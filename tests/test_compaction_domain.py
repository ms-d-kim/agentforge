"""Offline tests for the context-compaction / agent-memory domain.

All tests run fully offline with ``FakeLLMClient`` (no API key). They check:
  (a) each strategy returns a non-empty, strictly SHORTER compacted context;
  (b) on a task whose answer fact is LATE, ``truncate`` preserves it (reward 1.0)
      while ``extractive`` does not (reward 0.0) — strategies genuinely differ;
  (c) with the fake reader over 60 episodes, the selector concentrates on the
      best average arm (``hierarchical``) and rolling accuracy beats 1/3.
"""
from __future__ import annotations

import random
import unittest

from src.domains.compaction.reward import (
    answer_matches,
    compute_compaction_reward,
    make_fake_reader_responder,
    make_reward_closure,
)
from src.domains.compaction.strategies import make_strategies
from src.domains.compaction.tasks import TASKS, make_tasks
from src.llm import FakeLLMClient
from src.selector import WorkflowSelector


def _fake_reader():
    return FakeLLMClient(make_fake_reader_responder(TASKS))


def _task(task_id):
    return next(t for t in TASKS if t.task_id == task_id)


class TestTasks(unittest.TestCase):
    def test_suite_size_and_shape(self):
        self.assertGreaterEqual(len(TASKS), 8)
        self.assertLessEqual(len(TASKS), 10)
        for t in TASKS:
            # ~40-120 line logs.
            self.assertGreaterEqual(t.num_lines, 40)
            self.assertLessEqual(t.num_lines, 120)
            # Exactly one line carries the answer.
            hits = [ln for ln in t.context if t.answer.lower() in ln.lower()]
            self.assertEqual(len(hits), 1, f"{t.task_id}: answer not unique in context")
            # fact_line_index points at that line.
            self.assertIn(t.answer.lower(), t.context[t.fact_line_index()].lower())

    def test_facts_span_positions(self):
        # The buried fact should sit at varying positions (early/middle/late).
        fracs = [t.fact_line_index() / max(1, t.num_lines - 1) for t in TASKS]
        self.assertTrue(any(f < 0.2 for f in fracs), "no early facts")
        self.assertTrue(any(0.3 < f < 0.7 for f in fracs), "no middle facts")
        self.assertTrue(any(f > 0.8 for f in fracs), "no late facts")

    def test_make_tasks_deterministic(self):
        a = make_tasks()
        b = make_tasks()
        self.assertEqual([t.context for t in a], [t.context for t in b])


class TestStrategiesShorten(unittest.TestCase):
    """(a) Every strategy returns a non-empty, strictly shorter context."""

    def test_non_empty_and_shorter(self):
        strategies = make_strategies()
        self.assertEqual(set(strategies), {"truncate", "extractive", "hierarchical"})
        for name, fn in strategies.items():
            for t in TASKS:
                compacted = fn(t)
                self.assertIsInstance(compacted, str)
                self.assertTrue(compacted.strip(), f"{name}/{t.task_id}: empty")
                # Strictly shorter than the original both in chars and in lines.
                self.assertLess(
                    len(compacted),
                    len(t.context_text),
                    f"{name}/{t.task_id}: not shorter (chars)",
                )
                self.assertLess(
                    len(compacted.splitlines()),
                    t.num_lines,
                    f"{name}/{t.task_id}: not shorter (lines)",
                )


class TestRewardReflectsStrategy(unittest.TestCase):
    """(b) On a late fact, truncate preserves it but extractive may not."""

    def test_late_fact_truncate_beats_extractive(self):
        llm = _fake_reader()
        strategies = make_strategies()
        # deploy-port: fact is LATE and phrased WITHOUT the question keywords, and
        # high-overlap decoys sit early -> truncate keeps it, extractive drops it.
        task = _task("deploy-port")
        self.assertGreater(
            task.fact_line_index() / (task.num_lines - 1), 0.8, "fixture: fact not late"
        )

        trunc_ctx = strategies["truncate"](task)
        extr_ctx = strategies["extractive"](task)

        trunc_reward, trunc_out = compute_compaction_reward(task, trunc_ctx, llm)
        extr_reward, _ = compute_compaction_reward(task, extr_ctx, llm)

        self.assertEqual(trunc_reward, 1.0, "truncate should preserve the late fact")
        self.assertEqual(extr_reward, 0.0, "extractive should drop the late low-overlap fact")
        # The reward really came from the reader recovering the gold answer.
        self.assertIsNotNone(trunc_out)
        self.assertTrue(answer_matches(task.answer, trunc_out))

    def test_high_overlap_fact_extractive_preserves(self):
        llm = _fake_reader()
        strategies = make_strategies()
        # db-host: fact is EARLY but high-overlap -> truncate drops it, extractive
        # keeps it. The mirror image of the case above.
        task = _task("db-host")
        trunc_reward, _ = compute_compaction_reward(task, strategies["truncate"](task), llm)
        extr_reward, _ = compute_compaction_reward(task, strategies["extractive"](task), llm)
        self.assertEqual(extr_reward, 1.0, "extractive should keep the high-overlap fact")
        self.assertEqual(trunc_reward, 0.0, "truncate should drop the early fact")

    def test_fake_reader_unknown_when_fact_dropped(self):
        # When the fact is absent from the compacted context, the fake reader must
        # answer UNKNOWN (no answer present) -> reward 0.
        llm = _fake_reader()
        task = _task("deploy-port")
        # A compacted context that omits the fact line entirely.
        ctx_without_fact = "\n".join(task.context[:5])
        self.assertNotIn(task.answer.lower(), ctx_without_fact.lower())
        reward, out = compute_compaction_reward(task, ctx_without_fact, llm)
        self.assertEqual(reward, 0.0)
        self.assertEqual((out or "").strip().upper(), "UNKNOWN")


class TestPerArmProfile(unittest.TestCase):
    """Sanity on the differentiation the bandit will learn from."""

    def test_hierarchical_best_on_average(self):
        llm = _fake_reader()
        strategies = make_strategies()
        totals = {name: 0 for name in strategies}
        for name, fn in strategies.items():
            for t in TASKS:
                reward, _ = compute_compaction_reward(t, fn(t), llm)
                totals[name] += int(reward)
        best = max(totals, key=lambda n: totals[n])
        self.assertEqual(best, "hierarchical", f"profile={totals}")
        # Strict: hierarchical must out-preserve each of the others.
        self.assertGreater(totals["hierarchical"], totals["truncate"], f"{totals}")
        self.assertGreater(totals["hierarchical"], totals["extractive"], f"{totals}")


class TestSelectorConcentrates(unittest.TestCase):
    """(c) With the fake + 60 episodes, selection concentrates on hierarchical
    and rolling accuracy beats 1/3."""

    def _run(self, policy: str, episodes: int = 60, seed: int = 0):
        llm = _fake_reader()
        sel = WorkflowSelector(
            arms=make_strategies(),
            reward=make_reward_closure(llm),
            policy=policy,
            seed=seed,
        )
        rng = random.Random(seed)
        correct = 0
        for _ in range(episodes):
            result = sel.run(rng.choice(TASKS))
            correct += int(result.reward or 0)
        return sel, correct, episodes

    def test_epsilon_greedy_concentrates(self):
        sel, correct, episodes = self._run("epsilon-greedy")
        self.assertEqual(sel.best(), "hierarchical")
        stats = sel.stats()
        counts = {n: stats[n]["count"] for n in stats}
        # The winner should receive the clear majority of pulls.
        self.assertEqual(max(counts, key=lambda n: counts[n]), "hierarchical", counts)
        self.assertGreater(counts["hierarchical"], episodes / 2, counts)
        # Rolling accuracy beats uniform-random-over-3-arms (1/3).
        self.assertGreater(correct / episodes, 1.0 / 3.0)

    def test_ucb1_also_learns_winner(self):
        sel, correct, episodes = self._run("ucb1")
        self.assertEqual(sel.best(), "hierarchical")
        self.assertGreater(correct / episodes, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
