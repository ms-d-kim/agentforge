"""Offline tests for the agentic search/grep domain (no API key, no network).

Covers: (a) the corpus grep finds a known symbol at its real location; (b) the
reward returns 1.0 for an exact gold match and 0.0 for a wrong file; (c) under the
fake responder the selector concentrates on the best arm and rolling accuracy
beats chance (1/3).
"""
from __future__ import annotations

import random
import unittest

from src.domains.search.corpus import Corpus
from src.domains.search.tasks import SearchTask, get_tasks
from src.domains.search.workflows import (
    compute_search_reward,
    make_arms,
    make_fake_responder,
)
from src.llm import FakeLLMClient
from src.selector import WorkflowSelector


class TestCorpus(unittest.TestCase):
    def test_grep_finds_known_symbol(self):
        corpus = Corpus()
        hits = corpus.grep("EpsilonGreedyBandit")
        self.assertTrue(hits, "expected grep hits for a known symbol")
        files = {f for f, _line, _text in hits}
        # The symbol is referenced in several files (multi-hop); its *definition*
        # lives in src/bandit.py, which must be among the hits.
        self.assertIn("src/bandit.py", files)
        # Paths are repo-relative POSIX and the corpus excludes sibling domains.
        self.assertFalse(any(f.startswith("/") for f in files))
        self.assertFalse(any("domains" in f for f in files))

    def test_definition_anchor_pins_one_file(self):
        corpus = Corpus()
        hits = corpus.grep("^class EpsilonGreedyBandit")
        self.assertEqual([(f, _l) for f, _l, _t in hits], [("src/bandit.py", 16)])

    def test_scoped_grep_restricts_to_files(self):
        corpus = Corpus()
        hits = corpus.grep("^class ", scope=["src/bandit.py"])
        self.assertTrue(hits)
        self.assertEqual({f for f, _l, _t in hits}, {"src/bandit.py"})


class TestReward(unittest.TestCase):
    TASK = SearchTask(
        task_id="t",
        question="q",
        gold_file="src/bandit.py",
        gold_symbol="EpsilonGreedyBandit",
        difficulty="exact",
    )

    def test_exact_match_is_one(self):
        r, err = compute_search_reward(self.TASK, "src/bandit.py")
        self.assertEqual(r, 1.0)
        self.assertIsNone(err)

    def test_basename_match_is_one(self):
        r, _ = compute_search_reward(self.TASK, "bandit.py")
        self.assertEqual(r, 1.0)

    def test_wrong_file_is_zero(self):
        r, err = compute_search_reward(self.TASK, "src/workflows/prompts.py")
        self.assertEqual(r, 0.0)
        self.assertIsNotNone(err)

    def test_same_directory_is_half(self):
        r, _ = compute_search_reward(self.TASK, "src/llm.py")  # same dir, wrong file
        self.assertEqual(r, 0.5)

    def test_no_file_is_zero(self):
        r, err = compute_search_reward(self.TASK, "")
        self.assertEqual(r, 0.0)
        self.assertIsNotNone(err)


class TestArmsDifferentiateUnderFake(unittest.TestCase):
    def _reward(self, task, out):
        return compute_search_reward(task, out)[0]

    def test_arms_return_paths_for_every_task(self):
        llm = FakeLLMClient(make_fake_responder())
        arms = make_arms(llm, Corpus())
        for task in get_tasks():
            for name, fn in arms.items():
                out = fn(task)
                self.assertIsInstance(out, str)
                self.assertTrue(out, "%s returned an empty path on %s" % (name, task.task_id))

    def test_selector_concentrates_and_beats_chance(self):
        llm = FakeLLMClient(make_fake_responder())
        arms = make_arms(llm, Corpus())
        sel = WorkflowSelector(arms, reward=self._reward, policy="epsilon-greedy", seed=0)

        tasks = get_tasks()
        rng = random.Random(0)
        exact_hits = 0
        episodes = 60
        for _ in range(episodes):
            task = rng.choice(tasks)
            result = sel.run(task)
            exact_hits += int((result.reward or 0.0) == 1.0)

        stats = sel.stats()
        best = sel.best()
        best_count = stats[best]["count"]
        total = sum(v["count"] for v in stats.values())

        # The selector identifies a single best arm and concentrates pulls there
        # (far more than a uniform 1/3 share).
        self.assertEqual(total, episodes)
        self.assertGreater(best_count, episodes / 3)
        self.assertGreater(stats[best]["mean"], 0.5)

        # Rolling accuracy (exact file matches) clears chance with margin.
        accuracy = exact_hits / episodes
        self.assertGreater(accuracy, 1 / 3)

    def test_iterative_outperforms_simple_on_average(self):
        """Sanity check on the designed competence gap (iterative >= simple)."""
        llm = FakeLLMClient(make_fake_responder())
        arms = make_arms(llm, Corpus())
        tasks = get_tasks()
        means = {}
        for name, fn in arms.items():
            means[name] = sum(self._reward(t, fn(t)) for t in tasks) / len(tasks)
        self.assertGreater(means["iterative"], means["simple"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
