"""Offline tests for the deterministic core + a full fake-LLM episode loop.

Run: `python -m unittest tests.test_pipeline` (no API key, no BIRD download).
"""
from __future__ import annotations

import unittest

from src.bandit import EpsilonGreedyBandit
from src.executor import execute_sql
from src.reward import compute_reward, normalize_rows
from src.schema import filter_schema, list_tables, load_schema
from src.tasks import load_fixture_tasks
from src.workflows import build_workflows
from src.workflows.base import extract_sql, parse_relevant_tables, parse_sub_questions
from scripts.make_fixture import ensure_fixture


class FixtureMixin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_fixture()
        cls.tasks = load_fixture_tasks()
        cls.by_id = {t.task_id: t for t in cls.tasks}


class TestBandit(unittest.TestCase):
    def test_forced_init_sweeps_each_arm_once(self):
        b = EpsilonGreedyBandit(3, epsilon=0.1, seed=0, forced_init=True)
        reasons = []
        for _ in range(3):
            arm, reason = b.select_arm()
            reasons.append(reason)
            b.update(arm, 0.0)
        self.assertEqual(reasons, ["forced", "forced", "forced"])
        self.assertEqual(b.counts, [1, 1, 1])

    def test_running_mean_update(self):
        b = EpsilonGreedyBandit(1, seed=0, forced_init=False)
        for r in (1.0, 0.0, 1.0, 1.0):
            b.update(0, r)
        self.assertAlmostEqual(b.means[0], 0.75)
        self.assertEqual(b.counts[0], 4)

    def test_determinism_same_seed(self):
        def pulls(seed):
            b = EpsilonGreedyBandit(3, seed=seed, forced_init=False)
            return [b.select_arm()[0] for _ in range(50)]
        self.assertEqual(pulls(7), pulls(7))


class TestParsing(unittest.TestCase):
    def test_extract_sql_fenced(self):
        self.assertEqual(extract_sql("```sql\nSELECT 1;\n```"), "SELECT 1")

    def test_extract_sql_bare(self):
        self.assertEqual(extract_sql("SELECT a FROM t"), "SELECT a FROM t")

    def test_parse_relevant_tables_json(self):
        self.assertEqual(parse_relevant_tables('["a", "b"]'), ["a", "b"])

    def test_parse_relevant_tables_fallback(self):
        self.assertEqual(parse_relevant_tables("schools, students"), ["schools", "students"])

    def test_parse_sub_questions_numbered(self):
        self.assertEqual(
            parse_sub_questions("1. first\n2. second"), ["first", "second"]
        )


class TestRewardAndExecutor(FixtureMixin):
    def test_gold_matches_itself(self):
        for t in self.tasks:
            r, err = compute_reward(t.gold_sql, t.gold_sql, t.db_path)
            self.assertEqual(r, 1, f"gold!=gold for task {t.task_id}: {err}")

    def test_wrong_sql_scores_zero(self):
        t = self.by_id["1"]  # COUNT(*) == 7
        r, _ = compute_reward("SELECT 0", t.gold_sql, t.db_path)
        self.assertEqual(r, 0)

    def test_order_insensitive(self):
        t = self.by_id["4"]  # names in grade 12
        shuffled = t.gold_sql + " ORDER BY name DESC"
        r, _ = compute_reward(shuffled, t.gold_sql, t.db_path)
        self.assertEqual(r, 1)

    def test_syntax_error_is_zero_not_crash(self):
        t = self.by_id["1"]
        r, err = compute_reward("SELEKT bogus", t.gold_sql, t.db_path)
        self.assertEqual(r, 0)
        self.assertIn("generated_exec_error", err)

    def test_readonly_blocks_mutation(self):
        t = self.by_id["1"]
        with self.assertRaises(Exception):
            execute_sql("DROP TABLE students", t.db_path)
        # table still there
        self.assertIn("students", list_tables(t.db_path))

    def test_normalize_rows_floats_and_nulls(self):
        a = normalize_rows([(1.000001, None), (2, "x")])
        b = normalize_rows([("1.0", None), ("2", "x")])
        self.assertEqual(a, b)


class TestSchema(FixtureMixin):
    def test_load_and_filter(self):
        creates = load_schema(self.by_id["1"].db_path)
        self.assertEqual(set(creates), {"schools", "students"})
        only = filter_schema(creates, ["students"])
        self.assertIn("students", only)
        self.assertNotIn("CREATE TABLE schools", only)

    def test_filter_empty_falls_back_to_full(self):
        creates = load_schema(self.by_id["1"].db_path)
        out = filter_schema(creates, ["nonexistent"])
        self.assertIn("schools", out)
        self.assertIn("students", out)


class TestFakeEpisodeLoop(FixtureMixin):
    def test_bandit_learns_best_arm(self):
        """With the scripted fake, schema_explore is best — bandit should pull it most."""
        import random

        from src.experiment import run_bandit
        from src.llm import FakeLLMClient
        from scripts.dry_run import make_responder

        rng = random.Random(0)
        episode_tasks = [rng.choice(self.tasks) for _ in range(80)]
        llm = FakeLLMClient(make_responder(self.tasks))
        records = run_bandit(
            tasks=episode_tasks, workflows=build_workflows(), llm=llm,
            run_id="unittest_bandit_seed0", seed=0, n_episodes=80,
            logs_dir="/tmp/agentforge_test_logs",
        )
        self.assertEqual(len(records), 80)
        final = records[-1]
        means = final["arm_means_after"]
        counts = final["arm_counts_after"]
        # arm 1 == schema_explore is the designed-best arm
        self.assertEqual(means.index(max(means)), 1, f"means={means}")
        self.assertEqual(counts.index(max(counts)), 1, f"counts={counts}")
        # every record carries the canonical schema
        for key in ("run_id", "reward", "workflow_name", "arm_means_after", "selection_reason"):
            self.assertIn(key, final)


if __name__ == "__main__":
    unittest.main(verbosity=2)
