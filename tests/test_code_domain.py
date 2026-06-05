"""Offline tests for the code-generation selector domain (src/domains/code).

All tests run with zero API calls: the executor is real (subprocess + timeout),
and the LLM is the deterministic ``make_fake_responder`` stand-in. Run with::

    python -m unittest tests.test_code_domain
"""
from __future__ import annotations

import random
import unittest

from src.domains.code import (
    TASKS,
    compute_code_reward,
    execute_python,
    make_arms,
    make_fake_responder,
)
from src.llm import FakeLLMClient
from src.selector import WorkflowSelector


class TestExecutor(unittest.TestCase):
    def test_passes_correct_impl(self):
        passed, _out = execute_python(
            "def add(a, b):\n    return a + b\n",
            "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n",
        )
        self.assertTrue(passed)

    def test_fails_wrong_impl(self):
        passed, out = execute_python(
            "def add(a, b):\n    return a - b\n",
            "assert add(2, 3) == 5\n",
        )
        self.assertFalse(passed)
        self.assertTrue(out)  # a non-empty failure message

    def test_blocks_forbidden_pattern(self):
        # Even though this code would "work", the deny-list must reject it.
        passed, out = execute_python(
            "import os\ndef cwd():\n    return os.getcwd()\n",
            "assert isinstance(cwd(), str)\n",
        )
        self.assertFalse(passed)
        self.assertIn("forbidden", out)

    def test_blocks_open_call(self):
        passed, out = execute_python(
            "def leak():\n    return open('/etc/passwd').read()\n",
            "assert leak()\n",
        )
        self.assertFalse(passed)
        self.assertIn("forbidden", out)

    def test_empty_code_not_passed(self):
        passed, _out = execute_python("", "assert True\n")
        self.assertFalse(passed)

    def test_timeout_does_not_crash(self):
        # A tiny budget on a long loop must return cleanly, not raise.
        passed, out = execute_python(
            "def slow():\n"
            "    x = 0\n"
            "    for _ in range(10**9):\n"
            "        x += 1\n"
            "    return x\n"
            "slow()\n",
            "assert True\n",
            timeout_s=1,
        )
        self.assertFalse(passed)
        self.assertIn("timeout", out)


class TestGoldReferences(unittest.TestCase):
    def test_all_reference_impls_score_one(self):
        for task in TASKS:
            score, err = compute_code_reward(task, task.reference_impl)
            self.assertEqual(
                score,
                1.0,
                "reference for {} did not pass: {}".format(task.task_id, err),
            )

    def test_fixture_spans_difficulties(self):
        diffs = {t.difficulty for t in TASKS}
        self.assertEqual(diffs, {"easy", "medium", "hard"})
        self.assertGreaterEqual(len(TASKS), 10)


class TestArmDifferentiationUnderFake(unittest.TestCase):
    def test_per_arm_success_profile(self):
        """iterative solves everything; decompose the middle; direct the least."""
        llm = FakeLLMClient(make_fake_responder())
        arms = make_arms(llm)
        totals = {name: 0 for name in arms}
        for task in TASKS:
            for name, fn in arms.items():
                score, _err = compute_code_reward(task, fn(task))
                totals[name] += int(score)
        self.assertGreater(totals["iterative"], totals["decompose"])
        self.assertGreater(totals["decompose"], totals["direct"])
        self.assertEqual(totals["iterative"], len(TASKS))


class TestSelectorLearnsBestArm(unittest.TestCase):
    def test_concentrates_on_iterative_over_60_episodes(self):
        llm = FakeLLMClient(make_fake_responder())
        sel = WorkflowSelector(
            arms=make_arms(llm),
            reward=lambda task, code: compute_code_reward(task, code)[0],
            policy="epsilon-greedy",
            seed=0,
        )
        rng = random.Random(0)
        correct = 0
        episodes = 60
        for _ in range(episodes):
            result = sel.run(rng.choice(TASKS))
            correct += int(result.reward or 0)

        stats = sel.stats()
        # The selector should have identified and concentrated pulls on iterative.
        self.assertEqual(sel.best(), "iterative")
        self.assertGreater(stats["iterative"]["count"], stats["direct"]["count"])
        self.assertGreater(stats["iterative"]["count"], stats["decompose"]["count"])
        # Rolling accuracy must clear the 1/3 random-arm baseline.
        self.assertGreater(correct / episodes, 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
