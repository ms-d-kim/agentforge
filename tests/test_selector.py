"""Offline tests for the domain-agnostic WorkflowSelector SDK."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.selector import WorkflowSelector

# Deterministic, non-LLM arms: "good" always scores 1, "bad" always 0.
ARMS = {"good": lambda t: "yes", "bad": lambda t: "no", "meh": lambda t: "maybe"}
REWARD = lambda task, out: 1.0 if out == "yes" else 0.0  # noqa: E731


class TestWorkflowSelector(unittest.TestCase):
    def test_learns_best_arm_epsilon_greedy(self):
        sel = WorkflowSelector(ARMS, reward=REWARD, policy="epsilon-greedy", seed=0)
        for _ in range(80):
            sel.run(None)
        self.assertEqual(sel.best(), "good")
        stats = sel.stats()
        self.assertGreater(stats["good"]["count"], stats["bad"]["count"])
        self.assertAlmostEqual(stats["good"]["mean"], 1.0)
        self.assertAlmostEqual(stats["bad"]["mean"], 0.0)

    def test_learns_best_arm_ucb1(self):
        sel = WorkflowSelector(ARMS, reward=REWARD, policy="ucb1", seed=0)
        for _ in range(80):
            sel.run(None)
        self.assertEqual(sel.best(), "good")

    def test_run_returns_arm_output_reward(self):
        sel = WorkflowSelector(ARMS, reward=REWARD, seed=1)
        r = sel.run(None)
        self.assertIn(r.arm, ARMS)
        self.assertIn(r.output, {"yes", "no", "maybe"})
        self.assertIn(r.reward, {0.0, 1.0})

    def test_manual_update_without_reward_fn(self):
        sel = WorkflowSelector(ARMS, seed=0)  # no reward fn
        name = sel.select()
        self.assertIn(name, ARMS)
        sel.update(name, 1.0)
        self.assertEqual(sel.stats()[name]["count"], 1)

    def test_persistence_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            sel = WorkflowSelector(ARMS, reward=REWARD, persist=path, seed=0)
            for _ in range(20):
                sel.run(None)
            counts = {n: sel.stats()[n]["count"] for n in ARMS}
            # New selector loads prior state from disk.
            sel2 = WorkflowSelector(ARMS, reward=REWARD, persist=path, seed=0)
            counts2 = {n: sel2.stats()[n]["count"] for n in ARMS}
            self.assertEqual(counts, counts2)
            self.assertEqual(sel2.best(), "good")

    def test_persistence_round_trip_thompson(self):
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            s1 = WorkflowSelector(ARMS, reward=REWARD, policy="thompson", persist=path, seed=0)
            for _ in range(60):
                s1.run(None)
            i = s1.names.index("good")
            alpha_before = s1._bandit.alpha[i]
            self.assertGreater(alpha_before, 1.0)  # posterior actually moved
            s2 = WorkflowSelector(ARMS, reward=REWARD, policy="thompson", persist=path, seed=0)
            self.assertEqual(s2._bandit.alpha[i], alpha_before)  # posterior restored, not reset
            self.assertEqual(s2.best(), "good")

    def test_persistence_round_trip_contextual(self):
        import random

        import numpy as np

        arms = {"a0": lambda t: "good" if t == "A" else "bad",
                "a1": lambda t: "good" if t == "B" else "bad"}
        reward = lambda task, out: 1.0 if out == "good" else 0.0  # noqa: E731
        feat = lambda task: [1.0, 0.0] if task == "A" else [0.0, 1.0]  # noqa: E731
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / "state.json")
            s1 = WorkflowSelector(arms, reward=reward, policy="linucb", featurize=feat,
                                  feature_dim=2, persist=path, seed=0)
            rng = random.Random(0)
            for _ in range(200):
                s1.run(rng.choice(["A", "B"]))
            # Reload: the contextual mapping (A/b) must survive, not reset to identity.
            s2 = WorkflowSelector(arms, reward=reward, policy="linucb", featurize=feat,
                                  feature_dim=2, persist=path, seed=0)
            self.assertFalse(np.allclose(s2._bandit.A[0], np.identity(2)))  # state restored
            picks_a = [s2.select("A") for _ in range(20)]
            picks_b = [s2.select("B") for _ in range(20)]
            self.assertEqual(max(set(picks_a), key=picks_a.count), "a0")
            self.assertEqual(max(set(picks_b), key=picks_b.count), "a1")

    def test_empty_arms_raises(self):
        with self.assertRaises(ValueError):
            WorkflowSelector({}, reward=REWARD)

    def test_unknown_policy_raises(self):
        with self.assertRaises(ValueError):
            WorkflowSelector(ARMS, reward=REWARD, policy="nonexistent-policy")


if __name__ == "__main__":
    unittest.main(verbosity=2)
