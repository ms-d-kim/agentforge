"""Tests for the policy ladder: Thompson sampling + contextual LinUCB.

The headline test (`test_contextual_beats_average_case`) shows the payoff of the
contextual bandit: on tasks where the best arm DEPENDS on the task, LinUCB learns
the per-context mapping and clears the ~50% ceiling that an average-case bandit
(Thompson) is stuck at.
"""
from __future__ import annotations

import random
import unittest

from src.selector import WorkflowSelector

ARMS = {"good": lambda t: "yes", "bad": lambda t: "no"}
REWARD = lambda task, out: 1.0 if out == "yes" else 0.0  # noqa: E731


def _typed_arms():
    """arm a0 is correct for type 'A'; arm a1 is correct for type 'B'."""
    def make(correct_type):
        return lambda task: "good" if task["type"] == correct_type else "bad"
    return {"a0": make("A"), "a1": make("B")}


TYPED_REWARD = lambda task, out: 1.0 if out == "good" else 0.0  # noqa: E731
TYPED_FEATURIZE = lambda task: [1.0, 0.0] if task["type"] == "A" else [0.0, 1.0]  # noqa: E731


def _run_typed(policy, n=300, seed=0, **kw):
    sel = WorkflowSelector(_typed_arms(), reward=TYPED_REWARD, policy=policy, seed=seed, **kw)
    rng = random.Random(seed + 1)
    rewards = []
    for i in range(n):
        task = {"type": rng.choice(["A", "B"]), "id": i}
        rewards.append(sel.run(task).reward)
    return sel, rewards


class TestThompson(unittest.TestCase):
    def test_learns_best_arm(self):
        sel = WorkflowSelector(ARMS, reward=REWARD, policy="thompson", seed=0)
        for _ in range(80):
            sel.run(None)
        self.assertEqual(sel.best(), "good")
        s = sel.stats()
        self.assertGreater(s["good"]["count"], s["bad"]["count"])

    def test_snapshot_has_posterior(self):
        sel = WorkflowSelector(ARMS, reward=REWARD, policy="thompson", seed=0)
        for _ in range(10):
            sel.run(None)
        snap = sel._bandit.snapshot()
        self.assertIn("alpha", snap)
        self.assertIn("beta", snap)


class TestContextual(unittest.TestCase):
    def test_contextual_learns_per_context(self):
        _sel, rewards = _run_typed("linucb", n=300, feature_dim=2, featurize=TYPED_FEATURIZE)
        tail = sum(rewards[-100:]) / 100
        self.assertGreater(tail, 0.8, f"contextual tail accuracy {tail} should clear 0.8")

    def test_contextual_beats_average_case(self):
        # Same task distribution; the average-case bandit is capped near 0.5.
        _c, ctx_rewards = _run_typed("linucb", n=300, feature_dim=2, featurize=TYPED_FEATURIZE)
        _t, avg_rewards = _run_typed("thompson", n=300)
        ctx_tail = sum(ctx_rewards[-100:]) / 100
        avg_tail = sum(avg_rewards[-100:]) / 100
        self.assertLess(avg_tail, 0.65, f"average-case should be ~0.5, got {avg_tail}")
        self.assertGreater(ctx_tail, avg_tail + 0.2, f"contextual {ctx_tail} vs average {avg_tail}")

    def test_default_featurizer_probe(self):
        # With the default hashed featurizer, dim is learned from a probe (no crash).
        sel = WorkflowSelector(ARMS, reward=REWARD, policy="linucb", seed=0)
        for _ in range(20):
            sel.run("some task text")
        self.assertEqual(len(sel.stats()), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
