"""Tests for the multi-step RL layer (REINFORCE on the multi-hop env).

Verifies the headline claim: a learned, state-conditioned policy beats every
fixed-horizon baseline (and random) on tasks of mixed hop-depth.
"""
from __future__ import annotations

import unittest

from src.rl.environment import ANSWER, FOLLOW, MultiHopEnv, get_env_tasks
from src.rl.reinforce import (
    evaluate,
    fixed_horizon_baseline,
    random_baseline,
    train,
)


class TestMultiHopEnv(unittest.TestCase):
    def test_answer_at_terminal_rewards(self):
        env = MultiHopEnv(signal_noise=0.0, seed=0)
        task = next(t for t in get_env_tasks() if t.depth == 2)
        env.reset(task)
        env.step(FOLLOW)
        env.step(FOLLOW)
        _f, reward, done, info = env.step(ANSWER)
        self.assertTrue(done)
        self.assertTrue(info["correct"])
        self.assertGreater(reward, 0)

    def test_answer_too_early_fails(self):
        env = MultiHopEnv(signal_noise=0.0, seed=0)
        task = next(t for t in get_env_tasks() if t.depth == 3)
        env.reset(task)
        _f, reward, _d, info = env.step(ANSWER)  # answer at hop 0
        self.assertFalse(info["correct"])
        self.assertLess(reward, 0)


class TestREINFORCE(unittest.TestCase):
    def test_beats_fixed_and_random_baselines(self):
        agent, _hist = train(episodes=4000, seed=0)
        sr, _steps = evaluate(agent, episodes=300, greedy=True)
        best_fixed = max(fixed_horizon_baseline(k, episodes=300) for k in range(4))
        rnd = random_baseline(episodes=300)
        self.assertGreater(sr, 0.7, f"REINFORCE success {sr:.2f} should clear 0.7")
        self.assertGreater(sr, best_fixed + 0.2, f"RL {sr:.2f} vs best fixed {best_fixed:.2f}")
        self.assertGreater(sr, rnd, f"RL {sr:.2f} vs random {rnd:.2f}")

    def test_learning_improves_over_training(self):
        _agent, hist = train(episodes=4000, seed=1)
        early = sum(s for s, _ in hist[:500]) / 500
        late = sum(s for s, _ in hist[-500:]) / 500
        self.assertGreater(late, early + 0.1, f"late {late:.2f} should beat early {early:.2f}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
