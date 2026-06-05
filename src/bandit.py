"""Epsilon-greedy multi-armed bandit (spec §8).

One discrete selection decision per task over running mean rewards. This is the
*policy* layer — the project's contribution is a learned selector over multi-step
workflows, not a novel bandit algorithm (CLAUDE.md framing rules).

Determinism matters (spec §4): all randomness flows through ``self.rng``, seeded
and logged by the caller.
"""
from __future__ import annotations

import random


class EpsilonGreedyBandit:
    def __init__(
        self,
        n_arms: int,
        epsilon: float = 0.1,
        seed: int = 0,
        forced_init: bool = True,
    ):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.seed = seed
        self.forced_init = forced_init
        self.means = [0.0] * n_arms
        self.counts = [0] * n_arms
        self.rng = random.Random(seed)

    def select_arm(self) -> tuple[int, str]:
        """Return ``(arm_index, reason)`` where reason is one of
        ``"forced"`` | ``"explore"`` | ``"exploit"``.

        With ``forced_init`` on, each arm is pulled once (in index order) before
        the epsilon-greedy rule engages, so no arm is judged on a mean of 0.
        """
        if self.forced_init:
            for i, c in enumerate(self.counts):
                if c == 0:
                    return i, "forced"

        if self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_arms), "explore"

        # Exploit — break ties uniformly at random (important early on).
        max_val = max(self.means)
        candidates = [i for i, m in enumerate(self.means) if m == max_val]
        return self.rng.choice(candidates), "exploit"

    def update(self, arm: int, reward: float) -> None:
        """Incremental running-mean update for the pulled arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n

    def snapshot(self) -> dict:
        """Copy of current estimates, for JSONL logging."""
        return {"means": list(self.means), "counts": list(self.counts)}
