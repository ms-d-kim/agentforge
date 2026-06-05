"""Epsilon-greedy multi-armed bandit (spec §8).

One discrete selection decision per task over running mean rewards. This is the
*policy* layer — the project's contribution is a learned selector over multi-step
workflows, not a novel bandit algorithm (CLAUDE.md framing rules).

Determinism matters (spec §4): all randomness flows through ``self.rng``, seeded
and logged by the caller.
"""
from __future__ import annotations

import math
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

    def select_arm(self, context=None) -> tuple[int, str]:
        """Return ``(arm_index, reason)`` where reason is one of
        ``"forced"`` | ``"explore"`` | ``"exploit"``.

        ``context`` is accepted (and ignored) for a uniform policy interface with
        the contextual bandit. With ``forced_init`` on, each arm is pulled once (in
        index order) before the epsilon-greedy rule engages.
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

    def update(self, arm: int, reward: float, context=None) -> None:
        """Incremental running-mean update for the pulled arm."""
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n

    def snapshot(self) -> dict:
        """Copy of current estimates, for JSONL logging."""
        return {"means": list(self.means), "counts": list(self.counts)}


class UCB1Bandit:
    """UCB1 policy (Auer et al. 2002) — a no-tuning alternative to ε-greedy.

    Same interface as ``EpsilonGreedyBandit`` (select_arm / update / snapshot) so the
    ``WorkflowSelector`` can swap policies. Picks the arm maximizing
    ``mean + sqrt(2 ln t / count)``; deterministic exploration via the confidence bonus.
    """

    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self.epsilon = 0.0  # for logging-field compatibility
        self.means = [0.0] * n_arms
        self.counts = [0] * n_arms
        self.rng = random.Random(seed)

    def select_arm(self, context=None) -> tuple[int, str]:
        for i, c in enumerate(self.counts):
            if c == 0:  # pull each arm once before the UCB rule engages
                return i, "init"
        t = sum(self.counts)
        ucb = [
            self.means[i] + math.sqrt(2 * math.log(t) / self.counts[i])
            for i in range(self.n_arms)
        ]
        best = max(ucb)
        candidates = [i for i, u in enumerate(ucb) if u == best]
        return self.rng.choice(candidates), "ucb"

    def update(self, arm: int, reward: float, context=None) -> None:
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n

    def snapshot(self) -> dict:
        return {"means": list(self.means), "counts": list(self.counts)}


class ThompsonBandit:
    """Thompson sampling with a Beta-Bernoulli posterior per arm.

    For each arm, maintains Beta(α, β) (uniform prior α=β=1); samples a success
    probability per arm and pulls the argmax. Rewards in [0,1] update the posterior
    fractionally (α += r, β += 1−r) — the standard Bernoulli extension. No tuning
    knob (unlike ε), and typically lower regret than ε-greedy.
    """

    def __init__(self, n_arms: int, seed: int = 0):
        self.n_arms = n_arms
        self.epsilon = 0.0  # logging-field compatibility
        self.alpha = [1.0] * n_arms
        self.beta = [1.0] * n_arms
        self.counts = [0] * n_arms
        self.means = [0.0] * n_arms
        self.rng = random.Random(seed)

    def select_arm(self, context=None) -> tuple[int, str]:
        samples = [self.rng.betavariate(self.alpha[i], self.beta[i]) for i in range(self.n_arms)]
        best = max(samples)
        candidates = [i for i, s in enumerate(samples) if s == best]
        return self.rng.choice(candidates), "thompson"

    def update(self, arm: int, reward: float, context=None) -> None:
        r = max(0.0, min(1.0, float(reward)))
        self.alpha[arm] += r
        self.beta[arm] += 1.0 - r
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n

    def snapshot(self) -> dict:
        return {
            "means": list(self.means),
            "counts": list(self.counts),
            "alpha": list(self.alpha),
            "beta": list(self.beta),
        }
