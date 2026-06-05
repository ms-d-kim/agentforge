"""Contextual bandit (LinUCB) — selection conditioned on task features.

This is the step the average-case bandits can't take: instead of "which arm is best
on average," LinUCB learns "which arm is best *for a task that looks like this*."
That's the PILOT-style online contextual adaptation (arXiv:2508.21141), but the
arms are whole workflows, not single models.

Disjoint LinUCB (Li et al. 2010): per arm ``a`` keep ``A_a = I + Σ x xᵀ`` and
``b_a = Σ r x``; for context ``x`` score ``θ_aᵀx + α·sqrt(xᵀ A_a⁻¹ x)`` and pull
the argmax. numpy is imported lazily so the stdlib policies stay dependency-free.
"""
from __future__ import annotations

import hashlib
import math
import random
import re


def default_featurize(task, dim: int = 64) -> list[float]:
    """Domain-agnostic features: hashed bag-of-words of ``str(task)`` + bias term.

    Provider-free and works for any task object. Swap in your own
    ``featurize(task) -> list[float]`` (e.g. an embedding) for richer context.
    """
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9_]+", str(task).lower()):
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        vec[h] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    vec = [v / norm for v in vec]
    vec.append(1.0)  # bias feature
    return vec


class LinUCBBandit:
    """Disjoint LinUCB contextual bandit."""

    def __init__(self, n_arms: int, dim: int, alpha: float = 1.0, seed: int = 0):
        import numpy as np  # lazy

        self._np = np
        self.n_arms = n_arms
        self.dim = dim
        self.alpha = alpha
        self.epsilon = 0.0  # logging-field compatibility
        self.A = [np.identity(dim) for _ in range(n_arms)]
        self.b = [np.zeros(dim) for _ in range(n_arms)]
        self.counts = [0] * n_arms
        self.means = [0.0] * n_arms
        self.rng = random.Random(seed)

    def select_arm(self, context=None) -> tuple[int, str]:
        np = self._np
        if context is None:
            return self.rng.randrange(self.n_arms), "linucb_nocontext"
        x = np.asarray(context, dtype=float)
        scores = []
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for a in range(self.n_arms):
                A_inv = np.linalg.inv(self.A[a])
                theta = A_inv @ self.b[a]
                var = float(x @ A_inv @ x)  # ≥0 in theory; clamp for numerical safety
                score = float(theta @ x) + self.alpha * (max(0.0, var) ** 0.5)
                scores.append(score if math.isfinite(score) else float("-inf"))
        best = max(scores)
        candidates = [i for i, s in enumerate(scores) if s == best]
        return self.rng.choice(candidates), "linucb"

    def update(self, arm: int, reward: float, context=None) -> None:
        if context is not None:
            x = self._np.asarray(context, dtype=float)
            self.A[arm] = self.A[arm] + self._np.outer(x, x)
            self.b[arm] = self.b[arm] + float(reward) * x
        self.counts[arm] += 1
        n = self.counts[arm]
        self.means[arm] += (reward - self.means[arm]) / n

    def snapshot(self) -> dict:
        return {"means": list(self.means), "counts": list(self.counts)}
