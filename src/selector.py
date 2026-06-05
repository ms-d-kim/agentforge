"""WorkflowSelector — domain-agnostic, drop-in self-improvement for agent turns.

This is the reusable kernel of AgentForge, separated from the BIRD/SQL experiment.
Bring your own *strategies* (arms) and a *reward signal*; an online bandit learns
which strategy wins on your tasks and concentrates selection there — persisting
across runs so it keeps improving session to session.

    from agentforge import WorkflowSelector

    sel = WorkflowSelector(
        arms={"fast": fast_fn, "careful": careful_fn, "use_tools": tool_fn},
        reward=lambda task, out: 1.0 if passes_tests(out) else 0.0,
        policy="epsilon-greedy",          # or "ucb1"
        persist="selector_state.json",    # learns across runs
    )
    result = sel.run(task)                # picks an arm, runs it, scores, updates
    print(sel.stats(), sel.best())

The BIRD text-to-SQL experiment in this repo is one instantiation (SQL arms +
execution-match reward). Your agent is another (any callables + any 0..1 signal).

WHEN IT HELPS (learned from our own BIRD run): use it when your strategies
genuinely differ and you have a cheap automatic signal — it finds the winner and
drops the losers. If the strategies are statistically tied, it will surface that
too, so you can cut the complexity. No magic when there's nothing to learn.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .bandit import EpsilonGreedyBandit, ThompsonBandit, UCB1Bandit

_PROBE = "agentforge probe task"  # used once to learn the feature dimension for linucb


@dataclass
class SelectorResult:
    arm: str
    output: Any
    reward: Optional[float] = None


def _make_policy(policy: str, n: int, epsilon: float, seed: int, forced_init: bool, dim=None):
    if policy == "epsilon-greedy":
        return EpsilonGreedyBandit(n, epsilon=epsilon, seed=seed, forced_init=forced_init)
    if policy == "ucb1":
        return UCB1Bandit(n, seed=seed)
    if policy == "thompson":
        return ThompsonBandit(n, seed=seed)
    if policy == "linucb":
        from .contextual import LinUCBBandit

        if dim is None:
            raise ValueError("linucb requires a feature dimension")
        return LinUCBBandit(n, dim=dim, seed=seed)
    raise ValueError(
        f"unknown policy: {policy!r} (expected epsilon-greedy | ucb1 | thompson | linucb)"
    )


class WorkflowSelector:
    """Online selector over a fixed set of named strategies (arms)."""

    def __init__(
        self,
        arms: dict[str, Callable[[Any], Any]],
        *,
        reward: Optional[Callable[[Any, Any], float]] = None,
        policy: str = "epsilon-greedy",
        epsilon: float = 0.1,
        seed: int = 0,
        forced_init: bool = True,
        persist: Optional[str] = None,
        featurize: Optional[Callable[[Any], list]] = None,
        feature_dim: Optional[int] = None,
    ):
        if not arms:
            raise ValueError("WorkflowSelector needs at least one arm")
        self.names = list(arms.keys())
        self.arms = arms
        self.reward_fn = reward
        self.policy_name = policy
        self.persist = Path(persist) if persist else None
        self.featurize = featurize

        dim = None
        if policy == "linucb":
            if self.featurize is None:  # default: hashed bag-of-words context
                from .contextual import default_featurize

                self.featurize = default_featurize
            dim = feature_dim if feature_dim is not None else len(self.featurize(_PROBE))

        self._bandit = _make_policy(policy, len(self.names), epsilon, seed, forced_init, dim)
        self.history: list[tuple[str, float]] = []
        self._last_idx: Optional[int] = None
        self._last_context = None
        if self.persist and self.persist.exists():
            self.load()

    # -- core API ----------------------------------------------------------- #
    def select(self, task: Any = None) -> str:
        """Pick a strategy name (without running it). Use when you'll run it yourself.

        For the contextual policy, pass ``task`` so it can featurize the context;
        then call ``update`` for that same task before selecting again.
        """
        context = self.featurize(task) if (self.featurize and task is not None) else None
        self._last_context = context
        idx, _reason = self._bandit.select_arm(context)
        self._last_idx = idx
        return self.names[idx]

    def update(self, arm: str, reward: float) -> None:
        """Record the reward for a strategy and improve the policy (using the
        context from the most recent ``select``)."""
        idx = self.names.index(arm)
        self._bandit.update(idx, float(reward), self._last_context)
        self.history.append((arm, float(reward)))
        if self.persist:
            self.save()

    def run(self, task: Any) -> SelectorResult:
        """Select a strategy, run it on ``task``, and (if a reward fn was given)
        score + update automatically. Returns the arm, its output, and reward."""
        name = self.select(task)
        output = self.arms[name](task)
        reward = None
        if self.reward_fn is not None:
            reward = float(self.reward_fn(task, output))
            self.update(name, reward)
        return SelectorResult(arm=name, output=output, reward=reward)

    # -- inspection --------------------------------------------------------- #
    def stats(self) -> dict[str, dict[str, float]]:
        snap = self._bandit.snapshot()
        return {
            self.names[i]: {"mean": snap["means"][i], "count": snap["counts"][i]}
            for i in range(len(self.names))
        }

    def best(self) -> str:
        snap = self._bandit.snapshot()
        return self.names[max(range(len(self.names)), key=lambda i: snap["means"][i])]

    # -- persistence -------------------------------------------------------- #
    def save(self) -> None:
        if not self.persist:
            return
        snap = self._bandit.snapshot()
        self.persist.parent.mkdir(parents=True, exist_ok=True)
        self.persist.write_text(
            json.dumps(
                {
                    "policy": self.policy_name,
                    "names": self.names,
                    "means": snap["means"],
                    "counts": snap["counts"],
                    "updates": len(self.history),
                },
                indent=2,
            )
        )

    def load(self) -> None:
        """Restore arm estimates from disk. If the arm set changed, start fresh."""
        data = json.loads(self.persist.read_text())
        if data.get("names") != self.names:
            return
        self._bandit.means = list(data["means"])
        self._bandit.counts = list(data["counts"])
