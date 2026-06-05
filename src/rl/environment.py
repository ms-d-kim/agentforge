"""A multi-hop retrieval environment — the multi-step RL testbed.

The agent follows a pointer chain of unknown length and must ANSWER exactly when it
reaches the terminal entity — observed through a noisy "is this the answer?" signal —
not before (too early → wrong entity), not after (overshoots). This is the general
**iterate-or-stop** control problem that lives inside every multi-step workflow:
keep searching / keep repairing / keep hopping, or commit now?

The point: NO fixed-horizon policy can win — "answer after k hops" only solves
depth-k tasks. Only a STATE-CONDITIONED policy, learned here by REINFORCE from sparse
terminal reward, adapts the number of hops per task. That is exactly what a one-step
bandit (which commits to a whole workflow up front) cannot do — it's the next rung of
the ladder.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

FOLLOW, ANSWER = 0, 1
N_ACTIONS = 2
ACTION_NAMES = ["follow", "answer"]
N_FEATURES = 5


@dataclass(frozen=True)
class HopTask:
    task_id: str
    depth: int  # number of hops from the start to the terminal (answer) entity


class MultiHopEnv:
    """Follow a chain ``depth`` hops, then answer. State exposes a noisy terminal
    signal; the agent must learn to act on it."""

    def __init__(self, max_hops: int = 5, step_cost: float = 0.05,
                 signal_noise: float = 0.1, seed: int = 0):
        self.max_hops = max_hops
        self.step_cost = step_cost
        self.signal_noise = signal_noise
        self.rng = random.Random(seed)
        self.task: HopTask = None
        self.pos = 0

    def reset(self, task: HopTask) -> List[float]:
        self.task = task
        self.pos = 0
        return self._features()

    def _features(self) -> List[float]:
        # The "is the current entity the terminal/answer?" observation, with noise:
        truth = self.pos == self.task.depth
        signal = (not truth) if self.rng.random() < self.signal_noise else truth
        return [
            self.pos / self.max_hops,        # how far we've hopped (normalized)
            1.0 if signal else 0.0,          # noisy terminal signal (the key cue)
            1.0 if self.pos == 0 else 0.0,   # at start
            min(self.pos, 3) / 3.0,          # mild depth proxy
            1.0,                              # bias
        ]

    def step(self, action: int):
        reward = -self.step_cost
        done = False
        info: dict = {}
        if action == FOLLOW:
            if self.pos < self.max_hops:
                self.pos += 1
        elif action == ANSWER:
            done = True
            correct = self.pos == self.task.depth  # committed exactly at the terminal
            if correct:
                reward += 1.0
            info["correct"] = correct
        if self.pos >= self.max_hops and not done:
            done = True  # ran out of hops without committing → no reward
        return self._features(), reward, done, info


def get_env_tasks(n_per_depth: int = 4, depths=(1, 2, 3)) -> List[HopTask]:
    """A task pool spanning hop-depths 1..3, so no single fixed horizon can solve all."""
    return [HopTask(task_id=f"d{d}_{i}", depth=d) for d in depths for i in range(n_per_depth)]
