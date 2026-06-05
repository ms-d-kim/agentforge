"""Multi-step RL extension: from one-shot selection to a learned trajectory policy.

The bandits/selector make ONE decision per task (which workflow to run). This package
takes the next step the user asked for: increase the hops. A search *agent* issues a
SEQUENCE of actions (search / narrow / answer), reward arrives only at the end, and a
REINFORCE policy learns — via credit assignment over the trajectory — which action to
take in which state. This is the SMDP-over-workflows framing (cf. ARC, arXiv:2602.11574),
but online and from scratch.
"""
from .environment import ACTION_NAMES, HopTask, MultiHopEnv, N_ACTIONS, N_FEATURES
from .reinforce import (
    REINFORCE,
    evaluate,
    fixed_horizon_baseline,
    random_baseline,
    train,
)

__all__ = [
    "MultiHopEnv",
    "HopTask",
    "REINFORCE",
    "train",
    "evaluate",
    "fixed_horizon_baseline",
    "random_baseline",
    "ACTION_NAMES",
    "N_ACTIONS",
    "N_FEATURES",
]
