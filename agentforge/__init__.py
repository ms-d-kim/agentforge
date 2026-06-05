"""AgentForge — a learned selection layer for agentic workflows.

Public API for `pip install -e .` users. The reusable kernel:

    from agentforge import WorkflowSelector

`WorkflowSelector` is domain-agnostic — bring your own strategies (arms) and a
reward signal and it learns online which strategy to run each turn. The BIRD
text-to-SQL experiment in this repo (`src/`, `scripts/`) is one instantiation.
"""
from src.bandit import EpsilonGreedyBandit, ThompsonBandit, UCB1Bandit
from src.contextual import LinUCBBandit, default_featurize
from src.rl import REINFORCE, MultiHopEnv
from src.selector import SelectorResult, WorkflowSelector

# The learning ladder, low → high: a one-step bandit picks a whole workflow
# (EpsilonGreedy/UCB1/Thompson), a contextual bandit picks per task (LinUCB), and
# REINFORCE learns a multi-step control policy (when to keep going vs. commit).
__all__ = [
    "WorkflowSelector",
    "SelectorResult",
    "EpsilonGreedyBandit",
    "UCB1Bandit",
    "ThompsonBandit",
    "LinUCBBandit",
    "default_featurize",
    "REINFORCE",
    "MultiHopEnv",
]
__version__ = "0.3.0"
