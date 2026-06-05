"""AgentForge — a learned selection layer for agentic workflows.

Public API for `pip install -e .` users. The reusable kernel:

    from agentforge import WorkflowSelector

`WorkflowSelector` is domain-agnostic — bring your own strategies (arms) and a
reward signal and it learns online which strategy to run each turn. The BIRD
text-to-SQL experiment in this repo (`src/`, `scripts/`) is one instantiation.
"""
from src.bandit import EpsilonGreedyBandit, ThompsonBandit, UCB1Bandit
from src.contextual import LinUCBBandit, default_featurize
from src.selector import SelectorResult, WorkflowSelector

__all__ = [
    "WorkflowSelector",
    "SelectorResult",
    "EpsilonGreedyBandit",
    "UCB1Bandit",
    "ThompsonBandit",
    "LinUCBBandit",
    "default_featurize",
]
__version__ = "0.2.0"
