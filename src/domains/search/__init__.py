"""Agentic search/grep domain for the AgentForge selector SDK.

Locate a target (a class/def in a code corpus) by grepping. Three strategy "arms"
plug into the domain-agnostic :class:`~src.selector.WorkflowSelector`:

* ``simple``            — one keyword, one grep, top hit.
* ``iterative``         — grep → disambiguate with the LLM → refine (≤3 rounds).
* ``broad_then_narrow`` — broad grep → candidates → narrow to a ``^class``/``^def``.

The selector learns online which arm wins on these tasks. Everything runs offline
with :class:`~src.llm.FakeLLMClient` (grep is a stdlib-adjacent POSIX tool).

This domain is intentionally MULTI-HOP friendly and is the seed for a later RL
phase: see the design note at the top of ``workflows.py`` for how the arms' inner
"grep → observe → re-grep → answer" loop maps onto an MDP (state = search history
+ observations; actions = next grep / answer; reward = file match). The arms stay
single-decision here; the RL is deferred.
"""
from __future__ import annotations

from .corpus import Corpus, GrepHit
from .tasks import SearchTask, get_tasks
from .workflows import (
    broad_then_narrow_search,
    compute_search_reward,
    iterative_search,
    make_arms,
    make_fake_responder,
    simple_search,
)

__all__ = [
    "Corpus",
    "GrepHit",
    "SearchTask",
    "get_tasks",
    "make_arms",
    "make_fake_responder",
    "compute_search_reward",
    "simple_search",
    "iterative_search",
    "broad_then_narrow_search",
]
