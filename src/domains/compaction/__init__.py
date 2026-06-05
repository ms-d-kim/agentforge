"""Context-compaction / agent-memory domain for the AgentForge selector.

Turns context-compaction strategies into selector ARMS over a long-context QA
task: compact a long agent log, then answer a question using ONLY the compacted
context. The reward is downstream-QA success — whether the compaction preserved
the fact the question needs.

This domain is inspired by the prior project *ee392c-agent-mem* ("EE 392C:
Memory-Lifetime Characterization of LLM Agent-Workflow Replays", Stanford, Spring
2026; authors Minseok Kim & Kristen Guernsey), which profiles three memory
mechanisms — truncation, retrieval selectivity, and extractive context
compaction — but only MEASURES them. Here those mechanisms become arms and gain
the missing downstream-QA reward.

Public surface:
    CompactionTask, TASKS, make_tasks            (tasks.py)
    make_strategies, answer_from_context,        (strategies.py)
        truncate, extractive, hierarchical
    compute_compaction_reward, make_reward_closure,
        make_fake_reader_responder, answer_matches  (reward.py)
"""
from __future__ import annotations

from .reward import (
    answer_matches,
    compute_compaction_reward,
    make_fake_reader_responder,
    make_reward_closure,
)
from .strategies import (
    DEFAULT_BUDGET,
    answer_from_context,
    extractive,
    hierarchical,
    make_strategies,
    truncate,
)
from .tasks import TASKS, CompactionTask, make_tasks

__all__ = [
    "CompactionTask",
    "TASKS",
    "make_tasks",
    "DEFAULT_BUDGET",
    "make_strategies",
    "answer_from_context",
    "truncate",
    "extractive",
    "hierarchical",
    "compute_compaction_reward",
    "make_reward_closure",
    "make_fake_reader_responder",
    "answer_matches",
]
