"""Compaction-strategy arms + the downstream reader.

Each strategy maps a :class:`CompactionTask` to a *compacted context* (a string):
a much shorter version of the agent log that the downstream reader will answer
from. The strategies are the selector ARMS — an online bandit learns which one
preserves the answer fact most often across the task suite.

Three strategies, intentionally biased differently:

  * ``truncate`` — keep only the last ``budget`` lines (recency). Cheap and
    fully deterministic. Wins when the fact is late; loses early facts.
  * ``extractive`` — keep the ``budget`` lines with the highest keyword overlap
    with the question. Cheap and fully deterministic. Wins when the fact is
    phrased with the question's words; loses facts phrased without them.
  * ``hierarchical`` — split the log into ``budget`` contiguous chunks and keep
    one representative line per chunk (the highest question-overlap line in that
    chunk, falling back to the first line). Broad coverage across positions —
    the robust all-rounder. Deterministic.

The READER (:func:`answer_from_context`) is a fixed downstream step: a single LLM
call that answers the question using ONLY the compacted context, and must say
``"UNKNOWN"`` if the fact is not present. The reward (see ``reward.py``) runs this
reader and checks whether the gold answer survived compaction.

Inspired by *ee392c-agent-mem* (EE 392C, Stanford; Minseok Kim & Kristen
Guernsey), which profiled a log-ingest/compaction workflow but had no downstream
task to reward; here compaction is graded by what it preserves.
"""
from __future__ import annotations

import re
from typing import Callable, Dict, List

from .tasks import CompactionTask

# Compaction budget: how many lines a strategy may keep. Kept small relative to
# the ~60-80 line logs so the three strategies genuinely diverge in what they
# preserve. (~9-12% of a log.)
DEFAULT_BUDGET = 8

# Words ignored when computing question/line keyword overlap.
_STOPWORDS = frozenset(
    """
    a an the of for to in on at is are was were be been being and or but if then
    what which who whom whose where when why how this that these those it its
    with as by from into your you our we they them us do does did has have had
    will would can could should may might must not no yes value set used use
    """.split()
)

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def _content_words(text: str) -> frozenset:
    return frozenset(t for t in _tokens(text) if t not in _STOPWORDS and len(t) > 1)


def _overlap(line: str, question_words: frozenset) -> int:
    """Number of distinct question content-words present in ``line``."""
    return len(_content_words(line) & question_words)


# --------------------------------------------------------------------------- #
# Strategy arms: CompactionTask -> compacted context (string)
# --------------------------------------------------------------------------- #
def truncate(task: CompactionTask, budget: int = DEFAULT_BUDGET) -> str:
    """Keep only the last ``budget`` lines (recency window)."""
    lines = list(task.context)
    kept = lines[-budget:] if budget < len(lines) else lines
    return "\n".join(kept)


def extractive(task: CompactionTask, budget: int = DEFAULT_BUDGET) -> str:
    """Keep the ``budget`` lines with the highest question-keyword overlap.

    Ties (and the common all-zero case) break toward earlier lines, so this is
    fully deterministic. Selected lines are returned in their original order.
    """
    qwords = _content_words(task.question)
    lines = list(task.context)
    scored = [(-_overlap(line, qwords), i) for i, line in enumerate(lines)]
    scored.sort()  # highest overlap first; ties -> smaller index first
    keep_idx = sorted(i for _, i in scored[:budget])
    return "\n".join(lines[i] for i in keep_idx)


def hierarchical(task: CompactionTask, budget: int = DEFAULT_BUDGET) -> str:
    """Chunk the log into ``budget`` parts; keep one representative line per chunk.

    The representative is the highest question-overlap line in the chunk (ties ->
    earliest), falling back to the chunk's first line. This guarantees coverage
    across the whole log regardless of where the fact sits — as long as the fact
    line stands out within its local chunk, it survives.
    """
    lines = list(task.context)
    n = len(lines)
    k = max(1, min(budget, n))
    qwords = _content_words(task.question)

    reps: List[str] = []
    # Contiguous near-equal chunks covering [0, n).
    for c in range(k):
        start = (c * n) // k
        end = ((c + 1) * n) // k
        if start >= end:
            continue
        chunk = lines[start:end]
        best_local = max(
            range(len(chunk)),
            key=lambda j: (_overlap(chunk[j], qwords), -j),
        )
        reps.append(chunk[best_local])
    return "\n".join(reps)


def make_strategies(budget: int = DEFAULT_BUDGET) -> Dict[str, Callable[[CompactionTask], str]]:
    """Return the arm dict ``{name: task -> compacted_context}`` for the selector.

    Stable arm order: ``truncate``, ``extractive``, ``hierarchical``.
    """
    return {
        "truncate": lambda task: truncate(task, budget),
        "extractive": lambda task: extractive(task, budget),
        "hierarchical": lambda task: hierarchical(task, budget),
    }


# --------------------------------------------------------------------------- #
# Downstream reader (fixed step): answer using ONLY the compacted context
# --------------------------------------------------------------------------- #
_READER_SYSTEM = (
    "You are a precise reading-comprehension agent. Answer the question using "
    "ONLY the provided context. If the answer is not stated in the context, "
    'reply with exactly "UNKNOWN". Reply with the shortest possible answer '
    "(a few words or a number), no explanation."
)

_READER_PROMPT = (
    "Context (compacted agent log):\n"
    "---\n"
    "{context}\n"
    "---\n"
    "Question: {question}\n"
    "Answer:"
)


def answer_from_context(llm, compacted_context: str, question: str) -> str:
    """One LLM call: answer ``question`` from ``compacted_context`` only.

    Returns the reader's raw text (the reward layer normalizes it). The reader is
    instructed to emit ``"UNKNOWN"`` when the fact is absent from the compacted
    context, so the reward measures whether compaction preserved the fact.
    """
    prompt = _READER_PROMPT.format(context=compacted_context, question=question)
    return llm.complete(prompt, system=_READER_SYSTEM).strip()
