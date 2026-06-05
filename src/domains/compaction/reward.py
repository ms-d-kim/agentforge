"""Downstream-QA reward for the compaction domain — the signal ee392c-agent-mem lacked.

The prior project *ee392c-agent-mem* (EE 392C, Stanford; Minseok Kim & Kristen
Guernsey) profiled context compaction but only MEASURED memory behavior; it had
no downstream task to say whether a compaction was *good*. Here the reward closes
that loop: a compacted context is good iff the downstream reader can still answer
the question from it.

``compute_compaction_reward(task, compacted_context, llm)`` runs the fixed reader
(``strategies.answer_from_context``) on the compacted context and returns
``(reward, reader_output)``: ``1.0`` if the gold answer appears in the reader's
output (normalized substring/equality match), else ``0.0``. The arm's output is
the compacted context; the reward function itself invokes the reader, so the
selector's reward closure can pass the (real or fake) reader in.

``make_fake_reader_responder()`` is the fully-offline reader: it returns the gold
answer IFF the answer fact actually survived into the compacted context it is
handed (it reads that context and looks for the fact line / answer), else
``"UNKNOWN"``. With this fake, reward depends purely on whether the chosen
strategy preserved the fact — deterministic, no real LLM, no API key.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from .strategies import answer_from_context
from .tasks import CompactionTask


def _normalize(text: str) -> str:
    """Lowercase, collapse whitespace, strip surrounding punctuation/quotes."""
    text = (text or "").lower().strip()
    text = text.strip("\"'`.,:;!?()[]{} \t\n")
    return re.sub(r"\s+", " ", text)


def answer_matches(gold: str, output: str) -> bool:
    """True if ``gold`` appears in ``output`` (normalized equality or substring).

    Substring is checked on word boundaries so e.g. gold ``"v7"`` does not match
    inside ``"v70"``. Falls back to plain containment when the gold itself has no
    word characters.
    """
    g = _normalize(gold)
    o = _normalize(output)
    if not g:
        return False
    if g == o:
        return True
    if re.search(r"\w", g):
        return re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", o) is not None
    return g in o


def compute_compaction_reward(
    task: CompactionTask,
    compacted_context: str,
    llm,
) -> Tuple[float, Optional[str]]:
    """Run the reader on ``compacted_context``; reward preservation of the fact.

    Returns ``(reward, reader_output)``. ``reward`` is ``1.0`` when the gold
    answer appears in the reader's output, else ``0.0``. ``reader_output`` is the
    reader's raw text (or ``None`` if the reader produced nothing).
    """
    reader_output = answer_from_context(llm, compacted_context, task.question)
    reward = 1.0 if answer_matches(task.answer, reader_output) else 0.0
    return reward, (reader_output if reader_output else None)


def make_reward_closure(llm) -> Callable[[CompactionTask, str], float]:
    """Adapt the reward to the selector's ``reward=(task, output)->float`` shape.

    The selector arm returns the compacted context as its ``output``; this closure
    feeds that context plus the bound ``llm`` (real or fake reader) into
    :func:`compute_compaction_reward` and returns just the float.
    """

    def reward(task: CompactionTask, compacted_context: str) -> float:
        score, _ = compute_compaction_reward(task, compacted_context, llm)
        return score

    return reward


# --------------------------------------------------------------------------- #
# Offline fake reader
# --------------------------------------------------------------------------- #
def _extract_context(prompt: str) -> str:
    """Pull the compacted context back out of the reader prompt.

    The reader prompt fences the context between ``---`` markers (see
    ``strategies._READER_PROMPT``); we read it back so the fake can decide purely
    from what compaction actually kept.
    """
    parts = prompt.split("\n---\n")
    if len(parts) >= 3:
        return parts[1]
    # Fallback: everything before the trailing "Question:" line.
    return prompt.split("\nQuestion:", 1)[0]


def make_fake_reader_responder(
    tasks: List[CompactionTask],
) -> Callable[[str, Optional[str]], str]:
    """Build a deterministic offline reader responder over ``tasks``.

    The returned ``responder(prompt, system)`` inspects the compacted context
    embedded in the prompt and returns the gold answer of whichever task's fact
    line survived into that context; otherwise ``"UNKNOWN"``. This makes the
    reward a pure function of whether the strategy preserved the fact — exactly
    the property ee392c-agent-mem could not score.
    """
    # Map each task's answer to the lowercased fact line that carries it, so the
    # fake can detect survival by checking the compacted context for that line.
    fact_lines = [
        (task.answer, task.context[task.fact_line_index()].lower()) for task in tasks
    ]

    def responder(prompt: str, system: Optional[str] = None) -> str:
        context = _extract_context(prompt).lower()
        for answer, fact_line in fact_lines:
            # Survival = the fact-bearing line is present in the compacted context.
            if fact_line and fact_line in context:
                return answer
        return "UNKNOWN"

    return responder
