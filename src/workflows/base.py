"""Uniform workflow interface + output parsing helpers (spec §7).

All three arms implement ``run(task, llm) -> WorkflowResult`` so the bandit can
treat them as interchangeable actions (CLAUDE.md conventions). The parsing
helpers turn model text into the structured pieces each workflow needs.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class WorkflowResult:
    sql: str
    llm_calls: int = 0
    tokens_used: int = 0
    latency_ms: int = 0
    intermediate_outputs: dict = field(default_factory=dict)


class Workflow(ABC):
    name: str = "base"

    @abstractmethod
    def run(self, task, llm) -> WorkflowResult:  # pragma: no cover - interface
        ...


class track:
    """Context manager capturing per-workflow LLM-call/token deltas and latency.

        with track(llm) as t:
            ... llm.complete(...) ...
        WorkflowResult(sql, t.calls, t.tokens, t.latency_ms, ...)
    """

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0
        self.tokens = 0
        self.latency_ms = 0

    def __enter__(self):
        self._c0 = self.llm.total_calls
        self._t0 = self.llm.total_tokens
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.latency_ms = int((time.perf_counter() - self._start) * 1000)
        self.calls = self.llm.total_calls - self._c0
        self.tokens = self.llm.total_tokens - self._t0
        return False  # don't suppress exceptions


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
_SQL_FENCE = re.compile(r"```sql\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```\s*(.*?)```", re.DOTALL)
_JSON_ARRAY = re.compile(r"\[.*?\]", re.DOTALL)


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model response (prefers ```sql fences)."""
    if not text:
        return ""
    m = _SQL_FENCE.search(text) or _ANY_FENCE.search(text)
    sql = (m.group(1) if m else text).strip()
    return sql.strip().rstrip(";").strip()


def _parse_json_array(text: str) -> list[str] | None:
    m = _JSON_ARRAY.search(text or "")
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    if isinstance(arr, list):
        return [str(x).strip() for x in arr if str(x).strip()]
    return None


def parse_relevant_tables(text: str) -> list[str]:
    """Parse the explore step's relevant-table list (JSON array, comma, or lines)."""
    arr = _parse_json_array(text)
    if arr is not None:
        return arr
    # Fallbacks: comma- or newline-separated bare names.
    cleaned = re.sub(r"[`\[\]\"']", " ", text or "")
    parts = re.split(r"[,\n]", cleaned)
    return [p.strip() for p in parts if p.strip()]


def parse_sub_questions(text: str) -> list[str]:
    """Parse the decompose step's sub-question list (JSON array or numbered lines)."""
    arr = _parse_json_array(text)
    if arr is not None:
        return arr
    lines = []
    for line in (text or "").splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines
