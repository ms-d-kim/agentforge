"""Arm 1 — Direct: single-shot SQL from question + full schema (spec §7.1)."""
from __future__ import annotations

from ..schema import load_schema_string
from .base import Workflow, WorkflowResult, extract_sql, track
from .prompts import DIRECT_PROMPT


class DirectWorkflow(Workflow):
    name = "direct"

    def run(self, task, llm) -> WorkflowResult:
        with track(llm) as t:
            schema = load_schema_string(task.db_path)
            prompt = DIRECT_PROMPT.format(
                schema=schema,
                evidence=task.evidence or "(none)",
                question=task.question,
            )
            raw = llm.complete(prompt)
            sql = extract_sql(raw)
        return WorkflowResult(sql, t.calls, t.tokens, t.latency_ms, {"raw": raw})
