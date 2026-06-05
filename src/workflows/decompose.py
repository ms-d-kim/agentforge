"""Arm 3 — Decompose-then-compose (spec §7.3).

Step 1 splits the question into sub-questions; step 2 writes sub-SQL for each;
step 3 composes a final query. Hypothesis: wins on multi-join / nested questions.

Robustness (spec §17): if composition fails or yields empty SQL, fall back to the
last sub-SQL and record ``workflow_error`` rather than crashing the episode.
"""
from __future__ import annotations

from ..schema import load_schema_string
from .base import (
    Workflow,
    WorkflowResult,
    extract_sql,
    parse_sub_questions,
    track,
)
from .prompts import COMPOSE_PROMPT, DECOMPOSE_PROMPT, SUB_SQL_PROMPT


class DecomposeWorkflow(Workflow):
    name = "decompose"

    def run(self, task, llm) -> WorkflowResult:
        intermediate: dict = {}
        with track(llm) as t:
            schema = load_schema_string(task.db_path)

            decompose_raw = llm.complete(
                DECOMPOSE_PROMPT.format(
                    evidence=task.evidence or "(none)",
                    question=task.question,
                )
            )
            sub_questions = parse_sub_questions(decompose_raw) or [task.question]
            intermediate["sub_questions"] = sub_questions

            sub_sqls = []
            for sq in sub_questions:
                raw = llm.complete(SUB_SQL_PROMPT.format(schema=schema, sub_q=sq))
                sub_sqls.append(extract_sql(raw))
            intermediate["sub_sqls"] = sub_sqls

            try:
                compose_raw = llm.complete(
                    COMPOSE_PROMPT.format(
                        schema=schema,
                        question=task.question,
                        sub_sqls="\n\n".join(s for s in sub_sqls if s),
                    )
                )
                sql = extract_sql(compose_raw)
                if not sql:
                    raise ValueError("empty composed SQL")
            except Exception as exc:  # noqa: BLE001 — fall back, don't crash the run
                intermediate["workflow_error"] = f"compose_failed: {exc}"
                sql = next((s for s in reversed(sub_sqls) if s), "")

        return WorkflowResult(sql, t.calls, t.tokens, t.latency_ms, intermediate)
