"""Arm 2 — Schema-explore-then-write (spec §7.2).

Step 1 identifies relevant tables from the full schema; step 2 writes SQL against
only that subset. Hypothesis: wins on large/messy schemas where irrelevant tables
distract the model.
"""
from __future__ import annotations

from ..schema import filter_schema, format_schema, load_schema
from .base import (
    Workflow,
    WorkflowResult,
    extract_sql,
    parse_relevant_tables,
    track,
)
from .prompts import EXPLORE_PROMPT, WRITE_PROMPT


class SchemaExploreWorkflow(Workflow):
    name = "schema_explore"

    def run(self, task, llm) -> WorkflowResult:
        with track(llm) as t:
            creates = load_schema(task.db_path)
            full_schema = format_schema(creates)

            explore_raw = llm.complete(
                EXPLORE_PROMPT.format(
                    schema=full_schema,
                    evidence=task.evidence or "(none)",
                    question=task.question,
                )
            )
            tables = parse_relevant_tables(explore_raw)
            relevant_schema = filter_schema(creates, tables)

            write_raw = llm.complete(
                WRITE_PROMPT.format(
                    schema=relevant_schema,
                    evidence=task.evidence or "(none)",
                    question=task.question,
                )
            )
            sql = extract_sql(write_raw)

        return WorkflowResult(
            sql,
            t.calls,
            t.tokens,
            t.latency_ms,
            {"relevant_tables": tables, "explore_raw": explore_raw},
        )
