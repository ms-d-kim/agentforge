"""All prompt templates as string constants (spec §7).

DIRECT_PROMPT is reproduced verbatim from the spec; the others follow the same
house style. Steps that get parsed (explore, decompose) request a JSON array to
make extraction robust.
"""
from __future__ import annotations

DIRECT_PROMPT = """\
You are an expert SQL writer. Given the following SQLite database schema and question, write a single SQL query that answers the question.

Schema:
{schema}

External knowledge: {evidence}

Question: {question}

Return only the SQL query, no explanation. Wrap it in ```sql ... ``` fences.
"""

EXPLORE_PROMPT = """\
You are an expert database analyst. Given the following SQLite schema and a question, identify which tables are relevant to answering it.

Schema:
{schema}

External knowledge: {evidence}

Question: {question}

Return ONLY a JSON array of the relevant table names, e.g. ["table_a", "table_b"]. No prose.
"""

WRITE_PROMPT = """\
You are an expert SQL writer. Using the following (already filtered to be relevant) SQLite schema, write a single SQL query that answers the question.

Relevant schema:
{schema}

External knowledge: {evidence}

Question: {question}

Return only the SQL query, no explanation. Wrap it in ```sql ... ``` fences.
"""

DECOMPOSE_PROMPT = """\
You are an expert at breaking down complex database questions. Decompose the question below into a short ordered list of simpler sub-questions that, answered in sequence, lead to the final answer. Use at most 4 sub-questions; if the question is already simple, return a single-element list.

External knowledge: {evidence}

Question: {question}

Return ONLY a JSON array of sub-question strings. No prose.
"""

SUB_SQL_PROMPT = """\
You are an expert SQL writer. Given the SQLite schema and a single sub-question, write a SQL query that answers just that sub-question.

Schema:
{schema}

Sub-question: {sub_q}

Return only the SQL query, wrapped in ```sql ... ``` fences.
"""

COMPOSE_PROMPT = """\
You are an expert SQL writer. You are given a database schema, an original question, and draft SQL for each sub-question. Compose a SINGLE final SQL query that answers the original question. The drafts are hints — you may rewrite them entirely.

Schema:
{schema}

Original question: {question}

Sub-query drafts:
{sub_sqls}

Return only the final SQL query, wrapped in ```sql ... ``` fences.
"""
