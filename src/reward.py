"""Binary execution-match reward (spec §9).

Reward 1 iff the generated SQL's result set equals the gold SQL's. Comparison is
order-insensitive (set of row tuples) UNLESS the gold query has an ``ORDER BY``, in
which case row order is significant and we compare the rows as an ordered list (so a
right-rows/wrong-order answer to an ordered question is not over-credited). Any error
or timeout on the generated side yields reward 0. NULLs and floats are normalized so
cosmetic differences don't cause false mismatches.
"""
from __future__ import annotations

import re

from .config import EXEC_TIMEOUT_S, FLOAT_ROUND
from .executor import execute_sql

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _norm_cell(value: object) -> object:
    """Normalize a single cell for comparison: round floats, stringify the rest.

    Casting to string (per spec §9) makes ``1`` and ``"1"`` compare equal, which
    matters because the same logical answer can come back with different column
    affinities across two queries.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return f"{round(value, FLOAT_ROUND)}"
    return str(value)


def _norm_row(row) -> tuple:
    return tuple(_norm_cell(c) for c in row)


def normalize_rows(rows: list[tuple]) -> set:
    """Order-insensitive, type-normalized set of row tuples."""
    return {_norm_row(row) for row in rows}


def _ordered_rows(rows: list[tuple]) -> list:
    """Order-sensitive, type-normalized list of row tuples (for ORDER BY golds)."""
    return [_norm_row(row) for row in rows]


def compute_reward(
    generated_sql: str,
    gold_sql: str,
    db_path: str,
    timeout_s: int = EXEC_TIMEOUT_S,
) -> tuple[int, str | None]:
    """Return ``(reward, error_message_or_None)``.

    A failure executing the *gold* SQL is reported with a ``gold_exec_error``
    prefix so it can be told apart from a model mistake in the logs.
    """
    try:
        gen_rows = execute_sql(generated_sql, db_path, timeout_s)
    except Exception as exc:  # noqa: BLE001 — any failure is a miss
        return 0, f"generated_exec_error: {exc}"
    try:
        gold_rows = execute_sql(gold_sql, db_path, timeout_s)
    except Exception as exc:  # noqa: BLE001
        return 0, f"gold_exec_error: {exc}"

    if _ORDER_BY.search(gold_sql or ""):
        match = _ordered_rows(gen_rows) == _ordered_rows(gold_rows)  # order significant
    else:
        match = normalize_rows(gen_rows) == normalize_rows(gold_rows)  # order-insensitive
    return (1, None) if match else (0, None)
