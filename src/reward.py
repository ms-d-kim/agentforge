"""Binary execution-match reward (spec §9).

Reward 1 iff the generated SQL's result set equals the gold SQL's result set,
compared order-insensitively as a set of row tuples. Any error or timeout on the
generated side yields reward 0. NULLs and floats are normalized before comparison
so cosmetic differences don't cause false mismatches.
"""
from __future__ import annotations

from .config import EXEC_TIMEOUT_S, FLOAT_ROUND
from .executor import execute_sql


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


def normalize_rows(rows: list[tuple]) -> set:
    """Order-insensitive, type-normalized set of row tuples."""
    return {tuple(_norm_cell(c) for c in row) for row in rows}


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

    if normalize_rows(gen_rows) == normalize_rows(gold_rows):
        return 1, None
    return 0, None
