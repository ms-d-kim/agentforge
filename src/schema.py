"""Load and format a SQLite database schema for prompts (spec §7).

`load_schema` returns the CREATE TABLE statements (which carry column types,
primary keys, and foreign keys — exactly what the model needs). `format_schema`
renders them to a prompt string. `filter_schema` keeps only a relevant subset,
used by the schema-explore workflow.
"""
from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path


def _connect_ro(db_path: str) -> sqlite3.Connection:
    """Open a read-only connection (predicted SQL must never mutate the DB)."""
    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def load_schema(db_path: str) -> "OrderedDict[str, str]":
    """Map of {table_name: CREATE TABLE statement}, user tables only, in DB order."""
    con = _connect_ro(db_path)
    try:
        rows = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
            "ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    creates: "OrderedDict[str, str]" = OrderedDict()
    for name, sql in rows:
        if sql:
            creates[name] = sql.strip()
    return creates


def format_schema(creates: "OrderedDict[str, str]") -> str:
    """Render CREATE statements into a single prompt-ready string."""
    return "\n\n".join(creates[t] for t in creates)


def load_schema_string(db_path: str) -> str:
    """Convenience: full schema of a DB as one formatted string."""
    return format_schema(load_schema(db_path))


def list_tables(db_path: str) -> list[str]:
    return list(load_schema(db_path).keys())


def filter_schema(creates: "OrderedDict[str, str]", table_names: list[str]) -> str:
    """Keep only the named tables (case-insensitive). Falls back to the full
    schema if the requested names match nothing, so the writer never starves."""
    wanted = {n.strip().lower() for n in table_names if n.strip()}
    keep = OrderedDict((t, creates[t]) for t in creates if t.lower() in wanted)
    if not keep:
        keep = creates
    return format_schema(keep)
