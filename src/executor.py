"""Safe SQL execution against a SQLite DB, with a wall-clock timeout (spec §9, §17).

Connections are opened read-only so a stray ``DROP``/``UPDATE`` in generated SQL
cannot mutate the benchmark databases. Timeouts are enforced by interrupting the
connection from a watchdog timer thread (``Connection.interrupt`` is documented
safe to call cross-thread).
"""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path


class QueryTimeout(Exception):
    """Raised when a query exceeds the allotted wall-clock budget."""


def execute_sql(sql: str, db_path: str, timeout_s: int = 30) -> list[tuple]:
    """Execute one SQL statement and return all rows as a list of tuples.

    Raises on syntax errors, timeouts, or any other execution failure — callers
    (the reward function) translate those into reward 0.
    """
    if not Path(db_path).exists():
        raise FileNotFoundError(f"database not found: {db_path}")

    uri = f"file:{Path(db_path).as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    timed_out = {"flag": False}

    def _interrupt() -> None:
        timed_out["flag"] = True
        con.interrupt()

    timer = threading.Timer(timeout_s, _interrupt)
    timer.start()
    try:
        cur = con.execute(sql)
        rows = cur.fetchall()
    except sqlite3.OperationalError as exc:
        if timed_out["flag"]:
            raise QueryTimeout(f"query exceeded {timeout_s}s") from exc
        raise
    finally:
        timer.cancel()
        con.close()

    if timed_out["flag"]:
        raise QueryTimeout(f"query exceeded {timeout_s}s")
    return rows
