"""BIRD-SQL task loading (spec §6).

The ``Task`` dataclass is the unit the whole pipeline operates on. ``load_tasks``
reads BIRD's ``dev.json`` and resolves each task's SQLite path. Path discovery is
deliberately tolerant: BIRD's dev zip has shifted its top-level folder name across
releases, so we ``rglob`` for ``dev.json`` and ``dev_databases`` under the data
root rather than hardcoding one rigid layout.

BIRD field-name notes (verified against the dev set):
  - the gold SQL key is uppercase ``SQL``
  - ``evidence`` may be null/empty
  - ``question_id`` is the stable task id; ``difficulty`` is simple/moderate/challenging
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import BIRD_DIR, FIXTURE_DIR


@dataclass(frozen=True)
class Task:
    task_id: str       # BIRD's "question_id"
    db_id: str         # which database
    db_path: str       # path to the .sqlite file
    question: str      # natural-language question
    evidence: str      # BIRD's external-knowledge hint (may be "")
    gold_sql: str      # ground-truth SQL
    difficulty: str    # BIRD's labeled difficulty


def _discover(root: Path) -> tuple[Path | None, Path | None]:
    """Find (dev.json, dev_databases/) anywhere under ``root``, ignoring the
    macOS ``__MACOSX`` resource-fork junk that ships inside BIRD's zips."""

    def ok(p: Path) -> bool:
        return "__MACOSX" not in p.parts

    dev_json = next((p for p in sorted(root.rglob("dev.json")) if ok(p)), None)
    db_root = next((p for p in sorted(root.rglob("dev_databases")) if ok(p)), None)
    return dev_json, db_root


def load_tasks(
    bird_dir: str | Path = BIRD_DIR,
    db_subset: list[str] | None = None,
    limit: int | None = None,
    dev_json: str | Path | None = None,
    db_root: str | Path | None = None,
) -> list[Task]:
    """Load BIRD dev tasks.

    Args:
        bird_dir: root to search when explicit paths aren't given.
        db_subset: if set, keep only tasks whose ``db_id`` is in this list.
        limit: cap the number of tasks returned (after subsetting).
        dev_json / db_root: explicit overrides for the questions file and the
            directory containing per-database folders.
    """
    root = Path(bird_dir)
    if dev_json is None or db_root is None:
        found_json, found_dbs = _discover(root)
        dev_json = dev_json or found_json
        db_root = db_root or found_dbs

    if not dev_json or not Path(dev_json).exists():
        raise FileNotFoundError(
            f"Could not find dev.json under {root}. "
            "Run `python -m scripts.download_bird` first, or pass dev_json=... explicitly."
        )
    if not db_root or not Path(db_root).exists():
        raise FileNotFoundError(
            f"Could not find a dev_databases/ directory under {root}. "
            "Did the database archive get unzipped? See scripts/download_bird.py."
        )

    entries = json.loads(Path(dev_json).read_text())
    db_root = Path(db_root)
    subset = set(db_subset) if db_subset else None

    tasks: list[Task] = []
    for e in entries:
        db_id = e["db_id"]
        if subset is not None and db_id not in subset:
            continue
        task_id = str(e.get("question_id", e.get("task_id", len(tasks))))
        db_path = db_root / db_id / f"{db_id}.sqlite"
        tasks.append(
            Task(
                task_id=task_id,
                db_id=db_id,
                db_path=str(db_path),
                question=e["question"],
                evidence=(e.get("evidence") or "").strip(),
                gold_sql=(e.get("SQL") or e.get("sql") or "").strip(),
                difficulty=(e.get("difficulty") or "unknown"),
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def load_fixture_tasks(limit: int | None = None) -> list[Task]:
    """Load the tiny synthetic fixture used for offline pipeline tests."""
    return load_tasks(
        bird_dir=FIXTURE_DIR,
        dev_json=FIXTURE_DIR / "dev.json",
        db_root=FIXTURE_DIR / "dev_databases",
        limit=limit,
    )
