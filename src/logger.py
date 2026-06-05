"""Append-only JSONL episode logging (spec §10).

One file per run at ``logs/{run_id}.jsonl``, one JSON object per episode. Field
names are stable once chosen — the plots depend on them. ``episode_record``
builds the canonical schema so callers can't drift field names.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from .config import LOGS_DIR


def _utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def episode_record(
    *,
    run_id: str,
    episode_idx: int,
    seed: int,
    task,                       # src.tasks.Task
    selected_arm: int,
    workflow_name: str,
    selection_reason: str,
    generated_sql: str,
    reward: int,
    execution_error: str | None,
    latency_ms: int,
    llm_calls: int,
    tokens_used: int,
    arm_means_after: list[float],
    arm_counts_after: list[int],
    epsilon: float,
    model: str,
) -> dict:
    """Construct one canonical JSONL record (spec §10 schema)."""
    return {
        "run_id": run_id,
        "episode_idx": episode_idx,
        "timestamp": _utcnow_iso(),
        "seed": seed,
        "task_id": task.task_id,
        "db_id": task.db_id,
        "question": task.question,
        "difficulty": task.difficulty,
        "selected_arm": selected_arm,
        "workflow_name": workflow_name,
        "selection_reason": selection_reason,
        "generated_sql": generated_sql,
        "gold_sql": task.gold_sql,
        "reward": reward,
        "exec_match": bool(reward),
        "execution_error": execution_error,
        "latency_ms": latency_ms,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "arm_means_after": list(arm_means_after),
        "arm_counts_after": list(arm_counts_after),
        "epsilon": epsilon,
        "model": model,
    }


class JSONLLogger:
    """Append JSON objects to ``logs/{run_id}.jsonl``."""

    def __init__(self, run_id: str, logs_dir: str | Path = LOGS_DIR):
        self.run_id = run_id
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.logs_dir / f"{run_id}.jsonl"

    def log(self, record: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def __repr__(self) -> str:  # pragma: no cover
        return f"JSONLLogger(path={self.path})"
