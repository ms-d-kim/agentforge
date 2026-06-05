"""One episode: select arm -> run workflow -> reward -> update -> log (spec §5).

This is the seam the whole experiment turns on. ``run_episode`` is provider- and
script-agnostic: pass any bandit, any list of workflows (the arms), any LLM
client (real or fake), and a logger.
"""
from __future__ import annotations

from .logger import episode_record
from .reward import compute_reward
from .workflows.base import WorkflowResult


def run_episode(
    *,
    task,
    bandit,
    workflows,
    llm,
    logger,
    run_id: str,
    episode_idx: int,
    seed: int,
    forced_arm: int | None = None,
) -> dict:
    """Run a single episode and return the logged record.

    If ``forced_arm`` is given, that arm is used instead of the bandit's choice
    (used by baselines: random, single-arm, oracle data collection). The bandit
    is still updated, so single-arm baselines accumulate honest per-arm means.
    """
    if forced_arm is None:
        arm, reason = bandit.select_arm()
    else:
        arm, reason = forced_arm, "forced_baseline"

    workflow = workflows[arm]
    try:
        result = workflow.run(task, llm)
    except Exception as exc:  # noqa: BLE001 — a workflow blow-up is reward 0, not a crash
        result = WorkflowResult(
            sql="", intermediate_outputs={"workflow_error": f"run_failed: {exc}"}
        )

    reward, exec_error = compute_reward(result.sql, task.gold_sql, task.db_path)
    if exec_error is None:
        exec_error = result.intermediate_outputs.get("workflow_error")

    bandit.update(arm, reward)
    snap = bandit.snapshot()

    record = episode_record(
        run_id=run_id,
        episode_idx=episode_idx,
        seed=seed,
        task=task,
        selected_arm=arm,
        workflow_name=workflow.name,
        selection_reason=reason,
        generated_sql=result.sql,
        reward=reward,
        execution_error=exec_error,
        latency_ms=result.latency_ms,
        llm_calls=result.llm_calls,
        tokens_used=result.tokens_used,
        arm_means_after=snap["means"],
        arm_counts_after=snap["counts"],
        epsilon=bandit.epsilon,
        model=getattr(llm, "model", "unknown"),
    )
    logger.log(record)
    return record
