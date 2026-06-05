"""Shared experiment loops used by all scripts (spec §11).

Keeps the seeded task ordering and the four conditions (bandit, random,
single-arm, oracle) in one place so run_experiment / run_baselines / dry_run
can't drift apart. Sequential only — reproducibility over wall-clock (spec §14).
"""
from __future__ import annotations

import random

from . import config
from .bandit import EpsilonGreedyBandit
from .episode import run_episode
from .logger import JSONLLogger
from .reward import compute_reward


def shuffle_tasks(tasks, seed: int):
    """Deterministic per-seed task order (without replacement)."""
    rng = random.Random(seed)
    ordered = list(tasks)
    rng.shuffle(ordered)
    return ordered


def _episode_slice(tasks, seed, n_episodes):
    ordered = shuffle_tasks(tasks, seed)
    if n_episodes is not None:
        ordered = ordered[:n_episodes]
    return ordered


def run_bandit(
    *,
    tasks,
    workflows,
    llm,
    run_id: str,
    seed: int,
    n_episodes: int | None = None,
    epsilon: float = config.EPSILON,
    forced_init: bool = config.FORCED_INIT,
    logs_dir=config.LOGS_DIR,
) -> list[dict]:
    ordered = _episode_slice(tasks, seed, n_episodes)
    bandit = EpsilonGreedyBandit(
        len(workflows), epsilon=epsilon, seed=seed, forced_init=forced_init
    )
    logger = JSONLLogger(run_id, logs_dir)
    return [
        run_episode(
            task=task, bandit=bandit, workflows=workflows, llm=llm,
            logger=logger, run_id=run_id, episode_idx=i, seed=seed,
        )
        for i, task in enumerate(ordered)
    ]


def run_fixed_arm(
    *, tasks, workflows, llm, run_id, seed, arm: int,
    n_episodes=None, logs_dir=config.LOGS_DIR,
) -> list[dict]:
    """Single-arm baseline: always pull ``arm`` (spec §11.3)."""
    ordered = _episode_slice(tasks, seed, n_episodes)
    bandit = EpsilonGreedyBandit(len(workflows), seed=seed, forced_init=False)
    logger = JSONLLogger(run_id, logs_dir)
    return [
        run_episode(
            task=task, bandit=bandit, workflows=workflows, llm=llm,
            logger=logger, run_id=run_id, episode_idx=i, seed=seed, forced_arm=arm,
        )
        for i, task in enumerate(ordered)
    ]


def run_random(
    *, tasks, workflows, llm, run_id, seed, n_episodes=None, logs_dir=config.LOGS_DIR,
) -> list[dict]:
    """Random baseline: uniform arm choice, no learning (spec §11.3)."""
    ordered = _episode_slice(tasks, seed, n_episodes)
    bandit = EpsilonGreedyBandit(len(workflows), seed=seed, forced_init=False)
    rng = random.Random(seed * 7919 + 1)  # independent of the bandit's RNG
    logger = JSONLLogger(run_id, logs_dir)
    records = []
    for i, task in enumerate(ordered):
        arm = rng.randrange(len(workflows))
        records.append(
            run_episode(
                task=task, bandit=bandit, workflows=workflows, llm=llm,
                logger=logger, run_id=run_id, episode_idx=i, seed=seed, forced_arm=arm,
            )
        )
    return records


def run_oracle(
    *, tasks, workflows, llm, run_id, seed, n_episodes=None, logs_dir=config.LOGS_DIR,
) -> list[dict]:
    """Post-hoc oracle: run every arm per task, take best-of-N (spec §11.3).

    Writes one record per task with reward = max over arms and selected_arm = the
    arm that achieved it. This is the upper bound the bandit cannot beat; the
    response cache means re-running it is cheap.
    """
    ordered = _episode_slice(tasks, seed, n_episodes)
    logger = JSONLLogger(run_id, logs_dir)
    from .logger import episode_record

    records = []
    for i, task in enumerate(ordered):
        best_reward, best_arm, best_sql, best_err = 0, 0, "", None
        total_calls = total_tokens = total_latency = 0
        for arm, wf in enumerate(workflows):
            try:
                result = wf.run(task, llm)
            except Exception as exc:  # noqa: BLE001
                result_sql, calls, tokens, latency, err = "", 0, 0, 0, f"run_failed: {exc}"
                reward = 0
            else:
                reward, err = compute_reward(result.sql, task.gold_sql, task.db_path)
                result_sql, calls, tokens, latency = (
                    result.sql, result.llm_calls, result.tokens_used, result.latency_ms,
                )
            total_calls += calls
            total_tokens += tokens
            total_latency += latency
            if reward > best_reward:
                best_reward, best_arm, best_sql, best_err = reward, arm, result_sql, err
        rec = episode_record(
            run_id=run_id, episode_idx=i, seed=seed, task=task,
            selected_arm=best_arm, workflow_name="oracle",
            selection_reason="oracle_best_of_n", generated_sql=best_sql,
            reward=best_reward, execution_error=best_err, latency_ms=total_latency,
            llm_calls=total_calls, tokens_used=total_tokens,
            arm_means_after=[0.0] * len(workflows),
            arm_counts_after=[0] * len(workflows),
            epsilon=0.0, model=getattr(llm, "model", "unknown"),
        )
        logger.log(rec)
        records.append(rec)
    return records
