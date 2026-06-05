"""Pre-flight workflow-differentiation check (spec §11.1) — the gate before the bandit.

Runs all three workflows on a fixed set of tasks at temperature 0 and reports
per-workflow success rate. If the arms aren't sufficiently differentiated, the
bandit has nothing to learn — redesign the workflows before proceeding.

    python -m scripts.smoke_test                # real LLM + BIRD dev
    python -m scripts.smoke_test --fake         # offline, on the synthetic fixture
"""
from __future__ import annotations

import argparse

from src import config
from src.llm import FakeLLMClient, build_client
from src.reward import compute_reward
from src.tasks import load_fixture_tasks, load_tasks
from src.workflows import build_workflows

SPREAD_GATE = 0.05  # >= 5 percentage points between best and worst (spec §11.1)


def evaluate(tasks, workflows, llm) -> dict:
    """Return {task_id: [reward_per_arm]} running every arm on every task."""
    grid = {}
    for t in tasks:
        row = []
        for wf in workflows:
            try:
                result = wf.run(t, llm)
                reward, _ = compute_reward(result.sql, t.gold_sql, t.db_path)
            except Exception:  # noqa: BLE001
                reward = 0
            row.append(reward)
        grid[t.task_id] = row
        print(f"  task {t.task_id} [{t.difficulty:>11}]: {row}")
    return grid


def main() -> None:
    ap = argparse.ArgumentParser(description="Workflow differentiation smoke test.")
    ap.add_argument("--n", type=int, default=config.SMOKE_N_TASKS)
    ap.add_argument("--fake", action="store_true", help="offline run on the synthetic fixture")
    ap.add_argument("--provider", default=config.LLM_PROVIDER)
    ap.add_argument("--db-subset", default="")
    args = ap.parse_args()

    workflows = build_workflows()

    if args.fake:
        from scripts.dry_run import make_responder
        from scripts.make_fixture import ensure_fixture

        ensure_fixture()
        tasks = load_fixture_tasks()
        llm = FakeLLMClient(make_responder(tasks))
        print(f"[fake] {len(tasks)} fixture tasks")
    else:
        subset = [s for s in args.db_subset.split(",") if s] or (config.BIRD_DB_SUBSET or None)
        tasks = load_tasks(db_subset=subset, limit=args.n)
        llm = build_client(provider=args.provider)
        print(f"{len(tasks)} BIRD tasks, provider={args.provider}, model={llm.model}")

    grid = evaluate(tasks, workflows, llm)

    n = len(grid) or 1
    rates = [sum(grid[tid][a] for tid in grid) / n for a in range(config.N_ARMS)]
    print("\nPer-workflow success rate:")
    for a, name in enumerate(config.ARM_NAMES):
        print(f"  {name:<16} {rates[a]:.3f}")

    spread = max(rates) - min(rates)
    gate = "PASS" if spread >= SPREAD_GATE else "FAIL"
    print(f"\nbest-worst spread = {spread:.3f}  (gate >= {SPREAD_GATE}) -> {gate}")
    if gate == "FAIL":
        print("Workflows are too similar — redesign before running the bandit (spec §11.1, §17).")


if __name__ == "__main__":
    main()
