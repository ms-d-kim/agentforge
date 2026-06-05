"""End-to-end offline smoke run — NO API calls, NO BIRD download (test harness).

Drives the *real* pipeline (workflows -> bandit -> reward -> logger) against the
synthetic fixture with a scripted "fake" LLM whose success profile differs by
arm, so the bandit has something to learn:

    schema_explore : solves all 5 questions   (best, ~1.00)
    direct         : misses the hard join      (~0.80)
    decompose      : misses two easy ones       (~0.60)

Writes bandit / random / single-arm / oracle logs under logs/ with the ``dry``
prefix; run `python -m scripts.plot_results --prefix dry` to render the 4 plots.
"""
from __future__ import annotations

import argparse
import json
import random

from src import config
from src.experiment import run_bandit, run_fixed_arm, run_oracle, run_random
from src.llm import FakeLLMClient
from src.tasks import load_fixture_tasks
from src.workflows import build_workflows
from scripts.make_fixture import ensure_fixture

# Per-question success sets keyed by task_id (string). Defines arm differentiation.
DIRECT_OK = {"1", "2", "4", "5"}        # misses #3 (hard join/group/order)
EXPLORE_OK = {"1", "2", "3", "4", "5"}  # solves everything
DECOMPOSE_OK = {"1", "3", "5"}          # over-decomposes the two simple list queries
WRONG = "```sql\nSELECT 0\n```"


def make_responder(tasks):
    """Build a scripted responder(prompt, system) -> str over the fixture tasks."""
    by_question = {t.question: t for t in tasks}

    def gold(t):
        return f"```sql\n{t.gold_sql}\n```"

    def find_task(prompt: str):
        for q, t in by_question.items():
            if q in prompt:
                return t
        return None

    def responder(prompt: str, system) -> str:
        t = find_task(prompt)
        # Intermediate steps (arm-agnostic):
        if "identify which tables are relevant" in prompt:
            return '["schools", "students"]'
        if "Decompose the question below" in prompt:
            return json.dumps([t.question if t else "?"])
        if t is None:
            return WRONG
        # Final SQL-producing steps, distinguished by their unique prompt markers:
        if "already filtered to be relevant" in prompt:          # explore arm (WRITE)
            return gold(t) if t.task_id in EXPLORE_OK else WRONG
        if "draft SQL for each sub-question" in prompt:          # decompose arm (COMPOSE)
            return gold(t) if t.task_id in DECOMPOSE_OK else WRONG
        if "single sub-question" in prompt:                      # decompose SUB_SQL (feeds compose)
            return gold(t)
        if "Given the following SQLite database schema" in prompt:  # direct arm
            return gold(t) if t.task_id in DIRECT_OK else WRONG
        return gold(t)

    return responder


def _sample_with_replacement(tasks, n, seed):
    rng = random.Random(1000 + seed)
    return [rng.choice(tasks) for _ in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline end-to-end dry run on the fixture.")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--seeds", type=str, default=",".join(map(str, config.SEEDS)))
    ap.add_argument("--prefix", type=str, default="dry")
    args = ap.parse_args()

    ensure_fixture()
    fixture_tasks = load_fixture_tasks()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    workflows = build_workflows()

    for seed in seeds:
        llm = FakeLLMClient(make_responder(fixture_tasks))
        episode_tasks = _sample_with_replacement(fixture_tasks, args.n_episodes, seed)
        kw = dict(tasks=episode_tasks, workflows=workflows, llm=llm,
                  seed=seed, n_episodes=args.n_episodes)
        run_bandit(run_id=f"{args.prefix}_bandit_seed{seed}", **kw)
        run_random(run_id=f"{args.prefix}_random_seed{seed}", **kw)
        for arm in range(config.N_ARMS):
            run_fixed_arm(run_id=f"{args.prefix}_arm{arm}_seed{seed}", arm=arm, **kw)
        run_oracle(run_id=f"{args.prefix}_oracle_seed{seed}", **kw)
        print(f"seed {seed}: wrote bandit / random / 3 single-arm / oracle logs")

    print(f"\nDone. Render plots with:\n  python -m scripts.plot_results --prefix {args.prefix}")


if __name__ == "__main__":
    main()
