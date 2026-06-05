"""Baselines for comparison against the bandit (spec §11.3).

Same seeded task ordering as run_experiment, so episodes align by index:
  - random        : uniform arm choice, no learning
  - single-arm    : always-direct / always-explore / always-decompose
  - oracle        : post-hoc best-of-N (upper bound the bandit can't beat)

    python -m scripts.run_baselines --seeds 0,1,2 --n-episodes 60 --prefix exp
"""
from __future__ import annotations

import argparse

from src import config
from src.experiment import run_fixed_arm, run_oracle, run_random
from src.llm import build_client
from src.tasks import load_tasks
from src.workflows import build_workflows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run random / single-arm / oracle baselines.")
    ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    ap.add_argument("--n-episodes", type=int, default=config.N_EPISODES)
    ap.add_argument("--provider", default=config.LLM_PROVIDER)
    ap.add_argument("--prefix", default="exp")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--db-subset", default="")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true",
                    help="oracle runs all 3 arms per task (3x cost) — skip if budget-bound")
    args = ap.parse_args()

    subset = [s for s in args.db_subset.split(",") if s] or (config.BIRD_DB_SUBSET or None)
    tasks = load_tasks(db_subset=subset, limit=args.limit)
    workflows = build_workflows()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    print(f"loaded {len(tasks)} tasks; seeds={seeds}; n_episodes={args.n_episodes}")

    for seed in seeds:
        llm = build_client(provider=args.provider, cache_enabled=not args.no_cache)
        common = dict(tasks=tasks, workflows=workflows, llm=llm, seed=seed,
                      n_episodes=args.n_episodes)
        run_random(run_id=f"{args.prefix}_random_seed{seed}", **common)
        for arm in range(config.N_ARMS):
            run_fixed_arm(run_id=f"{args.prefix}_arm{arm}_seed{seed}", arm=arm, **common)
        if not args.skip_oracle:
            run_oracle(run_id=f"{args.prefix}_oracle_seed{seed}", **common)
        print(f"seed {seed}: baselines complete (llm calls={llm.total_calls}, "
              f"cache hits={llm.cache_hits})")

    print(f"\nNext: python -m scripts.plot_results --prefix {args.prefix}")


if __name__ == "__main__":
    main()
