"""Main bandit experiment loop (spec §11.2).

For each seed: deterministically shuffle the BIRD dev subset and run the
ε-greedy bandit for N episodes, writing one JSONL log per seed. Sequential by
design (spec §14) — the response cache keeps re-runs cheap.

    python -m scripts.run_experiment --seeds 0,1,2 --n-episodes 60
"""
from __future__ import annotations

import argparse

from src import config
from src.experiment import run_bandit
from src.llm import build_client
from src.tasks import load_tasks
from src.workflows import build_workflows


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the ε-greedy bandit over BIRD dev.")
    ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
    ap.add_argument("--n-episodes", type=int, default=config.N_EPISODES)
    ap.add_argument("--epsilon", type=float, default=config.EPSILON)
    ap.add_argument("--provider", default=config.LLM_PROVIDER)
    ap.add_argument("--prefix", default="exp")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap tasks loaded before shuffling (None = all)")
    ap.add_argument("--db-subset", default="",
                    help="comma-separated db_ids; empty = config.BIRD_DB_SUBSET or all")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    subset = [s for s in args.db_subset.split(",") if s] or (config.BIRD_DB_SUBSET or None)
    tasks = load_tasks(db_subset=subset, limit=args.limit)
    workflows = build_workflows()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
    print(f"loaded {len(tasks)} tasks; seeds={seeds}; n_episodes={args.n_episodes}")

    for seed in seeds:
        llm = build_client(provider=args.provider, cache_enabled=not args.no_cache)
        run_bandit(
            tasks=tasks, workflows=workflows, llm=llm,
            run_id=f"{args.prefix}_bandit_seed{seed}", seed=seed,
            n_episodes=args.n_episodes, epsilon=args.epsilon,
        )
        print(f"seed {seed}: bandit complete -> logs/{args.prefix}_bandit_seed{seed}.jsonl "
              f"(llm calls={llm.total_calls}, cache hits={llm.cache_hits})")

    print(f"\nNext:\n  python -m scripts.run_baselines --seeds {args.seeds} "
          f"--n-episodes {args.n_episodes} --prefix {args.prefix}\n"
          f"  python -m scripts.plot_results --prefix {args.prefix}")


if __name__ == "__main__":
    main()
