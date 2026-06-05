"""Reproducible real BIRD run via OpenRouter (the canonical 'reproduce my results' entrypoint).

Builds a FIXED, difficulty- and database-balanced pool of tasks so the three arms have room
to differ (a top model that solves everything leaves the bandit nothing to learn). Then runs
the bandit + every baseline across seeds, sharing the SQLite response cache.

Order is oracle-first: the oracle touches every (task × arm), warming the cache so the bandit,
random, and single-arm conditions become near-instant cache hits. Temperature 0 + caching make
the whole thing deterministic and cheap to re-run.

    python -m scripts.run_real --n-episodes 60 --seeds 0,1,2 \
        --dbs california_schools,financial,formula_1,superhero --prefix exp
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from src import config
from src.experiment import run_bandit, run_fixed_arm, run_oracle, run_random
from src.llm import build_client
from src.tasks import load_tasks
from src.workflows import build_workflows

DEFAULT_DBS = ["california_schools", "financial", "formula_1", "superhero"]


def build_pool(tasks, size: int, seed: int = 12345):
    """Round-robin sample across (db_id, difficulty) buckets for a balanced pool."""
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for t in tasks:
        buckets[(t.db_id, t.difficulty)].append(t)
    for b in buckets.values():
        rng.shuffle(b)
    keys = sorted(buckets)
    rng.shuffle(keys)
    pool, i = [], 0
    while len(pool) < size and any(buckets[k] for k in keys):
        k = keys[i % len(keys)]
        if buckets[k]:
            pool.append(buckets[k].pop())
        i += 1
    return pool[:size]


def main() -> None:
    ap = argparse.ArgumentParser(description="Real BIRD run via OpenRouter.")
    ap.add_argument("--n-episodes", type=int, default=60)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--dbs", default=",".join(DEFAULT_DBS))
    ap.add_argument("--provider", default="openrouter")
    ap.add_argument("--prefix", default="exp")
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    dbs = [d for d in args.dbs.split(",") if d]
    tasks = load_tasks(db_subset=dbs)
    pool = build_pool(tasks, args.n_episodes)
    workflows = build_workflows()
    seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]

    model = build_client(provider=args.provider).model
    print(f"provider={args.provider} model={model}")
    print(f"pool: {len(pool)} tasks | difficulty={dict(Counter(t.difficulty for t in pool))} "
          f"| dbs={dict(Counter(t.db_id for t in pool))}")

    for seed in seeds:
        llm = build_client(provider=args.provider)
        common = dict(tasks=pool, workflows=workflows, llm=llm, seed=seed, n_episodes=args.n_episodes)
        if not args.skip_oracle:
            run_oracle(run_id=f"{args.prefix}_oracle_seed{seed}", **common)   # warms the cache
        run_bandit(run_id=f"{args.prefix}_bandit_seed{seed}", **common)
        run_random(run_id=f"{args.prefix}_random_seed{seed}", **common)
        for arm in range(config.N_ARMS):
            run_fixed_arm(run_id=f"{args.prefix}_arm{arm}_seed{seed}", arm=arm, **common)
        print(f"seed {seed}: done (api_calls={llm.total_calls}, cache_hits={llm.cache_hits})")

    print(f"\nNext: python -m scripts.plot_results --prefix {args.prefix}")


if __name__ == "__main__":
    main()
