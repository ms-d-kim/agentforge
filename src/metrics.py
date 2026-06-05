"""Turn JSONL episode logs into the series the plots need (spec §12, CLAUDE.md).

Pure functions over lists of records — no plotting here, so they're unit-testable
and reusable. Field names match src/logger.episode_record.
"""
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    records.sort(key=lambda r: r.get("episode_idx", 0))
    return records


def rewards(records: list[dict]) -> list[int]:
    return [int(r["reward"]) for r in records]


def cumulative(seq) -> list[float]:
    out, total = [], 0.0
    for x in seq:
        total += x
        out.append(total)
    return out


def rolling_accuracy(records: list[dict], window: int = 10) -> list[float]:
    rw = rewards(records)
    out = []
    for i in range(len(rw)):
        w = rw[max(0, i - window + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def arm_pull_fractions(records: list[dict], n_arms: int) -> list[list[float]]:
    """Cumulative fraction of pulls each arm has received, per episode."""
    counts = [0] * n_arms
    series = [[] for _ in range(n_arms)]
    for i, r in enumerate(records):
        counts[r["selected_arm"]] += 1
        total = i + 1
        for a in range(n_arms):
            series[a].append(counts[a] / total)
    return series


def per_arm_means_over_time(records: list[dict], n_arms: int) -> list[list[float]]:
    """What the bandit 'thinks' of each arm over time (from arm_means_after)."""
    series = [[] for _ in range(n_arms)]
    for r in records:
        means = r.get("arm_means_after", [0.0] * n_arms)
        for a in range(n_arms):
            series[a].append(means[a] if a < len(means) else 0.0)
    return series


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion (spec §12)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def per_arm_summary(records: list[dict], n_arms: int) -> list[dict]:
    pulls = [0] * n_arms
    succ = [0] * n_arms
    for r in records:
        a = r["selected_arm"]
        pulls[a] += 1
        succ[a] += int(r["reward"])
    rows = []
    for a in range(n_arms):
        rate = succ[a] / pulls[a] if pulls[a] else 0.0
        lo, hi = wilson_interval(succ[a], pulls[a])
        rows.append(
            {"arm": a, "pulls": pulls[a], "successes": succ[a], "rate": rate, "ci": (lo, hi)}
        )
    return rows


def _by_episode(records: list[dict]) -> dict[int, int]:
    return {r["episode_idx"]: int(r["reward"]) for r in records}


def cumulative_regret(bandit_records: list[dict], oracle_records: list[dict]) -> list[float]:
    """Cumulative (oracle_reward - bandit_reward), aligned by episode index.

    Same seed => same task order => episode_idx is a valid join key.
    """
    oracle = _by_episode(oracle_records)
    out, total = [], 0.0
    for r in bandit_records:
        i = r["episode_idx"]
        total += oracle.get(i, r["reward"]) - int(r["reward"])
        out.append(total)
    return out


def mean_std_across(seqs: list[list[float]]) -> tuple[list[float], list[float]]:
    """Element-wise mean and population std across equal-ish-length sequences."""
    if not seqs:
        return [], []
    n = min(len(s) for s in seqs)
    means, stds = [], []
    for i in range(n):
        vals = [s[i] for s in seqs]
        means.append(sum(vals) / len(vals))
        stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
    return means, stds
