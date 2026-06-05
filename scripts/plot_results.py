"""Render the four evaluation plots + summary table from JSONL logs (spec §12).

Logs are grouped by run-id naming convention:
    {prefix}_bandit_seed{N}.jsonl
    {prefix}_random_seed{N}.jsonl
    {prefix}_arm{A}_seed{N}.jsonl
    {prefix}_oracle_seed{N}.jsonl

Produces, into plots/:
    {prefix}_01_cumulative_reward.png   — learning curve (headline)
    {prefix}_02_arm_fractions.png       — selection mass shifting to the best arm
    {prefix}_03_cumulative_regret.png   — vs the always-best oracle
    {prefix}_04_arm_means.png           — what the bandit thinks of each arm
    {prefix}_summary.txt                — per-arm pulls / success rate / Wilson CI
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config, metrics  # noqa: E402


def _seed_of(path: str) -> int:
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def discover(prefix: str, logs_dir: Path) -> dict:
    def g(pat):
        return sorted(glob.glob(str(logs_dir / pat)))

    groups = {
        "bandit": g(f"{prefix}_bandit_seed*.jsonl"),
        "random": g(f"{prefix}_random_seed*.jsonl"),
        "oracle": g(f"{prefix}_oracle_seed*.jsonl"),
    }
    for a in range(config.N_ARMS):
        groups[f"arm{a}"] = g(f"{prefix}_arm{a}_seed*.jsonl")
    return groups


def _load(files):
    return [metrics.load_jsonl(f) for f in files]


def _band(x, mean, std, color, alpha=0.15):
    lo = [m - s for m, s in zip(mean, std)]
    hi = [m + s for m, s in zip(mean, std)]
    plt.fill_between(x, lo, hi, color=color, alpha=alpha)


def plot_cumulative_reward(groups, out):
    plt.figure(figsize=(8, 5))

    def add_band(files, label):
        runs = _load(files)
        if not runs:
            return
        cums = [metrics.cumulative(metrics.rewards(r)) for r in runs]
        mean, std = metrics.mean_std_across(cums)
        x = list(range(1, len(mean) + 1))
        line, = plt.plot(x, mean, label=label)
        _band(x, mean, std, line.get_color())

    add_band(groups["bandit"], "bandit (ε-greedy)")
    add_band(groups["random"], "random")
    add_band(groups["oracle"], "oracle (best-of-N)")
    for a in range(config.N_ARMS):
        runs = _load(groups[f"arm{a}"])
        if not runs:
            continue
        cums = [metrics.cumulative(metrics.rewards(r)) for r in runs]
        mean, _ = metrics.mean_std_across(cums)
        plt.plot(range(1, len(mean) + 1), mean, "--", alpha=0.6,
                 label=f"always {config.ARM_NAMES[a]}")

    plt.xlabel("episode")
    plt.ylabel("cumulative reward")
    plt.title("Cumulative reward over episodes")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_arm_fractions(groups, out):
    runs = _load(groups["bandit"])
    if not runs:
        return
    per_arm = []
    for a in range(config.N_ARMS):
        series = [metrics.arm_pull_fractions(r, config.N_ARMS)[a] for r in runs]
        mean, _ = metrics.mean_std_across(series)
        per_arm.append(mean)
    n = min(len(s) for s in per_arm)
    per_arm = [s[:n] for s in per_arm]
    x = range(1, n + 1)
    plt.figure(figsize=(8, 5))
    plt.stackplot(x, *per_arm, labels=list(config.ARM_NAMES), alpha=0.85)
    plt.xlabel("episode")
    plt.ylabel("cumulative pull fraction")
    plt.ylim(0, 1)
    plt.title("Arm pull frequency over time (bandit)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_regret(groups, out):
    bandit_files, oracle_files = groups["bandit"], groups["oracle"]
    if not bandit_files or not oracle_files:
        return
    oracle_by_seed = {_seed_of(f): metrics.load_jsonl(f) for f in oracle_files}
    regrets = []
    for bf in bandit_files:
        seed = _seed_of(bf)
        if seed not in oracle_by_seed:
            continue
        regrets.append(metrics.cumulative_regret(metrics.load_jsonl(bf), oracle_by_seed[seed]))
    if not regrets:
        return
    mean, std = metrics.mean_std_across(regrets)
    x = list(range(1, len(mean) + 1))
    plt.figure(figsize=(8, 5))
    line, = plt.plot(x, mean, label="cumulative regret")
    _band(x, mean, std, line.get_color())
    plt.xlabel("episode")
    plt.ylabel("cumulative regret vs oracle")
    plt.title("Cumulative regret (sub-linear => learning)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_arm_means(groups, out):
    runs = _load(groups["bandit"])
    if not runs:
        return
    plt.figure(figsize=(8, 5))
    for a in range(config.N_ARMS):
        series = [metrics.per_arm_means_over_time(r, config.N_ARMS)[a] for r in runs]
        mean, _ = metrics.mean_std_across(series)
        plt.plot(range(1, len(mean) + 1), mean, label=config.ARM_NAMES[a])
    plt.xlabel("episode")
    plt.ylabel("running mean reward (bandit estimate)")
    plt.ylim(-0.05, 1.05)
    plt.title("Per-arm running mean reward")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def write_summary(groups, out_txt):
    runs = _load(groups["bandit"])
    all_records = [r for run in runs for r in run]
    if not all_records:
        return ""
    rows = metrics.per_arm_summary(all_records, config.N_ARMS)
    header = f"{'arm':<4}{'name':<16}{'pulls':>7}{'succ':>7}{'rate':>8}   wilson95"
    lines = [header, "-" * len(header)]
    for row in rows:
        lo, hi = row["ci"]
        lines.append(
            f"{row['arm']:<4}{config.ARM_NAMES[row['arm']]:<16}"
            f"{row['pulls']:>7}{row['successes']:>7}{row['rate']:>8.3f}"
            f"   [{lo:.3f}, {hi:.3f}]"
        )
    text = "\n".join(lines)
    Path(out_txt).write_text(text + "\n")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description="Render the 4 evaluation plots from logs.")
    ap.add_argument("--prefix", type=str, default="dry", help="run-id prefix to group logs by")
    ap.add_argument("--logs-dir", type=str, default=str(config.LOGS_DIR))
    ap.add_argument("--out-dir", type=str, default=str(config.PLOTS_DIR))
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = discover(args.prefix, logs_dir)

    if not groups["bandit"]:
        raise SystemExit(
            f"No bandit logs matching '{args.prefix}_bandit_seed*.jsonl' in {logs_dir}. "
            "Run an experiment (or `python -m scripts.dry_run`) first."
        )

    p = args.prefix
    plot_cumulative_reward(groups, out_dir / f"{p}_01_cumulative_reward.png")
    plot_arm_fractions(groups, out_dir / f"{p}_02_arm_fractions.png")
    plot_regret(groups, out_dir / f"{p}_03_cumulative_regret.png")
    plot_arm_means(groups, out_dir / f"{p}_04_arm_means.png")
    summary = write_summary(groups, out_dir / f"{p}_summary.txt")

    print(f"Plots written to {out_dir}/ (prefix '{p}'):")
    for name in ("01_cumulative_reward", "02_arm_fractions", "03_cumulative_regret", "04_arm_means"):
        print(f"  {p}_{name}.png")
    if summary:
        print("\nPer-arm summary (bandit, all seeds):\n" + summary)


if __name__ == "__main__":
    main()
