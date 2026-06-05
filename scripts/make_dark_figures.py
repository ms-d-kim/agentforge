"""Dark-themed variants of the result figures, for the demo video (black slides).

Regenerates from the SAME data as the light figures (logs/, results/sweep.json, a fresh
REINFORCE run) but styled for clear readability on black. Writes to results/dark/. The
light figures (results/*.png) are untouched — the README keeps those.

    python -m scripts.make_dark_figures
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from scripts._style import (  # noqa: E402
    BG, BLUE, FG, GREEN, HEAD, MAGENTA, MUTED, ORANGE, SERIES, TEAL, VIOLET, apply_dark_rc,
)

apply_dark_rc()
import matplotlib.pyplot as plt  # noqa: E402

from src import metrics  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "results" / "dark"


def _title(ax, text):
    ax.set_title(text, fontfamily=HEAD, fontsize=15, color=FG, fontweight="bold", pad=12)


def bird():
    logs = REPO / "logs"

    def band(prefix, label, color, ls="-"):
        files = sorted(glob.glob(str(logs / f"{prefix}_seed*.jsonl")))
        if not files:
            return
        cums = [metrics.cumulative(metrics.rewards(metrics.load_jsonl(f))) for f in files]
        m, s = metrics.mean_std_across(cums)
        x = range(1, len(m) + 1)
        plt.plot(x, m, ls, label=label, color=color, lw=2.2 if ls == "-" else 1.6,
                 alpha=1.0 if ls == "-" else 0.75)
        if ls == "-":
            plt.fill_between(x, [a - b for a, b in zip(m, s)], [a + b for a, b in zip(m, s)],
                             color=color, alpha=0.13)

    plt.figure(figsize=(10, 5.6))
    band("exp_bandit", "bandit (ε-greedy)", BLUE)
    band("exp_random", "random", ORANGE)
    band("exp_oracle", "oracle (best-of-N)", GREEN)
    for a, (nm, col) in enumerate([("direct", VIOLET), ("schema_explore", TEAL), ("decompose", MUTED)]):
        band(f"exp_arm{a}", f"always {nm}", col, ls="--")
    plt.xlabel("episode")
    plt.ylabel("cumulative reward")
    _title(plt.gca(), "Cumulative reward — BIRD-SQL")
    plt.legend(fontsize=8.5, labelcolor=FG, loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT / "exp_01_cumulative_reward.png", dpi=120)
    plt.close()


def sweep():
    data = json.loads((REPO / "results" / "sweep.json").read_text())
    mat, pols, doms = data["matrix"], data["policies"], list(data["matrix"])
    x = range(len(doms))
    width = 0.25
    plt.figure(figsize=(9.5, 5.2))
    for j, pol in enumerate(pols):
        means = [mat[d][pol]["final_mean"] for d in doms]
        stds = [mat[d][pol]["final_std"] for d in doms]
        plt.bar([xi + (j - 1) * width for xi in x], means, width, yerr=stds, capsize=3,
                label=pol, color=SERIES[j], edgecolor=BG, error_kw=dict(ecolor=MUTED))
    plt.xticks(list(x), doms)
    plt.ylim(0, 1)
    plt.ylabel("final rolling accuracy")
    _title(plt.gca(), "Selector across domains × policies (3 seeds)")
    plt.legend(fontsize=9, labelcolor=FG)
    plt.tight_layout()
    plt.savefig(OUT / "sweep_matrix.png", dpi=120)
    plt.close()


def rl():
    from src.rl.reinforce import evaluate, fixed_horizon_baseline, random_baseline, train

    agent, hist = train(episodes=5000, seed=0)

    def roll(h, w=200):
        return [sum(s for s, _ in h[max(0, i - w + 1): i + 1]) / len(h[max(0, i - w + 1): i + 1])
                for i in range(len(h))]

    sr, _ = evaluate(agent, episodes=500)
    best = max(fixed_horizon_baseline(k, episodes=500) for k in range(4))
    rnd = random_baseline(episodes=500)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    r = roll(hist)
    ax1.plot(range(1, len(r) + 1), r, color=BLUE, lw=2.2, label="REINFORCE (learned)")
    ax1.axhline(best, ls="--", color=MUTED, label=f"best fixed horizon ({best:.2f})")
    ax1.axhline(rnd, ls=":", color="#55555E", label=f"random ({rnd:.2f})")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("rolling success rate")
    ax1.set_ylim(0, 1)
    _title(ax1, "Multi-step RL learns when to stop")
    ax1.legend(fontsize=8.5, labelcolor=FG)
    ax2.bar(["REINFORCE", "best fixed", "random"], [sr, best, rnd],
            color=[GREEN, MUTED, "#3A3A42"], edgecolor=BG)
    for i, v in enumerate([sr, best, rnd]):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", color=FG, fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("success rate")
    _title(ax2, "Learned vs. fixed-horizon")
    plt.tight_layout()
    plt.savefig(OUT / "rl_multihop.png", dpi=120)
    plt.close()


def architecture():
    from scripts.make_architecture import render
    render(OUT / "architecture.png", dark=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bird()
    sweep()
    rl()
    architecture()
    print(f"dark figures → {OUT}")


if __name__ == "__main__":
    main()
