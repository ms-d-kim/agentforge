"""Render the AgentForge architecture flowchart -> results/architecture.png.

The self-improving selection loop: a task enters, the WorkflowSelector's policy
picks an arm, the chosen strategy runs, a reward scores the output, and the policy
updates online and persists. Domains plug in their own arms + reward. Used in both
the README and the demo video.

    python -m scripts.make_architecture
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK = "#0B1F3A"
ACCENT = "#1F6FEB"
BOX = "#EEF3FB"
SEL = "#DBE9FF"
GREEN = "#DCF3E6"
GOLD = "#FFF1D6"
EDGE = "#2B4A78"


def _box(ax, cx, cy, w, h, text, fill=BOX, fs=15, bold=False, edge=EDGE, tc=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12",
                 linewidth=1.8, edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, p1, p2, label=None, color=ACCENT, rad=0.0, lx=0, ly=0.28, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}",
                 arrowstyle="-|>", mutation_scale=18, linewidth=2.0,
                 color=color, zorder=1, linestyle=ls))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=12, color=color, zorder=4)


def render(path):
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(8, 8.5, "AgentForge — the self-improving selection loop",
            ha="center", fontsize=22, fontweight="bold", color=INK)
    ax.text(8, 7.95, "policy ladder:  ε-greedy → UCB1 → Thompson → LinUCB (contextual) → REINFORCE (multi-step)",
            ha="center", fontsize=13, color=ACCENT)

    # Main horizontal flow.
    y = 5.7
    _box(ax, 1.5, y, 2.0, 1.0, "Task", fill="white", bold=True)
    _box(ax, 5.2, y, 3.0, 1.7, "WorkflowSelector\n\n(online policy)", fill=SEL, fs=16, bold=True, edge=ACCENT)
    _box(ax, 9.3, y, 2.4, 1.4, "Arms\nstrategies\n(per domain)", fill=BOX, fs=14)
    _box(ax, 12.7, y, 1.9, 1.0, "Output", fill="white")
    _box(ax, 15.0, y, 1.9, 1.4, "reward\n(task, output)", fill=GREEN, fs=13, bold=True)

    _arrow(ax, (2.5, y), (3.7, y))
    _arrow(ax, (6.7, y), (8.1, y), "select arm")
    _arrow(ax, (10.5, y), (11.75, y), "run")
    _arrow(ax, (13.65, y), (14.05, y))

    # Feedback loop: reward -> update -> selector.
    _arrow(ax, (15.0, y - 0.7), (5.2, 4.2), color="#1B7F4B", rad=-0.16)
    _arrow(ax, (5.2, 4.2), (5.2, y - 0.85), color="#1B7F4B")
    ax.text(10.3, 4.62, "reward  →  update policy (online)", ha="center", fontsize=13,
            color="#1B7F4B", fontweight="bold")

    # Persistence.
    _box(ax, 2.7, 2.8, 3.3, 0.95, "state.json\npersists across runs", fill=GOLD, fs=12)
    _arrow(ax, (4.5, 5.0), (3.2, 3.35), color="#9A7B22", ls="--")

    # Domains plug into the same loop (labeled banner, no crossing arrows).
    _box(ax, 9.7, 1.25, 11.4, 1.25,
         "Four domains plug into the same loop — each supplies its own arms + reward:\n"
         "text-to-SQL (exec-match) · code (tests pass) · search (file-match) · compaction (downstream-QA)",
         fill="#F2F6FF", fs=13, edge="#9DB2CE")

    fig.tight_layout()
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    out = Path(__file__).resolve().parent.parent / "results" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
