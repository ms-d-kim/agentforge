"""Render the AgentForge architecture flowchart.

    python -m scripts.make_architecture            # light (results/architecture.png, for README)

The self-improving selection loop: a task enters, the WorkflowSelector's policy picks
an arm, the chosen strategy runs, a reward scores the output, and the policy updates
online and persists. Domains plug in their own arms + reward. `render(path, dark=True)`
produces the on-black variant used by the demo video.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from scripts._style import HEAD, MONO  # noqa: E402

LIGHT = dict(
    bg="white", title="#0B1F3A", accent="#1F6FEB", taskfill="white", taskedge="#2B4A78",
    boxfill="#EEF3FB", boxedge="#2B4A78", boxtext="#0B1F3A", selfill="#DBE9FF", seledge="#1F6FEB",
    rewfill="#DCF3E6", rewedge="#1B7F4B", persistfill="#FFF1D6", persistedge="#9A7B22",
    persisttext="#0B1F3A", domfill="#F2F6FF", domedge="#9DB2CE", domtext="#0B1F3A",
    flow="#1F6FEB", feedback="#1B7F4B", persistarrow="#9A7B22", domarrow="#9DB2CE",
)
DARK = dict(
    bg="#0A0A0B", title="#FFFFFF", accent="#5AA9FF", taskfill="#15151A", taskedge="#3A3A44",
    boxfill="#15151A", boxedge="#3A3A44", boxtext="#E8E8EE", selfill="#11233A", seledge="#5AA9FF",
    rewfill="#10271B", rewedge="#2BE06B", persistfill="#2A220E", persistedge="#C7A24A",
    persisttext="#C9C9D2", domfill="#101018", domedge="#3A3A44", domtext="#9AA0AA",
    flow="#5AA9FF", feedback="#2BE06B", persistarrow="#C7A24A", domarrow="#4A4A55",
)


def _box(ax, cx, cy, w, h, text, fill, edge, tc, fs=14, bold=False, font=MONO):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                 boxstyle="round,pad=0.02,rounding_size=0.12", linewidth=1.8,
                 edgecolor=edge, facecolor=fill, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontfamily=font, fontweight="bold" if bold else "normal", zorder=3)


def _arrow(ax, p1, p2, color, label=None, rad=0.0, lx=0, ly=0.28, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                 mutation_scale=18, linewidth=2.0, color=color, zorder=1, linestyle=ls))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=12, color=color,
                fontfamily=MONO, zorder=4)


def render(path, dark=False):
    t = DARK if dark else LIGHT
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(t["bg"])
    ax.set_facecolor(t["bg"])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8, 8.5, "AgentForge — the self-improving selection loop", ha="center",
            fontsize=22, fontweight="bold", color=t["title"], fontfamily=HEAD)
    ax.text(8, 7.95, "policy ladder:  ε-greedy → UCB1 → Thompson → LinUCB (contextual) → REINFORCE",
            ha="center", fontsize=12.5, color=t["accent"], fontfamily=MONO)

    y = 5.7
    _box(ax, 1.5, y, 2.0, 1.0, "Task", t["taskfill"], t["taskedge"], t["title"], bold=True, font=HEAD)
    _box(ax, 5.2, y, 3.0, 1.7, "WorkflowSelector\n\n(online policy)", t["selfill"], t["seledge"],
         t["title"], fs=16, bold=True, font=HEAD)
    _box(ax, 9.3, y, 2.4, 1.4, "Arms\nstrategies\n(per domain)", t["boxfill"], t["boxedge"], t["boxtext"])
    _box(ax, 12.7, y, 1.9, 1.0, "Output", t["taskfill"], t["taskedge"], t["title"])
    _box(ax, 15.0, y, 1.9, 1.4, "reward\n(task, output)", t["rewfill"], t["rewedge"], t["title"],
         fs=13, bold=True)

    _arrow(ax, (2.5, y), (3.7, y), t["flow"])
    _arrow(ax, (6.7, y), (8.1, y), t["flow"], "select arm")
    _arrow(ax, (10.5, y), (11.75, y), t["flow"], "run")
    _arrow(ax, (13.65, y), (14.05, y), t["flow"])

    _arrow(ax, (15.0, y - 0.7), (5.2, 4.2), t["feedback"], rad=-0.16)
    _arrow(ax, (5.2, 4.2), (5.2, y - 0.85), t["feedback"])
    ax.text(10.3, 4.62, "reward  →  update policy (online)", ha="center", fontsize=13,
            color=t["feedback"], fontweight="bold", fontfamily=MONO)

    _box(ax, 2.7, 2.8, 3.3, 0.95, "state.json\npersists across runs", t["persistfill"],
         t["persistedge"], t["persisttext"], fs=12)
    _arrow(ax, (4.5, 5.0), (3.2, 3.35), t["persistarrow"], ls="--")

    _box(ax, 9.7, 1.25, 11.4, 1.25,
         "Four domains plug into the same loop — each supplies its own arms + reward:\n"
         "text-to-SQL (exec-match) · code (tests pass) · search (file-match) · compaction (downstream-QA)",
         t["domfill"], t["domedge"], t["domtext"], fs=13)

    fig.tight_layout()
    fig.savefig(path, facecolor=t["bg"], bbox_inches="tight")
    plt.close(fig)


def main():
    out = Path(__file__).resolve().parent.parent / "results" / "architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(out, dark=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
