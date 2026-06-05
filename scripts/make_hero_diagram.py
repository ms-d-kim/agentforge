"""Generate the shareable 'hero' diagram for AgentForge (light + dark, 16:9).

Shows the core research idea in one frame: a task enters; an epsilon-greedy
selector picks one of three concrete SQL workflows (each with its learned value
Q); the chosen workflow runs and earns a binary reward; the reward updates the
selector online. The learned Q-bars encode the actual finding -- the weak
`decompose` arm is down-weighted.

Run: python -m scripts.make_hero_diagram
  -> assets/social/hero_diagram.png        (light, for LinkedIn / light decks)
  -> assets/social/hero_diagram_dark.png   (dark, for black decks)
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.font_manager import FontProperties, fontManager

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTDIR = os.path.join(REPO, "assets", "fonts")
OUTDIR = os.path.join(REPO, "assets", "social")


def fp(fname: str, size: float, default: str = "DejaVu Sans") -> FontProperties:
    path = os.path.join(FONTDIR, fname)
    if os.path.exists(path):
        try:
            fontManager.addfont(path)
        except Exception:
            pass
        return FontProperties(fname=path, size=size)
    return FontProperties(family=default, size=size)


def SG_B(s):  return fp("SpaceGrotesk-Bold.ttf", s)
def SG_SB(s): return fp("SpaceGrotesk-SemiBold.ttf", s)
def SG_M(s):  return fp("SpaceGrotesk-Medium.ttf", s)
def JB_B(s):  return fp("JetBrainsMono-Bold.ttf", s)
def JB_M(s):  return fp("JetBrainsMono-Medium.ttf", s)
def JB_R(s):  return fp("JetBrainsMono-Regular.ttf", s)


LIGHT = dict(
    BG="#FFFFFF", INK="#0B0B0C", MUTED="#6B7280", ARROW="#374151",
    BLUE="#2563EB", BLUE_FILL="#E7EEFE",
    GREEN="#15A34A", GREEN_FILL="#DCFCE7",
    GRAYE="#9CA3AF", GRAY_FILL="#F1F2F4", TRACK="#E5E7EB",
    NEUTRAL_FILL="#FFFFFF", NEUTRAL_EC="#0B0B0C",
    TASK_FILL="#FFFFFF", TASK_EC="#0B0B0C",
    CAP_FILL="#F8FAFC", CAP_EC="#E5E7EB",
)
DARK = dict(
    BG="#0A0A0B", INK="#F4F4F6", MUTED="#9A9AA3", ARROW="#AAB0BC",
    BLUE="#6AA6FF", BLUE_FILL="#15212E",
    GREEN="#2BE06B", GREEN_FILL="#0E2A19",
    GRAYE="#5B5B64", GRAY_FILL="#161619", TRACK="#2A2A31",
    NEUTRAL_FILL="#161619", NEUTRAL_EC="#3A3A42",
    TASK_FILL="#161619", TASK_EC="#E4E4E7",
    CAP_FILL="#121214", CAP_EC="#26262C",
)


def render(T: dict, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 9), dpi=160)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.patch.set_facecolor(T["BG"]); ax.set_facecolor(T["BG"])
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.set_aspect("equal"); ax.axis("off")

    def box(cx, cy, w, h, fc, ec, lw=1.6, r=0.14, z=2):
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=1))

    def arrow(x1, y1, x2, y2, color, lw=2.4, style="-|>", ls="-", rad=0.0, z=3, ms=18):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle=style, color=color, lw=lw, linestyle=ls,
            mutation_scale=ms, shrinkA=3, shrinkB=3,
            connectionstyle=f"arc3,rad={rad}", zorder=z, capstyle="round"))

    def text(x, y, s, font, color=None, ha="center", va="center", z=4):
        ax.text(x, y, s, fontproperties=font, color=color or T["INK"], ha=ha, va=va, zorder=z)

    # ---- title ----
    text(8, 8.45, "AgentForge: an agent that learns which workflow to run", SG_B(27))
    text(8, 7.92, "Workflow selection as a multi-armed bandit — learned online from your own reward signal",
         SG_M(13.5), color=T["MUTED"])

    # ---- Task ----
    box(1.45, 4.9, 1.7, 1.0, T["TASK_FILL"], T["TASK_EC"], lw=1.6)
    text(1.45, 4.9, "Task", SG_SB(15))

    # ---- Selector ----
    box(4.2, 4.9, 2.7, 1.9, T["BLUE_FILL"], T["BLUE"], lw=2.0)
    text(4.2, 5.42, "Selector", SG_B(17))
    text(4.2, 4.98, "ε-greedy bandit", JB_M(12), color=T["BLUE"])
    text(4.2, 4.58, "keeps value Q per arm", JB_R(9.5), color=T["MUTED"])

    # ---- Arms ----
    arms = [
        ("direct",         "one LLM call",          0.39, "neutral", 6.30),
        ("schema_explore", "find tables → write", 0.41, "best",  4.90),
        ("decompose",      "split → solve → join", 0.21, "weak", 3.50),
    ]
    aL, aW = 6.7, 4.4
    for name, desc, q, status, cy in arms:
        if status == "best":
            fc, ec, nmc, barc, lw = T["GREEN_FILL"], T["GREEN"], T["INK"], T["GREEN"], 2.0
        elif status == "weak":
            fc, ec, nmc, barc, lw = T["GRAY_FILL"], T["GRAYE"], T["MUTED"], T["GRAYE"], 1.4
        else:
            fc, ec, nmc, barc, lw = T["NEUTRAL_FILL"], T["NEUTRAL_EC"], T["INK"], T["BLUE"], 1.5
        box(aL + aW / 2, cy, aW, 1.16, fc, ec, lw=lw)
        text(aL + 0.28, cy + 0.24, name, JB_B(13), color=nmc, ha="left")
        text(aL + 0.28, cy - 0.24, desc, SG_M(10.5), color=T["MUTED"], ha="left")
        bx0, bw = aL + 2.62, 1.5
        text(bx0, cy + 0.32, f"Q = {q:.2f}", JB_M(10.5), color=nmc, ha="left")
        ax.add_patch(Rectangle((bx0, cy - 0.30), bw, 0.26, fc=T["TRACK"], ec="none", zorder=2))
        ax.add_patch(Rectangle((bx0, cy - 0.30), max(0.03, min(1.0, q / 0.5)) * bw, 0.26,
                               fc=barc, ec="none", zorder=3))
    text(aL + 0.28, 7.18, "3 workflow arms (the actions)", SG_SB(11), color=T["MUTED"], ha="left")

    # ---- Run + reward ----
    box(12.95, 4.9, 2.7, 1.35, T["GREEN_FILL"], T["GREEN"], lw=2.0)
    text(12.95, 5.28, "Run workflow", SG_SB(13))
    text(12.95, 4.88, "execute the SQL", JB_R(10), color=T["MUTED"])
    text(12.95, 4.50, "reward: 0 or 1", JB_M(11.5), color=T["GREEN"])

    # ---- arrows ----
    arrow(2.32, 4.9, 2.82, 4.9, T["ARROW"], lw=2.2)
    arrow(5.58, 4.9, 6.68, 4.9, T["GREEN"], lw=2.8)
    text(6.12, 5.18, "exploit (1-ε)", JB_M(9), color=T["GREEN"])
    arrow(5.5, 5.4, 6.68, 6.15, T["MUTED"], lw=1.5, ls="--", ms=14)
    text(5.78, 5.95, "explore ε", JB_M(8.5), color=T["MUTED"])
    arrow(11.1, 4.9, 11.6, 4.9, T["GREEN"], lw=2.8)
    text(11.35, 5.16, "run", JB_M(9), color=T["GREEN"])

    # ---- feedback loop ----
    arrow(12.95, 4.22, 4.2, 3.95, T["GREEN"], lw=2.4, rad=-0.30)
    text(8.55, 2.62, "update Q(selected arm)  —  online, and persists across runs",
         JB_M(11), color=T["GREEN"])

    # ---- caption ----
    box(8.0, 1.28, 13.8, 1.12, T["CAP_FILL"], T["CAP_EC"], lw=1.2, r=0.1)
    text(8.0, 1.52, "On BIRD-SQL it learns online to drop the weak arm (33% → 13% of pulls),", SG_M(12))
    text(8.0, 1.04,
         "beats a random baseline, and tracks just under the best fixed workflow — without being told the ranking.",
         SG_M(12))

    # ---- footers ----
    text(0.5, 0.32, "Stanford CS 153 · Frontier Systems", JB_R(9.5), color=T["MUTED"], ha="left")
    text(15.5, 0.32, "github.com/ms-d-kim/agentforge", JB_R(9.5), color=T["MUTED"], ha="right")

    fig.savefig(outpath, facecolor=T["BG"], dpi=160)
    plt.close(fig)
    print("wrote", outpath)


def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)
    render(LIGHT, os.path.join(OUTDIR, "hero_diagram.png"))
    render(DARK, os.path.join(OUTDIR, "hero_diagram_dark.png"))


if __name__ == "__main__":
    main()
