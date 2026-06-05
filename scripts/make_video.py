"""Generate the demo video: ElevenLabs TTS voiceover over slides with the real figures.

Pipeline (no system ffmpeg needed — uses the imageio-ffmpeg static binary):
  1. ElevenLabs TTS per narration segment        -> video/audio/seg_NN.mp3
  2. matplotlib renders a 1080p slide per segment -> video/slides/seg_NN.png
  3. ffmpeg builds a clip per segment (slide + audio), then concatenates
                                                  -> video/agentforge_demo.mp4

    python -m scripts.make_video                 # full build
    python -m scripts.make_video --slides-only    # render slides only (no API calls)

Needs ELEVENLABS_API_KEY in .env (gitignored).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from scripts._style import HEAD, MONO  # noqa: E402 — registers bundled fonts on import

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO = REPO / "video"
AUDIO_DIR = VIDEO / "audio"
SLIDE_DIR = VIDEO / "slides"
RESULTS = REPO / "results"

VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "XrExE9yKIg1WjnnlVkGX")  # Matilda — professional
TTS_MODEL = "eleven_multilingual_v2"

# --- theme (black, terminal / product aesthetic) --------------------------- #
BG = "#0A0A0B"       # near-black
FG = "#FFFFFF"       # headings
MUTED = "#8A8A93"    # body / captions
GREEN = "#2BE06B"    # terminal-green accent
MAGENTA = "#FF3DA6"  # pink accent
PANEL = "#141417"    # code panel
BORDER = "#2A2A30"   # hairline borders
W, H = 19.2, 10.8    # 1920x1080 at dpi=100


# --------------------------------------------------------------------------- #
# Narration + slides
# --------------------------------------------------------------------------- #
SNIPPET = (
    "from agentforge import WorkflowSelector\n\n"
    "sel = WorkflowSelector(\n"
    '    arms={"fast": f, "careful": g, "use_tools": h},\n'
    "    reward=my_signal,        # tests / eval / thumbs-up\n"
    '    policy="linucb",         # bandit -> contextual -> RL\n'
    '    persist="state.json",    # learns across runs\n'
    ")\n"
    "out, arm = sel.run(task)     # picks, runs, scores, improves"
)

COMPARE_LEFT = [
    "Every agent SDK",
    "(ADK · OpenAI · MS · LangGraph · CrewAI)",
    "",
    "• static orchestration",
    "• loop stays OPEN —",
    "  a human edits prompts between runs",
    '• OpenAI docs: "do not modify',
    '  the config during the run"',
]
COMPARE_RIGHT = [
    "AgentForge",
    "",
    "",
    "• online learned selection",
    "• closes the loop automatically",
    "• per-task, persists across runs",
    "• closest prior work: ARC —",
    "  but offline + research-only",
]

SEGMENTS = [
    dict(id="01", slide=dict(kind="title", title="AgentForge",
         subtitle="A self-improving agent SDK", foot="Stanford CS 153 · Frontier Systems"),
         text="AgentForge. A self-improving agent SDK. Most agents hardcode how they act. "
              "AgentForge learns which strategy to use, online, from your own reward."),
    dict(id="02", slide=dict(kind="bullets", title="The bottleneck", bullets=[
            "LLM agents are stochastic; different tasks want different strategies.",
            "Direct prompt? Explore the schema first? Decompose the problem?",
            "Today you hand-pick one workflow and freeze it.",
            "When it fails, a human reads logs and edits the prompt.",
            "Nobody is learning which workflow actually works.",
         ]),
         text="Here's the bottleneck. Language-model agents are stochastic, and different tasks "
              "want different strategies. Sometimes a direct prompt is enough; sometimes you need "
              "to explore first, or break the problem down. Today you hand-pick one workflow and "
              "freeze it. And when it fails, a human reads the logs and edits the prompt. Nobody "
              "is actually learning which workflow works. That's the gap AgentForge fills."),
    dict(id="03", slide=dict(kind="code", title="The idea — a learned selection layer", code=SNIPPET),
         text="The idea is simple. Give AgentForge a few strategies, any callables, and a reward "
              "signal you already have: tests passing, an eval score, a thumbs-up. An online "
              "multi-armed bandit learns which strategy wins on your tasks, concentrates on it, and "
              "keeps improving across runs. It's a harness for choosing your harness's strategy."),
    dict(id="3b", slide=dict(kind="figure", title="Architecture — one self-improving loop",
         image=str(RESULTS / "dark" / "architecture.png"),
         caption="select → run → reward → update (online) → persist · every domain plugs in arms + reward"),
         text="Here's the architecture. A task comes in, the selector's policy picks a strategy, "
              "that strategy runs and produces an output, and a reward function scores it. The "
              "reward updates the policy, online, and persists to disk. That's the whole "
              "self-improving loop. Every domain just plugs in its own strategies and its own reward."),
    dict(id="04", slide=dict(kind="figure", title="How it works — proven on text-to-SQL",
         image=str(RESULTS / "dark" / "exp_01_cumulative_reward.png"),
         caption="BIRD-SQL · gpt-4o-mini · the bandit beats random and approaches the oracle"),
         text="Here's the loop. The bandit picks a workflow, runs it, and gets a reward by "
              "executing the result. For our first domain, text-to-SQL, we run the generated query "
              "against the database and check the answer. The reward updates the policy. On the "
              "BIRD benchmark, with a deliberately mid-tier model so strategy still matters, the "
              "bandit learns online to drop the weak workflow and concentrate on the strong ones, "
              "beating random and approaching the oracle."),
    dict(id="05", slide=dict(kind="figure", title="It generalizes — four domains, one selector",
         image=str(RESULTS / "dark" / "sweep_matrix.png"),
         caption="SQL · code (tests pass) · search (file match) · compaction (downstream QA)"),
         text="But the selection layer isn't about SQL. It's domain-agnostic. We plugged in three "
              "more domains: code generation, rewarded by unit tests; agentic search, rewarded by "
              "finding the right file; and context compaction, rewarded by whether the answer "
              "survives. Same selector, different arms and reward. In every domain, it learns the "
              "best strategy."),
    dict(id="06", slide=dict(kind="figure", title="A ladder: bandit → contextual → multi-step RL",
         image=str(RESULTS / "dark" / "rl_multihop.png"),
         caption="Contextual: 1.00 vs 0.50 per-task · REINFORCE: 0.81 vs 0.35 best fixed-horizon"),
         text="And selection is a ladder. The basic bandits learn the best arm on average. A "
              "contextual bandit learns the best arm per task: on tasks where the winner depends on "
              "the input, it hits perfect accuracy where the average-case bandit is stuck at fifty "
              "percent. And at the top, we go multi-step: a REINFORCE policy learns the "
              "iterate-or-stop decision, when to keep going versus commit. It more than doubles any "
              "fixed strategy, because only a learned policy adapts the number of steps per task."),
    dict(id="07", slide=dict(kind="compare", title="How it's different",
         left=COMPARE_LEFT, right=COMPARE_RIGHT),
         text="So how is this different from every agent SDK out there? We checked. Google's ADK, "
              "OpenAI's Agents SDK, Microsoft, LangGraph, CrewAI: they're all static orchestration. "
              "OpenAI's own docs literally say, do not modify the config during the run. Improvement "
              "means a human edits prompts between runs. They optimize inside a fixed strategy, or "
              "just measure it. AgentForge optimizes over a portfolio of strategies and routes to "
              "winners online. That axis is unoccupied."),
    dict(id="08", slide=dict(kind="bullets", title="Verified, and ready to use", bullets=[
            "64 offline tests · real BIRD results · cross-domain sweep",
            "Multi-agent adversarial verification re-runs and refutes each claim",
            "pip install -e .  →  from agentforge import WorkflowSelector",
            "Closest research (ARC) is offline; we are online, persistent, packaged",
         ]),
         text="And we don't just claim it. Sixty-four offline tests, real benchmark results, and a "
              "multi-agent adversarial verification pass that re-runs every result and tries to "
              "refute it. To use it: pip install, wrap your strategies, plug in a reward, ten lines. "
              "The closest research work, ARC, does this offline. We do it online, persistent, and "
              "packaged."),
    dict(id="09", slide=dict(kind="bullets", title="What's next", bullets=[
            "Cost-aware reward — accuracy per dollar",
            "Richer contextual policies that pick per task",
            "Workflows themselves become learnable, improved from demonstrations",
            "…with the bandit selecting over an evolving library",
         ]),
         text="What's next? Cost-aware reward, accuracy per dollar. Richer contextual policies that "
              "pick per task. And the long game: making the workflows themselves learnable, improved "
              "from demonstrations, with the bandit selecting over an evolving library. The selection "
              "layer is the foundation."),
    dict(id="10", slide=dict(kind="title", title="Don't hand-pick your agent's workflow.",
         subtitle="Let it learn.", foot="github.com/ms-d-kim/agentforge", title_fs=54),
         text="AgentForge. Don't hand-pick your agent's workflow. Let it learn. The code is open "
              "source. Thank you."),
]


# --------------------------------------------------------------------------- #
# Slide rendering
# --------------------------------------------------------------------------- #
def _new_fig():
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor(BG)
    return fig


_KW = {"from", "import", "def", "return", "lambda", "class", "for", "in",
       "if", "else", "with", "as", "None", "True", "False"}


def _tokenize_code(line):
    """[(text, color)] spans for light syntax highlighting (monospace)."""
    spans, code, comment = [], line, ""
    h = line.find("#")
    if h != -1:
        code, comment = line[:h], line[h:]
    for m in re.finditer(r'"[^"]*"|\'[^\']*\'|[A-Za-z_][A-Za-z0-9_]*|\s+|[^\sA-Za-z0-9_]', code):
        t = m.group(0)
        if t[:1] in ('"', "'"):
            spans.append((t, GREEN))
        elif t in _KW:
            spans.append((t, MAGENTA))
        elif t[:1].isalpha() or t[:1] == "_":
            spans.append((t, "#E8E8EE"))
        elif t.isspace():
            spans.append((t, FG))
        else:
            spans.append((t, "#B7B7C0"))
    if comment:
        spans.append((comment, "#6E6E78"))
    return spans or [(" ", FG)]


def _footer(fig):
    fig.add_artist(Circle((0.073, 0.052), 0.006, transform=fig.transFigure, color=GREEN, zorder=5))
    fig.text(0.086, 0.05, "AgentForge — a self-improving agent SDK", color=MUTED,
             fontsize=14, va="center", fontfamily=MONO)
    fig.text(0.927, 0.05, "CS 153 · Frontier Systems", color=MUTED, fontsize=14,
             ha="right", va="center", fontfamily=MONO)


def _render_code(fig, spec):
    fig.text(0.5, 0.90, spec["title"], color=FG, fontsize=42, ha="center", va="center",
             fontweight="bold", fontfamily=HEAD)
    ax = fig.add_axes([0.135, 0.135, 0.73, 0.65])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.025",
                 transform=ax.transAxes, facecolor=PANEL, edgecolor=BORDER, lw=1.5, zorder=0))
    barh = 0.135
    ax.add_patch(Rectangle((0, 1 - barh), 1, barh, transform=ax.transAxes, facecolor="#1A1A1F",
                 edgecolor="none", zorder=1))
    for i, c in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        ax.add_patch(Circle((0.028 + i * 0.024, 1 - barh / 2), 0.0075, transform=ax.transAxes, color=c, zorder=2))
    ax.text(0.5, 1 - barh / 2, "selector.py", color=MUTED, fontsize=14, fontfamily=MONO,
            ha="center", va="center", transform=ax.transAxes, zorder=2)
    rows = [HPacker(children=[TextArea(t, textprops=dict(color=c, fontfamily=MONO, fontsize=20))
                              for t, c in _tokenize_code(ln)], align="baseline", pad=0, sep=0)
            for ln in spec["code"].split("\n")]
    box = VPacker(children=rows, align="left", pad=0, sep=8)
    ax.add_artist(AnchoredOffsetbox(loc="upper left", child=box, pad=0, borderpad=0,
                  bbox_to_anchor=(0.05, 1 - barh - 0.06), bbox_transform=ax.transAxes, frameon=False))


def render_slide(spec, path):
    fig = _new_fig()
    kind = spec["kind"]

    if kind == "title":
        fig.text(0.5, 0.595, spec["title"], color=FG, fontsize=spec.get("title_fs", 90),
                 ha="center", va="center", fontweight="bold", fontfamily=HEAD)
        fig.text(0.5, 0.47, spec.get("subtitle", ""), color=GREEN, fontsize=40,
                 ha="center", va="center", fontfamily=HEAD)
        fig.add_artist(Line2D([0.45, 0.55], [0.40, 0.40], color=GREEN, lw=2.5, transform=fig.transFigure))
        if spec.get("foot"):
            fig.text(0.5, 0.33, spec["foot"], color=MUTED, fontsize=23, ha="center",
                     va="center", fontfamily=MONO)

    elif kind == "bullets":
        fig.add_artist(Line2D([0.072, 0.072], [0.27, 0.80], color=GREEN, lw=4, transform=fig.transFigure))
        fig.text(0.105, 0.845, spec["title"], color=FG, fontsize=48, fontweight="bold",
                 va="center", fontfamily=HEAD)
        y = 0.665
        for b in spec["bullets"]:
            if b:
                fig.text(0.105, y, "▸", color=GREEN, fontsize=25, va="center")
                fig.text(0.135, y, b, color="#D7D7DD", fontsize=26, va="center", fontfamily=MONO)
            y -= 0.108

    elif kind == "code":
        _render_code(fig, spec)

    elif kind == "figure":
        fig.text(0.5, 0.925, spec["title"], color=FG, fontsize=40, fontweight="bold",
                 ha="center", va="center", fontfamily=HEAD)
        # Dark figures share the slide's black bg → place directly, with a hairline frame.
        fig.add_artist(FancyBboxPatch((0.085, 0.15), 0.83, 0.66, boxstyle="round,pad=0,rounding_size=0.008",
                       transform=fig.transFigure, facecolor="none", edgecolor=BORDER, lw=1.2, zorder=1))
        ax = fig.add_axes([0.10, 0.165, 0.80, 0.63], zorder=2)
        ax.axis("off")
        ax.imshow(plt.imread(spec["image"]))
        if spec.get("caption"):
            fig.text(0.5, 0.105, spec["caption"], color=MUTED, fontsize=22, ha="center",
                     va="center", fontfamily=MONO)

    elif kind == "compare":
        fig.text(0.5, 0.90, spec["title"], color=FG, fontsize=48, fontweight="bold",
                 ha="center", va="center", fontfamily=HEAD)
        fig.add_artist(Line2D([0.5, 0.5], [0.13, 0.80], color=BORDER, lw=2, transform=fig.transFigure))
        for x0, lines, headcol, bodycol in [(0.08, spec["left"], MUTED, "#A7A7B0"),
                                            (0.55, spec["right"], GREEN, "#E8E8EE")]:
            y = 0.75
            for i, ln in enumerate(lines):
                if i == 0:
                    fig.text(x0, y, ln, color=headcol, fontsize=32, fontweight="bold",
                             va="center", fontfamily=HEAD)
                elif ln:
                    fig.text(x0, y, ln, color=bodycol, fontsize=23, va="center", fontfamily=MONO)
                y -= 0.083

    _footer(fig)
    fig.savefig(path, facecolor=BG)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# TTS + assembly
# --------------------------------------------------------------------------- #
def tts(text, out_path):
    import requests

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set in .env")
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}?output_format=mp3_44100_128"
    resp = requests.post(
        url,
        headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        json={"text": text, "model_id": TTS_MODEL,
              "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                                 "style": 0.0, "use_speaker_boost": True}},
        timeout=120,
    )
    resp.raise_for_status()
    out_path.write_bytes(resp.content)


def _with_breaks(text, dur=0.35):
    """Insert short ElevenLabs pauses between sentences for a calmer, less rushed read."""
    parts = re.split(r"(?<=[.!?]) +", text.strip())
    return (f' <break time="{dur}s" /> ').join(parts)


def make_clip(slide_png, audio_mp3, out_mp4, tail_silence=0.7):
    # apad adds trailing silence so segments don't butt up against each other (breathing room).
    subprocess.run(
        [FFMPEG, "-y", "-loop", "1", "-i", str(slide_png), "-i", str(audio_mp3),
         "-c:v", "libx264", "-tune", "stillimage", "-r", "25",
         "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
         "-vf", "scale=1920:1080", "-af", f"apad=pad_dur={tail_silence}",
         "-shortest", str(out_mp4)],
        check=True, capture_output=True,
    )


def concat(clips, out_mp4):
    listfile = VIDEO / "concat.txt"
    listfile.write_text("".join(f"file '{c.resolve()}'\n" for c in clips))
    subprocess.run([FFMPEG, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile),
                    "-c", "copy", str(out_mp4)], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description="Build the AgentForge demo video.")
    ap.add_argument("--slides-only", action="store_true", help="render slides; skip TTS + ffmpeg")
    ap.add_argument("--reuse-audio", action="store_true",
                    help="skip TTS for segments whose mp3 already exists (only new/changed segments)")
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)

    try:  # light architecture for README + dark figures for the video slides
        from scripts.make_architecture import render as render_arch

        render_arch(RESULTS / "architecture.png")
        if not (RESULTS / "dark" / "sweep_matrix.png").exists():
            from scripts.make_dark_figures import main as make_dark

            make_dark()
    except Exception as exc:  # noqa: BLE001
        print(f"  (figure prep skipped: {exc})")

    print("rendering slides…")
    for seg in SEGMENTS:
        render_slide(seg["slide"], SLIDE_DIR / f"seg_{seg['id']}.png")
    if args.slides_only:
        print(f"slides in {SLIDE_DIR}")
        return

    print("generating voiceover (ElevenLabs)…")
    clips = []
    for seg in SEGMENTS:
        mp3 = AUDIO_DIR / f"seg_{seg['id']}.mp3"
        if not (args.reuse_audio and mp3.exists()):
            tts(_with_breaks(seg["text"]), mp3)
        clip = VIDEO / f"clip_{seg['id']}.mp4"
        make_clip(SLIDE_DIR / f"seg_{seg['id']}.png", mp3, clip)
        clips.append(clip)
        print(f"  seg {seg['id']} ✓")

    out = VIDEO / "agentforge_demo.mp4"
    concat(clips, out)
    size_mb = out.stat().st_size / 1e6
    print(f"\nDONE → {out}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
