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
import subprocess
from pathlib import Path

import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
VIDEO = REPO / "video"
AUDIO_DIR = VIDEO / "audio"
SLIDE_DIR = VIDEO / "slides"
RESULTS = REPO / "results"

VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "XrExE9yKIg1WjnnlVkGX")  # Matilda — professional
TTS_MODEL = "eleven_multilingual_v2"

# --- theme ----------------------------------------------------------------- #
BG = "#0B1F3A"      # navy
FG = "#F2F6FC"      # near-white
ACCENT = "#4DA3FF"  # blue
MUTED = "#9DB2CE"   # gray-blue
W, H = 19.2, 10.8   # 1920x1080 at dpi=100


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
    dict(id="04", slide=dict(kind="figure", title="How it works — proven on text-to-SQL",
         image=str(RESULTS / "exp_01_cumulative_reward.png"),
         caption="BIRD-SQL · gpt-4o-mini · the bandit beats random and approaches the oracle"),
         text="Here's the loop. The bandit picks a workflow, runs it, and gets a reward by "
              "executing the result. For our first domain, text-to-SQL, we run the generated query "
              "against the database and check the answer. The reward updates the policy. On the "
              "BIRD benchmark, with a deliberately mid-tier model so strategy still matters, the "
              "bandit learns online to drop the weak workflow and concentrate on the strong ones, "
              "beating random and approaching the oracle."),
    dict(id="05", slide=dict(kind="figure", title="It generalizes — four domains, one selector",
         image=str(RESULTS / "sweep_matrix.png"),
         caption="SQL · code (tests pass) · search (file match) · compaction (downstream QA)"),
         text="But the selection layer isn't about SQL. It's domain-agnostic. We plugged in three "
              "more domains: code generation, rewarded by unit tests; agentic search, rewarded by "
              "finding the right file; and context compaction, rewarded by whether the answer "
              "survives. Same selector, different arms and reward. In every domain, it learns the "
              "best strategy."),
    dict(id="06", slide=dict(kind="figure", title="A ladder: bandit → contextual → multi-step RL",
         image=str(RESULTS / "rl_multihop.png"),
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
         subtitle="Let it learn.", foot="github.com/ms-d-kim/agentforge"),
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


def _footer(fig):
    fig.text(0.04, 0.045, "AgentForge — a self-improving agent SDK", color=MUTED, fontsize=15)
    fig.text(0.96, 0.045, "CS 153 · Frontier Systems", color=MUTED, fontsize=15, ha="right")


def render_slide(spec, path):
    fig = _new_fig()
    kind = spec["kind"]

    if kind == "title":
        fig.text(0.5, 0.60, spec["title"], color=FG, fontsize=70, ha="center", va="center", fontweight="bold")
        fig.text(0.5, 0.46, spec.get("subtitle", ""), color=ACCENT, fontsize=40, ha="center", va="center")
        if spec.get("foot"):
            fig.text(0.5, 0.30, spec["foot"], color=MUTED, fontsize=24, ha="center", va="center")

    elif kind == "bullets":
        fig.text(0.07, 0.86, spec["title"], color=ACCENT, fontsize=46, fontweight="bold", va="center")
        y = 0.70
        for b in spec["bullets"]:
            fig.text(0.09, y, ("•  " + b) if b else "", color=FG, fontsize=30, va="center")
            y -= 0.115

    elif kind == "code":
        fig.text(0.07, 0.88, spec["title"], color=ACCENT, fontsize=44, fontweight="bold", va="center")
        ax = fig.add_axes([0.09, 0.16, 0.82, 0.60])
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#0A1830",
                                   edgecolor=ACCENT, lw=1.5))
        ax.text(0.04, 0.92, spec["code"], color="#E6EDF7", fontsize=23, family="monospace",
                va="top", ha="left", transform=ax.transAxes)

    elif kind == "figure":
        fig.text(0.5, 0.92, spec["title"], color=ACCENT, fontsize=42, fontweight="bold", ha="center", va="center")
        img = plt.imread(spec["image"])
        ax = fig.add_axes([0.10, 0.16, 0.80, 0.68])
        ax.imshow(img)
        ax.axis("off")
        if spec.get("caption"):
            fig.text(0.5, 0.10, spec["caption"], color=MUTED, fontsize=24, ha="center", va="center")

    elif kind == "compare":
        fig.text(0.5, 0.90, spec["title"], color=ACCENT, fontsize=46, fontweight="bold", ha="center", va="center")
        for x0, lines, col in [(0.07, spec["left"], MUTED), (0.55, spec["right"], FG)]:
            y = 0.76
            for i, ln in enumerate(lines):
                fs = 30 if i == 0 else 24
                fw = "bold" if i == 0 else "normal"
                c = ACCENT if i == 0 else col
                fig.text(x0, y, ln, color=c, fontsize=fs, fontweight=fw, va="center")
                y -= 0.085
        fig.add_artist(plt.Line2D([0.51, 0.51], [0.12, 0.80], color="#28406A", lw=2))

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


def make_clip(slide_png, audio_mp3, out_mp4):
    subprocess.run(
        [FFMPEG, "-y", "-loop", "1", "-i", str(slide_png), "-i", str(audio_mp3),
         "-c:v", "libx264", "-tune", "stillimage", "-r", "25",
         "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p",
         "-vf", "scale=1920:1080", "-shortest", str(out_mp4)],
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
    args = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    SLIDE_DIR.mkdir(parents=True, exist_ok=True)

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
        tts(seg["text"], mp3)
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
