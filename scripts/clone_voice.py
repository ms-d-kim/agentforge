"""Clone YOUR voice into ElevenLabs (Instant Voice Cloning) and wire it into the video.

Record ~1–3 minutes of clean speech (quiet room, one speaker, consistent tone), save as
mp3/wav/m4a, then:

    python -m scripts.clone_voice path/to/your_voice.mp3 --name "Daniel"

It uploads the sample(s), creates an Instant Voice Clone, prints the new voice_id, and writes
ELEVENLABS_VOICE_ID to .env so the next `python -m scripts.make_video` narrates in YOUR voice.

Note: Instant Voice Cloning requires a paid ElevenLabs tier (Starter+); it's not on the free plan.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")


def add_voice(name, audio_paths, key):
    import requests

    files = [("files", (p.name, open(p, "rb"), "application/octet-stream")) for p in audio_paths]
    resp = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        headers={"xi-api-key": key},
        data={"name": name, "remove_background_noise": "true"},
        files=files,
        timeout=300,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"ElevenLabs error {resp.status_code}: {resp.text[:300]}")
    return resp.json()["voice_id"]


def set_env_voice(voice_id):
    env = REPO / ".env"
    lines = [l for l in (env.read_text().splitlines() if env.exists() else [])
             if not l.startswith("ELEVENLABS_VOICE_ID=")]
    lines.append(f"ELEVENLABS_VOICE_ID={voice_id}")
    env.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Clone your voice into ElevenLabs (IVC).")
    ap.add_argument("audio", nargs="+", help="one or more clean voice samples (mp3/wav/m4a)")
    ap.add_argument("--name", default="MyVoice")
    args = ap.parse_args()

    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        raise SystemExit("ELEVENLABS_API_KEY not set in .env")
    paths = [Path(a) for a in args.audio]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"audio not found: {p}")

    print(f"uploading {len(paths)} sample(s) → cloning '{args.name}'…")
    voice_id = add_voice(args.name, paths, key)
    set_env_voice(voice_id)
    print(f"\ncloned voice_id = {voice_id}")
    print("wrote ELEVENLABS_VOICE_ID to .env")
    print("\nnow re-narrate the whole video in your voice:")
    print("  rm -f video/audio/*.mp3 && python -m scripts.make_video")


if __name__ == "__main__":
    main()
