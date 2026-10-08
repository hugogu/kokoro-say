#!/usr/bin/env -S uv run --script
"""Rebuild the audio clips the README plays.

    uv run scripts/make_audio.py

docs/audio/intro.mp3      ksay introducing itself, reading docs/audio/intro.txt
docs/audio/compare/*.mp3  three short passages read by ksay and by eSpeak NG

Every clip is loudness-matched to -16 LUFS with ffmpeg (two passes), so that no
engine sounds better just by being louder. Needs ffmpeg. There is nothing from
Apple's say here: its licence does not allow publishing recordings of its voices.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["kokoro-say", "numpy", "soundfile"]
#
# [tool.uv.sources]
# kokoro-say = { path = "..", editable = true }
# ///

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "docs" / "audio"
PARAGRAPH = (ROOT / "benchmarks" / "paragraph.txt").read_text().strip()
PASSAGES = {
    "sea": "The salt breeze came across from the sea.",
    "invoice": "The invoice total is $1,250.50, due on March 3, 2026.",
    "explainer": ". ".join(PARAGRAPH.split(". ")[:2]) + ".",
}
TARGET = "I=-16:TP=-1.5:LRA=11"


def loudness_matched_mp3(source: Path, target: Path) -> None:
    """Encode source as a speech-quality MP3 at -16 LUFS: measure, then correct."""
    first = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(source),
            "-af",
            f"loudnorm={TARGET}:print_format=json",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    measured = json.loads(
        re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", first.stderr).group(0)
    )
    correction = (
        f"loudnorm={TARGET}:measured_I={measured['input_i']}:measured_TP={measured['input_tp']}"
        f":measured_LRA={measured['input_lra']}:measured_thresh={measured['input_thresh']}"
        f":offset={measured['target_offset']}:linear=true"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-af",
            correction,
        ]
        + ["-ac", "1", "-c:a", "libmp3lame", "-b:a", "64k", str(target)],
        check=True,
    )


def main() -> int:
    import soundfile as sf

    from kokoro_say import cli

    sys.path.insert(0, str(ROOT / "benchmarks"))
    from compare import espeak

    kokoro = cli.load_kokoro(cli.ensure_model(cli.model_dir()))
    (AUDIO / "compare").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        raw = Path(folder) / "raw.wav"

        text = (AUDIO / "intro.txt").read_text()
        samples, rate = kokoro.create(text, voice=cli.DEFAULT_VOICE, lang="en-us")
        sf.write(raw, samples, rate)
        loudness_matched_mp3(raw, AUDIO / "intro.mp3")

        for name, text in PASSAGES.items():
            samples, rate = kokoro.create(text, voice=cli.DEFAULT_VOICE, lang="en-us")
            sf.write(raw, samples, rate)
            loudness_matched_mp3(raw, AUDIO / "compare" / f"ksay-{name}.mp3")
            samples, rate = espeak(text)
            sf.write(raw, samples, rate)
            loudness_matched_mp3(raw, AUDIO / "compare" / f"espeak-ng-{name}.mp3")
    for clip in sorted(AUDIO.rglob("*.mp3")):
        print(f"{clip.relative_to(ROOT)}  {clip.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
