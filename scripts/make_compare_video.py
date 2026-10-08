#!/usr/bin/env -S uv run --script
"""Render docs/audio/compare.mp4: ksay and eSpeak NG read the same passages in turn.

GitHub plays inline only a video uploaded through its own editor (see
make_intro_video.py), so the comparison clips from make_audio.py are strung together
into one video: the passage on screen, a label for who is speaking, and a waveform.

    uv run scripts/make_audio.py
    uv run scripts/make_compare_video.py

Needs ffmpeg and macOS system fonts. There is no Apple say here: its licence does not
allow publishing recordings of its voices.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "soundfile"]
# ///

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from math import gcd
from pathlib import Path

from make_audio import AUDIO, PASSAGES, ROOT
from make_intro_video import MONO, SANS, drawtext

RATE = 24_000
SPEAKERS = [("ksay", "ksay", "0x3fb950"), ("espeak-ng", "eSpeak NG", "0xf0883e")]
LEAD_IN, BETWEEN_VOICES, BETWEEN_PASSAGES = 0.8, 0.6, 1.4


def main() -> int:
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    pieces: list[np.ndarray] = []
    shown: list[tuple[str, float, float]] = []  # passage text, start, end
    speaking: list[tuple[str, str, float, float]] = []  # label, color, start, end
    clock = 0.0

    def add(samples: np.ndarray) -> float:
        nonlocal clock
        pieces.append(samples)
        clock += len(samples) / RATE
        return clock

    def pause(seconds: float) -> None:
        add(np.zeros(int(seconds * RATE), dtype="float32"))

    pause(LEAD_IN)
    for name, text in PASSAGES.items():
        passage_start = clock
        for index, (folder, label, color) in enumerate(SPEAKERS):
            wave, rate = sf.read(
                AUDIO / "compare" / f"{folder}-{name}.mp3", dtype="float32"
            )
            divisor = gcd(RATE, rate)
            start = clock
            add(resample_poly(wave, RATE // divisor, rate // divisor).astype("float32"))
            speaking.append((label, color, start, clock))
            if index < len(SPEAKERS) - 1:
                pause(BETWEEN_VOICES)
        shown.append((text, passage_start - 0.2, clock + BETWEEN_PASSAGES / 2))
        pause(BETWEEN_PASSAGES)

    duration = clock
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        sf.write(folder / "track.wav", np.concatenate(pieces), RATE)
        grey = "0x8b949e"
        filters = [
            "[v0]"
            + drawtext(
                folder,
                "title",
                "ksay or eSpeak NG?",
                MONO,
                x=80,
                y=44,
                fontsize=56,
                fontcolor="white",
            ),
            drawtext(
                folder,
                "sub",
                "The same text, the same loudness",
                SANS,
                x=80,
                y=124,
                fontsize=28,
                fontcolor=grey,
            ),
        ]
        for i, (text, start, end) in enumerate(shown):
            filters.append(
                drawtext(
                    folder,
                    f"text{i}",
                    "\n".join(textwrap.wrap(text, 46)),
                    SANS,
                    x=80,
                    y=250,
                    fontsize=40,
                    fontcolor="white",
                    line_spacing=14,
                    enable=f"'between(t,{start:.2f},{end:.2f})'",
                )
            )
        for i, (label, color, start, end) in enumerate(speaking):
            filters.append(
                drawtext(
                    folder,
                    f"who{i}",
                    label,
                    MONO,
                    x=80,
                    y=186,
                    fontsize=36,
                    fontcolor=color,
                    enable=f"'between(t,{start - 0.05:.2f},{end + 0.15:.2f})'",
                )
            )
        graph = (
            f"color=c=0x0d1117:s=1280x720:r=30:d={duration + 0.3:.2f}[bg];"
            "[0:a]asplit[a][w];"
            "[w]showwaves=s=1120x140:mode=cline:rate=30:colors=0x58a6ff:scale=lin,"
            "format=rgba,colorkey=black:0.1:0.1[wv];"
            "[bg][wv]overlay=80:540:shortest=1:format=auto[v0];"
            + ",".join(filters)
            + "[v]"
        )
        (folder / "graph.txt").write_text(graph)
        target = AUDIO / "compare.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                folder / "track.wav",
            ]
            + [
                "-filter_complex_script",
                folder / "graph.txt",
                "-map",
                "[v]",
                "-map",
                "[a]",
            ]
            + [
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "26",
                "-pix_fmt",
                "yuv420p",
            ]
            + [
                "-c:a",
                "aac",
                "-b:a",
                "96k",
                "-movflags",
                "+faststart",
                "-shortest",
                target,
            ],
            check=True,
        )
    size = target.stat().st_size / 1024
    print(f"{target.relative_to(ROOT)}  {size:.0f} KB  {duration:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
