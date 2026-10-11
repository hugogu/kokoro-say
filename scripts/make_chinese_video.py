#!/usr/bin/env -S uv run --script
"""Render docs/audio/chinese.mp4: ksay reads Chinese with English words in it, twice.

Each sentence is read first with the zh extra and then without it, the same voice at the
same loudness, so that what the extra changes can be heard. GitHub plays inline only a
video uploaded through its own editor (see make_intro_video.py).

    uv run --python 3.12 scripts/make_chinese_audio.py
    uv run scripts/make_chinese_video.py

Needs ffmpeg with the drawtext filter and macOS system fonts. There is no Apple say
here: its licence does not allow publishing recordings of its voices.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "soundfile"]
# ///

from __future__ import annotations

import sys
from pathlib import Path

from make_chinese_audio import AUDIO, CHINESE
from make_compare_video import render

CJK = Path("/System/Library/Fonts/Hiragino Sans GB.ttc")
SPEAKERS = [
    ("ksay", "ksay with the zh extra", "0x3fb950"),
    ("ksay-plain", "ksay without it", "0xf0883e"),
]


def main() -> int:
    if not CJK.exists():
        raise SystemExit(f"{CJK} not found; this script needs macOS system fonts")
    render(
        AUDIO / "chinese",
        CHINESE,
        SPEAKERS,
        "Chinese + English",
        "ksay, voice zf_xiaobei, the same loudness",
        CJK,
        26,
        AUDIO / "chinese.mp4",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
