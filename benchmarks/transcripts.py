#!/usr/bin/env -S uv run --script
"""Print what an English recognizer hears in each recording of a folder.

    uv run benchmarks/transcripts.py DIR

Whisper small.en transcribes every file in DIR with sampling off, so the same file
always gives the same text. It is the way to check a claim such as "this voice spells
JSON out": the transcript says "j, s, o, n" or "JSON". It is a poor judge of anything
else. On speech that is mostly Chinese it makes text up (often "Thank you for
watching"), and it hears the English words of a Chinese voice less well than a
listener does.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "soundfile", "faster-whisper"]
# ///

from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel
from scipy.signal import resample_poly

WHISPER_MODEL = "small.en"


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    for path in sorted(Path(sys.argv[1]).iterdir()):
        wave, rate = sf.read(path, dtype="float32")
        if wave.ndim > 1:
            wave = wave.mean(axis=1)
        divisor = gcd(16_000, rate)
        audio = resample_poly(wave, 16_000 // divisor, rate // divisor)
        segments, _ = model.transcribe(
            audio.astype("float32"),
            language="en",
            beam_size=5,
            temperature=0.0,
            condition_on_previous_text=False,
        )
        print(f"{path.name}: {' '.join(s.text.strip() for s in segments)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
