#!/usr/bin/env -S uv run --script
"""Show what moves a UTMOS22 score: damage clean recordings in known ways, score again.

    uv run benchmarks/compare.py --samples DIR
    uv run benchmarks/mos_probe.py DIR/ksay

Each condition takes the same recordings, changes them in one known way, and prints the
predicted mean opinion score next to that of the untouched recordings. It shows what the
model notices (bandwidth, noise, clipping) and what it does not (what is being said).
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "torchaudio", "numpy", "scipy", "soundfile"]
# ///

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from mos import load_predictor, predict


def conditions():
    """(label, function from a recording and its rate to the damaged recording)."""
    import numpy as np
    from scipy.signal import butter, resample_poly, sosfilt

    def noise(db):
        def apply(wave, rate):
            rng = np.random.default_rng(0)
            rms = np.sqrt(np.mean(wave**2))
            return wave + rng.standard_normal(len(wave)) * rms / 10 ** (db / 20)

        return apply

    def lowpass(wave, rate):
        return sosfilt(butter(8, 3400, "low", fs=rate, output="sos"), wave)

    def gain(db):
        return lambda wave, rate: wave * 10 ** (db / 20)

    def clipped(wave, rate):
        return np.clip(wave * 10 ** (24 / 20), -1, 1)

    def coarse(wave, rate):
        step = 2.0**-5  # six bits
        return np.round(wave / step) * step

    def speed(up, down):
        return lambda wave, rate: resample_poly(wave, up, down)

    def only_noise(wave, rate):
        return np.random.default_rng(0).standard_normal(len(wave)) * 0.1

    return [
        ("untouched", lambda wave, rate: wave),
        ("telephone bandwidth (low-pass at 3.4 kHz)", lowpass),
        ("white noise 20 dB below the speech", noise(20)),
        ("white noise 10 dB below the speech", noise(10)),
        ("30 dB quieter", gain(-30)),
        ("24 dB louder, then clipped", clipped),
        ("coarse six-bit quantisation", coarse),
        ("played 25% faster (the pitch rises too)", speed(4, 5)),
        ("played 20% slower (the pitch drops too)", speed(5, 4)),
        ("played backwards", lambda wave, rate: wave[::-1].copy()),
        ("white noise and no speech", only_noise),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path, help="a folder of clean recordings")
    parser.add_argument("--clips", type=int, default=10, help="how many to use")
    parser.add_argument("--json", type=Path, help="write the scores here")
    args = parser.parse_args()

    import soundfile as sf

    paths = sorted(args.directory.iterdir())[: args.clips]
    recordings = [sf.read(path, dtype="float32") for path in paths]
    predictor = load_predictor()
    results = {}
    for label, change in conditions():
        scores = [
            predict(predictor, change(wave, rate), rate) for wave, rate in recordings
        ]
        results[label] = {
            "mean": statistics.mean(scores),
            "stdev": statistics.stdev(scores),
        }
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    base = results["untouched"]["mean"]
    print(f"| Condition ({len(paths)} recordings) | predicted MOS | change |")
    print("| --- | ---: | ---: |")
    for label, row in results.items():
        change = "" if label == "untouched" else f"{row['mean'] - base:+.2f}"
        print(f"| {label} | {row['mean']:.2f} ± {row['stdev']:.2f} | {change} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
