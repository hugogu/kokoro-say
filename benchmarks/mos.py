#!/usr/bin/env -S uv run --script
"""Predict how natural each engine's recordings sound, with UTMOS22.

    uv run benchmarks/compare.py --wer --samples DIR
    uv run benchmarks/mos.py DIR

DIR holds one folder of recordings per engine, which is what --samples writes.
UTMOS22 (https://github.com/tarepan/SpeechMOS, pinned to v1.2.0) is a neural model
trained on listening-test scores: it predicts the mean opinion score listeners
would give, from 1 (bad) to 5 (excellent). It tracks listeners well enough to rank
engines, but it is a model and not a listener, so treat small gaps as noise and
listen for yourself. The first run downloads about 400 MB of weights from GitHub
and runs the model's code, which is why it asks torch.hub to trust that repository.
"""
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch", "torchaudio", "numpy", "scipy", "soundfile"]
# ///

from __future__ import annotations

import argparse
import json
import statistics
import sys
from math import gcd
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("samples", type=Path, help="a folder of folders of recordings")
    parser.add_argument("--json", type=Path, help="write the scores here")
    args = parser.parse_args()

    import soundfile as sf
    import torch
    from scipy.signal import resample_poly

    predictor = torch.hub.load(
        "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True, verbose=False
    )
    scores = {}
    for folder in sorted(path for path in args.samples.iterdir() if path.is_dir()):
        by_file = {}
        for path in sorted(folder.iterdir()):
            wave, rate = sf.read(path, dtype="float32")
            if wave.ndim > 1:
                wave = wave.mean(axis=1)
            divisor = gcd(16_000, rate)
            wave = resample_poly(wave, 16_000 // divisor, rate // divisor)
            with torch.no_grad():
                score = predictor(
                    torch.from_numpy(wave.astype("float32")).unsqueeze(0), 16_000
                )
            by_file[path.name] = float(score)
        values = list(by_file.values())
        scores[folder.name] = {
            "mean": statistics.mean(values),
            "stdev": statistics.stdev(values),
            "recordings": len(values),
            "by_file": by_file,
        }

    if args.json:
        args.json.write_text(json.dumps(scores, indent=2) + "\n")
    print("| Engine | predicted MOS (mean ± standard deviation) | recordings |")
    print("| --- | ---: | ---: |")
    for name, row in scores.items():
        print(
            f"| {name} | {row['mean']:.2f} ± {row['stdev']:.2f} | {row['recordings']} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
