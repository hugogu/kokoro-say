#!/usr/bin/env -S uv run --script
"""Download a few real human recordings, to give predicted-naturalness scores a scale.

    uv run benchmarks/fetch_human_reference.py DIR
    uv run benchmarks/mos.py DIR

DIR gets two folders of twelve clips each, taken from the start of two public speech
datasets through the Hugging Face dataset server: lj, a single studio speaker
reading non-fiction (LJ Speech, public domain), and libritts_r, many speakers reading
audiobooks, cleaned up by speech restoration (LibriTTS-R, test.clean, CC BY 4.0). They
are for scoring on your own disk only; do not republish them.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

SOURCES = {  # folder -> (dataset, config, split)
    "lj": ("MikhailT/lj-speech", "default", "full"),
    "libritts_r": ("mythicinfinity/libritts_r", "clean", "test.clean"),
}
HEADERS = {"User-Agent": "kokoro-say benchmarks"}


def get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS)) as reply:
        return reply.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("directory", type=Path)
    parser.add_argument("--clips", type=int, default=12, help="clips per dataset")
    args = parser.parse_args()
    for name, (dataset, config, split) in SOURCES.items():
        query = urllib.parse.urlencode(
            {"dataset": dataset, "config": config, "split": split, "offset": 0}
            | {"length": args.clips}
        )
        rows = json.loads(get(f"https://datasets-server.huggingface.co/rows?{query}"))
        target = args.directory / name
        target.mkdir(parents=True, exist_ok=True)
        for index, item in enumerate(rows["rows"]):
            audio = item["row"]["audio"][0]
            (target / f"{name}-{index:02d}.wav").write_bytes(get(audio["src"]))
        print(f"{target}: {len(rows['rows'])} clips from {dataset}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
