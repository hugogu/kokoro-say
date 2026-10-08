#!/usr/bin/env -S uv run --script
"""How soon each engine speaks text that is still being written.

    uv run benchmarks/live.py                   # a sentence every 2 s, then all at once
    uv run benchmarks/live.py --interval 1      # only: a sentence a second
    uv run benchmarks/live.py --json live.json

A producer prints the first sentences of paragraph.txt to a pipe, one every INTERVAL
seconds, as a language model or a build log would, and an engine reads that pipe. The
clock starts when the producer does, so every engine pays for starting up.

`say` waits for the end of its input before it speaks (its manual page says the text is
"spoken all at once"), and ksay --stream speaks each sentence as it arrives.

- first sound: when the first audio exists. For ksay that is its first write to the
  audio device, which is replaced by a recorder, so the real pipeline runs and only the
  loudspeaker is missing. For say it is when its output file first holds audio.
- speech over: when the last of it would have been heard, if each block is played as
  soon as it is handed over, or as soon as the one before it has finished.

Apple's licence does not allow publishing recordings of its voices; say only writes to a
temporary file here, which is deleted.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["kokoro-say", "numpy", "soundfile"]
#
# [tool.uv.sources]
# kokoro-say = { path = "..", editable = true }
# ///

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from compare import environment

HERE = Path(__file__).resolve().parent
PARAGRAPH = (HERE / "paragraph.txt").read_text().strip()
RATE = 24_000
SAY_HEADER = 8192  # bytes; an AIFF file holds only its header until audio is written

PRODUCER = """
import json, sys, time
sentences, interval = json.loads(sys.argv[1]), float(sys.argv[2])
stamps = []
for index, sentence in enumerate(sentences):
    if index and interval:
        time.sleep(interval)
    stamps.append(time.time())
    print(sentence, flush=True)
print(json.dumps(stamps), file=sys.stderr)
"""

# ksay with the audio device replaced by a recorder of when each block was handed over
RECORDER = """
import json, os, sys, time
from kokoro_say import cli

writes = []


class Output:
    def start(self): pass
    def write(self, samples): writes.append((time.time(), len(samples)))
    def stop(self): pass
    def close(self): pass


cli.open_output = lambda rate: Output()
status = cli.main(sys.argv[1:])
with open(os.environ["LIVE_WRITES"], "w") as file:
    json.dump(writes, file)
sys.exit(status)
"""


@dataclass
class Engine:
    name: str
    kind: str  # "ksay" or "say"
    arguments: list[str]


def engines() -> list[Engine]:
    found = [
        Engine("ksay --stream", "ksay", ["--stream"]),
        Engine("ksay, without --stream", "ksay", []),
    ]
    if sys.platform == "darwin" and shutil.which("say"):
        found += [
            Engine("say (default voice)", "say", []),
            Engine("say (Samantha)", "say", ["-v", "Samantha"]),
        ]
    return found


def sentences_of(text: str, count: int) -> list[str]:
    return re.split(r"(?<=[.!?])\s+", " ".join(text.split()))[:count]


def playback(writes: list[list[float]]) -> tuple[float, float]:
    """When a device playing each block as soon as it can would start and finish."""
    first = end = None
    for stamp, frames in writes:
        begin = stamp if end is None else max(stamp, end)
        first = begin if first is None else first
        end = begin + frames / RATE
    return first, end


def run_ksay(engine: Engine, producer: subprocess.Popen, folder: Path) -> tuple:
    writes = folder / "writes.json"
    environment_ = {**os.environ, "LIVE_WRITES": str(writes)}
    command = [sys.executable, "-c", RECORDER, *engine.arguments]
    done = subprocess.run(
        command, stdin=producer.stdout, env=environment_, capture_output=True, text=True
    )
    if done.returncode:
        raise RuntimeError(f"ksay exited with {done.returncode}: {done.stderr[-500:]}")
    return playback(json.loads(writes.read_text()))


def run_say(engine: Engine, producer: subprocess.Popen, folder: Path) -> tuple:
    import soundfile as sf

    out = folder / "out.aiff"
    process = subprocess.Popen(
        ["say", *engine.arguments, "-o", str(out)], stdin=producer.stdout
    )
    first = None
    while process.poll() is None:
        if first is None and out.exists() and out.stat().st_size > SAY_HEADER:
            first = time.time()
        time.sleep(0.02)
    if first is None:  # it finished between two looks
        first = time.time()
    if process.returncode:
        raise RuntimeError(f"say exited with {process.returncode}")
    return first, first + sf.info(out).duration


def measure(engine: Engine, sentences: list[str], interval: float) -> dict:
    with tempfile.TemporaryDirectory() as folder:
        started = time.time()
        producer = subprocess.Popen(
            [sys.executable, "-c", PRODUCER, json.dumps(sentences), str(interval)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        run = run_ksay if engine.kind == "ksay" else run_say
        first, over = run(engine, producer, Path(folder))
        producer.stdout.close()
        stamps = json.loads(producer.communicate()[1])
    return {
        "first_sound": first - started,
        "speech_over": over - started,
        "text_finished": stamps[-1] - started,
    }


def table(results: dict) -> str:
    lines = [
        "| Engine | first sound | speech over |",
        "| --- | ---: | ---: |",
    ]
    for name, runs in results.items():
        first = statistics.median(run["first_sound"] for run in runs)
        over = statistics.median(run["speech_over"] for run in runs)
        lines.append(f"| {name} | {first:.1f} s | {over:.1f} s |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sentences", type=int, default=10, help="how many to send")
    parser.add_argument(
        "--interval", type=float, help="seconds between sentences (default: 2, then 0)"
    )
    parser.add_argument("--runs", type=int, default=3, help="runs per engine")
    parser.add_argument("--json", type=Path, help="write the results here")
    args = parser.parse_args()

    sentences = sentences_of(PARAGRAPH, args.sentences)
    intervals = [args.interval] if args.interval is not None else [2.0, 0.0]
    report: dict = {
        "environment": environment(),
        "sentences": len(sentences),
        "scenarios": {},
    }
    for interval in intervals:
        scenario = {}
        for engine in engines():
            print(f"{engine.name}, a sentence every {interval:g} s...", file=sys.stderr)
            measure(
                engine, sentences[:2], 0
            )  # warm-up: the first start loads libraries
            scenario[engine.name] = [
                measure(engine, sentences, interval) for _ in range(args.runs)
            ]
        report["scenarios"][f"{interval:g}"] = scenario
        written = (len(sentences) - 1) * interval
        print(
            f"\nA sentence every {interval:g} s: the text is written in {written:.0f} s"
        )
        print(table(scenario))
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
