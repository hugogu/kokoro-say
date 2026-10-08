#!/usr/bin/env python3
"""How long ksay's libraries take to import under each Python given.

    python benchmarks/imports.py ENV-A/bin/python ENV-B/bin/python

Each argument is the interpreter of an environment where ksay is installed. Start-up
is mostly imports, and the same code imports at very different speeds on different
Python builds: this is the part of ksay's run time that its own code does not control.
Prints the median of seven runs after one warm-up run, for a bare interpreter and for
the libraries ksay loads before it can speak.
"""

from __future__ import annotations

import statistics
import subprocess
import sys
import time

LIBRARIES = "import numpy, onnxruntime, kokoro_onnx, phonemizer, sounddevice, soundfile"
RUNS = 7


def median_seconds(python: str, code: str) -> float:
    def once() -> float:
        start = time.perf_counter()
        subprocess.run([python, "-c", code], check=True, capture_output=True)
        return time.perf_counter() - start

    once()  # the first run pays for compiling bytecode and for the system's checks
    return statistics.median(once() for _ in range(RUNS))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    print("| Python | bare interpreter | ksay's libraries |")
    print("| --- | ---: | ---: |")
    for python in sys.argv[1:]:
        version = subprocess.run(
            [python, "-c", "import platform; print(platform.python_version())"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        bare = median_seconds(python, "pass")
        libraries = median_seconds(python, LIBRARIES)
        print(f"| {version} | {bare * 1000:.0f} ms | {libraries:.2f} s |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
