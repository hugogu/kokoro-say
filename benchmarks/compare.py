#!/usr/bin/env -S uv run --script
"""Compare ksay with macOS say and eSpeak NG on speed, memory and intelligibility.

    uv run benchmarks/compare.py                # speed and memory
    uv run benchmarks/compare.py --wer          # plus intelligibility, using Whisper
    uv run benchmarks/compare.py --samples DIR  # keep the audio, to listen to it

Speed is the wall time of one command, from launch until its file is written, so
it includes starting the program. Memory is that process's peak resident set size.
Intelligibility is the word error rate of Whisper's transcript of 30 sentences:
20 Harvard sentences (plain speech) and 10 with numbers, dates and names. Both
sides go through Whisper's English text normalizer, so "$1,250" and "twelve
hundred and fifty dollars" count as the same, and for the numbers set so do
"415-555-0132" and "415 555 0132". It tells how clearly the words come across, not
how natural the voice sounds; mos.py estimates that from the audio --samples keeps.

Unix only, because peak memory comes from wait4. say is skipped off macOS. Apple's
licence does not allow publishing recordings of its voices, so keep the audio that
--samples writes for say to yourself.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "kokoro-say",
#   "numpy",
#   "scipy",
#   "soundfile",
#   "faster-whisper",
#   "jiwer",
#   "whisper-normalizer",
# ]
#
# [tool.uv.sources]
# kokoro-say = { path = "..", editable = true }
# ///

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from math import gcd
from pathlib import Path

HERE = Path(__file__).resolve().parent
SHORT = "Good morning, how are you today?"
PARAGRAPH = (HERE / "paragraph.txt").read_text().strip()
MEDIUM = ". ".join(PARAGRAPH.split(". ")[:4]) + "."
SETS = {  # name -> (sentences, ignore formatting between digits)
    "plain": ((HERE / "harvard.txt").read_text().splitlines(), False),
    "numbers": ((HERE / "numbers.txt").read_text().splitlines(), True),
}
TEXTS = {"short": SHORT, "medium": MEDIUM, "long": PARAGRAPH}
WHISPER_MODEL = "small.en"


@dataclass
class Engine:
    name: str
    suffix: str
    command: Callable[[str, Path], list[str]]


def engines() -> list[Engine]:
    found = [
        Engine(
            "ksay",
            ".wav",
            lambda text, out: [
                sys.executable,
                "-m",
                "kokoro_say",
                text,
                "-o",
                str(out),
            ],
        )
    ]
    if sys.platform == "darwin" and shutil.which("say"):
        # Plain say speaks with the system voice, which can be a neural voice that
        # `say -v '?'` does not list; naming Samantha asks for the classic compact one.
        found.append(
            Engine(
                "say (default voice)",
                ".aiff",
                lambda text, out: ["say", "-o", str(out), text],
            )
        )
        found.append(
            Engine(
                "say (Samantha)",
                ".aiff",
                lambda text, out: ["say", "-v", "Samantha", "-o", str(out), text],
            )
        )
    found.append(
        Engine(
            "eSpeak NG",
            ".wav",
            lambda text, out: [sys.executable, __file__, "--espeak", str(out), text],
        )
    )
    return found


def espeak(text: str):
    """Synthesize with the eSpeak NG library that kokoro-onnx installs anyway."""
    import ctypes

    import espeakng_loader
    import numpy as np

    data = Path(espeakng_loader.get_data_path())
    if len(str(data)) >= 150:  # espeak-ng keeps this path in a 160-byte buffer
        link = Path(tempfile.mkdtemp()) / "data"
        link.symlink_to(data, target_is_directory=True)
        data = link
    library = ctypes.CDLL(espeakng_loader.get_library_path())
    library.espeak_Synth.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    chunks = []

    @ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
    )
    def collect(wave, count, events):
        if wave:
            chunks.append(np.ctypeslib.as_array(wave, shape=(count,)).copy())
        return 0

    rate = library.espeak_Initialize(2, 0, str(data).encode(), 0)  # 2: no playback
    library.espeak_SetSynthCallback(collect)
    library.espeak_SetVoiceByName(b"en-us")
    encoded = text.encode()
    library.espeak_Synth(encoded, len(encoded) + 1, 0, 0, 0, 1, None, None)  # 1: UTF-8
    library.espeak_Terminate()
    return np.concatenate(chunks).astype(np.float32) / 32768, rate


def run(command: list[str]) -> tuple[float, float]:
    """Run a command; return its wall time in seconds and its peak memory in MB."""
    start = time.perf_counter()
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    _, status, usage = os.wait4(process.pid, 0)
    seconds = time.perf_counter() - start
    if code := os.waitstatus_to_exitcode(status):
        raise RuntimeError(f"{command[0]} exited with {code}")
    per_mb = 1024 * 1024 if sys.platform == "darwin" else 1024  # bytes or KB
    return seconds, usage.ru_maxrss / per_mb


def speed_and_memory(engine: Engine, text: str, runs: int) -> dict:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / f"out{engine.suffix}"
        run(engine.command(text, out))  # warm-up: the first start loads libraries
        timed = [run(engine.command(text, out)) for _ in range(runs)]
        audio = sf.info(out).duration
    seconds = statistics.median(t for t, _ in timed)
    return {
        "seconds": seconds,
        "audio_seconds": audio,
        "times_real": audio / seconds,
        "peak_mb": max(m for _, m in timed),
    }


def synthesize(
    engine: Engine, texts: list[str], folder: Path, prefix: str
) -> list[Path]:
    """One file per text. ksay loads its model once instead of once per file."""
    import soundfile as sf

    paths = [
        folder / f"{prefix}-{index:02d}{engine.suffix}" for index in range(len(texts))
    ]
    if engine.name == "ksay":
        from kokoro_say import cli

        kokoro = cli.load_kokoro(cli.ensure_model(cli.model_dir()))
        for text, path in zip(texts, paths, strict=True):
            samples, rate = kokoro.create(text, voice=cli.DEFAULT_VOICE, lang="en-us")
            sf.write(path, samples, rate)
    else:
        for text, path in zip(texts, paths, strict=True):
            run(engine.command(text, path))
    return paths


def transcribe(paths: list[Path]) -> list[str]:
    import soundfile as sf
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    heard = []
    for path in paths:
        samples, rate = sf.read(path, dtype="float32")
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        divisor = gcd(16_000, rate)
        samples = resample_poly(samples, 16_000 // divisor, rate // divisor)
        segments, _ = model.transcribe(
            samples.astype("float32"),
            language="en",
            beam_size=5,
            condition_on_previous_text=False,
        )
        heard.append(" ".join(segment.text.strip() for segment in segments))
    return heard


def intelligibility(engine: Engine, samples: Path | None) -> dict:
    import jiwer
    from whisper_normalizer.english import EnglishTextNormalizer

    normalizer = EnglishTextNormalizer()

    def canonical(text: str, ignore_digit_formatting: bool) -> str:
        text = normalizer(text)
        if ignore_digit_formatting:  # 415-555-0132 is 415 555 0132 is 4155550132
            text = re.sub(r"(?<=\d)[\s,.:-]+(?=\d)", "", text)
        return text

    results = {}
    for name, (sentences, lenient) in SETS.items():
        with tempfile.TemporaryDirectory() as folder:
            paths = synthesize(engine, sentences, Path(folder), name)
            heard = transcribe(paths)
            if samples:
                target = samples / re.sub(
                    r"[^a-z0-9]+", "-", engine.name.lower()
                ).strip("-")
                target.mkdir(parents=True, exist_ok=True)
                for path in paths:
                    shutil.copy(path, target / path.name)
        references = [canonical(text, lenient) for text in sentences]
        hypotheses = [canonical(text, lenient) for text in heard]
        wrong = [
            {"said": text, "heard": got}
            for text, ref, hyp, got in zip(
                sentences, references, hypotheses, heard, strict=True
            )
            if ref != hyp
        ]
        results[name] = {
            "wer": jiwer.wer(references, hypotheses),
            "perfect": len(sentences) - len(wrong),
            "sentences": len(sentences),
            "mistakes": wrong,
        }
    return results


def environment() -> dict:
    import importlib.metadata as metadata

    if sys.platform == "darwin":
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
        system = f"macOS {platform.mac_ver()[0]}"
    else:
        chip, system = platform.processor() or platform.machine(), platform.platform()
    try:
        commit = subprocess.run(
            ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):  # no git, or not a checkout
        commit = "unknown"
    return {
        "date": time.strftime("%Y-%m-%d"),
        "system": system,
        "chip": chip,
        "cores": os.cpu_count(),
        "memory_gb": round(
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30
        ),
        "python": platform.python_version(),
        "onnxruntime": metadata.version("onnxruntime"),
        "kokoro_onnx": metadata.version("kokoro-onnx"),
        "ksay_commit": commit,
        "whisper_model": WHISPER_MODEL,
    }


def table(results: dict) -> str:
    names = list(results["speed"])
    lines = [
        "| Engine | short | paragraph | long | speed on long text | peak memory |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in names:
        row = results["speed"][name]
        times = " | ".join(f"{row[key]['seconds']:.2f} s" for key in TEXTS)
        long = row["long"]
        speed, memory = long["times_real"], long["peak_mb"]
        lines.append(f"| {name} | {times} | {speed:.1f}× real time | {memory:.0f} MB |")
    if "intelligibility" in results:
        lines += [
            "",
            "| Engine | plain speech: word error rate | plain: heard perfectly "
            "| numbers and names: word error rate | numbers: heard perfectly |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name, row in results["intelligibility"].items():
            plain, numbers = row["plain"], row["numbers"]
            cells = [f"{name}"]
            for part in (plain, numbers):
                cells += [
                    f"{part['wer'] * 100:.1f}%",
                    f"{part['perfect']} of {part['sentences']}",
                ]
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    if len(sys.argv) == 4 and sys.argv[1] == "--espeak":  # used by the speed runs
        import soundfile as sf

        samples, rate = espeak(sys.argv[3])
        sf.write(sys.argv[2], samples, rate)
        return 0

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--runs", type=int, default=5, help="timed runs per text")
    parser.add_argument("--wer", action="store_true", help="measure intelligibility")
    parser.add_argument("--samples", type=Path, help="keep the sentence audio here")
    parser.add_argument("--json", type=Path, help="write the results here")
    args = parser.parse_args()

    results: dict = {"environment": environment(), "speed": {}}
    for engine in engines():
        print(f"timing {engine.name}...", file=sys.stderr)
        results["speed"][engine.name] = {
            key: speed_and_memory(
                engine, text, args.runs if key != "long" else max(2, args.runs // 2)
            )
            for key, text in TEXTS.items()
        }
    if args.wer:
        results["intelligibility"] = {}
        for engine in engines():
            print(f"transcribing {engine.name}...", file=sys.stderr)
            results["intelligibility"][engine.name] = intelligibility(
                engine, args.samples
            )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["environment"]))
    print(table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
