#!/usr/bin/env -S uv run --script
"""Render docs/audio/intro.mp4: the introduction as a video GitHub can play inline.

GitHub removes audio and video tags from a README, and plays inline only a video
uploaded through its own editor. So the introduction is wrapped in a plain video:
a waveform, the command being described, and captions. Drag the file into a GitHub
comment box, copy the URL it gives you and put it on a line of its own in the README.

    uv run scripts/make_intro_video.py

Run scripts/make_audio.py first. Needs ffmpeg and macOS system fonts. Whisper times
the words, so that the commands and captions change when the narration reaches them.
"""
# /// script
# requires-python = ">=3.11"
# dependencies = ["faster-whisper", "numpy", "scipy", "soundfile"]
# ///

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import textwrap
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIO = ROOT / "docs" / "audio"
MONO = Path("/System/Library/Fonts/Menlo.ttc")
SANS = Path("/System/Library/Fonts/HelveticaNeue.ttc")

# What the panel shows while each sentence of intro.txt is spoken. None keeps
# the bare prompt.
COMMANDS = [
    None,
    None,
    None,
    'ksay "Hello there."',
    'ksay "Hello there." -o hello.mp3',
    "ksay --stream - < chapter.txt",
    "ksay -v '?'",
    "uv tool install git+https://github.com/hugogu/kokoro-say",
]


def sentences(script: str) -> list[str]:
    found = re.findall(r"[^.!?]+[.!?]", " ".join(script.split()))
    return [sentence.strip() for sentence in found]


def word_times(path: Path) -> list[tuple[str, float, float]]:
    import soundfile as sf
    from faster_whisper import WhisperModel
    from scipy.signal import resample_poly

    wave, rate = sf.read(path, dtype="float32")
    wave = resample_poly(wave, 16_000, rate).astype("float32")
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(
        wave,
        language="en",
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    return [(w.word, w.start, w.end) for s in segments for w in s.words]


def sentence_starts(
    script_sentences: list[str], words: list[tuple[str, float, float]]
) -> list[float]:
    """When the narration reaches each sentence, from aligning script and transcript."""
    tokens = [
        re.findall(r"[a-z0-9']+", s.lower().replace("-", " ")) for s in script_sentences
    ]
    flat = [token for sentence in tokens for token in sentence]
    owner = [i for i, sentence in enumerate(tokens) for _ in sentence]
    heard = [re.sub(r"[^a-z0-9']", "", word.lower()) for word, _, _ in words]
    starts: dict[int, float] = {}
    for tag, a1, a2, b1, b2 in SequenceMatcher(
        None, flat, heard, autojunk=False
    ).get_opcodes():
        if tag in ("equal", "replace") and b2 > b1:
            for index in range(a1, a2):
                spoken = b1 + min(b2 - b1 - 1, (index - a1) * (b2 - b1) // (a2 - a1))
                starts[owner[index]] = min(
                    starts.get(owner[index], 1e9), words[spoken][1]
                )
    if len(starts) != len(script_sentences):
        raise SystemExit("could not match every sentence of intro.txt to the recording")
    return [starts[i] for i in range(len(script_sentences))]


def drawtext(folder: Path, name: str, text: str, font: Path, **options: object) -> str:
    """A drawtext filter reading its text from a file, which needs no escaping."""
    (folder / name).write_text(text)
    settings = {
        "fontfile": font,
        "textfile": folder / name,
        "expansion": "none",
        **options,
    }
    return "drawtext=" + ":".join(f"{key}={value}" for key, value in settings.items())


def main() -> int:
    for font in (MONO, SANS):
        if not font.exists():
            raise SystemExit(f"{font} not found; this script needs macOS system fonts")
    mp3 = AUDIO / "intro.mp3"
    script = sentences((AUDIO / "intro.txt").read_text())
    starts = sentence_starts(script, word_times(mp3))
    duration = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                mp3,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )
    ends = [*starts[1:], duration + 0.5]

    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        grey, blue, green = "0x8b949e", "0x58a6ff", "0x3fb950"
        filters = [
            "[v0]drawbox=x=80:y=440:w=1120:h=110:color=0x161b22:t=fill",
            "drawbox=x=80:y=440:w=1120:h=110:color=0x30363d:t=2",
            drawtext(
                folder,
                "tag",
                "Natural text-to-speech for your terminal",
                SANS,
                x=80,
                y=160,
                fontsize=30,
                fontcolor=grey,
            ),
            drawtext(
                folder, "prompt", "$", MONO, x=110, y=478, fontsize=34, fontcolor=green
            ),
            drawtext(
                folder,
                "cursor",
                "_",
                MONO,
                x=146,
                y=478,
                fontsize=34,
                fontcolor="white",
                enable=f"'lt(t,{starts[3]:.2f})*lt(mod(t,1),0.6)'",
            ),
        ]
        for i, (command, start, end) in enumerate(
            zip(COMMANDS, starts, ends, strict=True)
        ):
            window = f"'between(t,{start:.2f},{end:.2f})'"
            if command:
                filters.append(
                    drawtext(
                        folder,
                        f"cmd{i}",
                        command,
                        MONO,
                        x=146,
                        y=478,
                        fontsize=34 if len(command) <= 40 else 28,
                        fontcolor="white",
                        enable=window,
                    )
                )
            caption = "\n".join(textwrap.wrap(script[i], 64))
            filters.append(
                drawtext(
                    folder,
                    f"cap{i}",
                    caption,
                    SANS,
                    x=80,
                    y=590,
                    fontsize=34,
                    fontcolor="white",
                    line_spacing=10,
                    enable=window,
                )
            )
        graph = (
            f"color=c=0x0d1117:s=1280x720:r=30:d={duration + 0.5:.2f}[bg];"
            "[0:a]asplit[a][w];"
            f"[w]showwaves=s=1120x180:mode=cline:rate=30:colors={blue}:scale=lin,"
            "format=rgba,colorkey=black:0.1:0.1[wv];"
            "[bg][wv]overlay=80:230:shortest=1:format=auto[v0a];"
            "[1:v]scale=400:-1[logo];[v0a][logo]overlay=72:20[v0];"
            + ",".join(filters)
            + "[v]"
        )
        (folder / "graph.txt").write_text(graph)
        target = AUDIO / "intro.mp4"
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", mp3]
            + ["-i", ROOT / "docs" / "logo" / "ksay-logo-dark.png"]
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
