"""Speak or save text with the Kokoro-82M voices, like macOS `say`."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import urllib.request
from collections.abc import AsyncIterable, Sequence
from pathlib import Path
from typing import NamedTuple

from kokoro_say import __version__

DEFAULT_VOICE = "af_heart"
SAMPLE_RATE = 24_000  # the only rate Kokoro-82M speaks at
MODEL_RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)


class ModelFile(NamedTuple):
    url: str
    size: int
    sha256: str


MODEL_FILES = {
    "kokoro-v1.0.onnx": ModelFile(
        f"{MODEL_RELEASE}/kokoro-v1.0.onnx",
        325_532_387,
        "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    ),
    "voices-v1.0.bin": ModelFile(
        f"{MODEL_RELEASE}/voices-v1.0.bin",
        28_214_398,
        "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
    ),
}

# The first letter of a Kokoro voice name is its language.
LANGUAGES = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "cmn",
}


def language_for(voice: str) -> str:
    """The espeak language a voice speaks, from its prefix."""
    return LANGUAGES.get(voice[:1], "en-us")


def model_dir(argument: str | None = None) -> Path:
    """Where the model lives: the argument, then $KOKORO_MODELS, then the cache."""
    configured = argument or os.environ.get("KOKORO_MODELS")
    return Path(configured or Path.home() / ".cache" / "kokoro-onnx").expanduser()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(model: ModelFile, target: Path) -> None:
    partial = target.with_name(target.name + ".part")
    try:
        with urllib.request.urlopen(model.url) as response, partial.open("wb") as out:
            done = 0
            while block := response.read(1 << 20):
                out.write(block)
                done += len(block)
                print(
                    f"\rdownloading {target.name}: {done * 100 // model.size}%",
                    end="",
                    file=sys.stderr,
                )
        print(file=sys.stderr)
        if _sha256(partial) != model.sha256:
            raise RuntimeError(f"{target.name} failed its checksum; try again")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def ensure_model(folder: Path, files: dict[str, ModelFile] = MODEL_FILES) -> Path:
    """Download whichever model files the folder lacks, once."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, model in files.items():
        if not (folder / name).exists():
            _download(model, folder / name)
    return folder


def shorten_espeak_path() -> None:
    """Point espeak-ng at its data through a short path.

    espeak-ng keeps the data path in a 160-byte buffer and silently falls back to
    its build machine's path when the real one is longer, which a deep virtual
    environment can be. phonemizer would resolve a link back to the long path,
    so its property is replaced as well.
    """
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    data = Path(espeakng_loader.get_data_path())
    if len(str(data)) < 150:
        return
    link = Path(tempfile.mkdtemp(prefix="kokoro-say-")) / "espeak-ng-data"
    try:
        link.symlink_to(data, target_is_directory=True)
    except OSError:  # e.g. Windows without symlink rights: keep the long path
        return
    EspeakWrapper.data_path = property(lambda self: link)


def load_kokoro(folder: Path):
    """A ready synthesizer; kept separate so tests can replace it."""
    shorten_espeak_path()
    from kokoro_onnx import Kokoro

    return Kokoro(*(str(folder / name) for name in MODEL_FILES))


def player_command(path: str) -> list[str] | None:
    """A command that plays a WAV file on this system, if one is installed."""
    system = platform.system()
    if system == "Darwin":
        return ["afplay", path]
    if system == "Linux":
        for command in (
            ["paplay", path],
            ["aplay", "-q", path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", path],
        ):
            if shutil.which(command[0]):
                return command
    return None


def play(samples, rate: int) -> None:
    import soundfile as sf

    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "speech.wav")
        sf.write(path, samples, rate)
        if platform.system() == "Windows":
            import winsound

            winsound.PlaySound(path, winsound.SND_FILENAME)
            return
        command = player_command(path)
        if command is None:
            raise RuntimeError("no audio player found; save to a file with -o")
        subprocess.run(command, check=True)


def open_output(rate: int):
    """A mono stream on the default audio device; kept separate for tests."""
    try:
        import sounddevice as sd
    except OSError as error:  # sounddevice loads PortAudio as it is imported
        raise RuntimeError(
            "--stream needs PortAudio; on Debian or Ubuntu: "
            "sudo apt install libportaudio2"
        ) from error
    try:
        return sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
    except sd.PortAudioError as error:
        raise RuntimeError(f"cannot open the audio output: {error}") from error


def stream(parts: AsyncIterable) -> None:
    """Play the parts of Kokoro.create_stream while later ones are generated.

    kokoro-onnx synthesizes the next part while one plays, and one output
    stream carries them all, so they join without a gap whenever synthesis
    keeps ahead of playback.
    """
    output = open_output(SAMPLE_RATE)  # first, so a missing device fails fast

    async def play() -> None:
        async for samples, _ in parts:
            output.write(samples)

    # Python would turn Ctrl-C into an exception and then wait for the batch
    # still being synthesized; the default action ends the process at once.
    interrupt = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        output.start()
        asyncio.run(play())
        output.stop()  # returns once the buffered audio has played
    finally:
        signal.signal(signal.SIGINT, interrupt)
        output.close()  # discards whatever an error left unplayed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kokoro-say",
        description="Speak or save text with the Kokoro-82M voices.",
        epilog="Example: kokoro-say -v bf_emma 'Hello there.' -o hello.wav",
    )
    parser.add_argument("text", nargs="?", help="text to speak, or - to read stdin")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o", "--output", help="save to this file (.wav, .flac or .ogg) instead"
    )
    destination.add_argument(
        "--stream",
        action="store_true",
        help="start speaking before the whole text has been generated",
    )
    parser.add_argument(
        "-v",
        "--voice",
        default=DEFAULT_VOICE,
        help=f"voice name, or ? to list them (default: {DEFAULT_VOICE})",
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=1.0, help="speaking rate (default: 1.0)"
    )
    parser.add_argument(
        "-l", "--lang", help="espeak language code (default: from the voice)"
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="list the voices and exit"
    )
    parser.add_argument(
        "--model-dir",
        help="model folder (default: $KOKORO_MODELS, else ~/.cache/kokoro-onnx)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.text is None and not (args.list_voices or args.voice == "?"):
        parser.error("give the text to speak, or - to read it from stdin")
    if not 0.5 <= args.speed <= 2.0:
        parser.error("speed must be between 0.5 and 2.0")

    try:
        kokoro = load_kokoro(ensure_model(model_dir(args.model_dir)))
        voices = sorted(kokoro.get_voices())
        if args.list_voices or args.voice == "?":
            print("\n".join(voices))
            return 0
        if args.voice not in voices:
            parser.error(
                f"unknown voice {args.voice!r}; list them with: kokoro-say -v '?'"
            )
        text = sys.stdin.read() if args.text == "-" else args.text
        if not text.strip():
            parser.error("there is no text to speak")
        lang = args.lang or language_for(args.voice)
        if args.stream:
            parts = kokoro.create_stream(
                text, voice=args.voice, speed=args.speed, lang=lang
            )
            stream(parts)
            return 0
        samples, rate = kokoro.create(
            text, voice=args.voice, speed=args.speed, lang=lang
        )
        if args.output:
            import soundfile as sf

            sf.write(args.output, samples, rate)
            print(f"saved {args.output} ({len(samples) / rate:.2f}s)", file=sys.stderr)
        else:
            play(samples, rate)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"kokoro-say: {error}", file=sys.stderr)
        return 1
    return 0
