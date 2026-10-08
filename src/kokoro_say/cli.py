"""Speak or save text with the Kokoro-82M voices, like macOS `say`."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import filecmp
import hashlib
import itertools
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import types
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


def reuse_espeak_copies(folder: Path) -> None:
    """Keep phonemizer's copies of the espeak-ng library between runs.

    phonemizer loads a private copy of the library for each espeak wrapper,
    four on first use, each written into a fresh temporary folder. macOS
    checks every newly written library before loading it, about half a second
    apiece, but remembers the files it has checked, so copies kept in `folder`
    pay that once instead of on every run.
    """
    from phonemizer.backend.espeak import api

    # Leave phonemizer alone if it has changed, or is patched already
    if getattr(api, "tempfile", None) is not tempfile:
        return
    if getattr(api, "shutil", None) is not shutil:
        return
    slots = itertools.count()

    def mkdtemp() -> str:  # one folder per wrapper, since each needs its own copy
        slot = folder / str(next(slots))
        slot.mkdir(parents=True, exist_ok=True)
        return str(slot)

    def copy(source, target, follow_symlinks=True) -> None:
        if os.path.exists(target) and filecmp.cmp(source, target, shallow=False):
            return
        partial = f"{target}.{os.getpid()}"
        shutil.copyfile(source, partial)
        os.replace(partial, target)  # never half written, even to another run

    def rmtree(path, *args, **kwargs) -> None:
        if not Path(path).is_relative_to(folder):
            shutil.rmtree(path, *args, **kwargs)

    api.tempfile = types.SimpleNamespace(**{**vars(tempfile), "mkdtemp": mkdtemp})
    api.shutil = types.SimpleNamespace(
        **{**vars(shutil), "copy": copy, "rmtree": rmtree}
    )


def performance_cores() -> int | None:
    """How many performance cores this Mac has, if it has two kinds.

    onnxruntime spreads synthesis over every core by default, and the slower
    efficiency cores hold up each parallel step: on an M2 Max, keeping to the
    eight performance cores synthesizes about a fifth faster.
    """
    if platform.system() != "Darwin":
        return None
    result = subprocess.run(
        ["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True
    )
    count = result.stdout.strip()
    return int(count) if result.returncode == 0 and count.isdigit() else None


def load_kokoro(folder: Path):
    """A ready synthesizer; kept separate so tests can replace it."""
    shorten_espeak_path()
    if platform.system() == "Darwin":  # where loading a fresh copy is slow
        reuse_espeak_copies(Path.home() / ".cache" / "kokoro-say" / "espeak")
    import onnxruntime
    from kokoro_onnx import Kokoro
    from kokoro_onnx.session import resolve_providers

    options = onnxruntime.SessionOptions()
    if cores := performance_cores():
        options.intra_op_num_threads = cores
    model, voices = (str(folder / name) for name in MODEL_FILES)
    session = onnxruntime.InferenceSession(
        model, options, providers=resolve_providers()
    )
    return Kokoro.from_session(session, voices)


def block_size() -> int:
    """Frames per audio buffer; 0 leaves the choice to PortAudio.

    PortAudio's low-latency default asks CoreAudio for the smallest buffer a
    device allows, 29 frames (0.3 ms) on a MacBook Pro's speakers. Such a stream
    overruns its I/O cycle as it stops, coreaudiod logs an "IO Overload", and
    the speech ends with a pop. 512 frames is what the built-in devices use
    otherwise, and speech needs no low latency.
    """
    return 512 if platform.system() == "Darwin" else 0


def open_output(rate: int):
    """A mono stream on the default audio device; kept separate for tests."""
    try:
        import sounddevice as sd
    except OSError as error:  # sounddevice loads PortAudio as it is imported
        raise RuntimeError(
            "playing speech needs PortAudio; on Debian or Ubuntu: "
            "sudo apt install libportaudio2, or save it with -o"
        ) from error
    try:
        return sd.OutputStream(
            samplerate=rate, channels=1, dtype="float32", blocksize=block_size()
        )
    except sd.PortAudioError as error:
        raise RuntimeError(f"cannot open the audio output: {error}") from error


@contextlib.contextmanager
def speaker():
    """The default audio device, opened before any speech is generated.

    Opening it first makes a missing device fail at once, and samples go
    straight to it instead of through a file and a player process. Python
    would turn Ctrl-C into an exception and then wait for the batch still
    being synthesized, so Ctrl-C keeps its default action and ends the process.
    """
    output = open_output(SAMPLE_RATE)
    interrupt = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        output.start()
        yield output
        output.stop()  # returns once the buffered audio has played
    finally:
        signal.signal(signal.SIGINT, interrupt)
        output.close()  # discards whatever an error left unplayed


def stream(output, parts: AsyncIterable) -> None:
    """Play the parts of Kokoro.create_stream while later ones are generated.

    kokoro-onnx synthesizes the next part while one plays, and one output
    stream carries them all, so they join without a gap whenever synthesis
    keeps ahead of playback.
    """

    async def play() -> None:
        async for samples, _ in parts:
            output.write(samples)

    asyncio.run(play())


def stdin_is_terminal() -> bool:
    """Whether nothing is piped in, so reading stdin would wait for a person."""
    return sys.stdin is None or sys.stdin.isatty()


def read_text(text: str | None, input_file: str | None) -> str:
    """The text to speak: the argument, a file, or stdin.

    Stdin is read for `-`, for `-f -`, and when neither an argument nor a file
    is given, the way `say` does.
    """
    source = text if input_file is None else input_file
    if source is None or source == "-":
        return sys.stdin.read()
    if input_file is None:
        return source
    try:
        # utf-8-sig, so a byte order mark from a Windows editor is not spoken
        return Path(input_file).expanduser().read_text(encoding="utf-8-sig")
    except OSError as error:
        raise RuntimeError(
            f"cannot read {input_file}: {error.strerror or error}"
        ) from error
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{input_file} is not UTF-8 text") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ksay",
        description="Speak or save text with the Kokoro-82M voices.",
        epilog="Example: ksay -v bf_emma 'Hello there.' -o hello.wav",
    )
    parser.add_argument(
        "text",
        nargs="?",
        help="text to speak; - reads stdin, and so does no text when input is piped",
    )
    parser.add_argument(
        "-f",
        "--input-file",
        metavar="FILE",
        help="read the text from FILE instead; - reads stdin",
    )
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "-o", "--output", help="save to this file (.wav, .flac, .ogg or .mp3) instead"
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
    listing = args.list_voices or args.voice == "?"
    if args.text is not None and args.input_file is not None:
        parser.error("give the text or -f FILE, not both")
    if not listing and args.text is None and args.input_file is None:
        if stdin_is_terminal():
            parser.error("give the text to speak, or -f FILE, or pipe it in")
    if not 0.5 <= args.speed <= 2.0:
        parser.error("speed must be between 0.5 and 2.0")

    try:
        # Read before loading the model, so a missing file fails at once
        text = None if listing else read_text(args.text, args.input_file)
        kokoro = load_kokoro(ensure_model(model_dir(args.model_dir)))
        voices = sorted(kokoro.get_voices())
        if listing:
            print("\n".join(voices))
            return 0
        if args.voice not in voices:
            parser.error(f"unknown voice {args.voice!r}; list them with: ksay -v '?'")
        if not text.strip():
            parser.error("there is no text to speak")
        lang = args.lang or language_for(args.voice)
        if args.output:
            import soundfile as sf

            samples, rate = kokoro.create(
                text, voice=args.voice, speed=args.speed, lang=lang
            )
            sf.write(args.output, samples, rate)
            print(f"saved {args.output} ({len(samples) / rate:.2f}s)", file=sys.stderr)
            return 0
        with speaker() as output:
            if args.stream:
                parts = kokoro.create_stream(
                    text, voice=args.voice, speed=args.speed, lang=lang
                )
                stream(output, parts)
            else:
                samples, _ = kokoro.create(
                    text, voice=args.voice, speed=args.speed, lang=lang
                )
                output.write(samples)
    except (OSError, RuntimeError) as error:
        print(f"ksay: {error}", file=sys.stderr)
        return 1
    return 0
