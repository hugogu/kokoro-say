"""Speak or save text with the Kokoro-82M voices, like macOS `say`."""

from __future__ import annotations

import argparse
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
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from kokoro_say import __version__, chinese
from kokoro_say.streaming import TextStream, speak, stdin_pieces

DEFAULT_VOICE = "af_heart"
MIN_SPEED = 0.5
MAX_SPEED = 2.0
SAMPLE_RATE = 24_000  # the only rate Kokoro-82M speaks at
SENTENCE_PAUSE = 0.25  # seconds of silence kokoro-onnx leaves between its own batches
CLAUSE_PAUSE = 0.1
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


MISSING_ZH = (
    "ksay: Chinese tones need Kokoro's own front end, the zh extra "
    "(kokoro-say[zh], Python 3.12 or older); speaking it with eSpeak NG, without tones"
)


MISSING_MCP = "ksay: --mcp needs the mcp extra (kokoro-say[mcp]): {}"


def to_kokoro(text: str, lang: str) -> tuple[str, bool]:
    """What to give Kokoro for this text, and whether that is phonemes already.

    Chinese goes through misaki when it is installed. Everything else is left to
    kokoro-onnx, which uses eSpeak NG.
    """
    if lang == "cmn" and chinese.available():
        return chinese.phonemize(text), True
    return text, False


def create(kokoro, spoken: str, phonemes: bool, **options):
    """Kokoro's samples and sample rate, or a RuntimeError saying why there are none."""
    try:
        return kokoro.create(spoken, is_phonemes=phonemes, **options)
    except ValueError as error:  # it found no phonemes in the text
        raise RuntimeError("there is nothing to say in that text") from error


def language_notes(text: str, lang: str, voice: str) -> list[str]:
    """Warnings for text that the chosen voice will not speak well."""
    if lang == "cmn" and not chinese.available():
        return [MISSING_ZH]
    if lang != "cmn" and chinese.has_chinese(text):
        return [
            f"ksay: the text has Chinese characters, which {voice} cannot read; "
            "try -v zf_xiaobei"
        ]
    return []


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
def device():
    """The default audio device, started, and released however the block ends."""
    output = open_output(SAMPLE_RATE)
    try:
        output.start()
        yield output
        output.stop()  # returns once the buffered audio has played
    finally:
        output.close()  # discards whatever an error left unplayed


@contextlib.contextmanager
def speaker():
    """The default audio device, opened before any speech is generated.

    Opening it first makes a missing device fail at once, and samples go
    straight to it instead of through a file and a player process. Python
    would turn Ctrl-C into an exception and then wait for the batch still
    being synthesized, so Ctrl-C keeps its default action and ends the process.
    That is a setting of the main thread, which is why `device` is apart.
    """
    interrupt = signal.signal(signal.SIGINT, signal.SIG_DFL)
    try:
        with device() as output:
            yield output
    finally:
        signal.signal(signal.SIGINT, interrupt)


def synthesizer(kokoro, voice: str, speed: float, lang: str):
    """A function from one sentence to the samples that speak it.

    Each sentence is led by the pause the one before it calls for, so the
    speech joins as kokoro-onnx joins its own batches and ends without a gap.
    """
    import numpy as np
    from kokoro_onnx.chunker import pause_after

    pause = 0.0
    told = False

    def synthesize(sentence: str):
        nonlocal pause, told
        if not told:  # once, for the first sentence that shows it
            told = bool(notes := language_notes(sentence, lang, voice))
            for note in notes:
                print(note, file=sys.stderr)
        spoken, phonemes = to_kokoro(sentence, lang)
        try:
            samples, rate = kokoro.create(
                spoken, voice=voice, speed=speed, lang=lang, is_phonemes=phonemes
            )
        except ValueError:  # nothing in it can be said, such as "***" or an emoji
            return None
        lead, pause = pause, pause_after(spoken, SENTENCE_PAUSE, CLAUSE_PAUSE)
        return np.pad(samples, (round(lead * rate), 0))

    return synthesize


def stdin_is_terminal() -> bool:
    """Whether nothing is piped in, so reading stdin would wait for a person."""
    return sys.stdin is None or sys.stdin.isatty()


def reads_stdin(text: str | None, input_file: str | None) -> bool:
    """Whether the text comes from stdin: `-`, `-f -`, or neither text nor file."""
    source = text if input_file is None else input_file
    return source is None or source == "-"


def read_text(text: str | None, input_file: str | None) -> str:
    """The text to speak: the argument, a file, or all of stdin."""
    if reads_stdin(text, input_file):
        return sys.stdin.read()
    if input_file is None:
        return text
    try:
        # utf-8-sig, so a byte order mark from a Windows editor is not spoken
        return Path(input_file).expanduser().read_text(encoding="utf-8-sig")
    except OSError as error:
        raise RuntimeError(
            f"cannot read {input_file}: {error.strerror or error}"
        ) from error
    except UnicodeDecodeError as error:
        raise RuntimeError(f"{input_file} is not UTF-8 text") from error


def check_output(target: Path) -> None:
    """Raise a RuntimeError if speech cannot be saved there, before any is made."""
    import soundfile as sf

    if target.suffix[1:].upper() not in sf.available_formats():
        raise RuntimeError(
            f"cannot save {target}: end the name in .wav, .flac, .ogg or .mp3"
        )
    if not target.parent.is_dir():
        raise RuntimeError(
            f"cannot save {target}: the folder {target.parent} does not exist"
        )


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
        help="speak each sentence as soon as it has arrived and been generated",
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
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="serve MCP on standard input and output, for an AI assistant to speak "
        "through, instead of speaking text (needs the mcp extra)",
    )
    parser.add_argument("--version", action="version", version=__version__)
    return parser


def serve_mcp(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """`ksay --mcp`: answer MCP requests instead of speaking text."""
    listing = args.list_voices or args.voice == "?"
    given = [args.text, args.input_file, args.output, args.lang]
    if any(value is not None for value in given) or args.stream or listing:
        parser.error("--mcp takes no text, -f, -o, --stream, -l or --list-voices")
    try:
        from kokoro_say import mcp_server
    except ImportError as error:  # not installed, or too old; say which
        print(MISSING_MCP.format(error), file=sys.stderr)
        return 1
    mcp_server.serve(args.voice, args.speed, args.model_dir)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not MIN_SPEED <= args.speed <= MAX_SPEED:
        parser.error(f"speed must be between {MIN_SPEED} and {MAX_SPEED}")
    if args.mcp:
        return serve_mcp(parser, args)
    listing = args.list_voices or args.voice == "?"
    if args.text is not None and args.input_file is not None:
        parser.error("give the text or -f FILE, not both")
    if not listing and args.text is None and args.input_file is None:
        if stdin_is_terminal():
            parser.error("give the text to speak, or -f FILE, or pipe it in")

    try:
        if args.output:
            check_output(Path(args.output))
        # Read before loading the model, so a missing file fails at once
        text = live = None
        if args.stream and not listing:
            pieces = (
                stdin_pieces()
                if reads_stdin(args.text, args.input_file)
                else [read_text(args.text, args.input_file)]
            )
            live = TextStream(pieces)  # reads on at once: text may arrive meanwhile
        elif not listing:
            text = read_text(args.text, args.input_file)
        kokoro = load_kokoro(ensure_model(model_dir(args.model_dir)))
        voices = sorted(kokoro.get_voices())
        if listing:
            print("\n".join(voices))
            return 0
        if args.voice not in voices:
            parser.error(f"unknown voice {args.voice!r}; list them with: ksay -v '?'")
        lang = args.lang or language_for(args.voice)
        if live is not None:
            with speaker() as output:
                synthesize = synthesizer(kokoro, args.voice, args.speed, lang)
                spoken = speak(output, synthesize, live)
            if not spoken:
                parser.error("there is no text to speak")
            return 0
        if not text.strip():
            parser.error("there is no text to speak")
        for note in language_notes(text, lang, args.voice):
            print(note, file=sys.stderr)
        spoken, phonemes = to_kokoro(text, lang)
        options = {"voice": args.voice, "speed": args.speed, "lang": lang}
        if args.output:
            import soundfile as sf

            samples, rate = create(kokoro, spoken, phonemes, **options)
            sf.write(args.output, samples, rate)
            print(f"saved {args.output} ({len(samples) / rate:.2f}s)", file=sys.stderr)
            return 0
        with speaker() as output:
            samples, _ = create(kokoro, spoken, phonemes, **options)
            output.write(samples)
    except (OSError, RuntimeError) as error:
        print(f"ksay: {error}", file=sys.stderr)
        return 1
    return 0
