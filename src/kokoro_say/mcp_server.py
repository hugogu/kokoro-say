"""Let AI assistants speak through ksay, over the Model Context Protocol.

`ksay --mcp` answers MCP on standard input and output, which is how Claude Code,
Claude Desktop, Cursor and other assistants start a local server. It offers
`speak`, which says text through the speakers, `save_speech`, which writes an audio
file, and `list_voices`. They use what the command uses: the model, the Chinese
front end, the sentence chunker and the sound card.

A command speaks once and exits. A server answers one request after another for
days, so it loads the model on the first request and keeps it, takes requests in
turn, stops speaking when a request is cancelled, and asks PortAudio to look at the
audio devices again before each speech.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import signal
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Annotated

import anyio.to_thread
import soundfile as sf
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from kokoro_say import __version__, cli, streaming

VOICE_HELP = (
    "Voice name. Its first letter is the language: a American English, b British "
    "English, e Spanish, f French, h Hindi, i Italian, j Japanese, p Brazilian "
    "Portuguese, z Mandarin Chinese; the second is f or m. A voice reads its own "
    "language only, so Chinese text needs a z voice such as zf_xiaobei. "
    "list_voices gives every name."
)
Text = Annotated[str, Field(description="The text to say.")]
Voice = Annotated[str, Field(description=VOICE_HELP)]
Speed = Annotated[
    float,
    Field(
        ge=cli.MIN_SPEED,
        le=cli.MAX_SPEED,
        description=f"Speaking rate, from {cli.MIN_SPEED} (slow) to "
        f"{cli.MAX_SPEED} (fast); 1.0 is natural.",
    ),
]
Destination = Annotated[
    str,
    Field(
        description="Where to write the file, ending in .wav, .flac, .ogg or .mp3. "
        "~ is the home folder. A relative path is relative to the folder the "
        "server runs in, so give a full path."
    ),
]


def rescan_devices() -> None:
    """Make PortAudio look at the audio devices again.

    PortAudio lists the devices once, as it starts. A command speaks seconds after
    that, but a server lives long enough for headphones to be plugged in and the
    default output to change. Between two speeches no stream is open, so starting
    PortAudio afresh is safe, and it takes about 4 ms.
    """
    try:
        import sounddevice as sd
    except OSError:  # opening the device explains that PortAudio is missing
        return
    try:
        sd._terminate()
        sd._initialize()
    except AttributeError:  # private functions, which a later release may drop
        return
    except sd.PortAudioError as error:
        raise RuntimeError(f"cannot open the audio output: {error}") from error


class Engine:
    """The model, loaded when first needed, and what it takes to share it.

    Requests arrive on worker threads. One speaks or saves at a time, in the order
    asked. The model is used by one thread at a time as well, which matters when
    the last sentence of a speech that was stopped is still being synthesized as
    the next speech begins.
    """

    def __init__(self, models: str | None = None):
        self.models = models
        self._kokoro = None
        self._turn = threading.Lock()
        self._model = threading.Lock()

    def kokoro(self):
        """The synthesizer, which the first call loads, after any download."""
        with self._model:
            if self._kokoro is None:
                folder = cli.ensure_model(cli.model_dir(self.models))
                self._kokoro = cli.load_kokoro(folder)
            return self._kokoro

    def voices(self) -> list[str]:
        return sorted(self.kokoro().get_voices())

    def prepare(self, text: str, voice: str):
        """The model, and the language of the voice, once there is something to say."""
        if not text.strip():
            raise RuntimeError("there is no text to speak")
        kokoro = self.kokoro()
        if voice not in kokoro.get_voices():
            raise RuntimeError(
                f"unknown voice {voice!r}; list them with the list_voices tool"
            )
        return kokoro, cli.language_for(voice)

    @contextlib.contextmanager
    def turn(self, stop: threading.Event) -> Iterator[None]:
        """Wait to be the one that speaks, unless a stop comes first."""
        while not self._turn.acquire(timeout=streaming.TICK):
            if stop.is_set():
                raise streaming.Stopped
        try:
            if stop.is_set():
                raise streaming.Stopped
            yield
        finally:
            self._turn.release()

    def speak(self, text: str, voice: str, speed: float, stop: threading.Event) -> str:
        kokoro, lang = self.prepare(text, voice)
        notes = cli.language_notes(text, lang, voice)
        say = cli.synthesizer(kokoro, voice, speed, lang)
        frames = 0

        def synthesize(sentence: str):
            nonlocal frames
            with self._model:
                samples = say(sentence)
            if samples is not None:
                frames += len(samples)
            return samples

        with self.turn(stop):
            rescan_devices()
            with cli.device() as output:
                spoken = streaming.speak(
                    output, synthesize, streaming.TextStream([text]), stop=stop
                )
        if not spoken:
            raise RuntimeError("there is nothing to say in that text")
        spoke = f"Spoke {frames / cli.SAMPLE_RATE:.1f} s of speech with {voice}."
        return "\n".join([spoke, *notes])

    def save(
        self,
        text: str,
        path: str,
        voice: str,
        speed: float,
        overwrite: bool,
        stop: threading.Event,
    ) -> str:
        target = Path(path).expanduser().resolve()
        cli.check_output(target)
        if target.exists() and not overwrite:
            raise RuntimeError(
                f"{target} already exists; set overwrite to true to replace it"
            )
        kokoro, lang = self.prepare(text, voice)
        notes = cli.language_notes(text, lang, voice)
        with self.turn(stop), self._model:
            spoken, phonemes = cli.to_kokoro(text, lang)
            options = {"voice": voice, "speed": speed, "lang": lang}
            samples, rate = cli.create(kokoro, spoken, phonemes, **options)
        if stop.is_set():  # the request is gone: leave no file behind
            raise streaming.Stopped
        sf.write(target, samples, rate)
        saved = f"Saved {target} ({len(samples) / rate:.1f} s of speech, {voice})."
        return "\n".join([saved, *notes])


async def run(function: Callable, *args):
    """Run a blocking call on a worker thread, and report an expected failure."""
    try:
        return await anyio.to_thread.run_sync(
            functools.partial(function, *args), abandon_on_cancel=True
        )
    except (OSError, RuntimeError) as error:
        raise ToolError(str(error)) from error


async def stoppable(function: Callable, *args):
    """`run`, with a stop signal as the last argument, set when the request ends.

    A cancelled request stops waiting for the thread at once, and the signal is what
    ends the thread. Set after a request that finished it does nothing.
    """
    stop = threading.Event()
    try:
        return await run(function, *args, stop)
    finally:
        stop.set()


def create_server(
    default_voice: str = cli.DEFAULT_VOICE,
    default_speed: float = 1.0,
    models: str | None = None,
) -> MCPServer:
    """The MCP server; a call that names no voice or speed gets the defaults."""
    engine = Engine(models)
    server = MCPServer(
        "ksay",
        version=__version__,
        description="Speak or save text with the Kokoro-82M neural voices, offline.",
    )
    # The SDK has set up logging now, and phonemizer's warnings would reach it, which
    # the command never shows: eSpeak NG spells out an acronym such as MCP, say. Only
    # propagation can be switched off, as phonemizer sets the level as each backend
    # is made, which happens when the model loads, after this.
    logging.getLogger("phonemizer").propagate = False

    def tool(**hints):
        """Register a function as a text tool, described by its docstring."""

        def register(function):
            server.add_tool(
                function,
                description=inspect.getdoc(function),
                annotations=ToolAnnotations(open_world_hint=False, **hints),
                structured_output=False,
            )
            return function

        return register

    @tool(title="Speak", read_only_hint=False, destructive_hint=False)
    async def speak(
        text: Text, voice: Voice = default_voice, speed: Speed = default_speed
    ) -> str:
        """Say text out loud through this computer's speakers, in a natural voice.

        It returns when the speech is over, which takes about as long as reading the
        text aloud. A call made meanwhile waits its turn, and cancelling a call stops
        the speech.
        """
        return await stoppable(engine.speak, text, voice, speed)

    @tool(title="Save speech to a file", read_only_hint=False, destructive_hint=True)
    async def save_speech(
        text: Text,
        path: Destination,
        voice: Voice = default_voice,
        speed: Speed = default_speed,
        overwrite: Annotated[
            bool, Field(description="Replace the file if it exists.")
        ] = False,
    ) -> str:
        """Save text as speech in an audio file, instead of playing it.

        The format follows the extension: .wav, .flac, .ogg or .mp3. A file that
        exists is left alone unless overwrite is true.
        """
        return await stoppable(engine.save, text, path, voice, speed, overwrite)

    @tool(title="List voices", read_only_hint=True, destructive_hint=False)
    async def list_voices() -> str:
        """The names of all the voices, one of which `speak` and `save_speech` take."""
        return ", ".join(await run(engine.voices))

    return server


def serve(voice: str, speed: float, models: str | None) -> None:
    """Answer MCP on standard input and output until the client closes them."""
    # As while speaking, Ctrl-C ends the process: the model cannot be interrupted
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    if cli.stdin_is_terminal():
        print("ksay: serving MCP on standard input; Ctrl-C stops it", file=sys.stderr)
    create_server(voice, speed, models).run()
