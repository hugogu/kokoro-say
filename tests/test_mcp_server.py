import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import types

import pytest
import soundfile as sf

pytest.importorskip("mcp")

import anyio  # noqa: E402
from mcp import Client, StdioServerParameters  # noqa: E402

from kokoro_say import cli, mcp_server  # noqa: E402

SENTENCE = "Save this sentence."


def call(server, tool, **arguments):
    """One call to a tool over an in-process session, and what came back."""

    async def ask():
        async with Client(server) as client:
            return await client.call_tool(tool, arguments)

    return anyio.run(ask)


def said(result) -> str:
    return "\n".join(block.text for block in result.content)


def wait_until(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


@pytest.fixture
def rescans(monkeypatch):
    """Stands in for PortAudio's restart, and counts the times it was asked for."""
    asked = []
    monkeypatch.setattr(mcp_server, "rescan_devices", lambda: asked.append(True))
    return asked


@pytest.fixture
def server(kokoro, output, rescans):
    return mcp_server.create_server()


@pytest.fixture
def held(output, monkeypatch):
    """The sound card holds each slice of speech until `release` is set."""
    writing, release = threading.Event(), threading.Event()
    play = output.write

    def write(block):
        play(block)
        writing.set()
        release.wait(10)

    monkeypatch.setattr(output, "write", write)
    yield types.SimpleNamespace(writing=writing, release=release)
    release.set()  # never leave a thread waiting, whatever the test did


def test_offers_speak_save_speech_and_list_voices(server):
    async def tools():
        async with Client(server) as client:
            return {tool.name: tool for tool in (await client.list_tools()).tools}

    offered = anyio.run(tools)
    assert set(offered) == {"speak", "save_speech", "list_voices"}
    speak = offered["speak"].input_schema
    assert speak["required"] == ["text"]
    assert speak["properties"]["voice"]["default"] == "af_heart"
    speed = speak["properties"]["speed"]
    assert (speed["minimum"], speed["maximum"], speed["default"]) == (0.5, 2.0, 1.0)
    assert offered["save_speech"].input_schema["required"] == ["text", "path"]
    assert offered["list_voices"].annotations.read_only_hint


def test_speak_says_the_text_through_the_sound_card(server, kokoro, output, rescans):
    text = "The first sentence is here. The second one follows it."
    result = call(server, "speak", text=text)
    assert not result.is_error
    assert re.fullmatch(r"Spoke \d\.\d s of speech with af_heart\.", said(result))
    assert kokoro.calls == [
        ("The first sentence is here.", "af_heart", 1.0, "en-us"),
        ("The second one follows it.", "af_heart", 1.0, "en-us"),
    ]
    assert output.calls == ["start", "stop", "close"]
    assert output.rate == 24_000
    # a second for each sentence, and the pause that the first one's stop calls for
    assert sum(len(block) for block in output.written) == 2 * 24_000 + 6_000
    assert rescans == [True]  # the audio devices were looked at again


def test_a_call_that_names_no_voice_or_speed_gets_the_servers(kokoro, output, rescans):
    server = mcp_server.create_server("bf_emma", 1.2)
    call(server, "speak", text="Spoken in the voice of the server.")
    call(server, "speak", text="Spoken as asked.", voice="af_heart", speed=0.8)
    assert [made[1:] for made in kokoro.calls] == [
        ("bf_emma", 1.2, "en-gb"),
        ("af_heart", 0.8, "en-us"),
    ]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            {"text": "Hello there.", "voice": "nope"},
            "unknown voice 'nope'; list them with the list_voices tool",
        ),
        ({"text": "Hello there.", "speed": 3}, "less than or equal to 2"),
        ({"text": "  \n"}, "there is no text to speak"),
    ],
)
def test_speak_refuses_what_it_cannot_say(server, kokoro, output, arguments, message):
    result = call(server, "speak", **arguments)
    assert result.is_error
    assert message in said(result)
    assert kokoro.calls == []
    assert output.calls == []  # the sound card was not even opened


def test_speak_says_so_when_nothing_in_the_text_can_be_said(server, kokoro, output):
    kokoro.unspeakable = "***"
    result = call(server, "speak", text="***")
    assert result.is_error
    assert "there is nothing to say in that text" in said(result)
    assert output.calls[-1] == "close"  # the sound card was let go all the same


def test_speak_reports_a_missing_sound_card(server, kokoro, monkeypatch):
    def no_device(rate):
        raise RuntimeError("cannot open the audio output: no device")

    monkeypatch.setattr(cli, "open_output", no_device)
    result = call(server, "speak", text="Hello there.")
    assert result.is_error
    assert "cannot open the audio output: no device" in said(result)
    assert kokoro.calls == []  # nothing was synthesized for nobody to hear


def test_speak_passes_on_what_the_voice_cannot_read(server):
    result = call(server, "speak", text="你好 world, how are you today")
    assert not result.is_error  # it speaks, and the caller is told how to do better
    assert "-v zf_xiaobei" in said(result)


def test_list_voices_names_every_voice(server):
    assert said(call(server, "list_voices")) == "af_heart, bf_emma, zf_xiaobei"


def test_save_speech_writes_the_file(server, kokoro, output, tmp_path):
    target = tmp_path / "speech.flac"
    result = call(server, "save_speech", text=SENTENCE, path=str(target))
    assert not result.is_error
    assert said(result) == f"Saved {target.resolve()} (1.0 s of speech, af_heart)."
    assert sf.info(target).duration == pytest.approx(1.0)
    assert kokoro.calls == [(SENTENCE, "af_heart", 1.0, "en-us")]
    assert output.calls == []  # saved, not played


def test_save_speech_leaves_an_existing_file_alone_unless_told_to(server, tmp_path):
    target = tmp_path / "speech.wav"
    target.write_bytes(b"precious")
    result = call(server, "save_speech", text=SENTENCE, path=str(target))
    assert result.is_error
    assert "already exists" in said(result)
    assert target.read_bytes() == b"precious"
    result = call(
        server, "save_speech", text=SENTENCE, path=str(target), overwrite=True
    )
    assert not result.is_error
    assert sf.info(target).duration == pytest.approx(1.0)


@pytest.mark.parametrize("name", ["speech.m4a", "speech", "nowhere/speech.wav"])
def test_save_speech_refuses_a_name_it_cannot_write(server, kokoro, tmp_path, name):
    result = call(server, "save_speech", text=SENTENCE, path=str(tmp_path / name))
    assert result.is_error
    assert "cannot save" in said(result)
    assert kokoro.calls == []


def test_save_speech_expands_the_home_folder(server, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # where Windows looks
    result = call(server, "save_speech", text=SENTENCE, path="~/speech.wav")
    assert not result.is_error
    assert (tmp_path / "speech.wav").exists()


def test_save_speech_says_where_a_relative_path_went(server, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = call(server, "save_speech", text=SENTENCE, path="speech.wav")
    assert str(tmp_path.resolve() / "speech.wav") in said(result)


def test_cancelling_a_save_leaves_no_file_behind(server, kokoro, tmp_path, monkeypatch):
    synthesizing, release = threading.Event(), threading.Event()
    create = kokoro.create

    def slow(*args, **options):
        synthesizing.set()
        release.wait(10)
        return create(*args, **options)

    monkeypatch.setattr(kokoro, "create", slow)
    target = tmp_path / "speech.wav"
    arguments = {"text": SENTENCE, "path": str(target)}

    async def scenario():
        async with Client(server) as client:
            async with anyio.create_task_group() as group:
                group.start_soon(client.call_tool, "save_speech", arguments)
                await anyio.to_thread.run_sync(synthesizing.wait, 10)
                group.cancel_scope.cancel()

    try:
        anyio.run(scenario)
    finally:
        release.set()
    # this has to wait for the turn that the cancelled call gave up
    second = tmp_path / "second.wav"
    assert not call(server, "save_speech", text=SENTENCE, path=str(second)).is_error
    assert second.exists()
    assert not target.exists()


def test_speeches_take_turns_on_the_sound_card(server, output, held):
    async def scenario():
        async with Client(server) as client:
            async with anyio.create_task_group() as group:
                group.start_soon(client.call_tool, "speak", {"text": "First, a while."})
                await anyio.to_thread.run_sync(held.writing.wait, 10)
                group.start_soon(client.call_tool, "speak", {"text": "Second, later."})
                await anyio.sleep(0.4)  # the second has had every chance to start
                assert output.calls == ["start"]  # and the card is still the first's
                held.release.set()

    anyio.run(scenario)
    assert output.calls == ["start", "stop", "close"] * 2


def test_cancelling_a_call_stops_the_speech_and_frees_the_sound_card(
    server, output, held
):
    text = "This is a sentence that is long enough to be one of many. " * 20

    async def scenario():
        async with Client(server) as client:
            async with anyio.create_task_group() as group:
                group.start_soon(client.call_tool, "speak", {"text": text})
                await anyio.to_thread.run_sync(held.writing.wait, 10)
                group.cancel_scope.cancel()

    anyio.run(scenario)
    played = len(output.written)
    held.release.set()  # the slice that was playing is over
    wait_until(lambda: output.calls == ["start", "close"])  # not played out: no stop
    assert len(output.written) == played  # and not one slice after it
    assert not call(server, "speak", text="Now say this instead.").is_error


def test_a_call_cancelled_while_it_waits_never_speaks(server, kokoro, output, held):
    async def scenario():
        async with Client(server) as client:
            async with anyio.create_task_group() as group:
                group.start_soon(client.call_tool, "speak", {"text": "The first one."})
                await anyio.to_thread.run_sync(held.writing.wait, 10)
                async with anyio.create_task_group() as second:
                    second.start_soon(
                        client.call_tool, "speak", {"text": "The second never comes."}
                    )
                    await anyio.sleep(0.3)  # it is waiting for its turn
                    second.cancel_scope.cancel()
                await anyio.sleep(0.3)  # long enough for it to have noticed
                held.release.set()

    anyio.run(scenario)
    assert output.calls == ["start", "stop", "close"]  # one speech, the first
    assert [made[0] for made in kokoro.calls] == ["The first one."]


def test_looks_at_the_audio_devices_again_by_restarting_portaudio(monkeypatch):
    steps = []
    sounddevice = types.SimpleNamespace(
        _terminate=lambda: steps.append("terminate"),
        _initialize=lambda: steps.append("initialize"),
        PortAudioError=Exception,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    mcp_server.rescan_devices()
    assert steps == ["terminate", "initialize"]


def test_says_so_when_portaudio_will_not_restart(monkeypatch):
    class PortAudioError(Exception):
        pass

    def initialize():
        raise PortAudioError("no host API")

    sounddevice = types.SimpleNamespace(
        _terminate=lambda: None, _initialize=initialize, PortAudioError=PortAudioError
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    with pytest.raises(RuntimeError, match="cannot open the audio output: no host API"):
        mcp_server.rescan_devices()


def test_goes_without_a_restart_that_sounddevice_no_longer_offers(monkeypatch):
    sounddevice = types.SimpleNamespace(PortAudioError=Exception)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    mcp_server.rescan_devices()  # the private functions are gone: nothing to do


def test_leaves_a_missing_portaudio_to_the_device_to_explain(monkeypatch):
    class NoPortAudio:
        def find_spec(self, name, path=None, target=None):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoPortAudio(), *sys.meta_path])
    mcp_server.rescan_devices()  # no error here: opening the device says what is wrong


def test_phonemizer_warnings_stay_out_of_the_log():
    mcp_server.create_server()
    assert not logging.getLogger("phonemizer").isEnabledFor(logging.WARNING)


def test_ksay_mcp_serves_with_the_options_it_was_given(monkeypatch):
    served = []
    monkeypatch.setattr(mcp_server, "serve", lambda *args: served.append(args))
    argv = ["--mcp", "-v", "bf_emma", "-s", "1.2", "--model-dir", "/models"]
    assert cli.main(argv) == 0
    assert served == [("bf_emma", 1.2, "/models")]


def test_serving_gives_ctrl_c_its_default_action(monkeypatch, capsys):
    seen = []

    class Server:
        def run(self):
            seen.append(signal.getsignal(signal.SIGINT))

    monkeypatch.setattr(mcp_server, "create_server", lambda *args: Server())
    monkeypatch.setattr(cli, "stdin_is_terminal", lambda: True)
    before = signal.getsignal(signal.SIGINT)
    try:
        mcp_server.serve("af_heart", 1.0, None)
    finally:
        signal.signal(signal.SIGINT, before)
    assert seen == [signal.SIG_DFL]  # onnxruntime cannot be interrupted any other way
    assert "Ctrl-C" in capsys.readouterr().err  # someone at a keyboard is told

    monkeypatch.setattr(cli, "stdin_is_terminal", lambda: False)
    try:
        mcp_server.serve("af_heart", 1.0, None)
    finally:
        signal.signal(signal.SIGINT, before)
    assert capsys.readouterr().err == ""  # a client that launched it hears nothing


INITIALIZE = (
    b'{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": '
    b'{"protocolVersion": "2025-06-18", "capabilities": {}, '
    b'"clientInfo": {"name": "test", "version": "0"}}}\n'
)


@pytest.fixture
def running():
    """A `ksay --mcp` process that has answered its first request."""
    process = subprocess.Popen(
        [sys.executable, "-m", "kokoro_say", "--mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        process.stdin.write(INITIALIZE)
        process.stdin.flush()
        assert b'"ksay"' in process.stdout.readline()  # it named itself
        yield process
    finally:
        process.kill()
        process.wait()
        process.stdin.close()
        process.stdout.close()


def test_ends_when_the_client_closes_its_end(running):
    running.stdin.close()
    assert running.wait(timeout=30) == 0


@pytest.mark.skipif(sys.platform == "win32", reason="Windows sends no SIGINT")
def test_ctrl_c_ends_the_server_as_it_ends_speech(running):
    running.send_signal(signal.SIGINT)
    assert running.wait(timeout=30) == -signal.SIGINT  # it died of it, no traceback


@pytest.mark.needs_model
def test_serves_over_standard_input_and_output_with_the_real_model(tmp_path):
    process = StdioServerParameters(
        command=sys.executable,
        args=["-m", "kokoro_say", "--mcp"],
        env=dict(os.environ),
    )
    target = tmp_path / "real.wav"

    async def session():
        async with Client(process) as client:
            tools = {tool.name for tool in (await client.list_tools()).tools}
            voices = await client.call_tool("list_voices", {})
            arguments = {"text": "Hello from the server.", "path": str(target)}
            return tools, voices, await client.call_tool("save_speech", arguments)

    tools, voices, saved = anyio.run(session)
    assert tools == {"speak", "save_speech", "list_voices"}
    assert "af_heart" in said(voices)
    assert not saved.is_error
    assert 0.5 < sf.info(target).duration < 5
