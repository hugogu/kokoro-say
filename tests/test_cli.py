import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import io
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import kokoro_say
from kokoro_say import cli


class Terminal(io.StringIO):
    """Standard input as a person at a keyboard has it."""

    def isatty(self):
        return True


def test_installs_the_command_as_ksay():
    commands = {
        entry.name: entry.value
        for entry in importlib.metadata.entry_points(group="console_scripts")
        if entry.dist.name == "kokoro-say"
    }
    assert commands == {"ksay": "kokoro_say.cli:main"}
    assert cli.build_parser().prog == "ksay"


def test_language_follows_the_voice_prefix():
    assert cli.language_for("af_heart") == "en-us"
    assert cli.language_for("bf_emma") == "en-gb"
    assert cli.language_for("zm_yunxi") == "cmn"
    assert cli.language_for("xx_unknown") == "en-us"


def test_model_dir_prefers_argument_then_environment_then_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("KOKORO_MODELS", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert cli.model_dir() == tmp_path / ".cache" / "kokoro-onnx"
    monkeypatch.setenv("KOKORO_MODELS", "/models/from-env")
    assert cli.model_dir() == Path("/models/from-env")
    assert cli.model_dir("/models/from-arg") == Path("/models/from-arg")


def test_saves_speech_to_a_file(kokoro, tmp_path, capsys):
    out = tmp_path / "hello.wav"
    assert cli.main(["Hello there.", "-v", "bf_emma", "-o", str(out)]) == 0
    assert kokoro.calls == [("Hello there.", "bf_emma", 1.0, "en-gb")]
    assert sf.info(out).duration == pytest.approx(1.0)
    assert f"saved {out} (1.00s)" in capsys.readouterr().err


@pytest.mark.parametrize("suffix", [".wav", ".flac", ".ogg", ".mp3"])
def test_saves_every_documented_format(kokoro, tmp_path, suffix):
    out = tmp_path / f"hello{suffix}"
    assert cli.main(["Hello there.", "-o", str(out)]) == 0
    assert sf.info(out).duration == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("name", ["speech.m4a", "speech"])
def test_refuses_a_file_name_that_names_no_audio_format(kokoro, tmp_path, capsys, name):
    out = tmp_path / name
    assert cli.main(["Hello there.", "-o", str(out)]) == 1
    assert capsys.readouterr().err == (
        f"ksay: cannot save {out}: end the name in .wav, .flac, .ogg or .mp3\n"
    )
    assert kokoro.calls == []  # it failed before the model was even used
    assert not out.exists()


def test_refuses_a_folder_that_does_not_exist(kokoro, tmp_path, capsys):
    out = tmp_path / "nowhere" / "speech.wav"
    assert cli.main(["Hello there.", "-o", str(out)]) == 1
    err = capsys.readouterr().err
    assert err == f"ksay: cannot save {out}: the folder {out.parent} does not exist\n"
    assert kokoro.calls == []


def test_saves_in_any_format_soundfile_writes(kokoro, tmp_path):
    out = tmp_path / "speech.aiff"  # not one of the four documented, yet it works
    assert cli.main(["Hello there.", "-o", str(out)]) == 0
    assert sf.info(out).duration == pytest.approx(1.0, abs=0.1)


def test_reads_text_from_stdin(kokoro, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Piped in."))
    assert cli.main(["-", "-o", str(tmp_path / "piped.flac")]) == 0
    assert kokoro.calls[0][0] == "Piped in."


def test_reads_piped_stdin_when_no_text_is_given(kokoro, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO("Piped in."))
    assert cli.main(["-o", str(tmp_path / "piped.flac")]) == 0
    assert kokoro.calls[0][0] == "Piped in."


@pytest.mark.parametrize("flag", ["-f", "--input-file"])
def test_reads_text_from_a_file(kokoro, tmp_path, flag):
    source = tmp_path / "chapter.txt"
    source.write_text("Once upon a time.\nThe end.\n", encoding="utf-8")
    assert cli.main([flag, str(source), "-o", str(tmp_path / "out.wav")]) == 0
    assert kokoro.calls[0][0] == "Once upon a time.\nThe end.\n"


def test_reads_non_ascii_text_from_a_file(kokoro, tmp_path):
    source = tmp_path / "chinese.txt"
    source.write_text("你好，世界。", encoding="utf-8")
    assert cli.main(["-f", str(source), "-o", str(tmp_path / "out.wav")]) == 0
    assert kokoro.calls[0][0] == "你好，世界。"


def test_reads_stdin_for_a_dash_file(kokoro, monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "stdin", io.StringIO("From stdin."))
    assert cli.main(["-f", "-", "-o", str(tmp_path / "out.wav")]) == 0
    assert kokoro.calls[0][0] == "From stdin."


def test_does_not_speak_a_byte_order_mark(kokoro, tmp_path):
    source = tmp_path / "bom.txt"
    source.write_bytes(b"\xef\xbb\xbfHello.")
    assert cli.main(["-f", str(source), "-o", str(tmp_path / "out.wav")]) == 0
    assert kokoro.calls[0][0] == "Hello."


def test_reports_a_file_it_cannot_read(kokoro, tmp_path, capsys):
    missing = tmp_path / "missing.txt"
    assert cli.main(["-f", str(missing)]) == 1
    assert capsys.readouterr().err.startswith(f"ksay: cannot read {missing}: ")
    assert kokoro.calls == []  # it failed before the model was even used


def test_reports_a_file_that_is_not_utf8(kokoro, tmp_path, capsys):
    source = tmp_path / "latin1.txt"
    source.write_bytes("café".encode("latin-1"))
    assert cli.main(["-f", str(source)]) == 1
    assert capsys.readouterr().err == f"ksay: {source} is not UTF-8 text\n"


def test_asks_for_text_instead_of_waiting_on_a_keyboard(kokoro, monkeypatch):
    monkeypatch.setattr(sys, "stdin", Terminal())
    with pytest.raises(SystemExit) as exit:
        cli.main([])
    assert exit.value.code == 2


def test_listing_voices_never_reads_stdin(kokoro, monkeypatch, capsys):
    class Unreadable(io.StringIO):
        def read(self, *args):
            raise AssertionError("listing the voices read stdin")

    monkeypatch.setattr(sys, "stdin", Unreadable())
    assert cli.main(["--list-voices"]) == 0
    assert "af_heart" in capsys.readouterr().out


@pytest.fixture
def no_server(monkeypatch):
    """A server module that fails the test if anything tries to start a server.

    Without it, a check that went missing would start a real server in the test.
    """

    def refuse(*args):
        raise AssertionError("a server was started")

    module = types.SimpleNamespace(serve=refuse, serve_http=refuse)
    monkeypatch.setitem(sys.modules, "kokoro_say.mcp_server", module)
    monkeypatch.setattr(kokoro_say, "mcp_server", module, raising=False)


@pytest.fixture
def misaki(monkeypatch):
    """The Chinese front end, replaced by one that wraps its input."""
    monkeypatch.setattr(cli.chinese, "available", lambda: True)
    monkeypatch.setattr(cli.chinese, "phonemize", lambda text: f"<{text}>")


def test_chinese_voices_speak_the_phonemes_of_the_chinese_front_end(
    kokoro, misaki, tmp_path
):
    out = tmp_path / "zh.wav"
    assert cli.main(["你好，世界。", "-v", "zf_xiaobei", "-o", str(out)]) == 0
    assert kokoro.calls == [("<你好，世界。>", "zf_xiaobei", 1.0, "cmn")]
    assert kokoro.phonemes == ["<你好，世界。>"]


def test_other_voices_are_left_to_kokoro_onnx(kokoro, misaki, tmp_path):
    assert cli.main(["Hello", "-v", "bf_emma", "-o", str(tmp_path / "en.wav")]) == 0
    assert kokoro.calls == [("Hello", "bf_emma", 1.0, "en-gb")]
    assert kokoro.phonemes == []


def test_chinese_without_misaki_is_spoken_from_the_text_and_says_so(
    kokoro, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli.chinese, "available", lambda: False)
    assert cli.main(["你好", "-v", "zf_xiaobei", "-o", str(tmp_path / "zh.wav")]) == 0
    assert kokoro.calls == [("你好", "zf_xiaobei", 1.0, "cmn")]
    assert kokoro.phonemes == []
    assert "kokoro-say[zh]" in capsys.readouterr().err


def test_chinese_text_in_an_english_voice_suggests_a_chinese_voice(
    kokoro, monkeypatch, tmp_path, capsys
):
    monkeypatch.setattr(cli.chinese, "available", lambda: False)
    assert cli.main(["你好 world", "-o", str(tmp_path / "zh.wav")]) == 0
    assert "-v zf_xiaobei" in capsys.readouterr().err


def test_english_text_gets_no_notes(kokoro, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cli.chinese, "available", lambda: False)
    assert cli.main(["Hello", "-o", str(tmp_path / "en.wav")]) == 0
    assert capsys.readouterr().err.startswith("saved ")


def test_plays_when_there_is_no_output_file(kokoro, output):
    assert cli.main(["Hello", "-s", "1.2"]) == 0
    assert kokoro.calls[0][2] == 1.2
    assert output.rate == 24_000
    assert output.calls == ["start", "stop", "close"]
    assert [len(samples) for samples in output.written] == [24_000]


def test_fails_before_synthesis_without_an_audio_device(kokoro, monkeypatch, capsys):
    def no_device(rate):
        raise RuntimeError("cannot open the audio output: no device")

    monkeypatch.setattr(cli, "open_output", no_device)
    assert cli.main(["Hello"]) == 1
    assert kokoro.calls == []  # nothing synthesized for nobody to hear
    assert capsys.readouterr().err.startswith("ksay: cannot open the audio output")


def test_streams_each_sentence_through_one_output(kokoro, output):
    text = "The first sentence is here. The second one follows it."
    assert cli.main([text, "--stream", "-v", "bf_emma"]) == 0
    assert kokoro.calls == [
        ("The first sentence is here.", "bf_emma", 1.0, "en-gb"),
        ("The second one follows it.", "bf_emma", 1.0, "en-gb"),
    ]
    assert output.rate == 24_000
    assert output.calls == ["start", "stop", "close"]
    # a sentence stop calls for 0.25 s of silence, before the next sentence only
    assert [len(block) for block in output.written] == [24_000, 24_000 + 6_000]
    assert all(block.dtype == np.float32 for block in output.written)


def test_streams_a_file_and_piped_text(kokoro, output, monkeypatch, tmp_path):
    source = tmp_path / "chapter.txt"
    source.write_text("Read from a file, sentence by sentence.", encoding="utf-8")
    assert cli.main(["-f", str(source), "--stream"]) == 0
    monkeypatch.setattr(sys, "stdin", io.StringIO("Read from a pipe, then say it."))
    assert cli.main(["--stream"]) == 0
    assert [call[0] for call in kokoro.calls] == [
        "Read from a file, sentence by sentence.",
        "Read from a pipe, then say it.",
    ]


def test_streams_chinese_sentence_by_sentence(kokoro, misaki, output, monkeypatch):
    monkeypatch.setattr(cli.chinese, "phonemize", lambda text: text.replace("。", "."))
    text = "今天我们来讨论一下这个非常重要的问题。然后大家一起来回答它并且给出想法。"
    assert cli.main([text, "--stream", "-v", "zf_xiaobei"]) == 0
    assert kokoro.phonemes == [
        "今天我们来讨论一下这个非常重要的问题.",
        "然后大家一起来回答它并且给出想法.",
    ]
    # the stop at the end of the first sentence calls for a pause, read in the phonemes
    assert [len(block) for block in output.written] == [24_000, 24_000 + 6_000]


def test_says_so_when_nothing_in_the_text_can_be_spoken(
    kokoro, output, tmp_path, capsys
):
    kokoro.unspeakable = "---"
    assert cli.main(["-o", str(tmp_path / "none.wav"), "--", "---"]) == 1
    assert capsys.readouterr().err == "ksay: there is nothing to say in that text\n"
    assert cli.main(["--", "---"]) == 1  # played, not saved
    assert capsys.readouterr().err == "ksay: there is nothing to say in that text\n"
    assert output.calls == ["start", "close"]  # the device was released all the same


def test_streaming_skips_what_cannot_be_said(kokoro, output):
    kokoro.unspeakable = "***"
    text = "The first sentence is here.\n\n***\n\nThe last sentence is here."
    assert cli.main([text, "--stream"]) == 0
    assert len(kokoro.calls) == 3  # it was tried
    assert len(output.written) == 2  # and left out


def test_streaming_nothing_is_an_error(kokoro, output, monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("  \n"))
    with pytest.raises(SystemExit) as exit:
        cli.main(["--stream"])
    assert exit.value.code == 2
    assert output.calls == ["start", "stop", "close"]  # the device was opened first


def test_the_device_is_opened_without_touching_signal_handlers(output):
    # signal.signal() works on the main thread only, and a server speaks on others
    errors = []

    def play():
        try:
            with cli.device() as device:
                device.write(np.zeros(10, dtype=np.float32))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=play)
    thread.start()
    thread.join()
    assert errors == []
    assert output.calls == ["start", "stop", "close"]


@pytest.mark.parametrize("argv", [["Hello"], ["Hello", "--stream"]])
def test_ctrl_c_ends_speech_at_once(kokoro, output, monkeypatch, argv):
    handlers = []

    def write(samples):
        handlers.append(signal.getsignal(signal.SIGINT))

    monkeypatch.setattr(output, "write", write)
    before = signal.getsignal(signal.SIGINT)
    assert cli.main(argv) == 0
    # The default action ends the process; Python's would wait for synthesis.
    assert set(handlers) == {signal.SIG_DFL}
    assert signal.getsignal(signal.SIGINT) is before


@pytest.mark.parametrize("argv", [["-v", "?"], ["--list-voices"]])
def test_lists_voices(kokoro, capsys, argv):
    assert cli.main(argv) == 0
    assert capsys.readouterr().out.split() == ["af_heart", "bf_emma", "zf_xiaobei"]


@pytest.mark.parametrize(
    "argv",
    [
        ["Hello", "-v", "nope"],  # unknown voice
        ["Hello", "-f", "notes.txt"],  # text or a file, not both
        ["Hello", "-s", "3"],  # speed out of range
        ["--voices", "bf_emma", "Hello"],  # not an option
        ["Hello", "--stream", "-o", "hello.wav"],  # plays or saves, not both
        ["--mcp", "-s", "3"],  # speed out of range, as for any other speech
    ],
)
def test_rejects_bad_arguments(kokoro, argv):
    with pytest.raises(SystemExit) as exit:
        cli.main(argv)
    assert exit.value.code == 2


@pytest.mark.parametrize(
    "argv",
    [
        ["--mcp", "Hello"],
        ["--mcp", "-f", "notes.txt"],
        ["--mcp", "-o", "hello.wav"],
        ["--mcp", "--stream"],
        ["--mcp", "-l", "en-us"],
        ["--mcp", "--list-voices"],
        ["--mcp", "-v", "?"],
    ],
)
def test_mcp_serves_requests_so_it_takes_nothing_to_speak(kokoro, no_server, argv):
    with pytest.raises(SystemExit) as exit:
        cli.main(argv)
    assert exit.value.code == 2
    assert kokoro.calls == []


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("8765", ("127.0.0.1", 8765)),
        (":8765", ("127.0.0.1", 8765)),
        ("0.0.0.0:8765", ("0.0.0.0", 8765)),
        ("192.168.1.10:80", ("192.168.1.10", 80)),
        ("mac.local:8765", ("mac.local", 8765)),
        ("[::1]:8765", ("::1", 8765)),
        ("[::]:8765", ("::", 8765)),
    ],
)
def test_parses_where_to_listen(value, expected):
    assert cli.parse_listen(value) == expected


@pytest.mark.parametrize(
    "value",
    ["", "host", "host:", "8765x", "0", "65536", "-1", "a b:80", "[::1:80", "ho_st:80"],
)
def test_rejects_an_address_it_cannot_listen_on(value):
    with pytest.raises(ValueError, match=r"\[HOST:\]PORT"):
        cli.parse_listen(value)


@pytest.mark.parametrize(
    ("host", "loopback"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("::", False),
        ("192.168.1.10", False),
        ("mac.local", False),
    ],
)
def test_knows_which_hosts_are_only_this_machine(host, loopback):
    assert cli.is_loopback(host) is loopback


def test_the_token_comes_from_a_file_before_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("KSAY_MCP_TOKEN", "e" * 20)
    token_file = tmp_path / "token"
    token_file.write_text("f" * 20 + "\n", encoding="utf-8")  # an editor adds a newline
    assert cli.read_token(str(token_file)) == "f" * 20
    assert cli.read_token(None) == "e" * 20


def test_there_is_no_token_unless_one_is_given(monkeypatch):
    monkeypatch.delenv("KSAY_MCP_TOKEN", raising=False)
    assert cli.read_token(None) is None
    monkeypatch.setenv("KSAY_MCP_TOKEN", "  ")
    assert cli.read_token(None) is None  # a blank variable is not a token


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("short", "at least 16 characters"),
        ("a token with spaces in it", "ASCII"),
        ("é" * 20, "ASCII"),
        ("\n", "is empty"),
    ],
)
def test_refuses_a_token_it_cannot_trust(tmp_path, text, message):
    token_file = tmp_path / "token"
    token_file.write_text(text, encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        cli.read_token(str(token_file))


def test_refuses_a_token_in_the_environment_that_is_too_short(monkeypatch):
    monkeypatch.setenv("KSAY_MCP_TOKEN", "short")
    with pytest.raises(RuntimeError, match="at least 16 characters"):
        cli.read_token(None)


def test_refuses_a_token_file_it_cannot_read(tmp_path):
    with pytest.raises(RuntimeError, match="cannot read .*missing"):
        cli.read_token(str(tmp_path / "missing"))


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--listen", "0.0.0.0:8765"], "needs a token"),
        (["--listen", "mac.local:8765"], "needs a token"),
        (["--listen", "[::]:8765"], "needs a token"),
        (["--listen", "nonsense"], "is not [HOST:]PORT"),
        (["--listen", "8765", "Hello"], "takes no text"),
        (["--listen", "8765", "-o", "hello.wav"], "takes no text"),
        (["--mcp", "--token-file", "token"], "goes with --listen"),
    ],
)
def test_listen_refuses_what_is_unsafe_or_meaningless(
    kokoro, no_server, monkeypatch, capsys, argv, message
):
    monkeypatch.delenv("KSAY_MCP_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exit:
        cli.main(argv)
    assert exit.value.code == 2
    assert message in capsys.readouterr().err


def test_a_token_that_is_too_weak_is_an_argument_error(
    kokoro, no_server, monkeypatch, capsys
):
    monkeypatch.setenv("KSAY_MCP_TOKEN", "short")
    with pytest.raises(SystemExit) as exit:
        cli.main(["--listen", "0.0.0.0:8765"])
    assert exit.value.code == 2
    assert "at least 16 characters" in capsys.readouterr().err


def test_mcp_without_its_extra_says_how_to_get_it(kokoro, monkeypatch, capsys):
    # What is already imported would hide the missing package: load it all afresh
    monkeypatch.setitem(sys.modules, "mcp.server", None)  # makes the import fail
    monkeypatch.delitem(sys.modules, "kokoro_say.mcp_server", raising=False)
    monkeypatch.delattr(kokoro_say, "mcp_server", raising=False)
    assert cli.main(["--mcp"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("ksay: --mcp needs the mcp extra (kokoro-say[mcp]): ")
    assert "Traceback" not in err


def test_downloads_missing_model_files_once(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"model")
    files = {
        "model.bin": cli.ModelFile(
            source.as_uri(), 5, hashlib.sha256(b"model").hexdigest()
        )
    }
    folder = cli.ensure_model(tmp_path / "models", files)
    assert (folder / "model.bin").read_bytes() == b"model"
    source.unlink()  # a second call must not download again
    cli.ensure_model(tmp_path / "models", files)


def test_rejects_a_download_that_fails_its_checksum(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupt")
    files = {"model.bin": cli.ModelFile(source.as_uri(), 7, "0" * 64)}
    with pytest.raises(RuntimeError, match="checksum"):
        cli.ensure_model(tmp_path / "models", files)
    assert list((tmp_path / "models").iterdir()) == []


def test_explains_a_missing_portaudio(monkeypatch):
    class NoPortAudio:
        def find_spec(self, name, path=None, target=None):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoPortAudio(), *sys.meta_path])
    with pytest.raises(RuntimeError, match="libportaudio2"):
        cli.open_output(24_000)


@pytest.mark.parametrize(
    ("system", "size"), [("Darwin", 512), ("Linux", 0), ("Windows", 0)]
)
def test_sizes_the_audio_buffer_for_the_system(monkeypatch, system, size):
    settings = {}

    class Stream:
        def __init__(self, **options):
            settings.update(options)

    sounddevice = types.SimpleNamespace(OutputStream=Stream, PortAudioError=Exception)
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    monkeypatch.setattr(cli.platform, "system", lambda: system)
    cli.open_output(24_000)
    assert settings["blocksize"] == size


def test_explains_a_missing_audio_device(monkeypatch):
    class PortAudioError(Exception):
        pass

    def no_device(**settings):
        raise PortAudioError("Error querying device -1")

    sounddevice = types.SimpleNamespace(
        PortAudioError=PortAudioError, OutputStream=no_device
    )
    monkeypatch.setitem(sys.modules, "sounddevice", sounddevice)
    with pytest.raises(RuntimeError, match="cannot open the audio output"):
        cli.open_output(24_000)


@pytest.mark.parametrize(
    ("system", "code", "out", "cores"),
    [
        ("Darwin", 0, "8\n", 8),  # Apple silicon
        ("Darwin", 1, "", None),  # an Intel Mac has one kind of core
        ("Linux", 0, "8\n", None),
    ],
)
def test_counts_the_performance_cores(monkeypatch, system, code, out, cores):
    def sysctl(command, **options):
        return subprocess.CompletedProcess(command, code, stdout=out)

    monkeypatch.setattr(cli.platform, "system", lambda: system)
    monkeypatch.setattr(cli.subprocess, "run", sysctl)
    assert cli.performance_cores() == cores


def test_onnxruntime_is_told_not_to_call_home_before_it_is_imported(monkeypatch):
    seen = []

    class Importing(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        """Stands in for onnxruntime, and notes what the environment says on import."""

        def find_spec(self, name, path, target=None):
            if name == "onnxruntime":
                return importlib.machinery.ModuleSpec(name, self)

        def create_module(self, spec):
            return None

        def exec_module(self, module):
            seen.append(os.environ.get("ORT_DISABLE_TELEMETRY"))
            raise ImportError("a stand-in that goes no further")

    monkeypatch.setenv("ORT_DISABLE_TELEMETRY", "")  # so that it is unset again after
    monkeypatch.delenv("ORT_DISABLE_TELEMETRY")
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    monkeypatch.setattr(cli, "shorten_espeak_path", lambda: None)
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    monkeypatch.setattr(sys, "meta_path", [Importing(), *sys.meta_path])
    with pytest.raises(ImportError, match="stand-in"):
        cli.load_kokoro(Path("models"))
    # it is not the Python function: that does not stop the connection
    assert seen == ["1"]


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need extra rights")
def test_shortens_a_long_espeak_data_path(monkeypatch, tmp_path):
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # Restored after the test, whatever shorten_espeak_path() puts there.
    monkeypatch.setattr(EspeakWrapper, "data_path", EspeakWrapper.__dict__["data_path"])
    deep = tmp_path / ("d" * 160)
    deep.mkdir()
    monkeypatch.setattr(espeakng_loader, "get_data_path", lambda: str(deep))
    cli.shorten_espeak_path()
    short = EspeakWrapper.data_path.fget(None)
    assert len(str(short)) < 150
    assert short.resolve() == deep.resolve()


def test_leaves_a_short_espeak_data_path_alone(monkeypatch):
    import espeakng_loader
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    original = EspeakWrapper.__dict__["data_path"]
    monkeypatch.setattr(espeakng_loader, "get_data_path", lambda: "/short/path")
    cli.shorten_espeak_path()
    assert EspeakWrapper.__dict__["data_path"] is original


def test_keeps_espeak_library_copies_between_runs(monkeypatch, tmp_path):
    from phonemizer.backend.espeak import api

    # Restored after the test, whatever reuse_espeak_copies() puts there.
    monkeypatch.setattr(api, "tempfile", api.tempfile)
    monkeypatch.setattr(api, "shutil", api.shutil)
    library = tmp_path / "libespeak-ng.dylib"
    library.write_bytes(b"espeak")

    def run():
        api.tempfile, api.shutil = tempfile, shutil  # each run starts unpatched
        cli.reuse_espeak_copies(tmp_path / "cache")
        folders = [api.tempfile.mkdtemp() for _ in range(2)]  # two wrappers
        for folder in folders:
            target = Path(folder) / library.name
            api.shutil.copy(library, target, follow_symlinks=False)
            api.shutil.rmtree(folder)  # as a wrapper does when it is done
        return [(Path(folder) / library.name).stat().st_ino for folder in folders]

    first = run()
    assert len(set(first)) == 2  # each wrapper loads a copy of its own
    assert run() == first  # the next run loads the very same files
    library.write_bytes(b"espeak, upgraded")
    assert run() != first  # while a changed library is copied afresh


@pytest.mark.needs_model
def test_loads_the_real_model_on_the_performance_cores():
    kokoro = cli.load_kokoro(cli.model_dir())
    threads = kokoro.sess.get_session_options().intra_op_num_threads
    assert threads == (cli.performance_cores() or 0)  # 0: onnxruntime decides


@pytest.mark.needs_model
def test_speaks_with_the_real_model(tmp_path):
    out = tmp_path / "real.wav"
    assert cli.main(["Hello from Kokoro.", "-o", str(out)]) == 0
    assert 0.5 < sf.info(out).duration < 5


@pytest.mark.needs_model
def test_streams_with_the_real_model(output):
    assert cli.main(["Hello from Kokoro. This part is streamed.", "--stream"]) == 0
    audio = np.concatenate(output.written)
    assert audio.dtype == np.float32  # what the float32 output stream accepts
    assert 1 < len(audio) / output.rate < 8


@pytest.mark.needs_model
def test_speaks_a_sentence_before_the_next_has_been_written(output, monkeypatch, pipe):
    heard = threading.Event()
    play = output.write

    def write(samples):
        play(samples)
        heard.set()

    monkeypatch.setattr(output, "write", write)
    early = []

    def produce():
        os.write(pipe, b"The first sentence is long enough to be spoken. ")
        # a process that waited for the end of its input would never get past this
        early.append(heard.wait(timeout=120))
        os.write(pipe, b"The second one comes afterwards.")
        os.close(pipe)

    producer = threading.Thread(target=produce)
    producer.start()
    assert cli.main(["--stream"]) == 0
    producer.join()
    assert early == [True]
    assert len(output.written) == 2
