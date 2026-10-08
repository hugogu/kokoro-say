import hashlib
import importlib.metadata
import io
import shutil
import signal
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from kokoro_say import cli

STREAM_PARTS = [
    np.linspace(-1, 1, 5_000, dtype=np.float32),
    np.ones(300, dtype=np.float32),
]

needs_model = pytest.mark.skipif(
    not (cli.model_dir() / "voices-v1.0.bin").exists(),
    reason="needs the Kokoro model files locally",
)


class FakeKokoro:
    def __init__(self):
        self.calls = []

    def get_voices(self):
        return ["af_heart", "bf_emma", "zf_xiaobei"]

    def create(self, text, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        return np.zeros(24_000, dtype=np.float32), 24_000

    async def create_stream(self, text, voice, speed, lang):
        self.calls.append((text, voice, speed, lang))
        for part in STREAM_PARTS:
            yield part, 24_000


class Terminal(io.StringIO):
    """Standard input as a person at a keyboard has it."""

    def isatty(self):
        return True


class FakeOutput:
    def __init__(self):
        self.calls = []
        self.written = []

    def start(self):
        self.calls.append("start")

    def write(self, block):
        self.written.append(block)

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


@pytest.fixture
def kokoro(monkeypatch, tmp_path):
    fake = FakeKokoro()
    monkeypatch.setattr(cli, "ensure_model", lambda folder: folder)
    monkeypatch.setattr(cli, "load_kokoro", lambda folder: fake)
    monkeypatch.setenv("KOKORO_MODELS", str(tmp_path / "models"))
    return fake


@pytest.fixture
def output(monkeypatch):
    fake = FakeOutput()

    def open_output(rate):
        fake.rate = rate
        return fake

    monkeypatch.setattr(cli, "open_output", open_output)
    return fake


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


def test_streams_the_parts_through_one_output(kokoro, output):
    assert cli.main(["Hello", "--stream", "-v", "bf_emma"]) == 0
    assert kokoro.calls == [("Hello", "bf_emma", 1.0, "en-gb")]
    assert output.rate == 24_000
    assert output.calls == ["start", "stop", "close"]
    assert len(output.written) == len(STREAM_PARTS)
    for written, part in zip(output.written, STREAM_PARTS, strict=True):
        np.testing.assert_array_equal(written, part)


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
    ],
)
def test_rejects_bad_arguments(kokoro, argv):
    with pytest.raises(SystemExit) as exit:
        cli.main(argv)
    assert exit.value.code == 2


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


@needs_model
def test_loads_the_real_model_on_the_performance_cores():
    kokoro = cli.load_kokoro(cli.model_dir())
    threads = kokoro.sess.get_session_options().intra_op_num_threads
    assert threads == (cli.performance_cores() or 0)  # 0: onnxruntime decides


@needs_model
def test_speaks_with_the_real_model(tmp_path):
    out = tmp_path / "real.wav"
    assert cli.main(["Hello from Kokoro.", "-o", str(out)]) == 0
    assert 0.5 < sf.info(out).duration < 5


@needs_model
def test_streams_with_the_real_model(output):
    assert cli.main(["Hello from Kokoro. This part is streamed.", "--stream"]) == 0
    audio = np.concatenate(output.written)
    assert audio.dtype == np.float32  # what the float32 output stream accepts
    assert 1 < len(audio) / output.rate < 8
