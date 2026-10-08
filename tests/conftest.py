import os
import sys

import numpy as np
import pytest

from kokoro_say import cli


def pytest_configure(config):
    """Keep onnxruntime from calling home in the test process as well.

    Tests with a fake model still import kokoro_onnx, and with it onnxruntime, without
    going through load_kokoro, which is what switches its telemetry off. Left alone,
    that client connects to Microsoft a few seconds in, and can abort the process as it
    exits, with exit status 134, after every test has passed.
    """
    os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")


def pytest_collection_modifyitems(items):
    """Skip the tests marked needs_model when the model is not on this machine."""
    if (cli.model_dir() / "voices-v1.0.bin").exists():
        return
    skip = pytest.mark.skip(reason="needs the Kokoro model files locally")
    for item in items:
        if "needs_model" in item.keywords:
            item.add_marker(skip)


class FakeKokoro:
    def __init__(self):
        self.calls = []
        self.phonemes = []  # what was given as phonemes rather than as text
        self.unspeakable = ""  # text that has no phonemes, as "***" has

    def get_voices(self):
        return ["af_heart", "bf_emma", "zf_xiaobei"]

    def create(self, text, voice, speed, lang, is_phonemes=False):
        self.calls.append((text, voice, speed, lang))
        if is_phonemes:
            self.phonemes.append(text)
        if self.unspeakable and self.unspeakable in text:
            raise ValueError(f"Nothing to synthesize, {text!r} produced no phonemes")
        return np.zeros(24_000, dtype=np.float32), 24_000


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
    """A model that says nothing, in place of the real one; records what it is asked."""
    fake = FakeKokoro()
    monkeypatch.setattr(cli, "ensure_model", lambda folder: folder)
    monkeypatch.setattr(cli, "load_kokoro", lambda folder: fake)
    monkeypatch.setenv("KOKORO_MODELS", str(tmp_path / "models"))
    return fake


@pytest.fixture
def output(monkeypatch):
    """A sound card that plays nothing, in place of the real one."""
    fake = FakeOutput()

    def open_output(rate):
        fake.rate = rate
        return fake

    monkeypatch.setattr(cli, "open_output", open_output)
    return fake


@pytest.fixture
def pipe(monkeypatch):
    """A real pipe as standard input; returns the end to write to."""
    read_end, write_end = os.pipe()
    stdin = os.fdopen(read_end, encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin)
    yield write_end
    stdin.close()
