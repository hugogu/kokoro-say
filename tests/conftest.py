import os
import sys

import pytest


@pytest.fixture
def pipe(monkeypatch):
    """A real pipe as standard input; returns the end to write to."""
    read_end, write_end = os.pipe()
    stdin = os.fdopen(read_end, encoding="utf-8")
    monkeypatch.setattr(sys, "stdin", stdin)
    yield write_end
    stdin.close()
