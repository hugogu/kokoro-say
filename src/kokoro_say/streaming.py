"""Speak text while it is still arriving.

A pipe can deliver a sentence a second, as a language model writes one, or a
log as it grows. `say` waits for the end of its input before it speaks anything.
Here the input is cut into sentences as it comes, and synthesis runs ahead of
playback, so the first sentence is heard while the rest is still being written.
"""

from __future__ import annotations

import codecs
import os
import queue
import re
import sys
import threading
from collections.abc import Callable, Iterable, Iterator

MINIMUM = 15  # characters; shorter sentences ("Dr.", "Yes.") join the next one
MAXIMUM = 300  # characters; text with no stop is cut at a space this far in
IDLE = 1.0  # seconds of silence after which an unfinished sentence is spoken

BOUNDARY = re.compile(
    r"""
      (?P<paragraph>\n[ \t\r]*\n)
    | (?P<item>\n(?=[ \t]*(?:\d{1,3}[.)]|[-*•])[ \t]))
    | (?P<sentence>[.!?…]+["'”’)\]]*(?=\s)|[。！？]+["”’）」』]*)
    """,
    re.VERBOSE,
)
# What ends before a stop that is not the end of a sentence: a title ("Dr"),
# an initial ("J"), or letters set off by stops ("U.S", "e.g")
ABBREVIATION = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|St|Mt|Jr|Sr|vs|Gen|Col|Lt|Capt|Sgt|Rev|Hon|[A-Z]"
    r"|(?:[A-Za-z]\.)+[A-Za-z])$"
)


def squeeze(text: str) -> str:
    """The text with every run of white space, newlines included, made one space."""
    return " ".join(text.split())


class Sentences:
    """Cuts text that arrives in pieces into sentences, each as soon as it is whole.

    A stop ends a sentence only when white space follows it, so "3.14" and
    "example.com" stay whole and the last stop of a piece waits for what comes
    next. Chinese and Japanese stops need no space. A blank line, or a line
    that starts a list item, ends one too. Lines that merely wrap do not.
    A sentence under `minimum` characters ("Dr.", "Yes.") joins the next one,
    and one with no stop for `maximum` characters is cut at a space.
    """

    def __init__(self, minimum: int = MINIMUM, maximum: int = MAXIMUM):
        self.minimum = minimum
        self.maximum = maximum
        self.pending = ""

    def feed(self, text: str) -> list[str]:
        """Add text; return the sentences it completed."""
        self.pending += text
        found, start = [], 0
        for match in BOUNDARY.finditer(self.pending):
            if match.lastgroup == "sentence":
                piece = self.pending[start : match.end()]
                if len(piece.strip()) < self.minimum:
                    continue  # a short answer or abbreviation: keep going
                if match[0].startswith(".") and ABBREVIATION.search(
                    self.pending[start : match.start()]
                ):
                    continue  # "Dr." or "J." or "U.S." is not the end
            else:  # a blank line or a list item ends the sentence before it
                piece = self.pending[start : match.start()]
            found += self._parts(piece)
            start = match.end()
        *full, self.pending = self._cut(self.pending[start:])  # the last may grow
        return found + [squeeze(part) for part in full if part.strip()]

    def flush(self) -> str | None:
        """Whatever has not ended in a stop, once there is nothing more to wait for."""
        tail, self.pending = squeeze(self.pending), ""
        return tail or None

    def _parts(self, piece: str) -> list[str]:
        return [squeeze(part) for part in self._cut(piece) if part.strip()]

    def _cut(self, text: str) -> Iterator[str]:
        """The text in parts of at most `maximum` characters, cut at spaces."""
        start = 0
        while len(text) - start > self.maximum:
            space = text.rfind(" ", start, start + self.maximum)
            end = space if space > start else start + self.maximum
            yield text[start:end]
            start = end
        yield text[start:]


def stdin_pieces() -> Iterator[str]:
    """Standard input in the pieces it arrives in, not once it has all arrived."""
    try:
        descriptor = sys.stdin.fileno()
    except (AttributeError, ValueError, OSError):  # not a real file, as in a test
        yield sys.stdin.read()
        return
    decoder = codecs.getincrementaldecoder(sys.stdin.encoding or "utf-8")("replace")
    while block := os.read(descriptor, 1 << 16):
        if text := decoder.decode(block):
            yield text
    if text := decoder.decode(b"", final=True):
        yield text


class TextStream:
    """Sentences from text that is still arriving.

    Reading starts at once, on a thread of its own, so text that arrives while
    the model is loading is not lost and a silent source cannot block the
    sentences already read. A sentence left unfinished for `idle` seconds is
    spoken as it is, and the last one is spoken when the text ends.
    """

    def __init__(self, pieces: Iterable[str], idle: float = IDLE):
        self.idle = idle
        self._arrived: queue.Queue = queue.Queue()
        threading.Thread(target=self._read, args=(pieces,), daemon=True).start()

    def _read(self, pieces: Iterable[str]) -> None:
        try:
            for piece in pieces:
                self._arrived.put(piece)
        except BaseException as error:  # reported where the sentences are consumed
            self._arrived.put(error)
        else:
            self._arrived.put(None)

    def __iter__(self) -> Iterator[str]:
        sentences = Sentences()
        while True:
            waiting = bool(sentences.pending.strip())
            try:
                piece = self._arrived.get(timeout=self.idle if waiting else None)
            except queue.Empty:  # the text stopped part-way through a sentence
                if tail := sentences.flush():
                    yield tail
                continue
            if isinstance(piece, BaseException):
                raise piece
            if piece is None:
                break
            yield from sentences.feed(piece)
        if tail := sentences.flush():
            yield tail


def speak(
    output,
    synthesize: Callable[[str], object | None],
    sentences: Iterable[str],
    ahead: int = 2,
) -> int:
    """Play the speech of each sentence while the next ones are synthesized.

    `synthesize` returns the samples for one sentence, or None when there is
    nothing in it to say. Synthesis runs on its own thread, up to `ahead`
    sentences in front of playback, so sentences join without a gap whenever
    synthesis is quicker than speech. Returns how many sentences were spoken.
    """
    ready: queue.Queue = queue.Queue(maxsize=ahead)

    def produce() -> None:
        try:
            for sentence in sentences:
                if (samples := synthesize(sentence)) is not None:
                    ready.put(samples)
        except BaseException as error:  # reported by the thread that plays
            ready.put(error)
        else:
            ready.put(None)

    threading.Thread(target=produce, daemon=True).start()
    spoken = 0
    while (item := ready.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        output.write(item)
        spoken += 1
    return spoken
