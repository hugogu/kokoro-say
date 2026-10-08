import io
import os
import sys
import threading
import time

import pytest

from kokoro_say.streaming import (
    MAXIMUM,
    SLICE,
    Sentences,
    Stopped,
    TextStream,
    speak,
    stdin_pieces,
)


def sentences(text):
    """What Sentences makes of text that arrives in one piece."""
    chunker = Sentences()
    found = chunker.feed(text)
    return found + ([tail] if (tail := chunker.flush()) else [])


def test_a_stop_waits_for_what_follows_it():
    chunker = Sentences()
    assert chunker.feed("The first sentence is here.") == []  # could be "here.5"
    assert chunker.feed(" And") == ["The first sentence is here."]
    assert chunker.flush() == "And"
    assert chunker.flush() is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Dr. Smith went to St. Louis yesterday. He came back home.",
            ["Dr. Smith went to St. Louis yesterday.", "He came back home."],
        ),
        (
            "Officials from the U.S. Army said so today. Then came more.",
            ["Officials from the U.S. Army said so today.", "Then came more."],
        ),
        (
            "The author J. K. Rowling wrote many books. Then she stopped.",
            ["The author J. K. Rowling wrote many books.", "Then she stopped."],
        ),
        (
            "Pi is about 3.14159 and e is 2.71828. Next one here.",
            ["Pi is about 3.14159 and e is 2.71828.", "Next one here."],
        ),
        (
            "Yes. No. Maybe so, then. Ok then, fine.",
            ["Yes. No. Maybe so, then.", "Ok then, fine."],
        ),
        (
            'He said, "Stop right there." Then he left the room.',
            ['He said, "Stop right there."', "Then he left the room."],
        ),
        (
            "Really?! Yes, absolutely. Why not, though?",
            ["Really?! Yes, absolutely.", "Why not, though?"],
        ),
        (
            "Visit example.com for more. It is free of charge.",
            ["Visit example.com for more.", "It is free of charge."],
        ),
    ],
)
def test_ends_sentences_where_a_reader_would(text, expected):
    assert sentences(text) == expected


def test_a_blank_line_ends_a_sentence_however_short():
    assert sentences("Chapter One\n\nIt was a dark and stormy night.") == [
        "Chapter One",
        "It was a dark and stormy night.",
    ]


def test_lines_that_merely_wrap_do_not_end_a_sentence():
    text = "The quick brown fox jumps\nover the lazy dog. Then it ran away.\n"
    assert sentences(text) == [
        "The quick brown fox jumps over the lazy dog.",
        "Then it ran away.",
    ]


def test_windows_line_endings_are_white_space_too():
    assert sentences("First line of text.\r\n\r\nSecond paragraph here.\r\n") == [
        "First line of text.",
        "Second paragraph here.",
    ]


@pytest.mark.parametrize(
    "text",
    ["1. First item\n2. Second item\n3. Third item\n", "- one\n- two\n- three\n"],
)
def test_each_list_item_is_spoken_alone(text):
    items = text.split("\n")[:-1]
    assert sentences("Here is the list:\n" + text) == ["Here is the list:", *items]


def test_chinese_needs_no_space_after_a_stop():
    assert sentences("你好，世界。我今天很高兴见到你。再见了朋友们。") == [
        "你好，世界。我今天很高兴见到你。",
        "再见了朋友们。",
    ]


def test_cuts_a_run_without_stops_at_a_space():
    text = "word " * 200
    found = sentences(text)
    assert len(found) > 1
    assert all(len(sentence) <= MAXIMUM for sentence in found)
    assert " ".join(found) == " ".join(text.split())


def test_the_pieces_text_arrives_in_do_not_change_its_sentences():
    text = (
        "Dr. Smith went to St. Louis yesterday. He came back with 3.5 liters!\n\n"
        "A new paragraph starts here and goes on for a while.\n"
        "1. First item\n2. Second item\n" + "word " * 100 + "The end of it all.\n"
    )
    bulk = sentences(text)
    chunker, found = Sentences(), []
    for character in text:
        found += chunker.feed(character)
    if tail := chunker.flush():
        found.append(tail)
    assert found == bulk


def test_gives_a_sentence_before_the_text_has_ended():
    release = threading.Event()

    def source():
        yield "The first sentence is here. "
        release.wait(5)
        yield "The second one is here too."

    stream = iter(TextStream(source(), idle=5))
    assert next(stream) == "The first sentence is here."  # the source is still waiting
    release.set()
    assert list(stream) == ["The second one is here too."]


def test_speaks_an_unfinished_sentence_after_a_pause_in_the_text():
    wait = threading.Event()

    def source():
        yield "Build finished without any errors"  # no stop ever comes
        wait.wait(5)

    stream = iter(TextStream(source(), idle=0.05))
    assert next(stream) == "Build finished without any errors"
    wait.set()
    assert list(stream) == []


def test_reports_a_source_that_fails():
    def source():
        yield "Some text that goes on and on. "
        raise OSError("the pipe broke")

    with pytest.raises(OSError, match="the pipe broke"):
        list(TextStream(source(), idle=0.05))


def test_reads_stdin_in_the_pieces_it_arrives_in(pipe):
    pieces = stdin_pieces()
    os.write(pipe, b"first piece ")
    assert next(pieces) == "first piece "  # before the writer is done
    os.write(pipe, "你".encode()[:2])  # half a character
    os.write(pipe, "你".encode()[2:] + b" and the tail")
    os.close(pipe)
    assert "".join(pieces) == "你 and the tail"


def test_stdin_that_is_not_utf8_is_not_fatal(pipe):
    os.write(pipe, b"caf\xe9 au lait")
    os.close(pipe)
    assert "".join(stdin_pieces()) == "caf� au lait"


def test_reads_stdin_that_has_no_file_descriptor(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO("all at once"))
    assert list(stdin_pieces()) == ["all at once"]


class Recorder:
    def __init__(self):
        self.written = []

    def write(self, samples):
        self.written.append(samples)


def test_plays_sentences_in_order_and_counts_them():
    output = Recorder()
    assert speak(output, str.upper, ["a", "b", "c"]) == 3
    assert output.written == ["A", "B", "C"]


def test_leaves_out_what_there_is_nothing_to_say_for():
    output = Recorder()
    assert speak(output, lambda s: None if s == "b" else s, ["a", "b", "c"]) == 2
    assert output.written == ["a", "c"]


def test_synthesizes_the_next_sentence_while_one_is_playing():
    second_started = threading.Event()

    def synthesize(sentence):
        if sentence == "b":
            second_started.set()
        return sentence

    class Output(Recorder):
        def write(self, samples):
            if not self.written:  # the first sentence is playing
                assert second_started.wait(5), "the next sentence was not started"
            super().write(samples)

    output = Output()
    assert speak(output, synthesize, ["a", "b"]) == 2
    assert output.written == ["a", "b"]


def test_does_not_run_far_ahead_of_playback():
    synthesized = []
    four = threading.Event()

    class Output(Recorder):
        def write(self, samples):
            if not self.written:  # the first sentence is playing
                four.wait(5)
                time.sleep(0.2)  # long enough for a runaway synthesis to show
                # two sentences queued, and the fourth held, waiting for room
                assert len(synthesized) == 4
            super().write(samples)

    def synthesize(sentence):
        synthesized.append(sentence)
        if len(synthesized) == 4:
            four.set()
        return sentence

    assert speak(Output(), synthesize, list("abcdefgh"), ahead=2) == 8


def test_stops_synthesizing_once_playback_has_failed():
    threads, synthesized = [], []

    def synthesize(sentence):
        threads.append(threading.current_thread())
        synthesized.append(sentence)
        return sentence

    class Failing:
        def write(self, samples):
            raise OSError("the sound card went away")

    with pytest.raises(OSError, match="sound card"):
        speak(Failing(), synthesize, list("abcdefghij"))
    threads[0].join(timeout=2)
    # a process that goes on speaking cannot keep a thread waiting for room for ever
    assert not threads[0].is_alive()
    assert len(synthesized) <= 4  # one in the player's hands, two queued, one held


def test_speech_that_can_be_stopped_is_written_in_slices():
    output = Recorder()
    stop = threading.Event()
    assert speak(output, lambda s: [0] * (2 * SLICE + 5), ["a"], stop=stop) == 1
    assert [len(block) for block in output.written] == [SLICE, SLICE, 5]


def test_speech_that_cannot_be_stopped_is_written_whole():
    output = Recorder()
    assert speak(output, lambda s: [0] * (2 * SLICE + 5), ["a"]) == 1
    assert [len(block) for block in output.written] == [2 * SLICE + 5]


def test_a_stop_ends_speech_within_a_slice():
    stop = threading.Event()

    class Output(Recorder):
        def write(self, samples):
            super().write(samples)
            stop.set()  # silence is asked for as the first slice plays

    output = Output()
    with pytest.raises(Stopped):
        speak(output, lambda s: [0] * (3 * SLICE), ["a", "b"], stop=stop)
    assert [len(block) for block in output.written] == [SLICE]


def test_a_stop_before_the_first_sound_writes_nothing():
    stop = threading.Event()
    stop.set()
    output = Recorder()
    with pytest.raises(Stopped):
        speak(output, str.upper, ["a"], stop=stop)
    assert output.written == []


def test_a_stop_does_not_wait_for_a_sentence_still_being_synthesized():
    stop, release, threads = threading.Event(), threading.Event(), []

    def synthesize(sentence):
        threads.append(threading.current_thread())
        release.wait(10)  # the model is busy with a long sentence
        return sentence

    threading.Timer(0.1, stop.set).start()
    started = time.monotonic()
    with pytest.raises(Stopped):
        speak(Recorder(), synthesize, ["a"], stop=stop)
    assert time.monotonic() - started < 3
    release.set()  # the sentence is finished, and nothing comes of it
    threads[0].join(timeout=2)
    assert not threads[0].is_alive()


def test_reports_a_synthesis_that_fails():
    def synthesize(sentence):
        raise RuntimeError("the model broke")

    with pytest.raises(RuntimeError, match="the model broke"):
        speak(Recorder(), synthesize, ["a"])
