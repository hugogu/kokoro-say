import sys

import pytest
import soundfile as sf

from kokoro_say import chinese, cli

needs_misaki = pytest.mark.skipif(
    not chinese.available(), reason="needs the zh extra (misaki, Python before 3.13)"
)
TONES = "→↗↓↘"


def test_finds_chinese_characters():
    assert chinese.has_chinese("你好")
    assert chinese.has_chinese("open the 设置 menu")
    assert not chinese.has_chinese("plain English, こんにちは, 안녕")


def test_is_not_available_without_misaki(monkeypatch):
    chinese.front_ends.cache_clear()
    monkeypatch.setitem(sys.modules, "misaki", None)  # makes the import fail
    try:
        assert not chinese.available()
    finally:
        chinese.front_ends.cache_clear()


@needs_misaki
def test_writes_the_tones_of_mandarin():
    phonemes = chinese.phonemize("你好")
    assert any(tone in phonemes for tone in TONES)
    assert not any(digit in phonemes for digit in "12345")


@needs_misaki
def test_turns_chinese_punctuation_into_what_kokoro_pauses_on():
    phonemes = chinese.phonemize("你好，世界。真的吗？")
    assert phonemes.count(",") == 1
    assert phonemes.count(".") == 1
    assert phonemes.count("?") == 1
    assert not any(mark in phonemes for mark in "，。？")


@needs_misaki
def test_speaks_numbers_in_chinese():
    assert "3" not in chinese.phonemize("第3名")


@needs_misaki
def test_phonemizes_english_words_inside_chinese_text():
    mixed = chinese.phonemize("请打开 Wi-Fi 设置")
    chinese_only = chinese.phonemize("请打开设置")
    assert "Wi" not in mixed  # not left as letters for the model to guess at
    assert "(en)" not in mixed and "(cmn)" not in mixed  # no language flags
    assert len(mixed) > len(chinese_only)
    assert any(tone in mixed for tone in TONES)


@needs_misaki
def test_a_curly_apostrophe_stays_inside_its_english_word():
    assert chinese.phonemize("今天 don\u2019t 来") == chinese.phonemize("今天 don't 来")


@needs_misaki
def test_leaves_nothing_for_blank_segments():
    assert chinese.phonemize("  \n ") == ""
    assert "  " not in chinese.phonemize("你好  machine   learning  世界")


@needs_misaki
@pytest.mark.needs_model
def test_speaks_chinese_and_mixed_text_with_the_real_model(tmp_path):
    plain = tmp_path / "plain.wav"
    mixed = tmp_path / "mixed.wav"
    assert cli.main(["你好，世界。", "-v", "zf_xiaobei", "-o", str(plain)]) == 0
    text = "今天我们来讨论一下 machine learning 的应用。"
    assert cli.main([text, "-v", "zm_yunxi", "-o", str(mixed)]) == 0
    assert 0.5 < sf.info(plain).duration < 6
    assert 2 < sf.info(mixed).duration < 12
