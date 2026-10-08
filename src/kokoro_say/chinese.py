"""Chinese text for the Kokoro voices, through Kokoro's own front end.

kokoro-onnx turns text into phonemes with eSpeak NG, which is the wrong tool for
Mandarin: eSpeak's tone marks are not in Kokoro's vocabulary, so the voice speaks
toneless syllables, and Chinese punctuation is lost. `misaki` is the front end the
Chinese voices were trained with. It writes the tones as arrows and speaks numbers
as Chinese. The English words of mixed text go to eSpeak NG and are spoken by the
same voice, with an accent, as a Chinese speaker would say them.

misaki is the optional `zh` extra, because it adds about 100 MB and declares
support only for Python before 3.13.
"""

from __future__ import annotations

import logging
import re
import warnings
from functools import lru_cache

HAN = re.compile(r"[一-鿿]")
# An English word or phrase, or everything between two of them
SEGMENT = re.compile(r"([A-Za-z][A-Za-z '\-]*[A-Za-z]|[A-Za-z])|([^A-Za-z]+)")


def has_chinese(text: str) -> bool:
    return HAN.search(text) is not None


@lru_cache(maxsize=1)
def front_ends():
    """The Chinese and English text-to-phoneme functions, or None without misaki."""
    try:
        with warnings.catch_warnings():  # jieba's old patterns warn as they compile
            warnings.simplefilter("ignore")
            import jieba
            from misaki import espeak, zh
    except ImportError:
        return None
    jieba.setLogLevel(logging.WARNING)  # it announces its dictionary on stderr
    return zh.ZHG2P(), espeak.EspeakG2P(language="en-us")


def available() -> bool:
    return front_ends() is not None


def phonemize(text: str) -> str:
    """The text as Kokoro's phonemes. Needs misaki: check `available()` first."""
    chinese, english = front_ends()
    parts = []
    for latin, other in SEGMENT.findall(text):
        if latin:
            parts.append(english(latin.strip())[0])
        elif other.strip():
            parts.append(chinese(other.strip())[0])
    return " ".join(parts)
