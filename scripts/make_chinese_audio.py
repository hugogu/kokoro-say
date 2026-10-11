#!/usr/bin/env -S uv run --script
"""Rebuild the Chinese clips the README plays.

    uv run --python 3.12 scripts/make_chinese_audio.py

docs/audio/chinese/ksay-<name>.mp3        ksay with the zh extra, voice zf_xiaobei
docs/audio/chinese/ksay-plain-<name>.mp3  the same without it, eSpeak NG the front end

Every clip is loudness-matched to -16 LUFS like the others (make_audio.py), and the zh
extra needs Python before 3.13. Needs ffmpeg. There is nothing from Apple's say here:
its licence does not allow publishing recordings of its voices.
"""
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["kokoro-say[zh]", "numpy", "soundfile"]
#
# [tool.uv.sources]
# kokoro-say = { path = "..", editable = true }
# ///

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from make_audio import AUDIO, ROOT, loudness_matched_mp3

VOICE = "zf_xiaobei"
CHINESE = {
    "json": "这个 API 返回 JSON 格式的数据，你可以用 Python 调用 requests 库来解析。",
    "settings": (
        "今天我们来讨论一下 machine learning 的应用，"
        "请在 Settings 里打开 Wi-Fi，然后点击 OK。"
    ),
    "hello": "你好，世界。今天天气很好，我们一起去公园散步吧。",
}


def main() -> int:
    import soundfile as sf

    from kokoro_say import chinese, cli

    kokoro = cli.load_kokoro(cli.ensure_model(cli.model_dir()))
    if not chinese.available():
        raise SystemExit(
            "needs the zh extra, so Python 3.12 or older: use --python 3.12"
        )
    folder = AUDIO / "chinese"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        raw = Path(temp) / "raw.wav"
        for name, text in CHINESE.items():
            spoken, phonemes = cli.to_kokoro(text, "cmn")
            samples, rate = kokoro.create(
                spoken, voice=VOICE, lang="cmn", is_phonemes=phonemes
            )
            sf.write(raw, samples, rate)
            loudness_matched_mp3(raw, folder / f"ksay-{name}.mp3")
            samples, rate = kokoro.create(text, voice=VOICE, lang="cmn")
            sf.write(raw, samples, rate)
            loudness_matched_mp3(raw, folder / f"ksay-plain-{name}.mp3")
    for clip in sorted(folder.glob("*.mp3")):
        print(f"{clip.relative_to(ROOT)}  {clip.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
