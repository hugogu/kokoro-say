# kokoro-say

Speak or save text with the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
voices from the command line, the way macOS `say` does. It runs locally through
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx): no API key, no GPU,
no network once the model is cached.

```sh
kokoro-say "Hello there."                          # speak it
kokoro-say -v bf_emma "Good evening." -o hi.wav    # save it (.wav, .flac, .ogg)
echo "Read from a pipe." | kokoro-say - -o out.flac
kokoro-say --stream - < chapter.txt                # speak long text as it is generated
kokoro-say -v '?'                                  # list the voices
```

## Install

```sh
uv tool install git+https://github.com/hugogu/kokoro-say
```

or `pipx install git+https://github.com/hugogu/kokoro-say`. Python 3.11 or later.

The first run downloads the model, about 350 MB, into `~/.cache/kokoro-onnx`
and checks its SHA-256; later runs work offline. Set `KOKORO_MODELS` or pass
`--model-dir` to use another folder, for example one shared by several
accounts on the same machine.

## Usage

```text
kokoro-say [text | -] [-o FILE | --stream] [-v VOICE] [-s SPEED] [-l LANG]
           [--list-voices] [--model-dir DIR]
```

| Option | Meaning |
| --- | --- |
| `text`, `-` | What to say; `-` reads standard input |
| `-o FILE` | Save to `.wav`, `.flac` or `.ogg` instead of playing |
| `--stream` | Start speaking before the whole text has been generated |
| `-v VOICE` | Voice name (default `af_heart`); `-v '?'` lists them |
| `-s SPEED` | Speaking rate from 0.5 to 2.0 (default 1.0) |
| `-l LANG` | espeak language code; by default it follows the voice |
| `--model-dir DIR` | Model folder (default `$KOKORO_MODELS`, else `~/.cache/kokoro-onnx`) |

A voice's first letter is its language: `a` American English, `b` British
English, `e` Spanish, `f` French, `h` Hindi, `i` Italian, `j` Japanese,
`p` Brazilian Portuguese, `z` Mandarin Chinese. The second letter is `f` or `m`.
The English voices sound best; kokoro-onnx turns text into sounds with
espeak-ng, which is weaker for Japanese and Chinese than Kokoro's own `misaki`.

Playback uses `afplay` on macOS, `paplay`, `aplay` or `ffplay` on Linux, and
`winsound` on Windows, once the whole passage has been generated. `--stream`
starts sooner on long text: kokoro-onnx generates speech in batches of up to
about half a minute, and each batch plays while the next one is generated,
joined without a gap as long as generation runs faster than speech. Text
shorter than one batch gains nothing. Streaming plays through PortAudio, which
the sounddevice wheels include on macOS and Windows; on Linux install it first,
for example `sudo apt install libportaudio2`.

## Related tools

- [nazdridoy/kokoro-tts](https://github.com/nazdridoy/kokoro-tts) is the fuller
  tool for documents: EPUB and PDF chapters, voice blending and MP3.
- [hexgrad/kokoro](https://github.com/hexgrad/kokoro), the official package,
  runs on PyTorch and pronounces English best (`python -m kokoro`).

## Credits

- Kokoro-82M by [hexgrad](https://huggingface.co/hexgrad/Kokoro-82M), Apache-2.0.
- kokoro-onnx and its model files by
  [thewh1teagle](https://github.com/thewh1teagle/kokoro-onnx), MIT.
- [espeak-ng](https://github.com/espeak-ng/espeak-ng), GPL-3.0, installed as a
  dependency of kokoro-onnx.

## License

MIT
