# kokoro-say

Speak or save text with the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
voices from the command line, the way macOS `say` does, as the command `ksay`.
It runs locally through
[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx): no API key, no GPU,
no network once the model is cached.

```sh
ksay "Hello there."                          # speak it
ksay -v bf_emma "Good evening." -o hi.wav    # save it (.wav, .flac, .ogg)
echo "Read from a pipe." | ksay - -o out.flac
ksay --stream - < chapter.txt                # speak long text as it is generated
ksay -v '?'                                  # list the voices
```

## Install

```sh
uv tool install git+https://github.com/hugogu/kokoro-say
```

or `pipx install git+https://github.com/hugogu/kokoro-say`. Python 3.11 or later.
The package is called kokoro-say; the command it installs is `ksay`.

The first run downloads the model, about 350 MB, into `~/.cache/kokoro-onnx`
and checks its SHA-256; later runs work offline. Set `KOKORO_MODELS` or pass
`--model-dir` to use another folder, for example one shared by several
accounts on the same machine. On macOS it also keeps copies of the espeak-ng
library in `~/.cache/kokoro-say`, which saves about two seconds on every start:
macOS checks each newly written library before loading it, and phonemizer
would otherwise write four of them on every run.

## Usage

```text
ksay [text | -] [-o FILE | --stream] [-v VOICE] [-s SPEED] [-l LANG]
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

Speech plays through PortAudio, which the sounddevice wheels include on macOS
and Windows; on Linux install it first, for example
`sudo apt install libportaudio2`. It starts once the whole passage has been
generated, or sooner on long text with `--stream`: kokoro-onnx generates speech
in batches of up to about half a minute, and each batch plays while the next
one is generated, joined without a gap as long as generation runs faster than
speech. Text shorter than one batch gains nothing.

## Platforms

| System | Speech to a file | Playback |
| --- | --- | --- |
| macOS, Apple silicon | CI | CI on the runner's audio device; by ear on an M2 Max |
| Linux, x86-64 | CI | CI, into a fake ALSA sound card; needs `libportaudio2` |
| Linux, arm64 | by hand, in a container | by hand, into a fake ALSA sound card |
| Windows, x64 | CI | not heard on real speakers; CI checks the message when there is no audio device |

CI runs the real model on GitHub Actions with Python 3.11 and 3.14. Intel Macs
are not supported, because onnxruntime stopped publishing macOS wheels for them
after 1.23.2. Windows on Arm has a wheel for every dependency but has not been
run. On some Linux virtual machines, GitHub's runners among them, onnxruntime
prints a harmless warning about PCI bus discovery.

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
