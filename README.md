<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/ksay-logo-dark.svg">
    <img src="docs/logo/ksay-logo-light.svg" alt="ksay" height="88">
  </picture>
</h1>

**Natural-sounding text-to-speech for your terminal.**<br>
Offline, free, and as easy to use as `say`.

[![CI](https://github.com/hugogu/kokoro-say/actions/workflows/ci.yml/badge.svg)](https://github.com/hugogu/kokoro-say/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![macOS, Linux, Windows](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

[Hear it](#hear-it) · [Install](#install) · [Usage](#usage) · [Comparison](#how-it-compares) · [Platforms](#platforms) · [Troubleshooting](#troubleshooting)

</div>

`ksay` reads text aloud, or saves it as an audio file, with the
[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) neural voices. It runs on your
CPU through [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx): no API key, no
account, no GPU, and no network once the model is cached.

```sh
ksay "Hello there."                          # speak it
ksay -v bf_emma "Good evening." -o hi.mp3    # save it (.wav, .flac, .ogg or .mp3)
echo "Read from a pipe." | ksay -            # read standard input
ksay --stream - < chapter.txt                # start speaking before the text is finished
```

## Hear it

https://github.com/user-attachments/assets/009fa92e-1835-455f-a6a4-e4e62cd645ec

Press play: `ksay` introduces itself in its own voice, with the commands it mentions on
screen. It reads [`docs/audio/intro.txt`](docs/audio/intro.txt), with the loudness matched
to −16 LUFS ([`scripts/make_audio.py`](scripts/make_audio.py) rebuilds it). Prefer a file?
[Download the MP3](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/intro.mp3).

<details>
<summary>Transcript</summary>

> Hi, I'm ksay, a text-to-speech command for your terminal.
>
> I speak with Kokoro, a neural voice model small enough to run on a laptop. Everything
> stays on your machine: no account, no API key, and no graphics card.
>
> Type ksay, then a sentence, and I'll say it out loud. Add a file name, and I'll save
> it instead. Pipe in a whole chapter with the stream option, and I'll start reading
> before I've finished generating the rest.
>
> I have fifty-four voices in eight languages, and I run on macOS, Linux, and Windows.
>
> Install me with uv, and let me read to you.

</details>

## Why ksay

- **Natural voices.** Kokoro-82M is a neural model with 54 voices in 8 languages. In
  the [comparison](#how-it-compares) below it scores 4.5 out of 5 on predicted
  naturalness, against 4.0 for macOS `say` and 2.2 for eSpeak NG, and about where the same
  model puts clean human recordings (4.3 to 4.4).
- **Private and offline.** Nothing you read leaves your computer, and there is
  nothing to sign up for.
- **Familiar.** `ksay "text"`, `-o file`, standard input, `-v voice`, `-s speed`.
- **Quick to start talking.** A second or two to the first word on an M2 Max, and
  `--stream` begins speaking a long text before it has been generated.
- **Fast enough.** About six times faster than real time on an M2 Max CPU: a 90-second
  passage in 15 seconds.
- **Everywhere.** macOS (Apple silicon), Linux and Windows, tested in CI with the real
  model.

## Install

```sh
uv tool install git+https://github.com/hugogu/kokoro-say
```

or `pipx install git+https://github.com/hugogu/kokoro-say`. It needs Python 3.11 or
later and installs the command `ksay`. To update, run the same command again with
`uv tool install --force`; to remove it, `uv tool uninstall kokoro-say`.

On Linux, playing sound needs PortAudio: `sudo apt install libportaudio2` on Debian
and Ubuntu. Saving to a file does not.

The first run downloads the model, 354 MB, into `~/.cache/kokoro-onnx` and checks its
SHA-256; after that `ksay` works offline. Set `KOKORO_MODELS` or pass `--model-dir` to
keep it elsewhere, for example one folder shared by several accounts.

## Usage

```sh
ksay "Hello there."                          # speak it
ksay -v bf_emma -s 1.1 "Good evening."       # another voice, a little faster
ksay "Save me." -o save.mp3                  # write a file instead of playing
cat chapter.txt | ksay --stream -            # read standard input, speaking at once
ksay -v '?'                                  # list the voices
```

```text
ksay [text | -] [-o FILE | --stream] [-v VOICE] [-s SPEED] [-l LANG]
     [--list-voices] [--model-dir DIR]
```

| Option | Meaning |
| --- | --- |
| `text`, `-` | What to say; `-` reads standard input |
| `-o FILE` | Save to `.wav`, `.flac`, `.ogg` or `.mp3` instead of playing |
| `--stream` | Start speaking before the whole text has been generated |
| `-v VOICE` | Voice name (default `af_heart`); `-v '?'` lists them |
| `-s SPEED` | Speaking rate from 0.5 to 2.0 (default 1.0) |
| `-l LANG` | espeak language code; by default it follows the voice |
| `--model-dir DIR` | Model folder (default `$KOKORO_MODELS`, else `~/.cache/kokoro-onnx`) |

A voice's first letter is its language and the second is `f` or `m`:

| Language | Voices | Examples |
| --- | ---: | --- |
| American English | 20 | `af_heart` (default), `af_bella`, `am_michael`, `am_adam` |
| British English | 8 | `bf_emma`, `bf_isabella`, `bm_george`, `bm_lewis` |
| Spanish | 3 | `ef_dora`, `em_alex` |
| French | 1 | `ff_siwis` |
| Hindi | 4 | `hf_alpha`, `hm_omega` |
| Italian | 2 | `if_sara`, `im_nicola` |
| Japanese | 5 | `jf_alpha`, `jm_kumo` |
| Brazilian Portuguese | 3 | `pf_dora`, `pm_alex` |
| Mandarin Chinese | 8 | `zf_xiaobei`, `zm_yunxi` |

The English voices sound best. kokoro-onnx turns text into sounds with eSpeak NG, which
is weaker for Japanese and Chinese than Kokoro's own `misaki`.

Speech plays through PortAudio and starts once the whole passage has been generated,
or sooner with `--stream`: kokoro-onnx generates speech in batches of up to about half
a minute, and each batch plays while the next one is generated, joined without a gap
as long as generation runs faster than speech. Text shorter than one batch gains
nothing from it.

## How it compares

`ksay` against macOS `say` (the Samantha voice, which is what plain `say` uses) and
eSpeak NG, a classic formant synthesizer. Measured on an Apple M2 Max with 64 GB, in
ordinary desktop use, on 2026-10-08. The full report, with every mistake, is in
[`benchmarks/results`](benchmarks/results/2026-10-08-apple-m2-max.md), and
[`benchmarks/`](benchmarks/) holds the scripts to repeat it.

### Speed, memory and size

| | ksay | say | eSpeak NG |
| --- | ---: | ---: | ---: |
| A six-word sentence, to a file | 1.7 s | 0.8 s | 0.4 s |
| 63 words, 22 seconds of speech | 4.4 s | 0.9 s | 0.4 s |
| 268 words, 91 seconds of speech | 14.9 s | 1.2 s | 0.4 s |
| Speed on the long text | 6× real time | 70× | 195× |
| Peak memory | 0.6 to 1.0 GB | 36 MB | 46 to 68 MB |
| Disk | 0.5 GB | built into macOS | 21 MB |

Times include starting the program. `ksay` needs a 354 MB model and 134 MB of packages.

### Quality

| | ksay | say | eSpeak NG |
| --- | ---: | ---: | ---: |
| Predicted naturalness (UTMOS22, 1 to 5) | **4.47** | 4.00 | 2.18 |
| Word error rate, 20 plain sentences | **0.6%** | 2.5% | 15.5% |
| Word error rate, 10 with numbers, dates and names | **0.0%** | **0.0%** | 4.5% |

Word error rate is how often Whisper transcribed the recording wrongly; lower is better.
Predicted naturalness comes from [UTMOS22](docs/utmos.md), a neural model trained on
listening tests. For scale, it gives real human recordings 4.41 (a studio speaker) and
4.25 (audiobook readers), so `ksay` is in their band, which the model cannot tell apart
from synthetic speech this good. It is not a listener, and with 30 sentences a gap of a
point or two in word error rate is noise. [What the score means, and what it
misses](docs/utmos.md).

### Hear the difference

https://github.com/user-attachments/assets/ffcdcc73-f221-4810-930f-c2e352e40f19

Press play: `ksay` reads each passage, then eSpeak NG, at the same loudness. Apple's
licence does not allow publishing recordings of its system voices, so there is no `say`
clip here; on a Mac, hear it yourself:

```sh
ksay "The salt breeze came across from the sea."
say  "The salt breeze came across from the sea."
```

<details>
<summary>The clips as MP3 files</summary>

| Passage | ksay | eSpeak NG |
| --- | --- | --- |
| The salt breeze came across from the sea. | [ksay](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/compare/ksay-sea.mp3) | [eSpeak NG](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/compare/espeak-ng-sea.mp3) |
| The invoice total is $1,250.50, due on March 3, 2026. | [ksay](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/compare/ksay-invoice.mp3) | [eSpeak NG](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/compare/espeak-ng-invoice.mp3) |
| Two sentences of prose | [ksay](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/compare/ksay-explainer.mp3) | [eSpeak NG](https://github.com/hugogu/kokoro-say/raw/main/docs/audio/compare/espeak-ng-explainer.mp3) |

</details>

### Which to use

- **ksay** when people will listen: narration, articles and books, accessibility,
  voice-overs, anything where a robotic voice is a distraction. It pays a second or
  two of start-up and up to a gigabyte of memory while it speaks.
- **say** for instant, tiny, built-in spoken alerts on a Mac, such as "build finished".
  It is the fastest to start and is already there.
- **eSpeak NG** when the footprint matters more than the voice: a small device, many
  languages, or text that must be spoken at once.

Not compared: Apple's downloadable Enhanced and Premium voices, which sound better than
Samantha; Windows' built-in voices; other neural engines such as Piper; cloud services.

## Platforms

| System | Speech to a file | Playback |
| --- | --- | --- |
| macOS, Apple silicon | CI | CI on the runner's audio device; by ear on an M2 Max |
| Linux, x86-64 | CI | CI, into a fake ALSA sound card; needs `libportaudio2` |
| Linux, arm64 | by hand, in a container | by hand, into a fake ALSA sound card |
| Windows, x64 | CI | not heard on real speakers; CI checks the message when there is no audio device |

CI runs the real model on GitHub Actions with Python 3.11 and 3.14. Intel Macs are not
supported, because onnxruntime stopped publishing macOS wheels for them after 1.23.2.
Windows on Arm has a wheel for every dependency but has not been run.

## How it works

```text
text ─► eSpeak NG (phonemes) ─► Kokoro-82M on ONNX Runtime (CPU) ─► audio ─► speakers or file
```

Long text is split into batches of at most 510 phonemes. With `--stream`, a batch plays
while the next is generated. On Apple silicon `ksay` also runs on the performance cores,
keeps eSpeak NG's library copies in `~/.cache/kokoro-say` (macOS checks every new
library once, which would cost about two seconds per run) and opens the audio device
with a 512-frame buffer.

## Troubleshooting

- **`playing speech needs PortAudio`** on Linux: install it, for example
  `sudo apt install libportaudio2`.
- **`cannot open the audio output`** on a server or in a container: there is no sound
  card. Save to a file with `-o`.
- **The first run after installing is slow**, several seconds longer: the model loads
  cold and macOS checks the new libraries once.
- **A warning about PCI bus discovery** on some Linux virtual machines comes from
  onnxruntime and is harmless.
- **Installation fails on an Intel Mac**: see [Platforms](#platforms).
- **The model downloads again**: it lives in `~/.cache/kokoro-onnx`, or in
  `$KOKORO_MODELS` if you set that; keep the variable set in every shell.

## Development

```sh
git clone https://github.com/hugogu/kokoro-say && cd kokoro-say
uv sync
uv run pytest                                # tests that speak for real need the model
uv run ruff check && uv run ruff format --check
```

CI runs the same on Linux, macOS and Windows. [`AGENTS.md`](AGENTS.md) records what was
learned the hard way, and is meant for people and coding agents alike. The recordings
in `docs/audio` are rebuilt with `scripts/make_audio.py`, the two videos made from them
with `scripts/make_intro_video.py` and `scripts/make_compare_video.py`; the numbers in
[Comparison](#how-it-compares) come from `benchmarks/compare.py`.

## Related tools

- [nazdridoy/kokoro-tts](https://github.com/nazdridoy/kokoro-tts) is the fuller tool for
  documents: EPUB and PDF chapters and voice blending.
- [hexgrad/kokoro](https://github.com/hexgrad/kokoro), the official package, runs on
  PyTorch and pronounces English best (`python -m kokoro`).

## Credits and licences

`ksay` is MIT-licensed. It stands on:

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, Apache-2.0.
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) and its model files by
  thewh1teagle, MIT.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng), GPL-3.0, which kokoro-onnx
  installs as a dependency; it is not part of this repository.

The logo is original; its wordmark is set in
[JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) (SIL OFL 1.1), as outlines.
The assets are in [`docs/logo`](docs/logo).

The benchmarks use [Whisper](https://github.com/openai/whisper) and
[UTMOS22 in SpeechMOS](https://github.com/tarepan/SpeechMOS), both MIT, the Harvard
sentences, and, for scale, clips of two public speech datasets, LJ Speech and LibriTTS-R.
None of them is needed to run `ksay`.
