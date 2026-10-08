<div align="center">

<h1>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/ksay-logo-dark.svg">
    <img src="docs/logo/ksay-logo-light.svg" alt="ksay" height="88">
  </picture>
</h1>

**Natural-sounding text-to-speech for your terminal, on macOS, Linux and Windows.**<br>
Offline, free, and as easy to use as `say`, which only macOS has.

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
echo "Read from a pipe." | ksay              # read standard input
ksay -f chapter.txt --stream                 # start speaking before the text is finished
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

- **The same command on macOS, Linux and Windows.** `say` exists only on macOS. On
  Linux the usual built-in, eSpeak NG, scores 2.2 out of 5 on predicted naturalness in
  the [comparison](#how-it-compares), and Windows keeps its natural Narrator voices to
  itself: other programs reach them only through
  [unofficial adapters](https://github.com/gexgd0419/NaturalVoiceSAPIAdapter). The
  natural-sounding offline choice there is an engine such as
  [Piper](https://github.com/OHF-Voice/piper1-gpl), which this page has not compared.
  `ksay` is one install and one command that speaks, the same on all three and tested in
  CI with the real model.
- **Natural voices.** Kokoro-82M is a neural model with 54 voices in 8 languages, English
  best of all. In the [comparison](#how-it-compares) below it scores 4.5 out of 5 on
  predicted naturalness: level with the default voice of macOS `say` (4.4) and with clean
  human recordings (4.3 to 4.4), ahead of the classic Samantha voice (4.0) and far ahead
  of eSpeak NG (2.2).
- **Chinese, and Chinese mixed with English.** With the optional `zh` extra the Chinese
  voices keep their tones and speak the English words of a Chinese sentence in the same
  voice, and JSON is read as a word, not spelled J, S, O, N ([how](#chinese)).
- **Private and offline.** Nothing you read leaves your computer, and there is
  nothing to sign up for.
- **Speaks while you write.** `--stream` speaks each sentence as it arrives, so the
  answer of a language model or a growing log is heard from its first sentence, not
  after its last. `say` waits for the end of its input.
- **Familiar.** `ksay "text"`, `-o file`, standard input, `-v voice`, `-s speed`.
- **Quick to start talking.** About a second to speak a short sentence on an M2 Max, and
  `--stream` starts a long text, or one still being written, as soon as its first
  sentence is there.
- **Fast enough.** About six times faster than real time on an M2 Max CPU: a 90-second
  passage in about 15 seconds.
- **Everywhere.** macOS (Apple silicon), Linux and Windows, tested in CI with the real
  model.

## Install

```sh
uv tool install git+https://github.com/hugogu/kokoro-say
```

or `pipx install git+https://github.com/hugogu/kokoro-say`. It needs Python 3.11 or
later and installs the command `ksay`. To update, run the same command again with
`uv tool install --force`; to remove it, `uv tool uninstall kokoro-say`. For Chinese,
add the `zh` extra: see [Chinese](#chinese).

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
ksay -f chapter.txt                          # read the text from a file
cat chapter.txt | ksay --stream              # read standard input, speaking at once
ksay -v '?'                                  # list the voices
```

```text
ksay [text | -] [-f FILE] [-o FILE | --stream] [-v VOICE] [-s SPEED] [-l LANG]
     [--list-voices] [--model-dir DIR]
```

| Option | Meaning |
| --- | --- |
| `text`, `-` | What to say; `-` reads standard input, and so does leaving it out when input is piped |
| `-f FILE`, `--input-file FILE` | Read the text from a UTF-8 file; `-` reads standard input |
| `-o FILE` | Save to `.wav`, `.flac`, `.ogg` or `.mp3` instead of playing |
| `--stream` | Speak each sentence as soon as it is generated, without waiting for the rest of the text to arrive |
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
is weaker for Japanese than Kokoro's own front end, `misaki`. For Chinese, see
[Chinese](#chinese).

Speech plays through PortAudio. It starts once the whole text has been read and
generated, or sooner with `--stream`: the text is cut into sentences, each is spoken as
soon as it is generated, and the next is generated while one plays, so they join
without a gap as long as generation runs faster than speech. Streaming does not wait
for the end of the input either, so text piped in as it is written is heard as it is
written:

```sh
# a sentence every two seconds; ksay speaks each one as it arrives
for s in "This is the first sentence." "Here is the second one."; do echo "$s"; sleep 2; done | ksay --stream
```

### Chinese

eSpeak NG does not suit Mandarin. Its tone marks are not in Kokoro's vocabulary, so the
voice speaks without tones. Chinese punctuation is lost, an English word inside Chinese
text arrives wrapped in language-switch markers that reach the model as phonemes, and an
English voice reads every character as "Chinese letter".

The optional `zh` extra installs `misaki`, the front end the Chinese voices were trained
with, and `ksay` then gives the Chinese voices (`zf_*`, `zm_*`) its phonemes instead. The
tones are kept, Chinese punctuation pauses the voice, numbers are spoken in Chinese, and
English words in a Chinese sentence are spoken by the same voice, with an accent:

```sh
uv tool install --force --python 3.12 'kokoro-say[zh] @ git+https://github.com/hugogu/kokoro-say'
ksay -v zf_xiaobei "今天我们来讨论一下 machine learning 的应用。"
```

The extra adds about 100 MB, and about 0.9 seconds to the start-up of a Chinese run. It
needs Python 3.12 or older, because that is all `misaki` supports. Without it `ksay` says
so and falls back to eSpeak NG. Chinese text given to an English voice, which is the
default, gets a hint to use a Chinese one.

Judged by ear on three mixed sentences, `ksay` keeps one voice for the English words and
says JSON as a word, where `say`'s Tingting voices change to a second voice for them and
read JSON as letters (a recognizer transcribes `say -v Tingting` as "j, s, o, n"). That
is one listener, not a measurement; the [notes](benchmarks/results/2026-10-08-chinese-mixed.md)
have the transcripts.

## How it compares

`ksay` against macOS `say` and eSpeak NG, a classic formant synthesizer. `say` is
measured twice, because its two usual voices are very different: plain `say` speaks with
the voice selected in the Mac's settings, which on the test Mac is a modern voice
rendered by a helper process, and `say -v Samantha` asks for the classic compact voice.
Measured on an Apple M2 Max with 64 GB, in ordinary desktop use, on 2026-10-08. The full
report, with every mistake, is in
[`benchmarks/results`](benchmarks/results/2026-10-08-apple-m2-max.md), and
[`benchmarks/`](benchmarks/) holds the scripts to repeat it.

### Speed, memory and size

| | ksay | say (default voice) | say -v Samantha | eSpeak NG |
| --- | ---: | ---: | ---: | ---: |
| A six-word sentence, to a file | 1.0 s | 1.3 s | 0.8 s | 0.2 s |
| 63 words, 22 seconds of speech | 3.9 s | 3.7 s | 0.9 s | 0.2 s |
| 268 words, 91 seconds of speech | 14.6 s | 11.8 s | 1.3 s | 0.2 s |
| Speed on the long text | 6× real time | 7× | 68× | 393× |
| Peak memory | 0.6 to 1.0 GB | 37 MB, plus a helper | 36 MB | 43 to 65 MB |
| Disk | 0.5 GB | built into macOS | built into macOS | 20 MB |

Times include starting the program, with uv's Python 3.11, which is what `uv tool
install` uses here; a slower Python build adds to every run (Homebrew's 3.14 imports
`ksay`'s libraries 0.4 s slower). `ksay` needs a 354 MB model and 132 MB of packages. The
helper process that renders the default voice peaked near 170 MB and one core; the
memory row counts the `say` process only.

### Quality

| | ksay | say (default voice) | say -v Samantha | eSpeak NG |
| --- | ---: | ---: | ---: | ---: |
| Predicted naturalness (UTMOS22, 1 to 5) | **4.47** | **4.43** | 4.00 | 2.18 |
| Word error rate, 20 plain sentences | **0.6%** | **0.6%** | 2.5% | 15.5% |
| Word error rate, 10 with numbers, dates and names | **0.0%** | **0.0%** | **0.0%** | 4.5% |

Word error rate is how often Whisper transcribed the recording wrongly; lower is better.
Predicted naturalness comes from [UTMOS22](docs/utmos.md), a neural model trained on
listening tests. For scale, it gives real human recordings 4.41 (a studio speaker) and
4.25 (audiobook readers), so `ksay` and the default `say` voice are both in their band,
which the model cannot tell apart from synthetic speech this good. It is not a
listener, and with 30 sentences a gap of a point or two in word error rate is noise.
[What the score means, and what it misses](docs/utmos.md).

On a Mac, then, `ksay` and plain `say` come out level on quality: the same predicted
naturalness and the same word error rate on plain speech. `ksay` is quicker to say a
single sentence (1.0 s against 1.3 s); `say` is about a fifth quicker on a long text
(11.8 s against 14.6 s) and far lighter. Samantha is the quickest of the Mac voices,
but it scores about 0.4 lower on predicted naturalness and had four of the twenty plain
sentences transcribed inexactly, against one each for the other two. So `ksay` is not a
better voice than the best one on a Mac. It is that quality on every other system as
well, offline, and it can speak text while that is still being written, which `say`
cannot.

### Text that is still being written

`say` reads all of its input before it speaks. `ksay --stream` speaks each sentence as it
arrives. Ten sentences, about a minute of speech, written to a pipe one every two
seconds, so that the text takes 18 seconds to write
([report](benchmarks/results/2026-10-08-live-speech.md)):

| | first sound | speech over |
| --- | ---: | ---: |
| ksay --stream | 1.4 s | 60 s |
| say (default voice) | 18.5 s | 72 s |
| say -v Samantha | 18.5 s | 72 s |
| ksay, without --stream | 27.7 s | 87 s |

`say` speaks 0.5 seconds after the last sentence has been written, however long that
takes: a language model that answers for a minute leaves it silent for a minute.
`ksay --stream` speaks 1.4 seconds after it starts, and does so as well when the text is
already there. In that case `say` is 0.2 seconds ahead with its default voice, and
because `ksay` speaks more slowly at its default speed (59 seconds for this text against
53), its speech ends later.

### Hear the difference

https://github.com/user-attachments/assets/ffcdcc73-f221-4810-930f-c2e352e40f19

Press play: `ksay` reads each passage, then eSpeak NG, at the same loudness. Apple's
licence does not allow publishing recordings of its system voices, so there is no `say`
clip here; on a Mac, hear it yourself:

```sh
ksay "The salt breeze came across from the sea."
say  "The salt breeze came across from the sea."
say -v Samantha "The salt breeze came across from the sea."
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

- **ksay** when the machine is not a Mac, when text arrives over time and should be spoken
  as it comes (a language model, a log), when a script must sound the same on every
  system, or when the audio will be published: Apple's licence does not allow
  recordings of its voices to be shared, and Kokoro's licence, Apache-2.0, sets no such
  limit. On a Mac it costs up to a gigabyte of memory while it speaks.
- **say** on a Mac, when the speech is only for you and the text is complete. It is
  already there, uses little memory, is faster on long texts, and its default voice is as
  natural as `ksay`'s. `say -v Samantha` answers in under a second, which suits spoken
  alerts such as "build finished".
- **eSpeak NG** when the footprint matters more than the voice: a small device, many
  languages, or text that must be spoken at once.

Not compared: Apple's downloadable Enhanced and Premium voices (no English one was installed);
Windows' built-in voices; other neural engines such as Piper; cloud services.

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

Long text is split into batches of at most 510 phonemes. With `--stream` the text is
first cut into sentences, a stop ending one only when a space follows it (so `3.14` and
`example.com` stay whole, and `Dr.` and `U.S.` do not end a sentence), as do a blank line,
a list item and the Chinese and Japanese stops. The input is read on a thread of its
own, a sentence is generated while the one before it plays, and text left without a stop
is spoken after a second of quiet. On Apple silicon `ksay` also runs on the performance cores,
keeps eSpeak NG's library copies in `~/.cache/kokoro-say` (macOS checks every new
library once, which would cost about two seconds per run) and opens the audio device
with a 512-frame buffer.

## Troubleshooting

- **`playing speech needs PortAudio`** on Linux: install it, for example
  `sudo apt install libportaudio2`.
- **`cannot open the audio output`** on a server or in a container: there is no sound
  card. Save to a file with `-o`.
- **`Chinese tones need Kokoro's own front end`**: the `zh` extra is not installed, or
  Python is 3.13 or newer, which `misaki` does not support. See [Chinese](#chinese).
- **Every run starts slowly**: the Python build matters. Importing `ksay`'s libraries took
  0.3 s on uv's own Python 3.11 and 0.7 s on Homebrew's 3.14. Install with
  `uv tool install --force --python 3.11 git+https://github.com/hugogu/kokoro-say` to use
  the quick one.
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
- [Piper](https://github.com/OHF-Voice/piper1-gpl) is a fast, local neural text-to-speech
  engine (GPL-3.0). It has not been compared with `ksay` here.

## Credits and licences

`ksay` is MIT-licensed. It stands on:

- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad, Apache-2.0.
- [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) and its model files by
  thewh1teagle, MIT.
- [eSpeak NG](https://github.com/espeak-ng/espeak-ng), GPL-3.0, which kokoro-onnx
  installs as a dependency; it is not part of this repository.
- [misaki](https://github.com/hexgrad/misaki) by hexgrad, Apache-2.0, Kokoro's own front
  end for Chinese, which the optional `zh` extra installs with jieba, pypinyin and cn2an
  (MIT).

The logo is original; its wordmark is set in
[JetBrains Mono](https://github.com/JetBrains/JetBrainsMono) (SIL OFL 1.1), as outlines.
The assets are in [`docs/logo`](docs/logo).

The benchmarks use [Whisper](https://github.com/openai/whisper) and
[UTMOS22 in SpeechMOS](https://github.com/tarepan/SpeechMOS), both MIT, the Harvard
sentences, and, for scale, clips of two public speech datasets, LJ Speech and LibriTTS-R.
None of them is needed to run `ksay`.
