# Benchmarks

The comparison in the [main README](../README.md): `ksay` against macOS `say` (its
default voice and Samantha) and eSpeak NG, on speed, memory, disk, intelligibility,
predicted naturalness and how soon each speaks text that is still being written. Each
number there comes from a report in [`results/`](results/), made by the scripts here (the
disk sizes were added up by hand).

```sh
uv run --python 3.11 benchmarks/compare.py --wer --samples out/ --json results.json
uv run --python 3.11 benchmarks/mos.py out/
uv run --python 3.11 benchmarks/live.py --json live.json
python benchmarks/imports.py ENV-A/bin/python ENV-B/bin/python
```

`compare.py` times and measures memory; with `--wer` it also checks intelligibility,
and with `--samples DIR` it keeps every recording. `mos.py` scores those recordings
for naturalness. `live.py` feeds sentences to each engine through a pipe at a set pace
and times the first sound. `imports.py` times the imports that start every `ksay` run
under the interpreters it is given. They run on macOS or Linux (`say` is skipped on
Linux). The first
`--wer` run downloads Whisper `small.en` (about 480 MB) and the first `mos.py` run
downloads PyTorch and about 400 MB of UTMOS weights; both are cached afterwards.

## What the numbers mean

| Measure | How it is taken |
| --- | --- |
| Time | Wall time of one command, from launch until its file is written, median of six runs (three for the long text) after a warm-up run. It includes start-up, because that is what a script pays, and start-up depends on the Python build, so the report names it and the commands above pin uv's own 3.11, which is what `uv tool install` gives. `ksay` is launched as `python -m kokoro_say`. |
| First sound | For `live.py`: when the first audio exists, counted from the moment the text starts to be written. `ksay`'s audio device is replaced by a recorder, so the real pipeline runs; `say` writes a file, and the time is when the file first holds audio. |
| Speed on the long text | Seconds of speech produced per second of wall time. |
| Memory | Peak resident set size of the command's process, read with `wait4`. For `say` that is the `say` process only. The speech services it calls were each under 20 MB when sampled with Samantha; the default voice runs in a helper process that peaked near 170 MB and one core, which the table leaves out. |
| Intelligibility | Each engine reads 30 sentences, Whisper transcribes the recordings, and the word error rate is computed after both texts pass through Whisper's English text normalizer. Lower is better. |
| Predicted naturalness | UTMOS22, a neural model trained on listening-test scores, predicts the mean opinion score listeners would give (1 bad, 5 excellent). [`docs/utmos.md`](../docs/utmos.md) explains it. |

## Test material

- [`paragraph.txt`](paragraph.txt): 268 words of original prose. The long text is all
  of it, the medium text is its first four sentences, and the short text is a
  six-word greeting.
- [`harvard.txt`](harvard.txt): Harvard sentences, lists 1 and 2 (IEEE Recommended
  Practice for Speech Quality Measurements, 1969). They are phonetically balanced and
  have no numerals, so they test how clearly the voice articulates.
- [`numbers.txt`](numbers.txt): ten sentences with numbers, currency, a date, an
  abbreviation and some hard names. They test how the front end reads what is not
  plain words. Formatting between digits is ignored when scoring them, so
  `415-555-0132` matches `415 555 0132`.

## Limits

- One machine and one day. Times depend on the hardware and on what else it is doing;
  [the report](results/2026-10-08-apple-m2-max.md) states the load.
- The word error rate depends on the recognizer. Whisper is forgiving of robotic
  speech, so it understates how tiring a poor voice is, and it sometimes writes a
  word differently from how it was spoken. Both texts go through the same normalizer,
  but some of those differences remain. Thirty sentences is a small sample: a gap of
  a point or two is noise.
- Predicted naturalness is a model of listeners, not listeners. It ranks engines that
  differ a lot but is not a verdict, and it cannot separate a very good voice from human
  speech: real recordings score 4.25 to 4.41 against `ksay`'s 4.47. Listen for yourself.
- `say` was tested twice: plain `say`, which speaks with the voice selected in the Mac's
  settings (on the test Mac a modern voice rendered by a helper process, not Samantha),
  and `say -v Samantha`, the classic compact voice. The default voice depends on the
  Mac, so on yours that row may measure another one. No English Enhanced or Premium
  voice that macOS can download was installed (one Chinese voice, Tingting, was). Apple's licence does not allow publishing
  recordings of its voices, so this repository holds no `say` audio; `--samples` writes
  it to your own disk for your own listening.
- Not tested: Windows' built-in voices, Piper and other neural engines, cloud services.

## Reading the naturalness score

`fetch_human_reference.py DIR` downloads twelve clips each of two public human-speech
datasets, so that `mos.py DIR` can show where real speech falls on the scale.
`mos_probe.py FOLDER` damages recordings in known ways, to show what moves a score.

## Results

- [2026-10-08, MacBook Pro with Apple M2 Max](results/2026-10-08-apple-m2-max.md)
- [2026-10-08, speech from text that is still being written](results/2026-10-08-live-speech.md)
- [2026-10-08, UTMOS22 scale and sensitivity](results/2026-10-08-utmos-calibration.md)

To add an engine, give `engines()` in `compare.py` a command that writes a file from
a text.
