# Benchmarks

The comparison in the [main README](../README.md): `ksay` against macOS `say` and
eSpeak NG, on speed, memory, intelligibility and predicted naturalness. Each number
there comes from a script here and a report in [`results/`](results/).

```sh
uv run benchmarks/compare.py --wer --samples out/ --json results.json
uv run benchmarks/mos.py out/
```

`compare.py` times and measures memory; with `--wer` it also checks intelligibility,
and with `--samples DIR` it keeps every recording. `mos.py` scores those recordings
for naturalness. They run on macOS or Linux (`say` is skipped on Linux). The first
`--wer` run downloads Whisper `small.en` (about 480 MB) and the first `mos.py` run
downloads PyTorch and about 400 MB of UTMOS weights; both are cached afterwards.

## What the numbers mean

| Measure | How it is taken |
| --- | --- |
| Time | Wall time of one command, from launch until its file is written, median of six runs (three for the long text) after a warm-up run. It includes start-up, because that is what a script pays. `ksay` is launched as `python -m kokoro_say`. |
| Speed on the long text | Seconds of speech produced per second of wall time. |
| Memory | Peak resident set size of the command's process, read with `wait4`. For `say` that is the `say` process; the speech services it calls in the background were each under 20 MB when sampled. |
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
- `say` was tested with Samantha, the voice plain `say` uses on the test Mac. The
  Enhanced and Premium voices that macOS can download were not installed, and score
  higher. Apple's licence does not allow publishing recordings of its voices, so this
  repository holds no `say` audio; `--samples` writes it to your own disk for your
  own listening.
- Not tested: Windows' built-in voices, Piper and other neural engines, cloud services.

## Reading the naturalness score

`fetch_human_reference.py DIR` downloads twelve clips each of two public human-speech
datasets, so that `mos.py DIR` can show where real speech falls on the scale.
`mos_probe.py FOLDER` damages recordings in known ways, to show what moves a score.

## Results

- [2026-10-08, MacBook Pro with Apple M2 Max](results/2026-10-08-apple-m2-max.md)
- [2026-10-08, UTMOS22 scale and sensitivity](results/2026-10-08-utmos-calibration.md)

To add an engine, give `engines()` in `compare.py` a command that writes a file from
a text.
