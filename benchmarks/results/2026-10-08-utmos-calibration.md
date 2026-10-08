# 2026-10-08, UTMOS22 scale and sensitivity

Context for the predicted-naturalness numbers in the README: where real human speech falls on
the same scale, and what changes a score. Same model as the comparison (UTMOS22 strong, from
SpeechMOS v1.2.0), scored on an Apple M2 Max. [`docs/utmos.md`](../../docs/utmos.md) explains the
model and how to read these.

## Where real speech falls

The human recordings come from `fetch_human_reference.py`: the first twelve clips of LJ Speech,
one studio speaker, and of LibriTTS-R test.clean, many audiobook readers after speech
restoration. The engines are the 30 sentences of the [main report](2026-10-08-apple-m2-max.md).

| Recordings | predicted MOS (mean ± standard deviation) | clips |
| --- | ---: | ---: |
| ksay | 4.47 ± 0.05 | 30 |
| Human, studio speaker (LJ Speech) | 4.41 ± 0.06 | 12 |
| Human, audiobook readers (LibriTTS-R) | 4.25 ± 0.09 | 12 |
| macOS say, Samantha | 4.00 ± 0.18 | 30 |
| eSpeak NG | 2.18 ± 0.23 | 30 |
| White noise, no speech (from the probe below) | 1.43 ± 0.11 | 10 |

## What changes a score

Ten `ksay` recordings, damaged one way at a time by `mos_probe.py`, and scored again.

| Condition | predicted MOS | change |
| --- | ---: | ---: |
| untouched | 4.51 ± 0.03 |  |
| telephone bandwidth (low-pass at 3.4 kHz) | 4.46 ± 0.04 | -0.05 |
| white noise 20 dB below the speech | 3.98 ± 0.10 | -0.53 |
| white noise 10 dB below the speech | 2.47 ± 0.26 | -2.04 |
| 30 dB quieter | 4.44 ± 0.05 | -0.07 |
| 24 dB louder, then clipped | 2.80 ± 0.13 | -1.71 |
| coarse six-bit quantisation | 3.20 ± 0.11 | -1.32 |
| played 25% faster (the pitch rises too) | 4.08 ± 0.14 | -0.44 |
| played 20% slower (the pitch drops too) | 4.48 ± 0.03 | -0.03 |
| played backwards | 1.60 ± 0.12 | -2.91 |
| white noise and no speech | 1.43 ± 0.11 | -3.08 |

Raw scores: [`2026-10-08-utmos-human.json`](2026-10-08-utmos-human.json) and
[`2026-10-08-utmos-probe.json`](2026-10-08-utmos-probe.json).
