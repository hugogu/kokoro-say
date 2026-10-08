# 2026-10-08, speech from text that is still being written

How soon each engine speaks when its text arrives over time, as the answer of a language model or a
growing log does. [`live.py`](../live.py) makes the measurements.

| | |
| --- | --- |
| System | macOS 26.7.1 |
| Chip | Apple M2 Max, 12 cores, 64 GB |
| Python | 3.11.14, uv's own build |
| ksay | commit 40d5d31, voice `af_heart` |
| say (default voice), say (Samantha) | as in the [main report](2026-10-08-apple-m2-max.md) |

A producer prints the first 10 sentences of `paragraph.txt` to a pipe, one every 2 seconds, and
closes it; an engine reads the pipe. The text is 18 s in the writing and 53 to 59 s in the
saying, depending on the voice. The clock starts when the producer does, so every engine pays for starting up.
Median of three runs after a warm-up run; the runs of one engine differ by at most 0.5 s. The one-minute
load average was 4.5 at the start and 6.3 at the end, on 12 cores.

- **First sound** is when the first audio exists. For `ksay` it is its first write to the audio device, which
  is replaced by a recorder, so the real pipeline runs and only the loudspeaker is missing. For `say` it is when
  its output file first holds audio, because `say` is asked for a file here, as in the main report.
- **Speech over** is when the last of it would have been heard, if every block is played as soon as it is
  handed over, or as soon as the one before it has finished.

## A sentence every two seconds

| Engine | first sound | speech over |
| --- | ---: | ---: |
| ksay --stream | 1.4 s | 59.9 s |
| ksay, without --stream | 27.7 s | 86.5 s |
| say (default voice) | 18.5 s | 71.8 s |
| say (Samantha) | 18.5 s | 72.4 s |

`say` speaks 0.4 to 0.5 s after the producer finished, in both voices: it reads all of its input first, as its
manual page says ("spoken all at once"), so its wait is as long as the producer takes. `ksay --stream` speaks
1.4 s after the producer started, the same moment as when all the text is there, because it needs only
the first sentence. Without `--stream`, `ksay` also reads everything first, and then generates all of it before
it plays anything.

## All the text at once

| Engine | first sound | speech over |
| --- | ---: | ---: |
| ksay --stream | 1.4 s | 59.9 s |
| ksay, without --stream | 10.4 s | 69.2 s |
| say (default voice) | 1.1 s | 54.4 s |
| say (Samantha) | 0.8 s | 54.8 s |

With the text already there, `say` starts 0.2 s (its default voice) and 0.5 s (Samantha) sooner than
`ksay --stream`.

At its default speed `ksay` speaks more slowly than `say`'s default voice: 58.5 s for this text against
53.2 s, which is why its speech is over later here although it starts about as soon. A speed of 1.1
(`-s 1.1`) should bring them level, but that was not measured.

The raw numbers are in [`2026-10-08-live-speech.json`](2026-10-08-live-speech.json).
