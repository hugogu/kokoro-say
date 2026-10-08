# UTMOS22: a score for how natural speech sounds

The [comparison in the README](../README.md#how-it-compares) gives each voice a
"predicted naturalness" between 1 and 5. That number comes from UTMOS22, a neural
network that listens to a recording and guesses the score a panel of people would give
it. This page explains what that means, where the model comes from, how `ksay` uses it
and, most important, how far to trust it.

**The short version**

- It estimates how natural a recording sounds, on the 1 (bad) to 5 (excellent) scale of
  a listening test, with no listeners and no reference recording.
- On this scale `ksay` scores 4.47, real human recordings 4.25 to 4.41, macOS `say` 4.00
  and eSpeak NG 2.18. Noise-only audio scores 1.43.
- It does not separate a very good synthetic voice from a human one: the top of the
  scale is crowded, and a gap of a few hundredths means nothing.
- It is a prediction, not a listening test. Use it to rank voices that differ a lot, and
  listen for yourself when they are close.

## Why a model instead of listeners

The standard way to judge synthetic speech is a listening test. People hear short clips
and rate each one on a five-point scale, and the average is the *mean opinion score*,
MOS. The ITU describes the method in Recommendation
[P.800](https://www.itu.int/rec/T-REC-P.800-199608-I), where the scale runs Excellent (5),
Good (4), Fair (3), Poor (2) and Bad (1).

It is the gold standard, and it is slow and costly: it needs many listeners, a
controlled setup and a fresh test for every system you want to compare. That cost is what
the UTMOS paper gives as the reason to measure quality automatically.

A model that predicts MOS from audio alone is called *non-intrusive*: unlike metrics such
as PESQ it needs no clean copy of the recording to compare against. That suits speech
synthesis, where there is no "correct" waveform for a sentence.

## Where UTMOS22 comes from

UTMOS (pronounced "u-t-mos") was built at the University of Tokyo (UTokyo-SaruLab, the
Saruwatari laboratory) for the
[VoiceMOS Challenge 2022](https://voicemos-challenge-2022.github.io/), the first of that
series of contests in predicting MOS, which 22 teams entered. The challenge's main track used the
BVCC dataset: listening-test ratings for English synthetic speech from 187 systems taken
from earlier Blizzard Challenges, Voice Conversion Challenges and ESPnet-TTS, gathered by
Cooper and Yamagishi. The part used for training holds 4,974 clips and 39,792 ratings from
288 listeners.

The UTMOS system was first on several measures of that contest. How well its predictions
agree with listeners on the test set, which includes systems, speakers and listeners the
model had not seen, is measured with the Spearman rank correlation (SRCC): 1.0 means the
same ranking as the listeners.

| SRCC with listeners | per recording | per system |
| --- | ---: | ---: |
| Full system, a stack of many models | 0.897 | 0.936 |
| One strong learner, which is what `ksay` runs | 0.881 | 0.925 |

"Per system" averages the predicted scores of every clip from one voice before comparing
the rankings. That is how the README uses it, averaging 30 sentences per engine, and it
agrees better with listeners than a score for a single clip does, because averaging
cancels the noise in individual clips. The figures are from Table 2 and section 4.2 of the
paper, and describe English synthetic speech in the challenge's own listening test.
They do not say how well the model agrees with listeners about Kokoro, `say` or eSpeak NG,
and no listening test of those has been run here.

## How the model works

```text
recording ─► wav2vec 2.0 ─► one vector for every 20 ms of speech
          ─► joined with a fixed "average listener" vector and a fixed "data domain" vector
          ─► bidirectional LSTM ─► small network ─► one score per 20 ms
          ─► the mean of those scores ─► × 2 + 3 ─► predicted MOS, 1 to 5
```

- **wav2vec 2.0** ([Baevski et al., 2020](https://arxiv.org/abs/2006.11477)) learned the
  structure of speech from large amounts of unlabelled audio. UTMOS uses the base-sized
  model pretrained on LibriSpeech and fine-tunes it, with the layers on top, on the BVCC
  ratings. The audio is resampled to 16 kHz first.
- **Listener and domain vectors.** Listeners differ: some rate harshly, some generously.
  During training the network is told who gave each rating and from which listening test,
  so it can learn those habits instead of being confused by them. When predicting, it uses
  a single "average listener".
- **Frame scores.** Training predicts a score for every frame and compares it with the
  recording's rating; a prediction is the mean over the frames. The scores live between
  −1 and 1 and are stretched onto the 1 to 5 scale.
- **Training losses.** The paper combines a clipped squared error with a contrastive loss
  that punishes getting the order of two clips wrong, which helps the rank correlation
  the challenge was judged on. It also changes speaking rate and pitch slightly to stretch
  the training data, and describes a phoneme encoder that feeds in speech-recognition output.

The full UTMOS system stacks up to 17 of these strong learners with 48 "weak learners",
simple regressions such as ridge regression and random forests on averaged wav2vec 2.0,
HuBERT and WavLM features.

**What `ksay` runs** is smaller than that. [SpeechMOS](https://github.com/tarepan/SpeechMOS)
reimplements one strong learner, `utmos22_strong`: audio in, score out, without the
phoneme encoder and without stacking. Its weights are a conversion of the official
[UTMOS22](https://github.com/sarulab-speech/UTMOS22) checkpoint, and both repositories are
MIT-licensed.

## How `ksay` uses it

[`benchmarks/mos.py`](../benchmarks/mos.py) scores every recording in a folder and prints
the mean and standard deviation per voice. It is pinned to SpeechMOS v1.2.0, mixes the
audio down to mono and resamples it to 16 kHz. The first run downloads PyTorch and about
400 MB of weights from GitHub, and runs the model's code from that repository, which
`torch.hub` asks to be told to trust. On CPU it takes a fraction of a second per clip: 24
clips and the model load took eight seconds on an Apple M2 Max.

```sh
uv run benchmarks/compare.py --samples out/    # records the sentences for every engine
uv run benchmarks/mos.py out/                  # one folder per voice, one score each
```

## Reading a score

These measurements were made with the scripts in [`benchmarks/`](../benchmarks/) on
2026-10-08; the [report](../benchmarks/results/2026-10-08-utmos-calibration.md) has the
raw scores.

**Where voices fall.** The human recordings come from two public datasets: twelve clips
of one studio speaker (LJ Speech) and twelve of many audiobook readers, cleaned by speech
restoration (LibriTTS-R). Fetch them with `benchmarks/fetch_human_reference.py`.

| Recordings | predicted MOS | clips |
| --- | ---: | ---: |
| ksay | 4.47 ± 0.05 | 30 |
| Human, studio speaker | 4.41 ± 0.06 | 12 |
| Human, audiobook readers | 4.25 ± 0.09 | 12 |
| macOS `say`, Samantha | 4.00 ± 0.18 | 30 |
| eSpeak NG | 2.18 ± 0.23 | 30 |
| White noise, no speech | 1.43 ± 0.11 | 10 |

`ksay` sits where clean human recordings sit, and 0.06 above the studio speaker is a tie
within the spread of the clips, not a sign that it sounds better than a person. The scale
cannot rank voices that good against each other; it can only say that both are far from
the audibly artificial ones.

**What changes a score.** Ten `ksay` recordings, damaged in one way at a time with
`benchmarks/mos_probe.py`, and scored again.

| Damage | predicted MOS | change |
| --- | ---: | ---: |
| none | 4.51 | |
| low-pass at 3.4 kHz, telephone bandwidth | 4.46 | −0.05 |
| 30 dB quieter | 4.44 | −0.07 |
| played 20% slower, the pitch drops too | 4.48 | −0.03 |
| played 25% faster, the pitch rises too | 4.08 | −0.44 |
| white noise 20 dB below the speech | 3.98 | −0.53 |
| coarse six-bit quantisation | 3.20 | −1.32 |
| 24 dB louder, then clipped | 2.80 | −1.71 |
| white noise 10 dB below the speech | 2.47 | −2.04 |
| played backwards | 1.60 | −2.91 |

The model reacts strongly to noise, distortion and clipping, and to audio that no longer
sounds like speech, as with the backwards clips. It hardly notices how loud a recording
is, or that everything above 3.4 kHz is missing. So it is better at spotting artefacts than at telling hi-fi from
mediocre audio.

## What it cannot tell you

- **Whether the words are right.** The model never sees the text: it takes audio only, so
  nothing in its input says what was meant to be spoken. The README pairs it with a
  Whisper word error rate for that reason.
- **Differences near the top.** It was trained to rank the systems of earlier challenges.
  In the measurements here, Kokoro and human recordings land within a few tenths of each
  other.
- **Anything outside its training.** The ratings are English, from one listening test. The
  challenge's out-of-domain track, Chinese speech from a different test, needed extra
  labelled data to work well, and Kokoro also speaks Japanese, Spanish and more. Every
  clip in the README comparison is English.
- **Style, emotion or long-form quality.** One number per short clip does not say whether
  a voice is expressive, or stays pleasant over a chapter.
- **A comparison with another predictor.** Scores from UTMOSv2, NISQA or DNSMOS follow
  different scales and training. Compare numbers only within one model.
- **A listener's verdict.** It is trained to imitate the average of many listeners, and
  it is wrong for individual clips and individual listeners. When two voices score
  within a couple of tenths of each other, listen.

A newer system from the same laboratory, UTMOSv2
([Baba et al., 2024](https://arxiv.org/abs/2409.09305)), is aimed at high-quality
synthetic speech. It has not been tried here.

## Using it on your own recordings

```sh
uv run benchmarks/mos.py my-voices/        # a folder of folders of .wav, .flac or .mp3
uv run benchmarks/fetch_human_reference.py reference/ && uv run benchmarks/mos.py reference/
uv run benchmarks/mos_probe.py my-voices/some-voice     # what damage does to its score
```

Put human recordings in the comparison, as above, whenever you quote a score: without
them a 4.5 reads as "nearly perfect", when it means "as high as this model scores clean
human speech".

## Further reading

- Saeki et al., [UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022](https://arxiv.org/abs/2204.02152), Interspeech 2022.
- Huang et al., [The VoiceMOS Challenge 2022](https://arxiv.org/abs/2203.11389), Interspeech 2022.
- Cooper and Yamagishi, [How do Voices from Past Speech Synthesis Challenges Compare Today?](https://arxiv.org/abs/2105.02373), SSW 2021: the listening test behind the training data.
- Cooper et al., [Generalization ability of MOS prediction networks](https://arxiv.org/abs/2110.02635), 2021: the fine-tuned-wav2vec baseline UTMOS builds on.
- Code: [UTMOS22](https://github.com/sarulab-speech/UTMOS22), [SpeechMOS](https://github.com/tarepan/SpeechMOS) and [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2).
