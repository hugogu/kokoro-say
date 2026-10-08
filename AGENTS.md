# kokoro-say

- `src/kokoro_say/cli.py` is the whole tool. The command is `ksay`; the package
  and repository keep the name kokoro-say. Tests replace `ensure_model`,
  `load_kokoro` and `open_output`, so no test plays sound and only the
  `real_model` tests need the 350 MB model; they skip when it is absent.
- On macOS the first phonemization, not the model, made start-up slow: phonemizer
  loads four freshly written copies of the espeak-ng library, and macOS checks
  each new library file once, 0.45 to 0.8 s apiece. `reuse_espeak_copies` keeps
  the copies in `~/.cache/kokoro-say`. Time each stage before blaming the model
  when start-up slows down.
- A pop at the end of speech was a CoreAudio I/O overload as the stream stopped,
  not the waveform (it ends 50 dB down) and not a cut-off tail (`stop()` waits for
  the audio to drain). PortAudio's low-latency default makes the built-in speakers
  run a 29-frame buffer, 0.3 ms, instead of their usual 512; `block_size()` fixes
  that. `coreaudiod` reports it in the unified log, so look there before guessing:
  `/usr/bin/log show --start "<time>" --predicate 'process == "coreaudiod"' | grep
  Overload` (plain `log` is a zsh builtin). It printed 1–2 events per default
  playback and none with an explicit block size.
- Linux playback is tested without a sound card: `~/.asoundrc` points ALSA's
  default device at its `file` plugin and the test counts the bytes written (see
  the last CI step; a container does the same locally). Windows playback has only
  been run against a machine without an audio device.
- `speaker()` gives SIGINT its default action while ksay generates and
  plays speech. Python's KeyboardInterrupt would wait for the batch onnxruntime
  is synthesizing, which cannot be interrupted, and a process that exits with
  130 instead of dying of SIGINT lets a calling shell loop carry on to its next
  turn.
- `benchmarks/` and `scripts/` hold the PEP 723 scripts behind the README's comparison
  and audio: `compare.py` and `mos.py` measure, `make_audio.py` and
  `make_intro_video.py` build `docs/audio`. Every number in the README's comparison
  must come from a report in `benchmarks/results`; rerun the benchmark and add a new
  report rather than editing a figure by hand.
- Never commit audio made with Apple's `say`. The macOS licence (section "Voices")
  allows system voices for personal, non-commercial use and rules out recording,
  publishing or redistributing them, even non-profit. Measuring `say` locally and
  publishing the numbers is fine.
- GitHub removes `<audio>` and `<video>` from a README. Its file page for an MP3 has no
  player, and the raw file is served with `content-disposition: attachment`, so a link
  to a clip downloads it. The only inline player is for a video uploaded through
  GitHub's own editor (there is no API for that; drag the file into a comment box and
  put the URL it returns on a line of its own). `docs/audio/intro.mp4` and
  `compare.mp4` are rendered for that upload and are not committed.
- Before pushing, run `uv run ruff check`, `uv run ruff format --check` and
  `uv run pytest`. CI runs them on Linux, macOS and Windows, each with the oldest
  and newest supported Python.
- onnxruntime decides which Python versions work. It publishes wheels only, so a
  release without a wheel for an interpreter cannot install there, yet `uv lock`
  still selects it. Read its wheel tags in `uv.lock` before widening
  `requires-python` or the CI matrix.
- `astral-sh/setup-uv` publishes no moving major tags; pin a full release such
  as `v10.2.0`.
- GitHub rejects changes under `.github/workflows` pushed with an OAuth token
  that lacks the `workflow` scope; push those over SSH.
