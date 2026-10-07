# kokoro-say

- `src/kokoro_say/cli.py` is the whole tool. Tests replace `ensure_model`,
  `load_kokoro` and `open_output`, so no test plays sound and only the
  `real_model` tests need the 350 MB model; they skip when it is absent.
- On macOS the first phonemization, not the model, made start-up slow: phonemizer
  loads four freshly written copies of the espeak-ng library, and macOS checks
  each new library file once, 0.45 to 0.8 s apiece. `reuse_espeak_copies` keeps
  the copies in `~/.cache/kokoro-say`. Time each stage before blaming the model
  when start-up slows down.
- `speaker()` gives SIGINT its default action while kokoro-say generates and
  plays speech. Python's KeyboardInterrupt would wait for the batch onnxruntime
  is synthesizing, which cannot be interrupted, and a process that exits with
  130 instead of dying of SIGINT lets a calling shell loop carry on to its next
  turn.
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
