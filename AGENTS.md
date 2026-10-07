# kokoro-say

- `src/kokoro_say/cli.py` is the whole tool. Tests replace `ensure_model` and
  `load_kokoro`, so only `test_speaks_with_the_real_model` needs the 350 MB
  model, and it skips when the model is absent.
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
