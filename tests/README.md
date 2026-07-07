# Test suite: two complementary halves

Bare `pytest` runs the **fast, mocked** suite — no weights, no GPU, CI-safe
(every real boundary — `from_pretrained`, the tokenizer, the pipeline — is a
fake or monkeypatch). `pytest -m live` runs the **live** suite instead —
`tests/test_integration.py` (end-to-end load + generate smoke) and
`tests/test_live_seams.py` (real `load_model`, real `decode_frames` on a
real per-step canvas, the real `DGemmaSampler` node) — against the actual
`google/diffusiongemma-26B-A4B-it` checkpoint (~53.6GB bf16) on a real CUDA
device. The two halves are complementary, not redundant: the mocked suite
buys its 100% coverage with fakes planted at exactly the boundaries
(`load_model`'s `from_pretrained`, `decode_frames`'s processor) a fake
cannot falsify; the live suite is the only instrument that reaches them.

Gating idiom (one coherent mechanism, not two): the `live` marker
(`pyproject.toml`) is the **selection** switch — `-m 'not live'` in the
default `addopts` excludes it, `pytest -m live` opts in. The
`require_live_weights` fixture (`conftest.py`) is the per-run **readiness**
gate — it `pytest.skip()`s (never errors) when the checkpoint isn't cached
or no CUDA device is present, so `pytest -m live` on a box without either
reports skips, not failures. A test that used to gate itself with an env
var (`DGEMMA_INTEGRATION=1`) now uses the marker for that same "explicit
opt-in" job instead — one idiom, not an env var and a marker doing
overlapping work.

Not every cache-gated test is `live`: `test_chat_template_thinking.py` only
loads the tokenizer/processor config (no forward pass, no GPU, no 53GB
weights), so it keeps its own lightweight module-level `skipif` and stays in
the fast default suite wherever that config happens to be cached.
