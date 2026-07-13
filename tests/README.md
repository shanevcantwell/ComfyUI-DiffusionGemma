# Test suite: three tiers

Bare `pytest` runs the **fast, mocked** suite — no weights, no GPU, CI-safe
(every real boundary — `from_pretrained`, the tokenizer, the pipeline — is a
fake or monkeypatch). `pytest -m live` runs the **live** tier instead —
`tests/test_integration.py` (end-to-end load + generate smoke) and
`tests/test_live_seams.py` (real `load_model`, real `decode_frames` on a
real per-step canvas, the real `DGemmaSampler` node) — against the actual
`google/diffusiongemma-26B-A4B-it` checkpoint (~53.6GB bf16) on a real CUDA
device, **importing the implementation directly** (`from dgemma.loop import
run_diffusion`, etc.) — an in-process seam test. `pytest -m e2e` runs the
third tier — `tests/e2e/` — the **black-box battery** (ADR-CDG-013): it
queues workflow-schema JSONs against a headless ComfyUI subprocess via
ComfyUI's own HTTP+websocket API and asserts only on `/history`/`/ws`/
`/object_info` responses, importing **nothing** from `dgemma`/`surfaces`/
`consumers` (enforced by `tests/e2e/test_e2e_import_guard.py`, mirroring
`tests/test_seam.py`'s subprocess pattern). The three tiers are
complementary, not redundant: the mocked suite buys its 100% coverage with
fakes planted at exactly the boundaries (`load_model`'s `from_pretrained`,
`decode_frames`'s processor) a fake cannot falsify; the live tier is the
only in-process instrument that reaches them; the e2e tier is the only tier
that proves the wiring ComfyUI itself constructs (node cache, `/prompt`
scheduling, websocket push, `/interrupt` propagation) — see ADR-CDG-013 for
why the live tier's in-process imports make it structurally blind to bugs
in that wiring (#9/#36/#38).

Gating idiom (one coherent mechanism, not two): the `live`/`e2e` markers
(`pyproject.toml`) are the **selection** switch — `-m 'not live and not
e2e'` in the default `addopts` excludes both, `pytest -m live` / `pytest -m
e2e` opt into each on its own. The `require_live_weights` fixture
(`conftest.py`) is the `live` tier's per-run **readiness** gate — it
`pytest.skip()`s (never errors) when the checkpoint isn't cached or no CUDA
device is present. The `e2e` tier has its own readiness gate
(`tests/e2e/conftest.py`'s server-launch fixture) that SKIPs (never errors)
when the ComfyUI install, the weights, or the GPU precondition is absent —
same discipline, one more tier. A test that used to gate itself with an env
var (`DGEMMA_INTEGRATION=1`) now uses a marker for that same "explicit
opt-in" job instead — one idiom, not an env var and a marker doing
overlapping work.

Not every cache-gated test is `live`: `test_chat_template_thinking.py` only
loads the tokenizer/processor config (no forward pass, no GPU, no 53GB
weights), so it keeps its own lightweight module-level `skipif` and stays in
the fast default suite wherever that config happens to be cached.
