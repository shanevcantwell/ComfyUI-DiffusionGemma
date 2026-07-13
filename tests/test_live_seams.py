"""tests/test_live_seams.py — live tests closing the two runtime seams the
mocked suite structurally cannot reach (see `tests/README.md` for the
`pytest` / `pytest -m live` two-halves convention):

1. `dgemma/model.py:load_model` is only ever exercised against a mocked
   `DiffusionGemmaForBlockDiffusion.from_pretrained`
   (`tests/test_model_load.py`) — the real bf16 `device_map="auto"` load on
   the real 26B checkpoint (CPU-spill, `_resolve_device` against a real
   `hf_device_map`) is untested by construction.
2. `dgemma/loop.py:decode_frames` — now on the ALWAYS-executed sampler path
   (`nodes/sampler.py:200`) — is only ever exercised against a
   `_FakeProcessor` decoding `torch.tensor([1, 2])` (`tests/test_frames.py`),
   never against a real tokenizer decoding a real per-step canvas tensor
   produced by a real pipeline run.

Both require the real ~53.6GB bf16 checkpoint AND a CUDA device
(`@pytest.mark.live` + the `require_live_weights` fixture, `tests/conftest.py`)
— gracefully SKIPPED, never ERRORed, when either is absent. The model load
itself happens ONCE per module via the module-scoped `live_model` fixture;
each test below still runs its own MINIMAL real generation (low
`num_inference_steps`, short `gen_length` — `gen_length` still rounds up to
one full `canvas_length`-sized canvas, so the step count is what actually
bounds wall time), since each test is proving a distinct seam and none of
them can substitute for another.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import (
    DEFAULT_CONFIDENCE,
    DEFAULT_ENTROPY_BOUND,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    decode_frames,
    run_diffusion,
)
from dgemma.model import load_model
from dgemma.types import DGemmaModel
from surfaces.comfyui.sampler import DGemmaSampler

pytestmark = pytest.mark.live

LIVE_QUANT = "none"  # the only quant value load_model accepts (issue #18)
LIVE_NUM_STEPS = 4  # minimal — proving the seam, not a quality benchmark
LIVE_GEN_LENGTH = 64  # pipeline rounds this up to canvas_length (256) regardless


@pytest.fixture(scope="module")
def live_model(require_live_weights) -> DGemmaModel:
    """Load the real DiffusionGemma model ONCE for every test in this
    module — a ~53.6GB bf16 load is not a per-test cost. Depends on
    `require_live_weights` (session-scoped, `tests/conftest.py`), so a box
    missing the checkpoint or a CUDA device skips every test below instead
    of erroring."""
    return load_model(quant=LIVE_QUANT)


def test_load_model_real(live_model):
    """Closes seam (1): `test_model_load.py` only ever calls `load_model`
    against a monkeypatched `from_pretrained`. Here the real
    `device_map="auto"` bf16 load runs against the real checkpoint, and
    `_resolve_device` resolves a real execution device from the real
    `hf_device_map` it produces — not a `FakeHfModel` stand-in."""
    assert isinstance(live_model, DGemmaModel)
    assert live_model.device  # resolves to a non-empty real device string
    assert live_model.dtype == "bfloat16"
    assert live_model.quant == "none"


def test_run_and_decode_frames_real(live_model):
    """Closes seam (2): `test_frames.py` only ever calls `decode_frames`
    against a `_FakeProcessor` decoding `torch.tensor([1, 2])`. Here one
    minimal real `run_diffusion` run produces real per-step canvas tensors,
    and `decode_frames(model.processor, canvas_trace.frames)` decodes them
    with the real tokenizer — this is the falsification a fake processor
    structurally cannot provide."""
    text, canvas_state, canvas_trace = run_diffusion(
        live_model,
        "Why is the sky blue?",
        seed=0,
        num_inference_steps=LIVE_NUM_STEPS,
        gen_length=LIVE_GEN_LENGTH,
    )

    frames = decode_frames(live_model.processor, canvas_trace.frames)

    assert isinstance(frames, list)
    assert len(frames) == len(canvas_trace.frames)
    assert all(isinstance(f, str) for f in frames)

    # Evidence of evolution — a `decode_frames` bug that silently returned
    # the same string for every frame (e.g. always decoding frame 0, or
    # decoding the wrong tensor) would pass the shape/type asserts above but
    # fail this one. Accept either signal: the decoded text visibly changed
    # across steps, OR the run reached (near-)full commitment by the last
    # captured frame — a run that fully converges in very few steps can
    # legitimately decode identically once it locks in early.
    evolved = frames[0] != frames[-1]
    near_converged = canvas_state.committed_fraction >= 0.9
    assert evolved or near_converged, (
        "no evidence of evolution across frames: frames[0] == frames[-1] "
        f"and committed_fraction={canvas_state.committed_fraction!r}"
    )


def test_sampler_node_frames_real(live_model):
    """Drives the actual `DGemmaSampler.sample()` node method (not just the
    underlying `run_diffusion`/`decode_frames` calls) against the real
    model, closing the gap between "the engine seam works" and "the node
    boundary wires it correctly" (`nodes/sampler.py`'s `decode_frames` and
    `render_frames_to_image_batch` calls).

    `unique_id=None` with no real `PromptServer`/ComfyUI process is not a
    reduced assertion — it is the exact headless path every other test in
    this suite already runs under: `_build_on_frame`'s live-push closure
    catches the `server` package's `ImportError` and no-ops
    (`nodes/sampler.py:119-120`; confirmed absent in this venv,
    `tests/test_seam.py`), so `sample()` runs to completion unchanged and
    this asserts the real 5-tuple / `OUTPUT_IS_LIST` shape it returns —
    including `frames_image` (issue #21 rework) rendered from a REAL decoded
    per-step series, not a fake tokenizer's stand-in strings.
    """
    node = DGemmaSampler()
    result = node.sample(
        live_model,
        "Why is the sky blue?",
        seed=0,
        num_inference_steps=LIVE_NUM_STEPS,
        t_min=DEFAULT_T_MIN,
        t_max=DEFAULT_T_MAX,
        entropy_bound=DEFAULT_ENTROPY_BOUND,
        confidence=DEFAULT_CONFIDENCE,
        gen_length=LIVE_GEN_LENGTH,
        thinking=False,
        unique_id=None,
    )

    assert isinstance(result, tuple)
    assert len(result) == 5
    text, canvas_state, canvas_trace, frames, frames_image = result
    assert isinstance(text, str)
    assert isinstance(frames, list)  # the OUTPUT_IS_LIST=True `frames` output
    assert all(isinstance(f, str) for f in frames)
    assert len(frames) == len(canvas_trace.frames)
    # `frames_image` is a single stacked (N, H, W, 3) batch tensor, NOT a
    # list (OUTPUT_IS_LIST=False) — see nodes/sampler.py's docstring.
    assert frames_image.dim() == 4
    assert frames_image.shape[0] == len(frames)
    assert frames_image.shape[-1] == 3
    assert frames_image.dtype == torch.float32
    assert torch.all(frames_image >= 0.0) and torch.all(frames_image <= 1.0)
