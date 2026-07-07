"""End-to-end DiffusionGemma load + generate — the original `live` test.

Gating idiom (reconciled 2026-07-06, was env-var `DGEMMA_INTEGRATION=1` +
an inline cache check): SELECTION is the `live` pytest marker (`pytest -m
live` opts in; the default `pytest` excludes it via `pyproject.toml`'s
`addopts`) — the env var did the same "explicit opt-in" job the marker now
does, so it was dropped rather than kept alongside it as a second gate.
Per-test READINESS (weights actually cached + a CUDA device present) is the
`require_live_weights` fixture (`tests/conftest.py`), which SKIPS — not
errors — when either is missing, so `pytest -m live` on a box without the
checkpoint/GPU reports a skip. The `google/diffusiongemma-26B-A4B-it`
checkpoint is ~53.6GB (CLAUDE.md) — a 26B load is not a unit-test cost,
which is exactly why it lives behind both the marker and the fixture.

QUANT CHOICE — visible, not buried: `INTEGRATION_QUANT` below defaults to
(and, since issue #18 removed the bnb nf4/int8 paths, only accepts) `"none"`
— unquantized bf16, `device_map="auto"`, accelerate may legally spill the
remainder to CPU. bitsandbytes could never quantize this model's fused 3D
MoE expert parameters (the NF4 footprint was ~45GiB regardless — grounded in
`loose-ends.md`, 2026-07-05 bnb-MoE entry), so the env-var override was
removed along with the dead code rather than left pointing at a `ValueError`.
A real quantized path is tracked in issue #4.

REDUCED KNOBS — named, not silent: `num_inference_steps=8` (vs. the grounded
default 48) because each denoising step is a full 26B forward with
CPU-offloaded experts; 8 steps proves the loop end-to-end (load → denoise →
frames → decode → validity readout) at ~1/6 the wall time. `gen_length=64`
requests the minimum, but note the pipeline rounds gen_length up to a
multiple of the model's `canvas_length` (256), so this still denoises one
full 256-token canvas — the reduction that actually bites is the step count.
"""
from __future__ import annotations

import time

import pytest

from dgemma.loop import run_diffusion
from dgemma.model import load_model

INTEGRATION_QUANT = "none"  # the only quant value load_model accepts (issue #18)
INTEGRATION_NUM_STEPS = 8
INTEGRATION_GEN_LENGTH = 64

pytestmark = pytest.mark.live


def test_load_and_generate_smoke(require_live_weights):
    t0 = time.perf_counter()
    model = load_model(quant=INTEGRATION_QUANT)
    load_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    text, canvas_state, canvas_trace = run_diffusion(
        model,
        "Why is the sky blue?",
        seed=0,
        num_inference_steps=INTEGRATION_NUM_STEPS,
        gen_length=INTEGRATION_GEN_LENGTH,
    )
    generate_seconds = time.perf_counter() - t1

    # Readback for the run log (run pytest with -s to stream it live).
    print(
        f"\n[integration readback] quant={INTEGRATION_QUANT} "
        f"load={load_seconds:.1f}s generate={generate_seconds:.1f}s "
        f"steps_used={canvas_state.steps_used} "
        f"({generate_seconds / max(canvas_state.steps_used, 1):.1f}s/step) "
        f"committed_fraction={canvas_state.committed_fraction:.4f} "
        f"converged={canvas_state.converged}"
    )
    print(f"[integration readback] text[:300]={text[:300]!r}")

    assert isinstance(text, str) and text
    assert 0.0 <= canvas_state.committed_fraction <= 1.0
    assert canvas_state.steps_used > 0
    assert canvas_trace.scheduler_name == "EntropyBoundScheduler"
    assert len(canvas_trace.frames) == canvas_state.steps_used

    # Seed determinism (reviewer gap): same seed, same output. Affordable
    # here because the model is already resident and generation measured
    # ~2.6s/step on the first green run (2026-07-05) — one extra 8-step pass.
    text_again, state_again, trace_again = run_diffusion(
        model,
        "Why is the sky blue?",
        seed=0,
        num_inference_steps=INTEGRATION_NUM_STEPS,
        gen_length=INTEGRATION_GEN_LENGTH,
    )
    assert text_again == text
    assert state_again.committed_fraction == canvas_state.committed_fraction
    assert len(trace_again.frames) == state_again.steps_used
