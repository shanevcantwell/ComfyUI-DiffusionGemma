"""End-to-end DiffusionGemma load + generate, gated behind an explicit opt-in
env var AND an HF-cache presence check.

The `google/diffusiongemma-26B-A4B-it` checkpoint is ~53.6GB (CLAUDE.md) — a
26B load is not a unit-test cost, so this only runs when both:
  1. `DGEMMA_INTEGRATION=1` is set (explicit opt-in), and
  2. the checkpoint is already present in the local HF cache.

QUANT CHOICE — visible, not buried: `INTEGRATION_QUANT` below defaults to
`"none"` (unquantized bf16, `device_map="auto"`, accelerate may legally spill
the remainder to CPU) because bitsandbytes cannot quantize this model's fused
3D MoE expert parameters, making the NF4 footprint ~45GiB — it does not fit
the 48GB dev box (grounded in `loose-ends.md`, 2026-07-05 bnb-MoE entry).
Override with `DGEMMA_INTEGRATION_QUANT=nf4|int8` once a working quantized
path exists.

REDUCED KNOBS — named, not silent: `num_inference_steps=8` (vs. the grounded
default 48) because each denoising step is a full 26B forward with
CPU-offloaded experts; 8 steps proves the loop end-to-end (load → denoise →
frames → decode → validity readout) at ~1/6 the wall time. `gen_length=64`
requests the minimum, but note the pipeline rounds gen_length up to a
multiple of the model's `canvas_length` (256), so this still denoises one
full 256-token canvas — the reduction that actually bites is the step count.
"""
from __future__ import annotations

import os
import time

import pytest

from dgemma.loop import run_diffusion
from dgemma.model import DEFAULT_REPO_ID, load_model

INTEGRATION_QUANT = os.environ.get("DGEMMA_INTEGRATION_QUANT", "none")
INTEGRATION_NUM_STEPS = 8
INTEGRATION_GEN_LENGTH = 64


def _weights_cached() -> bool:
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False
    return any(repo.repo_id == DEFAULT_REPO_ID for repo in cache_info.repos)


pytestmark = pytest.mark.skipif(
    os.environ.get("DGEMMA_INTEGRATION") != "1" or not _weights_cached(),
    reason=(
        "Set DGEMMA_INTEGRATION=1 and ensure google/diffusiongemma-26B-A4B-it "
        "is in the local HF cache (~53.6GB) to run this test."
    ),
)


def test_load_and_generate_smoke():
    t0 = time.perf_counter()
    model = load_model(quant=INTEGRATION_QUANT)
    load_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    text, canvas_state = run_diffusion(
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

    # Seed determinism (reviewer gap): same seed, same output. Affordable
    # here because the model is already resident and generation measured
    # ~2.6s/step on the first green run (2026-07-05) — one extra 8-step pass.
    text_again, state_again = run_diffusion(
        model,
        "Why is the sky blue?",
        seed=0,
        num_inference_steps=INTEGRATION_NUM_STEPS,
        gen_length=INTEGRATION_GEN_LENGTH,
    )
    assert text_again == text
    assert state_again.committed_fraction == canvas_state.committed_fraction
