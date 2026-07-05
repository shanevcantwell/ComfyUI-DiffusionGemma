"""dgemma — the ComfyUI-agnostic DiffusionGemma engine (ADR-CDG-003).

Plain Python + torch/transformers/diffusers. Imports and runs with **zero
ComfyUI present** — that is the whole reason this package exists, split out
from `nodes/` (see `tests/test_seam.py`, the enforcement surface). `nodes/`
imports from here; this package never imports from `nodes/` or `comfy`.

Owns the model (`model.py`), the dataclasses (`types.py`), the denoising loop
(`loop.py`), and — as of Phase 3 — the pure trace-analysis functions
(`sampling.py`: heatmap/avalanche-curve builders, mask-token corroboration).
`schedule.py` lands in a later phase (plan.md).
"""
from __future__ import annotations

from .loop import (
    DEFAULT_CONFIDENCE,
    DEFAULT_ENTROPY_BOUND,
    DEFAULT_GEN_LENGTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    THINK_TOKEN,
    run_diffusion,
)
from .model import DEFAULT_QUANT, DEFAULT_REPO_ID, load_model
from .sampling import (
    MaskTokenCorroboration,
    build_avalanche_curve,
    build_commit_heatmap,
    corroborate_no_mask_token,
)
from .types import CanvasState, CanvasTrace, DGemmaModel, DiffusionFrame

__all__ = [
    "CanvasState",
    "CanvasTrace",
    "DGemmaModel",
    "DiffusionFrame",
    "MaskTokenCorroboration",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_ENTROPY_BOUND",
    "DEFAULT_GEN_LENGTH",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_QUANT",
    "DEFAULT_T_MAX",
    "DEFAULT_T_MIN",
    "DEFAULT_REPO_ID",
    "THINK_TOKEN",
    "build_avalanche_curve",
    "build_commit_heatmap",
    "corroborate_no_mask_token",
    "load_model",
    "run_diffusion",
]
