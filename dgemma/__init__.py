"""dgemma — the ComfyUI-agnostic DiffusionGemma engine (ADR-CDG-003).

Plain Python + torch/transformers/diffusers. Imports and runs with **zero
ComfyUI present** — that is the whole reason this package exists, split out
from `nodes/` (see `tests/test_seam.py`, the enforcement surface). `nodes/`
imports from here; this package never imports from `nodes/` or `comfy`.

Owns the model (`model.py`), the dataclasses (`types.py`), and the denoising
loop (`loop.py`). `schedule.py`/`sampling.py` land in later phases (plan.md).
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
from .types import CanvasState, DGemmaModel, DiffusionFrame

__all__ = [
    "CanvasState",
    "DGemmaModel",
    "DiffusionFrame",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_ENTROPY_BOUND",
    "DEFAULT_GEN_LENGTH",
    "DEFAULT_NUM_INFERENCE_STEPS",
    "DEFAULT_QUANT",
    "DEFAULT_T_MAX",
    "DEFAULT_T_MIN",
    "DEFAULT_REPO_ID",
    "THINK_TOKEN",
    "load_model",
    "run_diffusion",
]
