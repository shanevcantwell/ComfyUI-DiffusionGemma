"""dgemma — the ComfyUI-agnostic DiffusionGemma engine (ADR-CDG-003).

Plain Python + torch/transformers/diffusers. Imports and runs with **zero
ComfyUI present** — that is the whole reason this package exists, split out
from `nodes/` (see `tests/test_seam.py`, the enforcement surface). `surfaces/`
imports from here; this package never imports from `surfaces/`, `consumers/`,
or `comfy`.

Owns the model (`model.py`), the dataclasses (`types.py`), and the denoising
loop (`loop.py`). Derived trace-analysis (heatmap/avalanche-curve builders,
mask-token corroboration) is **not** part of the core's public face — it
moved to `consumers/analysis.py` (ADR-CDG-008 Phase 3; Open Question #1
resolved to `consumers/`, see the ADR's amendment note and issue #55). The
core emits the canonical `CanvasTrace`; analysis parses it as a downstream
consumer, never the reverse (`tests/test_seam.py`'s CDG-008 Phase 4
assertion enforces that `import dgemma` never pulls `consumers.*` in).
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
from .model import AUTOROUND_REPO_ID, DEFAULT_QUANT, DEFAULT_REPO_ID, load_model
from .types import CanvasState, CanvasTrace, DGemmaModel, DiffusionFrame

__all__ = [
    "CanvasState",
    "CanvasTrace",
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
    "AUTOROUND_REPO_ID",
    "THINK_TOKEN",
    "load_model",
    "run_diffusion",
]
