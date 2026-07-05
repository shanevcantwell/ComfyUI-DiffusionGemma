"""nodes/sampler.py — DGemmaSampler: thin ComfyUI adapter (ADR-CDG-003).

P1 hardcodes the entropy-bound defaults (widgets land in P2, plan.md). Emits
`STRING` (decoded text) **plus** `DGEMMA_CANVAS_STATE` (validity readout) —
never a bare string, so the payload can't lie about whether the canvas
actually finished denoising (ADR-CDG-001 Addendum).
"""
from __future__ import annotations

from dgemma.loop import (
    DEFAULT_ENTROPY_BOUND,
    DEFAULT_GEN_LENGTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    run_diffusion,
)


class DGemmaSampler:
    """Drives the denoising loop for one prompt, EB defaults hardcoded (P1)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DGEMMA_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "DGEMMA_CANVAS_STATE")
    RETURN_NAMES = ("text", "canvas_state")
    FUNCTION = "sample"
    CATEGORY = "DiffusionGemma"

    def sample(self, model, prompt: str, seed: int):
        text, canvas_state = run_diffusion(
            model,
            prompt,
            seed=seed,
            gen_length=DEFAULT_GEN_LENGTH,
            num_inference_steps=DEFAULT_NUM_INFERENCE_STEPS,
            entropy_bound=DEFAULT_ENTROPY_BOUND,
            t_min=DEFAULT_T_MIN,
            t_max=DEFAULT_T_MAX,
        )
        return (text, canvas_state)
