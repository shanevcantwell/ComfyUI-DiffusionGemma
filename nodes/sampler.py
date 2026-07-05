"""nodes/sampler.py — DGemmaSampler: thin ComfyUI adapter (ADR-CDG-003).

P2 promotes the entropy-bound params, seed, and the thinking toggle to
widgets (plan.md Phase 2). Emits `STRING` (decoded text) **plus**
`DGEMMA_CANVAS_STATE` (validity readout) — never a bare string, so the
payload can't lie about whether the canvas actually finished denoising
(ADR-CDG-001 Addendum). Widget names match `dgemma.loop.run_diffusion`'s own
kwarg names 1:1 (`num_inference_steps`, `gen_length`, ...) rather than
introducing a separate node-facing vocabulary (plan.md's shorthand labels,
e.g. "max_steps"/"canvas_length", are prose labels for the same grounded
values, not a distinct parameter set) — this keeps `sample()` a pure
unpack-and-forward with no translation logic of its own (ADR-CDG-003).
Validation (`t_min < t_max`) lives on the engine side, in
`run_diffusion` itself — not scattered into this adapter.
"""
from __future__ import annotations

# Dual-context import, explicit package-depth gate — see nodes/loader.py for
# the full rationale (ComfyUI loader context vs. pytest/standalone; observed
# violation 2026-07-05, enforced by tests/test_comfyui_loader_context.py).
if __package__ and "." in __package__:
    from ..dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        run_diffusion,
    )
else:
    from dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        run_diffusion,
    )


class DGemmaSampler:
    """Drives the denoising loop for one prompt; EB params/seed/thinking are
    widgets (P2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("DGEMMA_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "num_inference_steps": (
                    "INT",
                    {"default": DEFAULT_NUM_INFERENCE_STEPS, "min": 1, "max": 1024},
                ),
                "t_min": ("FLOAT", {"default": DEFAULT_T_MIN, "min": 0.0, "max": 1.0, "step": 0.01}),
                "t_max": ("FLOAT", {"default": DEFAULT_T_MAX, "min": 0.0, "max": 1.0, "step": 0.01}),
                "entropy_bound": (
                    "FLOAT",
                    {"default": DEFAULT_ENTROPY_BOUND, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "confidence": (
                    "FLOAT",
                    {"default": DEFAULT_CONFIDENCE, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                "gen_length": ("INT", {"default": DEFAULT_GEN_LENGTH, "min": 1, "max": 8192}),
                "thinking": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "DGEMMA_CANVAS_STATE")
    RETURN_NAMES = ("text", "canvas_state")
    FUNCTION = "sample"
    CATEGORY = "DiffusionGemma"

    def sample(
        self,
        model,
        prompt: str,
        seed: int,
        num_inference_steps: int,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        confidence: float,
        gen_length: int,
        thinking: bool,
    ):
        text, canvas_state = run_diffusion(
            model,
            prompt,
            seed=seed,
            gen_length=gen_length,
            num_inference_steps=num_inference_steps,
            entropy_bound=entropy_bound,
            t_min=t_min,
            t_max=t_max,
            confidence=confidence,
            thinking=thinking,
        )
        return (text, canvas_state)
