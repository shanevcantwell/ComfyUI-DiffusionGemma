"""surfaces/comfyui/denoise.py — DGemmaDenoise: thin ComfyUI adapter (ADR-CDG-003).

ADR-CDG-012 (issue #62 Phase 3): the `KV_CACHE` seam's consumer node — IN-2,
"inject a known-provenance cache." Unpacks widget inputs, calls one
`dgemma.*` function (`dgemma.loop.run_diffusion`, threading `kv_cache=`
through unchanged), wraps the result. Mirrors `DGemmaSampler`'s knob surface
and body shape exactly (same widgets, same `on_frame` live-push wiring) with
one addition: an optional `kv_cache` (`DGEMMA_KV_CACHE`) input.

**Phase-3 scope boundary, named (not silently decided):** `run_diffusion`'s
`kv_cache=` door is still the Phase-2 SKELETON (issue #62 Phase 2 — ingress
validation + `CanvasTrace.injected_cache_provenance` stamp only; the decoder
is not yet driven off the injected cache's tensors). ADR-CDG-012 §4/§D.2
describes a fourth `DGEMMA_KV_CACHE` output ("OUT-1") gated by an optional
"stop at a block boundary" toggle — that mechanism requires the block loop
to actually expose a mid-run stop point, which does not exist until Phase
4's live drive body lands (issue #62 Q-2: gated on the ADR's real-weights
de-risk smoke test). Shipping a `stop_at_block` widget now, with no engine
support behind it, would be exactly the "silently degrade" failure this
pack's doctrine forbids (ADR-CDG-001) — a widget that looks live but does
nothing. This node therefore ships THREE outputs at Phase 3 (`text`,
`canvas_state`, `canvas_trace` — identical to `DGemmaSampler`'s first three);
the fourth (`DGEMMA_KV_CACHE` OUT-1, stop-at-block) is deferred to Phase 4
alongside the live drive body it depends on, not dropped. A reader can
already recover whether/how a run was cache-conditioned via
`canvas_trace.injected_cache_provenance` (OUT-3, live since Phase 2).
"""
from __future__ import annotations

import logging

# Dual-context import, explicit package-depth gate — see
# surfaces/comfyui/loader.py for the full rationale. Gate is
# `__package__.count(".") >= 2` — see loader.py's "GATE CORRECTION" comment.
# Issue #62 implementation plan §M: this file is a new consumer of the
# existing depth-2 predicate, not a fourth gate variant.
if __package__ and __package__.count(".") >= 2:
    from ...dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        run_diffusion,
    )
    from .socket_types import DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, DGEMMA_KV_CACHE, DGEMMA_MODEL
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
    from surfaces.comfyui.socket_types import DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, DGEMMA_KV_CACHE, DGEMMA_MODEL

# Event name for the live per-step push — same mechanism as
# `surfaces/comfyui/sampler.py`'s `DGEMMA_STEP_EVENT`, namespaced separately
# so a UI listening to one node type doesn't also catch the other's pushes.
DGEMMA_DENOISE_STEP_EVENT = "dgemma.denoise.step"


def _build_on_frame(unique_id):
    """Live-push closure — identical shape/guarding to
    `surfaces/comfyui/sampler.py`'s `_build_on_frame` (see that module's
    docstring for the full display-must-never-kill-generation rationale);
    duplicated rather than shared because the two nodes are independent
    thin adapters (ADR-CDG-003) and this closure is the one piece of
    ComfyUI-server-touching code each owns for its own event name."""

    def on_frame(frame) -> None:
        try:
            from server import PromptServer

            instance = PromptServer.instance
            if instance is None:
                return
            instance.send_sync(
                DGEMMA_DENOISE_STEP_EVENT,
                {
                    "node": unique_id,
                    "canvas_idx": frame.canvas_idx,
                    "step_idx": frame.step_idx,
                    "t": frame.t,
                    "temperature": frame.temperature,
                    "committed_fraction": frame.committed_fraction,
                },
            )
        except ImportError:
            return  # No live ComfyUI process (e.g. pytest) — skip the push, not an error.
        except Exception as exc:  # noqa: BLE001 — deliberate breadth: display-only, see docstring.
            logging.warning(
                "DGemmaDenoise live push failed (display only, generation continues): %s", exc
            )

    return on_frame


class DGemmaDenoise:
    """Drives the denoising loop for one prompt, optionally consuming a
    `DGEMMA_KV_CACHE` injected via `DGemmaEncode` (IN-2)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (DGEMMA_MODEL,),
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
            },
            "optional": {
                "kv_cache": (DGEMMA_KV_CACHE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE)
    RETURN_NAMES = ("text", "canvas_state", "canvas_trace")
    FUNCTION = "denoise"
    CATEGORY = "DiffusionGemma"

    def denoise(
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
        kv_cache=None,
        unique_id=None,
    ):
        text, canvas_state, canvas_trace = run_diffusion(
            model,
            prompt,
            seed=seed,
            gen_length=gen_length,
            num_inference_steps=num_inference_steps,
            entropy_bound=entropy_bound,
            t_min=t_min,
            t_max=t_max,
            confidence=confidence,
            kv_cache=kv_cache,
            on_frame=_build_on_frame(unique_id),
        )
        return (text, canvas_state, canvas_trace)
