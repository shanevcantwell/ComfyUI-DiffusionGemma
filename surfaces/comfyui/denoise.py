"""surfaces/comfyui/denoise.py — DGemmaDenoise: thin ComfyUI adapter (ADR-CDG-003).

ADR-CDG-012 (issue #62 Phase 3): the `KV_CACHE` seam's consumer node — IN-2,
"inject a known-provenance cache." Unpacks widget inputs, calls one
`dgemma.*` function (`dgemma.loop.run_diffusion`, threading `kv_cache=`
through unchanged), wraps the result. Mirrors `DGemmaSampler`'s knob surface
and body shape exactly (same widgets, same `on_frame` live-push wiring) with
one addition: an optional `kv_cache` (`DGEMMA_KV_CACHE`) input.

**Full connection parity with `DGemmaSampler` (issue #166 ratified scope,
operator scope decision 2026-07-30):** this node's output signature is
`DGemmaSampler`'s existing six-output socket set, adopted verbatim —
`text`, `canvas_state`, `canvas_trace`, `frames`, `images`, `run_config` —
built by the SAME shared helper the sampler calls
(`surfaces.comfyui.emission.build_sampler_shaped_outputs`), not a
per-node-duplicated copy. Input-side parity extends to the `thinking`
widget (same declaration/default as the sampler's), threaded to
`run_diffusion` the same way. Two deltas remain, both named rather than
silently absorbed (per the #166 decision comment, these are denoise's
legitimate deltas, not gaps):

1. **`kv_cache` input** (`DGEMMA_KV_CACHE`, optional) — the whole reason
   this node exists (IN-2), absent from the sampler by construction.
2. **OUT-1 deferral** — ADR-CDG-012 §4/§D.2 describes a fourth
   `DGEMMA_KV_CACHE` output ("OUT-1") gated by an optional "stop at a block
   boundary" toggle. That mechanism requires the block loop to actually
   expose a mid-run stop point, which does not exist until Phase 4's live
   drive body lands (issue #62 Q-2: gated on the ADR's real-weights
   de-risk smoke test). Shipping a `stop_at_block` widget now, with no
   engine support behind it, would be exactly the "silently degrade"
   failure this pack's doctrine forbids (ADR-CDG-001) — a widget that
   looks live but does nothing. OUT-1 stays deferred to Phase 4 alongside
   the live drive body it depends on, not dropped. A reader can already
   recover whether/how a run was cache-conditioned via
   `canvas_trace.injected_cache_provenance` (OUT-3, live since Phase 2).

Both nodes' socket signatures (modulo these two named deltas) are asserted
equal by `tests/test_sampler_denoise_parity.py` — the enforcement surface
that fails BY NAME if a future output is added to one node and not the
other.
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
        KNOB_DOCS,
        run_diffusion,
    )
    from .emission import build_sampler_shaped_outputs
    from .socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_KV_CACHE,
        DGEMMA_MODEL,
        DGEMMA_RUN_CONFIG,
    )
else:
    from dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        KNOB_DOCS,
        run_diffusion,
    )
    from surfaces.comfyui.emission import build_sampler_shaped_outputs
    from surfaces.comfyui.socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_KV_CACHE,
        DGEMMA_MODEL,
        DGEMMA_RUN_CONFIG,
    )

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
                # EXPERIMENTAL — same widget, same default, same honesty note
                # as `DGemmaSampler`'s (issue #166 input-side parity): the
                # injected system-turn path is one token short of native
                # `enable_thinking=True`; behavioral impact unverified
                # pending an E2E thinking-mode run on real weights. See
                # `dgemma.loop`'s `thinking` docstring / `KNOB_DOCS` mint.
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": KNOB_DOCS["thinking"],
                    },
                ),
            },
            "optional": {
                "kv_cache": (DGEMMA_KV_CACHE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, "STRING", "IMAGE", DGEMMA_RUN_CONFIG)
    RETURN_NAMES = ("text", "canvas_state", "canvas_trace", "frames", "images", "run_config")
    # Same shape as `DGemmaSampler.OUTPUT_IS_LIST` (issue #166 full
    # connection parity): `frames` is the one list-typed output (one
    # decoded string per captured step); `images` is a single stacked
    # batch tensor, not a list; `run_config` is one plain object.
    OUTPUT_IS_LIST = (False, False, False, True, False, False)
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
        thinking: bool,
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
            thinking=thinking,
            kv_cache=kv_cache,
            on_frame=_build_on_frame(unique_id),
        )
        # Output construction (RunConfig build, frames decode, images
        # render, tuple assembly) is the shared emission helper (issue
        # #166) — identical body `DGemmaSampler.sample` calls.
        return build_sampler_shaped_outputs(
            model=model,
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            t_min=t_min,
            t_max=t_max,
            entropy_bound=entropy_bound,
            confidence=confidence,
            gen_length=gen_length,
            thinking=thinking,
            text=text,
            canvas_state=canvas_state,
            canvas_trace=canvas_trace,
        )
