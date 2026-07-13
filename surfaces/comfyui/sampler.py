"""surfaces/comfyui/sampler.py — DGemmaSampler: thin ComfyUI adapter (ADR-CDG-003).

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

P3 adds a third output, `DGEMMA_CANVAS_TRACE` (plan.md Phase 3 (b)), a live
per-step push (plan.md Phase 3 (a)), and a fourth output, `frames` — a
`STRING` list (`OUTPUT_IS_LIST`), one decoded string per captured step, in
order (the in-graph "flipbook": noise -> coherent text). Decoding is
`dgemma.loop.decode_frames` over `canvas_trace.frames`, called here rather
than inside `run_diffusion` (ADR-CDG-003: the engine's 3-tuple return stays
unchanged; this is a node-boundary derivation from a value `run_diffusion`
already returns). `sample()` builds a closure over
`PromptServer.instance.send_sync` and hands it to `run_diffusion` as
`on_frame` — the ADR-CDG-003-respecting way to let a live view exist without
`dgemma/loop.py` ever importing ComfyUI. `PromptServer` is imported lazily,
inside `sample()`, guarded so its absence (the normal pytest/headless
condition — this pack has no `comfy`/`server` dependency, see
`tests/test_seam.py`) degrades to a no-op live push rather than crashing the
sampler; everything else (`text`, `canvas_state`, `canvas_trace`) proceeds
unchanged either way.

A fifth output, `frames_image` (issue #21, reworked from a standalone
`DGemmaFlipbook` node into a second sampler output): the same decoded
`frames` strings rendered as a single stacked
`(N, H, W, 3)` float32 `[0, 1]` `IMAGE` batch via
`surfaces.comfyui.frames_image.render_frames_to_image_batch` — the "watch it reason"
series made watchable/shareable (e.g. `SaveAnimatedWEBP`/VHS downstream), not
just inspectable as text. Reuses the `frames` list `decode_frames` already
produced (one decode, two renderings) rather than re-decoding
`canvas_trace.frames` a second time. Render params (width/font size/caption)
are fixed sensible defaults here, not new widgets — the sampler's knob
surface (P2) stays unchanged; this is a display rendering, not a sampling
parameter. Unlike `frames`, `frames_image` is `OUTPUT_IS_LIST=False`: it is
ONE stacked batch tensor, not a list of N single-frame tensors — the shape
`PreviewImage`'s scrubber, `SaveAnimatedWEBP`, and VHS nodes all expect from
an `IMAGE` output (a list here would fan out per-frame and break every one of
those consumers).

**Why a sampler output, not a standalone node (issue #21 rework):** the
earlier `DGemmaFlipbook` node took `CANVAS_TRACE` alone and needed a
tokenizer to decode it, but `CANVAS_TRACE` never carried one — forcing
`dgemma.types.CanvasTrace` to grow an optional `processor` field just to
carry a runtime object across a data-plane socket (a payload-purity smell,
ADR-CDG-001). This node already holds `model.processor` and already decodes
`frames` itself, so rendering the image batch here instead keeps
`CANVAS_TRACE` pure.

`DGEMMA_MODEL` / `DGEMMA_CANVAS_STATE` / `DGEMMA_CANVAS_TRACE` socket-type
strings come from the `socket_types` mint module (#35 R2, ADR-CDG-008 Phase
1) — no inline `DGEMMA_*` literal at this site; see
`surfaces/comfyui/socket_types.py`. `DGEMMA_STEP_EVENT` below is NOT part of
that mint — it's a WebSocket event name, not a ComfyUI socket type.

**Named trap (plan.md Risks): this MUST NOT touch `comfy.utils.ProgressBar`'s
`preview=` slot.** That path is structurally image-typed downstream
(`server.py:1293-1301`, `ProgressBar.update_absolute` -> `send_image`, which
calls `image.save(...)` on whatever it's handed) and throws on text. Text
goes out its own custom event via `send_sync`, never through `preview=`.
This is a review-gate risk, not a test-enforced one — there is no clean unit
test for "this code path never calls the wrong API" (plan.md Risks).
"""
from __future__ import annotations

import logging

# Dual-context import, explicit package-depth gate — see
# surfaces/comfyui/loader.py for the full rationale (ComfyUI loader context
# vs. pytest/standalone; observed violation 2026-07-05, enforced by
# tests/test_comfyui_loader_context.py). This module lives two levels under
# the pack root (surfaces/comfyui/), so the relative climb to dgemma/ is
# THREE dots (ADR-CDG-008 Phase 1 / issue #52 risk R-1). `.frames_image` and
# `.socket_types` stay ONE dot — both are siblings in this same directory,
# unaffected by the pack-root depth change. Gate is `__package__.count(".")
# >= 2`, not a bare dot-presence check — see loader.py's "GATE CORRECTION"
# comment: this module's own absolute package name ("surfaces.comfyui")
# contains a dot even under bare pytest, so a naive check would misfire.
if __package__ and __package__.count(".") >= 2:
    from ...dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        decode_frames,
        run_diffusion,
    )
    from .frames_image import render_frames_to_image_batch
    from .socket_types import DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, DGEMMA_MODEL
else:
    from dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        decode_frames,
        run_diffusion,
    )
    from surfaces.comfyui.frames_image import render_frames_to_image_batch
    from surfaces.comfyui.socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_MODEL,
    )

# Event name for the live per-step push (plan.md Phase 3 (a)). Namespaced
# under the pack's own prefix — `send_sync`'s receiving side has no
# event-name whitelist (`loose-ends.md`), so any string works, but a
# collision with another pack's event name would silently cross-wire two
# unrelated `web/` extensions' `addEventListener` handlers.
DGEMMA_STEP_EVENT = "dgemma.sampler.step"

# `frames_image` render defaults (issue #21 rework) — fixed, not widgets; see
# the `frames_image` output's docstring above for why this is a display
# rendering rather than a sampling parameter.
FRAMES_IMAGE_WIDTH = 512
FRAMES_IMAGE_FONT_SIZE = 20
FRAMES_IMAGE_CAPTION_STEP_INDEX = True


def _build_on_frame(unique_id):
    """Build the live-push closure handed to `run_diffusion` as `on_frame`.

    Lives here, not in `dgemma/loop.py` (ADR-CDG-003): this is the one place
    in the pack allowed to import ComfyUI server infrastructure. `PromptServer`
    is imported lazily inside the closure (not at module top) so this module
    stays importable — and the sampler still runs — with no ComfyUI process
    alive (the normal pytest condition); a real live session is the only
    context where the import succeeds and the push actually fires.

    Display must never kill generation (review finding, 2026-07-05): the
    whole push — import, instance lookup, `send_sync` — is guarded, and any
    failure (no server, serialization error, dropped websocket) is logged
    and swallowed rather than propagated. The guard lives HERE, not in
    `dgemma/loop.py`'s hook site, deliberately: the engine's `on_frame`
    contract propagates callback exceptions (see `_FrameCollector`'s
    docstring — an engine that silently ate a user's analysis-callback
    error would be its own dishonesty), so the display-only closure guards
    itself at the layer that owns the display concern. A `send_sync` hiccup
    must not abort a multi-step 26B generation run.
    """

    def on_frame(frame) -> None:
        try:
            from server import PromptServer

            instance = PromptServer.instance
            if instance is None:
                return
            instance.send_sync(
                DGEMMA_STEP_EVENT,
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
                "DGemmaSampler live push failed (display only, generation continues): %s", exc
            )

    return on_frame


class DGemmaSampler:
    """Drives the denoising loop for one prompt; EB params/seed/thinking are
    widgets (P2)."""

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
                # EXPERIMENTAL (issue #22 honesty finding, in the widget
                # itself, not just a docstring): the injected system-turn
                # path is pinned by tests/test_chat_template_thinking.py to
                # be exactly one token short of native
                # `enable_thinking=True` (the template's `| trim` eats the
                # newline after `<|think|>`, id 107) — the ONLY reachable
                # path through `pipeline.__call__`, see dgemma.loop's
                # `thinking` docstring. Token parity is structurally
                # unreachable via message content. Behavioral impact of that
                # one-token gap is UNVERIFIED — no E2E thinking-mode run has
                # been done (needs the real 26B weights on GPU). This toggle
                # ships as a documented, honest experiment, not a confirmed
                # feature.
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "EXPERIMENTAL: injects the <|think|> control token via a "
                            "system turn, one token short of native enable_thinking=True "
                            "(structurally unreachable gap, see dgemma.loop docstring). "
                            "Behavioral effect unverified — no E2E run on real weights yet."
                        ),
                    },
                ),
            },
            "hidden": {
                # Standard ComfyUI hidden-input idiom (grepped against the
                # live install: tests/execution/testing_nodes/.../
                # specific_tests.py) — the node's own graph id, so the
                # per-step live push (P3 (a)) can be routed to the right
                # node's widget rather than broadcast anonymously.
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, "STRING", "IMAGE")
    RETURN_NAMES = ("text", "canvas_state", "canvas_trace", "frames", "images")
    # `frames_image` is a single stacked (N, H, W, 3) batch tensor, NOT a
    # list — False here, unlike `frames`' True (see this module's docstring:
    # a list would fan out per-frame and break PreviewImage/SaveAnimatedWEBP/VHS).
    OUTPUT_IS_LIST = (False, False, False, True, False)
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
            on_frame=_build_on_frame(unique_id),
        )
        frames = decode_frames(model.processor, canvas_trace.frames)
        # Per-image canvas-index key (ADR-CDG-009 §2, #35 F7): one canvas_idx
        # per decoded frame, parallel to `frames`, so the flipbook caption is
        # the N-canvas `canvas k/N · step i/M` form keyed per image rather than
        # a flat running index reconstructed by a fragile 1:1 zip.
        canvas_indices = [frame.canvas_idx for frame in canvas_trace.frames]
        frames_image = render_frames_to_image_batch(
            frames,
            width=FRAMES_IMAGE_WIDTH,
            font_size=FRAMES_IMAGE_FONT_SIZE,
            caption_step_index=FRAMES_IMAGE_CAPTION_STEP_INDEX,
            canvas_indices=canvas_indices,
        )
        return (text, canvas_state, canvas_trace, frames, frames_image)
