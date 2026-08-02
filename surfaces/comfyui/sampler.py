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
already returns). `sample()` hands `live_view.build_on_frame(unique_id)`
(issue #188 ONE-MINT fix — previously a closure built inline here) to
`run_diffusion` as `on_frame` — the ADR-CDG-003-respecting way to let a live
view exist without `dgemma/loop.py` ever importing ComfyUI. `PromptServer`
is imported lazily, inside that shared closure, guarded so its absence (the
normal pytest/headless condition — this pack has no `comfy`/`server`
dependency, see `tests/test_seam.py`) degrades to a no-op live push rather
than crashing the sampler; everything else (`text`, `canvas_state`,
`canvas_trace`) proceeds unchanged either way.

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

**Metadata banner (issue #84, DECISION S-1, implementer's call — no new
widget):** `_build_frame_metadata` builds one `FrameMetadata` per decoded
frame from `canvas_trace.frames` (mirrors the `canvas_indices` construction
immediately above it) and threads it into `render_frames_to_image_batch` as
`frame_metadata=`, always on — operator requirement (a) asks to "draw all
available frame metadata into the flipbook," and every field
`FrameMetadata` needs (`t`/`temperature`/`committed_fraction`) is already
unconditionally populated on `DiffusionFrame`, so there is no meaningful
"banner off" state to gate behind a widget the way `thinking` gates an
experimental path. `mean_entropy` alone can render `—` per-frame
(`DiffusionFrame.entropy`'s own additive-optional discipline, ADR-CDG-014)
without that absence needing a whole-banner toggle.

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
`surfaces/comfyui/socket_types.py`. The live-view event name and the hidden-
input capability sentinel come from `surfaces/comfyui/live_view.py` (issue
#188 ONE-MINT fix) — the event name is not a ComfyUI socket type either, so
it stays out of the `socket_types` mint, same as before this fix.

**Named trap (plan.md Risks): this MUST NOT touch `comfy.utils.ProgressBar`'s
`preview=` slot.** That path is structurally image-typed downstream
(`server.py:1293-1301`, `ProgressBar.update_absolute` -> `send_image`, which
calls `image.save(...)` on whatever it's handed) and throws on text. Text
goes out its own custom event via `send_sync`, never through `preview=`.
This is a review-gate risk, not a test-enforced one — there is no clean unit
test for "this code path never calls the wrong API" (plan.md Risks).

**A sixth output, `run_config` (issue #72, Option A / D-1):** a
`DGEMMA_RUN_CONFIG`-typed `consumers.run_log.RunConfig` bundle assembled
from widget args and `model` attributes this method already holds (`seed`,
every knob, `model.repo_id`/`quant`/`device`/`dtype`, `prompt`) — a pure
unpack-and-forward, no new logic (ARCHITECTURE.md rule 2 stays intact, AC-8).
`run_diffusion`'s returned `CanvasTrace` does not carry these values (G-1:
`_build_result` never receives seed/confidence/gen_length/thinking/prompt),
so the sampler is the sole position that can assemble a correct header —
this output exists so `DGemmaRunLogWriter`
(`surfaces/comfyui/run_log_writer.py`) can build one without re-deriving it.
Wiring this output costs nothing when unwired (ComfyUI only computes what a
downstream node actually consumes at the socket level; an unconnected output
is simply not read) and stays surface-side per Option A's rejection of
widening the core's `_build_result` signature for a downstream-only value.

**Cancellation wiring (issue #140 sampler half, closes #38's surface gap):**
`_build_should_cancel` connects `comfy.model_management.processing_interrupted()`
to `run_diffusion`'s pre-existing `should_cancel` parameter, the same lazy-import
shape `live_view.build_on_frame` already uses for `PromptServer` — `comfy` is imported
inside the closure, never at module top, so this module keeps importing (and
`sample()` keeps running) with zero ComfyUI present. The engine-side seam
(`dgemma/composite.py`'s `_CancellationParticipant`, `dgemma/loop.py`'s
`should_cancel` param and `DiffusionCancelled` handling) already existed and
is already tested (`tests/test_run_diffusion_cancel.py`) — this is only the
surface connection ADR-CDG-003 reserves for this layer. No new cancellation
semantics are introduced here.
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
        KNOB_DOCS,
        run_diffusion,
    )
    from .emission import build_sampler_shaped_outputs
    from .live_view import LIVE_VIEW_HIDDEN_INPUT, build_on_frame
    from .socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
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
    from surfaces.comfyui.live_view import LIVE_VIEW_HIDDEN_INPUT, build_on_frame
    from surfaces.comfyui.socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_MODEL,
        DGEMMA_RUN_CONFIG,
    )

def _build_should_cancel():
    """Build the cancellation predicate handed to `run_diffusion` as
    `should_cancel` (issue #140 sampler half, closes #38's surface-wiring
    gap). `dgemma.composite._CancellationParticipant` and `run_diffusion`'s
    own `should_cancel` seam already exist and are already tested in
    isolation (`tests/test_run_diffusion_cancel.py`,
    `tests/test_step_end_composite.py`) — this closure is ONLY the surface
    connection ADR-CDG-003 reserves for this layer: `dgemma/loop.py` stays
    ComfyUI-agnostic and never imports `comfy`, exactly like
    `live_view.build_on_frame` (imported above).

    `comfy.model_management` is imported lazily, inside the returned closure
    (not at module top, and not even inside this builder) — same rationale
    as `live_view.build_on_frame`'s lazy `from server import PromptServer`:
    this module must keep importing with zero ComfyUI present (the normal
    pytest/headless condition, `tests/test_seam.py`,
    `tests/test_dual_context_import.py`). Unlike the display-only `on_frame`
    push, a failure here degrades to "no cancellation wiring" (`False`) —
    logged, not raised — since a live-push hiccup and an interrupt-check
    hiccup carry the same non-negotiable rule (display/interrupt plumbing
    must never kill generation), so the closure never propagates an
    unexpected exception into `run_diffusion`'s per-step composite.

    Cancellation semantics belong entirely to the engine (`dgemma/composite.py`'s
    `_CancellationParticipant` raises `DiffusionCancelled`, caught inside
    `run_diffusion` to return the partial `(text, CanvasState, CanvasTrace)`
    already captured, per #38's "a cancelled experiment run is still data"
    clause) — this closure adds no new semantics, it only supplies the
    predicate.
    """

    def should_cancel() -> bool:
        try:
            import comfy.model_management as model_management

            return bool(model_management.processing_interrupted())
        except ImportError:
            return False  # No live ComfyUI process (e.g. pytest) — never cancel.
        except Exception as exc:  # noqa: BLE001 — deliberate breadth: see docstring.
            logging.warning(
                "DGemmaSampler interrupt check failed (treating as not-cancelled): %s", exc
            )
            return False

    return should_cancel


class DGemmaSampler:
    """Drives the denoising loop for one prompt; EB params/seed/thinking are
    widgets (P2)."""

    DESCRIPTION = (
        "Drives the denoising loop for one prompt. Six outputs: text, "
        "canvas_state, canvas_trace, frames, images, run_config (wire the "
        "last four into DGemmaRunLogWriter for a run log, or canvas_trace "
        "into DGemmaTrace/DGemmaTokenTrace for analysis). Use DGemmaDenoise "
        "instead if you need to condition on a KV-cache."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    DGEMMA_MODEL,
                    {"tooltip": "Loaded DiffusionGemma model (from DGemmaLoader)."},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "The user turn to generate a response to.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": KNOB_DOCS["seed"],
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": DEFAULT_NUM_INFERENCE_STEPS,
                        "min": 1,
                        "max": 1024,
                        "tooltip": KNOB_DOCS["num_inference_steps"],
                    },
                ),
                "t_min": (
                    "FLOAT",
                    {
                        "default": DEFAULT_T_MIN,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": KNOB_DOCS["t_min"],
                    },
                ),
                "t_max": (
                    "FLOAT",
                    {
                        "default": DEFAULT_T_MAX,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": KNOB_DOCS["t_max"],
                    },
                ),
                "entropy_bound": (
                    "FLOAT",
                    {
                        "default": DEFAULT_ENTROPY_BOUND,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": KNOB_DOCS["entropy_bound"],
                    },
                ),
                "confidence": (
                    "FLOAT",
                    {
                        "default": DEFAULT_CONFIDENCE,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.001,
                        "tooltip": KNOB_DOCS["confidence"],
                    },
                ),
                "gen_length": (
                    "INT",
                    {
                        "default": DEFAULT_GEN_LENGTH,
                        "min": 1,
                        "max": 8192,
                        "tooltip": KNOB_DOCS["gen_length"],
                    },
                ),
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
                # feature. Tooltip text sourced from the KNOB_DOCS mint
                # (`dgemma/loop.py`) — same ONE-MINT discipline as every
                # other widget here, not a special case.
                "thinking": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": KNOB_DOCS["thinking"],
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
                # Live-view capability sentinel (issue #188 ONE-MINT fix):
                # merged from `live_view.LIVE_VIEW_HIDDEN_INPUT` rather than
                # inlined, so this node's presence in ComfyUI's
                # `/object_info` payload (`nodeData.input.hidden`) is what
                # `web/live_view.js` keys its decoration/listen logic off of
                # — no hardcoded node-type list on the JS side.
                **LIVE_VIEW_HIDDEN_INPUT,
            },
        }

    RETURN_TYPES = ("STRING", DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, "STRING", "IMAGE", DGEMMA_RUN_CONFIG)
    RETURN_NAMES = ("text", "canvas_state", "canvas_trace", "frames", "images", "run_config")
    # `frames_image` is a single stacked (N, H, W, 3) batch tensor, NOT a
    # list — False here, unlike `frames`' True (see this module's docstring:
    # a list would fan out per-frame and break PreviewImage/SaveAnimatedWEBP/VHS).
    # `run_config` (issue #72) is one plain `RunConfig` object, not a list.
    OUTPUT_IS_LIST = (False, False, False, True, False, False)
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
            on_frame=build_on_frame(unique_id),
            should_cancel=_build_should_cancel(),
        )
        # Output construction (RunConfig build, frames decode, images
        # render, tuple assembly) is the shared emission helper (issue
        # #166) — `DGemmaDenoise` calls the identical body, byte-identical
        # outputs to before this extraction.
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
