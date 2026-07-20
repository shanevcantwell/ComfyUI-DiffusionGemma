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
`surfaces/comfyui/socket_types.py`. `DGEMMA_STEP_EVENT` below is NOT part of
that mint — it's a WebSocket event name, not a ComfyUI socket type.

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
    from ...consumers.run_log import RunConfig
    from ...dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        KNOB_DOCS,
        decode_frames,
        run_diffusion,
    )
    from .frames_image import FrameMetadata, render_frames_to_image_batch
    from .socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_MODEL,
        DGEMMA_RUN_CONFIG,
    )
else:
    from consumers.run_log import RunConfig
    from dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        KNOB_DOCS,
        decode_frames,
        run_diffusion,
    )
    from surfaces.comfyui.frames_image import FrameMetadata, render_frames_to_image_batch
    from surfaces.comfyui.socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_MODEL,
        DGEMMA_RUN_CONFIG,
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


def _build_frame_metadata(frames: list) -> list:
    """Build the per-image `FrameMetadata` key (issue #84, DECISION S-1)
    from `canvas_trace.frames`, threaded into `render_frames_to_image_batch`
    the SAME way `canvas_indices` already is (parallel list, one entry per
    decoded frame, built here rather than inside the render helper — the
    render helper stays plain-data-in, ADR-CDG-003).

    `mean_entropy` is a scalar reduction of `DiffusionFrame.entropy`
    (`float32[canvas_len]` or `None`) — cheap (`~1 KB/step` tensor, ADR-CDG-014
    Decision 3) and the one non-trivial computation in this function; still
    not denoising-loop logic (rule 2), just an adapter-side reduction of a
    value the core already computed. `None` propagates as `None` (never a
    fabricated `0.0`), matching `DiffusionFrame.entropy`'s own "`None` means
    not captured this run" discipline."""
    metadata = []
    for frame in frames:
        mean_entropy = float(frame.entropy.mean().item()) if frame.entropy is not None else None
        metadata.append(
            FrameMetadata(
                step_idx=frame.step_idx,
                total_steps=len(frames),
                t=frame.t,
                temperature=frame.temperature,
                committed_fraction=frame.committed_fraction_per_example[0]
                if len(frame.committed_fraction_per_example) == 1
                else None,
                mean_entropy=mean_entropy,
            )
        )
    return metadata


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
            on_frame=_build_on_frame(unique_id),
        )
        frames = decode_frames(model.processor, canvas_trace.frames)
        # Per-image canvas-index key (ADR-CDG-009 §2, #35 F7): one canvas_idx
        # per decoded frame, parallel to `frames`, so the flipbook caption is
        # the N-canvas `canvas k/N · step i/M` form keyed per image rather than
        # a flat running index reconstructed by a fragile 1:1 zip.
        canvas_indices = [frame.canvas_idx for frame in canvas_trace.frames]
        # Per-image metadata key (issue #84, DECISION S-1): threaded the
        # same way as canvas_indices above — one FrameMetadata per decoded
        # frame, parallel to `frames`.
        frame_metadata = _build_frame_metadata(canvas_trace.frames)
        frames_image = render_frames_to_image_batch(
            frames,
            width=FRAMES_IMAGE_WIDTH,
            font_size=FRAMES_IMAGE_FONT_SIZE,
            caption_step_index=FRAMES_IMAGE_CAPTION_STEP_INDEX,
            canvas_indices=canvas_indices,
            frame_metadata=frame_metadata,
        )
        # `run_config` (issue #72, Option A / D-1): a plain unpack of args and
        # `model` attributes this method already holds — no re-derivation, no
        # new logic. This is the ONLY position holding seed+knobs+model-id
        # simultaneously (G-2), so it is assembled here rather than pushed
        # into `run_diffusion`'s core signature.
        run_config = RunConfig(
            prompt=prompt,
            model_repo_id=model.repo_id,
            seed=seed,
            num_inference_steps_requested=num_inference_steps,
            gen_length=gen_length,
            t_min=t_min,
            t_max=t_max,
            entropy_bound=entropy_bound,
            confidence=confidence,
            thinking=thinking,
            quant=model.quant,
            device=model.device,
            dtype=model.dtype,
        )
        return (text, canvas_state, canvas_trace, frames, frames_image, run_config)
