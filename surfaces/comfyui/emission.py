"""surfaces/comfyui/emission.py — shared output-construction helper for
`DGemmaSampler` and `DGemmaDenoise` (issue #166).

**Why this exists (#166 ratified scope):** the operator's scope decision
(issue #166 comment, 2026-07-30) adopts `DGemmaSampler`'s full six-output
signature (`text`, `canvas_state`, `canvas_trace`, `frames`, `images`,
`run_config`) verbatim as `DGemmaDenoise`'s contract too — "the sampler
output type for this version IS the sampler's existing socket set." Rather
than duplicate the RunConfig-build / frames-decode / images-render /
tuple-assembly block per node (the copy `DGemmaSampler.sample` already had
before this issue), that block is extracted ONCE here and both nodes call
it. `DGemmaSampler` is refactored to call this helper too — its outputs are
byte-identical to before this extraction (see `tests/test_loader_contract.py`
and `tests/test_socket_mint.py`, unchanged).

Surface-tier, not core (ADR-CDG-003 rule 2): this module renders `IMAGE`
tensors and assembles a ComfyUI-node-shaped return tuple — display/adapter
concerns, not denoising-loop logic. It imports `dgemma.loop.decode_frames`
and `consumers.run_log.RunConfig` (both already-consumed-by-surface
concerns; see `sampler.py`'s own docstring for why `RunConfig` lives in
`consumers/`, not `dgemma/`), plus this package's own `frames_image`
renderer. No new socket type is minted here — the six socket type strings
this helper's callers wire come from `socket_types.py` unchanged (rule 4,
`ONE-MINT`); this module only builds the plain-data VALUES those sockets
carry.

`_build_frame_metadata` and the `FRAMES_IMAGE_*` render defaults are moved
here verbatim from `sampler.py` (same docstrings retained on the function;
the fixed render params are simple module constants, not new knobs)."""
from __future__ import annotations

# Dual-context import, explicit package-depth gate — see
# surfaces/comfyui/loader.py for the full rationale (ComfyUI loader context
# vs. pytest/standalone). This module lives two levels under the pack root
# (surfaces/comfyui/), so the relative climb to dgemma/ and consumers/ is
# THREE dots — same depth as sampler.py/denoise.py. Gate is
# `__package__.count(".") >= 2`, not a bare dot-presence check.
if __package__ and __package__.count(".") >= 2:
    from ...consumers.run_log import RunConfig
    from ...dgemma.loop import decode_frames
    from .frames_image import FrameMetadata, render_frames_to_image_batch
else:
    from consumers.run_log import RunConfig
    from dgemma.loop import decode_frames
    from surfaces.comfyui.frames_image import FrameMetadata, render_frames_to_image_batch

# `frames_image` render defaults (issue #21 rework, moved here unchanged from
# `sampler.py`) — fixed, not widgets; a display rendering, not a sampling
# parameter, so both nodes share the identical fixed rendering rather than
# each picking its own.
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


def build_sampler_shaped_outputs(
    *,
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
    text: str,
    canvas_state,
    canvas_trace,
) -> tuple:
    """Build the shared six-output tuple (`text`, `canvas_state`,
    `canvas_trace`, `frames`, `images`, `run_config`) both `DGemmaSampler`
    and `DGemmaDenoise` return (issue #166 ratified scope) from a completed
    `run_diffusion` call's raw 3-tuple plus the widget args/model attributes
    the caller already holds.

    A pure unpack-and-forward (ARCHITECTURE.md rule 2, AC-8): no denoising
    logic, only decode/render/assemble of values `run_diffusion` and the
    caller's own widget args already produced. Moved here verbatim from
    `DGemmaSampler.sample` (issue #72's `run_config` assembly, issue #21's
    `frames`/`images` rendering) so `DGemmaDenoise` can call the identical
    body rather than a re-typed copy.
    """
    frames = decode_frames(model.processor, canvas_trace.frames)
    # Per-image canvas-index key (ADR-CDG-009 §2, #35 F7): one canvas_idx
    # per decoded frame, parallel to `frames`, so the flipbook caption is
    # the N-canvas `canvas k/N · step i/M` form keyed per image rather than
    # a flat running index reconstructed by a fragile 1:1 zip.
    canvas_indices = [frame.canvas_idx for frame in canvas_trace.frames]
    # Per-image metadata key (issue #84, DECISION S-1): threaded the same
    # way as canvas_indices above — one FrameMetadata per decoded frame,
    # parallel to `frames`.
    frame_metadata = _build_frame_metadata(canvas_trace.frames)
    images = render_frames_to_image_batch(
        frames,
        width=FRAMES_IMAGE_WIDTH,
        font_size=FRAMES_IMAGE_FONT_SIZE,
        caption_step_index=FRAMES_IMAGE_CAPTION_STEP_INDEX,
        canvas_indices=canvas_indices,
        frame_metadata=frame_metadata,
    )
    # `run_config` (issue #72, Option A / D-1): a plain unpack of args and
    # `model` attributes the caller already holds — no re-derivation, no
    # new logic. This is the ONLY position holding seed+knobs+model-id
    # simultaneously (G-2), so it is assembled here rather than pushed into
    # `run_diffusion`'s core signature.
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
    return (text, canvas_state, canvas_trace, frames, images, run_config)
