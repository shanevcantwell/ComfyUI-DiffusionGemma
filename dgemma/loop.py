"""dgemma/loop.py — the denoising-loop spine (ADR-CDG-004 drive seam).

Drives a preloaded `DiffusionGemmaForBlockDiffusion` (from `dgemma/model.py`)
through `diffusers.DiffusionGemmaPipeline` + `EntropyBoundScheduler`, per
ADR-CDG-004. Per-step frames are the loop's native contract from day one
(plan.md, `dgemma/loop.py` per-module notes): P1 keeps only the last frame
(`keep_frames="last"`), but the collection seam iterates every step
regardless, so P2 (knobs) and P3 (instrumentation) grow the same generator
without a reshape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from diffusers import DiffusionGemmaPipeline, EntropyBoundScheduler

from .types import CanvasState, DGemmaModel, DiffusionFrame

# Grounded defaults (CLAUDE.md / plan.md — first local run, Q4_K_M).
DEFAULT_NUM_INFERENCE_STEPS = 48
DEFAULT_T_MIN = 0.4
DEFAULT_T_MAX = 0.8
DEFAULT_ENTROPY_BOUND = 0.1
DEFAULT_GEN_LENGTH = 256


class DGemmaPipeline(DiffusionGemmaPipeline):
    """`DiffusionGemmaPipeline` subclass widening the per-step callback allowlist.

    The ONLY change from the base pipeline: `_callback_tensor_inputs` gains
    `"scheduler_output"`. The base class allowlist is `["canvas", "logits"]`
    (`pipeline_diffusion_gemma.py:76`); `check_inputs` validates
    `callback_on_step_end_tensor_inputs` against `self._callback_tensor_inputs`
    (`:155-161`), and the callback-kwargs extraction is generic —
    `callback_kwargs[k] = locals()[k]` (`:404-405`) — not a hardcoded
    two-key dispatch. Widening the allowlist here is therefore enough to hand
    the callback the full scheduler `.step()` output object (`accepted_index`,
    `sampled_probs`, `pred_logits`, ...) with no method override needed
    (ADR-CDG-004, resolved open question (a)).

    Caveat carried from that resolution: `"accepted_index"` alone is NOT a
    valid key — it is not a bound local in `__call__`'s scope. Only the
    `scheduler_output` container is.
    """

    _callback_tensor_inputs = ["canvas", "logits", "scheduler_output"]


def anneal_temperature(
    step_idx: int, num_inference_steps: int, t_min: float, t_max: float
) -> tuple[float, float]:
    """Replicate `EntropyBoundScheduler.step()`'s inlined anneal formula.

    Source: `scheduling_entropy_bound.py:153-155` (installed diffusers
    0.39.0) — the formula is inlined directly in `step()`, not exposed on
    `EntropyBoundSchedulerOutput`, so this dgemma layer recomputes it from the
    same inputs rather than reading it off the scheduler.

    Returns `(t, temperature)` where `t` is the normalized schedule fraction
    (1.0 at the hottest/first step, decreasing toward but not reaching 0) and
    `temperature = t_min + (t_max - t_min) * t`.
    """
    t = (num_inference_steps - step_idx) / num_inference_steps
    temperature = t_min + (t_max - t_min) * t
    return t, temperature


@dataclass
class _FrameCollector:
    """Per-step frame collector driving `callback_on_step_end`.

    Pure with respect to the diffusers pipeline: reads only the callback's
    own contract (`pipe, global_step, step_idx, callback_kwargs`) plus the
    scheduler config values needed for `anneal_temperature`, so it is
    unit-testable with a fake `scheduler_output` and no real pipeline
    (`tests/test_frames.py`).

    `keep_frames="last"` (P1 default) retains only the most recent frame —
    memory policy, not a change in what gets computed per step; `"all"`
    retains every frame (the seam P3's `CanvasTrace` grows into). `steps_used`
    counts every step regardless of retention policy.

    `canvas_idx` tracking: the pipeline's `step_idx` resets to 0 for each
    canvas/block (inner denoising loop nested in the outer canvas loop,
    `pipeline_diffusion_gemma.py:318,356`), and the callback contract carries
    no block coordinate of its own — so the collector infers it: a
    non-increasing `step_idx` between consecutive callbacks means a new block
    began. Detection is `step_idx <= previous`, not `step_idx == 0`, so a
    future mid-schedule start (variation runs, `loose-ends.md`) whose first
    step_idx is nonzero still registers as a new block.
    """

    num_inference_steps: int
    t_min: float
    t_max: float
    keep_frames: Literal["last", "all"] = "last"
    frames: list[DiffusionFrame] = field(default_factory=list)
    steps_used: int = 0
    _canvas_idx: int = -1
    _prev_step_idx: int | None = None

    def on_step_end(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict:
        """`callback_on_step_end(pipe, global_step, step_idx, callback_kwargs)`.

        Pure capture (P1): never overwrites the canvas, so it always returns
        `{}` — `callback_outputs.pop("canvas", canvas)` at the call site
        (`pipeline_diffusion_gemma.py:407`) then leaves the canvas unchanged.
        Mid-loop constraint injection (P5) is a different callback that
        returns `{"canvas": ...}`.

        Raises `ValueError` on a zero-length block: `accepted_index` with
        block dim 0 would make the per-example mean NaN, and a NaN
        committed_fraction would silently read as not-converged downstream —
        degenerate input is surfaced, not laundered into a validity field.
        """
        scheduler_output = callback_kwargs["scheduler_output"]
        canvas = callback_kwargs["canvas"]

        accepted_index = scheduler_output.accepted_index
        if accepted_index.shape[-1] == 0:
            raise ValueError(
                "Degenerate scheduler_output: accepted_index has block length 0 "
                f"(shape {tuple(accepted_index.shape)}); committed_fraction would be NaN."
            )

        if self._prev_step_idx is None or step_idx <= self._prev_step_idx:
            self._canvas_idx += 1
        self._prev_step_idx = step_idx

        t, temperature = anneal_temperature(step_idx, self.num_inference_steps, self.t_min, self.t_max)
        # Mean over the block dim ONLY — one fraction per example, never a
        # batch-blended scalar (review finding, 2026-07-05).
        committed_per_example = tuple(accepted_index.float().mean(dim=-1).tolist())

        frame = DiffusionFrame(
            canvas_idx=self._canvas_idx,
            step_idx=step_idx,
            t=t,
            temperature=temperature,
            committed_fraction_per_example=committed_per_example,
            canvas=canvas,
        )
        self.steps_used += 1
        if self.keep_frames == "last":
            self.frames[:] = [frame]
        else:
            self.frames.append(frame)
        return {}


def derive_canvas_state(*, text: str, canvas_ids: Any, frames: list[DiffusionFrame], steps_used: int) -> CanvasState:
    """Derive `CanvasState`'s validity fields from the captured frames.

    See `CanvasState.converged`'s docstring for what "converged" honestly
    does and does not claim.
    """
    if not frames:
        raise RuntimeError("No frames captured — the denoising callback never fired.")
    last = frames[-1]
    return CanvasState(
        text=text,
        canvas_ids=canvas_ids,
        converged=last.committed_fraction >= 1.0,
        committed_fraction=last.committed_fraction,
        steps_used=steps_used,
    )


def run_diffusion(
    dgemma_model: DGemmaModel,
    prompt: str,
    *,
    seed: int | None = None,
    gen_length: int = DEFAULT_GEN_LENGTH,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    entropy_bound: float = DEFAULT_ENTROPY_BOUND,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    keep_frames: Literal["last", "all"] = "last",
) -> tuple[str, CanvasState]:
    """Drive one prompt through the block-diffusion denoising loop.

    Constructs `EntropyBoundScheduler` directly with the entropy/temperature
    config (`entropy_bound`, `t_min`, `t_max`, `num_inference_steps`) — these
    live on the scheduler config, NOT on the pipeline's `__call__` (ADR-CDG-004:
    the pipeline only forwards `generator`/`mask_token_id`/`temperature` to
    `scheduler.step()`, filtered by that scheduler's own signature, and
    `EntropyBoundScheduler.step()` doesn't accept `mask_token_id` or
    `temperature` at all — it anneals its own). Wraps the loaded model in
    `DGemmaPipeline` (direct-constructor idiom, not `.from_pretrained`, since
    the model is already loaded).

    `confidence_threshold`/`stability_threshold`/`eos_early_stop` are left at
    the pipeline's own defaults (0.005 / 1 / True — already the grounded
    defaults, CLAUDE.md) rather than passed explicitly; P2 promotes them to
    widgets.

    Returns `(text, CanvasState)` — never a bare string (ADR-CDG-001 Addendum).
    """
    scheduler = EntropyBoundScheduler(
        entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
    )
    pipeline = DGemmaPipeline(model=dgemma_model.model, scheduler=scheduler, processor=dgemma_model.processor)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=dgemma_model.device).manual_seed(seed)

    collector = _FrameCollector(
        num_inference_steps=num_inference_steps, t_min=t_min, t_max=t_max, keep_frames=keep_frames
    )

    output = pipeline(
        prompt=prompt,
        gen_length=gen_length,
        num_inference_steps=num_inference_steps,
        generator=generator,
        callback_on_step_end=collector.on_step_end,
        callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
    )

    text = output.texts[0]
    canvas_ids = output.sequences[0]
    canvas_state = derive_canvas_state(
        text=text, canvas_ids=canvas_ids, frames=collector.frames, steps_used=collector.steps_used
    )
    return text, canvas_state
