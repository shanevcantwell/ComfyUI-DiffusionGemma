"""dgemma/types.py — engine-native dataclasses (ADR-CDG-001's socket payloads).

ComfyUI-agnostic (ADR-CDG-003). A custom ComfyUI socket type is just a string
matched by equality; the object riding it is the corresponding dataclass here,
passed through untouched — so these dataclasses ARE the public contract, not
an implementation detail behind one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DGemmaModel:
    """A loaded DiffusionGemma model + its processor + device/dtype info.

    Rides the `DGEMMA_MODEL` socket. Load seam per ADR-CDG-002 (unchanged by
    ADR-CDG-004): `DiffusionGemmaForBlockDiffusion.from_pretrained()` +
    `AutoProcessor.from_pretrained()`.
    """

    model: Any  # transformers.DiffusionGemmaForBlockDiffusion
    processor: Any  # transformers processor (AutoProcessor.from_pretrained(...))
    device: str
    dtype: str  # human-readable compute dtype label, e.g. "float16" (NF4 compute dtype)
    repo_id: str
    quant: str  # "nf4" | "int8" | "none" — how the checkpoint was loaded


@dataclass
class DiffusionFrame:
    """One denoising step's captured state.

    Keyed on `(canvas_idx, step_idx, t, temperature)` — block identity plus
    the absolute position in the noise schedule — never on loop index alone:

    - `step_idx` resets to 0 for every canvas/block (the denoising loop is
      nested inside the outer autoregressive canvas loop,
      `pipeline_diffusion_gemma.py:318,356`), so once `gen_length` exceeds
      the model's `canvas_length`, `(step_idx, t, temperature)` alone
      collides across blocks. `canvas_idx` says *which block's* anneal a
      frame belongs to; it is also part of the block identity a resumable
      save-state needs (ADR-CDG-005's schedule-position field, extended with
      the outer-loop coordinate).
    - a run that starts mid-schedule (a variation/renoise run,
      `loose-ends.md`) would make loop-index keying silently incomparable
      across runs (plan.md P3 rationale) — hence absolute `t`, not list
      position.

    `t` and `temperature` replicate `EntropyBoundScheduler.step()`'s inlined
    anneal formula (`scheduling_entropy_bound.py:153-155`, installed diffusers
    0.39.0) — the scheduler does not expose either value on its step output,
    so `dgemma.loop.anneal_temperature` recomputes them from the same inputs:

        fraction = (num_inference_steps - step_idx) / num_inference_steps
        temperature = t_min + (t_max - t_min) * fraction

    `t` is that `fraction`: the normalized position in the schedule, 1.0 at
    the hottest (first) step, decreasing toward (but not reaching exactly) 0
    at the last step. `temperature` is the resulting annealed sampling
    temperature. Both ride the frame so a reader never has to reconstruct one
    from the other against the scheduler config.

    Commit info is **per-example**: `committed_fraction_per_example[i]` is
    the fraction of example i's canvas positions accepted this step (mean of
    `scheduler_output.accepted_index[i]` over the block dim only — never the
    batch dim, which would silently blend examples). The scalar
    `committed_fraction` property is a batch_size==1 convenience and raises
    on batched frames rather than inventing a blended number.
    """

    canvas_idx: int
    step_idx: int
    t: float
    temperature: float
    committed_fraction_per_example: tuple[float, ...]
    canvas: Any  # torch.LongTensor snapshot of the canvas after this step (scheduler_output.prev_sample)

    @property
    def committed_fraction(self) -> float:
        """Scalar commit fraction — defined only for single-example frames."""
        if len(self.committed_fraction_per_example) != 1:
            raise ValueError(
                "committed_fraction is a batch_size==1 convenience; this frame has "
                f"{len(self.committed_fraction_per_example)} examples — use "
                "committed_fraction_per_example."
            )
        return self.committed_fraction_per_example[0]


@dataclass
class CanvasState:
    """Validity readout riding alongside the decoded `STRING` output.

    ADR-CDG-001 Addendum ("time-axis lying payload"): a bare string cannot say
    whether the canvas it was decoded from actually finished denoising — with
    too few steps or too wide an entropy bound, "finished-looking" text can
    still contain uncommitted renoise garbage. These fields are the evidence
    that keeps the `STRING` honest about its own completion state.

    `converged` is defined **honestly, not aspirationally**: `True` iff the
    *last captured frame's* `committed_fraction == 1.0` — every canvas
    position was simultaneously accepted under the entropy bound on the final
    denoising step this run captured. This is a per-step reading, not a
    ratchet: under `EntropyBoundScheduler` (this pack's P1 default scheduler)
    there is no persistent commit state to consult (ADR-CDG-001 Addendum,
    "scheduler-relative commit semantics") — `converged` says "the schedule
    bottomed out by the last step we saw," not "every position was locked in
    and has stayed locked since." It does NOT independently confirm EOS was
    emitted or that the pipeline's own `confidence_threshold` early-stop
    fired — that decision is internal to `DiffusionGemmaPipeline.__call__`'s
    loop and is not surfaced to `callback_on_step_end`. Do not read more into
    `converged` than the entropy-bound reading it actually is.

    Scope: single-example. P1's `run_diffusion` drives one prompt (batch 1);
    deriving a `CanvasState` from a batched frame raises via
    `DiffusionFrame.committed_fraction` rather than blending examples.
    """

    text: str
    canvas_ids: Any  # torch.LongTensor, final canvas token ids (prompt stripped)
    converged: bool
    committed_fraction: float
    steps_used: int
