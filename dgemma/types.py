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
    dtype: str  # human-readable compute dtype label — always "bfloat16" since issue #18
    repo_id: str
    quant: str  # "none" (issue #18 — bnb nf4/int8 removed, misled on this architecture)


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
class CanvasTrace:
    """The complete per-step record of one `run_diffusion` call — the
    `CANVAS_TRACE` socket payload (ADR-CDG-001), consumed post-hoc by
    `DGemmaTrace` (plan.md Phase 3 (b)).

    ADR-CDG-001's addendum on scheduler-relative commit semantics is
    non-negotiable here: a commit mask (or, as here, a per-step
    `committed_fraction` reading) means something different depending on
    which scheduler minted it — a ratchet under `BlockRefinementScheduler`,
    a stateless per-step reading under `EntropyBoundScheduler`. A
    `CanvasTrace` that carried `frames` alone, without saying which
    scheduler produced them and with what config, would be a lying payload
    the same way a disguised `SIGMAS` tensor is. `scheduler_name` +
    `scheduler_config` are therefore not optional metadata — they are the
    mint identity that gives the frames' commit readings their meaning.

    Frames are already self-keyed (`DiffusionFrame`'s own
    `(canvas_idx, step_idx, t, temperature)` identity, see that class's
    docstring) — this type carries the collection as-is, no new keying
    logic.
    """

    frames: list[DiffusionFrame]
    scheduler_name: str
    scheduler_config: dict


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

    `canvas_ids`/`text` are post thought-channel-excision (P2, issue #8): the
    model emits an id-100/id-101 `<|channel>...<channel|>` frame at turn
    start (empty when `thinking=False`, possibly non-empty when
    `thinking=True`) that upstream `skip_special_tokens=True` decode does not
    fully strip — `dgemma.loop.excise_thought_channel` removes every
    well-formed span from the canvas ids before either field is derived, so
    neither payload can leak a frame (ADR-CDG-001 payload-contamination
    discipline).
    """

    text: str
    canvas_ids: Any  # torch.LongTensor, final canvas token ids (prompt stripped, thought-channel excised)
    converged: bool
    committed_fraction: float
    steps_used: int
    thought: str | None = None
    """Decoded content of the excised thought channel(s), or `None` when the
    channel was empty (the `thinking=False` common case — the model card
    notes an empty channel "might still be emitted" even with thinking off)
    or absent entirely; multiple non-empty channels are joined with a blank
    line. Not surfaced on the `STRING` payload (ADR-CDG-001: payloads mean
    what they say — the answer text is the answer text); this is the
    "natural slot" decision for issue #8's optional thought-surfacing ask,
    landing on `CanvasState` since it already carries validity/diagnostic
    readouts alongside the answer."""

    stray_thought_delimiter: bool = False
    """`True` iff the canvas held an unmatched `<|channel>` start delimiter
    past the head of the generated region — a malformed frame
    `excise_thought_channel` deliberately did NOT excise, because
    excise-to-end there would silently destroy answer text (review finding,
    2026-07-05). The answer `STRING` keeps all surrounding text (the
    delimiter itself vanishes in `skip_special_tokens=True` decode); this
    flag is the validity-side signal that an anomalous frame remnant exists
    rather than letting the condition vanish."""

    turn_closed: bool = False
    """Issue #9's honesty-readout half: `True` iff `eos_token_id` was found
    committed somewhere in the (thought-excised) answer ids — the turn ran
    to a real stop, not just "the canvas is full of plausible-looking
    tokens." `False` covers BOTH of issue #9's named specimens: an
    all-thought/empty-answer turn (nothing to find EOS in) and a
    budget-truncated answer that hit `gen_length` mid-token with
    `converged=True, committed_fraction=1.0` — the exact gap `converged`
    alone cannot see (`CanvasState.converged`'s docstring already says it
    doesn't confirm EOS). `turn_closed` is deliberately independent of
    `converged`: a run can converge (every position locked in this step)
    while still not being turn_closed (it locked in on non-EOS filler
    because the canvas ran out first)."""

    answer_tokens: int = 0
    """Issue #9's companion honesty field: the count of (thought-excised)
    answer ids **before the first EOS** — the EOS itself and any trailing
    EOS/renoise fill run (a converged run pads the canvas tail with one;
    ~30 tokens observed live) are excluded, mirroring `_decode_ids`'s own
    trim, because a bare `len(canvas_ids)` would inflate the count by that
    padding and defeat the field's honesty purpose (review finding,
    2026-07-05). When no EOS is present, the full thought-excised length
    counts (every id is content the budget-truncated canvas actually
    holds); `0` when `canvas_ids` is unavailable (e.g. a unit test
    constructing `CanvasState` directly without a real canvas). Read
    alongside `turn_closed`: a small `answer_tokens` with
    `turn_closed=False` is the all-thought/empty-answer specimen; a large
    `answer_tokens` with `turn_closed=False` is the budget-truncated
    specimen — same field pair, different failure shape."""
