"""dgemma/capture.py — the per-step frame capture participant (ADR-CDG-018
Stage 3, issue #129's `dgemma/loop.py` decomposition).

`anneal_temperature`, `_build_pinned_mask`, `_FrameCollector` — extracted
verbatim from `dgemma/loop.py` (no behavior change — see
`tests/test_loop_golden_trace.py`, the decomposition's no-behavior-change
oracle). `_FrameCollector` is `dgemma.composite.StepEndComposite`'s `capture`
participant, which runs FIRST in the fixed order (`capture -> cancel ->
beta-rebuild -> pin -> walker`, ADR-CDG-010) — every other participant's
logits/canvas reads happen downstream of this module's own.

`loop.py` re-imports all three names, so every existing
`from dgemma.loop import _FrameCollector` / `anneal_temperature` /
`_build_pinned_mask` import site keeps working unchanged (the ratified
plan's facade ruling, issue #129).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

from .payloads import Constraints
from .types import DiffusionFrame


def anneal_temperature(
    step_idx: int, num_inference_steps: int, t_min: float, t_max: float
) -> tuple[float, float]:
    """Replicate `EntropyBoundScheduler.step()`'s inlined anneal formula.

    Source: `scheduling_entropy_bound.py:153-155` (installed diffusers
    0.39.0) — the formula is inlined directly in `step()`, not exposed on
    `EntropyBoundSchedulerOutput`, so this dgemma layer recomputes it from the
    same inputs rather than reading it off the scheduler.

    Enforcement surface for this replication (issue #35 R3 / PR #48 gate
    finding F-1): `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin`
    — `dgemma.compat`'s structural probe cannot see a formula-*body* change,
    so that test recovers the temperature the real installed scheduler's
    `step()` actually applied and asserts this function matches it. If you
    edit this formula, that test is the contract you are editing against.

    Returns `(t, temperature)` where `t` is the normalized schedule fraction
    (1.0 at the hottest/first step, decreasing toward but not reaching 0) and
    `temperature = t_min + (t_max - t_min) * t`.
    """
    t = (num_inference_steps - step_idx) / num_inference_steps
    temperature = t_min + (t_max - t_min) * t
    return t, temperature


def _build_pinned_mask(constraints: "Constraints | None", canvas: Any) -> Any | None:
    """Derive `DiffusionFrame.pinned_mask` from a validated `Constraints`
    payload (ADR-CDG-010 Decision 4, issue #64 Phase 2, gate correction A1).

    Static-from-`Constraints.pins` by construction: a boolean tensor shaped
    like one example's canvas (`canvas.shape[-1]`), `True` at every
    `pin.position`. Valid only because and only while pins are
    position-static (see `DiffusionFrame.pinned_mask`'s docstring for the
    full scope-guard reasoning) — no pin participant exists yet (Phase 3), so
    this reflects "which cells WOULD be pinned", not an observed write.

    `None` when `constraints` is `None` or carries no pins (`Constraints()`
    default, or an explicit `Constraints(pins=())`) — additive-optional
    discipline (ADR-CDG-014 Decision 1): absence, never an all-`False` mask
    standing in for "no pins".
    """
    if constraints is None or not constraints.pins:
        return None
    canvas_len = canvas.shape[-1]
    mask = torch.zeros(canvas_len, dtype=torch.bool)
    for pin in constraints.pins:
        mask[pin.position] = True
    return mask


@dataclass
class _FrameCollector:
    """Per-step frame collector driving `callback_on_step_end`.

    Pure with respect to the diffusers pipeline: reads only the callback's
    own contract (`pipe, global_step, step_idx, callback_kwargs`) plus the
    scheduler config values needed for `anneal_temperature`, so it is
    unit-testable with a fake `scheduler_output` (and, for the denominator,
    a fake scheduler exposing a `num_inference_steps` attribute) and no real
    pipeline (`tests/test_frames.py`).

    `num_inference_steps` (issue #20): NOT the user-requested value — a
    scheduler-like object read *lazily*, once per callback, via
    `.num_inference_steps`. Grounded against the installed diffusers 0.39.0
    pipeline (`pipeline_diffusion_gemma.py:280-297`): `set_timesteps(
    predictor_steps, ...)` runs at pipeline entry, before the per-step loop
    (`:356`) that fires `callback_on_step_end`, and `EntropyBoundScheduler.
    set_timesteps` (`scheduling_entropy_bound.py:87-91`) — and
    `BlockRefinementScheduler.set_timesteps`, `scheduling_block_refinement.py
    :83-100` — both reassign `self.num_inference_steps = num_inference_steps`
    there, the exact attribute `step()`'s inlined anneal formula divides by
    (`scheduling_entropy_bound.py:153`). So by the first callback the
    scheduler's own attribute already holds the *effective* denominator
    (`predictor_steps`, which differs from the user's `num_inference_steps`
    whenever a corrector scheduler folds `corrector_steps` sweeps into the
    same budget — `pipeline_diffusion_gemma.py:284-290`). Reading it lazily
    (not caching the value at collector-construction time, before the
    pipeline has called `set_timesteps`) is required: the collector is built
    by `run_diffusion` (`dgemma/loop.py`) before `pipeline(...)` runs, so a
    constructor-time snapshot would still be the stale user-requested count.
    Plain `EntropyBoundScheduler` (no `corrector_steps`) leaves
    `predictor_steps == num_inference_steps`, so this path is unchanged for
    today's only scheduler — the bug is latent, not yet observable, exactly
    per ADR-CDG-001's greenfield-exception framing (CLAUDE.md).

    `keep_frames="last"` (P1 default) retains only the most recent frame —
    memory policy, not a change in what gets computed per step; `"all"`
    retains every frame (the seam P3's `CanvasTrace` grows into). `steps_used`
    counts every step regardless of retention policy.

    `on_frame`, when given, is invoked once per captured step with the
    freshly built `DiffusionFrame` — regardless of `keep_frames` (a caller
    watching every step live still wants a callback even under `"last"`
    retention, which only governs what's kept afterward). Pure w.r.t.
    ComfyUI (ADR-CDG-003): this collector never imports or touches
    `PromptServer` itself — that's `nodes/sampler.py`'s closure, built and
    passed in from the node layer. `on_frame` runs after the retention
    policy is applied, so a callback exception never loses the frame itself.

    Engine contract on `on_frame` exceptions (deliberate, review finding
    2026-07-05): they PROPAGATE. The engine does not swallow a caller's
    callback error — a user's analysis callback silently eaten here would
    be its own dishonesty. A callback whose failure must not kill the run
    (e.g. a display-only push) guards itself at its own layer; that is what
    `nodes/sampler.py`'s live-push closure does.

    `canvas_idx` tracking: the pipeline's `step_idx` resets to 0 for each
    canvas/block (inner denoising loop nested in the outer canvas loop,
    `pipeline_diffusion_gemma.py:318,356`), and the callback contract carries
    no block coordinate of its own — so the collector infers it: a
    non-increasing `step_idx` between consecutive callbacks means a new block
    began. Detection is `step_idx <= previous`, not `step_idx == 0`, so a
    future mid-schedule start (variation runs, `loose-ends.md`) whose first
    step_idx is nonzero still registers as a new block.

    **Effective-knob telemetry (ADR-CDG-011 clause 7, issue #64 Phase 2):**
    `entropy_bound`/`t_min`/`t_max` are read fresh off `self.scheduler.config`
    on every callback — the same "never cached, always effective" discipline
    issue #20 already established for `num_inference_steps` above, extended
    to the three walker-mutable knobs (ADR-CDG-011's `MUTABLE_TARGETS`). No
    walker exists yet to write through them (Phase 4, `NOT-YET-IMPLEMENTED`),
    but reading live now — rather than only once a walker lands — is what
    makes a future walker bug that silently fails to write through visible
    in the trace the day it ships, instead of requiring a second migration of
    this read site. The ctor `t_min`/`t_max` fields remain: they are the
    values `anneal_temperature` falls back to when `self.scheduler` exposes
    no `.config` at all (a bare unit-test double lighter than the real
    scheduler/R4 fixture) — a named degradation, not a raise, mirroring
    `resolve_vocab_size`'s stub fallback. Every real `EntropyBoundScheduler`
    and the R4 `FakeEntropyBoundScheduler` fixture expose `.config`, so this
    fallback is exercised only by pre-R4-style bare test doubles.

    **`pinned_mask` (ADR-CDG-010 Decision 4, issue #64 Phase 2, gate
    correction A1):** derived once at construction from `constraints.pins`
    when a `Constraints` payload is supplied — `None` otherwise. No pin
    participant exists yet (Phase 3), so this is the validated-then-ignored
    payload's positions read directly, not an observed per-step write. Valid
    **only because and only while** pins are position-static (the D6
    hard-pin invariant: a hard pin re-asserts the same positions every step,
    so the pinned-position set is provably constant for the whole run) — see
    `DiffusionFrame.pinned_mask`'s docstring for the full A1 scope-guard
    reasoning and the labeled door for a future dynamic/re-pinning constraint
    type.
    """

    scheduler: Any
    """Object exposing a `.num_inference_steps` attribute (the real
    `EntropyBoundScheduler`/`BlockRefinementScheduler`, or a fake in tests) —
    read fresh on every callback, never cached, so the collector always
    reflects the scheduler's *effective* post-`set_timesteps` value (issue
    #20; see this class's docstring)."""

    t_min: float
    """Fallback anneal `t_min` used only when `self.scheduler` exposes no
    `.config.t_min` (see the class docstring's effective-knob-telemetry
    section) — otherwise superseded every callback by the live config read."""

    t_max: float
    """Fallback anneal `t_max`, same fallback-only role as `t_min` above."""

    keep_frames: Literal["last", "all"] = "last"
    on_frame: Callable[[DiffusionFrame], None] | None = None
    constraints: "Constraints | None" = None
    """ADR-CDG-010 Decision 4 / issue #64 Phase 2: the validated `Constraints`
    payload (or `None`), used only to derive each frame's static `pinned_mask`
    at construction time — see the class docstring's `pinned_mask` section.
    Not otherwise read; no participant consumes this yet (Phase 3)."""

    top_k: int = 0
    """ADR-CDG-014 Decision 3 Tier 1 (issue #61 P-B): the validated
    `CaptureSpec.top_k` value (or `0`), read fresh in `on_step_end` to derive
    `DiffusionFrame.top_k_ids`/`top_k_weights` from the same pre-pin `logits`
    Tier 0's `entropy` derives from. `0` (default) leaves both fields `None`
    (additive-optional absence, Decision 1/2) — byte-identical to every run
    before this phase."""

    capture_full_distribution: bool = False
    """ADR-CDG-014 Decision 3 Tier 2 (issue #61 P-C): the validated
    `CaptureSpec.capture_full_distribution` value (or `False`). `False`
    (default) leaves `DiffusionFrame.distribution` `None` on every frame
    (additive-optional absence, Decision 1/2) — byte-identical to every run
    before this phase. `True` derives `distribution = softmax(logits)` from
    the same pre-pin `logits` Tier 0/1 already read, subject to
    `max_full_distribution_steps`'s retention budget (Decision 5) below."""

    max_full_distribution_steps: int | None = None
    """ADR-CDG-014 Decision 3/5 Tier 2's budget (issue #61 P-C): the
    validated `CaptureSpec.max_full_distribution_steps` value. Caps the
    number of CAPTURED steps (in step order, counted by `self.steps_used`
    at the moment each callback fires — i.e. the first
    `max_full_distribution_steps` calls to `on_step_end`) whose frame
    retains a populated `distribution`; every step beyond the budget gets
    `distribution=None` on the RETAINED frame, regardless of `keep_frames`
    (Decision 5 — the budget caps retention, not the live stream).
    `on_frame`, when given, still receives every frame's `distribution` live
    while Tier 2 is on and the step is within budget — Decision 5's "a
    streaming consumer that does not retain gets the full stream" clause
    applies to whichever steps actually computed a distribution; a
    consumer's `on_frame` never sees a *different* value than what the
    matching retained frame holds for that same step. `None` (default)
    means "no budget declared"; ingress (`dgemma.ingress.validate_capture`)
    already rejects `capture_full_distribution=True` with no budget, so a
    live `_FrameCollector` never actually reaches `capture_full_distribution
    =True, max_full_distribution_steps=None` through `run_diffusion` — this
    field still defaults to `None` for callers driving the collector
    directly in tests, where `capture_full_distribution=False` makes the
    budget irrelevant."""

    frames: list[DiffusionFrame] = field(default_factory=list)
    steps_used: int = 0
    _canvas_idx: int = -1
    _prev_step_idx: int | None = None
    _pinned_mask: Any | None = field(default=None, init=False, repr=False)
    _pinned_mask_built: bool = field(default=False, init=False, repr=False)

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

        **Tier 0 entropy capture (ADR-CDG-014 Decision 3/4, issue #14):**
        this method IS the composite's `capture` participant, which runs
        FIRST in the fixed order (`capture -> cancel -> beta-rebuild -> pin`,
        ADR-CDG-010) — so `callback_kwargs["logits"]`, when present, is the
        model's pre-pin predictive distribution for this step, never a
        post-pin/post-constraint artifact. `DiffusionFrame.entropy` is
        always populated when `logits` is reachable (the always-on Tier 0
        default); `None` only when a caller drives this collector directly
        without requesting `logits` in `callback_on_step_end_tensor_inputs`
        (additive-optional discipline — absence, never a zero-valued
        stand-in, ADR-CDG-014 Decision 1/2).

        **Effective-knob telemetry (ADR-CDG-011 clause 7, issue #64 Phase
        2):** `entropy_bound`/`t_min`/`t_max` are read off `self.scheduler.
        config` fresh THIS callback — the values `step()` actually consumed
        producing this frame — falling back to the ctor `self.t_min`/
        `self.t_max` (and `None` for `entropy_bound`, which has no ctor
        fallback) only when `self.scheduler` exposes no `.config` at all.
        `t`/`temperature` are recomputed from the live `t_min`/`t_max`, so a
        walker-mutated anneal range (Phase 4) is reflected consistently
        across `t`/`temperature` and the `effective_*` fields together.

        **`pinned_mask` (ADR-CDG-010 Decision 4, issue #64 Phase 2):** built
        once, lazily, from `self.constraints.pins` on the first callback and
        reused for every subsequent frame this run — see the class
        docstring's `pinned_mask` section for the A1 scope-guard reasoning.
        `None` when no `Constraints` payload was supplied.

        **Tier 1 top-k capture (ADR-CDG-014 Decision 3, issue #61 P-B):**
        when `self.top_k > 0` and `logits` is reachable, `DiffusionFrame.
        top_k_ids`/`top_k_weights` are derived from the SAME pre-pin
        `logits` `entropy` reads (`logits.topk(k)` for ids/raw scores,
        `softmax` over just those k logits for weights — a per-position
        renormalization over the top-k slice, not the full-vocab softmax,
        since Tier 1 never materializes the full distribution) — so Tier 1
        inherits Tier 0's capture-pre-pin ordering guarantee for free, not a
        second derivation that could drift from it. `top_k=0` (default)
        leaves both fields `None` (additive-optional absence, Decision 1/2),
        matching every run before this phase byte-for-byte.

        **Tier 2 full-distribution capture (ADR-CDG-014 Decision 3/5, issue
        #61 P-C):** when `self.capture_full_distribution` is `True`,
        `logits` is reachable, AND this callback's step is still within
        `self.max_full_distribution_steps`'s budget (`self.steps_used`,
        read BEFORE incrementing — i.e. this is the Nth call, budget counts
        calls 0..budget-1), `DiffusionFrame.distribution` is
        `softmax(logits, dim=-1)` over the SAME pre-pin per-position logits
        entropy/top-k already read — one derivation, not a third drifting
        copy. Once the budget is exhausted, `distribution` stays `None` on
        every subsequent frame for the rest of the run (Decision 5: the
        budget caps *retained* frames regardless of `keep_frames`) — Tier 0/
        Tier 1 fields are completely unaffected by the Tier-2 budget running
        out (they have their own independent policies). `capture_full_
        distribution=False` (default) leaves `distribution` `None`
        unconditionally, byte-identical to every run before this phase.
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

        config = getattr(self.scheduler, "config", None)
        effective_t_min = getattr(config, "t_min", self.t_min) if config is not None else self.t_min
        effective_t_max = getattr(config, "t_max", self.t_max) if config is not None else self.t_max
        effective_entropy_bound = getattr(config, "entropy_bound", None) if config is not None else None

        t, temperature = anneal_temperature(
            step_idx, self.scheduler.num_inference_steps, effective_t_min, effective_t_max
        )
        # Mean over the block dim ONLY — one fraction per example, never a
        # batch-blended scalar (review finding, 2026-07-05).
        committed_per_example = tuple(accepted_index.float().mean(dim=-1).tolist())

        entropy = None
        top_k_ids = None
        top_k_weights = None
        distribution = None
        logits = callback_kwargs.get("logits")
        if logits is not None:
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            if entropy.dim() == 2:
                # `logits` may be `[batch, canvas_len, vocab]` (real
                # pipeline) or already `[canvas_len, vocab]` (some fake
                # fixtures) — single-example scope (ADR-CDG-014 Open
                # Questions: batched capture deliberately deferred to a
                # P4+ design pass), so batch index 0 is what every existing
                # single-example consumer expects.
                entropy = entropy[0]

            # Same batch-squeeze as entropy above, applied to logits itself
            # once — shared by Tier 1 (top-k) and Tier 2 (full distribution)
            # so both derive from the identical per-position row entropy
            # just read (one normalization, not drifting copies, ADR-CDG-014
            # Decision 4).
            per_position_logits = logits[0] if logits.dim() == 3 else logits

            if self.top_k > 0:
                top_k_values, top_k_ids = per_position_logits.topk(self.top_k, dim=-1)
                # Renormalize over just the top-k slice (a per-position
                # softmax restricted to the k candidates already selected) —
                # Tier 1 never materializes the full-vocab softmax (that is
                # Tier 2's `distribution` field below, budget-gated). This is
                # the top-k conditional distribution, not an approximation of
                # the full one; a consumer reading `top_k_weights` as
                # anything other than "renormalized over these k ids" would
                # be reading past what Tier 1 actually captured.
                top_k_weights = torch.softmax(top_k_values, dim=-1)

            if self.capture_full_distribution:
                # Budget check (ADR-CDG-014 Decision 3/5, issue #61 P-C):
                # `self.steps_used` is this callback's 0-indexed ordinal
                # (read BEFORE the increment below), so the budget retains
                # the FIRST `max_full_distribution_steps` captured steps —
                # `max_full_distribution_steps=None` (no budget declared,
                # only reachable when a caller drives the collector
                # directly rather than through `run_diffusion`'s ingress
                # gate) is treated as "no cap", matching
                # `capture_full_distribution`'s own unconditional meaning in
                # that direct-use case.
                budget = self.max_full_distribution_steps
                if budget is None or self.steps_used < budget:
                    # Full per-position softmax — Tier 2's ~134 MB/step
                    # payload (ADR-CDG-014 Decision 3's Tier-2 row).
                    distribution = torch.softmax(per_position_logits, dim=-1)

        if not self._pinned_mask_built:
            self._pinned_mask = _build_pinned_mask(self.constraints, canvas)
            self._pinned_mask_built = True

        frame = DiffusionFrame(
            canvas_idx=self._canvas_idx,
            step_idx=step_idx,
            t=t,
            temperature=temperature,
            committed_fraction_per_example=committed_per_example,
            canvas=canvas,
            entropy=entropy,
            top_k_ids=top_k_ids,
            top_k_weights=top_k_weights,
            distribution=distribution,
            pinned_mask=self._pinned_mask,
            effective_entropy_bound=effective_entropy_bound,
            effective_t_min=effective_t_min,
            effective_t_max=effective_t_max,
        )
        self.steps_used += 1
        if self.keep_frames == "last":
            self.frames[:] = [frame]
        else:
            self.frames.append(frame)
        if self.on_frame is not None:
            self.on_frame(frame)
        return {}
