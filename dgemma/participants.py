"""dgemma/participants.py — engine-built `StepEndComposite` participants
(ADR-CDG-010/011, issue #64 Phases 3/4).

Phase 3 landed `PinParticipant`, the canvas re-assertion mechanism (ADR-CDG-010
Decision 1(b)): given a validated `Constraints` payload, re-write each pin's
`token_id` at its `position` every step, in the composite's `pin` slot — the
LAST writer (`dgemma/composite.py`'s fixed `capture -> cancellation ->
beta_rebuild -> pin` order). Rejected positions renoise over the full
vocabulary on a real scheduler step (`EntropyBoundScheduler.step`'s
uniform-state resample, ADR-CDG-001 — no absorbing mask, every non-committed
cell is redrawn from the full vocab each step), so a given that is only
checked once at ingress and never re-asserted would drift the first time its
cell isn't the one the scheduler accepts. Re-assertion each step is what
guarantees *what conditions* the next forward pass sees (ADR-CDG-010's
two-mechanism framing; the sibling mechanism — restricting *what commits* — is
`dgemma/constraints_hook.py:build_logit_mask_hook`).

Phase 4 (issue #64, this module's `WalkerParticipant`) lands ADR-CDG-011's
control-signal walker: a `scheduler.config`-mutator, not a canvas-writer, so
it occupies the composite's separate `walker` slot
(`dgemma/composite.py:StepEndComposite`), invoked LAST — after capture, the
cancellation check, and every canvas-writer (`beta_rebuild`/`pin`) — per the
design-gate ratification on issue #64 (2026-07-13): walker-last satisfies
ADR-CDG-011 clause 6's lower bound (must run after capture reads the
current step's effective knobs) and is behaviorally inert relative to the
canvas-writers since it never touches the canvas, only `scheduler.config`.

`BetaRebuildParticipant` remains OUT of this phase (issue #64 §0: Phase 5) —
its ordered slot is `dgemma/composite.py`'s `beta_rebuild=`, still `()` by
default; later phases append it here rather than opening a second
participants module (ARCHITECTURE.md rule 7: engine-built participants have
one home).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .payloads import Constraints, ControlSignals


@dataclass
class PinParticipant:
    """Canvas re-assertion for a validated `Constraints` payload (ADR-CDG-010
    Decision 1(b), Decision 3's `pin` slot — the composite's LAST writer).

    Construct with `constraints` (a validated, non-`None`,
    non-empty-`pins` `Constraints` — `run_diffusion` only builds this
    participant when there is at least one pin; see `dgemma/loop.py`). Each
    call writes `token_id` at `position` into `callback_kwargs["canvas"]`
    (which, by pin's turn in the fixed order, already reflects any
    `beta_rebuild` writer's output — ADR-CDG-010 Decision 3: "beta-rebuild
    before pin ... a pin's re-assertion could be immediately overwritten by
    a renoise pass that doesn't know the cell was just pinned") and returns
    `{"canvas": <rewritten>}` so the composite threads it as the step's
    final canvas-writer output.

    **Shape assumption:** `canvas` is `[batch, canvas_len]` or `[canvas_len]`
    (both shapes appear across the fake-pipeline fixture and the real
    pipeline's per-example call); writing `canvas[..., position] = token_id`
    covers either via an ellipsis index — no batch-dim branching needed.

    **State contract (ADR-CDG-010 Decision 7):** holds only the immutable
    `constraints.pins` tuple from THIS call's payload — no cross-call state,
    no accumulation. A fresh `PinParticipant` is constructed by
    `run_diffusion` every call (`dgemma/loop.py`), never cached on
    `dgemma_model` or a module global — enforced by
    `tests/test_run_diffusion_statelessness.py`'s `TestPinStatePerRun`.
    """

    constraints: "Constraints"
    name: str = "pin"

    def __call__(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict | None:
        canvas = callback_kwargs["canvas"].clone()
        for pin in self.constraints.pins:
            canvas[..., pin.position] = pin.token_id
        return {"canvas": canvas}


@dataclass
class WalkerParticipant:
    """Control-signal walker (ADR-CDG-011 Decisions 1/2/3/4/6): mutates only
    `scheduler.config`, never the canvas — the composite's dedicated `walker`
    slot (`dgemma/composite.py`'s `StepEndComposite.walker`), invoked LAST,
    after capture and every canvas-writer.

    Construct with `control_signals` (a validated, non-`None`,
    non-empty-`bindings` `ControlSignals` — `run_diffusion` only builds this
    participant when there is at least one binding; see `dgemma/loop.py`) and
    `scheduler` (the SAME `EntropyBoundScheduler` instance `run_diffusion`
    constructed for this call — the walker writes through this object's
    `register_to_config`, never a copy).

    **Mechanism (ADR-CDG-011 Decision 4):** for each binding, map the raw
    unitless `signal[k] in [0,1]` into the declared `[low, high]` range —
    `value = low + (high - low) * signal[k]` — and write it via
    `scheduler.register_to_config(**{target: value})`, the ONLY real mutation
    path `ConfigMixin`'s frozen `.config` exposes (a whole-dict rebuild, never
    an in-place attribute set — see `dgemma/payloads.py:Binding`'s docstring
    and `tests/conftest.py`'s `FakeFrozenConfig`/`register_to_config` mirror).
    Every binding for a given step is folded into ONE `register_to_config`
    call (`{target: value for each binding}`) so bindings sharing a step
    (e.g. `t_min` and `t_max`, Decision 5's exact-per-step-temperature
    mechanism) land in a single whole-dict rebuild rather than clobbering
    each other across two separate calls.

    **Timing — "walker prepares the next step" (ADR-CDG-011 Decision 6, issue
    #64 gate ruling O1, 2026-07-13):** at the callback for `step_idx = k`,
    the walker writes `signal[k + 1]` — the config that will govern step
    `k + 1`'s forward pass and `step()` call — NEVER `signal[k]` itself.
    `signal[0]` is deliberately never applied by the walker: step 0 runs
    under the ctor-supplied `entropy_bound`/`t_min`/`t_max` `run_diffusion`
    was called with, and the walker's first write (at the end of step 0's
    callback) prepares step 1 from `signal[1]`. This keeps `register_to_config`
    the ONE config-write surface for the whole run (rule 6 `STATELESS-CORE`)
    — a second, ctor-time write path for `signal[0]` would split that surface
    in two. At the FINAL step (`step_idx == len(signal) - 1`, no step
    `k + 1` exists in the schedule), the walker is a no-op: writing past the
    signal's last index would read out of bounds, and there is no next
    step's forward pass left for the write to govern anyway.

    **State contract (ADR-CDG-011 clause 8 / F5):** holds only the immutable
    `control_signals.bindings` tuple from THIS call's payload — no cross-call
    state, no accumulated write history. A fresh `WalkerParticipant` is
    constructed by `run_diffusion` every call (`dgemma/loop.py`), never
    cached on `dgemma_model` or a module global — enforced by
    `tests/test_run_diffusion_statelessness.py`'s `TestWalkerStatePerRun`.

    **Return value:** always `None` — the walker is a config-mutator, not a
    canvas-writer (`dgemma/composite.py`'s `StepEndParticipant` protocol: a
    non-writer participant returns `None`/`{}`); its return is ignored by the
    composite (ARCHITECTURE.md, "The step-end intervention architecture").
    """

    control_signals: "ControlSignals"
    scheduler: Any
    name: str = "walker"

    def __call__(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict | None:
        next_step = step_idx + 1
        writes: dict[str, float] = {}
        for binding in self.control_signals.bindings:
            if next_step >= len(binding.signal):
                continue
            raw = binding.signal[next_step]
            value = binding.low + (binding.high - binding.low) * raw
            writes[binding.target] = value
        if writes:
            self.scheduler.register_to_config(**writes)
        return None
