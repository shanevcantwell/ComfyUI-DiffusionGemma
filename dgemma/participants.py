"""dgemma/participants.py — engine-built `StepEndComposite` participants
(ADR-CDG-010/011, issue #64 Phase 3).

Phase 3 lands `PinParticipant`, the canvas re-assertion mechanism (ADR-CDG-010
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

`WalkerParticipant`/`BetaRebuildParticipant` are OUT of this phase (issue #64
§0: Phase 4/5) — this module holds only `PinParticipant` for now; later
phases append siblings here rather than opening a second participants module
(ARCHITECTURE.md rule 7: engine-built participants have one home).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .payloads import Constraints


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
