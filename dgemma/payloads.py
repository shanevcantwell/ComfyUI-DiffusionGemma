"""dgemma/payloads.py — declarative ingress payloads (ADR-CDG-010/011, rule 7).

Rationale for a new module (not `types.py`, issue #64 §1.2): `types.py` holds
the *output/socket* contract dataclasses (`DiffusionFrame`/`CanvasTrace`/
`CanvasState`); these are *input* contract objects with their own ingress
validators (`dgemma/ingress.py`). Keeping them here keeps `types.py`'s "these
ARE the socket payloads consumed at the door" docstring honest and gives
ingress a single import home. Imports zero ComfyUI (ADR-CDG-003 rule 1),
exactly like `types.py`.

**Phase 1 scope (issue #64 §6):** this module lands the dataclasses and the
`MUTABLE_TARGETS` registry only. No participant reads these yet — Phases 3/4
wire `Constraints`/`ControlSignals` into `PinParticipant`/`WalkerParticipant`.
`run_diffusion` validates-then-ignores a payload this phase (dgemma/loop.py).

`CaptureSpec` is deliberately NOT defined here (issue #64 §7 O5 field-shape
ruling): ADR-CDG-014 Decision 7 rules the `capture=` payload's dataclass is
owned by the capture cluster (issue #61) — `pinned_mask`/`keep_frames` are
contributed into that shared dataclass as additive-optional fields, not
minted as a competing type in this module. Only `Constraints`/`ControlSignals`
are #64-owned and defined here.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Pin:
    """One id-level given (ADR-CDG-010 D6: pins are id-level).

    `position`: canvas index to pin (0 <= position < gen_length).
    `token_id`: the vocab id to assert at that position (must be in-vocab).

    Both mechanisms (ADR-CDG-010 D1) apply to every Pin: the logit mask
    restricts `position`'s vocabulary to `{token_id}` (commits first, ~zero
    entropy), and the canvas re-assertion writes `token_id` at `position`
    each step (Phase 3 — not built yet in Phase 1).
    """

    position: int
    token_id: int


@dataclass(frozen=True)
class Constraints:
    """The `CONSTRAINTS` socket payload (ADR-CDG-010 D6, minted in the R2
    socket-type module — `surfaces/comfyui/socket_types.py`).

    A collection of id-level pins. Empty tuple == no-op (validated but
    installs no participant, matching `None`)."""

    pins: tuple[Pin, ...] = ()


@dataclass(frozen=True)
class Binding:
    """One signal->knob binding (ADR-CDG-011 clause 3: units at the binding).

    `target`: the scheduler.config knob name — MUST be in `MUTABLE_TARGETS`
    (below); `num_inference_steps` is NOT in that set (ADR-CDG-011 clause 4,
    ingress reject).
    `signal`: the unitless precomputed per-step signal, length ==
    num_inference_steps.
    `low`/`high`: the declared range the unitless [0,1] signal maps into
    (polarity is encoded by low>high, ADR-CDG-011 clause 3)."""

    target: str
    signal: tuple[float, ...]
    low: float
    high: float


@dataclass(frozen=True)
class ControlSignals:
    """The control-signal payload (ADR-CDG-011 clause 1/2).

    A collection of bindings; each maps one unitless signal to one knob with
    an explicit range. Empty == no-op."""

    bindings: tuple[Binding, ...] = ()


# The ONLY scheduler.config knobs the walker may mutate (ADR-CDG-011 clause
# 4). `EntropyBoundScheduler.step()` reads each fresh from `self.config` on
# every call (`scheduling_entropy_bound.py:148-149,154`).
# `num_inference_steps` is DELIBERATELY EXCLUDED (clause 4, issue #20's
# desync mechanism foreclosed by construction, not by luck — see
# ADR-CDG-011's Context section for the full grounding).
#
# Placement (issue #64 §7 O4, ratified 2026-07-13): engine-side, in this
# module, beside `Binding` — NOT in `surfaces/comfyui/socket_types.py`. This
# names scheduler-config knobs the engine owns, not socket envelope strings
# the surface owns (IDENTITY-ENVELOPE split, ADR-CON-0001 one-home-per-
# concept). ADR-CDG-011 Open Question 1 left the placement undecided; the
# gate ratification comment (2026-07-13) confirms (a) engine-side.
MUTABLE_TARGETS = frozenset({"entropy_bound", "t_min", "t_max"})
