"""dgemma/payloads.py — declarative ingress payloads (ADR-CDG-010/011, rule 7).

Rationale for a new module (not `types.py`, issue #64 §1.2): `types.py` holds
the *output/socket* contract dataclasses (`DiffusionFrame`/`CanvasTrace`/
`CanvasState`); these are *input* contract objects with their own ingress
validators (`dgemma/ingress.py`). Keeping them here keeps `types.py`'s "these
ARE the socket payloads consumed at the door" docstring honest and gives
ingress a single import home. Imports zero ComfyUI (ADR-CDG-003 rule 1),
exactly like `types.py`.

**Phase 1 scope (issue #64 §6):** this module lands the `Constraints`/
`ControlSignals` dataclasses and the `MUTABLE_TARGETS` registry.
`constraints=` is LIVE end-to-end since issue #64 Phase 3 (`dgemma/loop.py`);
`control_signals=` is LIVE end-to-end since issue #64 Phase 4 (the
`WalkerParticipant`, `dgemma/participants.py`, wired in `dgemma/loop.py`).

`CaptureSpec` (below) is minted HERE, not in this module's Phase-1-era
placeholder location: ADR-CDG-014 Decision 7 rules the `capture=` payload's
dataclass is owned by the capture cluster (issue #61), not issue #64 — this
is that mint. `pinned_mask` rides `DiffusionFrame` instead (a per-frame trace
field, not a `capture=` knob — issue #64 Phase 2/3, `dgemma/types.py`); it is
not a `CaptureSpec` field. `keep_frames` stays the existing `run_diffusion`
keyword-only parameter (`dgemma/loop.py`) — `CaptureSpec.keep_frames` is
validated (`dgemma.ingress.validate_capture`, duck-typed since issue #64 P1)
but not yet wired to override it; that wiring is out of ADR-CDG-014 P-B's
scope (Tier 1 top-k derivation + ingress only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


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


@dataclass(frozen=True)
class CaptureSpec:
    """The `capture=` payload (ADR-CDG-014 Decision 7, issue #61 P-B).

    Owned by the capture cluster (issue #61 / ADR-CDG-014), not issue #64 —
    the ruling in ADR-CDG-014 Decision 7 that the `capture=` param's shape is
    this cluster's to define. Canonical fields are the retention-tier
    controls; `keep_frames` is validated here (duck-typed since issue #64
    P1, `dgemma.ingress.validate_capture`) but the `run_diffusion`
    `keep_frames=` keyword-only parameter remains the one that actually
    governs `_FrameCollector` retention this phase — wiring `CaptureSpec.
    keep_frames` through to override it is not part of P-B's scope (Tier 1
    top-k derivation + ingress only).

    Tier 2 fields (`capture_full_distribution`/`max_full_distribution_steps`,
    ADR-CDG-014 Decision 3's Tier-2 row) land in this phase (P-C): the full
    per-position distribution is EXPLICIT opt-in WITH A BUDGET — an
    unbounded request is rejected at ingress (`dgemma.ingress.
    validate_capture`), never silently honored, per the ADR's load-bearing
    clause (Decision 3 Tier 2, "an unbounded full-distribution request is
    rejected at ingress, never silently honored").
    """

    top_k: int = 0
    """Tier 1 knob (ADR-CDG-014 Decision 3's Tier-1 row): number of top
    candidate token ids/weights to derive per position from each step's
    pre-pin `logits`, alongside Tier 0's `entropy`. `0` (this field's
    default) is OFF — `DiffusionFrame.top_k_ids`/`top_k_weights` stay `None`
    (additive-optional absence, ADR-CDG-014 Decision 1/2), matching today's
    byte-identical behavior for every run that doesn't ask for Tier 1.
    Validated at ingress (`dgemma.ingress.validate_capture`): must be a
    non-negative int, and when `> 0` must not exceed the model's vocabulary
    size (an out-of-vocab k would silently clamp to `topk`'s own vocab-sized
    ceiling rather than raising the caller's actual mistake — rejected
    instead, per rule 5 `EMIT-CANONICAL / PARSE-AT-THE-DOOR`). The
    gate-ratified recommendation when a caller opts Tier 1 on at all (issue
    #61 design-gate comment, 2026-07-13) is k=16 — a caller's own choice
    (`CaptureSpec(top_k=16)`), not this field's default."""

    capture_full_distribution: bool = False
    """Tier 2 knob (ADR-CDG-014 Decision 3's Tier-2 row, issue #61 P-C):
    explicit opt-in for the full per-position distribution
    (`softmax(logits)`, `float32[canvas_len, vocab]`, ~134 MB/step). `False`
    (this field's default) is OFF — `DiffusionFrame.distribution` stays
    `None` on every frame (additive-optional absence, Decision 1/2),
    byte-identical to every run before this phase. Requesting Tier 2
    (`True`) WITHOUT also supplying `max_full_distribution_steps` is rejected
    at ingress (`dgemma.ingress.validate_capture`) — the load-bearing clause
    that prevents a 48 GB box dying mid-run because a ~6.4 GB/canvas heavy
    field defaulted on or was requested unbounded (rule 5, `EMIT-CANONICAL /
    PARSE-AT-THE-DOOR` — reject the unbounded ask, never silently OOM)."""

    max_full_distribution_steps: int | None = None
    """Tier 2's budget (ADR-CDG-014 Decision 3/5): the maximum number of
    steps for which `DiffusionFrame.distribution` is actually retained this
    run. `None` (this field's default) means "no budget declared" — paired
    with `capture_full_distribution=True` this is the unbounded-request
    shape ingress rejects; paired with `capture_full_distribution=False`
    (the common case) it is inert, matching every run that never touches
    Tier 2. When `capture_full_distribution=True`, must be a positive int:
    `0`/negative is rejected (a zero-or-negative budget can never retain
    anything, which is either a caller mistake or better expressed as
    `capture_full_distribution=False`). The budget caps *retained* frames
    under `keep_frames="all"` regardless of `keep_frames` (Decision 5) — the
    FIRST `max_full_distribution_steps` captured steps (in step order) are
    the ones that carry a populated `distribution`; every step beyond the
    budget carries `distribution=None`, the same absence semantics as Tier 2
    being off entirely for that step. `on_frame` still sees every frame's
    `distribution` live when Tier 2 is on (a streaming consumer that does
    not retain gets the full stream, Decision 5) — only what the retained
    `CanvasTrace` keeps is budget-limited."""

    keep_frames: Literal["last", "all"] = "all"
    """Validated (`dgemma.ingress.validate_capture`, issue #64 P1) but not
    yet wired to override `run_diffusion`'s own `keep_frames=` parameter —
    see this class's docstring."""
