"""dgemma/sampling.py — pure functions over a captured `CanvasTrace` (plan.md
Phase 3, module goes stub -> real per the module build-order table).

ComfyUI-agnostic (ADR-CDG-003): no `nodes/` import, no torch-autograd or live
pipeline dependency — everything here reads a `CanvasTrace` already produced
by `dgemma.loop.run_diffusion` and returns plain Python lists/dataclasses.
`nodes/trace.py` is the ComfyUI-side adapter that wraps these plain results
into `IMAGE`/`STRING` sockets; that wrapping does NOT belong here (the
one-line test, ADR-CDG-003: no ComfyUI-shaped tensor construction in this
module).

Working data note: `DiffusionFrame` (dgemma/types.py) carries the *aggregate*
`committed_fraction_per_example` per step, not a per-position commit mask —
the scheduler's raw `accepted_index` tensor is read once in
`_FrameCollector.on_step_end` and reduced to a mean; it is not retained
per-frame. Every function below that needs a per-position signal (the
heatmap, the mask-token corroboration) therefore derives one from consecutive
`frame.canvas` snapshots — "did this position's token id change since the
previous frame" — rather than from a stored mask that doesn't exist. This is
the forced reading of plan.md step 3's "entropy or commit-state per cell":
no per-position entropy is captured either, so commit-state (via canvas
diffing) is the only signal available, not an arbitrary pick between two
equally-available options.

Batch scope, stated rather than implied (review note, 2026-07-05): every
function here reads example 0 of each frame's canvas only
(`frame.canvas[0]`) — single-example scope, matching
`DiffusionFrame.committed_fraction`'s own batch_size==1 convenience and
`CanvasState`'s single-example contract. A batched trace's examples 1..N
are silently outside these functions' view; extending them is a real (P4+)
design question, not a missing loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .types import CanvasTrace


def build_commit_heatmap(trace: CanvasTrace, scale: int = 1) -> list[list[int]]:
    """2D array, one row per frame in `trace.frames` order, one column per
    canvas position (batch index 0 — single-example scope, matching
    `DiffusionFrame.committed_fraction`'s own batch_size==1 convenience).

    Cell value is `1` if that position's token id differs from the previous
    frame's snapshot (still being renoised / not locked in as of this step),
    `0` if it held steady. The first frame of each `canvas_idx` block (no
    prior-step canvas within the same block to diff against — see
    `DiffusionFrame.canvas_idx`'s docstring on block boundaries) reports
    every position as `1`: honestly, nothing has locked in yet at the start
    of a block.

    `scale` (operator finding, 2026-07-05: a raw steps×positions map — e.g.
    256×11 — is unreadably small as pixels) nearest-neighbor-upscales the
    grid by an integer factor on BOTH axes: each cell becomes a
    `scale`×`scale` block, so the output is `(steps*scale) x
    (positions*scale)`. `scale=1` is the identity. Pure list math here —
    the engine owns the scaling (ADR-CDG-003: `nodes/trace.py` stays a thin
    adapter; its `cell_px` widget threads straight through to this
    parameter). Raises `ValueError` for `scale < 1` (parse-at-the-door —
    a zero/negative scale would silently emit an empty grid).
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale!r}.")
    rows: list[list[int]] = []
    prev_positions: list[int] | None = None
    prev_canvas_idx: int | None = None
    for frame in trace.frames:
        positions = frame.canvas[0].tolist()
        if prev_positions is None or frame.canvas_idx != prev_canvas_idx:
            row = [1] * len(positions)
        else:
            row = [int(prior != current) for prior, current in zip(prev_positions, positions)]
        rows.append(row)
        prev_positions = positions
        prev_canvas_idx = frame.canvas_idx
    if scale == 1:
        return rows
    return [
        [cell for cell in row for _ in range(scale)]  # widen: each cell -> scale columns
        for row in rows
        for _ in range(scale)  # tallen: each row -> scale rows
    ]


def build_avalanche_curve(trace: CanvasTrace) -> list[float]:
    """The "Neither Parallel Nor Sequential" commit-fraction-over-step
    series: `DiffusionFrame.committed_fraction` (batch_size==1 convenience)
    read off each frame in `trace.frames` order, as a plain list of floats."""
    return [frame.committed_fraction for frame in trace.frames]


@dataclass
class MaskTokenCorroboration:
    """Result of `corroborate_no_mask_token` — item (c), the empirical
    corroboration of ADR-CDG-004's documentary "no MASK" confirmation.

    Tri-state `verdict` (issue #22 honesty finding, 2026-07-13): the
    original two-state shape (`no_fixed_sentinel: bool`) folded "genuinely
    varied evidence FOR uniform-state renoise" and "zero observed
    transitions, hence no evidence either way" into the same `True` value —
    a trace with no mid-renoise transitions at all (e.g. a single-frame or
    fully-converged-on-arrival trace) would print the same "uniform-state
    renoise supported" verdict as a trace that actually exhibited varying
    prior values. That is exactly the lying-payload shape ADR-CDG-001
    forbids: a summary claiming corroboration on zero evidence. `verdict`
    keeps the three outcomes distinct instead of collapsing two of them."""

    verdict: Literal["evidence_against_sentinel", "vacuous", "sentinel_found"]
    """`"evidence_against_sentinel"`: more than one distinct prior-value id
    was observed among positions caught mid-renoise — genuine evidence FOR
    uniform-state renoise (`mask_token_id=None`), a fixed sentinel is
    contradicted. `"vacuous"`: zero transitions were observed at all (no
    same-block frame pair had any position change) — nothing to corroborate
    OR contradict the no-mask hypothesis with; silent on the question, not
    supportive of it. `"sentinel_found"`: every observed transition's prior
    value was the same single repeated id — the signature an absorbing-MASK
    scheme would leave."""

    candidate_sentinel_id: int | None = None
    """Set iff `verdict == "sentinel_found"` — the one repeated id found."""


def corroborate_no_mask_token(trace: CanvasTrace) -> MaskTokenCorroboration:
    """Cheap empirical check (loose-ends.md, "near-free, ~15-20 lines") over
    already-captured frames' canvas tensors — deliberately a simple
    distinctness check, not a rigorous statistical test (see that entry's
    own sizing; if this starts growing real machinery, that is a scope
    signal to stop and flag, not silently expand).

    For each consecutive same-block frame pair, a position whose token id
    changed was — by construction — not yet locked in as of the *prior*
    frame. `prior_value` (the id it held right before that transition) is
    the value a not-yet-accepted position was actually holding. Under
    uniform-state renoise, that value is a freshly resampled vocabulary
    token each time a still-unaccepted position gets touched again, so the
    pool of `prior_value`s collected across the whole trace should vary.
    Under an absorbing-MASK scheme, every not-yet-accepted position holds
    the *same* fixed sentinel id until its own one-time reveal, so that pool
    would collapse to a single repeated value. Single-example scope: only
    example 0 of each frame's canvas is examined (module docstring).

    Collects `prior_value`, not `value`: the value a position lands ON after
    a change is its (generally distinct, per-position) committed content —
    checking that would trivially "vary" regardless of scheme and prove
    nothing. The pre-transition value is the one a fixed-sentinel scheme
    would pin.

    No block-crossing pairs are compared (mirrors `build_commit_heatmap`'s
    block-boundary handling, but this function does not call that one — the
    two derive independent things from the same raw frames and conflating
    them risked polluting this check with the heatmap's own "first frame of
    a block reports all positions as changed" convention, which is a
    completeness choice for rendering, not evidence about token identity).

    Tri-state return (issue #22): zero observed transitions is `"vacuous"`,
    NOT folded into the same verdict as genuinely varied evidence — a trace
    with nothing to diff (e.g. one frame, or every position already
    committed by the first captured step) has no evidence either way and
    must say so, rather than reporting the same "no fixed sentinel" verdict
    a trace with real varying transitions would earn. See
    `MaskTokenCorroboration`'s docstring for the three outcomes.
    """
    observed: set[int] = set()
    prev_positions: list[int] | None = None
    prev_canvas_idx: int | None = None
    for frame in trace.frames:
        positions = frame.canvas[0].tolist()
        if prev_positions is not None and frame.canvas_idx == prev_canvas_idx:
            for prior_value, current_value in zip(prev_positions, positions):
                if prior_value != current_value:
                    observed.add(int(prior_value))
        prev_positions = positions
        prev_canvas_idx = frame.canvas_idx

    if not observed:
        return MaskTokenCorroboration(verdict="vacuous")
    if len(observed) == 1:
        return MaskTokenCorroboration(
            verdict="sentinel_found", candidate_sentinel_id=next(iter(observed))
        )
    return MaskTokenCorroboration(verdict="evidence_against_sentinel")
