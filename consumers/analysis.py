"""consumers/analysis.py — pure functions over a captured `CanvasTrace`.

Consumer tier per ADR-CDG-008 Open Question #1 (settled `consumers/`, see the
amendment note in `decisions/adr-cdg-008-mcp-center-multi-surface-topology.md`
and issue #55 §1): analysis **parses** an already-emitted `CanvasTrace` — it
never wraps `load_model`/`run_diffusion` — so it lives outside `dgemma/`'s
import graph, not inside it. This module was relocated from
`dgemma/sampling.py` (CDG-008 Phase 3); `dgemma/__init__.py` no longer
re-exports it, and `tests/test_seam.py`'s subprocess assertion (CDG-008 Phase
4) enforces that `import dgemma` never pulls this module in.

ComfyUI-agnostic (ADR-CDG-003): no `surfaces/` import, no torch-autograd or
live pipeline dependency — everything here reads a `CanvasTrace` already
produced by `dgemma.loop.run_diffusion` and returns plain Python
lists/dataclasses. `surfaces/comfyui/trace.py` is the ComfyUI-side adapter
that wraps these plain results into `IMAGE`/`STRING` sockets; that wrapping
does NOT belong here (the one-line test, ADR-CDG-003: no ComfyUI-shaped
tensor construction in this module).

Working data note: `DiffusionFrame` (dgemma/types.py) carries the *aggregate*
`committed_fraction_per_example` per step, not a per-position commit mask —
the scheduler's raw `accepted_index` tensor is read once in
`_FrameCollector.on_step_end` and reduced to a mean; it is not retained
per-frame. `build_commit_heatmap`/`corroborate_no_mask_token` therefore
derive a per-position signal from consecutive `frame.canvas` snapshots —
"did this position's token id change since the previous frame" — rather
than from a stored mask that doesn't exist. This was, at the time those two
functions were written, the only per-position signal available (no
per-position entropy was captured yet); ADR-CDG-014 (issue #61) has since
landed genuine per-position `entropy`/`top_k_ids`/`top_k_weights`/
`distribution` capture on the frame (Tiers 0-2) — `build_entropy_heatmap`
below reads the real Tier-0 measurement rather than a canvas-diff proxy.
`build_token_identity_grid` (issue #11, P-D) is a third per-position view:
the raw token id itself (not a diff, not an entropy value), straight off
the same already-raw `frame.canvas` snapshots.

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

# Dual-context import, same discipline as the root `__init__.py` and
# `surfaces/comfyui/*.py` (ADR-CDG-003/CDG-008): ComfyUI's real loader gives
# this module a dotted `__package__` (`"<synthetic-pack-name>.consumers"`)
# with the pack root absent from `sys.path`, so a bare absolute
# `from dgemma.types import ...` raises `ModuleNotFoundError` in that context
# even though it resolves fine under pytest (repo root on `sys.path`).
# `consumers/` sits one level under the pack root — same depth as `dgemma/`
# itself — so climbing to the sibling `dgemma` package is TWO dots (one dot
# reaches this module's own package's parent, i.e. the pack root; the second
# descends into `dgemma.types`). Gate is `"." in __package__` (not bare
# truthiness) because `__package__` is the plain string `"consumers"` (zero
# dots) under pytest/standalone and `"<synthetic>.consumers"` (>= 1 dot)
# under the real loader — mirrors the root `__init__.py`'s depth-0 gate,
# not `surfaces/comfyui/*.py`'s depth-2 `>= 2` gate (this module is one
# level shallower). Found during CDG-008 Phase 3 execution (issue #55 did
# not anticipate this module needing its own dual-context gate — the
# original `dgemma/sampling.py` never needed one because its `.types` import
# was relative-within-`dgemma`, which resolves regardless of context).
if __package__ and "." in __package__:
    from ..dgemma.types import CanvasTrace
else:
    from dgemma.types import CanvasTrace


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
    the engine owns the scaling (ADR-CDG-003: `surfaces/comfyui/trace.py`
    stays a thin adapter; its `cell_px` widget threads straight through to this
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


def build_entropy_heatmap(trace: CanvasTrace, scale: int = 1) -> list[list[float]]:
    """2D array, one row per frame in `trace.frames` order, one column per
    canvas position — the Tier-0 DISTRIBUTION-seam sibling of
    `build_commit_heatmap` (ADR-CDG-014 issue #61 P-D). Cell value is that
    position's captured `frame.entropy[position]` (a `float`, nats — the
    natural-log base `torch.distributions.Categorical.entropy()` uses), NOT
    a 0/1 diff flag: entropy is a continuous per-position measurement, so
    this heatmap's cells ARE the measurement, unlike the commit heatmap's
    derived "did this change" boolean.

    Raises `ValueError` if `trace.frames` is non-empty and any frame's
    `entropy` is `None` — ADR-CDG-014 Decision 2's absence-vs-empty
    discipline: `None` means "Tier 0 was not captured this run" (e.g. a
    legacy trace, or a `logits`-unreachable run), and reading it as
    zero-entropy would be exactly the ADR-CDG-001 lying-payload trap this
    function must not commit. A consumer that wants a heatmap MUST have a
    Tier-0-capturing trace; there is no honest degraded rendering of an
    absent measurement (contrast `build_commit_heatmap`, which derives its
    signal from `frame.canvas` — always present — so it has nothing to be
    absent). Empty `trace.frames` returns `[]` without touching `entropy`,
    matching `build_commit_heatmap`'s and `_heatmap_to_image`'s existing
    empty-trace handling.

    `scale` — identical nearest-neighbor upscale contract as
    `build_commit_heatmap` (operator finding, 2026-07-05): raises
    `ValueError` for `scale < 1`; `scale=1` (default) is the identity.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale!r}.")
    rows: list[list[float]] = []
    for frame in trace.frames:
        if frame.entropy is None:
            raise ValueError(
                "build_entropy_heatmap: frame.entropy is None — Tier 0 entropy was not "
                "captured this run (ADR-CDG-014 Decision 2: absence is not zero-entropy). "
                "Re-run with logits reachable, or use build_commit_heatmap for a trace "
                "without Tier 0 capture."
            )
        row = [float(value) for value in frame.entropy.tolist()]
        rows.append(row)
    if scale == 1:
        return rows
    return [
        [cell for cell in row for _ in range(scale)]  # widen: each cell -> scale columns
        for row in rows
        for _ in range(scale)  # tallen: each row -> scale rows
    ]


def build_token_identity_grid(trace: CanvasTrace) -> list[list[int]]:
    """2D array, one row per frame in `trace.frames` order (keyed on each
    frame's own `(canvas_idx, step_idx, t, temperature)` identity, per
    `DiffusionFrame`'s docstring), one column per canvas position — the raw
    per-position TOKEN ID held at that step, batch index 0 (single-example
    scope, module docstring). This is issue #11's token-identity view: the
    per-step `frame.canvas` snapshots are already raw/pre-excision
    (`decode_frames`'s own "no excision" contract), so this function is a
    direct unpack, not a derived signal — contrast `build_commit_heatmap`,
    which reduces the same snapshots to a changed/unchanged diff.

    Deliberately returns raw integer ids, not decoded text: decoding needs a
    tokenizer, and this module stays tokenizer-free (module docstring,
    ADR-CDG-003) — the same reason `dgemma.loop.decode_frames` (not this
    module) owns text decoding. A decoded-token rendering is a surface-side
    concern for whichever adapter has the model's processor (mirrors how
    `consumers/tally_audit.py` takes already-decoded strings rather than
    decoding itself).

    `[]` for an empty trace."""
    return [frame.canvas[0].tolist() for frame in trace.frames]


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
