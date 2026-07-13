# ADR-CDG-009 — Make the two-canvas (block) structure legible in the flipbook and heatmap

**Status**: proposed
**Date**: 2026-07-13
**Related**: ADR-CDG-001 (native socket types — payloads mean what they say;
this ADR's `committed_fraction` relabeling and heatmap fill-sentinel are both
instances of that discipline applied to display, not just data), ADR-CDG-005
(`CANVAS_STATE` — canvas identity/boundaries; this ADR is the display-side
twin for `CANVAS_TRACE`/flipbook rather than the resumable-state contract)

---

## Context

DiffusionGemma generation spans **two canvases** (block-diffusion): a fixed-
width canvas denoises to convergence, then the block boundary advances and a
second, fresh canvas opens and denoises in turn. `DiffusionFrame.canvas_idx`
(`dgemma/types.py:73`) already tags every captured frame with which block it
belongs to — `_FrameCollector` (`dgemma/loop.py:104-202`) infers the boundary
from a non-increasing `step_idx` between callbacks
(`dgemma/loop.py:134-141,178-180`) and increments `_canvas_idx` there. The
data needed to distinguish canvases already exists per-frame; nothing
upstream of display needs to change to make it available.

Two display surfaces read `trace.frames` today and neither uses
`canvas_idx` for anything beyond the diff-baseline reset:

- **The flipbook** (`nodes/frames_image.py:render_frames_to_image_batch`,
  called from `nodes/sampler.py:239-244`): renders each decoded frame string
  to a fixed-size image, captioned only `"step {idx+1}/{total}"`
  (`frames_image.py:159`) — a running index over the whole trace, no block
  coordinate.
- **The heatmap** (`dgemma/sampling.py:build_commit_heatmap`, wrapped by
  `nodes/trace.py:_heatmap_to_image`): one row per frame, one column per
  canvas position. It already resets the diff baseline at a `canvas_idx`
  change (`sampling.py:70`, covered by
  `tests/test_sampling.py:test_new_canvas_idx_resets_the_diff_baseline`) —
  but the row width is `len(frame.canvas[0])`, i.e. that block's own canvas
  width, with no accommodation for a later block being a different width.
  `_heatmap_to_image` builds a single `torch.tensor(heatmap, ...)`
  (`nodes/trace.py:45`) over the whole `list[list[int]]`, which requires
  uniform row length — a `canvas_idx` transition with a differing width
  produces a jagged list, the "25-vs-29 frame/heatmap shape mismatch" issue
  #26 names from a live run.

Grounded facts from a 61-step cloze run (issue #26): canvas 1 converges
(`committed_fraction` → 1.0) by step 41; step 42 opens canvas 2 and
`committed_fraction` crashes 1.000 → 0.004, because the fraction is
**block-local** — `accepted_index.float().mean(dim=-1)`
(`dgemma/loop.py:185`) is computed over the active block's own tensor only,
with no persistent cross-block state (`EntropyBoundScheduler` holds none;
ADR-CDG-001's addendum on scheduler-relative commit semantics). Canvas 1's
content is preserved byte-for-byte across the boundary — the block *advances*,
it does not discard — but nothing in the current display says so. A reader
sees a converged canvas, then a wall of noise, then a sawtooth in the summary
curve, and the honest reading ("canvas 1 finished; canvas 2 began") is
indistinguishable from the wrong one ("canvas 1 got thrown out and
re-melted") without already knowing the block-diffusion mechanism going in.

This is a `EMIT-CANONICAL / PARSE-AT-THE-DOOR` problem in the same shape
ADR-CDG-001 already names for data payloads, applied here to display:
`committed_fraction`'s block-local *meaning* is currently only in a
docstring (`dgemma/types.py:65-70`), not on any surface a reader actually
looks at (the `DGemmaTrace` summary string, the flipbook caption). A number
that means "fraction of the **active block** committed" but is labeled
merely "committed_fraction" is exactly the kind of caption that reads as
something it isn't — a display-layer instance of the lying-payload family,
not a new failure mode this ADR invents.

## Decision

Three changes, all downstream of data that already exists — no new fields on
`DiffusionFrame` or `CanvasTrace`, no new capture-side logic:

### 1. Heatmap: pad to max canvas width with a documented fill sentinel, plus a boundary row

`build_commit_heatmap` (`dgemma/sampling.py`) computes
`max_width = max(len(frame.canvas[0]) for frame in trace.frames)` up front,
then pads every row to `max_width` on the right with a **fill sentinel value
of `2`** (not `0` or `1` — those are load-bearing commit-state values; `2` is
reachable by no real computation in this function, so it cannot collide with
a genuine reading). A row's sentinel-padded tail always corresponds to
positions past that block's own canvas end — "this position does not exist
in this block," not "this position is uncommitted." `_heatmap_to_image`
(`nodes/trace.py`) maps `0`/`1`/`2` to three visually distinct colors
(e.g. commit-state grayscale for `0`/`1`, a distinct hue such as flat blue
for `2`) so a padded region reads as *structurally absent*, not as
melted-and-never-committing.

Between the last frame of one `canvas_idx` and the first frame of the next,
insert a single **boundary marker row**: a uniform row of a fourth sentinel
value (`3`), rendered as a bright dividing line the height of one (unscaled)
row. This directly answers issue #26's ask for "a boundary marker in the
heatmap" and gives a reader a literal, unmissable line at every block
transition, independent of reading the commit-state coloring correctly.

This resolves the ragged-row/shape-mismatch bug (any-width canvases now
produce a rectangular array) as a side effect of making the boundary
visible, not as a separately-motivated fix.

### 2. Flipbook: per-block caption, boundary frame

`render_frames_to_image_batch`'s caption changes from
`f"step {idx + 1}/{total}"` to `f"canvas {frame.canvas_idx + 1} · step
{block_local_idx + 1}/{block_len}"` — the caption already available data
(`canvas_idx`), reformatted to say what it actually is: position within a
block, not position within the whole run. `render_frames_to_image_batch`
currently takes `frames: list[str]` only (decoded text, no frame metadata);
it must additionally take `canvas_indices: list[int]` (parallel array, one
`canvas_idx` per string — cheap to derive from `canvas_trace.frames` at the
`nodes/sampler.py` call site, which already holds both the decoded strings
and the original `DiffusionFrame`s) so the block-local numbering and
boundary detection can be computed without decoding.

At each detected `canvas_idx` transition, insert one synthetic **divider
frame** into the batch (same fixed canvas size as the real frames, distinct
background color, caption `"— canvas {N} → canvas {N+1} —"`) between the
last frame of the outgoing block and the first frame of the incoming one.
This makes the boundary a visible frame in any consumer that scrubs the
`images` batch (`PreviewImage`, `SaveAnimatedWEBP`, VHS `Video Combine`) —
the same mechanism issue #26 asks for, applied to the medium (a frame
sequence) each of those consumers actually understands, rather than a
side-channel annotation those consumers would not render.

### 3. `committed_fraction`: label as block-local at every display surface

`DGemmaTrace`'s summary string (`nodes/trace.py:_format_summary`) changes
its `"committed_fraction per step: ..."` line to
`"committed_fraction per step (block-local — resets to ~0 at each canvas
boundary, marked above): ..."`, and the live-view status line
(`web/live_view.js:72`, already showing `canvas_idx`) gets the same
one-line clarification in its legend/tooltip. No field renames, no schema
change — `DiffusionFrame.committed_fraction`'s docstring
(`dgemma/types.py:80-89`) already states the block-local scope correctly;
this is purely propagating that existing, correct meaning to the two
operator-facing text surfaces that currently don't say it. This directly
answers issue #26's "the sawtooth reads as re-melt" misreading — the fix is
captioning, not recomputation. (Whether a *cumulative*, cross-block reading
of commit progress is also wanted is deliberately out of scope — see Open
Questions.)

## Rationale

### Positive Consequences
- Resolves the ragged-heatmap bug (#26's "25-vs-29 shape mismatch") as a
  direct consequence of the boundary-legibility fix, not a separate patch —
  one change, two problems closed.
- No new capture-side state: `canvas_idx` already exists on every frame
  (`_FrameCollector`, unchanged); this ADR is entirely a rendering-layer
  change downstream of data already flowing through `CanvasTrace`.
  `ADR-CDG-003`'s thin-adapter discipline holds — `dgemma/sampling.py` gains
  padding/divider-row *list* logic (still pure, still no ComfyUI import),
  `nodes/trace.py` and `nodes/frames_image.py` gain color/caption mapping
  only.
- The boundary marker (heatmap divider row, flipbook divider frame) is
  legible without reading any caption — a structural visual cue, not solely
  a labeling fix — so a reader who never looks at the summary string still
  sees "something changed here" rather than misreading melt.
- `committed_fraction` relabeling costs nothing beyond string literals and
  is immediately correct-and-shippable independent of the heatmap/flipbook
  changes landing.

### Negative Consequences
- **Four-valued heatmap cells (`0`/`1`/`2`/`3`) instead of two-valued.** Any
  downstream consumer that assumed a strict binary commit-state heatmap (a
  hypothetical future analysis node reading the raw `list[list[int]]` rather
  than the rendered `IMAGE`) must be updated to know about the fill and
  divider sentinels, or it will misread padding-absence and the boundary
  marker as commit-state. **Enforcement surface:** `build_commit_heatmap`'s
  own docstring becomes the source of truth for the four-value contract;
  `tests/test_sampling.py` gets an explicit assertion pinning `2` (fill) and
  `3` (divider) as reserved values distinct from `{0, 1}`, so a future
  change to the commit-state encoding (e.g. widening past binary) cannot
  silently collide with these sentinels without a failing test.
- **Padding changes heatmap pixel dimensions** in a way existing saved
  screenshots/examples (`examples/`) will not match — a purely cosmetic
  regression for anyone diffing old images against new ones, not a
  behavioral one.
- **The flipbook's synthetic divider frame is not a real denoising step.**
  A consumer that assumes every frame in the `images` batch corresponds 1:1
  to a `DiffusionFrame` (e.g. a hypothetical future node that zips `images`
  back against `canvas_trace.frames` by index) would miscount once divider
  frames are inserted. **Enforcement surface:** `render_frames_to_image_batch`
  keeps a strict "N real frames in, N + (num_boundaries) frames out"
  contract, documented in its own docstring and pinned by a test asserting
  batch length equals `len(frames) + num_transitions` for a multi-canvas
  fixture — the count relationship is the checkable invariant, not "trust
  the docstring."
- **`gen_length` fixed-width assumption broken already, not introduced
  here.** `render_frames_to_image_batch` already computes one shared
  `(width, height)` for the whole batch from the tallest wrapped frame
  (`frames_image.py:151-155`); canvases of differing token width don't
  change the image pixel size (text is word-wrapped to a fixed pixel
  `width` regardless of token count), so this ADR does not need to solve a
  second sizing problem — noted so a reviewer doesn't go looking for one.

## Alternatives Considered

### Option A: Per-canvas sub-heatmaps (separate `IMAGE` per block) instead of one padded/divided heatmap

Emit a *list* of heatmap images, one per `canvas_idx`, each sized to its own
block's width — no padding, no sentinel values, no ragged-row problem
because there's never one shared array to be ragged.

**Why not chosen as the primary design:** `DGemmaTrace.RETURN_TYPES` is
currently `("IMAGE", "STRING")` — a single heatmap `IMAGE`, matching
`PreviewImage`'s single-image-or-uniform-batch expectation. Switching to a
list return is a breaking change to the node's output arity/type (an
`OUTPUT_IS_LIST` flip), forcing every existing workflow that wires
`DGemmaTrace`'s `heatmap` output downstream to be rebuilt, for a run that
today is architecturally exactly two canvases (P0–P3 scope) but could in
principle be more. It also loses the "one glance, whole run" property the
issue explicitly asks for ("makes the two-canvas structure legible" reads
most naturally as *one* legible artifact, not N artifacts a reader must
mentally re-stitch). Left as an **open question** below rather than fully
foreclosed: if a future run regularly spans many blocks of wildly differing
width, padding's overhead (unused sentinel area) could dominate and tip the
trade-off toward this option.

### Option B: Truncate/crop all rows to the minimum canvas width instead of padding to the max

Avoids inventing a fill sentinel entirely — every row is already the same
(smaller) width.

**Why rejected:** Silently discards real commit-state data for whichever
canvas is wider (typically canvas 2, the "over-provisioned for this prompt"
case issue #26 itself observed). A heatmap that quietly drops columns is a
lying payload in exactly ADR-CDG-001's sense — it looks complete and isn't.
Padding-with-a-documented-sentinel keeps every real cell visible; the cost
is a legend entry, not lost data.

### Option C: Cumulative (cross-block) `committed_fraction` as a second curve, instead of only relabeling the existing block-local one

Compute a running "total canvas committed" curve that does not crash to
near-zero at each boundary, addressing the "reads as re-melt" complaint by
changing the number's *meaning* rather than only its *label*.

**Why not chosen as the primary design:** No such notion is well-defined
without deciding how to weight blocks of differing width against each other
(a straight average would let canvas 2's over-provisioned padding drag the
number down in a way that doesn't correspond to any real un-commitment) —
that is real design work with its own trade-offs, not a captioning fix, and
risks exactly the kind of invented-metric scope creep this repo's
greenfield discipline warns against manufacturing without an anchoring
failure. The block-local relabeling (Decision §3) is a strict, low-risk
subset that ships now; a cumulative curve is deferred to an open question
rather than bundled in.

## Open Questions

- [ ] Does a run ever span more than two canvases in practice, and if so,
      does the padded-heatmap approach's wasted sentinel area become a real
      cost (Option A's rejected-for-now trade-off) rather than a
      theoretical one? **Resolution trigger:** revisit once a 3+-canvas
      trace is observed from a real run; P0–P3 grounding is two-canvas only.
- [ ] Is a cumulative (non-block-local) commit curve wanted alongside the
      relabeled block-local one (Option C), and if so, how should blocks of
      differing width be weighted? **Resolution trigger:** raise as a
      follow-on issue only if an operator finds the block-local-only curve
      insufficient after this ADR ships — do not build speculatively.
- [ ] Should the flipbook's synthetic divider frame be configurable
      (on/off, or a distinct color per canvas rather than a single generic
      divider) once there's real multi-run operator feedback on
      legibility? **Resolution trigger:** first operator usage of the
      shipped flipbook change.
- [ ] `nodes/trace.py`'s `cell_px` widget already upscales the heatmap
      (`build_commit_heatmap(scale=...)`); does the divider row need its
      own height distinct from `scale` (e.g. always `2*scale` tall) to stay
      visible at small `cell_px` values, or is `scale`-height sufficient?
      **Resolution trigger:** visual check against `cell_px=1` (the
      smallest legal value) once implemented.

## Supersession Relationships

**Supersedes:** none
**Superseded by:** TBD

## References

- `dgemma/types.py:31-90` (`DiffusionFrame`, `canvas_idx` field and
  `committed_fraction` docstring)
- `dgemma/loop.py:104-202` (`_FrameCollector`, `canvas_idx` boundary
  inference at `:134-141,178-180`)
- `dgemma/sampling.py:40-83` (`build_commit_heatmap`, existing
  `canvas_idx`-aware diff-baseline reset at `:70`)
- `nodes/trace.py:38-46` (`_heatmap_to_image`, the single-tensor build that
  requires uniform row width)
- `nodes/frames_image.py:133-165` (`render_frames_to_image_batch`, current
  running-index-only caption at `:159`)
- `nodes/sampler.py:239-244` (call site wiring `frames`/`frames_image`)
- `web/live_view.js:72` (existing live-view `canvas_idx` display, unaffected
  by this ADR beyond the `committed_fraction` legend clarification)
- `tests/test_sampling.py:test_new_canvas_idx_resets_the_diff_baseline`
  (existing same-width boundary coverage this ADR's implementation must not
  regress)
- Issue #26 (this ADR's originating ask, including the grounded 61-step
  cloze-run observations)
