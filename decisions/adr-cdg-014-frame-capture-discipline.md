# DiffusionFrame capture discipline: additive-optional fields + heavy-field retention tiers

**Status**: `accepted`
**Date**: 2026-07-13
**Related**: ADR-CDG-001 (native socket types / no lying payloads), ADR-CDG-005 (`CANVAS_STATE` small-per-step economy + save-state/display split), ADR-CDG-010 (composite ordering: capture pre-pin), ADR-CDG-011 (declarative payloads, effective-knob telemetry on the frame), issue #35 R6/F3, issue #61 (the design/plan this record decides), issues #14 / #11 / #9 / #3, ROADMAP Track B R0.

---

## Context

The liquid-phase research program (`docs/experiments/liquid-phase-decoding/`) is gated at
Track B **R0 — the bench gate**: without per-position `DISTRIBUTION` capture, H0-observe and
H0-project cannot run, and committed-state-only logging demonstrably hides the liquid (proven
empirically, n=5, `concept.md` "Empirical grounding"). R0 depends on two capture surfaces —
per-position entropy (#14) and candidate/raw ids (#11) — both of which must land on
`DiffusionFrame`/`CanvasTrace`, the core's canonical per-step contract.

The forces at play:

- **A full per-position distribution is enormous.** `canvas_length=256`, `vocab≈262 144`. One
  step's full distribution in fp16 is **~134 MB**; a 48-step canvas is **~6.4 GB**; a
  multi-canvas generation (`gen_length` past 256) is **~26 GB**. The dev box is a 48 GB
  RTX-8000 already near its limit loading the ~53 GB model quantized/offloaded. A heavy field
  that defaults on is an OOM kill mid-run.
- **ADR-CDG-005's economy is load-bearing.** `keep_frames="all"` (the P3 default) is justified
  precisely because per-step state is *small* (a `gen_length` int64 canvas + a per-example
  float). A naively-added full-distribution field breaks that justification: the trace stops
  being cheap to retain.
- **ADR-CDG-001 forbids the scalar-shadow trap.** ARCHITECTURE.md rule 5 names #14's
  entropy-only capture as "the scalar-shadow trap reborn" if it is mistaken for the
  distribution. Entropy is a legitimate *slice* of the distribution seam, not a stand-in for it.
- **`DiffusionFrame` is a shipped contract.** Once frames with a given field shape ride out to
  consumers (`consumers/analysis.py`, `CanvasState` derivation, the live view), reshaping them
  is a breaking change across the whole surface set. The extension shape must be decided
  **before** the fields ship, not refactored after.

This record decides the *discipline* every capture field lands under, and the *retention policy*
for the heavy ones. The concrete field-by-field capture design and phased plan live in issue #61;
this ADR is the invariant that plan executes against.

## Decision

### 1. Additive-optional field discipline (R6)

Every field added to `DiffusionFrame` or `CanvasTrace` for capture is **optional with a
default**. The pre-existing positional fields (`DiffusionFrame`: `canvas_idx`, `step_idx`, `t`,
`temperature`, `committed_fraction_per_example`, `canvas`; `CanvasTrace`: `frames`,
`scheduler_name`, `scheduler_config`) never move, never gain a required sibling. A new capture
field is a keyword-defaulted addition, never a positional insertion.

### 2. Default semantics = "not captured," never "captured empty"

A heavy field's default is `None`, meaning *this tier was off this run* — categorically distinct
from a captured-but-degenerate value (e.g. an all-zero entropy vector). A consumer reads absence
honestly (raises or skips), and **never** treats `None` as a zero-valued measurement. A `None`
entropy field is "no entropy captured," not "every position had zero entropy" — reading it as the
latter is exactly the ADR-CDG-001 lying-payload failure.

### 3. Heavy-field retention tiers

Three tiers, by cost, each with a distinct capture policy and enforcement surface:

| Tier | Field(s) | Per-step cost | Per-run cost (48 steps, 1 canvas) | Policy |
|---|---|---|---|---|
| **0 — scalar-derived** | per-position `entropy` (`float32[canvas_len]`) | ~1 KB | ~49 KB | **Always captured** when logits are reachable. Default on. |
| **1 — top-k** | `top_k_ids` + `top_k_weights` (`[canvas_len, k]`) | ~25 KB (k=16) | ~1.2 MB | **On request** — `top_k` knob, default 0 (off). |
| **2 — full distribution** | `distribution` (`[canvas_len, vocab]`) | ~134 MB | ~6.4 GB (single canvas) / ~26 GB (4-canvas gen) | **Explicit opt-in with a budget.** `capture_full_distribution=True` **plus** `max_full_distribution_steps`. An unbounded full-distribution request is **rejected at ingress**, never silently honored. |

Tier 2 is the load-bearing clause: the failure it prevents is a 48 GB box dying mid-run because
a heavy field defaulted on or was requested without a bound. Ingress rejection (fail on unknown /
fail on unbounded) over silent OOM is the `EMIT-CANONICAL / PARSE-AT-THE-DOOR` shape (rule 5).

### 4. Capture derives from pre-pin logits (ADR-CDG-010 ordering)

All three tiers derive from the step's **`logits`** (entropy = `Categorical(logits=...).entropy()`;
top-k = `logits.topk(k)`; distribution = `softmax(logits)`), read in the **capture participant**,
which runs *first* in the composite (`capture → cancel → beta-rebuild → pin`). Capture therefore
records the *model's* predictive distribution — pre-pin, pre-constraint-reassertion truth — not a
post-pin artifact. The `entropy`/`distribution` fields are the model's decidedness over the
canvas, which is the signal #14 and the observation face actually want. `logits` is already a
base-pipeline `_callback_tensor_inputs` key (`pipeline_diffusion_gemma.py:76`); `run_diffusion`
requests it in `callback_on_step_end_tensor_inputs`.

### 5. Heavy capture is `keep_frames`-aware

Under `keep_frames="all"`, an unbudgeted Tier-2 field is the OOM path — so the Tier-2 budget caps
*retained* full-distribution frames regardless of `keep_frames`. `on_frame` still sees every
frame's heavy field live (a streaming consumer that does not retain gets the full stream); the
retained `CanvasTrace` holds Tier-2 only for budgeted steps. Tiers 0 and 1 ride `keep_frames`
unchanged (they are cheap enough that "all" is fine).

### 6. Raw pre-excision ids ride the trace, not `CanvasState` (#11)

The raw, un-excised final canvas ids ride **`CanvasTrace.raw_canvas_ids`** (additive-optional),
populated in `_build_result` from `sequences` *before* `excise_thought_channel` runs.
`CanvasState.canvas_ids` stays post-excision (the #8 thought-channel contract is unchanged). This
keeps the display/probe raw view on the TRACE side and the honest save-state/answer view on the
`CanvasState` side — ADR-CDG-005's split is load-bearing (a resumable save-state must not carry a
thought-channel leak; a research probe must be able to see one). The per-step `frame.canvas`
snapshots are *already* raw (`decode_frames` documents "no excision"); only the *final* raw
sequence was previously unreachable — this closes that gap.

## Rationale

### Positive Consequences

- R0's minimum (per-position entropy + raw ids, Tiers 0 + #11) is reachable with **negligible
  memory** (~49 KB/run entropy + one raw id sequence) — the bench gate is not blocked on the
  heavy tier.
- The heavy DISTRIBUTION socket (Tier 2) is available for H0-observe/H0-project **without** being
  a default footgun — a researcher opts in with a budget scoped to the steps/positions they are
  studying.
- The frame contract is extended once, correctly — additive-optional means every existing
  consumer (`derive_canvas_state`, `build_commit_heatmap`, the live view) keeps working
  untouched, and a future rung adds a field the same way.
- Absence-vs-empty semantics keep the trace honest: a consumer can never mistake "not captured"
  for a real zero.

### Negative Consequences

- Three capture tiers plus a budget knob is more surface than a single `capture_distribution`
  bool. Justified: the 5-order-of-magnitude cost span (49 KB → 26 GB) makes a single toggle
  either useless (too cheap to be the real thing) or dangerous (defaults to OOM).
- Tier 0's "always on when logits reachable" adds an entropy compute per step even for a run that
  never looks at it. Cost is ~1 KB and one `Categorical(...).entropy()` call over `[256, 262144]`
  logits already materialized by the forward pass — cheap relative to the step itself, and the
  P3 "heatmap promised, never captured" deviation is the failure of *not* defaulting it on.

## Alternatives Considered

### Option A: Single `capture_distribution: bool`, full distribution always when on

**Why rejected:** Conflates the 49 KB scalar, the 1.2 MB top-k, and the 26 GB full distribution
into one toggle. On → OOM on any non-trivial generation; off → the scalar shadow (#14) is
unreachable even though it costs nothing. The cost span *is* the design; a single bool erases it.

### Option B: Tier 2 defaultable with a small default budget (e.g. 4 steps)

**Why rejected:** Re-opens the OOM footgun by making a ~134 MB/step field reachable without an
explicit ask. A default budget is still a heavy field a user did not know they enabled. The
budget must be an operator's deliberate scoping, not a silent default. (Reconsider only if usage
shows the explicit opt-in is friction with no counterbalancing safety win.)

### Option C: Put `raw_canvas_ids` on `CanvasState`

**Why rejected:** Breaks ADR-CDG-005's save-state/display split. `CanvasState` is a resumable
save-state whose `canvas_ids` are post-excision *by contract* (a save-state must not carry a
thought-channel leak). The raw view is a research/probe surface that specifically *wants* to see
the pre-excision truth (#9's EOS-in-thought-span probe). Two different consumers, two different
honesty requirements — they belong on two different types.

### Option D: Quantize the full distribution (int8) to shrink Tier 2

**Why rejected here, noted for later:** int8 halves Tier 2 (~67 MB/step, ~3.2 GB/canvas) but does
not change the *shape* of the problem — it is still a heavy field that must be budgeted, and
quantizing a distribution before a consumer has decided what precision it needs is a lossy
`EMIT-CANONICAL` violation baked into the core. If a confirmed H0 shows fp16 is wasteful, a
quantized *transport* tier is a follow-up ADR, not a reason to skip the budget.

## Open Questions

- [ ] Retention default confirmation (Tier 0 on / Tier 1 off / Tier 2 opt-in+budget). **Resolution:** operator decision on issue #61 before P-A implements; recommendation is the table above.
- [ ] Top-k default value when Tier 1 is requested (recommended k=16). **Resolution:** settled at P-B implementation against observed multi-modality in the first H0-observe run.
- [ ] Batched (N>1) distribution capture. **Resolution:** deliberately deferred — single-example scope matches every existing `consumers/analysis.py` function; trigger is a batched-trace design pass (P4+), not this ADR.
- [ ] Whether Tier 2 needs a per-*position* budget (capture full dist only at named positions) in addition to the per-*step* budget. **Resolution:** revisit if per-step budgeting alone still OOMs a study of a wide canvas; the ingress validator is the place it would land.

## Supersession Relationships

**Supersedes:** none (first record on frame capture discipline).
**Superseded by:** TBD — a confirmed H0 that graduates a socket type (e.g. a first-class
`DISTRIBUTION` socket per the bench inventory) may amend this record's transport/retention shape.

## Implementation Notes

Concrete field-by-field design, the `capture=` declarative payload shape, the fake-pipeline test
plan, and the dependency-ordered gateable phases (P-A…P-D) live in **issue #61** — this record is
the invariant that plan executes against, not a duplicate of it.

| File | Change Type | Description |
|------|-------------|-------------|
| `dgemma/types.py` | Modified | `DiffusionFrame` gains `entropy`/`top_k_ids`/`top_k_weights`/`distribution` (all optional, default `None`); `CanvasTrace` gains `raw_canvas_ids` (optional, default `None`) |
| `dgemma/loop.py` | Modified | capture participant derives tiers from `logits`; `run_diffusion` requests `"logits"` in `callback_on_step_end_tensor_inputs`; `_build_result` populates `raw_canvas_ids` pre-excision; `capture=` ingress validation (Tier-2 budget reject) |
| `consumers/analysis.py` | Modified | `build_entropy_heatmap` + a token-identity view over `raw_canvas_ids` |
| `surfaces/comfyui/trace.py` | Modified | `DGemmaTrace` `mode` widget (`commit` \| `entropy`) |
| `tests/test_frame_capture_discipline.py` | Created | additive-optional, tier gating, budget reject, capture-pre-pin ordering |
| `tests/test_raw_canvas_ids.py` | Created | pre-excision id conservation; EOS-in-thought-span probe |
| `ARCHITECTURE.md` | Modified | flip the "`DiffusionFrame` extension discipline" enforcement row from `NOT-YET-IMPLEMENTED` to in-force as phases land |

## References

- Issue #61 — the capture-surface design/plan this record decides.
- `docs/experiments/liquid-phase-decoding/concept.md` — the DISTRIBUTION seam ("the gate everything waits on") and the n=5 empirical proof that committed-state-only logging hides the liquid.
- ROADMAP.md Track B R0 (bench gate) — the rung this design closes.
- ADR-CDG-001 (no lying payloads), ADR-CDG-005 (small-per-step economy + save-state split), ADR-CDG-010 (capture pre-pin), ADR-CDG-011 (declarative payloads).
