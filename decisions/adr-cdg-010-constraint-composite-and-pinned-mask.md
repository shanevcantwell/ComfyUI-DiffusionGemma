# ADR-CDG-010 — Constraints as a two-mechanism model: logit mask + canvas re-assertion, composed through an engine-owned ordered composite

**Status**: accepted (ratified 2026-07-13, PR #43)
**Date**: 2026-07-13
**Related**: ADR-CDG-003 (node-engine seam — the core/adapter split this
composite lives core-side of), ADR-CDG-004 (drive seam — `run_diffusion`'s
signature is what this ADR widens), ADR-CDG-008 (MCP-center topology —
Correction 1's `STATELESS-CORE` posture this ADR's composite must not
violate), ADR-CDG-011 (control-signal/mod-matrix — the walker that shares this
ADR's composite slot; §"Cross-references" below), Issue #35 (architecture
review — the drafting spec both this ADR and ADR-CDG-011 transcribe)

---

## Context

Issue #35's grounding pass (2026-07-13, "intervention-surface sweep") found a
logit-mask seam that changes what a *given* (a pinned token, a masked region)
can mean. Two mechanisms exist, and they do different jobs:

- **Logit masking** — the installed `diffusers` pipeline consumes exactly the
  model module's returned `.logits` (`pipeline_diffusion_gemma.py:365-371` →
  `scheduler.step(model_output=logits, ...)`), so a `register_forward_hook` on
  `pipe.model` that mutates returned logits alters what the commit rule
  receives. Restricting a cell's vocabulary makes `token_entropy`
  (`scheduling_entropy_bound.py:158`) reflect true constrainedness — a cell
  masked to one token reads ~zero entropy and commits first, at ~zero cost
  against the entropy budget. The commit rule *becomes* most-constrained-first
  propagation, rather than approximating it by luck.
- **Canvas re-assertion** — masking alone is not sufficient for a given,
  because a rejected position is renoised by `torch.randint` over the **full**
  vocabulary on the next step (issue #26's root cause, corroborated in #28).
  Without re-asserting the given's value onto the canvas each step, the
  forward pass conditions on garbage at that cell.

`run_diffusion` (`dgemma/loop.py:465`) has exactly one hook slot today: a
single hardcoded `callback_on_step_end=collector.on_step_end`
(`dgemma/loop.py:619`, ARCHITECTURE.md F1). Five expansion participants
(β-renoise, walker, pin, capture, and any future canvas-writer) want that slot
with real ordering semantics — there is no composition layer, so a second
participant cannot be added without clobbering the first. This ADR names the
composite that closes F1 for the constraints half of the expansion; ADR-CDG-011
names it for the control-signal half. Both widen the same slot and must agree
on one ordering (see "Cross-references").

## Decision

1. **Constraints use both mechanisms, never one alone.** A `constraints=`
   payload on `run_diffusion` that pins a token or masks a region installs (a)
   a forward-hook logit mask shaping what commits, and (b) a per-step canvas
   re-assertion guaranteeing what conditions. Neither alone is a correct
   given: mask-only lets a pinned cell drift once accepted-elsewhere noise
   crosses it on a later step (nothing re-asserts the value after commit is
   read); re-assertion-only pays full entropy-budget cost for a cell whose
   answer is already known, and never gets the most-constrained-first commit
   ordering the mask buys.

2. **`run_diffusion` widens only by declarative payloads.** `constraints=` is
   a payload the engine validates at ingress (constraint ids in-vocab; fail on
   unknown) and turns into engine-built participants — never a surface-built
   closure or hook. `pipe.model` reachability from a step-end callback (the
   sealed-surface finding in #28) is explicitly **not** a sanctioned
   installation path for a surface; only the engine itself installs the
   forward hook, through the R5 lifecycle context manager (clause 5 below).
   This is `run_diffusion`'s rule 7 (ARCHITECTURE.md) applied to constraints
   specifically.

3. **The composite is an engine-internal ordered list of engine-built
   participants: β-renoise, walker, pin, capture — in that fixed order.**
   - **Capture runs before any canvas-writer.** Capture must read
     model-committed, pre-pin truth; if pin (or any other writer) ran first,
     capture would record constraint-asserted tokens as if the model had
     committed them, corrupting the trace's meaning of "committed" a second
     way beyond ADR-CDG-001's scheduler-relative-semantics addendum.
   - **β-rebuild runs before pin.** β-renoise (the liquid-phase-decoding bench
     participant that rebuilds/renoises canvas regions) must finish writing
     before pin re-asserts, or a pin's re-assertion could be immediately
     overwritten by a renoise pass that doesn't know the cell was just pinned.
   - **Pin is the last writer.** Every other participant has had its turn to
     write the canvas; pin's re-assertion is what actually reaches the next
     forward pass unclobbered. This is mechanism (b) from Decision 1, placed
     where it can guarantee what it promises.
   - **Live view (`on_frame`) is explicitly not a composite participant**
     (#35 delta Correction 2, corroborated in ARCHITECTURE.md's step-end
     intervention section). It stays on the existing engine-side `on_frame`
     read-only observer seam (`nodes/sampler.py:136-159` pattern,
     `run_diffusion(on_frame=…)` at `dgemma/loop.py:477`): receives a built
     `DiffusionFrame`, return ignored, structurally read-only. It needs no
     position among canvas-writers because pre-pin truth reaches it as
     **frame fields** (`pinned_mask`, effective knobs), not by observer
     ordering. It is the only executable crossing the surface owns; every
     other participant is engine-internal.

4. **`pinned_mask` rides every `DiffusionFrame`, distinguishing
   model-committed from constraint-asserted.** A frame's commit information is
   a lying payload (ADR-CDG-001's addendum sense) unless a reader can tell
   which cells the *model* committed under the entropy-bound rule and which
   cells the *constraint layer* re-asserted regardless of what the model
   would have chosen. `pinned_mask` is the field that makes this
   distinguishable — boolean per canvas position, `True` where clause 3's pin
   step (re-)wrote the cell this frame, independent of whatever the
   scheduler's own commit reading says. Capture's pre-pin ordering (clause 3)
   is what lets it record the scheduler's honest commit reading *and* the
   pin layer separately report which cells it touched — the two together are
   what keeps the trace honest; either alone would let a reader conflate
   "the model decided this" with "the constraint forced this."

5. **The logit-mask hook installs only through the R5 forward-hook lifecycle
   context manager.** The mask from Decision 1(a) is the engine-installed
   `register_forward_hook` on `pipe.model` (the only logit door, per #28 —
   a callback returning `{"logits": ...}` is silently discarded by the
   installed pipeline). Per ARCHITECTURE.md rule 6 (`STATELESS-CORE`) and F4,
   no hook may survive a `run_diffusion` call: R5's context manager is the
   sole installation path, guaranteeing teardown on both clean return and a
   raised exception. A hook installed any other way (e.g. an ad hoc
   `try/finally` local to the constraints code) would duplicate F4's failure
   mode inside the very feature meant to close it.

6. **`CONSTRAINTS` is minted in R2's socket-type module; pins are id-level.**
   The `CONSTRAINTS` socket (ADR-CDG-001 lists it, unspecified) carries a
   collection of pin/mask entries addressed by position or id, minted once in
   the R2 socket-type module (interim `nodes/socket_types.py`, target
   `surfaces/comfyui/socket_types.py` per ARCHITECTURE.md rule 4) — no inline
   `DGEMMA_*` literal at the constraints call site.

7. **Pin state is per-run, never cached across `run_diffusion` calls.** The
   accumulated pin mask is exactly the class of cross-call-mutable state
   ADR-CDG-008 Correction 1 forbids the MCP surface's state manager from
   retaining (a cached scheduler carrying a prior run's mutated dims forward
   is the observed 25-vs-29 heatmap precedent this generalizes from). Each
   `run_diffusion` call builds its pin state fresh from that call's
   `constraints=` payload; nothing persists it for the next call.

## Amendment — cancellation position in the composite order (2026-07-13, PR #45)

Decision 3 fixes the canvas-writer order (capture < β-rebuild < pin) but
predates issue #38's fold-in of a per-step cancellation check into the same
composite, so it is silent on where that check runs. R1's implementation
(PR #45) initially placed cancellation **first** (before capture); the gate
review adjudicated the position and the seat decided the opposite. The
ratified order is:

    capture -> cancellation check -> β-rebuild -> pin

**Why capture-then-cancel.** The composite fires at `callback_on_step_end` —
*after* the scheduler's `step()` has committed the step's canvas
(`pipeline_diffusion_gemma.py:365-371` → `:404-407`). At the moment the
cancellation check could trip, the current step's frame is therefore
**committed evidence, not an in-flight partial**. Capturing before the check
means a cancelled run's partial `CanvasTrace` retains its exact truncation
point: the committed frame of the very step the caller cancelled on — often
the most diagnostically interesting frame (it is the state the caller was
looking at when they decided to stop). This is #38's "a cancelled experiment
run is still data" taken at full strength, and it aligns with the pack's
instrumentation-first evidence posture (cf. #46).

**Alternative considered — cancel-first (rejected).** Checking cancellation
before capture saves one capture on the abandoned step. That optimizes a
cost (one frame's capture) the pack's own values would gladly pay, and it
silently discards a *committed* frame — a small lying-by-omission in the
trace: the run's record ends one step before where the run actually stopped.
Rejected in the 2026-07-13 gate adjudication (PR #45 review comment).

**What survives from the cancel-first rationale:** cancellation still
precedes every canvas-writer, so no β-rebuild/pin pass runs for a step whose
result will never be used — only the evidence side of the cancelled step
completes.

**Partial-return contract** (engine-side, landed in R1): on
`DiffusionCancelled`, `run_diffusion` returns the same
`(text, CanvasState, CanvasTrace)` shape as a completed run, built from all
captured frames including the truncation-point frame; with zero captured
frames (unreachable through the composite's own flow, defensively guarded)
it re-raises rather than fabricating an empty `CanvasState`. Note
`CanvasState` here is the P3 validity object — ADR-CDG-005's full resumable
save-state is still implementation-pending; the partial return is honest
about where the run stopped, which is the property this amendment fixes.

Enforcement surface: `tests/test_step_end_composite.py:TestCancellationSeam`
(capture-then-cancel, writer gating) and
`tests/test_run_diffusion_cancel.py` (truncation-frame inclusion,
no-evidence re-raise). Record trail: PR #45 (implementation + gate review
comment), issue #38 (decided-position comment).

## Rationale

### Positive Consequences

- **The commit rule becomes most-constrained-first propagation instead of
  approximating it.** Grounded directly in #28: masking a given to one token
  makes its entropy reading ~zero, so it commits first under the existing
  entropy-bound rule with no new commit-ordering logic needed.
- **The trace stops lying about who committed what.** `pinned_mask` closes
  the gap ADR-CDG-001's addendum opened but didn't resolve for constraints
  specifically: "committed" already carries scheduler-relative semantics;
  this ADR adds constraint-relative semantics on the same field family.
- **One installation path forecloses the second door.** Restricting hook
  installation to R5's context manager and constraint payloads to declarative
  ingress means a surface cannot reach `pipe.model` to install its own mask —
  the forbidden shape (a surface returning `{"canvas": ...}`, CDG-008's
  surface-resident-sampling-logic shape) becomes structurally unrepresentable
  for constraints specifically, not just asserted in prose.

### Negative Consequences

- **Composite ordering is now a real invariant to test, not an implementation
  detail.** Getting capture-before-pin or β-before-pin wrong produces a
  trace that looks plausible but silently misattributes commits — the
  failure is not a crash, it is a lying payload. This raises the bar for
  touching `dgemma/loop.py`'s composite: every change must be checked against
  the fixed order, not just against "does it run."
- **Two mechanisms per constraint means two failure surfaces.** A masking bug
  and a re-assertion bug can partially cancel (mask commits the token fast,
  re-assertion papers over a masking gap) and look correct while being
  fragile — testing must exercise each mechanism's absence independently
  (mask-only, re-assertion-only, neither) to catch a regression in either
  half.
- **`register_forward_hook` on a ~53 GB resident model is a sharp edge.**
  A hook that raises, or is left installed past its `run_diffusion` call,
  corrupts every subsequent run sharing the loaded model (F4). This ADR
  makes R5 mandatory rather than optional specifically because this cost is
  high and silent.

## Enforcement surfaces (per clause)

| Clause | Invariant | Enforcement surface | Status |
|---|---|---|---|
| 1 — constraints use both mechanisms | a `constraints=` payload always installs both a logit mask and a re-assertion write, never one alone | Composite-participant test asserting both effects for one constraint entry, over R4's shared fake-pipeline fixture | `NOT-YET-IMPLEMENTED` — #35 R1, gated on R4 |
| 2 — declarative payloads only | `run_diffusion(constraints=...)` accepts a payload, not a callable; ingress rejects a passed callable/hook | Ingress type/shape validation (constraint ids in-vocab; fail on unknown) | `NOT-YET-IMPLEMENTED` — ADR-CDG-010/011 ingress clause, ARCHITECTURE.md enforcement-surface table |
| 3 — fixed composite order (capture < β-rebuild < pin; capture before any writer) | two identical constraint runs produce identically-ordered composite effects; capture never reads post-pin state | Ordered-composite test over R4's shared fixture (`tests/conftest.py`): `tests/test_step_end_composite.py:TestFixedOrdering`, `TestOrderingIsStructural`, asserting the exact operation order via the fixture's recording model/scheduler | **In force** — #35 R1 (over R4's fixture), `dgemma/composite.py:StepEndComposite`; ARCHITECTURE.md enforcement-surface table |
| 3 — live view is not a participant | `on_frame` never blocks or reorders canvas-writers; removing all `on_frame` observers changes no canvas output | Existing read-only-observer contract (`_FrameCollector`'s docstring, `dgemma/loop.py`); no new test needed beyond today's `on_frame` exception-propagation coverage — flagged here as inherited, not newly created | **In force** (`nodes/sampler.py:114-161`, `dgemma/loop.py:477`) |
| Amendment — cancellation position (capture before cancellation; cancellation before any writer) | a cancelled step's committed truncation-point frame is captured before the cancel check raises; no canvas-writer runs on a cancelled step; partial return includes that frame | `tests/test_step_end_composite.py:TestCancellationSeam`; `tests/test_run_diffusion_cancel.py` | **In force** — PR #45 (amendment 2026-07-13) |
| 4 — `pinned_mask` per frame | every `DiffusionFrame` in `CanvasTrace` carries a `pinned_mask` field distinguishing constraint-asserted cells from model-committed ones | `DiffusionFrame` field addition (`dgemma/types.py`) + a trace-honesty test asserting a pinned cell's `pinned_mask` is `True` regardless of the scheduler's own commit reading that step | `NOT-YET-IMPLEMENTED` — #35 R1/R6 (rides `DiffusionFrame` extension discipline) |
| 5 — hook installs only via R5 | no `register_forward_hook` call for constraints exists outside the R5 context manager; no hook survives a `run_diffusion` call | R5 lifecycle context-manager test, clean **and** raising | `NOT-YET-IMPLEMENTED` — #35 R5 (F4); ARCHITECTURE.md enforcement-surface table |
| 6 — `CONSTRAINTS` minted once | no inline `DGEMMA_*` / constraint-shape literal outside the R2 mint module | Grep-gate test asserting against the module object | `NOT-YET-IMPLEMENTED` — #35 R2 |
| 7 — pin state per-run | two identical `run_diffusion(constraints=...)` calls on one loaded model yield identical effective pin telemetry; no pin state observable before the payload that created it | Same-in/same-out statelessness test (shared with ADR-CDG-011's F5) | `NOT-YET-IMPLEMENTED` — #35 R5/F5; ADR-CDG-008 Correction 1 |

## Alternatives Considered

### Option A: Logit masking only, no canvas re-assertion

**Why rejected:** Grounded directly in #28 — a rejected position is renoised
by `torch.randint` over the full vocabulary on the next step. A mask shapes
what the model is *allowed* to commit but does nothing to a cell the model
declines to commit this step; without re-assertion, that cell's canvas value
is garbage-conditioned noise for however many steps pass before the model
happens to accept it. This is not a tunable trade-off, it is a
correctness gap the grounding pass observed directly.

### Option B: Canvas re-assertion only, no logit mask

**Why rejected:** Correct but expensive and un-ordered: a given re-asserted
by canvas write alone still costs full entropy-budget weight every step (the
scheduler has no way to know the cell is already answered) and never gets
most-constrained-first propagation. The mask is what turns "eventually
correct" into "commits first, costs nothing" — dropping it gives up exactly
the mechanism #28 identified as the practical reason to prefer masking.

### Option C: A single unordered composite (participants run in registration order, no fixed semantics)

**Why rejected:** F1's five expansion participants have real ordering
dependencies (capture must see pre-pin truth; pin must write last); an
unordered composite would make correctness depend on the order callers
happen to register participants in, which is exactly the kind of
implicit, unenforced contract this repo's greenfield discipline exists to
avoid naming an invariant for only after it breaks. A fixed order, decided
once here, is checkable by a single ordered-composite test instead of by
convention.

## Open Questions

- [ ] **Exact `CONSTRAINTS` payload shape (per-position vs per-id addressing,
      mask-region syntax).** This ADR fixes the two-mechanism *model* and the
      composite *ordering*; it does not fix the wire shape of the `CONSTRAINTS`
      dataclass. **Resolution trigger:** settle when R2's mint module is
      authored, before any node wires a `constraints=` UI widget.
- [ ] **Does β-renoise ever need to run more than once per step (e.g. a
      multi-region renoise pass), and does that change the "β-rebuild before
      pin" ordering to "all β-rebuild passes before any pin pass"?** Current
      wording assumes a single β-rebuild phase per step. **Resolution
      trigger:** revisit if the liquid-phase-decoding bench's β-renoise
      participant (`docs/experiments/liquid-phase-decoding/concept.md`) turns
      out to need multiple ordered sub-phases.
- [ ] **Interaction between a mask-only constraint (no re-assertion desired,
      e.g. a soft bias) and this ADR's "constraints always use both
      mechanisms" clause.** Decision 1 assumes every constraint wants both
      mechanisms; a future soft-constraint use case might legitimately want
      mask-only. **Resolution trigger:** revisit if/when a soft-constraint
      requirement is grounded against real usage; until then, hard pins are
      the only constraint shape in scope and both mechanisms are mandatory
      for them.

**Resolution plan:** all three are resolved during R1/R2 implementation
planning, not by this ADR; none blocks recording the two-mechanism model and
composite ordering, and none should be silently decided by implementation
ahead of that pass.

## Cross-references

- **ADR-CDG-011** shares this ADR's composite slot: the walker (control
  signals) and the constraint participants (β-renoise, pin) both live in the
  one ordered list this ADR names. ADR-CDG-011 does not redefine the order;
  it places the walker's config-mutation step relative to it (walker prepares
  the *next* step's config before that step's forward pass; this ADR's
  capture/β-rebuild/pin sequence governs the *current* step's canvas). Where
  the two ADRs' composites interact (a walker-driven `entropy_bound` change
  and a pin re-assertion landing the same step), the fixed order in this
  ADR's Decision 3 is authoritative for canvas-write sequencing; ADR-CDG-011's
  config-mutation is orthogonal (it never writes the canvas).
- **ADR-CDG-008** Correction 1 (`STATELESS-CORE`) is the invariant Decision 7
  applies to pin state specifically; the MCP surface's Phase-2 state manager
  must never cache a scheduler or an accumulated pin mask across calls.
- **ARCHITECTURE.md** "The step-end intervention architecture" section states
  the target this ADR formalizes; its enforcement-surface table's composite-
  ordering and `pinned_mask` rows point back at this ADR by name.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD.

## References

- Issue #35 — architecture review; F1, F4, F5, R1, R2, R5, R6; the delta
  comment's Corrections 1–3.
- Issue #28 — logit-mask seam grounding (`register_forward_hook` on
  `pipe.model`; entropy-honest masking; canvas re-assertion requirement).
- Issue #26 — root cause (the callback sees only the current canvas),
  corroborated in #28's grounding comment, which directly observed the
  full-vocabulary `torch.randint` renoise on rejected positions.
- ADR-CDG-001 — native socket types; the scheduler-relative commit-semantics
  addendum this ADR's `pinned_mask` clause extends to constraint-relative
  semantics.
- ADR-CDG-008 — MCP-center topology, Correction 1 (`STATELESS-CORE` applied
  to the surface lifecycle object).
- `dgemma/loop.py:465,477,582,619` — `run_diffusion` signature, `on_frame`
  contract, the single hardcoded callback binding this ADR's composite
  replaces.
- `ARCHITECTURE.md` — "The step-end intervention architecture" section; rule
  7; the enforcement-surface table.
