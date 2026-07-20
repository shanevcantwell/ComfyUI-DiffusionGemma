# ADR-CDG-017 — Neighborhood remelt: a `RemeltSpec` payload occupying the `beta_rebuild` slot, resolving ADR-CDG-010 Open Question 2

**Status**: proposed
**Date**: 2026-07-20
**Related**: ADR-CDG-010 (constraint composite and pinned mask — the `beta_rebuild`
slot and Open Question 2 this ADR resolves), ADR-CDG-011 (control-signal CV/LFO —
the declarative-payload/engine-built-participant discipline this ADR follows),
ADR-CDG-014 (frame capture discipline — Tier-1 `top_k_ids`/`top_k_weights`, this
kernel's data source), ADR-CDG-015 (latent-field input-embedding seam — the
embedding-kNN kernel's future landing site, named not built), ADR-CDG-012 (KV-cache
seam — the KV-derived kernel's future landing site, named not built), ADR-CDG-016
(crystalline CA rule-table payloads — the sibling occupant of the same
`beta_rebuild` slot, sequenced relative to this ADR below), issue #111 / commit
`96eea85` (the `KNOB_DOCS` mint pattern this ADR extends), issue #103 (the rule-8
ComfyUI/MCP parity asymmetry this ADR must not reintroduce), issue #115 (the
originating idea and its pre-registration floor)

---

## Context

Issue #115 (2026-07-20) asks a well-formed sampling question CDG's own scheduler
source settles the mechanics of: `EntropyBoundScheduler.step()` renoises every
rejected position with `torch.randint(0, vocab_size)` — a full uniform-vocabulary
redraw, logit-independent, memoryless (`scheduling_entropy_bound.py:162-172`,
zero-instance-attribute mutation in `step()`). The idea: instead of redrawing to
pure entropy, redraw a rejected position to a *structured* neighborhood of the
token it first annealed toward — replaying the position's own captured
distribution rather than discarding it. This is squarely CDG's declared
territory (VISION.md's "every sampler scalar is a wireable per-step field") and
is exactly the shape ADR-CDG-010 named and deferred: **Open Question 2** ("does
β-renoise ever need to run more than once per step... revisit if the
liquid-phase-decoding bench's β-renoise participant turns out to need multiple
ordered sub-phases") and the conformance table's honest gap ("no fourth ingress
payload — ADR-CDG-010 Open Question 2 is still unresolved... `beta_rebuild=()`
stays the default at every real call site").

The landing site already exists and is proven: `dgemma/composite.py:
StepEndComposite`'s `beta_rebuild` tuple slot, fixed **before** `pin`
(`dgemma/composite.py:17`, ADR-CDG-010 Decision 3 — "a pin's re-assertion could
be immediately overwritten by a renoise pass that doesn't know the cell was just
pinned"), and `dgemma/participants.py:BetaRebuildParticipant` is a real,
mutation-verified, per-run-stateless canvas-writer occupying that slot today —
constructed with a static `writes: tuple[RebuildWrite, ...]` and never wired
from any `run_diffusion` payload (`dgemma/participants.py:140-152`). This ADR's
whole implementation shape is: **decide the payload that builds this
participant's `writes` each step**, not build a new participant class from
scratch.

The data this kernel replays already exists: ADR-CDG-014 Tier-1 capture banks
`top_k_ids`/`top_k_weights` per position from **pre-pin logits**
(`dgemma/loop.py:756-766`, `_FrameCollector.on_step_end`) — the position's own
captured candidate distribution at the step it first became eligible. Capture
runs first in the composite (`capture -> cancellation -> beta_rebuild -> pin`),
so by construction this data is available to `beta_rebuild` on later steps
without inventing a second capture path.

One continuous channel already crosses today's shallow/uniform remelt: the
pipeline feeds `pred_logits` back as self-conditioning
(`pipeline_diffusion_gemma.py:380-383`), for renoised positions too — so
"remelt to uniform" is already not pure entropy at the belief layer, only at
the canvas layer. This ADR's shallow-remelt scope inherits that same asymmetry
deliberately (see Decision 6); it does not attempt to close it.

**A sibling ADR occupies the same slot.** ADR-CDG-016 (accepted 2026-07-18,
crystalline CA rule tables) also targets `beta_rebuild` — its P2 phase
("engine participants: neighbor-read -> mask rebuild per step; local re-melt
participant over β-renoise") is explicitly sequenced *strictly downstream* of
"CDG-010/011 Phases 3/4" (ADR-CDG-016 Decision 5), and both ADRs use `top_p`
in overlapping but distinct senses (CDG-016's `top_p` is a rule alphabet
source; #115's is a truncation shape over a captured distribution — the
"terminology note" in issue #115 already flags this collision was raised and
dismissed once before as a non-issue since the underlying question is
well-formed regardless of the shared word). This ADR does not build
CDG-016's neighbor-coupled rule engine; it builds the narrower, single-position,
non-neighbor-coupled kernel #115 asks for. Sequencing relative to CDG-016 is
named explicitly in the roadmap (Decision 8) so a future participant author
does not have to rediscover the collision.

## Decision

### 1. A new declarative payload, `RemeltSpec`, is the fourth ingress door, resolving ADR-CDG-010 Open Question 2

`run_diffusion` gains a `remelt: RemeltSpec | None = None` keyword-only
parameter (`dgemma/loop.py`), joining `constraints=`/`control_signals=`/
`capture=` as the fourth declarative payload validated at ingress
(`dgemma/ingress.py:validate_ingress`) before any scheduler/pipeline
construction — same discipline, same door, no exception. `RemeltSpec` lives
in `dgemma/payloads.py` beside `CaptureSpec`/`Constraints`/`ControlSignals`
(frozen dataclass; unknown-key rejection is structural via the constructor,
per `dgemma/ingress.py`'s existing "fail-on-unknown for keys is structural"
discipline). `RemeltSpec(kernel="off")` (or `None`) is a no-op, byte-identical
to today's `beta_rebuild=()` default — the same "empty == no-op" shape
`Constraints()`/`ControlSignals()` already establish.

This resolves Open Question 2 **narrowly**: the answer is "no, β-rebuild does
not need multiple *ordered sub-phases* per step for this kernel" — one
`RemeltKernelParticipant` occupies the slot, produces one set of `RebuildWrite`s
per step, same shape `BetaRebuildParticipant` already proves. The question
remains open for a *different* future body that might genuinely need ordered
sub-phases (e.g. CDG-016's neighbor-mask-then-local-remelt two-step); this ADR
answers it for kernel-based single-position remelt only, and updates
ADR-CDG-010's own Open Question 2 checkbox to note the narrow resolution and
point here.

### 2. Kernel menu: phase 1 ships `captured_top_k` only; two future kernels are named, not built, and the payload vocabulary must not foreclose them

```
RemeltSpec:
    kernel: Literal["off", "captured_top_k"] = "off"
    shape: TruncationShape          # see Decision 3
    positions: Literal["all_rejected"] | tuple[int, ...] = "all_rejected"  # see Decision 5
```

Phase 1 builds exactly one kernel body, `captured_top_k`: replay the rejected
position's own Tier-1 `top_k_ids`/`top_k_weights` from the step it was last
captured, apply the truncation/reheat shape (Decision 3), and sample a
replacement id from the resulting distribution. `kernel` is a closed
`Literal` — not an open string — specifically so ingress can reject an
unknown kernel name loudly (rule 5, `EMIT-CANONICAL / PARSE-AT-THE-DOOR`)
rather than silently no-op on a typo.

**Embedding-kNN** (ADR-CDG-015 seam) and **KV-derived** (ADR-CDG-012 seam)
are named future values of the same `kernel` literal — `"embedding_knn"`,
`"kv_derived"` — explicitly **not** added to the `Literal` this phase (adding
an unimplemented enum member that ingress accepts but no participant body
handles would be exactly the trust-and-degrade gap this pack's greenfield
discipline forbids). The vocabulary is shaped so adding either later is a
`Literal` widening plus a new kernel-dispatch branch in the participant body —
not a payload redesign: `RemeltSpec`'s `shape`/`positions` fields are kernel-
agnostic (a truncation shape and a position selector are meaningful regardless
of where the candidate distribution comes from), so neither future kernel
requires new top-level fields, only a new source for "the candidate
distribution at this position."

### 3. Radius/shape controls: `TruncationShape` is its own frozen dataclass, all five controls are phase-1 fields, validated as a closed combination

```
TruncationShape:
    top_k: int | None = None          # fixed candidate count, <= captured k
    top_p: float | None = None        # cumulative-mass prefix, (0, 1]
    min_p: float | None = None        # peak-relative floor, (0, 1]
    reheat_temperature: float = 1.0   # T; >1 pushes outward, must be > 0
    exclude_peak: bool = False        # forced displacement of the argmax candidate
```

All five are phase-1 payload fields — issue #115's own framing names them as
one coherent "truncation family applied to the captured anneal-time
distribution," and splitting them across phases would mean the pre-registered
probe (steps-to-freeze, candidate-shape collapse) cannot be run with the
`min_p` control the experiment design calls out as the "on-manifold guardrail"
survivable under reheat. Ingress validation (`dgemma.ingress.validate_remelt`,
same C/V/P/H register style as `validate_constraints`/`validate_control_signals`):

- **At most one of `top_k`/`top_p`/`min_p` may be set** (mutually exclusive
  truncation families — combining them is not rejected because it is
  mathematically incoherent, it is rejected because #115 does not name a
  combination semantics and guessing one is exactly the kind of undecided-
  contract this pack's discipline refuses to paper over). `RemeltSpec` with
  all three `None` uses the full captured top-k slice unmodified (reheat/
  exclude-peak still apply) — a legal, named "no truncation narrowing"
  configuration, not a reject.
- `top_k` (of the shape): positive int, `<= capture.top_k` (validated against
  the SAME `CaptureSpec.top_k` this run supplied — a shape-level top_k wider
  than the captured candidate set has nothing to truncate and is rejected
  with a remedy naming the captured `top_k`, never silently clamped).
- `top_p`: `float`, `0 < top_p <= 1.0`.
- `min_p`: `float`, `0 < min_p <= 1.0`.
- `reheat_temperature`: `float`, `> 0` (a `<= 0` divisor is undefined for
  `softmax(z / T)`, the same guard `t_min >= t_max` already applies elsewhere
  in this loop).
- `exclude_peak`: plain `bool`.
- **`kernel="captured_top_k"` with no captured data reachable is rejected at
  ingress, not degraded at runtime.** Concretely: `remelt is not None and
  remelt.kernel != "off"` requires `capture is not None and capture.top_k > 0`
  — the kernel's only phase-1 data source is Tier-1 capture, so a caller who
  enables the kernel without enabling Tier-1 capture gets a loud ingress
  reject naming the missing `capture.top_k` precondition, never a silent
  fallback to uniform remelt (which would look like the kernel is "on" while
  behaving exactly like today's default — the precise trust-and-degrade shape
  ADR-CDG-001 forbids).

### 4. Remelt depth: phase 1 is shallow-only; deep remelt is deferred with a named trigger, not decided here

**Shallow remelt** (canvas-only — writes `RebuildWrite`s the same way
`BetaRebuildParticipant`/`PinParticipant` already do) is the entire phase-1
scope. **Deep remelt** (canvas rewrite *and* ablation of the position's
self-conditioning slice in `pred_logits`) requires a seam this participant
cannot reach: `pred_logits` self-conditioning happens inside
`pipeline_diffusion_gemma.py:380-383`, upstream of `callback_on_step_end` —
no `StepEndParticipant` can reach into the pipeline's own forward-pass input
construction; that is pipeline-internal state, not `callback_kwargs`. Building
deep remelt would require either (a) a new pipeline-level hook analogous to
`install_logit_shaping_hook` but for the self-conditioning tensor rather than
the model's returned logits, or (b) a diffusers-side pipeline override
(`DGemmaPipeline`'s existing narrow-subclass discipline, `dgemma/loop.py:
399-416`, could plausibly grow this — it already widens one allowlist entry
for exactly this kind of reach).

This ADR does **not** decide which. Issue #115 itself names deep remelt as
"needed as the honest baseline arm eventually," which is a real requirement,
not a nice-to-have — but the seam it needs is genuinely undesigned (a
self-conditioning ablation hook has no existing precedent to extend the way
`beta_rebuild` extends `BetaRebuildParticipant`). **Deferred-with-trigger**:
the trigger is the pre-registration battery (issue #115's floor) actually
running its shallow-remelt arm and needing the pure-entropy control to
distinguish "kernel changed the trajectory" from "removing self-conditioning
changed the trajectory" — at that point a follow-on ADR designs the
self-conditioning ablation seam on its own merits (candidate seam: a second,
narrower forward-hook-like context manager parallel to
`install_logit_shaping_hook`, scoped to `dgemma/hooks.py`). Recorded as an
open question below, not guessed at here.

### 5. Which positions remelt: `accepted_index` identifies this-step-rejected positions; phase 1 offers only `"all_rejected"`

`scheduler_output.accepted_index` (`dgemma/loop.py:710`, already read by
`_FrameCollector.on_step_end` to derive `committed_fraction_per_example`) is
per-position boolean-like acceptance for the step just committed — the exact
signal needed to identify "this step's rejected positions" without inventing
a new read off the scheduler. The kernel participant, occupying `beta_rebuild`
(which runs on the SAME `callback_kwargs` the capture participant just read),
has access to the same `scheduler_output` and therefore the same
`accepted_index` tensor.

`RemeltSpec.positions` is phase-1-scoped to exactly one literal value,
`"all_rejected"` — apply the kernel to every position `accepted_index`
reports rejected this step, mirroring `PinParticipant`'s and
`BetaRebuildParticipant`'s existing "apply to every entry in my own spec"
shape. A declarative subset (e.g. "only reject-positions inside range
[a, b)") is a real, plausible future need (issue #115's probe is a
*single-position* case study) but is not required to run the pre-registered
battery, which observes one already-known position's behavior under the
kernel applied uniformly — adding a subset-selection sub-language now would
be scope creep against a requirement the floor experiment does not name.
`positions` is typed as `Literal["all_rejected"] | tuple[int, ...]` (not a
bare `Literal`) specifically so a later phase widening to declarative subsets
is additive (a new legal value for an already-typed field), not a payload
shape change.

### 6. Trace honesty: a `remelted_mask` rides every frame, parallel to `pinned_mask`, with the same static-derivation scope guard

Per ADR-CDG-010 Decision 4's precedent (`pinned_mask` distinguishes
model-committed from constraint-asserted cells) and the operator's explicit
requirement that "the pinned_mask discipline extends," `DiffusionFrame` gains
`remelted_mask: Any | None = None` — boolean `[gen_length]`, `True` at every
position the `RemeltKernelParticipant` rewrote **this frame**. Unlike
`pinned_mask`, this field is **NOT** derivable statically at construction
time: which positions are rejected (hence remelt-eligible) varies every step
by definition (that is the entire premise of a rejection-triggered rewrite,
unlike a hard pin's provably-constant position set — see
`DiffusionFrame.pinned_mask`'s own docstring for why the static shortcut was
valid there and only there). `remelted_mask` must therefore be built from the
participant's **actual per-step write**, observed post-hoc by capture — which
requires `remelted_mask` to be populated the same way `entropy`/`top_k_ids`
already are: read fresh each callback from data available on
`callback_kwargs`.

**This creates a real ordering question capture-before-writer does not
resolve for free.** Capture runs *first* in the composite (before
`beta_rebuild`), so capture-time `callback_kwargs` does not yet contain this
step's remelt write — capture is recording the model-committed truth
*before* any rewrite, which is correct for `entropy`/`top_k_ids`/`pinned_mask`
(they must reflect pre-remelt state) but is exactly wrong for
`remelted_mask`, which must reflect *this step's remelt*, an event that has
not happened yet at capture time. Resolution: `remelted_mask` is populated
from `accepted_index` at capture time as "which positions WOULD be remelt-
eligible this step" (the same shape `_build_pinned_mask`'s static-from-spec
precedent uses for pins) **only when `remelt.positions == "all_rejected"`**
— i.e. `remelted_mask` == the rejection mask == `~accepted_index` whenever
the kernel is on, computed from data capture already reads
(`scheduler_output.accepted_index`), not from observing the participant's
actual write. This is the same "static-derivation, valid only while a stated
scope-guard condition holds" shape `pinned_mask`'s docstring already
establishes for a different reason — labeled the same way: **when phase-1's
`positions` widens beyond `"all_rejected"` to a declarative subset, this
derivation must switch to observing the participant's actual write** (the
same labeled-door discipline `pinned_mask`'s docstring names for a future
dynamic/re-pinning constraint). Named as a constraint on any future
`positions` widening, not silently left to rot.

Absence semantics: `None` when no `remelt=` payload was supplied (additive-
optional, ADR-CDG-014 Decision 1) — never an all-`False` mask standing in for
"remelt was off."

### 7. Per-run statelessness: the kernel's memory lives entirely in the run's own captured frames, never cached cross-call

`RemeltKernelParticipant` (new class, `dgemma/participants.py`, same file and
shape family as `BetaRebuildParticipant`) holds only the immutable
`RemeltSpec` from THIS call's payload, exactly the state contract
`PinParticipant`/`BetaRebuildParticipant`/`WalkerParticipant` already prove
(ADR-CDG-010 Decision 7 / rule 6 `STATELESS-CORE`). The "memory" the kernel
replays — a position's captured top-k at the step it first froze — is **not**
new cross-call state: it is read from `callback_kwargs`/the collector's
already-captured `DiffusionFrame` history for THIS run only, the same
in-run-only lifetime `_FrameCollector.frames` already has. Concretely: the
participant needs "the most recent captured `top_k_ids`/`top_k_weights` for
position `p`" — the cheapest correct implementation is the participant
holding a reference to the SAME `_FrameCollector` instance `run_diffusion`
already constructs for this call (read-only access to `collector.frames`,
never mutating it), not a second parallel cache. `run_diffusion` builds a
fresh `_FrameCollector` and a fresh `RemeltKernelParticipant` every call
(the existing pattern every other participant follows), so nothing survives
between calls — two identical `run_diffusion(remelt=...)` calls on one loaded
model yield identical effective remelt telemetry, the same same-in/same-out
proof `tests/test_run_diffusion_statelessness.py` already runs for
pin/walker.

### 8. Composite ordering: the kernel occupies `beta_rebuild`, alongside (not replacing) ADR-CDG-016's future occupant

`RemeltKernelParticipant` is added to the `beta_rebuild` tuple
`run_diffusion` builds when `remelt is not None and remelt.kernel != "off"` —
same slot, same "before pin" guarantee `BetaRebuildParticipant` already has
proven (`tests/test_step_end_composite.py::TestBetaRebuildBeforePinRealParticipants`).
This ADR does **not** amend the fixed composite order; it is the first real
occupant of a slot that has existed since ADR-CDG-010 Phase 5 but never had
an ingress payload building it. **Relative to ADR-CDG-016**: that ADR's own
Decision 5 already sequences its P2 (neighbor-mask + local-remelt
participant) strictly downstream of "CDG-010/011 Phases 3/4" — this ADR
*is* that downstream landing, for the non-neighbor-coupled case. A future
CDG-016 implementer adding a neighbor-coupled remelt kernel should expect to
either (a) add a second, distinguishable `kernel` literal value scoped to
neighbor-coupled rules, sharing this ADR's `TruncationShape`/`positions`
machinery where it applies, or (b) find that CDG-016's rule-table payload is
different enough (a transition spec, not a truncation shape) to warrant its
own `RuleTableSpec` payload occupying the same slot as a sibling, not a
`RemeltSpec` variant. This ADR takes no position on which — it names the
fork so CDG-016's author does not have to rediscover this ADR's shape from
scratch.

### 9. Surface shape — ComfyUI: a dedicated payload-emitting node, `DGemmaRemeltSpec`, over widgets on the sampler

A new `DGEMMA_REMELT` socket, minted in `surfaces/comfyui/socket_types.py`
(rule 4, `IDENTITY⊥ENVELOPE`) alongside `DGEMMA_CONSTRAINTS`/
`DGEMMA_CONTROL_SIGNALS`. A new thin node, `DGemmaRemeltSpec`
(`surfaces/comfyui/remelt.py`), exposes `kernel`/`top_k`/`top_p`/`min_p`/
`reheat_temperature`/`exclude_peak` as widgets, constructs and emits a
`RemeltSpec` on its one `DGEMMA_REMELT` output — the SAME shape
`DGEMMA_CONSTRAINTS`/`DGEMMA_CONTROL_SIGNALS` already establish for
`Constraints`/`ControlSignals` (minted, unwired-to-a-node-input, socket-typed
payload objects). `DGemmaSampler` gains a new optional `DGEMMA_REMELT` input,
threaded straight to `run_diffusion(remelt=...)` — one line of plumbing, no
new logic in the sampler body (ADR-CDG-003's thin-adapter discipline).

**Rejected: widget-on-sampler.** Six new widgets crammed onto `DGemmaSampler`
directly (mirroring how `t_min`/`t_max`/`entropy_bound` are today) was
considered and rejected: `constraints=`/`control_signals=` already
established the "dedicated payload-emitting node, socket-typed, no widget
sprawl on the sampler" precedent the moment they needed anything beyond a
scalar (`Constraints`/`ControlSignals` are collections; a `TruncationShape`
is a small closed struct, same shape class). Widgets directly on the sampler
are reserved for the P1/P2 always-present scalar knob surface
(`t_min`/`t_max`/`entropy_bound`/`confidence`/`seed`/`gen_length`/`thinking`);
a feature this deep in the ADR-CDG-010/011/014 expansion vocabulary follows
that expansion's own established node shape, not the P1/P2 scalar-widget
shape it would otherwise have to retrofit onto a node whose widget list is
already long (`surfaces/comfyui/sampler.py:254-372`).

### 10. Surface shape — MCP: schema mirrors `RemeltSpec` exactly, unpacked by a thin `_unpack_remelt`, validated core-side only

`surfaces/mcp/commands/generate.py`'s `generate` tool schema gains a
`"remelt"` property, structurally identical to the existing
`"constraints"`/`"control_signals"`/`"capture"` properties: a JSON object
schema whose shape mirrors `RemeltSpec`/`TruncationShape` field-for-field,
`additionalProperties: False` at every object level (fail-on-unknown at the
JSON door, matching the frozen-dataclass fail-on-unknown at the core door).
A new `_unpack_remelt(raw) -> RemeltSpec | None` thin-unpack function
(same shape as `_unpack_constraints`/`_unpack_control_signals`/
`_unpack_capture`) does field mapping ONLY — every value/range/combination
check (mutual exclusivity of `top_k`/`top_p`/`min_p`, the `capture.top_k`
consistency check, `reheat_temperature > 0`) stays core-side in
`dgemma.ingress.validate_remelt`, never re-implemented in the MCP schema
(rule 5).

### 11. Term minting: extend `KNOB_DOCS`, do not invent a second mint

Per the operator's explicit requirement and the #111 precedent
(`dgemma/loop.py:KNOB_DOCS`, commit `96eea85`): every new knob this ADR
introduces (`kernel`, `top_k` [shape-level, distinct from
`CaptureSpec.top_k` — see naming note below], `top_p`, `min_p`,
`reheat_temperature`, `exclude_peak`, `remelt`/`positions`) gets one entry
added to the SAME `KNOB_DOCS` dict in `dgemma/loop.py`, wired identically
into (a) `DGemmaRemeltSpec`'s widget `"tooltip"` keys and (b) the MCP
`generate` tool's `"remelt"` schema property `description`s — the same
`tests/test_units_glossary_mint.py`-style identity check (`is`, not `==`)
extends to cover the new entries, asserting both doors read the literal
same `KNOB_DOCS[...]` object. **No second mint module, no re-typed prose at
either door.**

**Naming collision to resolve at implementation time, named here so it is
not silently guessed:** `TruncationShape.top_k` (a truncation count over the
*already-captured* candidate set) and `CaptureSpec.top_k` (how many
candidates to capture in the first place) share a bare name across two
different dataclasses with genuinely different semantics — issue #115's own
"terminology note" already flags a near-miss on exactly this collision
class ("top_k" meaning the captured set, not sampler truncation). `KNOB_DOCS`
entries must be keyed distinctly (e.g. `"remelt_top_k"` vs. the existing
`"capture_top_k"`-shaped key, exact key name is an implementation-phase
naming call, not a design-phase one) so the mint dict itself does not
collide two different meanings under one string key — the mint is only
ONE-MINT if its keys are unambiguous, not just its dict object.

## Rationale

### Positive Consequences

- **Resolves a named, tracked open question instead of leaving it to rot.**
  ADR-CDG-010 Open Question 2 has sat unresolved since 2026-07-13; this ADR
  gives it a narrow, honest answer (single-kernel, single-sub-phase, for this
  payload shape) rather than a blanket one, and updates that ADR's checkbox.
- **Reuses a proven slot and a proven participant shape.** `BetaRebuildParticipant`
  already exists, is mutation-tested, and already sits in the right
  composite position — this ADR's entire engine-side lift is "give it a real
  `writes` source instead of a static test-only tuple," not new composite
  machinery.
- **The Tier-1 capture investment (ADR-CDG-014) pays off immediately.**
  `top_k_ids`/`top_k_weights` were captured with no consumer besides
  observability until now; this is the first payload that *uses* captured
  data to drive behavior, not just report it.
- **Parity from birth forecloses a #103-class asymmetry.** Building the
  ComfyUI node and the MCP schema property in the same phase, from the same
  `RemeltSpec` dataclass and the same `KNOB_DOCS` mint, makes the rule-8
  asymmetry class structurally harder to reintroduce for this payload
  specifically (though rule 8's general enforcement remains review-only,
  per ARCHITECTURE.md's own honest "known-fragile" framing for that row).

### Negative Consequences

- **A closed `Literal["off", "captured_top_k"]` is a real constraint on
  future kernels.** Adding `"embedding_knn"`/`"kv_derived"` later is a
  breaking-adjacent change to the `Literal` (though additive at the type
  level) and requires touching ingress, the participant's kernel-dispatch
  branch, and both surfaces' schemas in lockstep — the same "widening a
  closed vocabulary is a real review event, not a silent extension" cost
  ADR-CDG-011's `MUTABLE_TARGETS` frozenset already accepts for the same
  reason.
- **`remelted_mask`'s static-from-rejection derivation is a labeled trap for
  the same reason `pinned_mask`'s is.** A future `positions` widening beyond
  `"all_rejected"` that does not also update the `remelted_mask` derivation
  produces a frame that looks honest but silently misreports which cells the
  kernel actually touched — the exact lying-trace failure mode ADR-CDG-010
  Decision 4 exists to prevent, now with a second instance to keep in sync.
- **Deep remelt is a known, named gap, not a solved one.** The
  pre-registration floor's own experiment design wants deep remelt as an
  "honest baseline arm eventually" — this ADR explicitly does not deliver
  it, so the battery in issue #115 can run its shallow-remelt arm now but
  cannot yet run the control arm that isolates canvas-kernel effect from
  self-conditioning-channel effect. That is a real limitation on what
  conclusions phase 1 alone can support, named rather than silently
  deferred past the point someone tries to draw that conclusion.
- **Two payloads race for the same slot's future.** ADR-CDG-016's
  neighbor-coupled rule-table work and this ADR's single-position kernel
  work both want `beta_rebuild`; Decision 8 names the fork but does not
  resolve it, so a future implementer of either could still make an
  incompatible assumption about how the slot's tuple-of-participants
  composes two conceptually different rewrite sources on the same step.

## Alternatives Considered

### Option A: Fold the remelt kernel into `CaptureSpec` (extend the existing capture payload) instead of a new `RemeltSpec`

**Why rejected:** `CaptureSpec` is an **observability** payload — ADR-CDG-014
Decision 7 places its ownership with the capture cluster specifically because
it controls what is *recorded*, never what is *written back to the canvas*.
A capture-triggering side effect that also rewrites canvas state would
violate that payload's own "captures never mutate" framing implicitly
carried by every existing Tier 0/1/2 field (`entropy`/`top_k_ids`/
`distribution` are all read-derived, never write-back). Blending "how much
to capture" and "what to remelt to" into one dataclass also breaks the ADR
that already resolved this exact split (constraints vs. capture are
deliberately separate payloads with separate ingress functions,
`dgemma/ingress.py:validate_constraints` vs. `validate_capture`) — a fourth
payload keeps that precedent rather than special-casing it away.

### Option B: A single `kernel="embedding_knn"`/`"kv_derived"` literal value shipped now, participant body raising `NotImplementedError`

**Why rejected:** this is the exact "unimplemented enum member ingress
accepts" trap Decision 2 names and refuses. A caller who selects
`kernel="embedding_knn"` today would pass ingress validation (the literal is
legal) and only discover the gap at participant-construction or run time —
a worse failure mode than an ingress-time reject naming an unrecognized
kernel string, because the caller's mistake (or the pack's incompleteness)
surfaces later and less legibly. Naming the future kernels in ADR prose
(Decision 2) gets the same "don't foreclose the vocabulary" benefit without
this cost.

### Option C: Widget-on-sampler surface shape (six new `DGemmaSampler` widgets) instead of a dedicated node

**Why rejected:** covered in Decision 9 — this repeats the precedent
`Constraints`/`ControlSignals` already set (dedicated node, socket-typed
payload) rather than the P1/P2 scalar-widget precedent, because the payload
shape here (a closed struct with a mutually-exclusive-field constraint) is
categorically like those two, not like a bare float/bool knob.

### Option D: Ship deep remelt (self-conditioning ablation) in phase 1, since issue #115 calls it the honest baseline

**Why rejected:** covered in Decision 4 — the seam deep remelt needs (reach
into `pipeline_diffusion_gemma.py:380-383`'s self-conditioning construction)
has no existing precedent to extend and is genuinely undesigned; guessing a
hook shape for it now, under this ADR's already-large scope, risks exactly
the "committing to a hook shape before the real requirement is understood"
mistake this pack's process discipline (independent design-gate review,
waterfall) exists to prevent. Naming it as deferred-with-trigger (Decision 4)
keeps the requirement visible without forcing a premature design.

### Option E: Derive `remelted_mask` by having the participant report its own write back to the collector (observed-write derivation, matching the labeled-door alternative `pinned_mask`'s docstring names for dynamic pins)

**Considered, not chosen for phase 1, but recorded as the eventual correct
shape if `positions` ever widens beyond `"all_rejected"`.** This is more
correct in general (it reflects the participant's actual write, not an
inference from `accepted_index`) but requires either restructuring the
composite's `dict` return-value threading (today's contract is "return
`{"canvas": ...}`," not "also report which positions you touched") or a
second out-of-band channel between `beta_rebuild` and capture — real
composite-shape work `StepEndComposite`'s docstring calls out as unchanged
by every addition so far ("this shape survived R5/R2 without reshaping").
Deferred to the future widening named in Decision 6, not built now, because
`"all_rejected"`'s static equivalence to `~accepted_index` makes it
unnecessary for phase 1's scope.

## Open Questions

- [ ] **Deep remelt's ablation seam (self-conditioning slice removal) —
  shape undecided.** Decision 4 names the fork (a new hook-lifecycle
  primitive parallel to `install_logit_shaping_hook`, or a `DGemmaPipeline`
  subclass override) but does not design either. **Resolution trigger:**
  the pre-registration battery's shallow-remelt arm running and the
  self-conditioning-vs-canvas-channel ablation sub-question (issue #115's
  own "Sub-question") becoming the next thing blocking a conclusion.
- [ ] **`RemeltSpec.positions` widening beyond `"all_rejected"` to a
  declarative subset** (issue #115's single-position probe is a case study,
  not yet a payload requirement). **Resolution trigger:** a real experiment
  needing to apply the kernel to fewer than all rejected positions in one
  step; when it lands, `remelted_mask`'s derivation (Decision 6) MUST switch
  from the static `~accepted_index` equivalence to observing the
  participant's actual write (Option E), not silently keep the phase-1
  shortcut past its valid scope.
- [ ] **`TruncationShape`'s exact `KNOB_DOCS` key names** (the `top_k`/
  `top_p`/`min_p`/`reheat_temperature`/`exclude_peak` collision with
  `CaptureSpec.top_k` named in Decision 11). **Resolution trigger:** first
  implementation PR — the plan pass downstream of this ADR picks concrete
  key strings; this ADR fixes only that they must be distinct, not what they
  are.
- [ ] **CDG-016 neighbor-coupled remelt's eventual payload shape** (Decision
  8's named-not-resolved fork: a `kernel` literal widening vs. a sibling
  `RuleTableSpec`). **Resolution trigger:** CDG-016 P1/P2 implementation
  actually starting (currently sequenced downstream of this ADR's Phases 3/4
  per that ADR's own Decision 5) — whichever author picks up CDG-016 next
  makes this call with both ADRs in hand.
- [ ] **Whether `RemeltKernelParticipant` reads `collector.frames` directly
  (Decision 7's proposed mechanism) or needs its own lighter-weight
  "last captured top-k per position" index structure** for performance at
  large `gen_length`/`num_inference_steps` — a plan-level question once
  profiling data exists, not a design-level fork (both are the same
  in-run-only, no-cross-call-state shape; this is a performance/ergonomics
  choice, not an architectural one). **Resolution trigger:** implementation
  phase 2 (below), if a direct `collector.frames` scan proves too slow.

**Resolution plan:** the first two are pre-registration-battery-triggered
(issue #115's own floor is what will surface them); the third and fifth are
implementation-phase calls the `plan` pass downstream of this ADR resolves;
the fourth is deferred to whoever picks up CDG-016 next, with this ADR as
required reading at that time.

## Phased Roadmap

**Phase 1 — Payload + ingress + engine participant (no surface exposure yet).**
Delivers: `RemeltSpec`/`TruncationShape` dataclasses in `dgemma/payloads.py`
(alongside `Constraints`/`ControlSignals`/`CaptureSpec`); `dgemma.ingress.
validate_remelt` in `dgemma/ingress.py` (mutual-exclusivity, range, and
`capture.top_k`-consistency checks, wired into `validate_ingress`'s single
call site); `RemeltKernelParticipant` in `dgemma/participants.py` (new class,
same file/shape family as `BetaRebuildParticipant`); `run_diffusion` gains
the `remelt=` keyword-only parameter (`dgemma/loop.py`), builds the
participant into the composite's `beta_rebuild` tuple exactly like
`constraints=`/`control_signals=` build `pin`/`walker` today
(`dgemma/loop.py:1386-1389,1403-1405` is the pattern to extend); `entropy`/
`top_k_ids`/`top_k_weights` reads already exist (ADR-CDG-014) and need no
change. Verifiable when: a direct `run_diffusion(remelt=RemeltSpec(
kernel="captured_top_k", ...))` call against the R4 fake-pipeline fixture
produces a different canvas trajectory than an identical call with
`remelt=None`, and a same-in/same-out statelessness test
(`tests/test_run_diffusion_statelessness.py`-style) passes for the new
participant. Depends on nothing beyond what is already landed
(`BetaRebuildParticipant`, Tier-1 capture, the ingress register pattern).

**Phase 2 — `remelted_mask` trace field + honesty test.** Delivers:
`DiffusionFrame.remelted_mask` (`dgemma/types.py`), populated in
`_FrameCollector.on_step_end` from `~accepted_index` whenever
`remelt.positions == "all_rejected"` and the kernel is on (Decision 6); a
trace-honesty test analogous to `TestPinnedMask::
test_pinned_mask_true_regardless_of_scheduler_commit_reading` proving
`remelted_mask` is `True` exactly at rejected positions and `False`
elsewhere, independent of what the kernel wrote there. Depends on Phase 1
(the participant must exist and be wired before its trace-honesty can be
tested against a real run).

**Phase 3 — ComfyUI surface: `DGEMMA_REMELT` socket + `DGemmaRemeltSpec`
node + sampler input.** Delivers: the socket mint entry
(`surfaces/comfyui/socket_types.py`); `DGemmaRemeltSpec`
(`surfaces/comfyui/remelt.py`), widgets sourcing tooltips from the extended
`KNOB_DOCS`; `DGemmaSampler`'s new optional `DGEMMA_REMELT` input threaded to
`run_diffusion(remelt=...)`. Verifiable when: a minimal ComfyUI graph
(pattern: `tests/test_kv_cache_cold_wiring.py`'s DV.3c non-degenerate-graph
proof) wires `DGemmaRemeltSpec` -> `DGemmaSampler` and produces a different
trace than the unwired baseline; `tests/test_socket_mint.py`'s grep-gate
extended to cover the new socket. Depends on Phase 1 (the payload/participant
must exist for the node to have something real to construct and pass).

**Phase 4 — MCP surface: `"remelt"` schema property + `_unpack_remelt`.**
Delivers: the schema property on the `generate` tool
(`surfaces/mcp/commands/generate.py`), mirroring `RemeltSpec`/
`TruncationShape` field-for-field with descriptions sourced from the same
extended `KNOB_DOCS`; `_unpack_remelt`, same thin-unpack shape as the three
existing `_unpack_*` functions. Verifiable when: an MCP `generate` call
carrying a `"remelt"` object produces the same effective run as the
equivalent direct `run_diffusion(remelt=...)` call (the same "parity by
construction" proof `tests/test_units_glossary_mint.py` already runs for the
scalar knobs, extended to this payload). Depends on Phase 1 only (does not
depend on Phase 3 — the two surfaces are peers, buildable in either order or
in parallel once Phase 1 lands; listed after Phase 3 here only because the
ComfyUI node's widget shape is a useful reference for the MCP schema's field
shape, not because of a real dependency).

**Phase 5 (research, outside this ADR's build scope) — pre-registration
battery.** Issue #115's own floor (null hypothesis, the `▁\`/`▁explot` probe,
per-position entropy/candidate-shape/steps-to-freeze measurables) runs
against the Phase 1-4 machinery. This ADR's phases build the instrument;
Phase 5 is the experiment the instrument enables, tracked on issue #115
directly, not re-specified here.

## Risk and Observability

- **Architectural bottleneck: `beta_rebuild` is now a contested slot with
  two ADRs' futures in it (this one and CDG-016).** No enforcement surface
  prevents a future participant addition from assuming it is the *only*
  canvas-writer in that tuple; `StepEndComposite`'s existing "each sees the
  previous writer's output via callback_kwargs threading" contract
  (`dgemma/composite.py:190-193`) already composes multiple `beta_rebuild`
  participants correctly in *sequence*, but nothing tests two *different*
  kinds of `beta_rebuild` participant (this ADR's kernel and a hypothetical
  CDG-016 rule engine) composing correctly in the SAME run. Named as a debt
  vector for whoever lands CDG-016 next (Decision 8's open question), not a
  bug in this ADR's own scope.
- **Security/resource exposure: none beyond the existing pattern.** No new
  external input crosses a trust boundary this ADR's ingress validation
  doesn't already gate (same shape as every prior declarative payload);
  `TruncationShape`'s numeric ranges are bounded the same way `Binding`'s
  `low`/`high`/signal-range checks already are.
  `capture.top_k`-consistency (Decision 3's last bullet) is the one new
  cross-payload validation this ADR introduces — it must read BOTH
  `capture` and `remelt` at the same `validate_ingress` call site
  (`dgemma/ingress.py:238-259`already threads all payloads through one
  function, so this is additive, not a new call path).
- **Technical-debt vector: the `top_k` naming collision (Decision 11) is
  exactly the kind of "two things sharing one bare name" drift this pack's
  own greenfield discipline exists to catch before it ships**, named here so
  implementation does not silently pick colliding key strings under time
  pressure.
- **Observability:** `remelted_mask` (Phase 2) is the primary new signal —
  without it, a `CanvasTrace` from a remelt-enabled run cannot distinguish
  "the model committed here" from "the kernel forced a redraw here," which
  is precisely the lying-trace failure ADR-CDG-010 Decision 4 already
  named for constraints and this ADR extends to remelt, per the operator's
  explicit requirement. Degradation path: if `remelted_mask`'s derivation is
  ever wrong (e.g. after the Decision 6 labeled-door widening is missed),
  the failure is silent — a plausible-looking but incorrect trace, not a
  crash — which is why Phase 2's honesty test is load-bearing, not optional
  polish.

## Preconditions Missing

None block writing this ADR. Two items are genuinely undecidable without
further input and are recorded as Open Questions above rather than guessed:
deep remelt's ablation-seam shape (Open Question 1) and CDG-016's eventual
payload relationship to this one (Open Question 4). Neither blocks Phase 1-4
of this ADR's own roadmap, which is why they are open questions rather than
blocking preconditions.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD.
**Amends:** ADR-CDG-010's Open Question 2 (narrow resolution: no ordered
sub-phases needed for a single kernel occupying one `beta_rebuild` slot
entry; the question remains open for a body that might need them).

## References

- Issue #115 — the originating idea, grounded scheduler mechanics
  (`scheduling_entropy_bound.py:162-172` renoise; `:153-155` anneal formula),
  the two-remelt-depth framing, the kernel menu, the pre-registration floor.
- `dgemma/composite.py:StepEndComposite` — the `beta_rebuild` slot and fixed
  composite order this ADR's participant occupies without amending.
- `dgemma/participants.py:BetaRebuildParticipant` — the existing, proven,
  never-wired participant shape this ADR's `RemeltKernelParticipant`
  parallels.
- `dgemma/loop.py:_FrameCollector.on_step_end` (lines ~707-808) — Tier-0/1/2
  capture-pre-pin derivation, this kernel's data source; `dgemma/loop.py:710`
  — `scheduler_output.accepted_index`, the rejected-position signal.
- ADR-CDG-010 — two-mechanism constraints, composite ordering, `pinned_mask`,
  Open Question 2 (this ADR's resolution target).
- ADR-CDG-011 — declarative-payload/engine-built-participant discipline,
  `MUTABLE_TARGETS` closed-vocabulary precedent (Decision 2's rationale for
  not open-stringing `kernel`).
- ADR-CDG-014 — frame capture discipline; Tier-1 `top_k_ids`/`top_k_weights`
  (Decision 3/4, issue #61 P-B) this kernel replays.
- ADR-CDG-015 — latent-field input-embedding seam; the `embedding_knn`
  future kernel's landing site (named, not built, Decision 2).
- ADR-CDG-012 — KV-cache seam; the `kv_derived` future kernel's landing site
  (named, not built, Decision 2).
- ADR-CDG-016 — crystalline CA rule-table payloads; sibling occupant of the
  same `beta_rebuild` slot, sequencing fork named in Decision 8/Open
  Question 4.
- Issue #111 / commit `96eea85` — the `KNOB_DOCS` ONE-MINT pattern this ADR
  extends (Decision 11).
- Issue #103 — the rule-8 ComfyUI/MCP parity asymmetry; the
  `constraints=`/`control_signals=`/`capture=` parity-by-construction
  pattern (`surfaces/mcp/commands/generate.py`) this ADR's surfaces
  (Decisions 9/10) transcribe for `remelt=`.
- `ARCHITECTURE.md` rules 4/5/6/7 — socket minting, payload honesty,
  cross-run statelessness, declarative-payload-only ingress; all four
  bind this ADR's shape directly.
