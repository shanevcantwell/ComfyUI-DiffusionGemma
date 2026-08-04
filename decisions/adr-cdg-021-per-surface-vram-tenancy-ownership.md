# ADR-CDG-021 — Per-surface VRAM-lifecycle ownership: MCP→transformers (may OOM, fail-loud); ComfyUI surface delegates to comfy.model_management (must not OOM)

**Status**: `proposed` — **ratification authority: the OPERATOR** (per issue #229's `auto:draft` contract; the operator sets intent, reads conclusions, holds the veto). This is distinct from ADR-CDG-020's readback-gated pattern: CDG-021 is an **ordinary design proposal prompted by an operator observation** (see Context) — no standing operator directive exists; the operator holds the accept/reject call on it as on any lifecycle-policy proposal. Stops before implementation opens.
**Date**: 2026-08-04
**Related**:
- [Issue #229](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/229) — this ADR's tracking issue (the operator observation + four named resolution points)
- [Issue #226](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/226) — the `encode_sequence` bf16-CPU-spill OOM this ADR proposes reclassifying (bug-anywhere → bug-only-where-ComfyUI-fails-to-delegate)
- [Issue #228](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/228) — the S-B-PASS-vs-#226-FAIL provenance gap; the context divergence (bare transformers lane vs ComfyUI-server lane) this ADR converts from confusion into a named per-surface boundary
- [Issue #160](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/160) — lazy materialization / never-hit-VRAM offload (same directive family; relation decided below, §Decision-4)
- [Issue #92](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/92) (CLOSED) — ARCHITECTURE.md's "Lifecycle & tenancy — honest absence" section this ADR begins to fill
- ADR-CDG-019 / [#137](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/137) / [#138](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/138) — MCP-as-contract topology remediation; the seam this ADR reconciles (§Decision-3) **before the #138 bracket lands**
- ADR-CDG-003 (core is ComfyUI-agnostic — the seam every mechanism here preserves), ADR-CDG-002/004 (transformers load seam, diffusers drive seam)

---

## Context

CDG loads ~53 GB of weights via `dgemma.model.load_model` under accelerate's `device_map="auto"` (`dgemma/model.py:889`, grounded default `quant="none"`), which CPU-spills the overflow the 48 GB card cannot hold. The core is ComfyUI-agnostic and imports **zero** `comfy` — test-enforced out-of-process by `tests/test_seam.py` (`import dgemma` must pull in no `comfy`/`surfaces`/`consumers` module). Today there is **no lifecycle delegation**: whatever process calls `load_model` owns residency for its lifetime, single-tenant (ARCHITECTURE.md §"Lifecycle & tenancy — honest absence", lines 93–97).

Two same-day facts frame the observation this ADR builds on. Issue #226: `encode_sequence` OOMs **deterministically** under bf16 CPU-spill (45.33/47.26 GiB at raise) in the **bare transformers lane** — no `comfy.model_management` in the loop — while an equivalent no-cache `run_diffusion` succeeds in the identical VRAM state; the differential is an accelerate CPU-offload-hook materialization-shape difference between a bare module call and the pipeline's internal call. Issue #228: gate run 4's S-B result (that same `encode_sequence` **PASSING** under bf16 CPU-spill, `a68e29d`) ran through the **ComfyUI server** and was never banked as a function-scoped fact — so #226 was misfiled as "never worked" while a passing precedent sat in the record. The two runs differ in exactly one axis that matters: **which process, with or without ComfyUI's memory manager arbitrating VRAM.**

**Operator observation, verbatim (2026-08-04) — initially misread by the apparatus as a directive; corrected same day, see #229:** *"MCP->transformers, ComfyUI surfaces must hit ComfyUI. transformers may OOM but ComfyUI will not."*

This ADR proposes adopting the observed property as an architectural rule: **VRAM-lifecycle ownership is a per-surface property, not a core property.** The core stays ONE-DOOR and surface-agnostic (ADR-CDG-003 unchanged). What splits is *who owns the loaded model's residency* on each surface, and therefore *whether OOM is an allowed fail-loud outcome or a defect by definition* on that surface.

## Decision

### Decision-1 — Per-surface tenancy ownership is the architectural rule

Residency ownership is assigned per surface. This is a new lifecycle-plane rule; it does not touch the seven core rules.

1. **MCP / transformers lane — accelerate-owned; OOM is an allowed, fail-loud outcome.** The MCP surface (`StateManager` in `surfaces/mcp/state_manager.py:57`, `load()` → `dgemma.model.load_model`) drives the accelerate `device_map="auto"` path with **no memory arbiter above it**. Under a card that cannot hold the requested load, accelerate spills to CPU/disk; a call shape that then over-allocates (the #226 `encode_sequence` differential) is **permitted to raise `torch.OutOfMemoryError`** — loudly, at ingress/load, never silently degraded. This is `EMIT-CANONICAL / PARSE-AT-THE-DOOR` for memory: the lane does not pretend to fit what it cannot. #226's hardening lands **in this lane** as a fail-loud contract (a pre-flight fit check and/or a typed OOM at the `encode_sequence` door), not as a "make it always fit" fix — under this proposed rule, the bare lane OOMing is in-contract.
   - **Invariant (greenfield, names its anticipated failure):** *the MCP/transformers lane MUST fail loud — typed and at the earliest door — on a load or call it cannot satisfy; it MUST NOT silently spill, retry, or degrade into a partial/corrupt state.* **Anticipated failure prevented:** a silent CPU-spill that limps to a wrong or truncated result (or the opaque "Cannot copy out of meta tensor" class the `_assert_no_meta_tensors` guard already catches at load) — the trust-and-degrade failure ADR-CDG-001 forbids, reappearing at the memory plane. **Enforcement surface:** the existing `_assert_no_meta_tensors` load guard (`dgemma/model.py:475`) for the load-time half; a new typed-OOM/pre-flight assertion at the `encode_sequence` ingress for the call-time half (#226 hardening) — both core-side, ComfyUI-free.

2. **ComfyUI surface — delegates residency to `comfy.model_management`; OOM is a defect by definition.** The ComfyUI consumer MUST route the load through ComfyUI's own model manager (fine-grained offload / free-before-load), so that when a graph runs, ComfyUI has already made room. Under this delegation OOM is **not** an acceptable outcome: an OOM observed on the ComfyUI surface is a defect in the delegation wiring, not an allowed lane outcome. This adopts the observed ComfyUI property (*"ComfyUI will not [OOM]"* — the arbiter makes room) as a surface contract, rather than executing an instruction.
   - **Invariant (greenfield, names its anticipated failure):** *any weight materialization reachable from a ComfyUI node MUST be preceded by a `comfy.model_management` free/reserve call on the surface side; a ComfyUI-surface OOM is a defect ticket, never a closed-as-wontfix lane behavior.* **Anticipated failure prevented:** the #228 regression class — a ComfyUI-lane result silently reclassified against the wrong baseline because the surface bypassed ComfyUI's arbiter and inherited the bare lane's OOM. **Enforcement surface:** the delegation-shape test named in Decision-2; provisionally review-only for the "every materialization is preceded by a free/reserve" whole-graph property (honestly known-fragile prose until a mechanizable whole-graph check exists — named, not pretended).

**Why this split and not a uniform rule:** a single "never OOM" rule would force `comfy.model_management` semantics into the core or the MCP lane, which have no memory manager and are the deliberately un-ComfyUI path (the pack's load path is `from_pretrained` out of `HF_HOME`, ratified 2026-07-13, `surfaces/comfyui/loader.py:16`). A single "may OOM" rule would abandon the ComfyUI user to the exact #226 failure the S-B PASS proves is avoidable when ComfyUI arbitrates. The surfaces have genuinely different memory contracts; the rule follows the contract.

### Decision-2 — Mechanical delegation shape: surface-side lazy `comfy.model_management` free/reserve, mirroring the existing interrupt-closure idiom (NOT ModelPatcher registration)

The ComfyUI surface delegates via a **surface-side, lazily-imported `comfy.model_management` call issued before materialization**, structurally identical to the already-shipped interrupt closures (`_build_check_interrupted`, `surfaces/comfyui/loader.py:222`; `_build_should_cancel`, `surfaces/comfyui/sampler.py:169`): `import comfy.model_management` **inside** a closure, degrade-to-safe on `ImportError` (pytest/headless), never at module top.

Mechanically, on the ComfyUI load path (`DGemmaLoader.load`, `surfaces/comfyui/loader.py:293` / post-#138 `consumers/comfyui/loader.py`):
- Before the primitive/`load_model` call, a surface-side closure calls `comfy.model_management.free_memory(...)` / `unload_all_models()` (exact API pinned at implementation against the installed ComfyUI) to make room for the ~53 GB (or ~30 GB INT4) load — reserving against the same estimate accelerate will demand.
- The load itself stays core-side and unchanged; the core still only ever sees a path/`repo_id`, a `quant`, and the existing optional `check_interrupted` predicate. The surface passes an **additional optional closure** (a "reserve/free" hook) the same additive way `check_interrupted` was added (`dgemma/model.py:893`, `None` = no-op for every non-ComfyUI caller) — OR performs the free/reserve entirely surface-side *before* calling the primitive, requiring **no core signature change at all** (preferred: keeps the core untouched).

**What stays surface-side to preserve the seam test:** every `comfy.model_management` symbol. The core (`dgemma/`) touches none; `tests/test_seam.py`'s out-of-process `import dgemma` leak-check stays green because the delegation lives in the consumer, exactly where the interrupt closures already live. Post-#138, this closure is in `consumers/comfyui/loader.py`, and the CDG-019 contract-seam test (`tests/test_contract_seam.py`) must whitelist `comfy.model_management` as a **sanctioned surface-side import** (it is not a `dgemma.*` leak — the seam test forbids consumer→core direct imports, not consumer→comfy imports, which are expected).

**Rejected — ModelPatcher registration (wrap `DGemmaModel` in a `comfy.model_patcher.ModelPatcher` and hand it to `load_models_gpu`):** ComfyUI's `ModelPatcher`/`load_models_gpu` machinery assumes ComfyUI *owns placement* — it moves modules on/off GPU itself. But CDG's placement is **already owned by accelerate's `device_map="auto"`** with its own CPU-spill hooks (`AlignDevicesHook`, `dgemma/model.py:449-472`), and #119's tied-weight-under-split-placement fix (the note at `dgemma/model.py:35`; the load-time guard machinery at `dgemma/model.py:772` / `:868`) is fragile to a second mover. Two placement authorities on the same 53 GB load is the corruption surface #119 already had to harden against once. ModelPatcher would also force the core to expose a ComfyUI-shaped model object, breaking ADR-CDG-003. **Deciding factor:** delegation must mean "ask ComfyUI to make room," not "hand ComfyUI the steering wheel accelerate already holds."

**Rejected — memory-pressure callbacks (register a `dgemma`-side callback ComfyUI invokes under pressure to evict):** requires the core to hold an evict-me handle and cross-call mutable eviction state, violating rule 6 (STATELESS-CORE) and requiring the core to know about ComfyUI's pressure signal. Free-before-load is simpler and sufficient for single-tenant on one card: there is nothing to evict *to* — the whole point is to make room before, not to arbitrate during.

### Decision-3 — CDG-019 reconciliation: DEFERRED, with a named trigger tied to the #138 bracket, and a resolved constraint the bracket must honor

The tension is real and must be named, not silently resolved: CDG-019/#138 makes ComfyUI **consume `dgemma_mcp.primitives`** rather than import `dgemma/` directly. If the primitive runs **out-of-process** (JSON-RPC to an MCP server holding the weights), then the weights live in another process and `comfy.model_management` — which only manages *ComfyUI's own* process VRAM — **cannot** own their residency. That directly contradicts *"ComfyUI surfaces must hit ComfyUI."* If the primitive runs **in-process** (a plain Python call into `dgemma_mcp.primitives.load_model`, the `StateManager` living in the ComfyUI process), then `comfy.model_management` **can** own residency, and Decision-2's closure works unchanged.

**Proposed constraint (this ADR's input to #138, pending ratification):** to keep the observed ComfyUI-non-OOM property true under the CDG-019 topology, **this ADR proposes that the ComfyUI consumer's load path resolve to an in-process Python call into `dgemma_mcp.primitives`, NOT an out-of-process JSON-RPC round-trip.** CDG-019 already builds exactly this: its primitives layer is *callable as Python OR over JSON-RPC* (ADR-CDG-019 Decision, "Callable as Python OR over JSON-RPC"), and its import-rules table lets `consumers/comfyui/` import `dgemma_mcp.primitives` directly (in-process). So the reconciliation is **compatible by construction** — CDG-019's ComfyUI consumer is a Python-call consumer, not a network client. The out-of-process served-engine topology (ARCHITECTURE.md:97, issue #92) remains the separate, later, un-decided fork; when it is taken, *this* proposed constraint becomes its hardest one: a served-engine that moves weights out of the ComfyUI process **must** re-answer "how does ComfyUI not OOM" (candidate: ComfyUI reserves a proxy allocation, or the served engine reports pressure back — an ADR at that time, not now).

- **Trigger (named, tied to #138):** when the #138 topology bracket opens, its plan MUST cite this ADR and encode the in-process-Python-call constraint as an acceptance criterion of the ComfyUI import redirection (CDG-019 Phase 2). The contract-seam test (Phase 5) whitelist MUST include `comfy.model_management` surface-side.
- **Deferred (not decided here):** the served-engine / out-of-process case (#92). Its trigger is unchanged (a second concurrent surface needing the resident model); this ADR adds the requirement that *that* ADR carry the ComfyUI-non-OOM answer.

**Why defer rather than decide the served-engine case now:** deciding out-of-process residency arbitration before a second concurrent surface exists would be inventing an invariant with no observed violation and no consumer — the greenfield discipline's own limit. The in-process constraint is decidable now (it has a live consumer: the #138 bracket); the out-of-process constraint is not.

### Decision-4 — Relation to #160 (lazy materialization / never-hit-VRAM): KEEP SEPARATE, cross-referenced

#160 is the **same directive family** (operator memory-residency control) but a **different axis**: #160 is about *when* and *whether* weights ever reach VRAM (lazy-load defers materialization to first `run_diffusion`; never-hit-VRAM streams per-layer from RAM, targeting 24 GB/96 GB boxes). This ADR is about *who owns residency once materialization happens*. They compose: the ComfyUI delegation (Decision-2) is the substrate #160's never-hit-VRAM mode named as a candidate ("ComfyUI model-management integration are the candidate substrates", #160). Folding them would couple a *placement-mode* decision (#160, still a draft needing its own loader-knob surface and rule-6-under-laziness enforcement) to a *tenancy-ownership* decision (this ADR, decidable now).

**Decision:** keep #160 a separate ADR-draft. This ADR records the seam: **#160's never-hit-VRAM mode, when designed, delegates residency per Decision-1/2** (ComfyUI surface → `comfy.model_management`; MCP lane → accelerate/fail-loud). #160's lazy-materialization half interacts with the fail-loud invariant (Decision-1.1): a deferred materialization must still fail loud at first-run if it cannot fit — #160's ADR inherits that invariant rather than restating it.

### Decision-5 — ARCHITECTURE.md lifecycle-section update: named as downstream doc work, NOT performed here

ARCHITECTURE.md §"Lifecycle & tenancy — honest absence" (lines 93–97) currently states the tenancy plane as a pure absence ("no lifecycle delegation"). Post-ratification, that section must be updated to record the **per-surface split** this ADR decides: the absence is no longer uniform — the MCP lane's absence-of-delegation is now a *decision* (fail-loud, accelerate-owned), and the ComfyUI surface's delegation-to-`comfy.model_management` is a *named rule with its enforcement surface*. This doc change is **downstream work, tracked on #229, performed by the implementing PR — not by this ADR.** It also interplays with #92's served-engine trigger (Decision-3) and must preserve the "anticipated evolution" fork for the out-of-process case.

---

## Rationale

### Positive Consequences
- **#226 is reclassified, not left open-ended:** the bare-lane OOM becomes in-contract (fail-loud), and the ComfyUI-lane OOM becomes a defect with a delegation fix — the two lanes stop being confused (the #228 root cause).
- **Zero core change required** (Decision-2's preferred surface-side-before-primitive shape): the seam test stays green with no whitelist churn beyond `comfy.model_management` (already surface-side).
- **CDG-019 is unblocked, not blocked:** the reconciliation is compatible-by-construction (in-process Python call), and the #138 bracket gains a concrete acceptance criterion rather than a lurking contradiction.
- **Reuses a proven idiom:** the interrupt closures already prove the lazy-import/degrade-to-safe seam pattern works and stays test-clean.

### Negative Consequences
- **Two memory contracts to reason about** — a contributor must know which lane they are on to know whether an OOM is a bug. Mitigated by making the split explicit in ARCHITECTURE.md (Decision-5) and by the fail-loud typed error naming the lane.
- **The ComfyUI free/reserve estimate is approximate** — reserving against a ~53 GB estimate before an accelerate load whose exact spill split accelerate decides is best-effort; a mis-estimate could still pressure the card. Mitigated by free-before-load being strictly safer than no-free, and by OOM-on-ComfyUI being a defect that surfaces the mis-estimate loudly.
- **The out-of-process served-engine case is left unanswered** (deliberately, Decision-3) — a future topology change re-opens the ComfyUI-non-OOM question.

## Alternatives Considered

### Option A: Uniform "never OOM" — push `comfy.model_management` semantics into the core
Make the core memory-manager-aware so every lane benefits. **Rejected:** breaks ADR-CDG-003 (core imports zero comfy, test-enforced), forces a ComfyUI dependency into the deliberately-un-ComfyUI `from_pretrained` path, and the MCP/script/test lanes have no ComfyUI process to delegate to. Contradicts the observed split, under which the transformers lane is *allowed* to OOM (and the ADR's proposal to honor it).

### Option B: Uniform "may OOM everywhere" — document the bare lane as the only lane
Accept #226 as terminal, tell ComfyUI users to shrink the model. **Rejected:** the S-B PASS (#228, `a68e29d`) proves the ComfyUI lane *can* avoid the OOM when ComfyUI arbitrates, which is exactly the observed property this ADR proposes to adopt; abandoning it would be a service regression against demonstrated behavior.

### Option C: ModelPatcher registration for the ComfyUI surface
Covered in Decision-2. **Rejected:** two placement authorities (accelerate + ComfyUI) on one 53 GB load is the #119 corruption surface; forces a ComfyUI-shaped model object into the core.

### Option D: Decide the served-engine (out-of-process) residency arbitration now
Resolve the CDG-019 tension by fully specifying how an out-of-process MCP-served model reports pressure to ComfyUI. **Rejected:** no observed violation, no live consumer (no second concurrent surface exists yet) — inventing the invariant ahead of the failure, against the greenfield discipline's own limit. Deferred with a named trigger (Decision-3).

## Open Questions

- [ ] Exact `comfy.model_management` API for the free/reserve call (`free_memory` vs `unload_all_models` vs `load_models_gpu` with a size hint). **Resolution:** pinned at implementation against the installed ComfyUI version; the seam (surface-side lazy closure) is invariant regardless of which symbol.
- [ ] Whether the free/reserve is a core-side additive closure (like `check_interrupted`) or purely surface-side-before-primitive. **Resolution:** prefer surface-side-before-primitive (zero core change); decide finally in the #229 implementation plan.
- [ ] #226's `encode_sequence` fail-loud form — pre-flight fit check vs typed OOM at the door. **Resolution:** decided in #226's own hardening PR, which this ADR scopes to the MCP/transformers lane.

## Supersession Relationships

**Supersedes:** none.
**Amends ADR-CDG-019:** narrows the ComfyUI consumer's permitted transport from {in-process Python call, JSON-RPC} to {in-process Python call only}, for the VRAM-non-OOM reason in Decision-3 (an out-of-process JSON-RPC consumer puts the weights in another process where `comfy.model_management` cannot own residency). The reciprocal supersession note onto ADR-CDG-019 rides *this ADR's* ratification, mirroring CDG-019's own Phase-6 pattern of noting supersession onto ADR-CDG-008 — see Implementation Notes.
**Amends** ARCHITECTURE.md's uniform "honest absence" of lifecycle delegation (Decision-5, downstream doc work) — the absence becomes a per-surface decision.
**Superseded by:** TBD — the served-engine / out-of-process ADR (Decision-3, #92 trigger) will amend the ComfyUI-residency answer when weights leave the ComfyUI process.

## Implementation Notes

Downstream work, tracked on #229; **not performed by this ADR.** No code, ROADMAP, or ARCHITECTURE edit is made here.

| File | Change Type | Description |
|------|-------------|-------------|
| `surfaces/comfyui/loader.py` (post-#138 `consumers/comfyui/loader.py`) | Modified | Surface-side lazy `comfy.model_management` free/reserve closure before the load primitive (mirrors `_build_check_interrupted`) |
| `dgemma/model.py` (`encode_sequence` ingress in `dgemma/kv_cache.py`) | Modified | #226 fail-loud typed-OOM / pre-flight fit check — MCP/transformers lane only |
| `tests/test_contract_seam.py` (CDG-019 Phase 5) | Modified | Whitelist `comfy.model_management` as sanctioned surface-side import |
| `ARCHITECTURE.md` §Lifecycle & tenancy | Modified | Record the per-surface split (Decision-5) |
| #138 topology bracket plan | Constraint | Encode in-process-Python-call requirement as CDG-019 Phase 2 acceptance criterion (Decision-3) |
| `decisions/adr-cdg-019-mcp-as-contract-topology-remediation.md` | Doc-work (at ratification) | Add reciprocal amendment note: "ComfyUI-consumer transport narrowed to in-process Python call only by ADR-CDG-021 (VRAM-non-OOM)." Mirrors CDG-019's own Phase-6 supersession-note-onto-CDG-008 pattern. **Do NOT edit CDG-019 now — this edit rides CDG-021 ratification.** |
| Issue #138 (GitHub comment) | Durable-emission (at ratification) | Post the CDG-021->#138 acceptance-criterion cross-link as a comment on #138, so a cold reader of #138 finds the in-process-Python-call constraint without having to already know CDG-021 exists. |

## Gate findings resolved (2026-08-04)

Independent Opus gate: **PASS_WITH_FINDINGS** (source fidelity clean — zero drift; doctrine conformance clean; "compatible-by-construction" with CDG-019 adjudicated genuine). Four resolved recommendations applied to this file:

1. **(mechanical)** Supersession section now carries an explicit **"Amends ADR-CDG-019"** entry (transport narrowed {in-process, JSON-RPC} -> {in-process only}); Implementation Notes gains a reciprocal-amendment doc-work row onto CDG-019, rider on this ADR's ratification (CDG-019 itself is NOT edited now).
2. **(mechanical)** Decision-2 ModelPatcher-rejection #119 citation repathed `ARCHITECTURE.md:35` -> `dgemma/model.py:35` (+ guard machinery at `:772`/`:868`).
3. **(suggestion, applied)** Implementation Notes gains a durable-emission action: post the CDG-021->#138 acceptance-criterion cross-link as a comment on issue #138 at ratification.
4. **(cosmetic, applied)** References + Decision-2 body: sampler interrupt-closure cite corrected `surfaces/comfyui/sampler.py:180` -> `:169` (the `_build_should_cancel` def; `:180`/`:202` land inside the closure body).

---

## References
- Operator observation verbatim, 2026-08-04 (issue #229; provenance correction same day).
- `tests/test_seam.py` — core-imports-zero-ComfyUI, out-of-process enforcement.
- `surfaces/comfyui/loader.py:222-256`, `surfaces/comfyui/sampler.py:169-206` (the `_build_should_cancel` def is at `:169`; the `comfy.model_management` import lands inside the closure body at `:202`) — the proven lazy-import/degrade-to-safe closure idiom this ADR extends.
- `dgemma/model.py:449-517` — accelerate `device_map="auto"` placement + `_assert_no_meta_tensors` load guard (the fail-loud enforcement surface).
- ADR-CDG-019 — MCP-as-contract topology; the "callable as Python OR over JSON-RPC" property Decision-3 relies on.

---

## Provenance correction (2026-08-04)

**What was misread.** The quoted line in Context ("MCP->transformers, ComfyUI surfaces must hit ComfyUI. transformers may OOM but ComfyUI will not.") was an operator **observation** about how ComfyUI behaves, not a directive. The apparatus initially misread it as a ruling and the first draft of this ADR was framed to that brief. The operator clarified same day (see #229): no standing directive exists.

**What changed.** Authority framing only — every "ruling"/"directive"/"the operator's instruction" claim was recast as observation/proposal language, and the status line now describes this ADR as an ordinary lifecycle-policy proposal the operator accepts or rejects. **The design content is unchanged** (the per-surface split, both invariants, the mechanical delegation shape, the CDG-019 reconciliation, and the #160 relation all stand on the S-B PASS evidence and the observed behavior, not on any claimed instruction). The prior **Gate findings resolved (2026-08-04)** edits are also untouched in substance.

**Nothing binding had propagated.** The two rider actions this ADR names — the reciprocal supersession note onto ADR-CDG-019 and the CDG-021->#138 cross-link comment — were deferred to ratification and remain **unexecuted**; CDG-019 and #138 were never edited. So the misread did not leak past this file.
