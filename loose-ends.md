# Loose Ends Log

Tactical decisions that didn't qualify for ADR treatment but are worth
remembering. See `decisions/` for full ADRs.

**Created:** 2026-06-30
**Last updated:** 2026-07-05

---

## 2026-06-30 — Modular RES4LYF-shaped node topology

**Category:** single-option (idiomatic)
**Related ADR:** ADR-CDG-001 (socket types), ADR-CDG-002 (access path)
**Graduation trigger:** If `CANVAS_STATE` chaining between nodes proves to need
tight coupling (shared mutable state that leaks across node boundaries),
revisit node-boundary design as an ADR.

### Context
Deciding how to decompose the pack into nodes. RES4LYF (and ComfyUI's
CustomSampler convention generally) splits into a schedule node, a
guides/conditioning node, a sampler node, and a chain of options nodes.

### Decision
Mirror that decomposition: **EntropySchedule → Constraints → Sampler**, with a
chainable `Options_*` family feeding an `options` socket — rather than one
monolithic node. (Payloads are the ADR-CDG-001 types, not `SIGMAS`/`LATENT`/`GUIDES`.)

### Why Not an ADR?
- [ ] Hard to reverse? → Somewhat, but it's a structural convention, not tech lock-in.
- [x] Surprising without context? → No. Modular CustomSampler-shaped nodes are the
      idiomatic ComfyUI pattern; a reader expects this, not the reverse.
- [x] Real trade-off? → Weak. Monolithic is possible but non-idiomatic; modular is
      the obvious platform-native choice, especially given the dev's deep ComfyUI
      familiarity and the goal of "insane combinations."

### Implementation Notes
- Node families: `DGemmaLoader`, `DGemmaEntropySchedule`, `DGemmaConstraints`,
  `DGemmaSampler`, `DGemmaTrace`, `DGemmaOptions_*`.
- `bongmath`-equivalent toggle on the sampler is `self_conditioning` (the
  documented loop feeds output logits back as self-conditioning for the next step).

---

## 2026-07-05 — DGemmaRenoise (text-to-text variation, SDEdit analogue): feasible-soft, priced

**Category:** feasibility ceiling (accepted)
**Related ADR:** ADR-CDG-004 (drive seam); ADR-CDG-001 addendum (scheduler-relative commit semantics)
**Graduation trigger:** If renoise becomes a headline node (not just an
experiment), it gets an ADR — the pipeline-subclass surface and strength
semantics are real design surface once something is built on them.

### Context
A text-to-text variation node — inject existing text into the canvas at some
corruption "strength" and diffuse from there, the SDEdit analogue for
discrete text diffusion. Grounded against the installed diffusers 0.39.0
pipeline: no seeding parameter exists on `DiffusionGemmaPipeline` — canvas
init is a hardcoded `torch.randint` with no injection point
(`pipeline_diffusion_gemma.py:346-348`).

### Decision
The soft version is feasible and priced: a ~50-100-line pipeline subclass
corrupting the user's injected text by `strength` for the canvas init.
**Hard-lock semantics are declined against a grounded ceiling, not a guess:**
injected text can only ever be evidence, not a hard constraint, because
`BlockRefinementScheduler` unconditionally resets `_committed` at step 0 of
every block (`scheduling_block_refinement.py:266`), and
`EntropyBoundScheduler` / `DiscreteDDIMScheduler` hold no commit state at all
to lock against. True hard-lock (injected tokens that cannot be renoised
away under any scheduler) would require vendoring or monkeypatching
third-party scheduler internals — declined. The soft version plus per-step
callback re-assertion (the same mechanism as P5 hard pinning,
`pipeline_diffusion_gemma.py:407`) covers the intended "variation" use case.

### Why Not an ADR?
- [ ] Hard to reverse? → No — a pipeline subclass, swappable/discardable.
- [ ] Surprising without context? → Somewhat, but this entry carries the
      grounding; no separate record needed while it's unbuilt.
- [x] Real trade-off? → Yes (soft vs. hard lock), but not yet load-bearing —
      nothing is built on it yet.

### Implementation Notes
- Candidate node: `DGemmaRenoise`. Not yet built.
- Graduates to an ADR when it ships as a headline node (the subclass surface
  and strength semantics need a decision record at that point).

---

## 2026-07-05 — Analyzer mode (entropy map of existing text): near-free, ~15-20 lines

**Category:** feasibility ceiling (accepted)
**Related ADR:** ADR-CDG-004 (drive seam)
**Graduation trigger:** Wire as a `DGemmaTrace` input mode or a tiny
standalone node once Phase 3's trace plumbing exists (P3-adjacent); promote to
an ADR only if it grows scheduler-touching logic of its own.

### Context
Running the entropy/temperature view over *existing* text (no diffusion
loop) — an analyzer, not a generator. Grounded against the installed
diffusers 0.39.0 pipeline: the loop body minus the loop — encoder KV populate
+ mask build + one forward (`pipeline_diffusion_gemma.py:318-343,364-371`) —
needs no scheduler at all, and the per-step temperature view is replicable
standalone from the anneal formula (`scheduling_entropy_bound.py:153-155`).

### Decision
Near-free: ~15-20 lines, no scheduler dependency. One caveat, flagged
UNGROUNDED during this pass rather than assumed: `modeling_diffusion_gemma.py`'s
encoder/mask-build call ordering was not independently verified — a quick
check before wiring is warranted.

### Why Not an ADR?
- [ ] Hard to reverse? → No — a small, isolated function.
- [ ] Surprising without context? → No, once this entry exists.
- [ ] Real trade-off? → None found; the only open cost is the ordering check
      above.

### Implementation Notes
- Candidate: a `DGemmaTrace` input mode, or a tiny separate node. P3-adjacent.
- Before wiring: verify `modeling_diffusion_gemma.py`'s encoder → mask-build
  call ordering (flagged ungrounded during this pass, not yet checked).
