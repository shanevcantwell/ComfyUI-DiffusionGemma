# Loose Ends Log

Tactical decisions that didn't qualify for ADR treatment but are worth
remembering. See `decisions/` for full ADRs.

**Created:** 2026-06-30
**Last updated:** 2026-06-30

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
