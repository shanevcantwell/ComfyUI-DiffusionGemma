# ADR-CDG-016 — Crystalline CA: neighbor-rule dynamics as declarative rule-table payloads (split-flap games, no field required)

**Status**: accepted (ratified by independent design-gate review, 2026-07-18; operator veto standing)
**Date**: 2026-07-18
**Related**: ADR-CDG-010 (two-mechanism model + composite ordering — the machinery this rides), ADR-CDG-011 (declarative-payloads-only door — the clause this widens), ADR-CDG-015 (the liquid arm this deliberately does **not** depend on), issue #28 (S-track — the global-constraint sibling), Track B R1 (β-renoise — the re-melt mechanism)

---

## Context

DiffusionGemma's update is **synchronous across all positions** — genuinely CA-like —
but its coupling is **global**: bidirectional attention over the full canvas. Global
synchronous coupling settling to fixed points is attractor dynamics (basin-settling),
not a cellular automaton; there is no locality to propagate structure through, so the
churn organizes into re-melts, not gliders.

A *liquid* CA — local rules over a held superposition field — is blocked on
ADR-CDG-015's fork (the field needs an input-embedding seam and probably training). But
locality can be **imposed at the rule layer over committed ids** — crystalline states —
needing no field at all. The mechanical register is the split-flap departure board: a
cell does not hold a blend of destinations; it flips through a discrete stack until it
lands. Each position's flap stack is its `top_p` nucleus (the per-cell alphabet); what
DG lacks is only the neighbor coupling, and that is impose-able through doors that
already exist, output-side, training-free.

Word games arrive pre-equipped with **viability predicates** — Carroll's doublets
("change one letter, must stay a word") is a birth/survival rule with a dictionary as
oracle; the semantic ladder ("change one word, must stay coherent") is scored by the
model itself. Each is a cheap, falsifiable probe of rule-coupled settling.

## Decision

1. **CA rules enter as declarative rule-table payloads through the existing rule-7
   door.** A rule table is data — alphabet, neighborhood, transition spec — validated
   at ingress; the **engine builds the participant**. No closure escape hatch: this
   reaffirms ADR-CDG-011's foreign-callable rejection and is exactly the payload shape
   that clause anticipated. A neighbor-*dependent* rule is a widening of the constraint
   vocabulary, not a new door.

2. **The neighbor rule is realized by ADR-CDG-010's two mechanisms, unchanged:** a
   dynamic logit mask — recomputed each step as a function of neighbors' current
   committed ids — shapes *what commits*; canvas re-assertion/pin holds rule-fixed
   cells (*what conditions*); β-renoise **local re-melt** is the propagation mechanism:
   a flip re-anneals its neighborhood and the model re-settles around it. Participant
   ordering per CDG-010 (capture → … → pin last writer) is inherited, not amended.

3. **Crystalline-first is deliberate and load-bearing:** this track must acquire **no
   dependency on the latent field** (ADR-CDG-015). If the field lands later, a liquid
   CA is a separate follow-on decision. The model is the physics, not the rule: the
   rule layer supplies locality; DG's global attention supplies semantic coherence in
   everything the rule leaves unconstrained — the global coupling stops being the
   CA-killer and becomes what keeps the board readable while the local game runs.

4. **Phase coupling comes free from the anneal schedule:** a rule declares its live
   window over steps (early = gas, mutations cheap; late = frozen). "Possibly-phased
   simpler mutations" is a schedule declaration, not new machinery.

5. **Sequenced strictly downstream of ADR-CDG-010/011 Phases 3/4** (the participant
   bodies, currently `NOT-YET-IMPLEMENTED`). This ADR adds scope to that vocabulary; it
   does not open a parallel implementation path.

## Phases

| Phase | Work | Depends on |
|---|---|---|
| **P1** | Rule-table payload schema + ingress validation: alphabet source (explicit id-set v1; `top_p`-nucleus source is an open question below), neighborhood (window k), transition spec, phase window (live-step range) | CDG-010/011 Phase 3 (payload→participant machinery) |
| **P2** | Engine participants: neighbor-read → mask rebuild per step (ordered per CDG-010); local re-melt participant over β-renoise | P1; Track B R1 (β mechanism) |
| **P3** | Word-game battery (doublets-class first) + mint **H0-ca** in `experiment.md`: *under a neighbor rule, DG settles into rule-consistent configurations unconstrained runs do not visit, and the settling is phase-dependent (rule live in the anneal window ≠ post-freeze)* | P2; observable via existing capture machinery (ADR-CDG-014 tiers) |

## Rationale

This is the cheapest falsifiable probe the CA thread admits: H0-ca is statable today,
rides sequenced work (CDG-010/011 Phases 3/4) rather than undesigned work (the field),
and its falsification banks cleanly ("neighbor rules do not produce distinct settling —
locality-at-the-rule-layer is inert on this substrate").

**Negative consequences:** the rule-table vocabulary grows the ingress surface
(more validators to hold); a rule table expressive enough for interesting games may
creep toward a DSL — bounded by P1 shipping the minimal transition-spec grammar and
each widening being its own review.

## Alternatives Considered

- **Surface-side CA** (a ComfyUI node loops the graph, applying rules between runs).
  Rejected: sequencing in a surface body is rule 8's instant-fail
  (`ARCHITECTURE.md` §instant-fail); multi-*run* games belong to the orchestration
  plane above the surfaces, per-*step* games belong core-side as payloads.
- **Closure-based rules** (surface passes a transition function). Rejected per
  ADR-CDG-011: unvalidatable at ingress; re-opens the foreclosed second door.
- **Windowed-attention locality** (mask attention to create true local coupling).
  Rejected for now: model surgery, off-manifold for trained weights, and unnecessary —
  rule-layer locality is training-free.
- **Wait for the field (liquid CA first).** Rejected: crystalline is independently
  falsifiable today; serializing it behind ADR-CDG-015's fork couples an unblocked
  program to a blocked one.

## Open Questions

- [ ] Alphabet-from-`top_p`-nucleus requires per-position candidate ids at rule-time —
  does that pull in Track B R0 / issues #14/#11? **Resolution:** P1 schema decides;
  v1 ships explicit id-sets only if nucleus sourcing would gate on R0.
- [ ] Flip scheduling within a step: all rule-eligible cells synchronously, or one flip
  per step (classic split-flap)? **Resolution:** P2 design; H0-ca's phase-dependence
  claim must be stated against the chosen scheduling before observation.
- [ ] Where does the word-validity oracle live — payload data (closed alphabet,
  leaning) or a consumer-side scorer? **Resolution:** P1; a consumer-side oracle would
  make the rule non-declarative and likely violates the rule-7 clause, so the burden of
  proof is on that option.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD — a future liquid-CA ADR (post-CDG-015) would extend, not
replace, this: the crystalline rule-table vocabulary is the substrate a field-coupled
rule would still enter through.

## References

- ADR-CDG-010/011 — the two mechanisms and the declarative door (both accepted,
  ratified 2026-07-13).
- `docs/experiments/liquid-phase-decoding/concept.md` — the phase vocabulary
  (steam/frozen/sublimation) this track's phase windows are declared against.
- Neural CA lineage (Distill, Mordvintsev et al.) — adjacent but distinct: a
  `top_p`-alphabet, neighbor-coupled semantic CA over a diffusion canvas was not found
  in a targeted 2026-07-18 search (novelty unverified; claim held loosely).
