# ADR-CDG-015 — The latent field lives at the input-embedding seam: a seventh seam, a two-arm fork, gated by H0-hold

**Status**: accepted (ratified by independent design-gate review, 2026-07-18; operator veto standing)
**Date**: 2026-07-18
**Related**: ADR-CDG-001 (native payload types), ADR-CDG-002 (no-MASK confirmation — `mask_token_id=None`), ADR-CDG-010/011 (the declarative, *output-side* intervention doors this seam is deliberately not one of), ADR-CDG-012 (the mean-of-keys ≠ key-of-mean off-manifold hazard, here relocated), ADR-CDG-016 (the crystalline arm that deliberately does *not* depend on this), `docs/experiments/liquid-phase-decoding/concept.md` + `experiment.md`

---

## Context

The liquid-phase program's held-distribution faces (Track B rungs R4-observe, R5-project)
read the per-step distribution the loop computes — and the loop **discards it every
step**. USD's canvas stores token ids; the scheduler is stateless; every step computes a
distribution, samples an id, writes the id, renoises the rest with fresh random ids.
There is no state variable anywhere in the loop whose job is "the live top-k across
every position, persisted." The liquid object has no storage location. A position
flipping `\frac → explotfrac → \frac` is not resting in liquid; it is flipping between
two crystalline argmax landings.

Verified 2026-07-18 (exogenous fetch, not inherited from discussion): the held field
**exists in the literature**, absorbing-state only:

- **Soft-Masked Diffusion LMs** (arXiv 2510.17206, Hersche et al., IBM/ETH — accepted
  ICLR 2026; already cited at `concept.md:112,134` as a build-from pair). Holds, at each
  uncommitted position, a confidence-weighted convex combination of `[MASK]` and top-k
  predicted token embeddings — verbatim *"by superposing the [MASK] token and top-k
  predictions."* **Not training-free**: two-pass training; a learned scale climbs ~0→1
  during continued pretraining. The model is *taught* to consume soft input.
- **Latent Refinement Decoding** (arXiv 2510.11052, Zhu et al. — new to this record).
  Entropy-weighted mixing `ẽ = (1−α)·e_[MASK] + α·Σ p̄(v)·e_v`: high-entropy positions
  stay mask-like, confident positions firm up — a smooth per-position field. Reads as an
  inference-time drop-in over frozen LlaDA/Dream, but its training-free status is
  **unconfirmed** (not stated outright in the fetched material).

Both mechanisms anchor the blend on the `[MASK]` embedding — absorbing-state (MDLM)
machinery. DiffusionGemma is **uniform-state**: no mask token exists to blend toward
(ADR-CDG-002's resolved question). A targeted search found **no published instance of a
soft-superposition mechanism on a uniform-state diffusion LM** (two passes; "nothing
surfaced," not a confirmed negative).

## Decision

1. **The field's storage location is the input embedding — named as a seventh seam.**
   Not the canvas (stores ids), not the logit hook (biases the collapse but still
   collapses), not β-renoise (warms the gas but still samples an id). The six-seam
   inventory (`concept.md`: `DISTRIBUTION`, `SCHEDULE`, pin/mask, sampling operator,
   `KV_CACHE`, `CANVAS_STATE`) is all *output-side*; the input-embedding seam is the one
   place a held superposition can physically live. Phase 1 adds this row to the
   inventory. This names precisely why the existing training-free toolkit cannot hold
   the field: the toolkit is output-side; the field is input-side.

2. **Adopt the two-arm fork, both arms recorded, neither implemented by this ADR:**
   - **Inject arm (training-free):** feed a convex combination of token embeddings at
     uncommitted positions, zero training. The off-manifold risk is the mean-of-keys ≠
     key-of-mean hazard (banked for K/V blending, `concept.md:367` / ADR-CDG-012)
     relocated to the embedding input — DG's weights only ever saw hard ids there.
   - **Train arm:** transpose Soft-Masked Diffusion's recipe to USD with the
     **uniform-mixture anchor** `ē = E_uniform[e_v]` — the expected embedding of a
     renoised cell — replacing `e_[MASK]`: blend `(1−α)·ē + α·Σ p̄(v)·e_v`, learned
     scale via continued pretraining. *Steam as the anchor instead of mask.* The
     asymmetry vs. SMDM: `[MASK]` is a trained point, `ē` is not — which is exactly what
     the learned-scale continued pretraining exists to fix.

3. **Both arms gate behind H0-hold**, minted into `experiment.md` at Phase 1: *DG
   produces non-degenerate output when an uncommitted position's input embedding is a
   convex blend rather than a hard id, with zero training.* Cheap to falsify; its
   falsification banks "the inject arm is dead, only the train arm remains" — a path
   closed, not a failure.

4. **The train arm is operator-scheduled.** Finetuning the 26B model is a
   physical-infra precondition (GPU tenancy) and is explicitly *not* sequenced by this
   ADR. It becomes schedulable only after Phase 2's trigger fires.

## Phases

| Phase | Work | Trigger to proceed |
|---|---|---|
| **P1** | Mint H0-hold in `experiment.md`; add the seventh-seam row to `concept.md`'s inventory; one-off probe script (bench/consumer-tier — **no core door is added for a probe**) | this ADR ratified |
| **P2** | If H0-hold observes non-degenerate output: design the input-embedding door properly — declarative payload through the rule-7 register (ADR-CDG-011's pattern), native socket per ADR-CDG-001; measure the off-manifold degradation curve | H0-hold non-degenerate |
| **P3** | Train-arm design ADR + operator infra ask (USD soft-input continued pretraining with `ē` anchor) | inject-arm ceiling observed (P2's degradation curve banked) |

## Rationale

The program's rungs (R0–R6) do not need the field — they instrument the unmodified
model. But the existence proof landed (two published mechanisms, one at ICLR 2026), and
the USD transposition appears open. Recording the fork now, with the anchor candidate
named, converts a webchat insight into a falsifiable program without committing compute.
The alternative — leaving it in a chat log — is the exact failure mode the lab-notebook
floor exists to prevent.

**Negative consequences (anticipated-failure register, greenfield exception
harness-tools#18):** the inject arm may produce garbage (off-manifold), burning P1
effort — bounded by the probe being a script, not a door. The train arm, if reached,
is a heavyweight commitment on a model this repo treats as read-only weights today; P3
deliberately re-opens that as its own ADR rather than deciding it here.

## Alternatives Considered

- **Status quo (field-free program only).** The R-rungs stand alone. Rejected as the
  *only* posture because the existence proof changes the prior; retained as the default
  until H0-hold says otherwise — this ADR spends no compute.
- **Mask-token retrofit** (give DG a pseudo-`[MASK]` embedding to anchor toward).
  Rejected: reintroduces absorbing-state semantics into a model that never trained
  them; `ē` is the substrate-honest anchor (the gas state DG actually has).
- **Hold the field in K/V instead of the input embedding.** Rejected: ADR-CDG-012
  already banked the K/V blending off-manifold hazard, and the literature's working
  mechanisms are input-side.

## Open Questions

- [ ] Is LRD (2510.11052) actually training-free? **Resolution:** full-paper read
  before P2 design; if yes, its α-schedule is the inject arm's starting point.
- [ ] Is `ē` static (uniform expectation) or per-step (expectation under the current
  renoise distribution, which β-renoise reshapes)? **Resolution:** P2 design question;
  P1 probes static `ē` only.
- [ ] Does H0-hold need the `DISTRIBUTION` gate (Track B R0) to be observable, or is
  degeneracy readable from output text alone? **Resolution:** P1 names its observable
  before the probe runs (H0-before-observation floor).

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD — P3's train-arm ADR, if reached, will subsume the fork's
resolution.

## References

- arXiv 2510.17206 — *Soft-Masked Diffusion Language Models* (verified 2026-07-18:
  "superposing" verbatim; mask-anchored; two-pass training + learned scale; ICLR 2026
  accepted, OpenReview `Gba02UMvrG`).
- arXiv 2510.11052 — *Latent Refinement Decoding* (verified 2026-07-18: Eq. 3
  mask-anchored entropy-weighted blend; training-free status unconfirmed).
- Uniform-state search: nothing surfaced (2 targeted passes, 2026-07-18) — DUO-family
  work is sampling-efficiency, not a held field. Narrow search, not a confirmed negative.
