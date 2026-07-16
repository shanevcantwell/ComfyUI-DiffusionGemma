# AR-side authorship arrives as KVs, not tokens — the field-determination hypothesis

**Status:** hypothesis note, pre-discriminating-evidence. **Date:** 2026-07-16 (session note, banked verbatim at operator direction).
**Anchors:** issue #40 (sweep results comment, 2026-07-16), issue #47, issue #62 / ADR-CDG-012, issue #14, issue #28, `docs/experiments/2026-07-15-dg-numeral-counts-update-in-response/`.

## The two readings of "who authors the content"

Fixed-seed `entropy_bound` sweeps (0.01–0.10, `count_numerals_2026-07-16T00-36-18_*` and `T01-09-06_*`) pose the question: can the schedule knob reach the *content* of what the model draws, or only its *trajectory*?

- **Reading A — canvas negotiation.** The numerals are negotiated on the canvas; trajectory participates in authorship. Mechanism: commit schedule → RNG-stream consumption pattern → conditioning drift. Predicts row variation across the sweep, in basins (per #10's discontinuous trajectory-selector shape).
- **Reading B — field determination.** The content is determined upstream of the first denoise step by the cache-conditioned prior. Predicts schedule-invariant, noise-invariant content.

## The operator's correction that reshapes Reading B

The AR hemisphere never hands the decoder a sequence — **it hands a KV field, not tokens**. So "the answer is already in the cache" is a type error: there are no tokens in the cache to find. Reading B properly stated: *the KV field shapes the decoder's logit landscape so strongly that the content is determined without ever being represented* — decided the way a ball atop a single-basin valley is decided, not the way a written message is.

Consequences:
1. **The cache cannot be audited for the answer — only perturbed.** B's influence is influence-without-representation; it is measurable only through what the decoder does when the field changes (swap, splice, full-attention-layer ablation, #47's over-provisioning). This is why the ADR-CDG-012 MITM apparatus is the discriminating instrument and nothing cheaper substitutes.
2. **Sweep 1 is B-fingerprinted evidence.** Ten distinct commit schedules imply ten distinct consumption patterns of the random stream — yet evidence rows came out byte-identical across all ten `entropy_bound` values. The content did not ride the noise. Whatever authored it lives in the deterministic part: prompt KVs + weights. (Caveat: that sweep's rows were degenerate sequential cycles — maximal prior territory.)
3. **The direct observable for "one basin" is per-position entropy at early steps.** If numeral positions show near-zero entropy from step 1 — before any canvas evidence exists — the field had already collapsed them; the liquid was never liquid there. High-entropy-then-late-collapse = negotiation happened. Tier-0 entropy is captured on every frame (ADR-CDG-014 P-A); the per-position *display* is the half-delivered issue #14.

## Discriminator ladder (instruments in dependency order)

1. **Row-comparison across fixed-seed sweeps** (no new instrument): identical rows under a randomness-demanding prompt → B strengthens; basin-varying rows → A retains authorship territory, and divergence location (shared-prefix length, late-committed positions) localizes where trajectory gets a vote.
2. **Per-position entropy at early steps** (needs #14's display half): counts basins directly.
3. **Per-token commit indices** (ADR-CDG-014 P-B-adjacent): turns block-level freeze inference into token-precision.
4. **KV perturbation** (#62 Phase 2+, #47): the only direct test — swap the field under a fixed canvas and watch content.

## H0 (falsifiable)

*Numeral-position content is invariant under commit-schedule perturbation at fixed seed.* Falsified by basin-varying evidence rows in the 2026-07-16T01-09-06 sweep (analysis pending at time of writing; predictions for that sweep were stated pre-observation in the session record and #40).
