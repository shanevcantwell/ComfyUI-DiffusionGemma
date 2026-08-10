# Experiment — liquid-state decoding v2: closures, restatements, and the field's equation of motion

**Status**: `registered` — Opus design-gate PASS 2026-08-10; landed from an identifier-free webchat draft (per the write-adrs discipline), identifiers resolved at write time
**Date**: 2026-08-10 (webchat session, co-framed with operator)
**Related**: the liquid-phase concept note ([`concept.md`](concept.md), same directory); [ADR-CDG-015](../../../decisions/adr-cdg-015-latent-field-input-embedding-seam.md) — the latent-field input-embedding seam (the seventh seam); [ADR-CDG-010](../../../decisions/adr-cdg-010-constraint-composite-and-pinned-mask.md) — the constraint composite / pinned mask (the logit door)
**Supersedes (by name, in [`experiment.md`](experiment.md)):** the original `experiment.md`'s H0-observe, H0-project, H0-control, H0-substrate, and H0-cache interpolation-form statements. H0-renoise and the seventh-seam inject-arm probe (H0-hold) carry forward unchanged. Original statements are RETAINED with dispositions — append-only discipline: nothing deleted, everything superseded in place.

**Framing carried in:** the held distribution is a *field* (persistent, coupled, with its own dynamics), not a sample from existing dynamics. The H0s are existence claims over an intervention space; each probe falsifies one coupling, never the field concept itself.

**Decisions embedded here that pass the three-part ADR test** (hard to reverse in an append-only record, surprising without context, real trade-off) — flagged for the library, described in words, deliberately unnumbered:
1. **Closure-by-argument as a verdict category** — extending the verdict set with `retired-a-priori`, trading "run everything" completeness for not spending compute on mathematically pre-decided clauses. A future decision record on closure-by-argument verdicts should be minted through the live library if ratified.
2. **The field's explicit equation of motion** (Part 3) — adding a state variable to the loop is a reversible probe now but becomes load-bearing if H0-eom confirms; graduation to a decision record rides the existing trigger (confirmed H0 → ADR).

---

## Part 1 — Closures (retired-a-priori, arguments banked)

### C1. Original H0-observe, clause 1 — "equal scalar entropy, materially different candidate sets" — RETIRED-A-PRIORI

*Argument.* Entropy is a many-to-one map from the (K−1)-simplex to a scalar; its level sets are (K−2)-dimensional manifolds (K ≈ 256k). Equal-entropy-different-support is generic — falsification would require the model's predictives to occupy a measure-zero family. A dimension count wearing a hypothesis's clothes. **No experiment runs on it.**

*Residue that survives:* the scalar-vs-distribution question, restated as a predictive-value claim → H0-observe′.

### C2. Original H0-observe, clause 2 (averaged form) — "distributions narrow as context freezes" — RETIRED-A-PRIORI

*Argument.* H(X|C) ≤ H(X): conditioning reduces entropy **in expectation** for any model that uses context at all. The averaged claim is a theorem. Per-instance behavior is NOT guaranteed (entropy may rise at a given position as context commits) — that residue is real and moves into H0-observe′.

### C3. Original H0-project, falsification branch "directed operator yields the same token regardless" — RETIRED-A-PRIORI

*Argument.* Any reweighting/masking operator changes the sampled token by construction whenever off-mode mass exists; invariance requires a distribution degenerate on one mode — which is the *frozen* outcome, already owned by H0-control. The branch did no independent work. The contentful claim (coherence under projection) moves into H0-project′ with the metric it was missing.

---

## Part 2 — Restatements (predictions before observation; every load-bearing term operationalized or marked ⟨operator-set⟩)

### H0-control′ — self-conditioning dominance: the basin is a memory phenomenon, and the null is i.i.d.

*Structural theorem this is built on (banked 2026-08-10).* At a held position whose neighborhood is frozen, with uniform renoise and **no** cross-step memory channel, the per-step predictive is a fixed function of unchanged context plus a fresh uniform draw — the predictive sequence is i.i.d. An i.i.d. sequence cannot settle, narrow, or wander coherently. Any liquid behavior is therefore carried by the self-conditioning channel (`pred_logits`) or by neighborhood change — nowhere else exists (concept note, addendum item 3). H0-control′ tests the *strength* of that carrier, not the abstract existence of a basin.

*Protocol.* Hold ⟨n⟩ positions per run under the balanced heat/threshold regime. Two arms, identical seeds: **(A)** intact self-conditioning; **(B) ablated** — `pred_logits` at held positions zeroed or shuffled each step via the engine-installed forward hook (declarative payload only, per the architecture review's constraint — never a surface-built closure).

*Pre-registered observables.* **Flip rate** f (per-step top-1 changes at held positions). **Memory index** M = (f_ablated − f_intact) / f_ablated — the stability fraction attributable to self-conditioning. **Candidate radius** r — mean pairwise cosine distance among top-⟨k⟩ candidates in the model's own input-embedding space (the pre-registered geometry for the whole program; see H0-project′).

*Prediction.* Intact arm: M > ⟨operator-set; candidate default 0.3⟩ AND r stays below ⟨r_liquid⟩ across the hold (mobile within a tight embedding neighborhood). Ablated arm: statistics consistent with the i.i.d. null (M ≈ 0 mechanically; this arm calibrates the null's noise floor for every downstream H0).

*Falsified if* M ≤ threshold (self-conditioning too weak to dominate renoise — the sublimation is not separable by this lever, now quantified), or intact-arm r rises to the steam regime, or held positions collapse immediately at every tested heat/threshold pair. **Every branch yields a number, not a shrug.**

### H0-observe′ — narrowing is instance-level and *local*: the architecture makes a "where" prediction

*Prediction (two clauses, both falsifiable).*
1. **Instance-level narrowing:** the fraction of held positions whose predictive entropy net-decreases over the hold as neighborhood commits exceeds ⟨operator-set; candidate 0.7⟩. (The theorem guarantees only the average; instances may anti-narrow. Counting them is the experiment.)
2. **Locality:** per-position entropy drop correlates with commits inside the position's **sliding-attention window** (~1K, 25 of 30 layers) more strongly than with equidistant-in-sequence commits outside it — Spearman ρ_inside − ρ_outside > ⟨margin⟩. The verified 25-sliding/5-global split predicts narrowing is *recency-and-window-shaped*, not uniform. First H0 in the program that uses the verified architecture to say **where**, not just whether.

*Falsified if* narrowing is no more frequent than anti-narrowing (clause 1), or the window-locality signal is absent/inverted (clause 2 — itself a bankable finding: the 5 global layers dominate conditioning flow).

*Predictive-value rider (successor to closure C1):* mutual information I(top-k set identity; committed token's embedding-cluster) − I(scalar entropy; same) > ⟨margin⟩ on held positions. If set identity buys no prediction over the scalar, the per-position-entropy heatmap's shadow is the whole signal *in the only sense that matters* — and DISTRIBUTION capture is telemetry, not a field readout.

### H0-project′ — modes exist only in a geometry; coherence and attribute-shift trade against a budget

*Operationalization (pre-registered, closing the "multi-modal" gap).* A categorical distribution has no modes without a metric. **Geometry:** the model's own input-embedding matrix, cosine distance. **Mode criterion:** cluster the top-⟨k⟩ candidates (weights carried); ≥2 clusters with silhouette > ⟨s; candidate 0.4⟩, each cluster mass > ⟨m; candidate 0.1⟩ ⇒ the position is multi-modal *in this geometry*. Terms of art now carry constraints: "multi-modal" means this and nothing else.

*Prediction.*
1. **Incidence:** ≥ ⟨fraction⟩ of held-liquid positions are multi-modal per the criterion.
2. **Projection with a coherence budget:** masking sampling to a chosen cluster yields committed text whose fluency degradation (ΔPPL under ⟨ONE pre-registered judge: the model's own re-score OR an external LM — operator fixes one before running⟩) stays ≤ ⟨ε⟩ **while** a measured style attribute (pre-registered register/tense classifier) shifts by ≥ ⟨δ⟩. The claim is the *conjunction* — either half alone is cheap.
3. **Freeze-time ordering (carried forward — already falsifiable):** register-projectable positions outnumber tense-projectable positions per run (the attribute-schedule finding of *Steering Without Breaking*, arXiv:2605.10971, with "projectable" now defined by clauses 1–2).

*Falsified if* incidence is below floor **in the pre-registered geometry** (banked as: modes don't live in input-embedding space — try ⟨one⟩ alternative geometry, then stop), or no (ε, δ) pair satisfies the conjunction (style lives in trajectory/guidance, not distribution shape), or the register/tense ordering inverts.

### H0-substrate-{a,b,c} — the split: three registered variants replace one unfalsifiable placeholder

The original H0-substrate was, by its own admission, not yet a hypothesis ("the measure chosen determines what this H0 tests"), and the three candidate measures are **provably non-equivalent**. Disposition: register all three; the operator decision becomes *selection*, not rewriting.

- **-a (co-liquidity):** a non-causal-prefill diffusion LM (candidate: LLaDA) sustains more *simultaneously* held positions than DG at matched hold protocol.
- **-b (per-position breadth):** its held positions show higher multi-modality incidence (H0-project′'s criterion — the geometry ports).
- **-c (cross-seed diversity):** fixed prompt, matched sampling: higher full-sequence diversity (⟨pre-registered metric: self-BLEU or embedding dispersion — pick one⟩).

*Confound (carried, still binding):* mask-vs-uniform noise and scratch-vs-adapted training move together across the model pair. Attribution stays at the causal-prefill *mechanism*, never "AR heritage in the weights" (training provenance undocumented). Variants falsify independently; any single confirmation rescues the differential-diagnosis role.

### H0-cache-B′ — interpolation, instrumented at the two sites where linearity actually breaks

*Mathematical grounding (banked 2026-08-10, replacing the undifferentiated "off-manifold" hazard).* Attention logits are bilinear in K: q·(αK_A+(1−α)K_B) = α(q·K_A)+(1−α)(q·K_B) — **pre-softmax scores interpolate exactly**, and RoPE stays consistent (matched positions, same rotations). Nonlinearity enters at exactly two sites: **(i)** softmax renormalization (Jensen), **(ii)** the pairing of resulting attention weights with *blended* V rows. Mean-of-keys ≠ key-of-mean is real but **localized**.

*Protocol.* α-sweep on ⟨two pre-registered contrasting prefills⟩. Instrument per layer: D_softmax(α) = KL(attention weights under blended cache ‖ α-mixture of single-cache attention weights); D_pair(α) = divergence attributable to the V-blend at matched weights. Output metrics: fluency (same judge as H0-project′), style attribute (same classifier).

*Prediction.* Style attribute responds **monotonically** in α with fluency degradation ≤ ⟨ε⟩ across an interior α-range of width ≥ ⟨w⟩; degradation, where it occurs, **correlates with D_softmax + D_pair** — breakage at the predicted sites, not diffuse.

*Falsified if* the attribute response is non-monotone/chaotic, or fluency collapses at all interior α (no usable knob), or degradation occurs *without* the site-divergences moving (breakage somewhere bilinearity didn't predict — itself a bankable finding about attention geometry).

*The concatenation form carries forward unchanged from the original, needing only its metric fixed: ⟨judge + statistical test, operator-set⟩.*

---

## Part 3 — New: the field gets an equation of motion

### H0-eom — an explicit memory field with decay λ produces settling dynamics that vanish at λ=0

*Why this exists.* The field framing demands dynamics, not just storage and couplings. Currently the field's evolution is whatever self-conditioning happens to do, uninspected. Make it explicit: **F ← λF + (1−λ)·p_t** at held positions — F an exponential moving average of the per-step predictives. λ is the field's literal viscosity/memory constant; **λ=0 recovers the i.i.d. null by construction**, unifying this H0 with H0-control′'s ablation arm.

*Two arms, sequenced by seam availability.*
- **H0-eom-logit (runnable NOW):** F lives in logit space, blended into the position's logits through the logit door — the forward-hook path established by [ADR-CDG-010](../../../decisions/adr-cdg-010-constraint-composite-and-pinned-mask.md) — before the commit rule reads them. No new seam; declarative payload; one state tensor, one decay constant, one line in the participant.
- **H0-eom-embed (GATED):** F re-expressed as a convex embedding blend at the input — the seventh seam's inject arm *with dynamics*, per [ADR-CDG-015](../../../decisions/adr-cdg-015-latent-field-input-embedding-seam.md). Not run until that decision's inject-arm probe (H0-hold) observes non-degenerate output. If H0-hold falsifies, the logit arm is the whole program — banked in advance.

*Prediction (logit arm).* Across a λ-sweep at fixed β and hold protocol: flip rate f(λ) decreases monotonically; candidate radius r contracts with an empirical time-constant increasing in 1/(1−λ); a ⟨λ-range⟩ exists where positions are mobile-but-tight (liquid per H0-control′'s r_liquid) that does not exist at λ=0.

*Pre-stated failure modes.* **(1) λ-invariance:** statistics flat in λ ⇒ either self-conditioning already saturates the memory channel (compare against H0-control′'s M) or the commit rule's entropy read is insensitive to the blend — both bankable. **(2) Premature crystallization:** F sharpens logits, entropy drops, the cascade fires — the self-collapse mode shared with H0-renoise. β and λ are the two viscosity knobs, but the 2D sweep is explicitly OUT of scope until each 1D axis is banked separately (discipline: one knob per probe).

*Falsified if* no λ opens a held intermediate absent at λ=0 under any tested (heat, threshold) pair — then memory is not the missing term, and the basin, if it exists anywhere, needs a coupling this program hasn't named.

---

## Part 4 — Carried unchanged

- **H0-renoise** — as pre-registered 2026-07-13. Best-formed probe in the program; one caveat now banked alongside it: no intermediate-value argument guarantees a phase between β→1 steam and β→0 self-collapse — first-order transitions are permitted; the sweep is the only oracle. The self-collapse branch has independent a-priori support (self-sampling loops contract toward modes).
- **H0-hold** (the seventh-seam inject-arm probe, minted in [ADR-CDG-015](../../../decisions/adr-cdg-015-latent-field-input-embedding-seam.md); that ADR's P1 mint into the experiment record had not been executed as of this landing — this entry completes it) — unchanged; now additionally load-bearing as H0-eom-embed's gate, and its outcome doubles as archaeology on the model's training recipe: a Duo-style soft-input curriculum would pass it trivially (banked 2026-08-10 session).

---

## Open Questions

- [ ] Operator commits every ⟨operator-set⟩ threshold **before the first run** of the H0 that uses it. **Resolution trigger:** pre-registration is void for any H0 executed with unset thresholds; the observation table refuses the row.
- [x] Verdict-vocabulary extension (`retired-a-priori`) ratified? **RESOLVED 2026-08-10:** ratified by operator at write time. Part 1's closures stand as `retired-a-priori`; the flagged closure-by-argument decision record remains a candidate, mintable through the live library.
- [ ] Judge selection for H0-project′/H0-cache-B′ (self-re-score vs external LM). **Resolution:** operator fixes ONE before either H0 runs; both H0s must use the same judge or the shared-metric comparison is void.

## Supersession Relationships

**Supersedes:** the original liquid-state experiment record's ([`experiment.md`](experiment.md)) H0-observe, H0-project, H0-control, H0-substrate, and H0-cache interpolation-form statements (by name; retained there with dispositions and retirement rows per append-only discipline).
**Superseded by:** TBD (a confirmed H0-eom would graduate the equation-of-motion into a decision record via the existing graduation trigger; that record gets its identifier when it exists, and this line is updated then).

## Write-time handoff (for the receiving model — per the write-adrs library-addition procedure)

This draft is **identifier-free by design**. Do not trust, copy, or mint from any code-shaped string; there should be none. At write time:

1. Read the live decision library (`decisions/`) and resolve every "(resolve identifier at write time)" reference — the latent-field input-embedding-seam decision and the constraint-composite / pinned-mask decision — to their real, existing identifiers. If either title cannot be matched to exactly one live record, **STOP and ask** — do not guess, do not mint.
2. This document itself lives in `docs/experiments/liquid-phase-decoding/` — an unnumbered library. Its filename is a slug decision (suggested: keep `experiment.md`, retaining the original content per Part 1's append-only structure, or land as a sibling if the operator prefers) — operator call, not yours.
3. Append the three retirement rows to the observation table; never edit the original H0 prose.
4. The two flagged ADR-candidate decisions (front matter) are **not minted here**. If the operator ratifies either, mint it through the live library as its own record, then update this document's references to it.
5. Verify: grep the landed document for identifier residue (digit-prefixed record names, decision-record prefixes followed by digits, placeholder patterns). External live references — issue-tracker numbers, arXiv IDs — are out of scope; they are externally minted and real.

**Executed — write-time record (2026-08-10):**
1. Identifier resolution: "latent-field input-embedding-seam decision" → ADR-CDG-015; "constraint-composite / pinned-mask decision" → ADR-CDG-010. Each matched exactly one live record; no minting.
2. Landing shape: sibling `experiment-v2.md` — operator call at write time. The ecosystem-level supersession-shape convention question is banked as an operating-doctrine issue (same session).
3. Three retirement rows appended to `experiment.md`'s observation table; original H0 prose untouched; supersession banner and verdict-vocabulary note added append-only.
4. Neither flagged ADR-candidate minted; both remain candidates in the live library's queue.
5. Residue grep: `grep -nE "(ADR-[A-Z]+-TBD|XXX|TODO|resolve (identifier|to its library identifier) at write time|⟨resolve)"` against this document — one hit, at the "Write-time handoff" section's own procedural instruction text (item 1 above, describing the resolution step performed), not an unresolved live placeholder; both flagged decisions are already resolved in the front matter and body to ADR-CDG-015 / ADR-CDG-010. No unresolved identifier residue found.
6. Defect found and repaired during resolution: ADR-CDG-015 P1's "mint H0-hold in `experiment.md`" was never executed — H0-hold existed only in the ADR and a ROADMAP row. Part 4's H0-hold entry completes the mint.
7. Design-gate ratification (2026-08-10, independent Opus reviewer, PASS, no blocking findings). Resolved recommendation, recorded as a decision: the file's ~4% overage past the ~5k-token salience ceiling (char-estimate) is accepted — this is a front-loaded pre-registration record, read once per program, not iterated hot; if the record grows further, the named split point is the Part 2 restatements.

## Room for observations

Append-only. Never retro-fit a prediction to a result. Verdict ∈ {untested, observed, falsified, held, retired-a-priori (operator-ratified 2026-08-10)}.

| date | H0 | setup | observation | verdict |
|---|---|---|---|---|
| 2026-08-10 | H0-observe (v1, clause 1) | — | dimension count: entropy level sets are (K−2)-dimensional; clause generic | retired-a-priori |
| 2026-08-10 | H0-observe (v1, clause 2, averaged) | — | H(X\|C) ≤ H(X); averaged narrowing is a theorem | retired-a-priori |
| 2026-08-10 | H0-project (v1, operator-invariance branch) | — | invariance ⇔ degenerate ⇔ frozen (owned by H0-control) | retired-a-priori |
| — | H0-control′ / H0-observe′ / H0-project′ / H0-substrate-{a,b,c} / H0-cache-B′ / H0-eom-logit | — | (none yet) | untested |
| — | H0-eom-embed | — | gated on H0-hold | untested |
