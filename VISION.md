# VISION — What a Diffusion LLM Might Let You See

This document is not a roadmap and not a set of promises. It is a record of the
questions this instrument was built to ask, and an honest account of which of
them are already answered, which are hypotheses we can now test, and which are
open. Roadmap items live in `decisions/` (ADRs); shipped behavior lives in
`README.md` and the code. This file is the *why it might matter*.

Every claim below is tagged:

- **[established]** — grounded in the model, the scheduler source, or published
  work; verifiable today.
- **[hypothesis]** — a specific, testable conjecture this tool is designed to
  probe. Not yet demonstrated here.
- **[open]** — a genuinely unsettled question, including ones the field has not
  yet framed.

The house rule, inherited from how this project is built: a term of art must
carry its constraints, not just its prestige. Where a word does real work below
(*converged*, *sampler*, *meaning*), it is used in its narrow sense, and the
narrowing is stated. Every literature claim tagged **[established]** carries a
handle in the [References](#references) — verify it, don't trust it.

---

## 1. The substrate that makes this possible

DiffusionGemma is, as of mid-2026, the first open-weight text diffusion model
built on **uniform-state** (not masked / absorbing-state) discrete diffusion.
**[established]** The distinction is load-bearing and is the reason this
instrument can exist:

- In **masked** diffusion (the dominant line — the LLaDA / Dream / MDLM family),
  corruption replaces a token with a single `[MASK]` sentinel. Noise is one
  symbol. Once a position is unmasked, it is fixed.
- In **uniform-state** diffusion, an un-accepted position is re-drawn uniformly
  from the *entire* vocabulary — for this model, all 262,144 tokens. There is no
  mask token; the scheduler's own source confirms it and the pack's
  `corroborate_no_mask_token` check re-derives it empirically. **[established]**

Two consequences follow directly, and both are the seed of everything else here:

1. **The noise is the maximum-entropy state of the token space itself.** A
   uniform draw over 262,144 tokens is 18 bits per position (≈12.48 nats; the
   scheduler's `entropy_bound` is denominated in nats), and because the
   Gemma vocabulary is multilingual, that noise renders as a polyglot cascade —
   katakana, Bengali, CJK, reserved `<unused>` tokens — not as a monotonous
   field of one sentinel. What you see in the early flipbook frames is not
   incidental garble; it is the literal, visible signature of uniform-state
   corruption. **[established]**

2. **Self-correction is native.** Because the scheduler is stateless per step
   and re-noises whatever it does not accept, a position can commit and then
   *re-melt* before it settles. This is observable in the trace: committed
   fraction can fall between steps before it climbs. **[established]** Masked
   diffusion cannot do this — an unmasked token is committed.

## 2. The reframe: meaning as an annealing process

The useful mental model is not "denoising" in the image sense but **simulated
annealing** in the statistical-mechanics sense. **[established, as an analogy
with a precise mapping]**

- The temperature schedule is a monotonic **cool**, `t_max → t_min`. Dividing
  logits by a shrinking temperature sharpens the distribution, so per-position
  entropy falls as the run proceeds.
- The **entropy bound** is the freezing criterion: the scheduler accepts the
  lowest-entropy prefix of positions whose cumulative entropy stays under the
  bound. This acceptance rule is not ad hoc — it is the *entropy-bounded
  unmasking* procedure of **EB-Sampler** (Ben-Hamu et al., FAIR/Meta,
  [arXiv:2505.24857](https://arxiv.org/abs/2505.24857), NeurIPS 2025): accept
  the lowest-entropy prefix within an error budget. The model's
  `EntropyBoundScheduler` *is* this paper's rule, and it is the exact mechanism
  everything below instruments. **[established]**
- The **renoise** is thermal agitation: un-frozen positions return to the
  18-bit melt each step.

So meaning does not *appear*; it **crystallizes** out of a maximum-entropy melt
as the schedule drops positions below the bound. The knob is not a fluctuation
you impose — it is the cooling rate.

This reframe is the spine of the vision, because crystallization carries a
constraint that matters: **the crystal you get depends on the cooling
protocol.** That is the hinge for everything in §3.

## 3. What the instrument might let you see

### 3.1 The representation, read off the freezing order

**[established finding, [hypothesis] for the extension]**

The order in which positions freeze is not arbitrary. Published trajectory
analysis of masked diffusion models ("What Gets Unmasked First?",
[arXiv:2605.31564](https://arxiv.org/abs/2605.31564)) finds an emergent,
unsupervised **content-first** order — entities first, then relations and
function words, structural tokens last. And Stability-Weighted Decoding
([arXiv:2604.17068](https://arxiv.org/abs/2604.17068)) makes the connection
rigorous: a token's temporal instability, quantified by the KL divergence
between its consecutive step distributions, is a *strict lower bound* on its
mutual information with the still-unresolved (masked) context. **[established]**
(Both results are stated over *masked* diffusion; their transfer to
uniform-state is the [hypothesis] below.)

Put together: **the freezing order is a readout of the model's learned
conditional-dependency structure.** The commit heatmap is not decoration; it is
a measurement of what the model treats as independent (freeze early) versus
dependent (freeze late).

The buildable experiment — **[hypothesis]** — is to ask whether that order is
*model-intrinsic* (invariant when you swap the sampler) or *process-imposed*
(it shifts). We already know a training choice can distort it (supervised
fine-tuning has been shown to prematurely anchor structural tokens, fixing
length before content — the "What Gets Unmasked First?" finding). So the
trajectory is sensitive to both the representation and the process, which is
exactly what makes it a probe rather than a constant. This instrument is the
bench on which that comparison runs.

### 3.2 Sampler signatures — dispositions, not aesthetics

**[open / emerging]**

In image diffusion, "a Heun image" and "a Karras schedule" became evocative
because an interface let thousands of people swap solvers and *feel* the
difference. The math predates the vocabulary; the interface created it. The same
zoo is now forming for text — remasking and decoding policies (confidence-order
[LLaDA, Nie et al. 2025], low-confidence remask [ReMDM, Wang et al. 2025],
learned quality scores [Jazbec et al., [arXiv:2512.09106](https://arxiv.org/abs/2512.09106)],
stability weighting, path selection over unmasking orders [Lee et al.,
[arXiv:2511.05563](https://arxiv.org/abs/2511.05563)]) and schedule families
(including hyperschedules that interpolate between autoregressive and diffusion
[HDLM, Fathi et al., [arXiv:2504.06416](https://arxiv.org/abs/2504.06416)]).
**[established that these exist]**

But the term of art must carry its constraint. An image sampler's signature is
*aesthetic* — every solver approximates the same score field, so you get the
same referent rendered differently. A text sampler crosses from *how* into
*what*: different commitment dynamics converge on different token sequences,
which are different propositions. The tell is already in the literature — the
discrete-diffusion sampler papers ask whether a sampler is *correct* (Tang et
al., "Is Your Diffusion Sampler Actually Correct?",
[arXiv:2602.19619](https://arxiv.org/abs/2602.19619)), a truth-predicate no one
applies to an image sampler. **[established]**

So the conjecture — **[hypothesis]** — is that a text sampler's signature is a
**disposition and its failure modes**, not a look: a corrector-heavy,
stability-weighted policy that re-checks dependencies before freezing (fewer
cascade errors, more compute) versus an aggressive parallel-commit policy that
paints itself into corners (fast, more premature-commitment incoherence — the
documented SFT length-anchoring failure is exactly this). What this instrument
could contribute is not the equations but the **bench where those dispositions
become a shared, felt vocabulary** — the thing that turned image samplers
evocative in the first place.

### 3.3 The renoise axis — the one masked models don't have

**[open]**

Uniform-state diffusion has a free axis that masked diffusion structurally
lacks: **what un-accepted positions are re-drawn from.** DiffusionGemma draws
uniformly. But the renoise distribution is a swappable component, and the most
philosophically loaded swap available is to bias it toward *semantically
plausible* alternatives rather than uniform noise. This is not a loose idea: it
is named directly in HDLM's own future-work (Fathi et al.,
[arXiv:2504.06416](https://arxiv.org/abs/2504.06416)) — replace uniform renoise
with distributions reflecting *plausible, on-policy errors* — and the
generalized noise process that makes such swaps first-class is von Rütte et
al.'s Generalized Interpolating Discrete Diffusion (GIDD,
[arXiv:2503.04482](https://arxiv.org/abs/2503.04482)). Under that swap, the
intermediate melt would no longer be structureless polyglot soup — it would be a
cloud of near-meanings, and the crystallization would proceed through a
landscape of plausible states rather than random ones. Almost nothing has been
done here in practice, because almost no open model was uniform-state until now.
This is the least-explored, most diffusion-native frontier the tool opens.

### 3.4 Polymorphism — which swaps preserve meaning, and where the boundaries are

**[hypothesis]**

If meaning crystallizes and the crystal depends on the cooling protocol, then
different samplers are different cooling protocols, and the sharp question is
which swaps are **polymorph-selecting** (a different crystal — a different
claim) versus **defect-level** (the same crystal, paraphrase-grade variation).
The outcome taxonomy is standard: a swap yields a meaning-preserving paraphrase,
a meaning-altering flip (a negation is a truth-conditional fork), or sub-semantic
garbage — and off-the-shelf natural-language-inference (entailment) models
classify those buckets automatically. That is a general NLI capability, not a
single citable result; treat it as available tooling, not a claim.

What is **[open]** is the *map*: a phase diagram of meaning over
(entropy-bound × cooling rate × renoise distribution) — regions where the claim
is stable, boundaries where it flips. No one has charted it, because until an
instrumented, swappable interface existed, no one could turn the knob and watch
the claim move. That map is the concrete deliverable the vision points at.

### 3.5 Self-correction, made watchable

**[hypothesis]**

A standing objection to diffusion reasoning (raised by Sean Goedecke in a
non-academic blog critique, our April thread — flagged as non-academic): why
would a model emit "wait, I was wrong" mid-generation — wouldn't denoising just
edit it out? Uniform-state is the answer masked diffusion cannot give:
**re-noising is the edit.** A committed token that drops below threshold
re-melts and can be replaced. The re-melt is already visible in the trace
numbers; whether it corresponds to *semantically meaningful* revision — the
model genuinely reconsidering — rather than mere stochastic churn is exactly the
kind of thing this instrument exists to let someone watch, frame by frame,
instead of argue about. The academic line making this a design principle is
explicit: self-correcting masked diffusion (Schiff et al.,
[arXiv:2602.11590](https://arxiv.org/abs/2602.11590)), self-reflective remasking
(Huang et al., [arXiv:2509.23653](https://arxiv.org/abs/2509.23653)), and the
native self-correction of GIDD and HDLM above. **[established that the mechanism
is studied; [hypothesis] that the trace shows meaningful revision.]**

## 4. What this is *not* claiming

This section is load-bearing, and it is deliberately deflationary.

- **No ultimate answers.** Nothing here promises that inspecting a denoising
  trajectory reveals what meaning "is." The floor of the vision is the
  representation probe (§3.1), which is real and modest; the ceiling is open and
  should stay labeled open.
- **The interesting cases are a minority.** For a tuned model on a well-posed
  prompt, most sampler swaps will land in the paraphrase bucket. Truth-flips
  cluster at the edges — underspecified prompts, aggressive settings, questions
  on a decision boundary. The right question is not "is meaning process-relative"
  (too broad to be true) but "*where are the boundaries*" (§3.4). Overstating
  this into "nothing means anything" is the failure mode to avoid.
- **The field is building these as correctness tools, not interpretation
  tools.** The sampler-centric evaluation and remasking-policy literature is real
  and moving fast, but it is framed almost entirely as quality control toward a
  presumed-correct answer. The *interpretive* question — what it means that the
  process co-determines the meaning — is the open lane, not a solved one. This
  tool is not first to the science; it may be first to the bench where a
  non-researcher can watch it.
- **First-mover is a fragile moat.** As of this writing, searches turn up no
  comparable instrumented text-diffusion sampler for ComfyUI — but a negative
  existence claim cannot be proven, and the ComfyUI core team fast-follows Gemma.
  The loader will be commoditized. The **instrumentation** is the durable asset.
- **`converged` is not `correct`.** The pack's validity fields say the canvas
  stopped changing, not that the answer is right or that reasoning was sound. The
  document that promises otherwise is the document to distrust.

## 5. The through-line: an image-native tool for a text-native problem

One observation ties the rest together. ComfyUI is image-*output*-native, and
that assumption shows up as the same gap from three angles: the sampler zoo is
stranded on `SIGMAS`/`LATENT` (an image-process substrate); text-out rendering
is thin (no strong string-to-image node, because text is an input there, not an
output); and numeric display is absent (the ecosystem's own tensor inspectors —
ComfyUI-ViewData, `TensorInfo` in basic_data_handling — show *shape*, not
*values*, and Mixlab's `GridOutput` arranges *images*; none inspect data,
because it is not a data-inspection environment). **[established — these are node
repos, not papers]** Each gap is small, and each is a small moat.

The bet this project makes is that the durable contribution is not the model
wrapper — which core will absorb — but the **interface where a text-diffusion
sampling vocabulary can form**, the same mechanism by which image samplers went
from opaque names in 2022 to evocative styles today: not new equations, but a
place to swap them and feel the difference. The model is what you point it at,
and it will change. The instrument is the thing worth building well.

---

## References

Grouped by the section that leans on them. arXiv IDs verified 2026-07-07.
Descriptive-only cites (no pinned ID) are marked. Non-academic references are
flagged inline where they appear.

**§2 — the entropy-bound rule (keystone; this is the mechanism the tool instruments):**
- Ben-Hamu, Gat, Severo, Nolte, Karrer (FAIR/Meta), *Accelerated Sampling from
  Masked Diffusion Models via Entropy Bounded Unmasking* (EB-Sampler),
  [arXiv:2505.24857](https://arxiv.org/abs/2505.24857), NeurIPS 2025.

**§3.1 — freezing order as representation readout:**
- *What Gets Unmasked First? Trajectory Analysis of Diffusion Models for
  Graph-to-Text Generation*, [arXiv:2605.31564](https://arxiv.org/abs/2605.31564)
  (2026) — content-first emergent order; SFT premature-structural-anchoring.
- Wu & Huang, *Stability-Weighted Decoding for Diffusion Language Models*,
  [arXiv:2604.17068](https://arxiv.org/abs/2604.17068) (2026) — temporal
  instability as a strict lower bound on mutual information with masked context
  (verified against the paper's stated result).

**§3.2 — sampler signatures / the zoo / "correct":**
- Tang, Yu, Zhang, Ver Steeg, *Is Your Diffusion Sampler Actually Correct? A
  Sampler-Centric Evaluation of Discrete Diffusion Language Models*,
  [arXiv:2602.19619](https://arxiv.org/abs/2602.19619), ICML 2026.
- Lee, Kim, Park, Park, *Lookahead Unmasking Elicits Accurate Decoding in
  Diffusion Language Models*, [arXiv:2511.05563](https://arxiv.org/abs/2511.05563)
  (2025) — path selection over unmasking orders.
- Jazbec et al., *Learning Unmasking Policies for Diffusion Language Models*,
  [arXiv:2512.09106](https://arxiv.org/abs/2512.09106) (2026).
- Confidence-order baseline: Nie et al. 2025 (LLaDA), descriptive. Low-confidence
  remasking: Wang et al. 2025 (ReMDM), descriptive.

**§3.3 — the renoise axis / semantically meaningful noise:**
- Fathi et al. (ServiceNow), *Unifying Autoregressive and Diffusion-Based
  Sequence Generation* (HDLM), [arXiv:2504.06416](https://arxiv.org/abs/2504.06416)
  (2025) — hyperschedules; the plausible-error-renoise conjecture is HDLM's own
  future-work.
- von Rütte, Fluri, Ding, Orvieto, Schölkopf, Hofmann, *Generalized Interpolating
  Discrete Diffusion* (GIDD), [arXiv:2503.04482](https://arxiv.org/abs/2503.04482)
  (2025) — generalized/hybrid noise process with native self-correction.

**§3.4 — polymorphism / meaning-preservation buckets:**
- The paraphrase / entailment / contradiction taxonomy is standard; classification
  is a general natural-language-inference capability, not a single citable result.
  (No specific paper pinned — treated as available tooling, not a claim.)

**§3.5 — self-correction:**
- Schiff et al., *Learn from Your Mistakes: Self-Correcting Masked Diffusion
  Models*, [arXiv:2602.11590](https://arxiv.org/abs/2602.11590) (2026).
- Huang, Wang, Chen, Qi, *Don't Settle Too Early: Self-Reflective Remasking for
  Diffusion Language Models*, [arXiv:2509.23653](https://arxiv.org/abs/2509.23653)
  (2025).
- The self-editing *objection* is Sean Goedecke's, from a non-academic blog
  critique — a reference, not a peer-reviewed source.

**§5 — the image-native gap (node repos, not papers):**
- ComfyUI-ViewData and `TensorInfo` (basic_data_handling) report tensor *shape*,
  not values; Mixlab's `GridOutput` arranges *images*.

---

*This document is speculative by design. Claims tagged **[hypothesis]** and
**[open]** are invitations, not results. If a future version demonstrates one of
them, it should be promoted to **[established]** here with a pointer to the
evidence — and if one is falsified, it should be struck through, not quietly
deleted.*
