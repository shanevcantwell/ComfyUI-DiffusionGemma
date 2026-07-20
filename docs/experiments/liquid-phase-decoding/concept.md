# The Forgotten Phase — liquid-state decoding in a sublimating sampler

**Status:** incubating (repo research record — `docs/experiments/`) · **minted:** 2026-07-12 (co-framed with operator this session)
**Concept handle:** *liquid-state decoding* / *the liquid register*
**Home repo:** `ComfyUI-DiffusionGemma` (migrated here from the `design-docs` pools 2026-07-12 — this is repo-specific design material, not cross-repo intake)
**Lab-notebook sibling:** [`experiment.md`](experiment.md) (the falsifiable H0s + observation table)

---

## The reframe (the spine)

**DiffusionGemma sublimates.** Its commit rule is a sharp threshold: a position is either
above `entropy_bound` (call it **steam** — high-entropy, renoised, structure gone) or below
it (**frozen** — committed, crystallized text). A position crosses **directly** between the
two. There is no **liquid** basin for it to rest in — no stable intermediate where a token is
*mobile but still coherent*.

The three phases, mapped to the mechanics:

| phase | mechanics | role |
|---|---|---|
| **frozen** | entropy below bound → committed | the crystallized token |
| **liquid** | held near the bound — mobile **and** coherent | **the forgotten intermediate — the whole idea** |
| **steam** | fully renoised, high-entropy, no structure | the noise the sampler anneals away |

The claim is not "the liquid phase is under-used." It is that **the liquid phase does not
currently exist in the dynamics** — the sampler sublimes solid↔gas and skips it. The
exploration is to *open a basin* for the liquid state where the current physics has only a
phase boundary.

## Why this is a step past a banked null, not a fresh guess

`ComfyUI-DiffusionGemma/loose-ends.md` (2026-07-06, entropy-bound decoder mechanics) already
tried the naive version: `confidence=0.0` keeps the system "open" longer, but *"keeping it hot
longer grants more escape attempts, it does not steer toward right."* Issue #10 corroborates —
the confidence dead-zone (0.005–0.1) is byte-identical, with a cliff at 0.001.

In the phase framing, that null is **sublimation to steam**: push heat and the position doesn't
melt to liquid, it boils straight to gas — undirected escape, no basin to land in. So the
prior attempt failed *because it overshot liquid*, and it did so because heat alone has no term
that holds a position in the intermediate. The new hypothesis is precise: **balance heat
re-injection against the annealing/commit threshold** — the threshold is the term that could
open and hold a liquid basin, converting escape-attempts into steering.

## The two faces (both require a liquid phase to exist)

A sublimating sampler offers neither of these — you can only steer freezing-order, or read a
resting distribution, if there is a liquid state to hold in.

- **Control face — deferred commitment / freeze-last.** Hold a chosen position liquid while its
  neighborhood crystallizes, so it freezes **last**, collapsing at **maximal conditioning**. The
  perfect word chosen after the paragraph is finished around it. Promotes `entropy_bound` from a
  scalar *stopping rule* into a steerable, per-position *field*; makes commit-**order** a
  controllable axis rather than the emergent percolation front that #7 only observes. This is
  the mechanism behind #3's "mot juste" *goal*.

- **Observation face — capture the near-superposition.** Softmax/sampling is a *projective
  measurement*: one sample and the superposition of candidate meanings is gone ("softmax
  violence"). A position held liquid can be **read** across timesteps — its per-position
  distribution (top-k candidates + weights) reconstructs the cloud of near-fitting meanings a
  single softmax destroys. This is a **distinct output type**: not a string, but a position
  annotated with its live distribution. #14's per-position entropy is the **scalar shadow** of
  this — it records *how much* ambiguity while discarding *which* meanings.

## Feasibility — corrected to ground truth (2026-07-12)

- **Per-step reach into the sampler is PROVEN**, not speculative: the P0–P3 live per-step view
  and ADR-CDG-004's `callback_on_step_end` already read logits and can re-assert a pin every
  step. Both faces are implementable **inside a single node execution**, in-callback.
- **The gap is a ComfyUI node-graph incrementer.** There is no clean per-graph-execution stepping
  to advance one diffusion step per run and thread state back through the UI. So the experiment
  runs *in-callback*, not as a stepped node graph; cross-execution resume, if needed, rides
  ADR-CDG-005 `CANVAS_STATE` / ADR-CDG-006 step-window resume — not a UI incrementer.
- **The artifact home** for the observation face is a native socket / `CANVAS_TRACE` field
  carrying a per-position distribution (ADR-CDG-001, native socket types — reject "lying
  scalars" the way it rejects lying sigmas).

## Where the pieces already live (synthesis was missing, not the parts)

Cross-refs into `ComfyUI-DiffusionGemma`:

- **#3** — gap-fill / "mot juste": the *goal*, not this mechanism.
- **#7** — commit-front morphology (dendritic vs percolation): the *emergent* freeze dynamics
  this proposes to *steer*.
- **#10** — confidence discontinuity / dead-zone / cliff: **evidence the bound is a phase
  boundary** — the surface a token would be held on. The strongest existing anchor.
- **#11** — raw pre-excision canvas token ids: committed ids, not the live distribution.
- **#14** — per-position entropy heatmap: the **scalar shadow** of the observation face.
- **#23** — `ENTROPY_SCHEDULE` seam / per-step input scheduling: the natural control home.
- **ADR-CDG-001** — native socket types: home for a distribution-valued output.
- **ADR-CDG-004** — logits reachable via `callback_on_step_end`: the proven capture point.
- **ADR-CDG-005 / 006** — resumable `CANVAS_STATE` / step-window resume: stand-in for the
  missing node incrementer.
- **loose-ends.md 2026-07-06** — the banked null this steps past (heat alone → steam).
- **plan.md Phase 5** — `BlockRefinementScheduler.editing_threshold` (re-melt *committed*
  tokens): the *opposite* direction (solid→liquid reopen) — worth relating, not the same move.

## Prior art & where the novelty actually is (2026-07-12 literature scan, ~50 papers, 5 search fans)

The field is exactly as active as suspected. The scan splits our idea into halves of very
different maturity — and locates the unclaimed composite precisely.

- **Control face (defer / commit-order) is a CROWDED 2026 cluster** — and not our novelty.
  `EB-Sampler` (arXiv:2505.24857, entropy-bounded multi-token unmasking) is the *likely direct
  ancestor of DiffusionGemma's `EntropyBoundScheduler`*; `ReMDM` (2503.00307) is the remask
  progenitor; `LESS` (2606.16908) commits only when top-1 stops drifting (JS-divergence
  stability — literal "metastable, then commit"); `TraceLock` (2605.24697) already argues
  commitment strategy is a *separate learnable axis*; `Deferred Commitment Decoding`
  (2601.02076). **But every one frames deferral as an accuracy/speed lever — never as a
  creative/stylistic knob.**
- **Observe face (read the held distribution) is PRECEDENTED — as a fidelity tool, not a style
  tool.** `Soft-Masked DLM` (2510.17206, ICLR 2026) blends the mask embedding with top-k
  predicted-token embeddings — *literally holds a liquid intermediate* — and is the mechanism to
  build on. Joint-sampling-correction papers (2605.13681, 2509.22738) name the exact failure of
  collapsing per-position marginals too early. **Their goal is faithfulness to the true joint,
  not exploiting the distribution's *shape* for control.**
- **Steering axes: register/formality/sentiment done; tense/mood apparently unclaimed.** `RegDiff`
  (2510.06386) steers sentiment/toxicity/formality/register (Shakespearean↔modern);
  `PoetryDiffusion` (2306.08456) does metrical/poetic control via step-wise loss injection. No
  paper found targeting grammatical **tense** or **mood** via diffusion steering — and none reads
  the *held pre-collapse distribution itself* as the lever (they inject guidance/regularization).
- **The corroboration that upgrades a hunch to a measurable prediction:** `Steering Without
  Breaking` (2605.10971) finds **different attributes commit at different points in the denoising
  schedule** — topic settles <2% in, sentiment ~20% — and builds an attribute-timed intervention
  scheduler. That is direct evidence for the earlier claim that *the axes you can project out are
  the ones committed context leaves open*: each axis has its own freeze-time, so its liquid window
  sits at a different schedule position. It makes "syntax pins tense harder than register" an
  **experiment, not a metaphor** — measure where each axis commits.

**Novelty verdict (defensible):** the composite — *deliberately hold a position liquid, then
sample its distribution to pull out a different register / tense / mood as a controllable
**creative** axis* — was not found across the surveyed corpus. The mechanics exist; using them as
a **style/semantic selector rather than a correctness lever** is the unclaimed move. Build-from
pair: `Soft-Masked DLM` (2510.17206, hold the blended state open) × `Steering Without Breaking`
(2605.10971, attributes freeze on different schedules).

**Grounding caveat (banked honestly):** the numeric DiffusionGemma defaults cited around this repo
(`entropy_bound=0.1`, `confidence=0.005`, 48 steps, temp 0.8→0.4) trace to a *single* HF model-card
fetch and may be circular with this repo's own `CLAUDE.md`; the official explainer confirms the
*mechanism* but publishes no numbers. Treat as plausible-not-verified until a second source
confirms. (Third-party sampler instrumentation exists — "Neither Parallel Nor Sequential",
2606.14620 — finding confidence predicts correctness only in *structured* domains: AUROC 0.75 on
GSM8K vs ~0.47 on factual recall. Relevant to whether `entropy_bound` is even a trustworthy handle
outside math/code.)

## Third face — distribution-space style projection (downstream; presupposes both H0s)

Once a liquid state exists *and* is readable, the sampling **operator** over it becomes a control
surface. The liquid distribution at a position is **multi-modal**, not a single fuzzy blob: it
holds separated candidate clusters — `walked / strode / ambled` (register), `runs / ran / running`
(tense), indicative↔subjunctive (mood). A *directed* sampling operator (reweight toward
low-frequency/high-surprisal mass; mask to subjunctive-inflected candidates) **projects out** which
mode collapses. That reframes "mot juste" once more: not the single best word, but a **manifold of
near-fits whose projection *is* the stylistic commitment** — a style knob at the distribution level,
not a post-hoc rerank. Testable constraint (per `Steering Without Breaking`): the axes you can
project are those committed context leaves open, and each has its own freeze-time — so register
freedom should survive in more positions than tense freedom. This is a **third face**: it depends
on H0-control (liquid exists) and H0-observe (distribution readable), so it is downstream, and it
is the location of the scan's "apparently novel" verdict. Tracked as H0-project in the experiment.

## Architecture (verified 2026-07-12) — and the variety ceiling it implies

Verified against primary sources (Google "explained" page, vLLM integration blog):

- **One backbone, two attention modes on shared weights** — not separate AR and diffusion
  components. Verbatim: *"Rather than using separate models, a single backbone dynamically
  toggles between two modes."* Causal self-attention for **prefill** (writes the KV cache),
  bidirectional self-attention over the **full 256-token canvas** for **denoise** (reads that
  cache). No cross-attention module, no block-local sub-windowing. The "shared weights" claim is
  now literal at the parameter level: **the encoder and decoder weights are *tied* per layer** —
  q/k/v/o projections, embeddings, and norms (`modeling_diffusion_gemma.py:1481-1491`). "One
  backbone, two modes" is not a framing convenience; the same tensors serve both attention modes.
- **The cache is written once per block, read-only during denoise.** This *verifies the
  feasibility win*: holding a position liquid = re-running denoise steps over a fixed prefill
  cache; the prompt is never recomputed. Cheap.
- **Cross-block mechanics — DG is AR at block scale, diffusion within the block.** The encoder
  **runs twice per block** (vLLM integration blog): a committed block is **causally re-encoded**
  into the KV cache, not appended from its denoise states. Denoise never feeds the cache
  directly — a block crystallizes, then that crystallized text is re-encoded through the causal
  path to become conditioning for the next block. The consequence for this program: **the liquid
  window is architecturally bounded at one canvas.** Within a 256-token block, denoise is fully
  bidirectional and the liquid can span it; across blocks, the channel is the causally-encoded
  cache and nothing else. Block N+1 conditions on block N's *committed* text, never on block N's
  liquid.
- **Training provenance is UNCONFIRMED.** Google claims architectural lineage ("based on",
  "builds upon Gemma-4") but never a training-initialization ("initialized from" / "fine-tuned
  from" an AR checkpoint). Whether the weights are AR-adapted or trained-from-scratch on this
  architecture is not documented — treat as open, not as "adapted from AR Gemma-4." Provenance
  footnote (suggestive, still UNCONFIRMED): Google's own explainer phrases the toggle as the AR
  path *"isn't used natively but fine-tuned to support the different tasks"* — a data point
  consistent with AR-adaptation, not a confirmation of it. Do not lean on it as sourcing.

## Limitation — the causal-prefill variety ceiling (H0-substrate)

Operator observation (2026-07-12): DiffusionGemma's autoregressive-flavored structure **confounds
the variety a pure bidirectional diffusion across full weights would allow** — late positions are
pre-pinned by early commits, so the liquid they hold is narrower than a symmetric joint would
support. The legible scaffold-early/content-late stratification *is* the confound: the same
ordering that makes DG readable is what spends the joint's degrees of freedom early.

Refined by the architecture verify, the confound splits:
- **Real at the mechanism level:** the prompt cache is built causally and fixed; a partial L→R
  commit bias is *measured* (τ≈0.43–0.60, arXiv:2606.14620). Both cap co-liquidity.
- **Attribution unconfirmed:** the L→R bias may be *emergent* from the causal-prefill design +
  data, not *inherited* from AR pretraining (provenance is undocumented, above).
- **Consequence:** the substrate to remove for more variety is **the causal prompt prefill
  itself** (a fully-bidirectional-including-prompt model), not merely "trained from scratch."
  Within the 256-canvas DG is already fully bidirectional, so the cap is the causal cache + the
  commit-order bias, not block-locality.

**H0-substrate** (greenfield anticipated-failure-mode, repo ADR idiom — name the failure before it
bites): a diffusion LM without a causal prompt-prefill confound (candidate: LLaDA, from-scratch
masked diffusion) exhibits **richer liquid than DG**. If H0-project ever fails on DG, this is the
differential diagnosis: *wrong substrate, not wrong idea.* Honest tradeoff: pure-diffusion variety
trades against coherence (SEDD/MDLM underperform AR on fluency), so the program must decide whether
it wants *maximum* variety or *coherent* variety.

**Open — operational definition of "variety"** (operator to settle; provisional candidates):
(a) **co-liquidity** — how many positions are simultaneously liquid; (b) **per-position breadth** —
how multi-modal each held distribution is; (c) **cross-seed diversity** — how different full
sequences are at fixed prompt. H0-substrate measures whichever of these is chosen; they are not
equivalent.

## Empirical grounding (DG-runs, n=5, 2026-07-08 logs)

Five distinct runs (six files; two `initial_tests` are byte-identical — duplicate save). Full-canvas
realized-token snapshots per timestep; **no per-position distribution/confidence logged.**

- **Morphology confirmed, all 5 runs:** discourse/formula skeleton locks early, broadly L→R;
  load-bearing content (numeric answers, key adjectives, closing clauses) commits in a **late
  burst in the final 15–30%** of the step budget. The operator's "AR layers under DG" read is
  legible in the trace.
- **The instrumentation gap is a GATE, now empirically:** the logs capture *committed state only*.
  The liquid is invisible except where it *leaks* as churn (~6–8 positions across 5 runs — a
  position flipping among *plausible* fillers before settling). H0-observe and H0-project cannot
  run on this logging; **#14 (per-position entropy) + #11 (candidate ids) + full-distribution
  capture are the prerequisite gate**, not optional telemetry.
- **Re-melt datum:** an already-correct `\frac` (row 195) regressed to `explotfrac` (row 204) then
  re-corrected (row 215) — a *non-monotonic* commit. Liquid (and re-melting) is in the model's
  repertoire; connects `plan.md` Phase 5 `BlockRefinementScheduler`.
- **First thread of H0-control evidence, in-data:** tighter `entropy_bound` (0.03 vs 0.05)
  surfaced *visibly more churn* — one A/B pair on one prompt (caveated, could be prompt artifact),
  but it points the predicted direction: **the threshold is a liquid-window lever.**

### Second empirical pass — the first operator knob-sweep on the instrument (issue #40, 2026-07-13)

The first sweep run *on the built instrument* (`EntropyBoundScheduler`, `t=[0.4,0.8]`, 48 steps, prompt
"Write about the ocean using only D words," only `entropy_bound` varied across 10 points, 0.01→0.10) adds
five findings that touch this record directly. Full data and heatmaps in issue #40; pointers here:

- **Liquid sighted at `committed_fraction=1.0` — the dimension↔diversity distinction made concrete.** A
  fully-committed canvas is *not* a spent one: the sweep shows structure still varying at cf=1.0 across
  runs. This sharpens the "variety" open question above — **co-liquidity (how many positions are
  simultaneously mobile) is a different axis from cross-run diversity**, and cf=1.0 can hide the latter.
  A canvas can read "done" on the scalar while its trajectory still carried the freedom the program wants
  to read.
- **`committed_fraction` is a lying convergence signal — quantified at n=10.** Every run shows a large
  **step-2 re-melt** (committed_fraction collapses, e.g. 0.5195 → 0.2539, before the monotone climb): the
  model routinely discards ~half its step-1 commitments as the *standard opening move*, not an occasional
  curiosity. This is direct corroboration that committed-state-only logging (the DG-runs gate above) hides
  the real dynamics — the scalar reports monotone-ish progress over a canvas that is visibly breathing. The
  step-2 re-melt is a **free, robust observable** every hold-and-release protocol (roadmap R2) will interact
  with.
- **Constraint-blind freezing / the fossil mechanism.** All 10 outputs open with a verbatim knob-invariant
  preamble; the lipogram's most consistent violation ("waves") lives *inside* that fossil — a constraint
  failure that froze early and was **never renegotiated at any knob value**. Constraint adherence visibly
  begins only ~step 5-7, after the polyglot melt clears, at *every* bound value. This is prior evidence for
  H0-control's difficulty: **early-frozen structure is very stable** — pinning against it (or melting it) is
  the real test, not a corollary.
- **Nested-prefix step-1 structure.** If the sweep was seed-fixed (see the gap below), the step-1 acceptances
  are **nested prefixes of one entropy-sorted order**, making the whole sweep a naturally controlled
  commit-order-causality experiment: all downstream divergence is traceable to *which extra positions froze*.
  This is a free experimental design the instrument produced incidentally — but it hangs on the seed question.
- **Telemetry gap: seed not logged.** The knob-invariant preamble *suggests* a fixed seed across the sweep,
  but **seed is not recorded in the log properties** — so the nested-prefix reading above is unconfirmed.
  This is a concrete, cheap instrumentation fix (log the seed) that would upgrade the sweep from suggestive
  to a controlled experiment. Banked here as the empirical-grounding twin of the DG-runs "no per-position
  distribution logged" gate: **the instrument's own telemetry is still lossy in ways that block the reads we
  want.**

## Polyphonic prefill (H0-cache) — turn the confound into a control surface

Operator move (2026-07-12): rather than *remove* the causal prefill to escape the variety ceiling
(H0-substrate, needs another model), **enrich it** — assemble the prefix KV cache from *multiple*
prefills and diffuse off the richer field. The verified architecture makes it clean: the cache is
written once and **read-only during denoise**, so a torch-level node can build any cache out-of-band
and hand it to the loop. This is the **upstream twin of H0-project**: shape what the liquid condenses
*from*, rather than sampling the liquid at the output.

Two forms, different risk:
- **(A) Concatenate** prefills' K/V along the sequence dim; bidirectional denoise attends to all.
- **(B) Blend/interpolate** K/V directly (`α·K_A+(1−α)·K_B`) as a continuous register knob.

*Feedstock strategy (orthogonal to A/B — what you prefill, not how you combine; operator 2026-07-12):*
**high-temperature run-on prefill.** Generate a long, high-entropy, associative AR run-on and prefill the
cache with *that*, then let the cool canvas condense off the enriched field. The synthesis it unlocks:
**same heat, opposite role by pipeline position** — heat *in the canvas* boils to steam (the 2026-07-06
null: escape attempts, no steering), but heat *in the prefill* is rich feedstock the canvas distills from.
*Don't heat the crystal; heat the feedstock.* It's the two modes used for what they're built for —
AR **divergent** (wide association), diffusion **convergent** (anneal a coherent answer out). Bonus: a
run-on is **one continuous causal stream**, so its KV is self-consistent — it sidesteps the concat
mutual-blindness caveat and needs no bridging recompute; likely *more* on-manifold than concatenated clean
framings. *Risk:* off-topic drift may distract rather than enrich (empirical; interacts with the `thinking`
toggle and #9's "thinking can consume the whole canvas"). Covered by H0-cache's novelty umbrella; a
dedicated "self-generated/high-temp context as conditioning" prior-art check is deferred, not run.

*Feedstock leverage is **total**, and the pipeline mechanics make it cheap (delta 2026-07-13):* the
cross-block re-encode (see Architecture, "DG is AR at block scale") means **the KV cache is the *sole*
cross-block conditioning channel** — nothing but the causally-encoded cache carries information into the
next canvas. So shaping the feedstock is not one lever among several; it is *the* lever on what a fresh
canvas condenses from. And the mechanics keep it in reach: **prefill never samples.** The prefill pass
only *encodes* — heat lives in feedstock **generation**, not in the encode. That decouples authorship from
encoding: **any AR model can author the run-on text** (a hotter, wilder, or differently-tuned model than
DG), and DG then encodes that text with *its own* weights into a cache the canvas can read. Two feedstock
knobs fall out of this:
- **EOS-suppression as a feedstock lengthener.** Suppressing the end-of-sequence token during feedstock
  *generation* makes the AR author run on further, producing a longer, richer associative field to
  prefill from — a direct dial on how much feedstock the canvas has to distill.
- **In-canvas EOS masking** — a *separate* knob, applied inside the denoise loop via the CDG-010 logit
  hook (not during feedstock generation): mask EOS in the canvas distribution to keep positions from
  committing to end-of-turn early. Interacts with #9's "thinking can consume the whole canvas" budget
  question — the two EOS knobs pull in opposite pipeline stages and should not be conflated.

*Feedstock geometry — how much, through what, weighted how (delta 2026-07-13, grounded from the cached
checkpoint's `config.json`):*
- **No num_KV:num_token ratio caps this — the ceiling is a quarter-million tokens.** `max_position_embeddings
  = 262,144`; the canvas is 256, so the conditioning cache can be ~**261,888 tokens** — a >1000:1
  conditioning:generation ratio. The upper bound on "what a canvas can diffuse from" is genuinely enormous;
  feedstock is not scarce by architecture.
- **Long range is a 5-layer channel, not the whole stack.** `layer_types` splits **25 sliding_attention**
  (window ~1K, θ=10K) against **5 full_attention** (proportional RoPE, `partial_rotary_factor` 0.25, θ=1M;
  `global_head_dim=512` vs 256 local). Distant feedstock reaches the canvas **only through the 5 global
  layers**; the 25 sliding layers give canvas positions a recency-weighted view of the feedstock **tail**.
  Two design consequences: **feedstock ordering matters — put what should dominate at the end** (the sliding
  window sees the tail hardest), and **AR-style recency survives into bidirectional mode via the window
  pattern**, not via causality. The tail-dominance the sliding layers impose is the geometry knob under the
  feedstock-ordering lever above.
- **VRAM estimate (verify hybrid-cache behavior before leaning on it).** If the sliding layers use a bounded
  hybrid cache, long-cache cost is dominated by the 5 global layers: ≈ **80 KB/token** (8 KV heads × 512
  head_dim × K+V × bf16 × 5 global layers) ⇒ ~**8 GB per 100K feedstock tokens**. A 50–100K-token hot
  run-on feedstock is plausibly runnable on the RTX-8000 with quantized weights. This assumes the sliding
  layers don't hold a full-length cache — **confirm the hybrid-cache behavior before the estimate is load-bearing.**
- **Open empirical question (the experiment, not a blocker):** whether denoise-mode conditioning *uses* very
  long caches well is unknown — the fine-tuning distribution of cache lengths is undocumented. The ceiling is
  architectural headroom, not a demonstrated capability.
- *Incidental (feedstock-adjacent):* `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` is already present in the
  HF hub cache — #4's lead quantized-checkpoint candidate is downloaded, so a quantized run of the above is
  not gated on a fetch.

**Novelty (2026-07-12 scan, cross-verified):** the diffusion-LM-native, training-free, multi-prefill
cache assembly *for steering* is **apparently novel** — adjacent work exists on every side, nothing at
the intersection. Efficiency-concat: `CacheBlend` (arXiv:2405.16444). Composable-KV framing: `Models
Take Notes at Prefill` (2606.17107). Single-vector cache steering: `KV Cache Steering` (2507.08799).
Image-domain literal K/V swap: `MasaCtrl` (2304.08465). Form (B) continuous K/V interpolation as a
style knob is **doubly novel** — nearest cousins interpolate *weights* (2404.07117) or the residual
stream, never the cache. **Build-from / distinguish-against pair:** `MasaCtrl` (mechanism) ×
`Models Take Notes at Prefill` (composable framing); ours differs by being diffusion-LM-native +
steering + interpolation.

**Honest constraints the literature already measured (design around these, don't rediscover them):**
- **Concat is not free.** Independently-prefilled chunks are *mutually blind* — they never attended to
  each other — so concatenation loses cross-segment attention. `Cache Merging` (2607.01308): works at
  k=2, degrades past k=2 without recompute. Fix = `CacheBlend`-style **selective recompute of a token
  fraction** to restore cross-attention; DG's read-only cache makes it affordable. "Attention just
  blends them" is wrong — a bridging recompute is required, and many-framing assembly has a ceiling.
- **Blend (B) is off-manifold-risky and unvalidated.** No paper tests whether raw K/V averaging stays
  on-distribution for attention (mean-of-keys ≠ key-of-mean). Treat as an **open empirical question our
  ablation answers first**, not a solved one.

## The deliverable is the bench — a seam inventory

Framing (operator, 2026-07-12): **ComfyUI is a lab bench. The deliverable is not any one of these
techniques — it is the primitives and their ins/outs, factored so honestly that the variety composes
from the bench, not from us.** This is ADR-CDG-001 (native socket types, reject lying scalars) applied
at scale: a badly-factored socket *forecloses* the variety, and the DG-runs finding proves it — the
current committed-state-only logging is a lossy I/O that hides the liquid, and every downstream idea is
impossible until that socket tells the truth.

The whole session collapses to ~6 seams:

| primitive (socket) | node(s) | idea it unlocks | status |
|---|---|---|---|
| **`DISTRIBUTION`** — per-position top-k candidates + weights, per step | distribution tap (reads logits in the ADR-CDG-004 callback) | H0-observe; anti-#14-scalar; **the gate everything waits on** | built — Tier 0/1/2 capture + `capture=` ingress + MCP door + `DGemmaTokenTrace` display, across [PR #66](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/66)/[#99](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/99)/[#106](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/106)/[#107](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/107)/[#104](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/104); #11's presentation scope delivered by #107 (issue itself still open on the tracker) |
| **`SCHEDULE` — *any* sampler field** (`entropy_bound`, `min/max_temp`, `confidence`, …) as a per-step/per-position input *field*, not static config | schedule builder (uniform / ramp / per-position pin), one per field | H0-control + the general control surface — **every knob is a wire** | per-**step** general form landed — the control-signal walker (`WalkerParticipant`, [PR #100](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/100)) + `MUTABLE_TARGETS`/`Binding` payload drive any bound scalar per step; per-**position** field (a wire per canvas position, not per step) remains unbuilt |
| **pin/mask control** — freeze-last / commit-order assertions | pin-map builder (in-callback) | control face, commit-order steering | live end-to-end — [PR #71](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/71) (`PinParticipant` + `constraints_hook.py`'s logit mask, wired through `run_diffusion`'s `constraints=`) |
| **sampling operator** over `DISTRIBUTION` → token | temperature / top-p / directed-reweight / mood-mask nodes | H0-project — plug-and-play style projection | needs `DISTRIBUTION` first |
| **`KV_CACHE`** — the prefix cache as a first-class handle | cache assembly: concat (+bridging recompute) / blend | H0-cache (polyphonic prefill) | types + ingress + nodes landed ([PR #68](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/68)/[#98](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/98)/[#102](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/102) — `DGemmaEncode`/`DGemmaDenoise`); decoder-drive body + serialization (Phase 4) pending — apparently novel; ablation-gated |
| **`CANVAS_STATE`** — resumable savestate; step-window advance | (exists) ADR-CDG-005/006 | substitutes for the missing node incrementer | designed |

**Parameter-as-wire (the general principle, operator 2026-07-12):** no sampler scalar is static config
— *every field is a potential per-step/per-position input hook.* `SCHEDULE` above is the general form;
`#23`'s `ENTROPY_SCHEDULE` is just its first instance. Exposing the sampler's whole parameter surface as
wireable fields is the deepest cut of the bench principle: scheduling itself becomes a user composition
(ramp temperature while stepping the bound while pinning positions — a control graph never specced here).

**Enforceable discipline:** every socket is an `EMIT-CANONICAL / PARSE-AT-THE-DOOR` surface. `DISTRIBUTION`
carries the real distribution or it is the #14 lying-scalar trap reborn; a sampling-operator node
validates its input at ingress. Build these six with honest I/O and the technique space — liquid
sampling, style projection, polyphonic prefill, and compositions no one specced — becomes a graph a user
wires up. That is the bench doing its job.

## Name provenance

Water-phase vocabulary (frozen / liquid / steam) and the sublimation frame ("DiffusionGemma is
acting through state sublimation; the exploration is toward the forgotten intermediate state")
are the operator's, this session. Metallurgical framings (zone refining, crystal growth) were
considered and dropped as niche — water wins because it names the **failure mode** (boil to
steam = the banked null) as well as the target (liquid), which crystal does not.

## Disposition

- **This note:** mints and synthesizes the concept. Incubating in-repo under `docs/experiments/`.
- **Experiment sibling:** [`experiment.md`](experiment.md) — **five** falsifiable H0s
  (control / observe / project / substrate / cache), stated before data, with an observation table
  (untested). Registered by pointer in the ecosystem lab-notebook index
  (`design-docs/experiments/README.md`).
- **Graduation trigger:** a confirmed H0 → an ADR in `decisions/` (a socket type and/or scheduler
  seam from the seam inventory). Until then it incubates here.

---

## Addendum (2026-07-13) — intervention surface grounded; the mechanisms are real

The 2026-07-13 intervention-surface sweep (grounded against the pipeline/scheduler source, banked
in issues #23 / #28 / #35 / #36) changes this note's standing on one axis: the mechanisms the two
faces require are no longer *reachable-in-principle* — they are **located in source**, and the
record's biggest implementability question is closed. Pointer-only below; the derivations live in
the cited issue comments.

1. **Knob liveness is proven — every EB scalar is live-mutable per step.**
   `EntropyBoundScheduler.step()` reads `entropy_bound`, `t_min`, `t_max` fresh from `self.config`
   on *every* call; nothing is baked at `set_timesteps` time. A step-end callback mutates
   `pipe.scheduler.config` and the change takes the next step. **Exact per-step temperature falls
   out:** `t_min = t_max = v` degenerates the anneal formula to `v`. The only guard rail is
   `num_inference_steps` (mutate it mid-run and the pipeline's cached step counts desync — #20's
   mechanism). Source: #23 comments (2026-07-13). This is H0-control's drive mechanism, in hand —
   no vendoring.

2. **The logit door — the per-position heat field the control face required.**
   The pipeline has no `logits_processor` param, but a `register_forward_hook` on `pipe.model`
   (reachable from the callback's `pipe` arg, or engine-installed before the run) mutates the
   returned `.logits` the commit rule consumes, and propagation is coherent — self-conditioning
   carries `pred_logits` derived from the *same* mutated tensor, so no split-brain between
   constraint and conditioning. **Flatten the logits ⇒ hold a position liquid; sharpen ⇒ commit
   early.** This closes the record's biggest open implementability question: the per-position heat
   field is a forward hook, not a wish. Sealed alternatives named honestly (`self_conditioning_logits`,
   `argmax_history`, `cur_input_ids` are `__call__`-scope locals; returning `{"logits": ...}` from
   the callback is silently discarded — the hook is the *only* logit door). Source: #28 comment
   (2026-07-13).

3. **Where the liquid actually lives.** The liquid is carried by the **self-conditioning channel**
   (`pred_logits`) over the **fixed prefill KV** — not by the canvas. The canvas is the
   **measurement register**: rejected positions are renoised each step (`torch.randint` over the
   full vocab) and the scheduler is stateless, so a held position's mobility is read *off the
   distribution*, not off canvas persistence. This is why the observation face needs the widened
   callback (DISTRIBUTION capture): the liquid is observable there and nowhere in committed state.

4. **Equilibrate-then-quench is a canvas-scale protocol under H0-control.** A **hold phase**
   (an ADSR-style envelope on temperature / entropy_bound / β, driven by the per-step knob mutation
   of (1)) holds the canvas in the liquid basin; then a **quench** cools it to commitment.
   Because the held state is a `CANVAS_STATE` (ADR-CDG-005), **N quenches can fork from one held
   state** — the multi-sample-from-one-equilibrium move #36's comment names as belonging on the
   one contract, not the graph loop.

5. **β-viscosity renoise / the superposition cloze — nearly free.** The scheduler *already*
   computes `sampled_tokens` for **every** position each step and discards them at rejected
   positions; a β-mixture renoise (draw held positions from top-k of the step's own distribution,
   weight β against uniform) is essentially **one `torch.where`**. β is the **viscosity knob** —
   the VISION §3.3 convergence named directly (uniform renoise → structureless polyglot soup;
   β<1 → a cloud of near-meanings). This also **inverts #6**: #6 asked whether *plausible noise
   fools the commit rule* (an adversarial attack); read as an instrument, that same plausible-noise
   renoise *is* the viscosity term — attack becomes control surface. (One β-sweep protocol answers
   both; see `experiment.md` H0-renoise.)

6. **Architecture reviewed for this load — verdict and the constraints it fixed.** The 2026-07-13
   Opus-tier review (#35) returned **"needs targeted refactors first, not a redesign"**: the bones
   (CDG-003 seam, fake-pipeline testing, native-socket discipline) survive, ~a week of seam work.
   Two constraints this addendum's mechanisms must respect, both firmed by the review's delta pass:
   the composite of engine participants (β-renoise, walker, pin, capture) is **composer-ordered**
   (capture pre-pin, pin last writer); and `run_diffusion` widens by **declarative payloads only**
   (`constraints=` / `control_signals=` / `capture=`, validated at ingress) — **never
   surface-built closures or hooks** (the callback's `pipe.model` reachability is explicitly *not*
   a sanctioned installation path; engine-installed hooks ride the R5 lifecycle manager). See #35
   and [`ARCHITECTURE.md`](../../../ARCHITECTURE.md).
