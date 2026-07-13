# Experiment — liquid-state decoding (the forgotten intermediate)

**Status:** untested · **opened:** 2026-07-12 (curator seat, co-framed with operator)
**Concept note:** [`concept.md`](concept.md)
**Target instrument:** `ComfyUI-DiffusionGemma` (per-step callback, ADR-CDG-004)

---

## Reasoning at decision time

DiffusionGemma's commit rule is a sharp threshold: a position sits above `entropy_bound`
(steam — renoised, high-entropy) or below it (frozen — committed), crossing **directly**
between them. There is no **liquid** basin — no stable intermediate where a token is mobile
*and* coherent. The system **sublimates**.

The prior attempt to keep positions "open" longer (`confidence=0.0`, banked in
`ComfyUI-DiffusionGemma/loose-ends.md` 2026-07-06) produced *"escape attempts, not steering"* —
i.e. it sublimed straight to steam. Issue #10's byte-identical dead-zone (0.005–0.1) corroborates
that naive heat changes nothing. The hypothesis under test is that **balancing heat re-injection
against the annealing/commit threshold** opens a liquid basin the prior attempt overshot.

Feasibility is grounded, not assumed: per-step reach into the sampler is proven (ADR-CDG-004
`callback_on_step_end`). The known constraint is that ComfyUI offers **no node-graph
incrementer**, so this runs **in-callback**, inside one node execution — not as a stepped graph.

## Expected behavior, pre-stated (before any observation)

### H0-control — the liquid basin steers where heat alone did not

*Prediction.* With heat re-injection balanced against a raised commit threshold at a chosen
position, that position stays **mobile-but-coherent** across additional timesteps: its per-step
top-1 candidate stays within a **small, semantically-related candidate set** (liquid) rather than
going **uniform-random** (steam) or collapsing immediately (frozen). Freezing it **last** yields
a committed token that (a) differs from its natural-commit-order token and (b) scores higher on a
context-fit metric than natural order.

*Falsified if* held positions either (a) collapse immediately anyway — the threshold cannot hold
them, no basin — or (b) go uniform-random across the held steps — straight to steam, reproducing
the 2026-07-06 null. Either outcome says the liquid phase does not exist under this control. **A
miss here is a banked finding: the sublimation is not separable by this lever.**

### H0-observe — the held distribution carries meaning the scalar discards

*Prediction.* At a position held liquid, the captured per-step distribution (top-k candidates +
weights) carries usable multi-meaning structure beyond #14's scalar entropy: two positions with
**equal scalar entropy** show **materially different candidate sets**, and a mot-juste slot's
candidate set **narrows meaningfully** as its context freezes around it.

*Falsified if* candidate sets at equal-entropy positions are interchangeable/uninformative, or
do not narrow as context commits — then capturing the distribution buys nothing over the scalar,
and #14's shadow is the whole signal.

### H0-project — a directed operator over the held distribution selects among co-present stylistic modes

*(Downstream — presupposes H0-control AND H0-observe. This is where the 2026-07-12 lit scan placed
the idea's "apparently novel" verdict: metastable-hold + distribution-sampling as a **creative/style**
axis selector, not a correctness lever, was unclaimed across ~50 papers.)*

*Prediction.* A held-liquid position's distribution is **multi-modal** — it carries separated
candidate clusters corresponding to distinct stylistic/semantic readings (register `walked/strode/
ambled`; tense `runs/ran/running`; mood indicative/subjunctive). A **directed** sampling operator
(e.g. reweight toward low-frequency/high-surprisal mass; mask to mood-inflected candidates) selects
which mode collapses — the *same slot* yields a poetic vs literal, or subjunctive vs indicative,
token depending on the operator, **without re-running generation**. Per `Steering Without Breaking`
(arXiv:2605.10971, "attributes commit on different schedules": topic <2%, sentiment ~20%), each axis
has its own freeze-time, so **register freedom survives in more positions than tense freedom** (syntax
pins tense harder than register).

*Falsified if* held distributions are effectively unimodal / carry no separable stylistic modes, or a
directed operator yields the same token regardless — then style lives in the *trajectory/guidance*
(the `RegDiff` / `Steering Without Breaking` approach) and not in the held distribution's shape, and
our distinct claim collapses into existing guidance methods. Build-from pair if it holds:
`Soft-Masked DLM` (2510.17206) × `Steering Without Breaking` (2605.10971).

### H0-substrate — a non-causal-prefill diffusion LM holds richer liquid than DG

*(Substrate-comparison. Greenfield anticipated-failure-mode: the differential diagnosis if
H0-project fails on DG — wrong substrate, not wrong idea.)*

*Prediction.* DiffusionGemma is one Gemma-4 backbone with a **causal prompt prefill** (verified
2026-07-12) and a *measured* partial L→R commit bias (τ≈0.43–0.60, arXiv:2606.14620) — both cap
co-liquidity. A diffusion LM **without** a causal prompt-prefill confound (candidate: LLaDA,
from-scratch masked diffusion) exhibits **richer liquid** than DG.

*Operational "variety" — OPEN, operator to settle before running.* Provisional, non-equivalent
candidates: (a) **co-liquidity** (positions simultaneously liquid); (b) **per-position breadth**
(multi-modality of each held distribution); (c) **cross-seed diversity** (full-sequence variety at
fixed prompt). The measure chosen determines what this H0 tests.

*Falsified if* the non-causal-prefill model shows no richer liquid on the chosen measure — then the
variety ceiling is not the substrate and DG is a fair testbed. *Note the confound:* mask-vs-uniform
noise and scratch-vs-adapted training move together across DG↔LLaDA; isolate carefully.
*Honest tradeoff to hold:* pure-diffusion variety trades against coherence (SEDD/MDLM underperform
AR on fluency).

*Caveat carried from architecture verify:* DG's training provenance (AR-adapted vs from-scratch) is
**undocumented** — do not attribute the confound to "AR heritage in the weights"; attribute it to
the causal-prefill *mechanism* + emergent commit-order, which is what is actually verified.

### H0-cache — polyphonic prefill: assembling the KV cache from multiple prefills steers the liquid

*(Upstream twin of H0-project — shape the conditioning field rather than sample the output. Apparently
novel per 2026-07-12 scan: diffusion-LM-native, training-free, multi-prefill cache assembly for steering
was not found in 2024–2026.)*

*Prediction.* Assembling DiffusionGemma's write-once/read-only prefix KV cache from multiple prefills and
denoising off it steers generation in a way a single prefill cannot. **(A) Concatenation** of N framings'
K/V lets bidirectional denoise attend to all; **(B) interpolation** (`α·K_A+(1−α)·K_B`) yields a
continuous register/style knob.

*Constraints to design around (measured in adjacent literature, not to be rediscovered):* concat loses
cross-segment attention because independently-prefilled chunks are mutually blind — needs `CacheBlend`-style
(arXiv:2405.16444) selective recompute; `Cache Merging` (2607.01308) shows fusion degrades past k=2 without
it. Form (B) off-manifold validity (mean-of-keys ≠ key-of-mean) is **untested anywhere — this experiment's
ablation is the first check**, not a solved question.

*Falsified if* (A) assembled-cache generation is indistinguishable from best-single-prefill even after
bridging recompute, or (B) interpolation produces off-manifold degradation rather than a smooth style path.
*Build-from / distinguish:* `MasaCtrl` (2304.08465, mechanism) × `Models Take Notes at Prefill` (2606.17107,
composable-KV framing) — ours differs by diffusion-LM-native + steering + interpolation.

### H0-renoise — β-viscosity renoise: drawing held positions from their own distribution opens a liquid basin uniform noise skips

*(Pre-registered 2026-07-13, before any observation. The cheapest falsification in the program —
runs on today's callback plus one `torch.where`, no vendoring. The scheduler already computes
`sampled_tokens` for every position each step and discards them at rejected positions; a β-mixture
renoise reweights that discard. Grounded in the 2026-07-13 intervention-surface sweep, `concept.md`
Addendum item 5; VISION §3.3's renoise axis.)*

*Prediction.* With renoise at held positions drawn from the **top-k of the step's own distribution**
(a β<1 mixture against uniform) instead of uniform noise, held positions stay **mobile-but-coherent**:
their per-step draws cluster in a **small, semantically-related candidate set** (liquid), where uniform
renoise **sublimates** — boils the position straight to structureless polyglot steam (the banked
2026-07-06 null). β is the viscosity knob: β→1 (uniform) reproduces the sublimation; β<1 should open
and hold the intermediate basin.

*Pre-stated failure mode — self-collapse.* The model was trained to denoise **uniform** noise. Feeding
it plausible-but-uncommitted tokens as renoise may read as *signal* rather than noise: entropy drops
everywhere, the commit rule sees false confidence, and the cascade fires prematurely — the cloud
**crystallizes instead of holding.** This is the specific, anticipated way β<1 could destroy the very
liquid it was meant to open.

*Falsified if* the β-sweep shows only the **two existing phases** — immediate collapse (self-collapse,
above) or steam (β too close to uniform) — with **no held intermediate** at any β. Then uniform-state
renoise has no viscosity axis and the liquid basin is not reachable by shaping the renoise
distribution.

*Note — one protocol, two questions.* This is the **same mechanism as issue #6's adversarial-renoise
question, inverted.** #6 asked whether *plausible noise fools the entropy-bound commit rule* (framed as
an attack); the self-collapse branch above **is exactly that failure** — plausible noise fooling the
commit rule into premature crystallization. One β-sweep protocol answers both: read as viscosity
control it is H0-renoise; read as an attack it is #6. A confirmation here is a control knob; the
self-collapse branch is #6's "yes, plausible noise fools the rule."

## Room for observations

Append-only. Never retro-fit a prediction to a result. Verdict ∈ {untested, observed, falsified,
held}.

| date | H0 | setup (steps / t / entropy_bound / threshold / held positions) | observation | verdict |
|---|---|---|---|---|
| — | — | — | (none yet) | untested |
| — | H0-renoise | — | (pre-registered 2026-07-13; not yet run) | untested |
