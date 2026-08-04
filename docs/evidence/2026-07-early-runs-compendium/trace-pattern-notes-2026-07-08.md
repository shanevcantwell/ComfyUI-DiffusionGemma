# Trace-pattern notes — 2026-07-08 (working notes, local/gitignored)

Source traces (local, gitignored provenance in `handoffs/`):
`DG_ocean_starts_with_d_01.txt`, `DG_volume_of_a_sphere.txt`,
`DG_volume_of_a_sphere_triple_inegral.txt`, `initial_tests_2026-07-08T00-25-57.txt`.

Purpose: distillation source for VISION.md §1/§3.5 tag revisions and the
experiment-issue set. Observations are tagged **[observed]** (seen directly in a
trace this pass) vs **[inferred]** (consistent with observation + the architecture
note, but not directly confirmed here). Nothing here is `[established]` until it
survives the paper-verification gate and a fresh-head VISION pass.

---

## 1. The `committed_fraction` sawtooth is BLOCK-LOCAL, not canvas re-melt

**The correction.** `DG_volume_of_a_sphere.txt` logs `committed_fraction` per step
(40 values) and it sawtooths: ramps to `1.0000`, crashes to ~`0.01`, three cycles
(crash at steps 11, 20, 32). A first-pass digest read this as "non-monotonic
commit / re-melt (Strong)." **That reading is wrong at the document level.**

- **[observed]** At the step logged `committed_fraction=1.0000`, the canvas tail is
  *still polyglot noise* (the `### 2. Solving the integral` region is unfinished,
  trailing into `░NOISE░`). A fraction of 1.0 with a visibly-noisy canvas is only
  possible if the fraction is measured over the **active block/window**, not the
  whole 256-token canvas.
- **[observed]** Across the 11 text snapshots, the crystallized *prefix* grows
  monotonically. `### 1. Setting up the integral` firms by ~snapshot 7 and is
  stable + correct through 8/9/10. No snapshot shows a settled prefix reverting to
  noise.
- **[inferred]** Therefore the sawtooth crashes are block-boundary accounting (a
  completed block commits → the next block's fraction starts near zero from the
  melt), **not** the document re-melting and re-annealing. This matches the
  block-diffusion + static-prompt-KV architecture (see architecture note, pending).

**Consequence for VISION.** §1 point 2 and §3.5 currently cite "committed fraction
can fall between steps" as the visible signature of re-melt / self-correction.
That signal, in a multi-block run, is dominated by block advancement — it is the
WRONG signal to cite for self-correction. Cite the token-level evidence (§2 below)
instead, and describe the sawtooth honestly as the block structure made visible.

**SETTLED (by `initial_tests`, see §6).** The block-advance vs. re-anneal question
is now answered at the block level: in `initial_tests` the `committed_fraction`
crashes to `0.0039` at snapshot 21 while sections 1&2 (already resolved to
`exhausted`/`energized`) remain **stable and intact**, and sections 3&4 appear
fresh. A completed block's content survives the next block's cf-crash → the crash
is **a new block from the melt, not canvas re-anneal.** Issue **#11 (raw canvas
token ids)** is downgraded from prerequisite to *nice-to-have* — still needed for
the finer *within-block* per-position question, no longer for the block-level one.

---

## 2. Genuine re-melt is TOKEN-LEVEL, within a block — the ocean trace shows it clean

`DG_ocean_starts_with_d_01.txt` (prompt: ocean sentences starting with "D"), 12
steps, no `committed_fraction` logged, short enough to be ~one block.

- **[observed]** A single fixed slot oscillates: position 2 reads
  `depths`(s5) → `கோட்ப`(s6) → `depths`(s7–8) → `çarp`(s9) → `arabe`(s10) →
  `depths`(s11–12). A position commits, **re-melts to noise, and re-commits** —
  non-monotonic at the token level, within a coherent single-block evolution.
  This is the real mechanism VISION §3.5 depends on, and it is clean here.
- **[inferred]** So "re-melt" lives at two scales: (a) token/position re-melt
  inside a block's annealing = genuine, self-correction-relevant; (b) the
  block-boundary sawtooth = advancement, not re-melt. VISION should name both and
  not conflate them.

---

## 3. Frozen defects: the "sunlight / Vast survive" pair, live

Same ocean trace. The "starts with D" constraint holds everywhere except one slot.

- **[observed]** `Dazzling sunlight descends.` locks by s9 and survives untouched
  to the end — an early, correct crystallization that never re-melts.
- **[observed]** One slot's *skeleton* ("_____, dream-filled domain.") locks early
  but its leading adjective never resolves to a D-word:
  `немає`(s10) → `LG`(s11) → **`Vast`**(s12). The entropy bound accepts a
  constraint-violating token because the surrounding skeleton froze before the
  slot's filler resolved, and nothing forces a re-melt once cumulative entropy is
  under bound.
- **[inferred]** This is a concrete, watchable instance of **premature-commitment /
  structural-anchoring failure** — the SFT length-anchoring pathology VISION §3.2
  names in the abstract, here visible at a single token. Candidate figure for
  VISION §3.2/§3.5. It also motivates the **anti-EB / acceptance-inversion**
  experiment: does inverting the accept order rescue or worsen these frozen-defect
  slots?

---

## 4. Length blow-out in the final block

- **[observed]** In the sphere trace, snapshots 1–10 are ~800–1400 chars (block 1
  building); snapshot 11 (the final state) is ~26 KB — the last block "blows out"
  and generates the entire remaining derivation + answer at once.
- **[observed — CONFIRMED in the 61-step cloze run, upgraded from parked]** The
  blow-out is **over-generation into an over-provisioned canvas**. In
  `initial_tests_...05-41` the genuine answer converges in block 1 (~900 chars,
  cf→1.0 by step 41); at the step-42 block boundary a much larger region opens
  (cf crashes to 0.004) and the tail fills with **repeated/overlapping copies of
  the answer** — the final 22.7 KB canvas contains `### 4.` **×11** and the closing
  line **×5**, interleaved with melt. The canvas is far longer than the prompt's
  content needs, so the model pads it with template repetition. This is the
  length-anchoring / tail-repetition pathology, now strongly evidenced (was only
  weakly seen in the sphere trace).
- **[observed] Block-advance re-confirmed at the crash.** Snapshot 42's prefix is
  byte-identical to 41 (sections 1&2 intact) with sections 3&4 appended — the
  block boundary advances, it does not discard the completed block. Third
  independent confirmation of §1's block-local-cf finding.
- **[RESOLVED — operator-confirmed 2026-07-08] Two canvases, not a dump artifact.**
  The 900→22.7 KB jump is a genuine **second canvas** opening at the block
  boundary, not a frame-dump scope switch. This is why the boundary reads as "it
  threw the whole thing out": the display doesn't represent the two-canvas
  structure. Filed as **#26** (display handling for the multi-canvas structure —
  per-canvas panels / boundary marker / block-local `committed_fraction` label).
  Residual mechanics still worth a read for the fix: exact `canvas_length` per
  canvas and whether canvas 2 is over-provisioned/tunable (the padding-repetition
  source).

---

## 5. Polyglot noise signature — confirmed uniform-state, across traces

- **[observed]** Ocean s1–4, sphere s1–2, `initial_tests` s1–10 all show dense
  multilingual/CJK/Indic + reserved `<unusedNNNN>` token garble in early frames,
  clearing as content crystallizes. This is the visible signature of uniform-state
  renoise (18-bit draw over the full vocab) exactly as VISION §1 point 1 claims —
  the traces corroborate that claim directly. `<unused1223>`, `<unused2911>`,
  `<unused5506>` etc. appear verbatim in the melt.

---

## 6. `initial_tests` — the semantic-map layer (a whole different level)

Prompt (meta, about semantic precision): *"I think I know precisely what I mean /
when I say it's a [what?] day"*. 38 steps, two blocks (cf ramps to 1.0 at step 20,
crashes to 0.0039 at step 21, ramps again to 1.0 at step 38).

- **[observed] Block-advance confirmed** (see §1 SETTLED): block 1 (steps 1–20)
  resolves sections **1. overwhelmed/exhausted → "heavy" day** and
  **2. productive/energized → "momentum" day**; the step-21 cf-crash begins block 2
  which adds **3. quiet/reflective → "slow" day** and **4. things going wrong →
  "glitchy" day**. Sections 1&2 stay verbatim-stable across the crash.
- **[observed] Per-slot semantic search.** The condition slot in §1 digs through
  candidates before freezing on the right synonym-partner of "overwhelmed":
  `lifting`(s9) → `consiste`(s11) → `parasitic`(s12) → `Ily`(s15) →
  `stucco`(s17) → **`exhausted`**(s18). Some candidates are noise; the trajectory
  trends toward the semantic target. The commit heatmap for such a slot is a
  *watchable semantic search*, not just a freeze time.
- **[observed] The content is a polymorphism table.** Asked for one precise word,
  the model enumerates the field — `heavy / momentum / slow / glitchy`, each tagged
  with the selecting condition and a gloss. This is a slice of the §3.4 "phase
  diagram of meaning over a slot" that VISION says *no one has charted* — the model
  charts it in-content, and the trace shows the per-cell search that finds it. The
  instrument (watch a slot dig) and the content (model enumerating candidates) are
  the same phenomenon at two levels; the prompt being *about* semantic precision
  makes the resonance explicit. Strong candidate centerpiece for an outreach
  writeup and for a VISION §3.4 figure.

- **[inferred] Nuance for §3.3.** §3.3 frames the "cloud of near-meanings" as
  something you must *engineer* by biasing the renoise distribution toward
  semantic neighbors. But this search happens under **uniform** renoise: once a
  slot's context is frozen, the model's *own conditional* over that position is
  already concentrated on plausible fillers, so the cloud emerges from the model,
  not the noise source. §3.3 should be revised to distinguish "cloud from the
  model's conditional" (free, happens now) from "cloud from engineered renoise"
  (the frontier swap) — they are different mechanisms and the trace shows the
  first.

---

## 7. Falsification run — cloze @ `entropy_bound=0.030`, 61 steps (2026-07-08T05-41)

Same prompt as §6, `entropy_bound` lowered 0.05→0.030 (operator intent: discourage
token-pinning). Result: 61 steps / **61 images/frames** to converge.

- **[observed] Monotonic front-advance HELD under the sharpest test.** Block-1
  de-noised text is **byte-identical across snapshots 25/33/41** — the full
  17-snapshot near-converged plateau. Settled sections do NOT re-melt while the
  frontier resolves. The `committed_fraction` wobbles (`0.996→0.992→0.996`) are
  1–2 frontier positions flickering, not document-level re-melt. The strongest
  claim in these notes survived 61 steps + tighter bound. **Not falsified.**
- **[observed] Block structure invariant to eb.** Still 2 blocks (single cf-crash,
  step 42) exactly as the 38-step run. Block *count* is a property of the
  content/canvas; step-*dwell* per block scales ~1/eb (block 1: 41 steps here vs
  20 at eb=0.05).
- **[observed → hypothesis] `entropy_bound` is a RESOLUTION knob on the semantic
  map.** eb=0.05 gave four one-word sections (`heavy`/`momentum`/`slow`/`glitchy`).
  eb=0.030 enumerates each section's whole neighborhood (§1: `survival`/`heavy`/
  `low-battery`/`brain-fog`; §2: `main-character`/`momentum`/`flow`). **Both
  eb=0.05 answers (`heavy`, `momentum`) appear as items inside the eb=0.030
  enumeration** — the coarse run collapses each cluster to one representative; the
  fine run materializes the cluster. Same crystal, higher resolution — NOT a
  truth-flip. Directly instruments §3.3 (cloud of near-meanings) and §3.4 (phase
  diagram): eb dials how much of the cloud precipitates. **[hypothesis, n=2 runs]**
  — confirm with an eb sweep (0.05 / 0.03 / 0.01) checking for monotonic
  elaboration. Strong VISION-grade candidate.
- **[observed] Convergence overran the config.** `num_inference_steps=48` but the
  run took 61 steps → stopping is convergence-gated and can EXCEED the nominal
  step count. The frame/step axis (dim_1 of the heatmap; dim_0 of the images batch)
  is therefore unbounded by the config knob and ranges widely with eb (38 @0.05 →
  61 @0.030). This is the same fact underlying the 25-vs-29 frame-axis mismatch:
  any node hardcoding a frame count breaks when eb changes. Enforcement-surface
  gap, unchanged from §1's note.
- **[observed] Compute cost of low eb: a dead plateau.** ~17 snapshots (25→41) where
  the block is otherwise fully formed but the scheduler keeps spending full forward
  passes dithering on the last 1–2 positions before block-advance. Under the
  "every step is a full pass" frame, that's ~17 near-wasted passes on a settled
  block. What those flickering positions ARE needs per-position logging (**#11**).

**Net:** the falsification attempt did not falsify — front-advance and block
structure held under a tighter bound — but it yielded the eb-as-resolution-knob
finding, which is stronger than the confirmation it was testing.

---

## Downstream actions (for the fresh-head pass, do NOT auto-apply)

1. **VISION §1 pt 2 + §3.5** — rewrite the re-melt evidence: cite token-level
   oscillation (§2), NOT the `committed_fraction` sawtooth (which is block
   structure). Add the two-scale distinction. Honest correction, not a new claim.
2. **VISION §3.2/§3.5** — the sunlight/Vast frozen-defect pair is a real, shippable
   figure of premature commitment. Promote from abstract to observed-on-this-model.
3. **Issue #11** is the prerequisite instrumentation to settle block-advance vs
   re-anneal (per-step, per-position commit logging). Note the dependency.
4. **Anti-EB / acceptance-inversion experiment** gains a concrete motivation: the
   frozen-defect slot. Fold into that issue when opened.
5. `committed_fraction` should be reported/labeled as **block-local** in the trace
   node UI, or it will keep being misread as canvas re-melt (the digest just did).
