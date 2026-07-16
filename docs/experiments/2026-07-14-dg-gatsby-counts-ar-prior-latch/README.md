> Relocated 2026-07-15 from design-docs/experiments/2026-07-14-dg-gatsby-counts-ar-prior-latch/ (originally committed there at 6dfb07d, amended 6637fc2); design-docs is not cross-repo-written — this repo is the record's home.

# DiffusionGemma Gatsby-counts: AR-prior latch vs. diffusion-computed tally

> **SUPERSEDING NOTE (2026-07-15):** the generalization of this run's H0-confirmation to
> "the diffusion performs no checkable work under default usage" is OVERTURNED by
> `../2026-07-15-dg-numeral-counts-update-in-response` — on a clean single-token numeral
> task the diffusion visibly updates counts against the annealing canvas (count of '4':
> 2→3 as the 10th position resolves). The one-shot freezes recorded below are now
> attributed to a MULTI-TOKEN-WORD confound, not to an absence of reconciliation. The
> observations below stand as data; the generalized conclusion does not.

**Status:** falsified (H0 CONFIRMED — the process performs no meaningful checkable work under default usage) (conclusion superseded 2026-07-15 — see note above)
**Date:** 2026-07-14 (run timestamp `2026-07-14T01-31-09` UTC)
**Repo touched:** this repo (the instrument); the record now lives beside it.
**Discipline:** locked hypothesis stated before observation; verdict reached by two independent Opus precision-review passes over the per-step canvases + commit heatmap.

## Reasoning at decision time (why this run)

A word-frequency tally is *checkable work*: the correct answer is derivable, and — crucially
— the **commit order** across diffusion steps reveals whether the process computed the counts
or merely pattern-completed them. If the count numerals freeze before the evidence they
depend on (the recalled source sentences, the co-listed words) ever materializes in-canvas,
then the counts cannot have been computed *from* anything in the canvas. That is the
signature this run was designed to expose.

## H0 — expected behavior, pre-stated (before observation)

> Under default usage the diffusion process performs no meaningful checkable work — count
> values may freeze **before / without** the evidence they depend on (AR-prior emission,
> not computation).

Falsifiable prediction: if the process *computes*, count numerals freeze **after** the
recalled source text and co-listed words stabilize (evidence-first ordering). If H0 holds,
numerals freeze one-shot from a memorized answer-shape, ordered independently of — or ahead
of — any usable in-canvas evidence.

## Method

- **Task prompt** (verbatim): "Alphabetize and tally each of the words appearing more than
  one time in the first 2 sentences of The Great Gatsby."
- **Scheduler config:** `EntropyBoundScheduler`, `entropy_bound=0.05`, `t=[0.4, 0.8]`,
  `num_inference_steps_requested=48`, `num_inference_steps_effective=18` (18 committed
  steps; the txt emits exactly 18 thought-delimited canvas blocks, one per step).
- **Telemetry captured:** per-step `committed_fraction` (block-local), per-step rendered
  canvas (thought channel + answer channel), and a per-token commit heatmap PNG
  (8192×576 skyline: x = token position, bar height = freeze step).
- **Review:** two independent Opus precision-review passes over the artifacts —
  - Pass 1: ground truth, final-state accuracy, commit-timeline table, telemetry fidelity, verdict.
  - Pass 2: numeral/label revision history (Check 1) + same-step evidence sync (Check 2),
    which re-blocked the file and issued a **step-alignment correction** superseding pass 1's
    compressed step numbering (correct alignment: 18 thought-delimited blocks = steps 1–18;
    numeral freezes fall on s10–s17). All step numbers below use the corrected alignment.

## Run artifacts

- `design-docs:incoming-ideas/DG-runs/counts/gatsby_counts_2026-07-14T01-31-09_0000.txt` (222 lines: header, per-step `committed_fraction`, per-step canvases) — sibling repo, not relocated with this record
- `design-docs:incoming-ideas/DG-runs/counts/gatsby_counts_2026-07-14T01-31-09_0000_heat_00001_.png` (per-token freeze-step heatmap) — sibling repo, not relocated with this record

## Results

### Ground truth (Scribner text)

Real first two sentences:

> "In my younger and more vulnerable years my father gave me some advice that I've been
> turning over in my mind ever since. 'Whenever you feel like criticizing any one,' he told
> me, 'just remember that all the people in this world haven't had the advantages that
> you've had.'"

Correct alphabetized tally of words appearing >1 time —

**Convention A** (lowercase; punctuation stripped; contractions kept whole — `I've`,
`haven't`, `you've` as single tokens; the most defensible default):

| word | count |
|---|---|
| had | 2 |
| in | 3 |
| me | 2 |
| my | 3 |
| that | 3 |
| the | 2 |

(6 words.) Convention variants: **B** (apostrophe as delimiter, `I've`→`i`/`ve`) adds
`ve: 2` and `you: 2`; **C** (case-sensitive) would drop `in` to 2 (sentence-initial "In").
Case-folding is near-universal for this task, so A is the reference answer.

### Final-state accuracy scorecard

Run's final frozen answer (step 18): `and:2, any:2, as:2, father:2, have:2, I:3, in:2,
it:2, me:2, my:3, of:2, that:2, to:2, you:2` (14 lines).

Scored against the Convention-A truth set `{had:2, in:3, me:2, my:3, that:3, the:2}`:

- **Only 4 of 14 listed words** are real duplicates: `in`, `me`, `my`, `that`.
- **Exactly 2 of 14 counts are fully correct:** `me:2` and `my:3` (both arguably coincidental — see headline evidence).
- **Real duplicates omitted entirely:** `had`, `the`.
- **10 of 14 listed words** are spurious — not duplicated in the real text (`and, any, as, father, have, I, it, of, to, you`).

**The run never materializes the real Gatsby sentences at any step.** Final recalled text
(step 18) is a confabulated paraphrase:

> "In my younger and more sensitive years my father gave me some advice inclined to make
> judgments. 'Whenever you feel like criticizing anybody,' he said to me, 'remember that all
> the world has not had the advantages that you have.'"

Corruptions vs. real: "sensitive" (real: *vulnerable*), "advice inclined to make judgments"
(real: *advice that I've been turning over in my mind ever since*), "anybody" (real: *any
one*), "he said to me" (real: *he told me*), "all the world has not had" (real: *all the
people in this world haven't had*), "you have" (real: *you've had*). Internal-consistency
failure: tallying the run's **own** recalled text yields `{me:2, my:2, that:2, the:2, to:2,
you:2}` — matching **neither** the ground truth **nor** the run's own emitted list.

### Check 1 — numeral & label revision history

Method: slot-mapped each of the 14 final tally lines backward through every step where a
tally list exists (steps 7–18), reading the value token verbatim at each step. Positional
correspondence exact for steps 12–18 (full 14-line list, 1:1 with final); steps 7–11
(fragmentary lists of 8–13 lines) mapped by alphabetical slot position.

| word | full value history (step: verbatim value) | numeral froze | class |
|---|---|---|---|
| and | s7 `ofx` (L18) → s8 `dst` (L28) → s9 `([],` (L41) → s10 `2` (L57) → 2 thereafter | s10 | (b) one-shot |
| any | s8 `ADDRESS` → s9 `ન્ડ` → s10 `serializers` → s11 `skies` → s12 `극` (L93) → s13 `2` (L111) → 2 | s13 | (b) one-shot |
| as | s8 `Convention` → s9 `стоят` → s10 `2` (L59, under garbage label `সমর্থ`) → s11 `2` (L76) → 2 | **s10** | (b) one-shot — see label note |
| father | s8 `inverso` (L31) → s9 `அது` (L44) → s10 `2` (L60) → 2 | s10 | (b) one-shot |
| have | s8 `Bless` → s9 `দর্শ` → s10 `একজনকে` → s11 `inac` (L78) → s12 `2` (L96) → 2 | s12 | (b) one-shot |
| I | s9 `convocatoria` (L46) → s10 `BURGH` (L62) → s11 `items` → s12 `nutritious` → s13 `3` (L115) → 3 | s13 | (b) one-shot |
| in | s9 `метр` → s10 `SPDX` (L63) → s11 `berhasil` → s12 `2` (L98) → 2 | s12 | (b) one-shot |
| it | s10 `伖` (L64) → s11 `Kissinger` → s12 `परि` → s13 `ು` (L117) → s14 `2` (L137) → 2 | s14 | (b) one-shot |
| me | s8 `refunded` (L36) → s9 `dunia` → s10 `اة` → s11 `செல்லும்` → s12 `皓` → s13 `厳しい` → s14 `2` (L138) → 2 | s14 | (b) one-shot |
| my | s8 `receb` (L37) → s9 `GUIContent` → s10 `フォー` → s11 `晝` → s12 `чрезвы` → s13 `holiness` → s14 `Fetch` → s15 `infrequently` (L159) → s16 `otheby` (L179) → s17 `3` (L199) → s18 `3` (L219) | s17 | (b) one-shot — latest freezer |
| of | s9 `લું` (slot-inferred) → s10 `新疆` (L67) → s11 `Giv` → s12 `2` (L102) → 2 | s12 | (b) one-shot |
| that | s10 `UO` (L68) → s11 `プレ` → s12 `इसी` → s13 `2` (L121) → 2 | s13 | (b) one-shot |
| to | s11 `locatie` (L86) → s12 `')}>` → s13 `Stalingrad` (L122) → s14 `Вин` (L142) → s15 `görev` (L162) → s16 `2` (L182) → 2 | s16 | (b) one-shot |
| you | s12 `<unused2341> madrid` (L105) → s13 `2` (L123) → s14 `2aarrggbb` → s15 `2anyika` → 2 | s13 | (b) one-shot |

**Tally: 0 revised / 14 one-shot / 0 indeterminate.** No line ever held a different legible
numeral — every value slot went straight from non-numeric renoise garbage to its final digit
and never moved.

Label history notes:
- **`it` label flicker** (the one observed label reversal): `it` at s10 (L64) → `Descriptive`
  at s11 (L81, same slot 8) → `it` again at s12 (L99) — subject to the s11 slot-mapping caveat.
- **`as` numeral froze one step BEFORE its own label:** s10 (L59) reads `** সমর্থ:** 2` in the
  as-slot; the label becomes `as` only at s11 (L76). The count committed one step ahead of the
  word it counts.

### Check 2 — same-step evidence sync

At each word's stabilization step S, count legible whole-word occurrences of that word across
step S's full canvas (thought header + recalled-sentence line + list header), excluding the
tally label token itself.

| word | emitted | step S | visible occurrences at S (excl. label) | match? |
|---|---|---|---|---|
| and | 2 | s10 | **1** — "my younger and more sensitive" (L53) | NO |
| as | 2 | s10 | **0** — no standalone "as" in L53–68 | NO |
| father | 2 | s10 | **0** — s10 recall breaks off before "father" (L53) | NO |
| have | 2 | s12 | **1** — "hossz have headache" (L88) | NO |
| in | 2 | s12 | **1** — "*\"In my younger" (L88, case-folded) | NO |
| of | 2 | s12 | **1** — scaffold-only: "sentences **of** *The Great Gatsby*" (L88); zero in the quoted text | NO |
| any | 2 | s13 | **0** — only "anybody" (L106), not whole-word | NO |
| I | 3 | s13 | **0** — no standalone "I" in L106–108 | NO |
| that | 2 | s13 | **2** — "ато that perniciousiate", "advantages that you have" (L106) | YES (2) |
| you | 2 | s13 | **2** — "Whenever you feel", "that you have" (L106) | YES |
| it | 2 | s14 | **0** — none whole-word in L124–128 | NO |
| me | 2 | s14 | **1 clean + 1 garbled** — "said to me," clean; "gave meíc fB💏" (L126) ambiguous | NO (1 unambiguous) |
| to | 2 | s16 | **2** — "adviceвате to make judgments", "he said to me" (L166) | YES |
| my | 3 | s17 | **2** — "In my younger", "years my father" (L186); canvas fully legible, no garble to hide a third | **NO — canvas says 2, model froze 3** |

**Read-off-able: 3/14 (that, you, to). Evidence-preceding: 11/14.** Even the 3 matches are
ambiguous evidence for computation — all three are equally explained by prior-pattern
collision (e.g. `that:2` matches the confabulated canvas but the true count is 3; a
pattern-prior and a canvas-read give the same digit there).

## Headline evidence

- **`my: 3` frozen at s17 against a FULLY LEGIBLE canvas containing exactly 2 "my".** 3 is
  the true count for the *real* Gatsby text, which the canvas never contained. The digit is
  right for a text that isn't there — only the memorized prior supplies it. Strongest single
  datum: at s17 there is no garble to hide a third "my".
- **`as` digit froze one step before its own label** (s10 `** সমর্থ:** 2`; label "as" first
  legible at s11): the count committed ahead of the word it purports to count.

## Honest caveats

- **Step-alignment method.** Steps are inferred from the 18 thought-delimited canvas blocks
  (block index = step index 1–18), corroborated by the `committed_fraction` curve. Pass 2's
  re-blocking corrected pass 1's compressed 12–18 mapping; the verdict is unchanged and
  strengthened. Freeze-step precision is MEDIUM (block-level, not raw per-token indices).
- **No per-token commit flags.** The txt emits rendered canvas strings only. "Fraction of
  frozen tokens" is not independently countable from the artifact; a digit sampled-then-
  remelted *between* logged steps, or during a step where its slot rendered as garbage,
  **cannot be excluded** from rendered text alone (though no alternative digit is visible
  anywhere).
- **Telemetry qualitatively consistent, not digit-verifiable.** `committed_fraction` (18
  values, monotone 0.0352→1.0000) and the heatmap skyline are qualitatively consistent with
  the canvas coherence progression, but NOT independently checkable to the digit from these
  two artifacts — a gap, not a disagreement.

## Verdict

**H0 CONFIRMED (prediction falsified in the direction H0 predicted).** This run does not
show diffusion performing meaningful checkable work on this task under default usage.
Selection = freeze: 0 of 14 numerals ever displayed a different legible value before settling
(0 revised / 14 one-shot), so annealing did no visible revision work on the numbers; in two
lines the digit committed **before** its own word label (`as`, s10) or **against** a fully
legible canvas contradicting it (`my: 3`, s17). Only 3/14 counts were read-off-able from
their own stabilization-step canvas, and all three are equally explained by prior-pattern
collision. The counts are **one-shot emissions from a memorized answer-shape** — an AR-prior
template lock — never reconciled against in-canvas evidence. Confidence: HIGH on the
confabulation verdict (robust to step-alignment uncertainty); MEDIUM on exact per-word freeze
steps (block-level inference, not raw per-token indices).

## Follow-ups (stable handles)

1. **Novel-text tally** — run the same task over text with no memorized prior available, so
   any correct count must be computed rather than recalled. Isolates computation from
   template recall.
2. **Pinned-evidence tally via constraints** — pin the real source sentences into the canvas
   and require the tally over *that* fixed evidence.
   `shanevcantwell/ComfyUI-DiffusionGemma` PR #71 (ADR-010/011 P3, pin participant +
   logit-mask hook), CDG-010/011 P3.
3. **Inverse probe** — pin the counts and require the text to conform; measures whether the
   process can be driven from the answer side.
4. **KV-provenance separation** (incl. over-provisioned-KV variant) — inject
   known-provenance caches and diffuse over a deliberately over-provisioned cache, turning
   the cache from a completion template into a selection problem.
   `shanevcantwell/ComfyUI-DiffusionGemma` ADR-CDG-012, #62, #47.
5. **Per-token commit emission** — emit per-token commit-step indices + raw pre-excision
   token ids so the count-token-froze-at-N vs. evidence-token-froze-at-M claim can be made at
   the token level instead of inferred from block-level coherence. Closes the two-caveat gap
   above. `shanevcantwell/ComfyUI-DiffusionGemma` #72, #11, #61.
