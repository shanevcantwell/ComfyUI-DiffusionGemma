> Relocated 2026-07-15 from design-docs/experiments/2026-07-15-dg-numeral-counts-update-in-response/ (originally committed there at 6637fc2); design-docs is not cross-repo-written — this repo is the record's home.

# DiffusionGemma numeral-counts: counts update against the annealing canvas (overturns the gatsby AR-prior-latch generalization)

**Status:** H0 from the 2026-07-14 gatsby run (`../2026-07-14-dg-gatsby-counts-ar-prior-latch/`)
**OVERTURNED** on a clean single-token task — the diffusion demonstrably updates counts in
response to the annealing canvas.
**Date:** 2026-07-15
**Repo touched:** this repo (the instrument); the record now lives beside it.
**Supersedes (partially):** the generalized conclusion of
`../2026-07-14-dg-gatsby-counts-ar-prior-latch/` — see that record's superseding note. The
gatsby run's raw observations still stand; its generalization does not.

## Setup

- **Task:** "Sum the appearances of each numeral" over a set that anneals to the 13
  single-digit tokens `[4, 7, 2, 4, 9, 1, 7, 2, 2, 4, 7, 6, 9]`.
- **Scheduler config:** `EntropyBoundScheduler`, `entropy_bound=0.05`, `t_min=0.4`,
  `t_max=0.8`, `num_inference_steps` 48 requested / 48 effective, **26 steps observed**.
- **Telemetry:** `committed_fraction` climbs to `1.0000`.

## Observation — the load-bearing cells ('4' and '7')

- **Count of '4': 2 → 3.** Holds at 2 (correct for the two 4s visible at that point,
  positions 1 and 4) while the 10th canvas position thrashes through `5` and garbage
  tokens; updates to 3 the moment position 10 anneals to `4`. The tally **tracks a canvas
  position as it resolves** — this is the single load-bearing datum of the run.
- **Count of '7': 2 → 3.** Corrects an initial undercount (three 7s already visible at
  positions 2, 7, 11) up to 3 — reconciliation toward the visible evidence.
- **Final tally fully correct:** `0:0, 1:1, 2:3, 3:0, 4:3, 5:0, 6:1, 7:3, 8:0, 9:2` (13
  numerals total).

## Interpretation

- **FALSIFIED:** the general claim "under default usage the diffusion performs no
  meaningful checkable work; counts freeze one-shot and are never reconciled against
  evidence." `4: 2→3` slaved to the position-10 resolution is a direct counterexample —
  the count is reconciled against in-canvas evidence.
- **REINTERPRETED, not discarded:** the gatsby "H0 CONFIRMED" verdict was a
  **task-design confound** — multi-token words (case variants, leading-space tokens,
  multi-token spans) gave the reconciliation no clean target. Single-token digits remove
  the confound and the reconciliation becomes visible. The gatsby word-count **failure is
  still real**; its cause is tokenization/representation, not "diffusion does nothing."

## Honest riders

- **Noisy reconciliation.** Position 10 thrashes through garbage before settling; '7'
  flickers `3 → garbage → 3`. The path to the correct answer is not monotone.
- **Correlational within one run.** The count/canvas correspondence is tight, but this is
  one run — a tight correlation, not a proven causal read. No intervention (e.g. forcing
  position 10 to a different digit) was performed to test counterfactually.

## Sudoku / unsloth-finetune hypothesis (operator free-association, 2026-07-15)

Recorded verbatim, not re-edited, per discipline (hypotheses are banked as stated, not
smoothed into house voice):

> "[0-9] are super cheap to produce from KVs, and serving as both the sample set and both
> sides of the tally, the tokens themselves maybe are abundant produced from a KV set
> containing all the necessary numerals."

Framed as a mechanism hypothesis: the unsloth sudoku finetune may work partly *because* a
small, closed, single-token vocabulary `[0-9]` is exactly the regime where the diffusion's
evidence-reconciliation (demonstrated in this run) is cleanest — no multi-token confound,
and the KV set can hold the entire symbol alphabet abundantly. Untested; a hypothesis to
carry into the KV-separation follow-up, not a conclusion.

**Cross-refs:** `shanevcantwell/ComfyUI-DiffusionGemma` issue #28 (Sudoku-class flagship
experiments), issue #47 (KV_CACHE seam), ADR-CDG-012 (AR/diffusion KV seam).

## Follow-ups (stable handles)

1. **KV-separation test of the sudoku hypothesis** — inject a deliberately small/closed
   vocabulary KV set (digit-analogue) vs. an over-provisioned one, and compare
   reconciliation cleanliness. Rides the same KV-provenance-separation seam named in the
   gatsby record's follow-up 4 (`#47`, ADR-CDG-012).
2. **Counterfactual intervention** — force an incorrect digit into position 10 mid-run and
   check whether the tally's dependent cell (`4`) tracks the forced value, to move the
   correlational read toward causal.
