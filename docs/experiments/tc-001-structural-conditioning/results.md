# TC-001 Results: Structural conditioning vs. basin re-assertion

**Status**: complete (2026-08-09)
**Protocol**: [`protocol.md`](./protocol.md) (registered; pre-observation)
**Tracking**: #286
**Raw data**: `design-docs/experiments/ComfyUI-DiffusionGemma/tc-001-structural-conditioning/` (private annex per ADR-CDG-022 — run logs, blinded texts, blind judgments, key, `scored.json`)
**Engine identity**: pack `4c99b68` + ComfyUI core `7cf4e783` (deployed `/srv/dev/ComfyUI`); harness from main `efb5de2`; bf16 (`quant="none"`), EB defaults, `gen_length=1024`, seeds 101–110, sequential runs.

## Verdict

**H1 supported.** Median paired compliance gap (B − A) = **2.0** (rule: ≥ 2); exact one-sided Wilcoxon signed-rank **p = 0.00098** (W = 55, n = 10, no zero differences, all 10 pairs positive; scipy absent — full 2^10 permutation computed). The basin did not re-assert at flash length: a committed structural skeleton in the prefix shifted every pair toward the anti-default targets. **No parroting**: zero literal skeleton echoes across all 20 texts; the pre-registered rephrase iteration was never needed.

## Primary — per pair (compliance 0–4 at B-targets)

| premise | A | B | gap (B−A) | F3 NO-ENDING (A / B) |
|---|---|---|---|---|
| 1 | 2 | 3 | +1 | yes / yes |
| 2 | 1 | 2 | +1 | yes / no |
| 3 | 1 | 3 | +2 | yes / no |
| 4 | 2 | 3 | +1 | no / yes |
| 5 | 0 | 2 | +2 | no / no |
| 6 | 1 | 3 | +2 | no / yes |
| 7 | 1 | 4 | +3 | no / no |
| 8 | 1 | 3 | +2 | yes / no |
| 9 | 0 | 3 | +3 | no / no |
| 10 | 3 | 4 | +1 | no / no |

## Primary — per feature (at-target /10)

| Feature | A | B | Pre-registered rule outcome |
|---|---|---|---|
| F1 theme unstated | 8 | 9 | uninformative axis this run (see confounds) |
| F2 non-chronological | 1 | 9 | **clear pass** (B ≥ 8, A ≤ 2) |
| F3 external-event resolution | 1 | 7 | partial; **prediction "F3 hardest" falsified** |
| F4 ≥3 real entities | 2 | 5 | hardest observed; **prediction "F4 easiest" falsified**; no formal basin lock (B > 4), but nearest to one |

## Confounds (named)

- **Ceiling truncation (pre-registered handling, banked on #286 before judging).** 8/10 A runs saturated the 1020-token ceiling. F3 NO-ENDING calls are flagged above; note NO-ENDING ≠ truncation — judges also called NO-ENDING on two turn-closed B stories whose endings resolve nothing.
- **F1's A-side is plausibly truncation-inflated.** A was already 8/10 at target — diverging from the StoryScope marginal (~77% theme announced). Theme statements concentrate in codas; a story cut before its coda cannot state its moral. The H1 verdict does not lean on F1 (A = 8, B = 9).
- **`turn_closed` and narrative closure are distinct measures.** Blind judges' completeness calls disagreed with the mechanical flag in 5/20 cases (3 closed-B judged truncated; 2 saturated-A judged complete). Both readings are banked.

## Secondary (exploratory, not gated)

- **Turn-closure asymmetry (unpredicted).** B closed its turn 9/10 (654–953 tokens); A closed 2/10 (saturated otherwise). The skeleton appears to license an ending. Candidate axis for a successor experiment.
- **Coherence collapse: 1/10 B** (premise 6, `tc001_p06_B_s106`): degeneration into token repetition + multilingual noise. Conditioning-cost rate at this scale: 1/10.
- **Word counts**: A mean 764.9 (right-censored by ceiling), B mean 628.3.
- Canvas-level commit-timing analysis (protocol's exploratory secondary) not run this session; traces are banked in the annex for later eyeballing.

## Outcome routing (per protocol table)

Nearest row: **"H1 supported, F-subset only" → rank axes; the locked set is the interesting object.** Observed ranking (B at-target): F1 9 ≈ F2 9 > F3 7 > F4 5 — with F1 discounted as uninformative. F4 (grounded real-world reference) is the axis most resistant to prefix conditioning. TC-002 (length scaling) remains conditional per the table. Design note for any successor: give the control condition headroom above the token ceiling (A saturated 8/10 at `gen_length=1024`), or the control's own endings are unmeasurable.
