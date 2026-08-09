# TC-001: Structural conditioning vs. basin re-assertion

**Status**: registered (2026-08-09, Opus design-gate PASS)
**Date**: 2026-08-09
**Tracking**: #286
**Runs on**: ComfyUI-DiffusionGemma, bf16 path, EB defaults (γ=0.1, temp 0.8→0.4, max 48 steps). Sampler untouched by design — the intervention lives entirely in the prefix.

---

## Hypothesis

**H1**: Committing a structural skeleton in the prefix shifts discourse-level
narrative features of generated flash fiction away from the AI marginals
(StoryScope: theme announced ~77%, linear timeline ~80%, no fourth wall,
few named real entities) toward assigned anti-default targets.

**H0 (the null to try to keep alive)**: The basin re-asserts. Conditioned and
unconditioned generations show no meaningful difference in structural
compliance — the skeleton is ignored, or parroted as summary rather than
enacted.

## Design

Paired: 10 premises × 2 conditions, same seed per pair. 20 generations total,
~800 words each (3–4 canvases).

- **Condition A (control)**: premise + "Write a short story, about 800 words."
- **Condition B (conditioned)**: identical, plus the skeleton block below,
  verbatim, before the premise.

### Skeleton block (Condition B)

```
STRUCTURE — commit to these before writing; enact them, never mention them:
1. The theme is never stated. No narrator moral, no character voicing the
   lesson. If a sentence explains what the story means, delete it.
2. Open mid-crisis. Exactly one flashback, entered without announcement.
3. The ending is caused by an external event. The protagonist's plan fails.
4. Name at least three real things: a real book, a real place, a real brand.
```

Each directive targets the anti-default pole of a StoryScope-measured axis.
That is the point: targets sit where the AI marginals say the model won't go
unconditioned.

### Premises (reskinned from distinct human openings; use as-is)

1. A young woman sits alone on a New York subway; 365 days without a date.
2. A night-shift radiologist finds his own name on a scan dated next month.
3. Two brothers split their late father's beehives; one hive won't accept either.
4. A ferry cook in the Aegean discovers the ship has been sailing in circles for a day.
5. A retired forger is asked to authenticate a painting she faked forty years ago.
6. A rural lineman keeps restoring power to a house the county says is vacant.
7. A translator realizes the dictator's speech she's rendering live contains a message for her.
8. A minor-league umpire calls his last game the day his eyesight officially fails.
9. A landlady inherits a tenant's ashes and his unpaid rent in the same envelope.
10. A glacier guide finds last season's missing client's camera, still recording.

## Measurements

### Primary — text level (gated)

Four binary features per story, judged with quoted-evidence requirement
(judge must cite the span or the absence that decides each call):

| # | Feature | Target (B) | AI default (A expected) |
|---|---------|-----------|------------------------|
| F1 | Theme stated explicitly anywhere | NO | YES (~77%) |
| F2 | Non-chronological structure present | YES | NO (~80% linear) |
| F3 | Resolution caused by external event / plan fails | YES | NO (protagonist-choice resolution) |
| F4 | ≥3 real named entities | YES | NO (~half human rate) |

Judge: a second model with per-feature yes/no + evidence span, or manual
(~2 min/story; binary structural facts are low-inference). Judge-shares-basin
risk is low for binary facts but the evidence-span requirement is mandatory.

Compliance score per story = count of features at target (0–4).

### Secondary — canvas level (exploratory, not gated)

From CANVAS_TRACE per run:
- Commit-step timing of the flashback-region tokens in B: does structure
  freeze early or late in the anneal?
- Commit-front topology difference A vs B (heatmap eyeball first; metric later
  only if the eyeball shows something).

### Validity gate

`turn_closed` / `answer_tokens` honesty readout must pass. Incoherent output
under B is not discarded — it is recorded as a conditioning-cost result.

## Decision rule (pre-registered)

- **H1 supported**: median paired compliance gap (B − A) ≥ 2 features,
  Wilcoxon signed-rank p < 0.05, n = 10 pairs. Report per-feature exact
  counts regardless.
- **Clear per-feature pass**: B ≥ 8/10 at target while A ≤ 2/10.
- **Parroting failure** (distinct from H0): skeleton restated as summary but
  not enacted — e.g., narrator *says* "there is no moral here" (violating F1
  in spirit), or announces the flashback. Counts against compliance. One
  pre-registered rephrase iteration of the skeleton block is permitted if
  parroting dominates; a second is not — at that point parroting is the
  finding.
- **Per-feature basin lock**: any feature where B ≤ 4/10 despite explicit
  instruction is reported as prompt-inaccessible. Prediction on record:
  F3 (external-event resolution) is the hardest; F4 the easiest.

## Outcomes → next actions

| Result | Meaning | Next |
|--------|---------|------|
| H1 supported, all features | Prefix conditioning defeats basin at flash length | TC-002: length scaling (does compliance decay per canvas?) |
| H1 supported, F-subset only | Some axes basin-locked below prompt level | Rank axes; locked set is the interesting object |
| Parroting dominates after rephrase | Instruction-following ≠ structural generation | Strongest available evidence for the video's side; write it up as such |
| H0 holds | Basin re-asserts even with committed prefix | The conditioning thesis from the trendslop thread is falsified at this scale; stop recommending it |
| Coherence collapse in B | Conditioning has measurable cost | Quantify: compliance vs. validity-gate pass rate |

## Cost

20 generations × ~800 words on the bf16 rig + judging: one afternoon.
No new code beyond a judge prompt; harness is examples/ + trace node as-is.

## Non-goals

No sampler modification, no RHT, no EB parameter sweeps, no GGUF. Any urge to
add a knob mid-run is out of scope and goes to the loose-ends ledger instead.
