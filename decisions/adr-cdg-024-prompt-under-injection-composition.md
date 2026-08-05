# ADR-CDG-024 — Prompt-under-injection composition: independent encoder context (KV) + templated denoiser turn

**Status**: Accepted
**Date**: 2026-08-05
**Related**: ADR-CDG-012 (MITM seam / `KV_CACHE` — this ADR amends §D.1 IN-2's
open-question resolution, does not reopen the rest), ADR-CDG-001 (native
socket types — `EMIT-CANONICAL / PARSE-AT-THE-DOOR`, the ground this ADR's
supersession stands on), Issue #245 (design mandate — primary source), Issue
#248 (interim exclusivity invariant, merged
https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/253 at
`50ea909` — **superseded by this ADR**, see below), Issue #255 (shipped-defect
record this ADR is the design gate for), Issue #254 (commit/de-commit
semantics under per-step recompute — cited for evidence-chain context, not
amended here), Issue #62 (Phase 4 ledger, `ac3c832` / PR #242).

## Ratification

**Verdict: PASS** — independent Opus design-gate review (a reviewer that did
not author this ADR), 2026-08-05, PR
https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/256. Ratified per
this repo's waterfall process (design completes and is gate-ratified before
implementation opens); this note flips the Status to Accepted and records each
resolved gate check as a decision.

Resolved gate checks:

1. **Doctrine conformance — CONFIRMED.** ARCHITECTURE.md rule 5 (payloads mean
   what they say) and rule 7 (declarative ingress) are read correctly. The
   supersession argument is doctrinally sound rather than a waiver: once
   `prompt` is tokenized, chat-templated, and prefilled onto the cache (§1),
   "connected but ignored" no longer describes the pair, so the
   `EMIT-CANONICAL / PARSE-AT-THE-DOOR` (ADR-CDG-001) ground for #248's
   rejection is genuinely removed. Forward-pass framing is honoured — §1 gives
   explicit token-sequence layouts (K/V attended by every canvas position), not
   a chat-level "add a directive" framing. The greenfield anticipated-failure
   rule is satisfied: §4 names four failure modes, each with a "Prevented by"
   clause. Verified against `dgemma/ingress.py:238` (`reject_prompt_and_kv_cache`),
   `dgemma/loop.py:131,238-240,258-274` (the with-cache drive body and block>0
   re-encode the prefill generalizes), and `dgemma/loop.py:736-744` (the
   `prompt_kwargs` construction §4 reuses).

2. **Operator requirement — CONFIRMED.** The design cites the banked #245
   evidence chain in full (two independent KV-injection stalls plus the
   discriminating AIO control) and pins the mechanism (conditioning without an
   attractor). §5 defines a falsifiable H0 with a concrete acceptance test — one
   of the two banked stall traces must converge under the composed drive body,
   with a predicted convergence range and an explicit falsification condition —
   satisfying the "KV path must converge / stall prompt converging" requirement
   as a live, non-mocked evaluation against the same real-weights fixture. The
   ADR honestly flags (§Negative Consequences) that this run is not yet
   performed and carries the obligation forward to the implementation PR.

3. **Internal consistency — CONFIRMED.** Decision (Option B) matches the
   Alternatives section, §1's token layout, and #245's title/use case. No
   contradictions found across Decision, Rationale, Failure modes, and
   Supersession sections.

4. **Cross-artifact seams — CONFIRMED.** The mandatory #248 supersession is
   explicit and named (§3 and §Supersession Relationships): the chosen
   disposition is **re-permit + supersede** — #248's acceptance criterion
   ("KV_CACHE connected AND non-empty prompt → ingress rejects") is reversed by
   name, and `reject_prompt_and_kv_cache`'s removal/tombstone is scoped in §2
   and Open Questions. The ADR-CDG-012 §D.1 IN-2 amendment relationship is
   explicit (amends the open-question resolution only, does not reopen the
   rest). #254 is correctly cited for evidence-chain context and explicitly not
   amended.

Advisory (non-blocking, no design change): §4's citation of
`dgemma/loop.py:274` as the point `decoder_start` derives from
`past_key_values.get_seq_length()` is a near-miss — `decoder_start` is computed
at loop.py:269 from `cached_len + canvas_idx * canvas_length`, and
`get_seq_length()` is read at loop.py:273 for the attention mask. The design
intent (derive the splice offset from the cache's own advanced length rather
than a hand-computed constant) is correct and matches the block>0 re-encode
pattern; the implementer should bind the prefill splice to the cache's actual
post-prefill `get_seq_length()`. Recorded for the implementation PR, not a
ratification blocker.

---

## Context

ADR-CDG-012 §D.1 IN-2's ratified interim state (PR #242 gate, 2026-08-04) is
"prompt-under-injection = prompt ignored, drive off cache": when
`DGemmaDenoise` receives a `kv_cache=`, `_run_pipeline_with_injected_cache`
(`dgemma/loop.py:131-336`) drives the decoder directly off `kv_cache.cache`
with a fresh `torch.randint` canvas and never tokenizes `prompt`
(`dgemma/loop.py:238-240`). That gate ruled the omission byte-identical to
the proven `scratch/q2-skeleton-2026-08-04` smoke and explicitly deferred the
composition question to this issue (#245) rather than answering it.

`DGemmaEncode`'s mint/advance body (`dgemma/kv_cache.py:245`, called from
`surfaces/comfyui/encode.py:96`) is `tokenizer.encode(text)` — a bare
tokenizer call with no role markers, no `<start_of_turn>model\n`
generation-prompt suffix, no `<|think|>` scaffold. The no-cache path, by
contrast, drives through `pipeline.__call__`, which applies
`processor.apply_chat_template(..., add_generation_prompt=True)` to `prompt`
(cited at `dgemma/loop.py:381-383`) — optionally with a `<|think|>`
system-turn injection (`dgemma/loop.py:736-742`, pinned byte-for-byte against
the real tokenizer by `tests/test_chat_template_thinking.py`). A cache minted
by `DGemmaEncode` therefore carries **raw context tokens with no model-turn
scaffold** — there is no `<start_of_turn>model\n` position for the field to
converge onto.

**Evidence (banked on issue #245, 2026-08-04 — full chain, cited here rather
than reproduced in prose elsewhere):** two independent KV-injection stalls —
a 48-step trace flat at the scheduler's structural 1/256 floor for 34
consecutive steps (~62% never committed), and a 64-step trace ending 4/256
committed, `converged: false`, 252–256/256 positions re-noised every step.
A discriminating AIO control (same prompt family, seed, `entropy_bound=0.05`,
no cache) converged cleanly: 6/256 at step 0, >50% by step 3, 255/256 by step
15, early-stop at step 16. This excludes both a broad Phase-4 drive-body
regression and the entropy-bound setting — the AIO path used the same bound
and converged. The stall is specific to the injection path; mechanism
pinned: conditioning without an attractor. A raw `tokenizer.encode` context
gives the decoder a K/V history with no `<start_of_turn>model\n`-shaped
completion site, so the posterior field over the canvas never crystallizes.
Full trace data and mechanism derivation are on issue #245's thread (cited,
not reproduced here).

**Consequence named by the operator (issue #245, reframe comment,
2026-08-04):** this design is the gate for issue #255, the shipped-defect
record — composition is the *enabling fix* for the KV-injection path to
produce convergent output at all, not an enhancement on top of working
behavior.

**Interim invariant in force:** issue #248 (merged `50ea909`, PR #253) rejects
`prompt` and `kv_cache` supplied together at `run_diffusion`'s ingress
(`dgemma/ingress.py:238-268`, `reject_prompt_and_kv_cache`), on the ground
that a connected-but-ignored `prompt` is exactly the trust-and-degrade
ARCHITECTURE.md rules 5/7 forbid. #248 named itself explicitly interim,
conditional on #245: "The design artifact here should land on top of that
recorded state" and "#245's ADR MUST explicitly supersede this invariant if
it composes prompt+cache."

## Decision

**Adopt Option B: independent encoder context (KV) + chat-templated denoiser
turn, composed at drive time.** `kv_cache` and `prompt` become jointly
permitted. `DGemmaEncode` stays template-free (unchanged — the raw-encode
path is issue #47's cache-perturbation instrument and stays available
verbatim, per #245's explicit non-negotiable constraint). `DGemmaDenoise.prompt`,
when supplied alongside a `kv_cache`, is treated as **the current
model-turn** and is chat-templated exactly as the no-cache path templates it
today — `processor.apply_chat_template` (with the same `thinking=`-gated
`<|think|>` injection), then prefilled onto the injected cache before the
decode loop begins.

### 1. Token-sequence layout (forward-pass framing — exactly what conditions emission)

Today's no-cache path conditions each denoise step on:

```
[ <chat-templated prompt tokens>, <generation-prompt suffix "<start_of_turn>model\n"> ]
                                   └── K/V of these tokens, attended to by every canvas position
```

Today's with-cache path (IN-2 as amended, ADR-CDG-012) conditions on:

```
[ <kv_cache's raw context tokens, no scaffold> ]
                                   └── K/V attended to; no turn-boundary token exists
```

**This ADR's composed layout** — the cache's context is prefilled with a
templated denoiser turn before the decode loop starts:

```
[ <kv_cache's raw context tokens (unchanged, from DGemmaEncode)>,
  <chat-templated "prompt" tokens: role markers + prompt text>,
  <generation-prompt suffix "<start_of_turn>model\n"> ]
                                   └── K/V of ALL of the above, attended to by every canvas position
```

The prefill is an **encoder pass over the templated turn**, appended to
`kv_cache.cache` in place of the skipped "no `prompt` re-encode" behavior
IN-2 names today — mechanically the same shape as `_run_pipeline_with_
injected_cache`'s existing block-boundary re-encode (`dgemma/loop.py:258-267`,
which already calls `model.model.encoder(input_ids=..., past_key_values=...,
position_ids=...)` against a cache whose starting length is the injected
cache's), generalized to run once, before block 0, for the templated
turn instead of a committed canvas block. `decoder_position_ids` for block 0
then start at `cached_len + len(templated_turn_ids)`, not `cached_len` — the
canvas positions shift by exactly the templated-turn length, mirroring how
`decoder_start` already shifts by `canvas_length` per block today
(`dgemma/loop.py:269`).

`prompt=""`/`None` alongside `kv_cache` degrades to **today's exact IN-2
behavior** (skip-first-encode, no prefill, decode straight off the injected
cache) — additive-optional, not a breaking change to the pure-injection
shape #47's instrument depends on. Composition only activates when both
inputs are non-empty.

### 2. Ingress changes

- `dgemma.ingress.reject_prompt_and_kv_cache` is **removed** (or reduced to a
  no-op retained only as a documented tombstone — implementer's choice,
  named as an open question below) — the pair it rejected is now the
  supported composition shape.
- `run_diffusion`'s `KV_CACHE` ingress validator (`validate_kv_cache_ingress`,
  V1–V6, ADR-CDG-012 §D.3) is unchanged — it validates the cache's own
  well-formedness, orthogonal to whether `prompt` is also supplied.
- No new ingress validator is required for the composed pair itself: an
  empty `prompt` alongside `kv_cache` was already legal (the intended
  injection-only shape); a non-empty `prompt` alongside `kv_cache` moves from
  rejected to accepted-and-composed. This is a **strict widening** of what
  IN-2 accepts, not a new door.
- `DGemmaDenoise`'s `prompt`/`kv_cache` widget tooltips (`surfaces/comfyui/
  denoise.py:147-157,243-253`) are rewritten: the exclusivity language
  ("exactly one of prompt or kv_cache is permitted", "a non-empty prompt
  alongside a connected kv_cache is rejected at ingress") is replaced with
  composition language — `prompt` supplied alongside `kv_cache` is the
  current-turn text, chat-templated and prefilled onto the cache before
  decoding.

### 3. Issue #248 supersession — explicit, named

**This ADR supersedes issue #248's exclusivity invariant by name.** #248's
rule (`reject_prompt_and_kv_cache`, merged `50ea909`) was correct and
necessary as an interim measure — the with-cache drive body at the time
genuinely ignored a connected `prompt`, and rejecting a connected-but-ignored
input at the door is the right call under `EMIT-CANONICAL /
PARSE-AT-THE-DOOR` (ADR-CDG-001). That ground is removed by this decision:
once `prompt` is tokenized, templated, and prefilled onto the cache (§1),
"connected but ignored" no longer describes the pair — `prompt` conditions
the run exactly as much as `kv_cache` does. #248's acceptance criterion "KV_CACHE
connected AND non-empty prompt supplied → ingress rejects" is **reversed**:
that pair becomes the primary composition path this ADR exists to enable.
`decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md`'s §D.1 IN-2
write-in (2026-08-04) is amended by reference: this ADR is IN-2's next
open-question resolution, superseding the "prompt ignored" interim wording
with "prompt, when non-empty, is the templated denoiser turn."

### 4. Anticipated failure modes (greenfield rule — every invariant names the failure it prevents)

- **Position-id drift at the splice.** Prefill and decode-loop
  `decoder_position_ids` disagreeing on where the templated turn ends and
  the canvas begins attends canvas positions under the wrong RoPE offset —
  silent incoherence, not a crash. **Prevented by:** deriving `decoder_start`
  from the prefill's own `past_key_values.get_seq_length()` (as the existing
  block>0 re-encode already does, `dgemma/loop.py:274`), not a
  separately hand-computed length.
- **Double-templating.** A future caller pre-templating `prompt` before
  `DGemmaDenoise` (copying `DGemmaEncode`'s raw-encode convention) would have
  the drive body apply `apply_chat_template` twice, corrupting turn
  structure. **Prevented by:** `DGemmaDenoise.prompt` stays contractually raw
  text; templating happens exactly once, inside the drive body.
- **Silent divergence from the no-cache template shape.** A composed-path
  `apply_chat_template` call drifting from the no-cache call (different
  `thinking=` handling or `add_generation_prompt` default) would make the
  two paths non-comparable for the same semantic operation. **Prevented
  by:** reusing the exact `prompt_kwargs` construction already built for the
  no-cache path (`dgemma/loop.py:736-744`) — one template-construction site,
  two consumers.
- **`DGemmaEncode`'s raw-encode instrument silently regressed.** Collapsing
  `DGemmaEncode` into template-aware encoding "for consistency" would lose
  issue #47's raw substrate. **Prevented by:** this ADR rules Option A out
  (§Alternatives) and reasserts the #245 constraint that `DGemmaEncode` stays
  template-free.

### 5. Falsifiable acceptance evidence

**H0:** the with-cache non-convergence is caused (at least in significant
part) by the missing model-turn scaffold. **Falsification test:** re-run one
of the two banked stall traces (issue #245 — the "count 23 randomly selected
numerals" 48-step trace or the 0.5.1_test 64-step trace) under this ADR's
composed drive body, same prompt, knobs, and seed. **Predicted result under
H0:** the composed run converges within a step count comparable to the AIO
control's healthy range (early-stop well inside the 48-step budget, not a
full-budget non-convergent run) — a repeat of the flat 1/256 structural floor
or the 252–256/256 full-canvas re-noise pattern would falsify H0 and reopen
the mechanism question. This is a live, non-mocked evaluation against the
same real-weights fixture the evidence chain already used — no new
infrastructure is required, only the composed drive body to run it against.

## Rationale

### Positive Consequences
- The KV-injection path becomes usable for its stated use case (operator,
  issue #245: "An encoder prompt for KVs independent of the denoiser prompt
  is a use case") — the composed shape is exactly what #245's title names:
  independent encoder context, denoiser turn.
- `DGemmaEncode`'s raw-encode contract is preserved untouched — issue #47's
  instrument keeps working exactly as before; this ADR is additive to the
  node pair's capability, not a replacement of an existing one.
- The templating logic is not duplicated — the composed path reuses the
  no-cache path's existing `prompt_kwargs`/`apply_chat_template` construction,
  so the two paths cannot silently diverge in turn-structure semantics
  (named as a failure mode above, closed by this reuse).
- #248's exclusivity rule, which the operator directive framed as interim
  pending this issue, is closed out on schedule rather than left as
  permanent scar tissue on the ingress door.

### Negative Consequences
- The with-cache drive body (`_run_pipeline_with_injected_cache`) grows a new
  branch (prefill-if-prompt-non-empty) — more surface than the current
  unconditional skip-first-encode, though the branch collapses to today's
  exact behavior when `prompt` is empty.
- The falsification test (§5) is **not yet run** — this ADR is Proposed on
  the strength of the mechanism evidence and the design's internal
  consistency, not yet on a confirmed convergence result. The design-gate
  review and implementation PR carry that obligation forward.
- Composition adds one more shape to the IN-2 door's contract surface
  (cache-only, prompt-only, cache+prompt-composed) that `DGemmaDenoise`'s
  docstring and tooltips must keep honest — a documentation-maintenance cost
  named here rather than left implicit.

## Alternatives Considered

### Option A: Template the encoder — `DGemmaEncode` applies the chat template + generation-prompt suffix

`DGemmaEncode` grows a template mode: role/turn wrapping (and optionally
`thinking`) applied to `text` before `tokenizer.encode`, so a minted cache is
itself a scaffolded turn-prefix. Simplest to implement — no drive-body
splice, no dual-templating-site risk.

**What is lost:** the "independent encoder context" concept collapses. Issue
#245's operator use case is explicitly *"An encoder prompt for KVs
independent of the denoiser prompt"* — a donor context that is NOT itself a
denoiser turn (e.g., a document, a prior exchange, arbitrary conditioning
text). Templating the encoder forces every minted cache to look like a
chat turn, which is wrong for that use case and directly regresses issue
#47's raw-context cache-perturbation instrument (#245 names this
constraint explicitly: "the raw template-free encode path must remain
available"). Rejected: violates a named, non-negotiable requirement.

### Option B: Independent encoder context + templated denoiser turn (chosen)

See Decision above. Composes `kv_cache` (raw, from `DGemmaEncode`,
unchanged) with `prompt` (templated, at drive time, in
`_run_pipeline_with_injected_cache`) — matches #245's title and use case
verbatim, and is the shape the evidence chain's H0 predicts will converge.

**Cons weighed and accepted:** more drive-body branching than Option A;
requires the position-id/splice care named in §Failure modes; requires the
falsification run (§5) to confirm, not just design-argue, the fix.

### Option C: Hold — KV path stays experimental, exclusivity stands

Leave #248's rejection in force; document the KV-injection path as
non-convergent-by-design pending further research, closing neither #245 nor
#255 for now.

**Why rejected:** the evidence chain (issue #245) already pins the mechanism
with a live discriminating control — this is not an open research question
requiring more investigation before a design can be attempted, it is a
scoping decision the operator has already made ("this design is now the gate
for shipped-defect #255... composition is the enabling fix, not an
enhancement"). Holding indefinitely leaves a shipped, advertised feature
(`KV_CACHE` injection, ADR-CDG-012, landed `ac3c832`) permanently
non-functional for its own motivating use case with a known, designable
fix sitting unimplemented. Rejected as inconsistent with the operator's
explicit reframing of #255 as the gated defect this design closes.

## Open Questions

- [ ] **`reject_prompt_and_kv_cache` disposition: delete vs. tombstone.**
      Should the function be removed outright from `dgemma/ingress.py`, or
      retained as a dead, documented no-op referencing this ADR (so a reader
      grepping issue #248 lands on live code explaining the reversal)?
      **Resolution:** implementer's call at implementation time; either
      satisfies this ADR's supersession as long as the pair is accepted at
      ingress and the code (if retained) is unreachable/no-op, not
      partially-enforcing.
- [ ] **`thinking=` interaction with the composed turn.** The no-cache path's
      `<|think|>` system-turn injection (`dgemma/loop.py:736-742`) is a
      `messages=[...]` construction; this ADR's §1 layout assumes the same
      construction is reused for the composed prefill. Confirm during
      implementation that `thinking=True` alongside `kv_cache=`+`prompt=`
      composes without a third divergent code path. **Resolution trigger:**
      implementation PR; covered by the existing
      `tests/test_chat_template_thinking.py` pin if the same
      `prompt_kwargs` construction is genuinely reused (§1's design intent).
- [ ] **OUT-1 (advanced-cache output) interaction.** ADR-CDG-012 §D.2 OUT-1
      defers emitting the advanced cache from the with-cache drive body.
      This ADR does not change that deferral, but the prefilled cache
      (context + templated turn) is a different object than the pure
      injected cache OUT-1 would eventually emit. **Resolution:** deferred
      to whichever future work lands OUT-1 — not decided here, named so that
      work does not silently assume the pre-composition cache shape.

**Resolution plan:** the disposition and `thinking=` questions resolve during
the implementation PR that lands this ADR's decision; OUT-1 stays deferred
to its own future work, unaffected by this ADR either way.

## Supersession Relationships

**Supersedes:** Issue #248's exclusivity invariant (`reject_prompt_and_kv_cache`,
merged `50ea909`, PR #253) — see §Decision/3 for the explicit disposition.
Does not supersede ADR-CDG-012 itself; amends its §D.1 IN-2 open-question
resolution only, per that ADR's own "carried forward... see the implementing
PR's deviations section" framing, which named this issue (#245) as the
question's actual resolution point.
**Superseded by:** TBD.

## References

- Issue #245 (design mandate, evidence chain) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/245
- Issue #248 (interim exclusivity, merged `50ea909`) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/248
- Issue #255 (shipped-defect record, this ADR's gate) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/255
- Issue #254 (commit/de-commit semantics, evidence provenance) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/254
- ADR-CDG-012 (§D.1 IN-2 is the channel amended here) — `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md`
- ADR-CDG-001 (`EMIT-CANONICAL / PARSE-AT-THE-DOOR` — ground for both #248's
  original rejection and this ADR's supersession)
- PR #242 (landed Phase 4 drive body, `ac3c832`, the "prompt ignored" interim
  state this ADR amends) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/242
- PR #253 (landed #248, `50ea909`, superseded here) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/253
