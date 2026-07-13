# ADR-CDG-012 — MITM the AR/diffusion seam: a `KV_CACHE` socket + `DGemmaEncode`/`DGemmaDenoise` node pair

**Status**: proposed
**Date**: 2026-07-13
**Related**: ADR-CDG-001 (native socket types — this ADR's fingerprint/ingress
rule is a direct instance of "payloads mean what they say"), ADR-CDG-005
(`CANVAS_STATE` resumable save-state — KV's exclusion there was economy, not
impossibility; this ADR does not reopen it), ADR-CDG-006 (step-window resume —
this ADR extends its ownership model one axis, the cache), ADR-CDG-008
(MCP-center topology — this decision lands core-side of the surface seam, so
every surface inherits it), ADR-CDG-010 / ADR-CDG-011 (composite ordering and
declarative-ingress discipline — the node pair inherits both), Issue #47
(primary source — grounding report, payload-richness refinement, serialization
correction), Issue #46 / Issue #40 (research motivation — backward-asymmetry
discriminators, AR-dominance, fossil waves)

---

## Context

Issue #47's grounding pass (2026-07-13, read against installed diffusers
0.39.0 / transformers 5.13.0) found that DiffusionGemma's encoder and decoder
are already separate organs, not one undifferentiated "model":

- **Encoder = the AR hemisphere.** Causal; the sole cache writer
  (`modeling_diffusion_gemma.py:350-351`, the only `past_key_values.update()`
  call path); runs per block on uncached tokens only, sliced by
  `cached_len = past_key_values.get_seq_length()`
  (`pipeline_diffusion_gemma.py:324,326-333`).
- **Decoder = the diffusion hemisphere.** Bidirectional over the canvas;
  computes K/V from canvas tokens only; never updates the cache
  (`modeling_diffusion_gemma.py:422-449`; forward returns no cache, `:1327`).
- **Exactly two inter-hemisphere crossings, both at block granularity:**
  cache-read (encoder → decoder) and committed-canvas re-encode (decoder →
  encoder, `pipeline_diffusion_gemma.py:429` concat then next-block encode).

Issue #46 and Issue #40's research threads (backward-asymmetry discriminators,
the "waves" fossil, KV-mediated garbling) independently converge on the same
seam as the next experimental lever: the cache is DiffusionGemma's sole
cross-block memory channel, so manipulating what crosses it is the only way to
test hypotheses about AR-dominance and fossil permanence without touching the
model.

`past_key_values` is internal to the pipeline's `__call__` today — not a
parameter, not in the output (#47 grounding). The model-level encoder and
`DiffusionGemmaForBlockDiffusion.forward` are directly callable with an
external cache today, so a prefill-from-sequence entry point is buildable
outside the pipeline (#47 grounding, `model.model.encoder(input_ids=…,
past_key_values=cache, position_ids=…)`).

## Decision

**The pack interposes man-in-the-middle on the two block-boundary crossings
between DiffusionGemma's AR hemisphere (encoder) and its diffusion hemisphere
(decoder).** Mechanism: a `KV_CACHE` native socket type plus a
`DGemmaEncode` / `DGemmaDenoise` node pair, with per-layer cache surgery as
the tier-2 experimental surface.

### 1. Governing invariant: a cache crosses any boundary with its provenance record

Generalized from #47's serialization-correction comment: **a cache crosses
any boundary — node-to-node, disk, or process — carrying its provenance
record.** The record is:

- the **minting sequence** (the token ids the encoder consumed to produce
  this cache), when one exists — tier 1, rebuild-via-prefill remains the
  cheap path (ADR-CDG-005 unaffected); or
- **minting sequence(s) + edit-script**, when the cache has been perturbed
  (spliced, ablated, scaled) and no single prefill reproduces it — tier 2.

Disk is a data-plane crossing like any other (`CONSERVE-ACROSS-THE-DATA-BOUNDARY`):
carry the record from the mint, assert the fingerprint at re-entry. This is
the same discipline as ADR-CDG-001's fingerprint/ingress rule, applied to a
live cache object instead of a tensor payload.

### 2. `KV_CACHE` payload schema

Per #47's payload-richness refinement (every structural blocker the grounding
report found converts to schema field or ingress rejection, never a silent
hazard):

- the **live cache object** (`DynamicCache`, per-layer `DynamicLayer` /
  `DynamicSlidingWindowLayer`);
- the **provenance record** (§1);
- **per-layer `cumulative_length`** — the grounding report's ranked #1
  blocker (`cache_utils.py:254`, mask offsets computed from it at `:270`);
  carried as a field `DGemmaEncode` fills, so no consumer hand-tracks it;
- a **geometry fingerprint**: layer-types pattern (5 full-attention / 25
  sliding, at indices `(i+1)%6`), sliding-window size, batch size, dtype,
  per-layer-type RoPE params (full: proportional, θ=1e6, partial_rotary
  0.25; sliding: default, θ=1e4).

**Ingress rule:** the consuming node validates the geometry fingerprint
against the loaded model at ingress and **fails on mismatch** — no silent
mis-masking. This is `EMIT-CANONICAL / PARSE-AT-THE-DOOR` (ADR-CDG-001)
applied to `KV_CACHE`; the socket type is where the honesty lives, not a
limit on what crosses (#47, payload-richness comment).

### 3. Ownership semantics: advance-returns-new-payload is the default

The cache is mutable and the encoder advances it. Fan-out to two consumers
plus one advance is a cross-consumer contamination hazard — the
`STATELESS-CORE` failure shape in miniature (ADR-CDG-008 Correction 1).
**Default: advance-returns-new-payload** — functional, matching
`CANVAS_STATE`'s existing shape (ADR-CDG-005/006) — the encoder/denoise
advance emits a new `KV_CACHE` payload rather than mutating the input in
place.

**Rejected alternatives** (recorded per #47's two-points-the-richness-does-
not-dissolve comment):

- **Copy-on-advance** — every consumer defensively deep-copies before
  advancing. Rejected as the default: pays a real per-block tensor-copy cost
  on every crossing to guard against a fan-out case that is the exception,
  not the norm; the aliasing-contamination failure this guards against is
  better foreclosed by the functional contract than paid for on every call.
- **Documented single-consumer ownership** (convention, no structural
  guard) — rejected as the sole mechanism: a prose-only "don't fan this out"
  rule is exactly the kind of invariant this repo's greenfield discipline
  requires an enforcement surface for, not a comment. Advance-returns-new-
  payload makes the failure mode structurally harder to hit without paying
  copy-on-every-call cost.

### 4. Node pair

- **`DGemmaEncode`** — sequence (`STRING`/token ids) in, `KV_CACHE` out. A
  near-wrapper over the separately-callable encoder (#47 grounding:
  `model.model.encoder(...)` is directly callable today) — thin per
  ADR-CDG-003's node/engine seam.
- **`DGemmaDenoise`** — canvas + `KV_CACHE` in. The block loop moves to
  engine/node ownership (extending ADR-CDG-006's step-window ownership model
  one axis: the cache, alongside the canvas/schedule-position axis
  ADR-CDG-006 already owns). Skips the first encode when a `KV_CACHE` is
  supplied (mirrors ADR-CDG-006's `start_at_step > 0` requires-a-resume-input
  gate). Optional stop-at-block-boundary, emitting the advanced cache as a
  new `KV_CACHE` payload (§3).
- **Save/load pair** (or torch-serializable `KV_CACHE` fields) for tier-2
  artifacts — belongs in the eventual node set alongside the encode/denoise
  pair (#47 serialization-correction comment). Mechanically: `DynamicCache`
  is per-layer K/V tensors + bookkeeping scalars; `torch.save`/load plus the
  provenance envelope is the mechanism; the fingerprint-at-ingress check
  (§2) already covers deserialization hazards (device placement, dtype,
  geometry) since it was designed for caches of unknown history.

### 5. Tier-2 experimental surface: per-layer cache surgery

The two-tier structure (#47):

- **Tier 1 — with-distribution conditioning.** A cache with an intact
  minting sequence; rebuild-via-prefill remains the cheap reproduction path;
  no conflict with ADR-CDG-005.
- **Tier 2 — against-distribution perturbation.** A cache with no single
  minting sequence — splice, ablation, scaling — reproducible only via its
  edit-script (§1). **This requires serialization** (#47
  serialization-correction comment): the tensors are the primary
  experimental artifact once perturbed, and reproducibility demands
  persisting them, not just their recipe.

The 5-full-attention/25-sliding geometry (#47 grounding) makes **full-attention
ablation** (zero/ablate the 5 full-attention layers' cache entries) a direct
utility function on top of tier-2 surgery — a direct test of whether #40's
fossil waves ride the long-range layers, per #47's node-level framing.

## Rationale

### Positive Consequences

- **A cache-manipulation bench becomes buildable without touching model
  code.** The encoder and decoder's already-separate-organ structure (#47
  grounding) means MITM-ing the seam is additive instrumentation, not a
  model fork.
- **Direct empirical purchase on #40/#46's open questions.** Cache/window
  manipulation is a second discriminator (alongside ADR-CDG-010's pin
  mechanism) for retrieval-limited vs. prior-limited backward-asymmetry
  hypotheses (#46), and full-attention ablation is a direct test of whether
  fossil waves ride the long-range layers (#40, #47).
- **No conflict with ADR-CDG-005.** `CANVAS_STATE`'s KV exclusion was
  economy (routine save-states shouldn't bloat with recomputable tensors),
  not impossibility. `KV_CACHE` is a distinct, deliberate experimental
  payload type for a distinct purpose; tier-1 rebuild-via-prefill remains
  `CANVAS_STATE`'s cheap path unchanged.
- **The fingerprint/ingress rule forecloses silent mis-masking by
  construction**, not by discipline — the same structural move ADR-CDG-001
  and ADR-CDG-010/011 already use for other payloads.

### Negative Consequences

- **Orphan-cache poisoning downstream conclusions.** A `KV_CACHE` payload
  crossing without its provenance record is a lying payload in the
  ADR-CDG-001 sense: a downstream analysis (e.g. a fossil-wave ablation
  study) run against a cache of unknown minting history produces a
  conclusion that looks grounded but isn't. This is the failure §1's
  invariant exists to prevent.
- **Wrong `cumulative_length` silently corrupts masks.** The grounding
  report's ranked #1 blocker (`cache_utils.py:254,270`) — an
  uninitialized or stale per-layer `cumulative_length` does not raise; it
  produces a plausible-looking but wrong attention mask. This is why §2
  makes it a schema field `DGemmaEncode` fills, never a value a consumer
  hand-tracks.
- **Aliasing contamination across graph branches.** Without §3's default,
  a `KV_CACHE` fanned to two consumers plus one advance lets one branch's
  mutation bleed into another's — a `STATELESS-CORE` violation in miniature
  (ADR-CDG-008 Correction 1's failure shape, here on a cache instead of a
  scheduler).
- **Silent geometry mismatch.** A `KV_CACHE` built (or perturbed) against
  one model's layer-type pattern, fed to a differently-configured model,
  produces wrong attention geometry with no crash — the reason §2 makes
  fingerprint validation mandatory at ingress rather than advisory.
- **Perturbed-cache irreproducibility without an edit record.** A tier-2
  cache with no minting sequence and no edit-script is an experimental
  artifact nobody can reproduce or audit — the reason §1 generalizes the
  provenance requirement to cover edit-scripts, not just minting sequences.

## Alternatives Considered

### Option A: Copy-on-advance as the default ownership semantics

**Why rejected:** Guards against the fan-out/aliasing hazard by paying a
tensor-copy cost on every advance, regardless of whether the run ever forks.
Advance-returns-new-payload (the `CANVAS_STATE` precedent) achieves the same
safety without a copy on the common, non-forking path.

### Option B: Documented single-consumer ownership, no structural guard

**Why rejected:** A convention enforced only by a docstring is exactly the
"invariant enforced only by prose" pattern this repo's discipline treats as
one refactor from gone. Advance-returns-new-payload makes the hazard
structurally harder to hit instead of merely discouraged.

### Option C: Treat `KV_CACHE` as reopening ADR-CDG-005's KV exclusion

**Why rejected:** Conflates two different questions. ADR-CDG-005 excluded
KV from the *routine save-state* on economy grounds (recomputable, not worth
the bloat). `KV_CACHE` is a *deliberate experimental payload* for a
different purpose (cache surgery, cross-block memory manipulation) that
ADR-CDG-005 never addressed. Treating this ADR as amending ADR-CDG-005 would
misrepresent both decisions' scope; they compose without conflict, per #47's
serialization-correction comment.

### Option D: Skip serialization; treat tier 2 as in-graph-only

**Why rejected — corrected mid-thread (#47).** Initially framed as "the
`KV_CACHE` socket is in-graph only; persistence reopens ADR-CDG-005
territory." Operator push-back: serialization is not a barrier, and tier 2
*requires* it — a perturbed cache has no minting sequence, so no prefill
reproduces it; the tensors themselves are the experimental artifact.
Mechanically trivial (`torch.save`/load + the provenance envelope), and the
fingerprint-at-ingress check (§2) already covers the deserialization
hazards. Rejecting persistence would make tier-2 experiments irreproducible
by design.

## Open Questions

- [ ] **Untested assumption: the decoder driven with a caller-built cache
      the pipeline didn't create.** Position/mask math should hold per the
      cited source (#47 grounding), but this is unverified against real
      weights. **Resolution trigger:** the designated de-risk experiment — a
      first real-weights smoke test — MUST run before `DGemmaDenoise`
      implementation proceeds past its skeleton.
- [ ] **Scope boundary, stated explicitly: in-block bidirectional attention
      severing is OUT OF SCOPE for this ADR.** Cache surgery (§5) cuts
      *context routes* only — which prior tokens' K/V a block's decoder can
      attend to. Intervening on the decoder's own bidirectional mask
      *within* a block (severing attention between canvas positions in the
      same denoise step) is a different mechanism entirely and would need
      its own ADR. This ADR does not decide, and does not imply, anything
      about in-block mask intervention.
- [ ] **Block loop ownership: graph-side vs. node-internal.** `DGemmaDenoise`
      §4 states the block loop "moves to engine/node ownership" but does not
      decide whether iteration is node-internal (mirroring ADR-CDG-006's
      step-windowed-pipeline-subclass shape) or graph-side (a For/While loop
      pack this checkout doesn't ship). This is ADR-CDG-006's territory
      verbatim — inherited here, not re-litigated. **Resolution trigger:**
      same as ADR-CDG-006's own open state; no in-tree For/While primitive
      exists today, so node-internal iteration is the only buildable shape
      until that changes.

**Resolution plan:** the real-weights smoke test gates all implementation
past a skeleton; the scope boundary and block-loop-ownership questions are
recorded as open and must not be silently decided by implementation ahead of
their resolution triggers.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD.

## References

- Issue #47 — primary source: grounding report (encoder/decoder anatomy,
  cache class/lifecycle, injection seams, ranked blockers), payload-richness
  refinement, serialization correction.
- Issue #46 — backward-asymmetry discriminators, KV-not-tokens mechanism
  note, cache/window manipulation as a discriminator.
- Issue #40 — AR-dominance findings, fossil waves, cache as sole cross-block
  memory channel.
- ADR-CDG-001 — native socket types; `EMIT-CANONICAL / PARSE-AT-THE-DOOR`;
  the fingerprint/ingress rule this ADR's §2 instances.
- ADR-CDG-005 — `CANVAS_STATE` resumable save-state; KV exclusion (economy,
  not impossibility, per #47's correction).
- ADR-CDG-006 — step-windowed resumable sampler; the ownership model §4
  extends one axis.
- ADR-CDG-008 — MCP-center topology; `STATELESS-CORE` (Correction 1), the
  invariant §3's default ownership semantics protects.
- ADR-CDG-010 / ADR-CDG-011 — declarative-payload ingress discipline (rule
  7) the node pair inherits.
- `modeling_diffusion_gemma.py:350-351,422-449,1327` — encoder cache-write
  path; decoder never writes the cache (installed transformers 5.13.0).
- `pipeline_diffusion_gemma.py:324,326-333,429` — cached-length slicing;
  committed-canvas re-encode crossing (installed diffusers 0.39.0).
- `cache_utils.py:254,257-264,270,1499-1604` — per-layer `cumulative_length`,
  sliding-window crop, `DynamicLayer`/`DynamicSlidingWindowLayer` (same
  package).
