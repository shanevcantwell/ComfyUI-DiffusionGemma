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

## Data channels

*Added 2026-07-13 per operator review of PR #51 ("better definition of how the
actual data 'channels' of input and output work, to review before
implementation"). This section makes the seam's I/O concrete enough that an
implementer cannot misread it. It **does not alter** the Decision above — it
pins the shapes, provenance tags, entry points, ride locations, and per-door
validation the Decision leaves at schema-level. Where a shape cannot be pinned
without a de-risk read against real weights, that is surfaced as an Open
Question, not invented.*

Two node boundaries carry `KV_CACHE`; each has an INPUT face and an OUTPUT
face. The mint identity that makes a cache non-lying (§1, §2) is a single
dataclass — call it the **provenance envelope** — that rides *every* crossing.
Its concrete fields are pinned in §D.0, then each channel says which door it
crosses and what validation fires there.

### D.0 The `KV_CACHE` payload — concrete dataclass

The payload riding the `KV_CACHE` socket is a `dgemma/types.py` dataclass (the
identity, not the ComfyUI socket string — `IDENTITY⊥ENVELOPE`, ARCHITECTURE.md
rule 4), shaped like the existing `CanvasTrace`/`CanvasState` payloads: a live
object plus the mint metadata that keeps it honest. Proposed name `KVCache`
(dataclass) on socket string `DGEMMA_KV_CACHE`.

| field | type | meaning / shape | filled by |
|---|---|---|---|
| `cache` | `Any` (`transformers.DynamicCache`) | the live per-layer K/V store. Per layer `i`: `key_cache[i]`, `value_cache[i]` each a tensor of shape `(batch, num_kv_heads, seq_len, head_dim)`, dtype `bfloat16`, on the model's device. Layer count == the loaded model's decoder-layer count (30 for `26B-A4B`: 5 full-attention + 25 sliding, at indices `(i+1)%6==0` full). | `DGemmaEncode` |
| `cumulative_length` | `tuple[int, ...]` | per-layer running committed length — the grounding report's ranked-#1 blocker (`cache_utils.py:254`, mask offsets at `:270`). One entry per layer; a consumer NEVER hand-tracks it (§2, Neg-Consequences). | `DGemmaEncode` (advances on each encode) |
| `geometry` | `dict` | the geometry fingerprint (§2): `layer_types` pattern (the 5-full/25-sliding mask), `sliding_window` size, `batch`, `dtype`, per-layer-type RoPE params (full: proportional, θ=1e6, partial_rotary 0.25; sliding: default, θ=1e4). This is what ingress validates against the loaded model. | `DGemmaEncode` (read from `model.config`) |
| `provenance` | `Provenance` (dataclass) | the mint record (§1): `minting_sequence: tuple[int, ...] | None` (token ids the encoder consumed — present for tier 1; `None` once perturbed), `edit_script: tuple[EditOp, ...]` (empty for tier 1; the splice/ablate/scale ops for tier 2), `model_repo_id: str` + `tokenizer_fingerprint: str` (which model/vocab minted it — the identity ingress checks vocab alignment against). | `DGemmaEncode`; `edit_script` appended by any tier-2 surgery op |

`minting_sequence is None and edit_script == ()` is an **illegal state** — an
orphan cache with no provenance at all (the Neg-Consequences "orphan-cache
poisoning" failure). Ingress rejects it (§D.3). This is the ADR-CDG-001 lying-
payload rule in dataclass-invariant form.

### D.1 INPUT channels (what enters, where, provenance, entry point)

**IN-1 — sequence → `DGemmaEncode` (mint a fresh cache).** Enters as token ids
(`STRING`/token-id list). Provenance: this node is the *mint*; it stamps
`provenance.minting_sequence = <these ids>`, `edit_script = ()`,
`model_repo_id`/`tokenizer_fingerprint` from `dgemma_model`. No `KV_CACHE`
ingress validation (nothing crossed yet); the mint is where the envelope is
*created*, not checked. Foreign-AR-model authorship (concept.md: "any AR model
can author the run-on text") is handled here by feeding foreign-authored ids —
but DiffusionGemma's *own* encoder weights encode them, so the cache geometry
is always DG's, never the foreign model's. **Failure this prevents:** a cache
minted with no record of which ids produced it — unreproducible tier-1
conditioning.

**IN-2 — `KV_CACHE` → `DGemmaDenoise` (inject a known-provenance cache).** This
is issue #47's motivating capability. The cache enters as a `KV_CACHE` payload
(the dataclass above). Entry point into the one contract: a **new declarative
parameter on `run_diffusion`**, `kv_cache: KVCache | None = None` — additive-
optional, defaulting `None` (today's exact behavior: `DGemmaDenoise` with no
injected cache mints its own via the first encode, mirroring ADR-CDG-006's
`start_at_step > 0` requires-a-resume-input gate). This honors rule 7: the
cache is a **declarative payload validated at ingress, never an executable
participant** — `run_diffusion` receives *data* (a cache object + its envelope),
not a closure or a hook. `None` leaves rule-6 `STATELESS-CORE` trivially
satisfied (no injected state crosses; the run mints fresh). **Failure this
prevents:** the injection door being opened as a second executable seam (a
surface handing in code), which rule 7 forecloses — it is a data door only.

**IN-3 — `KV_CACHE` → `DGemmaEncode` (advance an existing cache, cross-block
re-encode).** The decoder→encoder committed-canvas re-encode crossing
(`pipeline_diffusion_gemma.py:429`) surfaced as a node input: a prior block's
`KV_CACHE` plus the newly-committed canvas ids enter `DGemmaEncode`, which
encodes the committed block into the cache and emits a **new** payload (§3
advance-returns-new-payload; OUT-2). Provenance: `minting_sequence` extends by
the committed ids (tier 1 stays tier 1); `cumulative_length` advances per
layer. **Failure this prevents:** a re-encode that mutates the input cache in
place, contaminating a fan-out branch (§3, the `STATELESS-CORE`-in-miniature
aliasing hazard).

**IN-4 — serialized `KV_CACHE` → load node (deserialize a tier-2 artifact).**
A `torch.save`d cache + its envelope re-enters from disk (§4 save/load pair).
Disk is a data-plane crossing like any node-to-node one (§1,
`CONSERVE-ACROSS-THE-DATA-BOUNDARY`): the same §D.3 ingress fires on the
deserialized payload. Provenance for tier-2 artifacts is `minting_sequence =
None` + a non-empty `edit_script` (the only reproduction path once perturbed).
**Failure this prevents:** a deserialized cache built against a different
model's geometry (device/dtype/layer-type drift) attaching silently — caught
by the geometry-fingerprint check, which was designed for caches of unknown
history (§4).

### D.2 OUTPUT channels (what comes back, where it rides, retention policy)

**OUT-1 — `DGemmaDenoise` → `KV_CACHE` (the advanced cache).** When
`DGemmaDenoise` stops at a block boundary (§4 optional stop), it emits the
advanced cache as a **new** `KV_CACHE` payload (§3), with `cumulative_length`
and `provenance.minting_sequence` advanced by the committed block. Rides its
own socket, not a trace field. **Retention policy — named, because KV caches
are large:** the advanced-cache output is emitted **only when the stop-at-block
boundary is requested**; a run to completion does not retain intermediate
per-block caches (they are advanced-through, not accumulated). A `KVCache`
payload is one live `DynamicCache` at a time — O(context) memory, not
O(context × blocks). The tier-2 serialization artifact (§5) is opt-in via the
save node; nothing auto-persists caches.

**OUT-2 — `DGemmaEncode` → `KV_CACHE` (the minted/advanced cache).** IN-1's
mint and IN-3's advance both exit here as a fresh payload. Same
advance-returns-new-payload discipline; same one-cache-at-a-time retention.

**OUT-3 — injection provenance → `CanvasTrace` (the record that a cache was
injected).** When a run is driven with an injected cache (IN-2), the fact and
identity of the injection must be recoverable from the trace, or a downstream
analysis cannot tell a conditioned run from an unconditioned one — the
`CanvasTrace` "mint identity gives the readings their meaning" discipline
(`types.py:99-124`) applied to the cache axis. Rides as an **additive-optional
field on `CanvasTrace`** under the #35 R6 additive-optional discipline
(`ARCHITECTURE.md` R6 row): `injected_cache_provenance: Provenance | None =
None` (default `None` — unchanged for every non-injected run today). It carries
the envelope's *identity* (minting sequence hash / edit-script summary /
`model_repo_id`), **not** the tensors — the tensors are large and already have
their own OUT-1/save-node home; duplicating them into the trace would violate
OUT-1's retention policy. **Failure this prevents:** a fossil-wave ablation
study whose trace cannot say the cache it ran against was injected/perturbed —
a conclusion that looks grounded but isn't (Neg-Consequences orphan-cache
poisoning, trace side).

**Per-step deltas are OUT OF SCOPE for this channel set.** Capturing how the
injected cache changes the *per-step* canvas distribution is a `DISTRIBUTION`-
tap concern (the seam-inventory `DISTRIBUTION` primitive, gated on #11/#14),
not a `KV_CACHE` output. This ADR's OUTPUT channels carry caches and injection
provenance; they do not carry per-step distribution deltas. Recorded so an
implementer does not fold a distribution capture into the cache payload.

### D.3 VALIDATION at each door (ingress checks + enforcement surface)

Every `KV_CACHE` ingress (IN-2, IN-3, IN-4) runs one validator —
`validate_kv_cache_ingress(payload, dgemma_model)` — before the payload is
used. Fail-on-mismatch, never trust-and-degrade (rule 5,
`EMIT-CANONICAL / PARSE-AT-THE-DOOR`). Each check names the failure it prevents
(greenfield anticipated-failure anchoring):

| # | ingress check | failure it prevents | enforcement surface |
|---|---|---|---|
| V1 | layer count of `cache` == loaded model's decoder-layer count | a cache from a differently-sized model attaching with a truncated/over-long layer set — silent wrong-geometry attention | `test_kv_ingress_layer_count_mismatch_raises` (unit, synthetic cache vs. model config) |
| V2 | `geometry.layer_types` / `sliding_window` / RoPE params == `model.config` derivation | the Neg-Consequences "silent geometry mismatch" — a cache built against one layer-type pattern fed to another produces wrong masks with no crash | `test_kv_ingress_geometry_fingerprint_mismatch_raises` |
| V3 | `cumulative_length` present, one entry per layer, non-negative | the ranked-#1 blocker: a stale/uninitialized `cumulative_length` silently corrupting mask offsets (`cache_utils.py:254,270`) — plausible-but-wrong mask | `test_kv_ingress_missing_or_ragged_cumulative_length_raises` |
| V4 | `provenance.tokenizer_fingerprint` / `model_repo_id` match the loaded model | vocab misalignment — a cache minted under a different tokenizer conditioning the canvas on token ids that mean something else (the orphan-provenance poisoning, vocab flavor) | `test_kv_ingress_vocab_mismatch_raises` |
| V5 | provenance non-orphan: NOT (`minting_sequence is None` and `edit_script == ()`) | a cache with no reproduction path at all — unreproducible, unauditable experimental input (§D.0 illegal state) | `test_kv_ingress_orphan_provenance_raises` |
| V6 | `cache` dtype / device match the loaded model | a CPU-loaded or fp32 deserialized cache (IN-4) attaching to a bf16-on-GPU model — device/dtype drift that would error deep in attention rather than at the door | `test_kv_ingress_dtype_device_mismatch_raises` |

Enforcement-surface home: these rows extend `ARCHITECTURE.md`'s enforcement-
surface table with a `KV_CACHE` ingress row (`NOT-YET-IMPLEMENTED` until the
node pair lands), alongside the existing declarative-payload ingress row (rule
7). The validator is engine-side (`dgemma/`, ComfyUI-agnostic per ADR-CDG-003);
the socket string is surface-side.

### D.4 Channel diagram

```
                         AR HEMISPHERE                    │  SEAM  │           DIFFUSION HEMISPHERE
                     (encoder — sole cache writer)        │ (MITM) │        (decoder — reads, never writes)
 ─────────────────────────────────────────────────────── │ ────── │ ───────────────────────────────────────

   token ids (own or foreign-authored AR text)
        │  IN-1
        ▼
   ┌───────────────┐   OUT-2                              │        │
   │ DGemmaEncode  │────────────►  KV_CACHE payload  ─────┼──[V1..V6]──►  ┌────────────────┐
   │  (mint /      │               { DynamicCache          │ ingress │     │  DGemmaDenoise  │
   │   advance)    │◄──── IN-3       + cumulative_length    validate │     │  (block loop,   │
   └───────────────┘   advance an    + geometry fp          │        │     │   canvas in)    │
        ▲                existing     + provenance envelope }│  IN-2  ├────►│                 │
        │                cache +                             (inject) │     └────────┬────────┘
        │                committed                          │        │              │
        │                canvas                             │        │      OUT-1 (stop-at-block):
        │                                                   │        │      advanced KV_CACHE ──┐
        └───────────────────────────────────────────────────────────┼──────────────────────────┘
                    committed-canvas re-encode (pipeline_diffusion_gemma.py:429)   (feeds IN-3)

   disk  ──[torch.load + envelope]──►  IN-4  ──[V1..V6]──►  (same ingress as IN-2)     │
                                                                                        ▼
                                                              CanvasTrace.injected_cache_provenance
                                                              (OUT-3 — identity only, not tensors)
```

Read: `DGemmaEncode` is the sole minter/advancer (mirrors the model's
sole-cache-writer encoder, `modeling_diffusion_gemma.py:350-351`);
`DGemmaDenoise` is the sole consumer (mirrors the read-only decoder). Every
arrow crossing the seam into a consumer passes `[V1..V6]` ingress. The
re-encode crossing (bottom) is the model's own decoder→encoder loop
(`:429`) surfaced as the IN-3 node input. OUT-3 records injection identity on
the trace without duplicating the tensors.

## Delivery & verification contract

*Added 2026-07-13 per operator review of PR #51 — the delivery/acceptance
contract stated verbatim: "inter-node data channels [are] fully exercised as
part of 100% code row coverage profiled across the greater system, with actual
workflow integrations ensuring that when this is wired by the user, at very
least the complete integration workflow(s) are provided, with a guarantee that
wiring them up independently in a new workflow will be effortless to get valid
results."* This section binds the Data-channels definition above to acceptance:
the channels are not "done" when the shapes are pinned — they are done when
their code is exercised end-to-end, shipped as working graphs, and provably
effortless to rewire fresh. It **does not alter** the Decision or the Data
channels — it names what makes them shippable and what enforces each guarantee,
because an acceptance bar enforced only by prose is one refactor from gone
(the greenfield discipline, `harness-tools#18`). Each clause names the failure
it prevents (anticipated-failure anchoring), consistent with §D's per-channel
"failure this prevents" register.

### DV.1 — 100% row coverage on channel-crossing code, in the FULL-SUITE profile

**Clause.** Every channel-crossing code row lands at **100% row coverage**,
measured in the greater-system profile (the full mocked suite), not a
unit-local run of one test module. The scoped set is the code the §D channels
traverse:

- the `KVCache` dataclass and its `Provenance`/`EditOp` members (`dgemma/types.py`
  additions, §D.0);
- `validate_kv_cache_ingress` and every V1–V6 branch (§D.3) — including each
  raise path, not just the happy path;
- the `run_diffusion` `kv_cache` ingress→use path (IN-2, §D.1) and the
  advance-returns-new-payload emit path (OUT-1/OUT-2, §D.2);
- trace provenance emission — the `CanvasTrace.injected_cache_provenance`
  population path (OUT-3, §D.2);
- the surface node bodies wrapping these (`DGemmaEncode` / `DGemmaDenoise` /
  the tier-2 save-load nodes) at the thin node/engine seam (ADR-CDG-003).

**Failure this prevents.** A channel-crossing branch — most acutely a V1–V6
*reject* path or the tier-2 `edit_script` provenance path — that is defined in
§D but never executed by a test, so a regression that silently disables an
ingress check ships green. The §D validators are the seam's honesty layer; an
un-exercised validator is a lying payload's open door.

**Enforcement surface.** A **per-module coverage floor** naming these modules
at `100`, distinct from the repo-wide number. The repo runs `pytest-cov`
in-process today (`tests/README.md`: the mocked suite "buys its 100% coverage
with fakes planted at exactly the boundaries"; `tests/test_dual_context_import.py`
notes the in-process constraint), but **no committed coverage config declares a
floor** — the 100% is achieved-by-convention, not gate-enforced. This clause
requires introducing the mechanism: a `[tool.coverage.report]` section in
`pyproject.toml` (the repo's single config home) that pins the new channel
modules at 100% — either via `fail_under = 100` scoped by an `include`/`omit`
that isolates the channel modules, or (if a single global `fail_under` cannot
express a per-module floor — see the Open Question below) a dedicated
coverage-gate test that reads the `.coverage` data and asserts the named
channel files are at 100%, failing CI by module name. **Scope is explicit and
not to be inflated:** the floor is the *new channel modules only*. Repo-wide
coverage is ~99% today with a pre-existing `dgemma/model.py` gap (the
`from_pretrained` boundary a fake cannot falsify, reachable only by the `live`
suite per `tests/README.md`); this clause does **not** claim, require, or imply
repo-wide 100%, and must not be read to regress the model.py gap into a blocker
it is not.

### DV.2 — shipped, validated integration workflows

**Clause.** The implementation ships **complete, working ComfyUI workflow
file(s)** exercising the seam end-to-end — source cache → ingress → generate →
trace/provenance readout — at minimum the **tier-1 honest-cache path**
(`DGemmaEncode` mints → `DGemmaDenoise` consumes → trace reads back the run),
plus a **tier-2 perturbed path** *if* per-layer surgery (§5) is in the
implementation's scope. The workflows live **in-repo under `examples/`**, the
existing home for shipped graphs (`examples/README.md`: `.api.json` = the
`/prompt` POST body, `.ui.json` = the canvas format), banked with provenance in
the same `examples/README.md` table as the existing `ping-smoke` / `p3-trace`
graphs — consistent with what the pack already ships for its P0–P3 nodes, not a
new convention.

**Failure this prevents.** A user who installs the pack finds the KV_CACHE
nodes on the palette but no reference wiring — the seam is buildable in
principle and unreachable in practice, tribal knowledge locked in the ADR. And
worse, the *rot* case: a shipped example that silently stops matching the node
definitions after a signature change — a shipped-but-rotted graph is worse than
none, because it fails at the user's hands with no CI warning.

**Enforcement surface.** A **workflow-conformance test** that loads each shipped
`examples/*.json` KV_CACHE graph and validates it against the *current* node
definitions — node `class_type` names resolve to registered nodes, wired socket
types match the nodes' declared `INPUT_TYPES`/`RETURN_TYPES` (the `DGEMMA_*`
native types, §DV.3a), and every required input is present. A node-signature
change that orphans a shipped workflow then **fails CI by the workflow's file
name**. This validation surface does **not exist today** (no test currently
loads `examples/*.json` against node defs — the existing E2E probe in
`examples/README.md` needs a *running* ComfyUI); this clause requires
introducing it as a no-GPU, no-running-server static conformance check in the
mocked suite. (Distinct from the live `.api.json` POST probe, which remains the
real-server E2E and stays `live`-gated.)

### DV.3 — effortless independent rewiring (the cold-wiring guarantee)

**Clause.** A user wiring the KV_CACHE nodes **fresh in a new workflow** — not
opening a shipped example — gets valid, non-degenerate results **without tribal
knowledge**. This is the operator's "guarantee that wiring them up
independently in a new workflow will be effortless to get valid results,"
decomposed into three *distinct* enforcement surfaces, because "effortless" is
not one mechanism:

**DV.3a — invalid connections are unwirable at the graph level.** The new
sockets are **native `DGEMMA_*` types** minted in `dgemma/types.py`
(`DGEMMA_KV_CACHE` per §D.0, plus any surgery/provenance socket the node set
introduces), extending ADR-CDG-001's discipline to this ADR's sockets. ComfyUI
refuses a wire between incompatible native types at the canvas, so a user
*cannot* connect a cache output to a canvas input by mistake — the invalid
graph is unbuildable, not merely discouraged. **Failure this prevents:** a
mis-wire that type-checks as `*`/`STRING` and fails deep in `run_diffusion`
instead of at the wire — the ADR-CDG-001 "lying socket" failure on the new
axis. **Enforcement surface:** the socket strings live in `dgemma/types.py`
(SSoT), asserted present and distinct by a node-contract test — the same shape
as the existing native-type contract tests.

**DV.3b — every ingress failure message is self-remedying.** Each V1–V6 reject
(§D.3) raises a message that names **both** the violated precondition **and**
the actionable remedy (e.g. V4 vocab mismatch → "cache minted under tokenizer
X, model loaded is Y; re-mint with the matching model or load model X"). A cold
user who mis-wires *around* the type system (e.g. a serialized tier-2 artifact
from a different model, IN-4) is told what is wrong and what to do, not handed a
bare assertion. **Failure this prevents:** a fail-fast validator that fails
*opaquely* — technically honest, practically tribal, sending the user to the
source to decode it. **Enforcement surface:** the §D.3 `test_kv_ingress_*_raises`
tests are strengthened to assert each message contains **both** the precondition
token and the remedy token (`pytest.raises(..., match=...)` on both), not merely
that it raised.

**DV.3c — the minimal legal wiring produces valid output.** Defaults are chosen
so that the *smallest* legal KV_CACHE graph — `DGemmaEncode` → `DGemmaDenoise`
with all parameters at default and no perturbation — yields a valid,
non-degenerate result. The grounded local-run defaults (CLAUDE.md: `max_steps=48
t=[0.4,0.8] entropy_bound=0.1 confidence=0.005 canvas_length=256`) are the
starting point; any KV_CACHE-specific default (e.g. `kv_cache=None` meaning
"mint fresh," IN-2) must keep the minimal graph legal-and-non-degenerate.
**Failure this prevents:** a node set where "wire the two nodes together" is
insufficient — a required-but-undefaulted knob, or a default that produces an
empty/degenerate canvas, so the user must *already know* the magic values to
get output. **Enforcement surface:** a **cold-wiring test** that constructs the
minimal node graph **programmatically** (calling the node bodies directly with
all-default parameters — NOT loading a shipped `examples/*.json`, so it is
independent of DV.2's fixtures and cannot be satisfied by a hand-tuned example)
and asserts the result is valid and non-degenerate (converged / non-empty
committed canvas). This test is the executable form of "effortless."

### DV.4 — enforcement-surface home and gating

The four new enforcement surfaces (DV.1 per-module coverage floor; DV.2
workflow-conformance test; DV.3a native-socket contract; DV.3b message-content
assertions; DV.3c cold-wiring test) extend `ARCHITECTURE.md`'s enforcement-
surface table alongside the §D.3 `KV_CACHE` ingress row, `NOT-YET-IMPLEMENTED`
until the node pair lands — the doc and the code move together
(ARCHITECTURE.md's own discipline). None of these clauses is `live`-gated: all
run in the mocked, no-GPU, no-running-server default suite, so they gate every
PR, not only a weights-available run. The real-weights smoke test (Open
Questions) remains the separate `live` gate on *correctness*; DV.1–DV.3 gate
*deliverability*.

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

- [ ] **`CANVAS_STATE` resume under an injected tier-2 cache — a genuine
      tension with ADR-CDG-005 the channel definition surfaced.** ADR-CDG-005
      excludes KV from the resume save-state on the ground that it is
      *recomputable from the committed prefix via one prefill pass* (ADR-CDG-005
      §"KV cache is deliberately excluded", Option B). That ground holds for a
      **tier-1** cache (its minting sequence prefills it back). It does **not**
      hold for a **tier-2** perturbed cache: no prefill reproduces it (§5,
      Option D — that is precisely why tier 2 requires serialization). So a
      `CANVAS_STATE` captured mid-run *while a tier-2 cache is injected* is not
      self-sufficient — resuming it re-prefills a cache that never matches the
      perturbed one that produced the frames. This does **not** reopen
      ADR-CDG-005's routine exclusion (tier-1 resume is unaffected, per Option
      C); it names a case ADR-CDG-005's recomputability premise did not cover.
      **Resolution trigger:** decide, before `CANVAS_STATE`+`KV_CACHE`
      co-capture is built, whether a tier-2-injected resume must reference the
      serialized cache artifact (by envelope identity, like OUT-3) rather than
      relying on prefill — a per-run flag on `CANVAS_STATE`, not a change to its
      default shape. This is a channel-composition question, not a Decision this
      ADR makes.

- [ ] **DV.1 mechanism: can a single `fail_under` express a *per-module* floor,
      or is a coverage-gate test required?** The operator's bar is 100% on the
      *new channel modules specifically*, held distinct from repo-wide (~99%,
      pre-existing `dgemma/model.py` gap). `coverage.py`'s `fail_under` is a
      single global threshold — it cannot natively assert "these named files at
      100%, the rest unchanged." Two candidate mechanisms (DV.1): (a) an
      `include`/`omit`-scoped coverage config run that isolates the channel
      modules and applies `fail_under = 100` to *that* scoped run, or (b) a
      dedicated gate test that reads the `.coverage` data and asserts the named
      channel files at 100%. The bar is **not weakened** by this uncertainty —
      the number stays 100% on the scoped set regardless of which mechanism
      lands. **Resolution trigger:** decided when the coverage-gate is
      implemented alongside the node pair; the mechanism is an implementation
      choice, the 100%-on-channel-modules floor is not. Recorded here rather
      than silently picking a mechanism that might not express the per-module
      floor.

**Resolution plan:** the real-weights smoke test gates all implementation
past a skeleton; the scope boundary and block-loop-ownership questions are
recorded as open and must not be silently decided by implementation ahead of
their resolution triggers. The `CANVAS_STATE`-under-tier-2 tension is recorded
as open (not decided here) and gates any `CANVAS_STATE`+`KV_CACHE` co-capture
work. The DV.1 coverage-mechanism question is recorded as open on *mechanism
only* — the 100%-on-channel-modules bar itself is not open.

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
