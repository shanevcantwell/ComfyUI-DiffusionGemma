# ADR-CDG-025 — MCP KV-path parity via a server-side cache-handle registry

**Status**: Accepted
**Date**: 2026-08-05
**Related**: ADR-CDG-012 (`KV_CACHE` socket + encode/denoise nodes — primitives
1+2 of the data-boundary discipline this ADR instances on a live object
instead of a tensor payload), ADR-CDG-024 (prompt-under-injection
composition — this ADR's `generate` tool design is composition-compatible
with, not exclusive against, ADR-CDG-024's accepted contract), ADR-CDG-008
(MCP-center topology — rule 2, the invariant this ADR repairs), Issue #103
(parity-tracking issue — the driving comment, 2026-08-04, is this ADR's
evidence), Issue #255 (KV-injection non-convergence defect — this ADR's
falsifiable acceptance evidence), Issue #257 (ADR-CDG-024's implementation,
open as of this writing — see §Composition compatibility)

---

## Context

ARCHITECTURE.md rule 2 states MCP is the canonical surface and ComfyUI
consumes it, not `dgemma/` directly. Issue #103's 2026-08-04 comment makes
the violation concrete: `surfaces/mcp/commands/generate.py` exposes no
`kv_cache` argument and no encode tool exists, while ComfyUI's
`DGemmaEncode`/`DGemmaDenoise` pair (ADR-CDG-012) has driven the KV path
since Phase 4 landed (`ac3c832`, PR #242). A surface-independent repro of
defect #255 (the KV-injection stall) is impossible through MCP today — the
canonical surface has fallen behind the contract it is supposed to track,
inverting rule 2.

`generate.py`'s own docstring names the blocker as it understood it:
`KVCache.cache` is a live `transformers.DynamicCache` with no JSON/disk
encoding, and inventing that serialization scheme was treated as a
prerequisite. That premise is wrong. The repo's own data-boundary crossing
discipline (ARCHITECTURE.md §"The data-boundary crossing discipline";
primitives (1) mint-identity guard and (2) self-distrust on resume) already
answers bulk-artifact crossings **by pointer**, not by value — serialization
is one candidate transport, not the only one. A live cache never needs to
leave the server process at all: it can be held server-side, next to the
model, and handed to the caller as an opaque handle.

**The tension this ADR must resolve, not dodge (per the design brief):**
ARCHITECTURE.md rule 6 says the model load is the *only* persisted object
this surface holds (`surfaces/mcp/state_manager.py`'s own docstring:
"deliberately holds nothing else: no scheduler, no canvas, no run-state, no
cache keyed on prompt/knobs" — and names a hypothetical cached-scheduler as
exactly the violation it exists to foreclose). A server-side cache registry
is a second persisted resource by construction. This ADR amends rule 6
rather than silently violating it or refusing to build the feature (see
§Decision/1).

## Decision

**Add a bounded, model-scoped KV-cache registry to `StateManager`, exposed
through two new MCP tools (`encode`, and a `kv_cache_id` parameter on the
existing `generate`), resolved by handle at ingress with fail-on-unknown.**
No disk serialization; the cache never leaves the server process. This
mirrors `DGemmaEncode`/`DGemmaDenoise`'s ComfyUI-side shape (mint/advance
node → live socket payload → consuming node) with the MCP transport's
actual constraint substituted for ComfyUI's in-graph object reference: a
JSON-safe string handle standing in for a live Python object a JSON
payload cannot carry.

### 1. Rule 6 amendment: cache lifecycle is a second, explicitly bounded persisted class

Rule 6 today reads: *"The core is stateless across runs; only the model load
persists... The ~53 GB model load is the only persisted object."* This ADR
amends it to:

> The core is stateless across runs. The surface tier persists exactly two
> object classes: the model load (rule 6, unchanged — one per server
> process, replaced wholesale on `load_model`), and a **bounded KV-cache
> registry scoped to the currently loaded model** (this ADR) — entries
> keyed by opaque handle, evicted on model unload/swap, with no other
> surface-held cross-call state permitted. A future third persisted class
> requires its own ADR amendment; this is not an open door.

This is not a waiver of `STATELESS-CORE` — the core (`dgemma/`) gains no new
persisted state; `run_diffusion` still builds a fresh scheduler/collector/
composite every call. The registry lives in `surfaces/mcp/state_manager.py`,
the surface tier's own named exception (the model load), widened by one
sibling class with its own lifecycle contract (§4), not folded silently into
the model-load exception's existing wording.

**Why amend rather than scope per-session/per-connection (the rejected
alternative):** MCP's stdio/JSON-RPC transport (`surfaces/mcp/server.py`) is
one process, one model load, request/response per tool call — there is no
per-connection session object today (`cancel_run`'s `_active_runs` registry,
the one precedent for call-scoped server state, is deliberately
process-global and keyed by caller-chosen `run_id`, not a connection handle).
Inventing session scoping *for this feature alone* would add a second
lifecycle concept the rest of the surface doesn't have, to solve a problem
(cross-caller cache leakage) this repo's actual deployment shape — one
server process, one operator, one model at a time (ARCHITECTURE.md
"Lifecycle & tenancy — honest absence") — does not yet present. See
§Alternatives Option A for the full weighing.

### 2. `encode` tool: text in, `kv_cache_id` out

New MCP tool, thin adapter over `dgemma.kv_cache.encode_sequence` — same
shape as `DGemmaEncode`'s ComfyUI body (`surfaces/comfyui/encode.py:90-97`):
tokenize with the loaded model's tokenizer (`tokenizer.encode(text)`, no
chat template — mirrors `DGemmaEncode`'s deliberately raw-encode contract,
issue #47's cache-perturbation instrument), call `encode_sequence(model,
token_ids, into=<resolved prior cache or None>)`, store the returned
`KVCache` in the registry, return `{"kv_cache_id": <handle>}`.

```
encode(prompt: str, kv_cache_id: str | None = None) -> {"kv_cache_id": str}
```

- `kv_cache_id=None` (or omitted): fresh mint (IN-1 parity).
- `kv_cache_id=<existing handle>`: advance (IN-3 parity) — resolved at
  ingress against the registry (§3); a dangling or unknown handle is
  rejected, never silently treated as "start fresh." Per
  advance-returns-new-payload (ADR-CDG-012 §3), the **input** handle's
  registry entry is replaced by the **output** — same handle string, new
  `KVCache` value — mirroring the ComfyUI node's own "new payload, not a
  mutation" contract without inventing a second handle per advance.

No `advance`-specific tool: `DGemmaEncode`'s "same node body, `into is
None` dispatches mint vs. advance" shape (encode.py's own docstring)
transcribes directly to "same tool, `kv_cache_id` presence dispatches
mint vs. advance" — one tool, not two, for parity with the one-node
precedent.

### 3. `generate` tool: `kv_cache_id` parameter, resolved at ingress, fail-on-unknown

`generate`'s JSON schema gains one new optional property:

```json
"kv_cache_id": {
  "type": "string",
  "description": "Handle from a prior encode call. Resolved server-side "
                  "against the currently loaded model's cache registry; "
                  "an unknown or expired handle is rejected, never silently "
                  "substituted with a fresh cache."
}
```

At the top of `generate()` (`surfaces/mcp/commands/generate.py`), before any
`run_diffusion` call: if `kv_cache_id` is supplied, resolve it against
`manager.resolve_kv_cache(kv_cache_id)`. **Fail-on-unknown is a designed
rejection, not a degrade path** (`EMIT-CANONICAL / PARSE-AT-THE-DOOR`,
ADR-CDG-001/rule 5): a dangling handle — model was unloaded/swapped since
`encode`, handle was mistyped, entry was evicted (§4) — raises a structured
`{"error": "..."}` naming the handle and why it's unresolvable, exactly the
same posture `state_manager.require_model()` already takes for a missing
model load (`state_manager.py:71-81`, "a loud RuntimeError naming the
missing precondition — never a silent None handed on to
dgemma.run_diffusion"). The resolved `KVCache` is passed straight to
`run_diffusion(kv_cache=..., ...)` — `run_diffusion`'s own
`validate_kv_cache_ingress` (V1–V6, ADR-CDG-012 §D.3) still runs beneath
this; the handle resolve is a *new, earlier* door (does this handle exist
and still name a live cache), not a replacement for the *existing* one
(is the cache well-formed against the loaded model). Two doors, two
different failure classes, both fail-loud.

**Prompt composes with the handle — this is composition-compatible with
ADR-CDG-024, not exclusive against it.** ADR-CDG-024 (Accepted) supersedes
issue #248's exclusivity invariant: `prompt` and `kv_cache` are jointly
permitted at `run_diffusion`'s ingress, with `prompt` (when non-empty)
chat-templated and prefilled onto the cache before the decode loop. This
ADR's `generate` tool changes nothing about that contract — `prompt` and
`kv_cache_id` are both ordinary optional parameters on the same tool call,
and whichever combination `run_diffusion`'s ingress currently accepts is
exactly what a resolved-handle-plus-prompt call reaches. **Status note:**
ADR-CDG-024's implementation (issue #257) is open as of this writing —
`dgemma/ingress.py:238`'s `reject_prompt_and_kv_cache` still enforces the
interim #248 exclusivity in the code this ADR's implementation will land
against. This ADR's tool schema does not re-implement or duplicate that
check (thin-adapter discipline, same as every other `generate` door); it
simply hands `prompt` and the resolved `kv_cache` to `run_diffusion`
unconditionally, so the MCP surface's composition-readiness is automatic
the moment #257 merges, with no MCP-side follow-up change required.

### 4. Lifecycle: eviction on unload/swap, bounded registry, no orphan survives its model

**A cache surviving its model is the anticipated failure this section
exists to prevent** (greenfield rule — every new invariant names the
failure it prevents). A `DynamicCache`'s tensors are shaped by the minting
model's layer count, geometry, dtype, device (ADR-CDG-012 §D.0); a stale
handle resolving after `StateManager.load()` has replaced `_model` with a
different model would hand `run_diffusion` a cache whose geometry no longer
matches what `validate_kv_cache_ingress` checks it against — V1/V2/V6
(ADR-CDG-012 §D.3) would catch the mismatch at the *second* door, but
letting a plainly-dead handle reach that far is a wasted round trip and an
honesty gap: the registry itself should already know the entry is dead.

- **Eviction trigger: model unload/reload.** `StateManager.load()`
  (`state_manager.py:57-69`) already unconditionally replaces `_model`
  ("(Re)load the model, replacing whatever was previously held"). This ADR
  adds: on every `load()` call, clear the entire KV-cache registry first.
  No partial survival, no "same repo_id so keep the caches" heuristic — a
  reload is a new model object even if the weights are byte-identical
  (`load_model` is not memoized; re-running it is a real new load), and a
  cache is scoped to the model *object* that minted it, not the repo_id
  string.
- **Identity sidecar, checked at resolve time, not just at ingress.**
  Every registry entry carries the same `Provenance` (`model_repo_id`,
  `tokenizer_fingerprint`) ADR-CDG-012 §D.0 already mints on every
  `KVCache`. `resolve_kv_cache` compares the entry's `provenance.
  model_repo_id`/`tokenizer_fingerprint` against the *currently loaded*
  model before returning it — belt-and-suspenders against a same-process,
  different-model-generation race the unconditional-clear-on-load should
  already foreclose, but the identity check is what makes the foreclosure
  structural rather than sequencing-dependent (a future code path that
  loads a model without going through `StateManager.load()` would still be
  caught here). Mismatch is a resolve-time rejection, same fail-loud
  posture as an unknown handle (§3) — a mint-identity guard (data-boundary
  discipline primitive 1) applied to a live object instead of a
  disk-crossing tensor artifact.
- **Bounded registry — no unbounded growth.** A cap (fixed count, e.g. a
  small constant such as 8 concurrent handles, or an LRU eviction on a
  size ceiling — the implementer's call, named as an open question below,
  not silently decided) prevents an MCP client that calls `encode`
  repeatedly without ever calling `generate` from growing server VRAM
  usage unbounded. `KVCache.cache` tensors are per-layer, per-position —
  the same "large, don't accumulate" concern ADR-CDG-012 §D.2 OUT-1 names
  for the ComfyUI-side advanced-cache output ("a `KVCache` payload is one
  live `DynamicCache` at a time — O(context) memory, not O(context ×
  blocks)"), here applied across concurrent *callers* instead of across
  blocks within one run.
- **No idle-TTL requirement.** Unlike a network session, there is no
  transport-level "connection closed" signal to key eviction off in MCP's
  stdio/request-response model (the same absence `generate.py`'s
  cancellation docstring already names for `cancel_run`'s wiring). Model
  unload/swap plus the bounded-count ceiling are the two eviction levers;
  no additional wall-clock expiry is introduced by this ADR.

### 5. What is explicitly NOT this ADR

- **Phase-3 disk serialization** (ADR-CDG-012 §4 "save/load pair... for
  tier-2 artifacts") is orthogonal and stays deferred. This ADR's registry
  answers a different question — how a live cache crosses the MCP tool
  boundary within one server process's lifetime — not how a cache survives
  a process restart or moves between machines. A future disk-serialization
  ADR can add persistence *underneath* the same handle vocabulary this ADR
  mints (a handle resolving to either an in-memory entry or a lazily
  deserialized one) without changing this ADR's `encode`/`generate` tool
  contracts.
- **Tier-2 per-layer cache surgery** (ADR-CDG-012 §5) — no surgery tool is
  added here; the registry is agnostic to whether a stored `KVCache` is
  tier-1 (intact minting sequence) or tier-2 (edit-scripted). A future
  surgery tool would resolve-then-mutate-then-restore the same way
  `generate` resolve-then-consumes.
- **OUT-1 (advanced-cache output from `DGemmaDenoise`'s stop-at-block)** —
  ADR-CDG-012 §D.2 defers this ComfyUI-side; this ADR does not add an
  MCP-side equivalent (a `generate` call does not return an advanced
  `kv_cache_id`). Naming this so a future OUT-1 implementer does not
  silently assume `generate`'s registry semantics already cover it.

## Rationale

### Positive Consequences

- **Closes the rule-2 parity gap named by #103's driving comment**: a
  surface-independent repro of #255 becomes possible (§Falsifiable
  acceptance evidence, below) — the concrete acceptance test the parity
  violation itself demanded.
- **No invented serialization scheme.** The blocking premise in
  `generate.py`'s current docstring (no JSON/disk encoding for
  `DynamicCache`) is dissolved by pointer-passing, not solved by building
  a new tensor container format under time pressure — Phase-3
  serialization remains a real, separately-scoped design question with
  its own container-format tradeoffs (per #103's own tracking comment:
  "Tensor container format for serialized kv_cache remains an
  implementation detail for the first implementer"), not conflated with
  this feature.
- **Mirrors the proven ComfyUI-side shape.** `DGemmaEncode`/`DGemmaDenoise`
  already established mint/advance-returns-new-payload and
  ingress-validates-at-consumption as the working pattern (ADR-CDG-012,
  landed and Q-2-smoke-proven); this ADR transcribes that shape onto MCP's
  transport constraint (string handle standing in for an in-graph
  reference) rather than inventing a new one.
- **Rule 6 gains an explicit, bounded amendment instead of silent
  erosion.** A future reviewer asking "why does `state_manager.py` hold
  more than the model now?" finds this ADR's answer, not a drifted
  docstring.

### Negative Consequences

- **`StateManager` grows real lifecycle logic it did not have before** —
  eviction-on-load, a bounded-count ceiling, identity-check-at-resolve.
  `tests/test_mcp_statelessness.py` (which "mutation-checks this file
  directly... asserts a hypothetical cached-scheduler shape would be
  caught") needs updating to distinguish "a new, ADR-sanctioned bounded
  cache-registry class" from "an unsanctioned third persisted-state axis"
  — named here so the implementer does not silently loosen that test's
  intent while making it pass.
- **A registry entry is a live GPU-resident tensor object held between
  tool calls** — the MCP server process's VRAM footprint is no longer
  fully characterized by "the model load" alone. This is the direct cost
  of the rule-6 amendment (§1) and is why the bound (§4) is load-bearing,
  not cosmetic.
- **Two fail-loud doors instead of one** (handle-resolve, then
  cache-ingress-validate) adds a small amount of surface-tier logic
  (§3) that has no ComfyUI-side analog — ComfyUI's graph wiring makes an
  unresolvable handle structurally impossible (a disconnected socket has
  no value to pass), so this failure class is MCP-transport-specific, not
  a gap ComfyUI's existing tests already cover by transcription.

## Alternatives Considered

### Option A: Per-session/per-connection cache scoping instead of a named registry

Scope each cache to the MCP client connection that minted it (session-local
state), rather than a single process-global registry.

**Why rejected:** MCP's stdio/JSON-RPC transport in this repo is
single-connection, single-process, single-model, one-operator
(ARCHITECTURE.md "Lifecycle & tenancy — honest absence": "in-process,
single-tenant... nothing external starts, stops, swaps, or arbitrates
tenancy"). There is no existing session/connection object anywhere in
`surfaces/mcp/` to scope onto — `state_manager.StateManager` is
constructed once, module-scope, in `server.py`, and `cancel_run`'s
`_active_runs` (the only precedent for call-scoped state) is
process-global-by-caller-supplied-id, not connection-scoped. Building
session scoping for this one feature would introduce a second lifecycle
concept (session vs. process) the rest of the surface doesn't have, to
solve cross-caller cache leakage that isn't a live problem at today's
single-tenant deployment shape (ARCHITECTURE.md's own "NOT-YET-BUILT"
framing for multi-tenant). If/when a served-engine, multi-surface topology
(the ADR-candidate ARCHITECTURE.md names, issue #92) actually lands,
session scoping becomes the right question to ask again — deferred to
that trigger, not decided now on a hypothetical.

### Option B: Exposing raw `encode` vs. composition-only (`generate` accepts inline text to encode-then-drive in one call, no standalone handle)

Fold encode into `generate` itself — a `context_text` parameter that
`generate` encodes internally before driving, never exposing a
standalone, reusable `kv_cache_id`.

**Why rejected:** Collapses the "independent encoder context" capability
ADR-CDG-024 exists to enable — #245's operator-named use case is
explicitly a donor context (a document, a prior exchange) reusable across
multiple `generate` calls without re-encoding it each time (the IN-3
advance parity, §2). A composition-only door also breaks parity with
`DGemmaEncode`'s standalone-node shape on the ComfyUI side: ComfyUI can
mint a cache once and fan it into `DGemmaDenoise` across separate graph
runs without re-encoding; folding encode into `generate` would make MCP
strictly *less* capable than ComfyUI on this axis, reopening rule 2 from
the other direction. Rejected as inconsistent with parity being the
actual goal (ARCHITECTURE.md rule 2, "MCP is the canonical surface" — a
canonical surface that can only do the union of what a single tool call
does is not tracking the contract).

## Falsifiable Acceptance Evidence

The #255 stall repro becomes runnable through MCP: `encode(prompt=<the
banked #245 stall-trace context>)` → `kv_cache_id`, then
`generate(kv_cache_id=<handle>, ...<same knobs/seed as the banked trace>)`.
**Pre-#257-fix** (today's `dgemma/ingress.py:238` interim state, if
`prompt` is also supplied) or **post-#257-fix composed drive body**
(ADR-CDG-024 §1 landed): observe the same stall signature the #245 evidence
chain already banked (flat 1/256 structural floor, or 252–256/256
full-canvas re-noise) pre-fix, and convergence within the AIO control's
healthy range post-fix — via MCP tool calls only, no direct `dgemma.*`
import, no ComfyUI graph. This is the acceptance test #103's parity
violation itself demanded ("a surface-independent repro of defect #255...
is impossible today"); it becomes possible the moment this ADR's `encode`
tool and `generate`'s `kv_cache_id` parameter exist, independent of whether
#257 has landed yet — the pre-fix stall and the post-fix convergence are
both valid, distinct observations this same MCP-only repro path produces at
different points in #257's landing.

## Open Questions

- [ ] **Registry bound: fixed count vs. LRU-on-size-ceiling.** §4 names the
      need for a bound but not its concrete mechanism (a small fixed slot
      count vs. a byte-size ceiling with LRU eviction). **Resolution:**
      implementer's call at implementation time; either satisfies this
      ADR's "no unbounded growth" requirement, chosen and documented in the
      implementing PR.
- [ ] **Handle format.** Whether `kv_cache_id` is a random opaque token
      (e.g. `uuid4().hex`), a monotonic counter, or caller-suppliable (like
      `run_id` on `cancel_run`/`generate`) is unresolved. **Resolution:**
      implementer's call; a caller-supplied option would mirror `run_id`'s
      precedent (§3's cancellation mechanism) but is not required for this
      ADR's contract — a server-minted opaque handle satisfies the design
      as written.
- [ ] **`tests/test_mcp_statelessness.py` update shape.** The negative
      consequence above names that this test needs to distinguish a
      sanctioned bounded registry from an unsanctioned state axis, but does
      not specify the new assertion shape. **Resolution:** implementation
      PR; the intent (catch a *third*, unbounded, unevicted persisted class)
      must survive the update, verified by the design-gate review.

**Resolution plan:** all three open questions are implementation-detail
choices within this ADR's contract, not decisions that change the contract
itself (handle-in/handle-out, fail-on-unknown, evict-on-reload,
identity-checked-at-resolve). They resolve in the implementing PR and are
reviewable there without reopening this ADR.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD.

## Ratification

Ratified 2026-08-05 by an independent Opus design-gate review (a reviewer that
did not author this ADR), per the repo's strict-waterfall process convention
(CLAUDE.md, "Process conventions"). PASS. Every load-bearing claim was checked
against origin/main ground truth, not accepted on prose.

**Doctrine conformance — verified:**

- **Rule 2 (canonical surface tracks the contract):** The rule-2 inversion is
  real and correctly diagnosed — the ComfyUI KV path landed at Phase 4
  (`ac3c832`, PR #242) while `surfaces/mcp/commands/generate.py` still declines
  `kv_cache` (verified: the docstring at `generate.py:43-54` names the
  by-value-serialization premise this ADR dissolves by pointer-passing).
  Option B's rejection (§Alternatives) guards the repair from re-opening rule 2
  from the other direction (MCP made strictly less capable than ComfyUI).
- **Rule 6 tension RESOLVED, not waved at:** §1 amends rule 6 with the exact
  before-text quoted from ARCHITECTURE.md and a lifecycle-complete after-text —
  the second persisted class is faced honestly with eviction (§4:
  evict-entire-registry on `StateManager.load()`), invalidation
  (identity-checked-at-resolve against the currently-loaded model), and
  boundedness. Per-session scoping is the *considered-and-rejected* alternative
  (Option A), grounded in a verified fact: there is no connection/session object
  anywhere in `surfaces/mcp/`, and the sole precedent for call-scoped state,
  `_active_runs` (`generate.py:110`), is process-global and caller-keyed, not
  connection-scoped. Either resolution the design brief permitted (amended
  clause with lifecycle, or per-session scoping faced honestly) is present; this
  ADR takes the first with the second explicitly weighed.
- **Rule 7 ingress:** The resolved handle is a JSON-safe string parameter, not a
  surface-supplied closure — rule 7's declarative-payload door is preserved. The
  resolved `KVCache` still passes the existing `validate_kv_cache_ingress`
  (V1–V6, verified present in `dgemma/kv_cache.py`); §3's handle-resolve is a
  new, earlier door, not a replacement for the existing one.
- **Pointer + identity sidecar:** Verified — `Provenance`
  (`model_repo_id`/`tokenizer_fingerprint`) exists on every `KVCache` and is
  checked at resolve time (§4), instancing data-boundary primitive (1)
  mint-identity guard on a live object rather than a disk-crossing tensor.
- **EMIT-CANONICAL (dangling handle = designed rejection):** §2/§3 state
  fail-on-unknown with no silent fallback to a fresh cache; matches the
  `require_model()` loud-RuntimeError posture (`state_manager.py:71-81`,
  verified).
- **Greenfield anticipated-failure rule:** §4 opens by naming
  cache-outlives-model as the anticipated failure it prevents.

**Requirements — verified:** parity is framed as rule-2 invariant repair per the
#103 operator ruling, not a feature-add; composition-compatibility with
ADR-CDG-024 (Accepted, verified to supersede #248's exclusivity) is by
construction — the tool hands `prompt`+`kv_cache` unconditionally to
`run_diffusion` and defers to whatever its ingress accepts, re-implementing no
exclusivity check at the MCP layer (and the ADR accurately notes
`dgemma/ingress.py:238`'s interim #248 rejection still stands on main until #257
lands); no dependency on Phase-3 disk serialization (§5 keeps it deferred and
orthogonal).

**Cross-artifact seams — verified:** #103 (OPEN, driving evidence), #255 (OPEN,
falsifiable acceptance via MCP-only repro), #257 (OPEN, ADR-CDG-024's
implementation — accurately flagged as pre-fix at time of writing), ADR-CDG-012
primitives (`encode_sequence(into=...)`, `Provenance`, V1–V6 all verified to
exist), and the rule-6 text change named with an exact quote.

**Resolved recommendations:** none blocking. The three Open Questions (registry
bound mechanism, handle format, `test_mcp_statelessness.py` update shape) are
genuine implementation-detail choices within the fixed contract
(handle-in/handle-out, fail-on-unknown, evict-on-reload,
identity-checked-at-resolve) and resolve in the implementing PR, reviewable
there — the gate confirms the contract invariants are held fixed and the
statelessness test's intent (catch a *third*, unbounded, unevicted persisted
class) must survive its update.

## References

- Issue #103 — parity-tracking issue; 2026-08-04 comment ("Parity violation
  made concrete") is this ADR's primary driving evidence —
  https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/103
- Issue #255 — KV-injection non-convergence defect; this ADR's acceptance
  test reproduces it through MCP —
  https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/255
- Issue #257 — ADR-CDG-024's implementation (open as of this ADR) —
  https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/257
- ADR-CDG-012 — `KV_CACHE` socket + encode/denoise node pair; §D.0 payload
  shape, §D.3 V1–V6 ingress, §3 advance-returns-new-payload — this ADR's
  registry entries are the same `KVCache` dataclass, held server-side
  instead of riding a ComfyUI graph edge —
  `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md`
- ADR-CDG-024 — prompt-under-injection composition; this ADR's `generate`
  tool is composition-compatible by construction (§3) —
  `decisions/adr-cdg-024-prompt-under-injection-composition.md`
- ADR-CDG-008 — MCP-center multi-surface topology; rule 2 (MCP is
  canonical), rule 6 (model load is the only persisted object, amended by
  §1 of this ADR) — `decisions/adr-cdg-008-mcp-center-multi-surface-topology.md`
- `surfaces/mcp/commands/generate.py:43-54` — the docstring naming the
  by-value serialization premise this ADR replaces with by-pointer
- `surfaces/mcp/state_manager.py:1-27` — the model-load-is-the-only-
  persisted-object doctrine this ADR amends
- `surfaces/comfyui/encode.py` — the sibling ComfyUI-surface capability
  this ADR achieves parity with (same mint/advance shape, MCP-transport
  handle substituted for a graph-edge object reference)
- ARCHITECTURE.md — rule 2 (MCP canonical), rule 6 (persisted-state
  discipline, amended §1), §"The data-boundary crossing discipline"
  (primitives 1+2, this ADR's pointer + identity-sidecar mechanism)
