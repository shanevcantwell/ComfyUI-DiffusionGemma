# ADR-CDG-012 Phase 4 gate — Q-2 smoke + liveness-sweep + rung-1 probe: pre-registered protocol

Status: DRAFT protocol artifact for the Phase-4 gate. Not a decision record — bank
the run outputs and the resulting verdict as a comment on issue #62
(https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62); this file
does not itself close the gate. `docs/experiments/`-vs-`docs/evidence/` disposition
and the `runs/` JSONL convention follow issue #101
(https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/101,
"lab-notebook floor... in-repo runs/ convention + retroactive salvage") — cited as
the target convention for raw-artifact banking, **not ratified** (issue is OPEN,
proposal still draft per its comment 3, 2026-08-04). Where this protocol needs a
banking location today, it names the current de-facto pattern
(`docs/experiments/<entry>/`, `.md` narrative + attachments) and flags the #101 gap
explicitly rather than inventing a JSONL schema #101 hasn't ratified.

Written 2026-08-04. All H0s below are stated BEFORE any run in this window
executes. Pass/fail thresholds are fixed here, not adjusted post-observation.

---

## 0. Context pointers (verify without re-deriving)

- `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md:618-623` — Open Q #1,
  the untested assumption this smoke resolves, verbatim: *"the decoder driven
  with a caller-built cache the pipeline didn't create. Position/mask math should
  hold per the cited source (#47 grounding), but this is unverified against real
  weights."* Resolution trigger stated in the same block: the real-weights smoke
  MUST run before `DGemmaDenoise` implementation proceeds past its skeleton.
- `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md:222-234` (IN-2) — the
  channel this smoke exercises: `kv_cache: KVCache | None = None` on
  `run_diffusion`; `None` (today) mints fresh via first encode; a supplied cache
  should skip that first encode and drive the decoder off it instead.
- `dgemma/loop.py:400-432` — the current fail-loud door. `kv_cache is not None`
  runs `validate_kv_cache_ingress` (V1–V6) then unconditionally raises
  `NotImplementedError`, naming issue #62 Phase 4 and the ADR's smoke gate as the
  remedy path. This is the code the drive-body rewrite replaces; the smoke's PASS
  path is what licenses replacing it.
- `dgemma/kv_cache.py:103-243` (`validate_kv_cache_ingress`) — V1 (layer count) →
  V2 (geometry) → V4 (vocab/tokenizer fingerprint) → V3 (cumulative_length) → V6
  (dtype/device) → V5 (orphan-provenance), fail-on-first-mismatch. This already
  runs today on every `kv_cache=` call and is NOT what Q-2 tests — V1–V6 confirm
  the payload is well-formed; Q-2 tests whether the **decoder**, once the door is
  open, produces correct output when actually driven off that payload.
- `dgemma/kv_cache.py:245-...` (`encode_sequence`) — mints (IN-1) or advances
  (IN-3) a `KVCache` via the encoder. Its own docstring (`:291-295`) states it is
  **not** gated on Q-2: Q-2 is scoped to the decoder consuming a caller-built
  cache (`DGemmaDenoise`/Phase-4 drive body), not the encoder's own call path,
  "already exercised by every existing `run_diffusion` call today."
- `ROADMAP.md:62-67` — the only place the liveness gate is named: "the
  fixed-seed encoder-text sweep (byte-identical canvases predicted pre-Phase-4)
  is the liveness gate." This protocol gives that sentence a runnable shape.
- `gh issue view 59` §5 / evidence comment (2026-08-03) — physical preconditions
  (GPU free of llauncher tenants, weights cached, pack loadable) and the proven
  live regime: bf16 load, RTX-8000, CPU-spill (the encoder docstring names
  "bf16 CPU spill" as one of three regimes it runs in and is the one with actual
  live E2E evidence — 5 passed/1 xfailed, 2026-08-03, main@f04688f). No live
  evidence exists in-record for AutoRound INT4 driving `run_diffusion` end-to-end
  through the ComfyUI node surface (INT4 evidence in #131/#183/#185 is a
  different code path — bare `diffusion-cli`/transformers probes, not this
  pack's `run_diffusion`). **[PROTOCOL-CHOICE]** this smoke uses bf16 CPU-spill,
  the only combination with a live E2E precedent through this pack's own entry
  point, not "the proven combination" in the abstract — named here because the
  prompt's phrasing implied one exists per-quant; the record only proves it for
  bf16.
- `gh issue view 62 --comments`, 2026-08-04 bracket-opening comment — Phase 2
  (PR #98) and Phase 3 (PR #102) landed; Phase-4 engineering preconditions
  (#187, #207/#209) closed; outstanding gate is exactly this smoke; and the
  comment itself names the design gap this protocol closes: "the fixed-seed
  encoder-text sweep liveness gate... its protocol (seeds, sample size,
  pass/fail threshold) will be specced and banked on this issue before the GPU
  window is requested." This artifact is that spec; banking it back onto #62 is
  the last step of this window, not a side effect. **Phrasing correction:** the
  same #62 comment calls the OUT-3 stamp at `dgemma/loop.py:589,602` "dead code
  today" — that phrasing is imprecise in the same way the gate flagged for this
  protocol; the stamp is LIVE code with an unreachable `kv_cache is not None`
  branch (see §1 observable #4). This protocol's phrasing supersedes the #62
  wording on that point.
- `gh issue view 131 --comments`, "Rung-1 measurement plan" (2026-08-03 comment,
  final one) — 5 named measurements: (1) visibility tax (entropy-readback
  on/off step time, #24427 path), (2) #24423 visual-server frame capture vs
  Tier-0 contract, (3) conformance run (same prompt/seed/knobs across CDG /
  #24423 / #24427 — tests the self-conditioning-fidelity H0), (4) determinism
  probes (seed handling, both PRs vs CDG), (5) Q4_K_M tok/s on sm_75 (Turing
  audience datapoint). Rung-1 is a **separate GGUF-engine probe**, not this
  ADR's KV-cache smoke — sharing the GPU window is a scheduling convenience
  (§3 below), not a shared H0.

---

## 1. Q-2 smoke protocol (ADR Open Q #1)

**Runnability floor (read first).** The with-cache observables below are checkable
only against the skeleton-under-test — the Phase-4 drive body that consumes an
injected cache. Under today's door (`dgemma/loop.py:422-432`, unconditional raise),
no with-cache `run_diffusion` call reaches decode; the no-cache arm is the only
fully-runnable-today arm. This §1 protocol pre-registers what to observe once the
skeleton is under test; it is not runnable end-to-end against `main` as it stands.

### H0 (stated verbatim from the ADR, before observation)

> The decoder can be driven with a caller-built cache the pipeline didn't
> create; position/mask math holds on real weights.

Decomposed into two falsifiable sub-claims the smoke actually observes:

- **H0a (numerical):** a `DGemmaDenoise`-equivalent run given a `KVCache` minted
  by a prior `encode_sequence(prompt_A)` call produces a canvas conditioned on
  `prompt_A`'s content — not garbage, not silently ignoring the cache, not a
  crash from a position/mask mismatch.
- **H0b (mechanical):** the run completes end-to-end (no exception from
  attention masking, position-id arithmetic, or cache-shape mismatch) across
  the full block loop, not just a single forward pass.

### Run matrix

| Axis | Value | Basis |
|---|---|---|
| Model / quant | bf16, `google/diffusiongemma-26B-A4B-it` | only combination with live E2E precedent through `run_diffusion` (§0) |
| Hardware | RTX-8000 (this host), CPU-spill accepted | CLAUDE.md grounded fact; #59 2026-08-03 evidence |
| `gen_length` | 256 | CLAUDE.md "Local run defaults" |
| `num_inference_steps` | 48 | CLAUDE.md "Local run defaults" (`max_steps=48`) |
| `entropy_bound` | 0.1 | CLAUDE.md grounded default |
| `t_min` / `t_max` | 0.4 / 0.8 | CLAUDE.md grounded default |
| `confidence` | 0.005 | CLAUDE.md grounded default |
| `thinking` | `False` | **[PROTOCOL-CHOICE]** — avoids compounding with #9's open thinking/converged-STRING contradiction (open, xfail'd in the E2E battery); Q-2 tests the KV-cache seam, not #9 |
| Seeds | `{7, 13, 21}` (N=3) | **[PROTOCOL-CHOICE]** — minimal N to distinguish "always mechanically sound" from "one lucky seed"; sized down from the liveness sweep's N (§2) because Q-2 is pass/fail-on-mechanism, not a distributional claim |
| Encoder prompt (cache donor) | `"The lighthouse keeper counted every wave before dawn."` | **[PROTOCOL-CHOICE]** — one fixed, semantically dense, prompt-injection-free sentence; content is arbitrary, chosen only to be unambiguously checkable by a cold reader in the observable below |
| Comparison arm | same seed, same prompt, **no** `kv_cache=` (today's supported path) | isolates the cache-injection effect from ordinary seed variance |

Total runs: 3 seeds × 2 arms (with-cache, without-cache) = **6 runs**, plus a
0th run to mint the donor cache once per seed (`encode_sequence(prompt_A)`,
reused across both arms at that seed) = 3 mint calls. **9 total decoder-relevant
calls.**

### Exact observable per run

For each of the 6 `run_diffusion` calls, record:

1. `status` — did the call return `(str, CanvasState, CanvasTrace)` or raise?
   If it raises, capture the full exception type + message + traceback.
2. The returned `STRING` (decoded output).
3. `CanvasState.converged`, `CanvasState.committed_fraction`,
   `CanvasState.steps_used` — the same internal-consistency triple the E2E
   battery's S2 assertion already checks (`tests/e2e/test_battery.py`,
   confirmed pattern from #59 evidence).
4. `CanvasTrace.injected_cache_provenance` — for the with-cache arm, must be
   non-`None` and must match the donor cache's `Provenance` (mint sequence /
   `model_repo_id` / `tokenizer_fingerprint`) exactly; for the no-cache arm,
   must be `None` (today's default, unchanged). This is OUT-3's own contract
   (`decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md:276-291`). Precise
   status of the stamp: the OUT-3 line at `dgemma/loop.py:589,602`
   (`injected_cache_provenance=kv_cache.provenance if kv_cache is not None else
   None`) is **LIVE code**, not dead — it executes on every completed and
   cancelled run and stamps `None` whenever `kv_cache is None` (i.e. on every
   no-cache run today). What is unreachable is only its `kv_cache is not None`
   *branch*, foreclosed upstream by the `:422-432` door that raises before
   decode. This observable is therefore exercisable only once the skeleton drive
   body runs a with-cache call; when it does, verify the rewrite **preserved**
   the stamp (that the non-`None` branch now actually reaches `_build_result`
   with the injected provenance intact) rather than dropping it in the door
   replacement.
5. Wall-clock per run (informational, not gating).

### Typed outcomes

- **PASS (gate opens; drive body may proceed past skeleton).** Split into a
  mechanical predicate that GATES the verdict and a human observation that is
  RECORDED but does not gate:

  **(a) GATING mechanical predicate — all of the following, per run, no
  averaging:** all 6 runs complete with `status=success` (no exception); for all
  3 with-cache runs, `injected_cache_provenance` is present and correct (matches
  the donor `Provenance` per observable #4); for all 3 with-cache runs, the
  returned STRING is **not byte-identical** to that seed's no-cache-arm STRING
  **and** is non-empty. Byte-difference-from-the-no-cache-arm is the mechanical
  proxy for "the cache actually reached the decoder": a silently-ignored cache
  produces output byte-identical to the no-cache arm at the same seed, exactly
  the accepted-and-ignored failure `dgemma/loop.py:414-421`'s comment names.
  `CanvasState`'s internal triple is self-consistent in all 6 runs (independent
  of #9 — `thinking=False` forecloses that contradiction here). This predicate
  is fully decidable from recorded fields; it is what opens the gate.

  **(b) RECORDED, not gating — human coherence observation:** whether the
  with-cache STRING reads as coherent text plausibly conditioned on the donor
  prompt's content. This is a soft human judgment, banked in the artifact
  alongside the run, but it does NOT gate this smoke — coherence-of-generation
  is the drive body's own design-gate concern (per §4, the Phase-4 body carries
  its own Opus review), not the KV-cache-seam liveness question Q-2 resolves.
  Record it so the drive-body gate inherits a datapoint; do not let a
  subjective coherence call flip the mechanical PASS above.
- **FAIL (gate stays closed; drive body blocked at skeleton; finding banked on
  #62 and, if it reveals a design gap rather than an implementation bug, on the
  ADR's Open Questions):** any with-cache run raises from position/mask
  arithmetic (attention shape mismatch, index-out-of-range on position_ids,
  RoPE-parameter mismatch) — this falsifies H0b directly. Or: all 6 runs
  complete, but a with-cache STRING is **byte-identical** to the no-cache arm at
  the same seed, or is empty, or `injected_cache_provenance` is absent/incorrect
  (cache accepted but not actually driving the decoder — the trust-and-degrade
  failure, now observed at the decoder rather than merely warned against at the
  door) — this falsifies H0a via the gating mechanical predicate (a) above.
  (Incoherence per observation (b) is recorded and referred to the drive-body's
  own gate; it is NOT a Q-2 FAIL on its own — see PASS (b).) Either
  gating sub-failure is a FAIL; do not average across seeds to
  paper over one bad seed — 3/3 with-cache runs must independently satisfy the
  gating predicate PASS (a) above (**[PROTOCOL-CHOICE]**: no partial-credit threshold; N=3 is
  small enough that "2 of 3 passed" is not evidence of anything, it is
  underpowered — treat any single clean falsification as gate-closing and
  re-run only after a fix, not as a statistics problem).
- **BLOCKED (infra; re-schedule, no verdict):** GPU unavailable / weights not
  cached / llauncher tenant contention (per #59 §5) / OOM under CPU-spill before
  any decoder code runs / a crash traceable to something outside the KV-cache
  seam entirely (e.g. an unrelated import error). Record the blocking condition
  on #62 and re-open the window; do not count a BLOCKED run as a FAIL data
  point.

---

## 2. Fixed-seed encoder-text sweep (liveness gate)

`ROADMAP.md:62-67` names this as the thing gating Phase 4, predicting
byte-identical canvases pre-Phase-4. Restated as a falsifiable claim:

### H0 (falsifiable, stated before observation)

> Before the Phase-4 drive-body lands, passing a `kv_cache=` built from
> different encoder texts has **zero observable effect** on the committed
> canvas, because the current code path (`dgemma/loop.py:422-432`) rejects
> every `kv_cache is not None` call before any scheduler/pipeline
> construction — the run never reaches a decode step with the cache attached
> in a way that could vary the output. Equivalently: **runs are not
> reachable at all today** (every call raises `NotImplementedError` at the
> ingress door), so the "byte-identical canvases" claim is trivially true in
> the strongest possible sense — not "the outputs happen to match" but "no
> output is ever produced to compare."

This reframes the ROADMAP's "byte-identical canvases predicted pre-Phase-4" as
a **liveness check on the current fail-loud door**, not a numerical-equivalence
experiment — there is no decode path today for varying encoder texts to
influence. The numerical-equivalence reading is not merely less apt but
unbuildable pre-Phase-4 — encoder text reaches the decoder only via
`kv_cache=`, which raises before decode; there is no cache-free path for varying
encoder text to condition output. This sweep's job is to confirm that prediction
is what actually happens (the door truly is unconditional and text-invariant) before the
drive-body rewrite removes it, so any post-rewrite divergence is legibly new
behavior rather than a pre-existing gap being uncovered.

### Protocol

- **Seeds:** N=3, `{7, 13, 21}` (same set as §1, for economy — **[PROTOCOL-
  CHOICE]**: reusing the Q-2 seed set avoids inventing a second arbitrary set
  with no stronger justification).
- **Encoder texts:** M=4 — **[PROTOCOL-CHOICE]**, chosen for lexical/length
  diversity so a hypothetical leak would be visible along more than one axis:
  1. `"The lighthouse keeper counted every wave before dawn."` (§1's donor,
     reused for cross-check)
  2. `""` (empty string — degenerate/edge case)
  3. `"Quarterly revenue exceeded forecast by twelve percent in the northern region."` (long, distinct vocabulary)
  4. `"xyzzy plugh"` (out-of-distribution tokens, tests tokenizer-edge handling)
- **Sample size justification:** N=3 × M=4 = 12 calls. This is a liveness check
  (does the door reject uniformly, yes/no), not a distributional estimate — 12
  is enough to rule out "it happens to pass for the one text we tried" while
  staying inside a single GPU window's budget alongside §1 and §3
  (**[PROTOCOL-CHOICE]**: no power calculation performed; a liveness predicate
  is binary per call, and 12/12 agreeing is already a strong uniformity signal
  at this cost).
- **Procedure per (seed, text) pair:** mint a cache via `encode_sequence(text)`,
  call `run_diffusion(..., seed=seed, kv_cache=<that cache>)` with the §1 knob
  defaults, and record the outcome.
- **Comparison method:** record `(status, exception_type_or_None)` per call.
  Compute `sha256` of the tuple `(status, exception_type)` per call (not of a
  committed canvas — no canvas is ever committed under the current door, so
  "hash of committed canvas ids" as literally stated in the prompt does not
  apply pre-rewrite; noted here as a correction to avoid inventing a canvas
  hash for calls that never reach canvas construction). All 12 hashes must be
  **identical** to each other (all `NotImplementedError`, same exception type)
  for the sweep to confirm the door's current text-invariance.
- **Thresholds (fixed in advance):**
  - **PASS (liveness confirmed; safe to proceed to the drive-body rewrite with
    a clean pre-rewrite baseline):** all 12/12 calls raise the identical
    `NotImplementedError` from `dgemma/loop.py:424-432`, regardless of seed or
    encoder text — confirming the door is unconditional and that no encoder
    text can currently produce a differing canvas (because none produces a
    canvas at all). This is the ROADMAP's prediction, confirmed.
  - **FAIL (a design-record assumption is wrong; halt before the rewrite,
    bank on #62):** any call does NOT raise, or raises a different exception,
    or (if somehow a canvas is produced) two different encoder texts at the
    same seed produce byte-identical committed canvases where the cache
    *should* differentiate them once the door is open — that would mean the
    injected cache is silently inert even where the code claims to consume
    it, a live instance of the accepted-and-ignored failure.
  - **BLOCKED:** same infra causes as §1.
- **Note for AFTER Phase 4 lands:** this sweep's H0 is specific to the
  *pre-rewrite* fail-loud door. Once the drive body replaces
  `dgemma/loop.py:422-432`, the liveness gate's claim changes shape entirely —
  different encoder texts SHOULD now produce different canvases (that is the
  feature landing). Re-running this exact sweep post-rewrite and expecting the
  same PASS condition would be a category error; flagging so the gate is not
  reused unmodified as a regression test without restating its H0.

---

## 3. GPU-window runsheet

One operator-schedulable window. Precedent for wall-clock estimates: the
2026-08-03 E2E battery window (issue #59 evidence comment) — model load +
fixture setup amortized once (~16s), then per-scenario calls in the 35–62s
range at `gen_length`≈default/`num_inference_steps` in that same ballpark.
Only estimates traceable to a cited record are given; anything else is marked
**[NO ESTIMATE — record silent]** rather than invented.

### Setup

1. **llauncher swap-out** — stop/swap `localhost:8081` (chat model) and
   confirm `localhost:8082` (embeddinggemma, kept-resident per #59's
   2026-08-03 precedent) is the only tenant, via the `llauncher` MCP
   start/stop/swap/status tools (per `~/.claude/CLAUDE.md`'s environment
   section — this is infra coordination, not a repo-code step).
2. **`nvidia-smi` baseline** — confirm VRAM free matches the #59 2026-08-03
   pattern (embedding service ~1.4GB resident, nothing else >2GB) before
   proceeding. Bank the baseline text in the run's artifact folder (§
   "Artifact-banking checklist" below).
3. **Weights-cached check** — `scan_cache_dir()` confirms
   `google/diffusiongemma-26B-A4B-it` present (per #59 §5.3 / 2026-08-03
   evidence: ~51.7G at that check). SKIP-not-ERROR if absent, per the E2E
   battery's own precondition discipline (`tests/e2e/conftest.py`) — do not
   invent a different failure mode for this window.
4. **Pack/tree sync** — confirm the working tree used for this window is at
   the commit that carries the fail-loud door referenced in §0
   (`dgemma/loop.py:422-432`) and the current `validate_kv_cache_ingress`
   (V1–V6). If the drive-body rewrite has already landed by the time this
   window runs, §2's liveness-gate H0 is stale — check the ADR's Open
   Questions / issue #62 status before running §2, and re-derive its H0
   against the landed code rather than running it unmodified (per §2's
   trailing note).

### Ordered run list

| Order | Item | Purpose | Est. wall-clock | Basis |
|---|---|---|---|---|
| 1 | §1 Q-2 smoke (6 `run_diffusion` calls + 3 mint calls) | resolve ADR Open Q #1 | **[NO ESTIMATE — record silent on with-cache-arm timing since the arm doesn't exist yet; no-cache arm ≈ 35-62s/call per #59's S2/S1 pattern at comparable knobs]**, so ≈3-6 min for the no-cache arm alone, with-cache arm unknown until it runs | #59 2026-08-03 evidence table |
| 2 | §2 liveness sweep (12 calls, all expected to raise before scheduler construction) | confirm pre-rewrite door invariance | **[PROTOCOL-CHOICE estimate]** seconds per call (raises before model-heavy work — `validate_kv_cache_ingress` runs first per `dgemma/loop.py:391-432`'s ordering, itself lightweight tensor-shape checks, not a forward pass) — a small multiple of 12, not minutes | inferred from the code path's ordering, not a cited timing record; flagged as inferred, not measured |
| 3 | #131 rung-1 inference/telemetry probe (5 measurements per the 2026-08-03 comment: visibility-tax, #24423 frame capture, conformance run across CDG/#24423/#24427, determinism probes, Q4_K_M tok/s on sm_75) | separate GGUF-engine probe, riding this window for GPU-tenancy economy only | **[NO ESTIMATE — record silent]**; rung-0 build times (CPU 1m27s, CUDA 9m46s) are build, not inference, and don't bound inference wall-clock | #131 rung-0 comment (2026-08-03) |

Order rationale: §1 and §2 both touch `dgemma/loop.py`'s current KV-cache door
and share the same model load — run them back-to-back to amortize the one
model load this window pays for (per the CLAUDE.md-noted standing constraint
that DGemma and the llauncher chat model can't coexist, the load itself is the
expensive, single-pay step). #131's rung-1 probe is a **different model
entirely** (a GGUF checkpoint through `llama-diffusion-cli`, not this pack's
loaded transformers model) — it does not share the load, so it is ordered last
and could equally run in a separate window if this one runs long; naming it
last here reflects "don't let a separate probe block the ADR gate," not a
dependency.

### Teardown

5. **llauncher swap-back** — restore whatever `localhost:8081` was serving
   before step 1, via the same MCP tools.
6. **`nvidia-smi` confirm-clean** — VRAM returns to the pre-window baseline
   (minus the resident embedding service, unchanged throughout, per #59
   convention).

### Artifact-banking checklist (per #101's draft convention — cited, not
ratified; see the header note)

- **Location:** `docs/experiments/2026-08-0X-adr-cdg-012-q2-smoke/` (dated on
  the day the window actually runs), following the existing directory-per-
  entry pattern already in `docs/experiments/` (confirmed present:
  `2026-07-14-...`, `2026-07-30-autoround-unified-path-split-check`, etc.) —
  this is today's de-facto convention, not #101's proposed `runs/` JSONL
  subdirectory (that subdirectory does not exist in any entry today; #101's
  own comment 1 confirms zero `.jsonl` files in-tree as of its filing). Using
  the JSONL shape here would be inventing ratification #101 hasn't received;
  using the narrative-`.md`-plus-attachments shape matches what every existing
  entry actually does.
- **Contents to bank per run (§1 and §2, all 18 calls):** the exact observable
  fields listed in §1 (`status`, STRING, the `CanvasState` triple,
  `injected_cache_provenance`, wall-clock) and §2 (status/exception hash) —
  raw values, not a narrative summary of them, per #101's own complaint that
  narrative-only banking is the failure this convention exists to fix. Include
  seed, encoder text (or its identifying index 1-4), and arm (with-cache /
  no-cache) per row so each row is independently re-derivable.
- **Host identity tag:** per the 2026-07-30 floor-amendment proposal on #101 —
  canonical host name + GPU model/VRAM/compute-capability on every banked
  result (`inference-host`, RTX-8000, 48GB, sm_75, for this window). Not yet
  ratified as a hard requirement, but cheap to include and this window's
  results would otherwise be exactly the kind of untagged-table risk that
  proposal names.
- **#131 rung-1 artifacts:** per that thread's own convention once it runs —
  this protocol does not redefine #131's banking shape, only schedules its
  execution inside this window (§3 rationale).
- **Verdict comment:** after the window, post a single comment to
  https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62 stating
  the typed outcome (PASS/FAIL/BLOCKED) for §1 and §2 each, linking the
  `docs/experiments/` artifact directory — this is the bracket-opening
  comment's own ask ("specced and banked on this issue before the GPU window
  is requested" — the spec is this file; the banking-back is this comment,
  owed after the window, not before).

---

## 4. Non-goals

This window does **not** decide:

- **Pin choice for the GGUF engine** (`#24423` vs `#24427`) — explicitly
  deferred to the future engine ADR per #131's 2026-08-03 "Reframe 2" comment
  ("Pin choice... is deferred to that ADR; rung-0 de-risks against #24423 per
  the Unsloth card's known-good path"). Rung-1's conformance-run measurement
  (§0, item 3) produces *evidence* toward that ADR; it does not constitute the
  decision.
- **Engine-ADR content** (the "more-complete PR built on the existing
  community-hardened base" reopened in #131's "Named reopen" — visibility
  delta, protocol shape, abandonability mitigation). Rung-1 is explicitly
  scoped by that same comment as evidence-gathering ("rung 1 (inference probe;
  measures what the existing CLI actually emits per-step — this bounds the
  visibility delta)"), with the comment's own closing line: "All decisions
  beyond rung 0–1 execution are operator-gated."
- **Rung-4 packaging** (bespoke llama.cpp binaries in the release package,
  CI-built from the owned pin — named in `ROADMAP.md:54-55` as a separate
  next-bracket item, "(b) Bespoke llama.cpp binaries in the release package —
  #131 rung 4"). Nothing in this window's runsheet builds or ships a binary;
  rung-0's build already happened (2026-08-03, banked on #131) and is not
  repeated here.
- **Whether the Phase-4 drive body's actual implementation is correct** beyond
  what §1's typed outcomes observe. A PASS licenses the drive body to
  *proceed past its skeleton* (the ADR's own resolution-trigger wording,
  `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md:621-623`) — it is
  not a substitute for that implementation's own Opus design-gate review once
  built, per this repo's strict-waterfall process convention (CLAUDE.md
  "Process conventions").
- **The `CANVAS_STATE`-under-tier-2-cache tension** (ADR Open Question,
  `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md:642-660`) — tier-2
  caches are out of scope for this smoke entirely (§1's run matrix uses only
  tier-1 caches, minted via plain `encode_sequence`, no surgery ops); this
  window produces no evidence either way on that open question.
- **The DV.1 coverage-gate mechanism choice** (per-module 100% floor
  expression, `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md:662-677`)
  — a test-suite mechanism question, orthogonal to this live-weights smoke.

---

## Gate findings resolved (2026-08-04)

Independent design-gate review returned PASS_WITH_FINDINGS. The four resolved
recommendations, each recorded as a decision applied to this artifact:

1. **§2 reframe — buildability of the numerical-equivalence reading.** Finding:
   the reframe called the numerical-equivalence reading "less apt" but did not
   state it is *unbuildable* pre-Phase-4. Resolution: added the decisive sentence
   to §2's reframe paragraph — encoder text reaches the decoder only via
   `kv_cache=`, which raises before decode (`dgemma/loop.py:422-432`), so there
   is no cache-free path for varying encoder text to condition output. The
   equivalence experiment cannot be built pre-Phase-4, not merely that it is a
   worse framing.

2. **§1 PASS predicate — split gating mechanics from soft coherence.** Finding:
   "coherent text materially distinguishable" conflated a decidable mechanical
   check with a subjective human judgment inside one gating bullet. Resolution:
   split PASS into **(a)** a GATING mechanical predicate (status=success AND
   `injected_cache_provenance` present-and-correct AND with-cache STRING
   not-byte-identical-to-no-cache-arm AND non-empty, per run, no averaging) and
   **(b)** coherence as a RECORDED-but-not-gating human observation, banked for
   the drive body's own Opus design gate. FAIL predicate re-aligned to match:
   incoherence alone is no longer a Q-2 FAIL.

3. **§1 observable #4 — drop the "dead code" framing.** Finding: calling the
   OUT-3 stamp "dead code" is inaccurate — `dgemma/loop.py:589,602` is LIVE code
   that stamps `None` on every no-cache run; only its `kv_cache is not None`
   branch is unreachable under today's `:422-432` door. Resolution: restated
   observable #4 to say the stamp is live, its non-`None` branch is what is
   currently unreachable, the observable is exercisable only against the skeleton
   drive body under test, and the verification is that the rewrite **preserved**
   the stamp. Also added a runnability-floor statement at the top of §1: the
   with-cache observables are checkable only against the skeleton-under-test; the
   no-cache arm is the only fully-runnable-today arm.

4. **#62 phrasing cross-check (cheap verification the gate requested).** Verified
   via `gh issue view 62 --comments`: the 2026-08-04 bracket-opening comment does
   call the OUT-3 stamp "dead code today." Resolution: added a one-line note in §0
   that the #62 phrasing is imprecise in the same way, and that this protocol's
   phrasing (stamp is LIVE, only the with-cache branch is unreachable) supersedes
   it on that point.
