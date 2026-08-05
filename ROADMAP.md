# ROADMAP — where ComfyUI-DiffusionGemma is headed

ComfyUI-DiffusionGemma exposes **DiffusionGemma** — text generation by
uniform-state discrete diffusion — as an instrumentable ComfyUI graph you can
watch, instrument, and take apart. For what the pack *is* and what already works,
see [`README.md`](README.md).

**What this file is:** the forward-looking view, in two tracks — engineering
(the seam work that makes expansion cheap) and research (the liquid-phase
program). It is deliberately pointer-heavy and holds no content of its own:

- **[`VISION.md`](VISION.md)** holds the *why* — the questions the instrument was
  built to ask, each tagged `[established]` / `[hypothesis]` / `[open]`.
- **[`decisions/`](decisions/)** holds the *decided* — the ADRs, the load-bearing
  choices and their trade-offs.
- **`plan.md`** is a **tombstone** — retired 2026-07-31 (operator direction);
  its phases P0–P3 per-phase PASS record is archived on closed issue #199,
  not in a live working-tree file. This file does not restate it.

Two independent `R`-namespaces appear below and must not be conflated:
**Track A's R1–R6 are engineering refactors** (issue #35); **Track B's R0–R6 are
research rungs** (the liquid-phase program). Same letter, different ledgers.

---

## Roadmap (recorded 2026-08-05, operator-set)

**Spine, stated once so it can't be missed:** the operator priority for this
bracket is **get a working `kv_cache` channel out** — end to end, both
surfaces — **without shipping regression or incomplete speculative ideas
alongside it.** Everything below is sequenced against that spine: what's
closed on it, what's guarded-open on it, and what's explicitly parked off
it. Nothing parked below gained a new promise in this pass.

**Version namespace note.** The "0.5.0 AutoRound INT4 release" attempt died
unshipped — no version literal, tag, or registry artifact ever bore the name
(post-mortem: [`docs/postmortems/2026-07-31-0.5.0.md`](docs/postmortems/2026-07-31-0.5.0.md)).
The name is therefore reused below. 0.4.2 remains available as the stabilization
tag.

- **0.5.0 — the refactor version, and only that** — **shipped 2026-08-03**, tag
  `v0.5.0` @ `c4beb3e`.
  [ADR-CDG-018](decisions/adr-cdg-018-decompose-loop-py.md): `dgemma/loop.py`
  decomposed into `config`/`compat`/`capture`/`excision` behind a re-export
  facade; pure refactor proven by golden-trace byte-identity + AST comparison;
  live-verified 4/4 on real weights (#129, closed). The registry still serves
  0.4.2 — a 0.5.0 registry publish is a separate operator call. Post-tag on
  `main`: #119 offload-aware tied-weights guard (`f04688f`), rides the next tag.
- **Composition/KV bracket — CLOSED.** The Encode/Denoise visibility
  disposition that this roadmap previously carried as "operator call
  pending" is resolved by supersession chain, in order:
  1. [ADR-CDG-024](decisions/adr-cdg-024-prompt-under-injection-composition.md)
     (independent encoder context + templated denoiser turn, prefilled onto
     the injected cache) — **Accepted**, ratified at `b01a992` (PR #256,
     2026-08-05) — supersedes #248's interim prompt+KV_CACHE exclusivity
     invariant by name.
  2. **#257 implements ADR-CDG-024** — merged `0b6bd1a` (PR #262,
     2026-08-05): the drive body composes `prompt` + `kv_cache` instead of
     ignoring `prompt` under injection.
  3. **Live-verified at AIO parity, operator-field-confirmed** during PR
     #262's acceptance session (2026-08-05): the banked #245 stall trace
     converges under the composed drive body, matching the AIO control's
     healthy range — the falsifiable acceptance ADR-CDG-024 §5 named.
  4. **#266 — run-log injection-provenance fields** — merged `ad8a8e7`
     (PR #268, 2026-08-05): closes the attribution gap the acceptance
     session hit twice (KV-path vs AIO indistinguishable in a banked log).
  The kv_cache channel's ComfyUI-surface half is therefore **shipped and
  verified**, not pending — #103's original "operator call pending" framing
  no longer applies to this bracket.
- **Remainders — guarded, not silent, not "just a bugfix."** PR #262's
  Opus design-gate review surfaced two defects in the composed path,
  **both fenced by the interim ingress guards of PR #270 (merged `921f944`,
  2026-08-05)** rather than shipped open or rushed to a fix that skips
  design. This
  guard-then-fix sequencing *is* the "no regression, no incomplete ideas"
  mechanism for this bracket — name it as such when reading the two rows
  below:
  - **#263 — block>0 splice offset is short by the prefilled-turn length**
    under a composed (`prompt`+`kv_cache`) multi-block run
    (`gen_length > canvas_length`). Not a plain bugfix: the fix needs a
    small design confirmation first (whether/how block>0 re-encode should
    condition on the prefilled turn at all is ADR-CDG-024-adjacent,
    undecided). Guarded today by `reject_multi_block_composed_prefill`
    (PR #270, merged `921f944`) — single-block composed runs and multi-block
    pure injection remain allowed; only the uncovered shape is rejected at
    ingress. Retires when the design-confirmed fix lands.
  - **#265 — `prefill_templated_turn` mutates the injected `KVCache` in
    place**, so a caller reusing the same cache object (observed live:
    ComfyUI node-result caching reusing an unchanged `DGemmaEncode` output)
    silently inherits a prior run's prefilled turn. Architectural, not a
    plain bugfix: sits against the stateless-core rule (ARCHITECTURE.md
    rule 6) and needs an ADR-CDG-024 amendment to pick a shape (prefill-onto-
    copy vs. defensive re-mint/invalidation vs. documented caller contract)
    before implementation. Guarded today by ingress check **V7** in
    `validate_kv_cache_ingress` (PR #270, merged `921f944`) — rejects when
    a cache's actual `get_seq_length()` exceeds its minted
    `cumulative_length`. Retires when the amendment-confirmed fix lands.
- **Next implementation bracket — #259, MCP KV-path parity.** Design ratified
  as [ADR-CDG-025](decisions/adr-cdg-025-mcp-kv-cache-handle-registry.md)
  (Accepted, merged `d6b9991`, PR #258, 2026-08-05); issue #259 is
  `auto:fix pri:now`, unblocked. This is the kv_cache channel's **second
  surface** — an `encode`/`generate(kv_cache_id=...)` MCP door over a bounded,
  model-scoped cache-handle registry in `StateManager` — and its completion
  is what makes "a working kv_cache channel" true across both surfaces, not
  just ComfyUI's. Repairs the rule-2 inversion tracked by #103.
- **Parked explicitly — not on the KV-channel line, no new promises added.**
  #260 (Tier-2 cache surgery: splice/ablate/scale + disk serialization) is
  design-first and sequenced behind #257/#259 — untouched by this pass. The
  research-track probes (Track B below, S/C/X/G/F-tracks) stay in their
  existing research section, unchanged.
- **Quant — re-sequenced, gated.** #264 (`_assert_tie_integrity` crashes on
  AutoRound's `QuantLinear`, no `.weight` attribute — every int4-AutoRound
  load dies post-load) sits inside #211's quantized-engine bracket and
  **gates** it: the AutoRound lane must load end-to-end before any
  alternate quant engine (e.g. #211's GPTQModel scoping) is worth
  considering. **bf16 (`quant="none"`) remains the only working load path**
  today — see #269 (merged, `0d348ea`) for the full corrected quant/GGUF
  ground state on `README.md`/`AGENTS.md`.
- **#131 — GGUF rung-1: parked until an operator-scheduled GPU window.**
  Read as parked, not in-flight: rung-0 build is green (`c3fb972`, pinned
  upstream draft-PR branch off ggml-org/llama.cpp#24423, not owned code),
  but the rung-1 inference/telemetry probe has not run. [ADR-CDG-020](decisions/adr-cdg-020-gguf-engine-sourcing-pinned-pr-branch.md)
  stays **proposed** — ratification and the #24423-vs-#24427 pin choice
  wait on that probe's readback, which waits on the GPU window.
- **Salience debt — reconciled as sibling ungated items.** #243 (`dgemma/loop.py`
  drifted to ~908 lines / ~12k tokens, past ADR-CDG-018's 5k threshold) and
  #219 (decompose `model.py` + evacuate prose from `loop.py`/`types.py`,
  broader salience-ceiling pass) are both open, both ungated, tracked at
  their own labeled priorities (#243 `pri:later`, #219 `pri:next`) — no
  topology-bracket dependency on either; the prior framing that gated #219
  behind the #138 topology bracket named no actual technical dependency and
  is retired.
- **Next bracket — topology (version minted at opening):**
  [ADR-CDG-019](decisions/adr-cdg-019-mcp-as-contract-topology-remediation.md)
  (accepted 2026-08-03; sequenced after #129; `dgemma_mcp/` rename per gate
  finding I1) → topology remediation (#138), import-gate consolidation (#57).
  The interrupt fix (#140) is **closed** (2026-08-05) — no longer rides here.
- **Lifecycle proposal pending:**
  [ADR-CDG-021](decisions/adr-cdg-021-per-surface-vram-tenancy-ownership.md)
  (per-surface VRAM tenancy — MCP/transformers lane may OOM fail-loud;
  ComfyUI surface integrates `comfy.model_management`; grown from an operator
  observation, #229) — proposed, operator disposition open; binds the #138
  bracket to in-process consumption if accepted.
- **The seam (post-refactor) — done.**
  [ADR-CDG-012](decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md)
  Phase 4 landed `ac3c832` (PR #242): the Q-2 real-weights de-risk smoke
  PASSed (#62), and the decoder is now genuinely driven off injected caches,
  composed with a templated prompt turn per ADR-CDG-024 above. #47
  (known-provenance cache-perturbation) is **closed** (2026-08-05); its
  Tier-2 remainder is tracked by #260, parked above.
- **Release line — v0.5.2** (operator ruling: "0.5.0 was refactor and this
  is activating latent functionality"). PR #270 (the #263/#265 interim
  guards) merged (`921f944`, 2026-08-05); the v0.5.2 line now waits only on
  #163's release gate: a seat-run fresh-install + live smoke on the dev
  host before the operator sees a version. #196's version-bump/tag-timing
  question resolves in that same act (mint the literal and the tag at gate
  PASS). The release's known limitations are exactly the guarded #263/#265
  pair above plus the standing quant/GGUF state (#264-gated AutoRound,
  #131-parked GGUF, bf16-only working path).
- **#175 — in-UI node explanations.** PR #270's gate PASS (merged `921f944`)
  assessed #175 as **partially delivered** by its tooltip/`DESCRIPTION`
  build-out (the `DGemmaDenoise` prompt/kv_cache composition-vs-exclusivity
  language, the guard-rejection naming) — issue rescoped (2026-08-05) to
  the remainder: DGemmaRunLogWriter, the gen_length tooltip, and
  DGemmaTokenTrace.
- **Research arc.** The capture instrument is complete (Tiers 0–2 + display
  consumers, [ADR-CDG-014](decisions/adr-cdg-014-frame-capture-discipline.md)).
  Queued: #186 (bf16-vs-INT4 trace comparison), #28 (flagship global-constraint
  problems), #3 (mot-juste probe), #7 (commit-front morphology), #118
  (hidden-state steering door), #115 /
  [ADR-CDG-017](decisions/adr-cdg-017-neighborhood-remelt-kernel.md)
  (neighborhood remelt). Gated on #59's E2E live preconditions and card tenancy.
- **Standing constraint.** DGemma INT4 whole-fit (30 GiB floor) and the resident
  llauncher chat model (~35 GiB) cannot coexist on the RTX-8000. Every live
  bracket schedules its llauncher swap explicitly rather than discovering
  occupancy at load time.

The tracks below are the standing two-track view this near-term roadmap draws
from; their per-rung status is maintained independently.

---

## Track A — Engineering (seam work + topology)

The 2026-07-13 Opus-tier architecture review (issue #35) returned
**"needs targeted refactors first — not a structural problem":** the CDG-003
seam, the fake-pipeline testing discipline, and the native-socket rule all
survive; roughly a week of seam work, not a redesign. Every expansion capability
lands **core-side of the seam**, so CDG-008's MCP surface and any future human UI
inherit it for free.

### The seam cluster (issue #35 + its 2026-07-13 delta comment)

Ordered per the delta pass (**R4 before R1**: the shared fixture lands first so
R1's composition-ordering tests are written against it). One line each, what it
enables:

| Refactor | What it enables | Status |
|---|---|---|
| **R4** — shared fake-pipeline/scheduler fixture in `tests/conftest.py` (N steps, mutable `scheduler.config`, hook-recording model) | Gates testable composition for R1/R5 — the fixture the ordering tests need | **done** — [PR #44](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/44); `tests/conftest.py:fake_pipeline_factory`, self-tested by `tests/test_conftest_fake_pipeline.py` |
| **R1** — callback-composition layer in `dgemma/loop.py`: ordered participants, canvas-write threading, per-participant exception policy, `_FrameCollector` first | Opens the single hardcoded callback slot (F1, **ONE-DOOR**) to the expansion participants — β-renoise, walker, pin, capture — that everything downstream needs | **done** — [PR #45](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/45); `dgemma/composite.py:StepEndComposite`, `tests/test_step_end_composite.py` |
| **R5** — forward-hook lifecycle context manager; invariant "no hook survives a `run_diffusion` call" | Closes hook-leakage across executions (F4, **STATELESS-CORE**) — the per-position heat field installs and tears down cleanly | **done** — [PR #49](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/49); `dgemma/hooks.py:install_logit_shaping_hook`, `tests/test_hook_lifecycle.py` |
| **R3** — diffusers version guard + structural probe (scheduler kwargs, `accepted_index`, `_callback_tensor_inputs`) | Fails loud on a diffusers bump instead of silently reporting a wrong re-derived temperature (F6, **EMIT-CANONICAL**) | **done** — [PR #48](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/48) gate finding F-1; `dgemma/loop.py:_check_diffusers_version`/`_check_diffusers_structure`, `tests/test_diffusers_version_guard.py` |
| **R2** — socket-type mint module + grep-gate test (no inline `DGEMMA_*` literal outside it) | One mint home for socket strings (F2, **ONE-MINT**); lands with/before CDG-008 Phase 1, in `surfaces/comfyui/socket_types.py` | **done** — landed with CDG-008 Phase 1, [PR #53](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/53); `tests/test_socket_mint.py` |
| **R6** — `DiffusionFrame` extension discipline (optional-with-defaults; heavy-field retention policy) | Lets rung-4's heavy `DISTRIBUTION` field ride the frame additively without breaking ADR-CDG-005's small-per-step economy (F3, **EMIT-CANONICAL**); rides research rung R4-observe, analysis functions go to the CDG-008 Phase-3 home | **done** — [PR #66](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/66) (merge `7508113`), issue #61 Phase P-A, ADR-CDG-014; `dgemma/types.py:DiffusionFrame`, `tests/test_frame_capture_discipline.py` |

Sequencing (issue #35, delta-corrected): **R4 → R1 → R5 cluster + R3** before any
research rung lands; **R2** with/before CDG-008 Phase 1; rung-4 analysis behind
CDG-008 Phase 3. R1–R6 above are now landed (verified against
ARCHITECTURE.md's enforcement-surface table and the cited tests).

### The topology move — ADR-CDG-008's five phases

[ADR-CDG-008](decisions/adr-cdg-008-mcp-center-multi-surface-topology.md) (accepted)
adopts an **MCP-center, multi-surface, single-repo** topology: `dgemma/`
(`load_model` + `run_diffusion`) is the one contract, MCP is the base surface,
ComfyUI is one peer surface among others. The published repo name stays
(`IDENTITY⊥ENVELOPE`); only the internal directory vocabulary changes.

| Phase | Move | Status |
|---|---|---|
| **1** | Rename `nodes/` → `surfaces/comfyui/`, move `web/` → `surfaces/comfyui/web/` | **done** — [PR #53](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/53), issue #52; `surfaces/comfyui/`, `tests/test_comfyui_loader_context.py` |
| **2** | Add `surfaces/mcp/` — the base surface over `load_model` + `run_diffusion` (transcribe `semantic-kinematics-mcp`, with the two `STATELESS-CORE` / `ONE-DOOR` corrections) | **done** — [PR #54](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/54); `surfaces/mcp/`, `tests/test_mcp_surface_seam.py` |
| **3** | Relocate analysis out of `dgemma/`'s import graph into a consumer home | **done** — [PR #56](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/56) (merge `e2aefd1`), issue #55; `consumers/analysis.py` |
| **4** | Add the boundary test: base contract imports no analysis (flips the prose-only row to in-force) | **done** — [PR #56](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/56) (merge `e2aefd1`), issue #55; `tests/test_seam.py::test_dgemma_does_not_import_consumers_package` |
| **5** | Rewrite `ARCHITECTURE.md` against the governance template | **done, in two passes.** [PR #37](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/37) landed the initial governance-template rewrite (against the then-old topology); this row's numbering was then reused for CDG-008's actual Phase 5 — the final doc pass closing Track A, flipping every Phase-3/4 row to landed once #56 merged (this ADR's own Phase-5 execution note, above). |

Two ADRs are in ratification, named as drafting specs by issue #35's "required
clauses": **ADR-CDG-010** (givens/constraints — the two-mechanism model: logit
mask shapes *what commits*, canvas re-assertion guarantees *what conditions*) and
**ADR-CDG-011** (per-step control — the declarative-socket / closure-walker split,
units-at-binding, `t_min=t_max=v` as the exact-temperature mechanism). Both carry
the **declarative-payloads-only** clause (issue #35 delta correction 3): foreign
callables are rejected as a design; `run_diffusion` widens by validated payloads
(`constraints=`, `control_signals=`, `capture=`), never surface-built closures.

---

## Track B — Research (the liquid-phase program)

The research program is a **dependency-ordered walk through VISION.md's tag
ledger**. Each rung is an experiment; a confirmation promotes a `[hypothesis]` or
`[open]` tag toward `[established]`, a falsification strikes it through — both are
banked gains, and that promote-or-strike rule is VISION.md's own (its closing
note). The organizing spine is the *liquid-phase* reframe: DiffusionGemma
**sublimates** (a position crosses directly between *steam* and *frozen* with no
*liquid* basin between); the program opens and instruments that missing
intermediate. The full framing lives in
[`docs/experiments/liquid-phase-decoding/`](docs/experiments/liquid-phase-decoding/)
(`concept.md` = the synthesis; `experiment.md` = the falsifiable H0s + observation
table).

The rungs, cheapest-falsification-first as ratified:

| Rung | Interesting if it works | Path it closes if falsified | Depends on | VISION / H0 pointer | Status |
|---|---|---|---|---|---|
| **R1 — cloze-renoise (β)** | Renoise drawn from top-k of the step's own distribution (β<1) instead of uniform holds positions *mobile-but-coherent* — the liquid the sublimating sampler skips | If the β-sweep shows only the two existing phases (immediate collapse or steam), the liquid basin does not exist under a renoise knob | none (cheapest falsification; runs on today's callback + a `torch.where`) | VISION §3.3 (the renoise axis); H0-renoise in `experiment.md` | **mechanism pending** — the composite `beta_rebuild` slot landed ([PR #105](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/105), `dgemma/participants.py:BetaRebuildParticipant`), ordered before `pin` per ADR-CDG-010 Decision 3, per-run stateless — but it is the **slot only**: the participant's own docstring names the β-viscosity/top-k mixture math as explicitly OUT (ADR-CDG-010 Open Question 2 still unresolved, gate ruling O3), and `run_diffusion` builds no `BetaRebuildParticipant` from any payload (`beta_rebuild=()` at every real call site). H0-renoise is not yet runnable. |
| **R0 — bench gate** | Full per-position `DISTRIBUTION` capture makes every held distribution observable — the socket the rest of the program reads from | Without it, H0-observe / H0-project cannot run; committed-state-only logging hides the liquid (proven empirically, n=5) | issues #14 (per-position entropy) + #11 (candidate ids); ADR-CDG-010/011 | `concept.md` "the gate everything waits on"; DISTRIBUTION seam | **closed** — Tier 0/1/2 capture landed across [PR #66](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/66) (Tier 0 + raw ids), [PR #99](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/99) (Tier 1 top-k + `capture=` ingress), [PR #106](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/106) (Tier 2 full distribution + budget-reject); display/consumers (entropy heatmap, `DGemmaTokenTrace`) in [PR #107](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/107), delivering the presentation surface issue #11 named as its remaining scope (issue itself still open on the tracker). The bench gate is real: the full `DISTRIBUTION` socket (Tier 2) is the concrete artifact H0-observe/H0-project read, and it is captured, budget-enforced, and displayable end to end. |
| **R2 — hold-and-release** | An equilibrate-then-quench protocol (hold under H0-control, then quench) makes canvas-scale liquid a controllable state, N quenches from one held state | If nothing holds — positions collapse or boil during the hold — the sublimation is not separable by this lever | R1, R0 | H0-control in `experiment.md`; concept.md control face | **R0 dependency satisfied** (above). The single-trajectory hold-and-release envelope (drive `entropy_bound`/`t_min`/`t_max` per step, hold then quench) is drivable **today** via `control_signals=` ([PR #100](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/100), `dgemma/participants.py:WalkerParticipant`, wired in `dgemma/loop.py`). The N-quenches-from-one-held-state half is still unbuilt — it needs `CANVAS_STATE` forking (branch one held canvas into multiple independent continuations), which no landed surface provides; `run_diffusion`/the MCP `generate` tool remain single-trajectory, single-shot calls. |
| **R3 — per-position heat** | A per-position heat *field* (freeze-last / commit-order steering) turns `entropy_bound` from a scalar stopping-rule into a steerable field | If order can't be steered, commit-order stays the emergent percolation front (#7 observes it, can't drive it) | R0; the R5 forward-hook | VISION §3.1 (freezing order as representation readout); concept.md control face | R0 dependency satisfied (above); the R5 forward-hook (`dgemma/constraints_hook.py:build_logit_mask_hook`, PR #71) is the per-position mechanism already landed. Per-position *field* scheduling (a wire per canvas position, not just per step) remains unbuilt — the walker (PR #100) drives per-**step**, not per-**position**. |
| **R4 — read-the-cloud** | The held distribution (top-k + weights) carries multi-meaning structure the scalar discards: equal-entropy positions show different candidate sets | If equal-entropy candidate sets are interchangeable, #14's scalar shadow is the whole signal and capturing the distribution buys nothing | R0, R2 | VISION §3.2 (sampler signatures); H0-observe in `experiment.md` | R0 dependency satisfied (above, Tier 1/2 capture is the readable artifact H0-observe needs). R2's dependency is only half-satisfied (single-trajectory envelope yes, N-quench forking no) — H0-observe's single-run narrowing claim is measurable now; the cross-quench comparison is not. |
| **R5 — project (the novelty)** | A directed operator over the held distribution selects among co-present stylistic modes (register / tense / mood) as a *creative* axis — the 2026-07-12 scan's "apparently novel" verdict | If held distributions are effectively unimodal, style lives in the trajectory/guidance and our claim collapses into existing methods | R2, R4 | VISION §3.4 (polymorphism); H0-project in `experiment.md` | Unstarted — depends on R2's unbuilt N-quench half and R4's downstream sampling-operator node (not yet designed), per R0/R2/R4 rows above. |
| **R6 — phase diagram** | The charted map of meaning over (entropy-bound × cooling rate × renoise distribution) — regions where the claim is stable, boundaries where it flips | The named deliverable is unreachable without the rungs beneath it; a null here is a null map, not a null idea | R1–R5 | VISION §3.4's named deliverable ("that map is the concrete deliverable the vision points at") | Unstarted — gated on R1 (mechanism pending, above) and R5 (unstarted, above). |

### Runnable today (post 2026-07-19 run)

Two probes are drivable right now with no further engineering, both over the MCP `generate` door ([PR #104](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/104)): **(a) the pin-complement re-melt/freeze-last protocol** — two `generate` calls (a full run, then a pins-on-keeps rerun holding everything but the target position via `constraints=`), Tier-2 capture (`capture=`) on both, reads H0-control's deferred-commitment face and H0-observe's narrowing claim without new code; **(b) the step-0-vs-late-step support probe** — one Tier-2 capture call comparing the cache-only (step 0) distribution against a late-step canvas-self-conditioned distribution at the same position. Neither needs the unbuilt CANVAS_STATE-forking half of R2. Lab floor: pre-register the H0 in `experiment.md` before either run, and bank the raw artifact per issue #101's proposed `runs/` convention — a verdict without a banked pointer doesn't count.

Parallel and conditional tracks:

- **S-track — Sudoku-class global-constraint problems (issue #28).** Runs in
  parallel: the capability demo for what a diffusion LM does that an AR model
  structurally cannot (global, order-free constraints). Rides the existing P0–P3
  surface for its weak versions; #14/#11 unlock the strong ones. The logit-mask
  seam (issue #28's 2026-07-13 comment) makes constraint propagation a decoding
  *dynamic*, not a prompting trick.
- **C-track — polyphonic prefill (H0-cache).** Assemble the write-once/read-only
  prefix KV cache from multiple prefills and diffuse off the richer field —
  "shape what the liquid condenses *from*." Apparently novel per the scan;
  ablation-gated (concat needs bridging recompute; blend off-manifold validity is
  untested). See `concept.md` "Polyphonic prefill".
- **X-track — substrate check (H0-substrate), conditional.** The differential
  diagnosis if R5 fails on DG: *wrong substrate, not wrong idea.* A
  non-causal-prefill diffusion LM (candidate: LLaDA) may hold richer liquid than
  DG's causal-prefill design. Only runs if R5 misses; "variety" must be
  operationally defined first.
- **G-track — crystalline CA / word-games
  ([ADR-CDG-016](decisions/adr-cdg-016-crystalline-ca-rule-table-payloads.md),
  accepted).** The *local-rule* counterpart of S-track's global constraints:
  neighbor-rule dynamics as declarative rule-table payloads over **committed ids**
  (split-flap register — each cell flips through its `top_p` nucleus until it
  lands), phase windows declared against the anneal schedule, β-renoise local
  re-melt as propagation. Field-free by design; H0-ca statable today. Depends on
  ADR-CDG-010/011 Phases 3/4 (+ R1's β mechanism for re-melt).
- **F-track — latent field (H0-hold)
  ([ADR-CDG-015](decisions/adr-cdg-015-latent-field-input-embedding-seam.md),
  accepted), conditional.** The held superposition lives at the
  **input-embedding seam** — a seventh seam the output-side inventory was missing.
  Existence proof landed absorbing-state only (Soft-Masked Diffusion, 2510.17206,
  ICLR 2026; Latent Refinement Decoding, 2510.11052); the USD transposition
  (uniform-mixture anchor `ē` in place of `e_[MASK]`) appears open. Two arms —
  training-free inject (off-manifold risk named) vs. continued-pretraining —
  both gated behind H0-hold; the train arm is operator-scheduled infra, not
  sequenced.

---

## Pointers (one home per concept — this file duplicates none of them)

- **[`VISION.md`](VISION.md)** — the *why*; the tag ledger this research walk
  promotes/strikes against.
- **[`docs/experiments/liquid-phase-decoding/`](docs/experiments/liquid-phase-decoding/)** —
  `concept.md` (the liquid-phase synthesis, seam inventory, prior-art scan) and
  `experiment.md` (the five falsifiable H0s + append-only observation table).
- **[`decisions/`](decisions/)** — the ADRs; ADR-CDG-008 (topology), ADR-CDG-010 /
  011 (accepted, ratified 2026-07-13 — the constraint + control seams),
  ADR-CDG-015 / 016 (accepted, ratified 2026-07-18 — the F-track field fork and
  the G-track crystalline CA).
- **`plan.md`** — tombstone (retired 2026-07-31); closed-phase evidence P0–P3
  archived on closed issue #199, not a live working-tree file.
- **Issues** — engineering: #35 (architecture review + R1–R6). Research and
  grounding: #23 (per-step control / mod-matrix), #28 (Sudoku-class flagship +
  logit-mask seam), #36 (loop-cache sweep hazard), #14 / #11 (the DISTRIBUTION
  gate), #10 (confidence dead-zone — the phase-boundary anchor), #7 (commit-front
  morphology), #6 (adversarial renoise — R1's failure branch inverted), #3
  (mot-juste goal).
