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
- **`plan.md`** holds the *closed-phase evidence* — phases P0–P3, each with its
  per-phase PASS record. That ledger is done and stays put (it lives in the working
  tree, not the published tree); this file does not restate it.

Two independent `R`-namespaces appear below and must not be conflated:
**Track A's R1–R6 are engineering refactors** (issue #35); **Track B's R0–R6 are
research rungs** (the liquid-phase program). Same letter, different ledgers.

---

## Roadmap (recorded 2026-07-31, operator-set)

**Version namespace note.** The "0.5.0 AutoRound INT4 release" attempt died
unshipped — no version literal, tag, or registry artifact ever bore the name
(post-mortem: [`docs/postmortems/2026-07-31-0.5.0.md`](docs/postmortems/2026-07-31-0.5.0.md)).
The name is therefore reused below. 0.4.2 remains available as the stabilization
tag.

- **0.4.2 — stabilization patch (next ship).** The product-fix batch on the
  reconciled trunk: #191 (VRAM guard reports the measured condition, not a menu),
  #187 (encode device pin under whole-fit), #188 (live-view single mint), #169
  (F0 test baseline), #151 (mcp<2.0 verification), #161 (CI actions bump), #175
  (node explanations, draft tier), plus the Encode/Denoise visibility disposition
  (operator call pending — the kv_cache door is inert until the seam bracket).
  Ship discipline per #195 (gate integrity axis) and #196 (version/tag timing).
- **0.5.0 — the refactor version, and only that** — **shipped 2026-08-03**, tag
  `v0.5.0` @ `c4beb3e`.
  [ADR-CDG-018](decisions/adr-cdg-018-decompose-loop-py.md): `dgemma/loop.py`
  decomposed into `config`/`compat`/`capture`/`excision` behind a re-export
  facade; pure refactor proven by golden-trace byte-identity + AST comparison;
  live-verified 4/4 on real weights (#129, closed). The registry still serves
  0.4.2 — a 0.5.0 registry publish is a separate operator call. Post-tag on
  `main`: #119 offload-aware tied-weights guard (`f04688f`), rides the next tag.
  Banked beyond the topology bracket: #219 (salience decomposition: model.py
  map + prose evacuation).
- **Next bracket — opened 2026-08-04 (run ledger: #225; ordering: KV-encoder
  live, GGUF design beside):**
  (a) **KV-cache encoder — in flight, gated on a re-run.** The Q-2 smoke ran
  2026-08-04 and typed **BLOCKED**: `encode_sequence` OOMs on the bare
  transformers lane before any decoder code (#226 — a regression, bisect
  window `a68e29d..33551d5`; empty-string ingress gap filed as #227). The
  ratified re-run route is **Amendment 1 on #62**: identical pre-registered
  protocol driven through the ComfyUI API lane (S-B's lane); the with-cache
  skeleton is banked on `scratch/q2-skeleton-2026-08-04`. GPU window is
  operator-scheduled. Knowledge-locality fix (function-scoped live-proof
  banking + promote an Encode scenario into the E2E battery) is #228.
  (b) **GGUF engine/packaging — corrected per #223.** "Rung 4" was roadmap
  shorthand, not the issue-thread sequence. Actual gate chain: rung-1 probe
  (partial 2026-08-04: #24423 leg live at 55.5–74.6 tok/s Q8_0,
  byte-identical seed-rerun; #24427 legs absent from host) → engine ADR
  [ADR-CDG-020](decisions/adr-cdg-020-gguf-engine-sourcing-pinned-pr-branch.md)
  (proposed; ratification **HOLD** until the #24427 legs run) → CI packaging.
  Note: `c3fb972` is a pinned upstream draft-PR branch, not owned code —
  "owned pin" language retired.
- **Lifecycle proposal pending:**
  [ADR-CDG-021](decisions/adr-cdg-021-per-surface-vram-tenancy-ownership.md)
  (per-surface VRAM tenancy — MCP/transformers lane may OOM fail-loud;
  ComfyUI surface integrates `comfy.model_management`; grown from an operator
  observation, #229) — proposed, operator disposition open; binds the #138
  bracket to in-process consumption if accepted. Topology (#138) and #219
  remain queued below.
- **Next bracket — topology (version minted at opening):**
  [ADR-CDG-019](decisions/adr-cdg-019-mcp-as-contract-topology-remediation.md)
  (accepted 2026-08-03; sequenced after #129; `dgemma_mcp/` rename per gate
  finding I1) → topology remediation (#138), import-gate consolidation (#57),
  and the interrupt fix (#140, rides the restructure per operator 2026-08-01).
- **The seam (post-refactor).**
  [ADR-CDG-012](decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md)
  Phase 4: the Q-2 real-weights de-risk smoke, then the decoder genuinely driven
  off injected caches (#62). #187 is a precondition; the fixed-seed encoder-text
  sweep (byte-identical canvases predicted pre-Phase-4) is the liveness gate.
  Then #47's known-provenance cache-perturbation experiments open.
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
- **`plan.md`** — closed-phase evidence, P0–P3 (working-tree only, not published).
- **Issues** — engineering: #35 (architecture review + R1–R6). Research and
  grounding: #23 (per-step control / mod-matrix), #28 (Sudoku-class flagship +
  logit-mask seam), #36 (loop-cache sweep hazard), #14 / #11 (the DISTRIBUTION
  gate), #10 (confidence dead-zone — the phase-boundary anchor), #7 (commit-front
  morphology), #6 (adversarial renoise — R1's failure branch inverted), #3
  (mot-juste goal).
