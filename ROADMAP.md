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
| **R4** — shared fake-pipeline/scheduler fixture in `tests/conftest.py` (N steps, mutable `scheduler.config`, hook-recording model) | Gates testable composition for R1/R5 — the fixture the ordering tests need | not-started |
| **R1** — callback-composition layer in `dgemma/loop.py`: ordered participants, canvas-write threading, per-participant exception policy, `_FrameCollector` first | Opens the single hardcoded callback slot (F1, **ONE-DOOR**) to the expansion participants — β-renoise, walker, pin, capture — that everything downstream needs | not-started |
| **R5** — forward-hook lifecycle context manager; invariant "no hook survives a `run_diffusion` call" | Closes hook-leakage across executions (F4, **STATELESS-CORE**) — the per-position heat field installs and tears down cleanly | not-started |
| **R3** — diffusers version guard + structural probe (scheduler kwargs, `accepted_index`, `_callback_tensor_inputs`) | Fails loud on a diffusers bump instead of silently reporting a wrong re-derived temperature (F6, **EMIT-CANONICAL**) | not-started |
| **R2** — socket-type mint module + grep-gate test (no inline `DGEMMA_*` literal outside it) | One mint home for socket strings (F2, **ONE-MINT**); lands with/before CDG-008 Phase 1, in `surfaces/comfyui/socket_types.py` | not-started |
| **R6** — `DiffusionFrame` extension discipline (optional-with-defaults; heavy-field retention policy) | Lets rung-4's heavy `DISTRIBUTION` field ride the frame additively without breaking ADR-CDG-005's small-per-step economy (F3, **EMIT-CANONICAL**); rides research rung R4-observe, analysis functions go to the CDG-008 Phase-3 home | not-started |

Sequencing (issue #35, delta-corrected): **R4 → R1 → R5 cluster + R3** before any
research rung lands; **R2** with/before CDG-008 Phase 1; rung-4 analysis behind
CDG-008 Phase 3.

### The topology move — ADR-CDG-008's five phases

[ADR-CDG-008](decisions/adr-cdg-008-mcp-center-multi-surface-topology.md) (accepted)
adopts an **MCP-center, multi-surface, single-repo** topology: `dgemma/`
(`load_model` + `run_diffusion`) is the one contract, MCP is the base surface,
ComfyUI is one peer surface among others. The published repo name stays
(`IDENTITY⊥ENVELOPE`); only the internal directory vocabulary changes.

| Phase | Move | Status |
|---|---|---|
| **1** | Rename `nodes/` → `surfaces/comfyui/`, move `web/` → `surfaces/comfyui/web/` | not-started |
| **2** | Add `surfaces/mcp/` — the base surface over `load_model` + `run_diffusion` (transcribe `semantic-kinematics-mcp`, with the two `STATELESS-CORE` / `ONE-DOOR` corrections) | not-started |
| **3** | Relocate analysis out of `dgemma/`'s import graph into a consumer home | not-started |
| **4** | Add the boundary test: base contract imports no analysis (flips the prose-only row to in-force) | not-started |
| **5** | Rewrite `ARCHITECTURE.md` against the governance template | **done** — [PR #37](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/pull/37) (merged) |

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

| Rung | Interesting if it works | Path it closes if falsified | Depends on | VISION / H0 pointer |
|---|---|---|---|---|
| **R1 — cloze-renoise (β)** | Renoise drawn from top-k of the step's own distribution (β<1) instead of uniform holds positions *mobile-but-coherent* — the liquid the sublimating sampler skips | If the β-sweep shows only the two existing phases (immediate collapse or steam), the liquid basin does not exist under a renoise knob | none (cheapest falsification; runs on today's callback + a `torch.where`) | VISION §3.3 (the renoise axis); H0-renoise in `experiment.md` |
| **R0 — bench gate** | Full per-position `DISTRIBUTION` capture makes every held distribution observable — the socket the rest of the program reads from | Without it, H0-observe / H0-project cannot run; committed-state-only logging hides the liquid (proven empirically, n=5) | issues #14 (per-position entropy) + #11 (candidate ids); ADR-CDG-010/011 | `concept.md` "the gate everything waits on"; DISTRIBUTION seam |
| **R2 — hold-and-release** | An equilibrate-then-quench protocol (hold under H0-control, then quench) makes canvas-scale liquid a controllable state, N quenches from one held state | If nothing holds — positions collapse or boil during the hold — the sublimation is not separable by this lever | R1, R0 | H0-control in `experiment.md`; concept.md control face |
| **R3 — per-position heat** | A per-position heat *field* (freeze-last / commit-order steering) turns `entropy_bound` from a scalar stopping-rule into a steerable field | If order can't be steered, commit-order stays the emergent percolation front (#7 observes it, can't drive it) | R0; the R5 forward-hook | VISION §3.1 (freezing order as representation readout); concept.md control face |
| **R4 — read-the-cloud** | The held distribution (top-k + weights) carries multi-meaning structure the scalar discards: equal-entropy positions show different candidate sets | If equal-entropy candidate sets are interchangeable, #14's scalar shadow is the whole signal and capturing the distribution buys nothing | R0, R2 | VISION §3.2 (sampler signatures); H0-observe in `experiment.md` |
| **R5 — project (the novelty)** | A directed operator over the held distribution selects among co-present stylistic modes (register / tense / mood) as a *creative* axis — the 2026-07-12 scan's "apparently novel" verdict | If held distributions are effectively unimodal, style lives in the trajectory/guidance and our claim collapses into existing methods | R2, R4 | VISION §3.4 (polymorphism); H0-project in `experiment.md` |
| **R6 — phase diagram** | The charted map of meaning over (entropy-bound × cooling rate × renoise distribution) — regions where the claim is stable, boundaries where it flips | The named deliverable is unreachable without the rungs beneath it; a null here is a null map, not a null idea | R1–R5 | VISION §3.4's named deliverable ("that map is the concrete deliverable the vision points at") |

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

---

## Pointers (one home per concept — this file duplicates none of them)

- **[`VISION.md`](VISION.md)** — the *why*; the tag ledger this research walk
  promotes/strikes against.
- **[`docs/experiments/liquid-phase-decoding/`](docs/experiments/liquid-phase-decoding/)** —
  `concept.md` (the liquid-phase synthesis, seam inventory, prior-art scan) and
  `experiment.md` (the five falsifiable H0s + append-only observation table).
- **[`decisions/`](decisions/)** — the ADRs; ADR-CDG-008 (topology), ADR-CDG-010 /
  011 (in ratification, the constraint + control seams).
- **`plan.md`** — closed-phase evidence, P0–P3 (working-tree only, not published).
- **Issues** — engineering: #35 (architecture review + R1–R6). Research and
  grounding: #23 (per-step control / mod-matrix), #28 (Sudoku-class flagship +
  logit-mask seam), #36 (loop-cache sweep hazard), #14 / #11 (the DISTRIBUTION
  gate), #10 (confidence dead-zone — the phase-boundary anchor), #7 (commit-front
  morphology), #6 (adversarial renoise — R1's failure branch inverted), #3
  (mot-juste goal).
