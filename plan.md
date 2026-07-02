# Build Plan — DiffusionGemma ComfyUI Node Pack

Working roadmap. Decisions referenced here are recorded in `decisions/`; this
file is the *what-to-do-next*, not the *why* (the ADRs own the why).

**Created:** 2026-06-30
**Last updated:** 2026-06-30

## Shape (template stolen from RES4LYF, payloads from ADR-CDG-001)

RES4LYF stays faithful to ComfyUI's CustomSampler topology; this pack keeps that
**topology** and swaps every socket **payload** for entropy-native types
(ADR-CDG-001). The image-graph reference (`ClownScheduler → ClownsharKSampler`,
with `ClownGuides` and the `ClownOptions_*` chain) maps node-for-node:

| RES4LYF node            | This pack                | Payload change                          |
|-------------------------|--------------------------|-----------------------------------------|
| `ClownScheduler`        | `DGemmaEntropySchedule`  | `SIGMAS` → `ENTROPY_SCHEDULE`           |
| `ClownGuides`           | `DGemmaConstraints`      | `GUIDES` → `CONSTRAINTS` (slot pins)    |
| `ClownsharKSampler`     | `DGemmaSampler`          | sigmas/latent in → `STRING + CANVAS_TRACE + CANVAS_STATE` |
| `bongmath` toggle       | `self_conditioning` toggle | logit-feedback refinement step        |
| `ClownOptions_*` chain  | `DGemmaOptions_*` chain  | commit-policy / renoise-rule swaps      |
| (none)                  | `DGemmaTrace`            | new — entropy heatmap + commit curve    |

## Code decomposition (ADR-CDG-003)

Two packages: `nodes/` are thin adapters (unpack → call one `dgemma.*` fn →
wrap tuple, no logic); `dgemma/` is the ComfyUI-agnostic engine. The seam exists
so the denoising loop is runnable and testable with no ComfyUI process — the
precondition for the instrumentation phase.

```
ComfyUI-DiffusionGemma/
├── __init__.py          # aggregates NODE_CLASS_MAPPINGS, nothing else
├── nodes/               # thin adapters, NO logic
│   ├── loader.py  schedule.py  constraints.py  sampler.py  trace.py  options.py
└── dgemma/              # engine: imports with zero ComfyUI present
    ├── model.py  types.py  schedule.py  loop.py  sampling.py
```

## Module build order (the dependency spine)

Which modules come alive in which phase. Read down a column for "what this phase
touches," across a row for "how this module grows."

**Legend:** `●` new/real · `○` stubbed · `+` grows · `→` I/O changes · blank absent

| Module                | P1 | P2 | P3 | P4 | P5 | P6 |
|-----------------------|----|----|----|----|----|----|
| `dgemma/model.py`     | ●  |    |    |    |    |    |
| `dgemma/types.py`     | ●○ |    | +  | +  | +  |    |
| `dgemma/loop.py`      | ●  | +  | +  |    |    |    |
| `dgemma/schedule.py`  |    |    |    | ●  |    |    |
| `dgemma/sampling.py`  |    |    | ○● |    | +  |    |
| `nodes/loader.py`     | ●  |    |    |    |    |    |
| `nodes/sampler.py`    | ●  | +  | →  | →  | →  |    |
| `nodes/trace.py`      |    |    | ●  |    |    |    |
| `nodes/schedule.py`   |    |    |    | ●  |    |    |
| `nodes/constraints.py`|    |    |    |    | ●  |    |
| `nodes/options.py`    |    |    |    |    | ●  |    |
| `__init__.py`         | ●  | +  | +  | +  | +  |    |
| packaging + LICENSE   |    |    |    |    |    | ●  |

Per-module notes:

- **`dgemma/loop.py` is the spine.** It's the one file that grows across three
  phases: `.generate()` wrapper (P1) → accepts the EB params (P2) → per-step
  capture (P3). Everything else is roughly one-file-per-phase.
- **`dgemma/types.py` grows monotonically:** `DGemmaModel` real + `CanvasState`
  stub (P1) → `CanvasTrace` (P3) → `EntropySchedule` (P4) → `Constraints` (P5).
- **`dgemma/sampling.py` is the one fork (from ADR-CDG-002).** It only appears in
  P3 *if* you reimplement the entropy-bound commit/renoise/stop rather than
  hooking `TextDiffusionStreamer`. If the streamer exposes enough, this file
  slips to P5 and P3 stays a pure capture task. Deciding this resolves the
  `mask_token=4` open question on ADR-CDG-002.
- **`nodes/sampler.py` is the one node that keeps changing shape:** `STRING` out
  (P1) → +widgets (P2) → +`CANVAS_TRACE` out (P3) → consumes `ENTROPY_SCHEDULE`
  instead of raw widgets (P4) → consumes `CONSTRAINTS` + options (P5). Expect to
  touch it every phase; keep it thin so that's cheap.

Dependency spine in one line: **model → loop → (knobs) → trace → schedule →
constraints/options → publish.** Nothing downstream is buildable before the loop
runs, which is why P1 is the keystone.

## Phases

### Phase 0 — Recon & spec *(paper)*
Access path locked (ADR-CDG-002). ADRs 001–003 + this plan written. **Done.**

### Phase 1 — Thin vertical slice *(the reverse-engineerable artifact)*
`DGemmaLoader` + `DGemmaSampler` wrapping `.generate()`, EB defaults hardcoded,
structured like ComfyUI-Llama. **Deliverable:** prompt in → text out, in the graph.

### Phase 2 — Expose the knobs
Promote EB params to widgets, defaults from the live run: `max_steps=48`,
`t=[0.4, 0.8]`, `entropy_bound=0.1`, `confidence=0.005`, `canvas_length=256`,
plus seed and thinking toggle. **Deliverable:** entropy_bound sweep on a fixed prompt.

### Phase 3 — Instrumentation *(playground switches on)*
Grow `dgemma/loop.py` to per-step capture; add `CanvasTrace`; build `DGemmaTrace`
(entropy heatmap + commit-per-step avalanche curve + live denoise).
**Deliverable:** watch the late-burst on your own runs; replicate the
"Neither Parallel Nor Sequential" curve. **Resolve ADR-CDG-002 open question here.**

### Phase 4 — Schedule node + curve zoo
Split out `DGemmaEntropySchedule` with a curve selector (linear / linear-quadratic
/ tangent) on the entropy/temperature axis — the honest `bong_tangent`.
**Deliverable:** A/B a late-pivoted tangent entropy curve vs. linear.

### Phase 5 — Constraints + options chain
`DGemmaConstraints` (pin tokens at slots → bidirectional-ripple experiment) +
first `DGemmaOptions_*` swapping commit policy (entropy / confidence / margin / KL).
**Deliverable:** the experiments; the "insane combinations" surface.

### Phase 6 — The 🤪 phase *(maintenance)*
ComfyUI Manager registration, README flip from "aspirational", **LICENSE file**
(tracked in loose-ends), and the inevitable "runs on the 4090, detonates on a
Mac" issues (Metal / multi-GPU bidirectional-KV is a known DiffusionGemma sore spot).

## Grounded defaults (from the first local run, Q4_K_M)
```
diffusion_eb: max_steps=48 t=[0.400,0.800] entropy_bound=0.1000
              stability=1 confidence=0.0050 kv_cache=on
```
Note: pass `-ngl 99` (+ `-cmoe` / `--n-cpu-moe` for overflow) — the first run hit
24 tok/s only because MoE experts spilled to CPU; in-step parallel was 456 tok/s.
