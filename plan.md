# Build Plan — DiffusionGemma ComfyUI Node Pack

Working roadmap. Decisions referenced here are recorded in `decisions/`; this
file is the *what-to-do-next*, not the *why* (the ADRs own the why).

**Created:** 2026-06-30

## Shape (template stolen from RES4LYF, payloads from ADR-CDG-001)

RES4LYF stays faithful to ComfyUI's CustomSampler topology; this pack keeps that
**topology** and swaps every socket **payload** for entropy-native types
(ADR-CDG-001). The image-graph reference (`ClownScheduler → ClownsharKSampler`,
with `ClownGuides` and the `ClownOptions_*` chain) maps node-for-node:

| RES4LYF node            | This pack                | Payload change                          |
|-------------------------|--------------------------|-----------------------------------------|
| `ClownScheduler`        | `DGemmaEntropySchedule`  | `SIGMAS` → `ENTROPY_SCHEDULE`            |
| `ClownGuides`           | `DGemmaConstraints`      | `GUIDES` → `CONSTRAINTS` (slot pins)    |
| `ClownsharKSampler`     | `DGemmaSampler`          | `(model, SIGMAS, GUIDES, LATENT)` → `(DGEMMA_MODEL, ENTROPY_SCHEDULE, CONSTRAINTS, CANVAS_STATE)` → `STRING + CANVAS_TRACE + CANVAS_STATE` |
| `bongmath` toggle       | `self_conditioning` toggle | logit-feedback refinement step        |
| `ClownOptions_*` chain  | `DGemmaOptions_*` chain  | commit-policy / renoise-rule swaps      |
| (none)                  | `DGemmaTrace`            | new — entropy heatmap + commit curve    |

## Phases

### Phase 0 — Recon & spec *(paper)*
Lock the access path (ADR-CDG-002: transformers + `TextDiffusionStreamer`; 3d-gen
Space as the per-step reference). Write the one-page type-and-socket spec.
**Deliverable:** the spec; ADRs 0001–0002 (done).

### Phase 1 — Thin vertical slice *(the reverse-engineerable artifact)*
`DGemmaLoader` + `DGemmaSampler` wrapping `.generate()`, EB defaults hardcoded,
structured like ComfyUI-Llama.
**Deliverable:** prompt in → text out, in the graph. The thing to take apart.

### Phase 2 — Expose the knobs
Promote EB params to widgets, defaults taken straight from the live run:
`max_steps=48`, `t=[0.4, 0.8]`, `entropy_bound=0.1`, `confidence=0.005`,
`canvas_length=256`, plus seed and thinking toggle.
**Deliverable:** entropy_bound sweep on a fixed prompt.

### Phase 3 — Instrumentation *(playground switches on)*
Drop to the documented loop / streamer; capture per-slot entropy + commit set
per step → `CANVAS_TRACE`. Build `DGemmaTrace`: entropy heatmap, commit-per-step
avalanche curve, live denoise view.
**Deliverable:** watch the late-burst on your own runs; replicate the
"Neither Parallel Nor Sequential" commit curve. **Resolve open question** on
ADR-CDG-002 (`mask_token=4`) here.

### Phase 4 — Schedule node + curve zoo
Split out `DGemmaEntropySchedule` with a curve selector
(linear / linear-quadratic / tangent) on the entropy/temperature axis — the
honest `bong_tangent`.
**Deliverable:** A/B a late-pivoted tangent entropy curve vs. linear.

### Phase 5 — Constraints + options chain
`DGemmaConstraints` (pin tokens at slots → bidirectional-ripple experiment) +
first `DGemmaOptions_*` swapping commit policy (entropy / confidence / margin / KL).
**Deliverable:** the experiments; the "insane combinations" surface.

### Phase 6 — The 🤪 phase *(maintenance)*
ComfyUI Manager registration, README, and the inevitable "runs on the 4090,
detonates on a Mac" issues (Metal / multi-GPU bidirectional-KV pain is already a
known DiffusionGemma sore spot). The point where it stops being yours and
becomes the on-ramp that didn't exist.

## Grounded defaults (from the first local run, Q4_K_M)
```
diffusion_eb: max_steps=48 t=[0.400,0.800] entropy_bound=0.1000
              stability=1 confidence=0.0050 kv_cache=on
```
Note: pass `-ngl 99` (+ `-cmoe` / `--n-cpu-moe` for overflow) — the first run hit
24 tok/s only because MoE experts spilled to CPU; in-step parallel was 456 tok/s.
