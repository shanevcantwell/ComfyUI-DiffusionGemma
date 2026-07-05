# Build Plan — DiffusionGemma ComfyUI Node Pack

Working roadmap. Decisions referenced here are recorded in `decisions/`; this
file is the *what-to-do-next*, not the *why* (the ADRs own the why).

**Created:** 2026-06-30
**Last updated:** 2026-07-05

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
| `web/` (JS extension) |    |    | ●  |    |    |    |
| `__init__.py`         | ●  | +  | +  | +  | +  |    |
| packaging + LICENSE   |    |    |    |    |    | ●  |

Per-module notes:

- **`dgemma/loop.py` is the spine, and its contract is per-step frames from
  day one**, not something Phase 3 invents. Via the one-line pipeline subclass
  (ADR-CDG-004 open question (a), resolved) widening `_callback_tensor_inputs`
  to include `"scheduler_output"`, it yields `(step, canvas, commit_mask,
  entropy_stats)` every step across all three phases it touches: P1 keeps only
  the last frame, P2 threads the EB params through the same per-step
  generator, P3 is presentation over frames that were already flowing —
  wiring the retained ones into `CanvasTrace`/`DGemmaTrace` instead of
  discarding them.
- **`dgemma/types.py` grows monotonically:** `DGemmaModel` real + `CanvasState`
  stub (P1 — with real validity fields from the start: `converged`,
  `committed_fraction`, `steps_used`, not just `STRING`; see Phase 1) →
  `CanvasTrace` (P3 — frames keyed by absolute noise level `(t, temperature,
  step_idx)`, never loop index alone) → `EntropySchedule` (P4) →
  `Constraints` (P5).
- **`dgemma/sampling.py` fork resolved (ADR-CDG-004).** The pack drives
  DiffusionGemma via the Diffusers pipeline + scheduler, not raw `.generate()`
  + `TextDiffusionStreamer` — the scheduler's `.step()` output natively carries
  the commit mask, so there is no entropy-bound commit/renoise/stop to
  reimplement. P3 is a pure capture task via `callback_on_step_end`; a custom
  scheduler subclass (not a `LogitsProcessor`) is the P4 extension point for
  curve swaps.
- **`nodes/sampler.py` is the one node that keeps changing shape:** `STRING` +
  validity readout out (P1) → +widgets (P2) → +`CANVAS_TRACE` out (P3) →
  consumes `ENTROPY_SCHEDULE`
  instead of raw widgets (P4) → consumes `CONSTRAINTS` + options (P5). Expect to
  touch it every phase; keep it thin so that's cheap. It drives the Diffusers
  pipeline (ADR-CDG-004), not `.generate()`.
- **`web/` is new in P3, for the LIVE-view split only** — a
  `WEB_DIRECTORY`-registered JS extension (`nodes.py:2269-2272` registration,
  `server.py:1225-1226` static serving) that listens for `DGemmaSampler`'s
  per-step `send_sync` custom events and renders the canvas as it denoises.
  It is not the `DGemmaTrace` analysis node (that stays in `nodes/trace.py`,
  post-hoc over `CANVAS_TRACE`) — see Phase 3.

Dependency spine in one line: **model → loop → (knobs) → trace → schedule →
constraints/options → publish.** Nothing downstream is buildable before the loop
runs, which is why P1 is the keystone.

## Phases

### Phase 0 — Recon & spec *(paper)*
Access path locked (ADR-CDG-002). ADRs 001–003 + this plan written. **Done.**

### Phase 1 — Thin vertical slice *(the reverse-engineerable artifact)*
`DGemmaLoader` + `DGemmaSampler` wrapping the Diffusers `DiffusionGemmaPipeline`
(ADR-CDG-004; loads via transformers, drives via Diffusers), EB defaults
hardcoded, structured like ComfyUI-Llama. `dgemma/loop.py`'s contract is
per-step frames from day one — `(step, canvas, commit_mask, entropy_stats)`
via the one-line pipeline subclass (ADR-CDG-004 open question (a), resolved)
— with P1 keeping only the last frame. The sampler emits `STRING` **plus** a
validity readout (`converged` / `committed_fraction` / `steps_used` on the
`CanvasState` stub), not a bare string: with wrong knobs the final text can
still contain uncommitted renoise garbage sitting inside otherwise-plausible
output, and a bare `STRING` has no way to say so (ADR-CDG-001 addendum,
2026-07-05). **Deliverable:** prompt in → text out + validity readout, in the
graph.

### Phase 2 — Expose the knobs
Promote EB params to widgets, defaults from the live run: `max_steps=48`,
`t=[0.4, 0.8]`, `entropy_bound=0.1`, `confidence=0.005`, `canvas_length=256`,
plus seed and thinking toggle. **Deliverable:** entropy_bound sweep on a fixed prompt.

### Phase 3 — Instrumentation *(playground switches on)*
`dgemma/loop.py` has yielded per-step frames since P1; this phase is
presentation over data that's already flowing, not the phase the capture
itself gets invented. Wire the retained frames into `CanvasTrace` — keyed by
absolute noise level `(t, temperature, step_idx)`, never loop index alone,
because variation runs (Renoise, `loose-ends.md`) start mid-schedule and
loop-index keying would make cross-run traces silently incomparable.
ADR-CDG-002's `mask_token` open question is already resolved documentarily
(ADR-CDG-004, 2026-07-05); this phase supplies the empirical corroboration.

**P3 splits into two deliverables, because ComfyUI's execution model forces
the split** — a node's outputs exist only once its `FUNCTION` returns, so a
downstream node cannot receive per-step frames live through a socket; there
is no partial-return mechanism to hand them off mid-loop.

- **(a) LIVE view — a feature of `DGemmaSampler`'s own node body, not a
  downstream node.** Per-step canvas is pushed via
  `PromptServer.instance.send_sync("<custom_event>", payload)` called from
  inside the sync `FUNCTION` — thread-safe by construction, since `send_sync`
  just does `call_soon_threadsafe` onto the asyncio message queue
  (`server.py:1374-1376`) and there is no event-name whitelist on the receiving
  side (`send_json`, `server.py:1364-1372`), so a custom event name is free to
  use. The frontend side is a `WEB_DIRECTORY`-registered JS extension —
  registration is `nodes.py:2269-2272` (checks `module.WEB_DIRECTORY`, mounts
  it into `EXTENSION_WEB_DIRS`), served as a static route at
  `server.py:1225-1226`. Adds a `web/` directory to the pack, registered from
  `__init__.py` (see the module build order table below).
  **Named trap:** do not smuggle this through `ProgressBar`'s `preview=` slot
  — that path is structurally image-typed downstream (`comfy/utils.py`'s
  `ProgressBar.update_absolute` → `main.py`'s hook → `send_image`,
  `server.py:1293-1301`, which does `image.save(...)` on whatever it's
  handed and throws on text). Text must go out its own custom event, not
  `preview=`.
  **Named residuals:** no in-tree precedent for per-step *text* push exists
  to copy (checked `comfy_extras/*.py` for `send_sync` usage — none found);
  this pack establishes the pattern. The frontend `addEventListener` idiom
  for a custom event is confirmed only at the shim level in the minified
  core bundle, not walked through in a worked first-party example — verify
  against the actual JS API when P3 builds this (tracked in
  `loose-ends.md`).
- **(b) ANALYSIS — `DGemmaTrace` over the complete `CANVAS_TRACE` socket.**
  Heatmap, avalanche curve, replay — all post-hoc and lossless, built from
  the full trace once the node has returned. This is the deliverable the
  original P3 text already described; it is unaffected by the live/analysis
  split above.

**Deliverable:** watch the late-burst live during your own runs (a), and
replicate the "Neither Parallel Nor Sequential" curve from the complete trace
after the fact (b).

**Beyond P3 — graph-driven stepping (deferred, envelope not yet built).** The
engine's `step()` primitive is proven extractable: the loop body factors
cleanly into KV populate → mask build → forward → `scheduler.step()`
(ADR-CDG-004). A `DGemmaStepSampler` node (`CANVAS_STATE` in, `CANVAS_STATE`
out, per ADR-CDG-005) could let the *graph* drive iteration instead of the
node's own internal loop — but this checkout's `comfy_extras` ships no
For/While pair to drive it with (grepped: only `RepeatImageBatch` exists,
no loop-control nodes), so graph-side iteration would need a third-party
loop pack or an eventual own For/While pair. Deferred by design, not
oversight: ADR-CDG-005 fixes the state contract precisely so this decision
can stay open without blocking anything — the envelope (what drives the
loop) is free to vary later because the identity (what crosses each step
boundary) is already settled.

### Phase 4 — Schedule node + curve zoo
Split out `DGemmaEntropySchedule` with a curve selector (linear / linear-quadratic
/ tangent) on the entropy/temperature axis — the honest `bong_tangent`.
**Deliverable:** A/B a late-pivoted tangent entropy curve vs. linear.

### Phase 5 — Constraints + options chain
`DGemmaConstraints` (pin tokens at slots → bidirectional-ripple experiment) +
first `DGemmaOptions_*` swapping commit policy (entropy / confidence / margin / KL).
Hard pinning is grounded, not speculative: re-assert the pinned slots in
`callback_on_step_end`'s canvas-overwrite return every step
(`pipeline_diffusion_gemma.py:407`, fires after `scheduler.step`) — no
diffusers internals touched. Candidate addition to the `DGemmaOptions_*`
commit-policy family: `BlockRefinementScheduler`'s `editing_threshold` knob,
an opt-in re-opening of already-committed tokens
(`scheduling_block_refinement.py:280-287`). **Deliverable:** the experiments;
the "insane combinations" surface.

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
