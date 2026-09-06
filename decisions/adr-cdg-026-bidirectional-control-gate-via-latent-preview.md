# ADR-CDG-026 — Bidirectional control gate via latent preview callback monkeypatch

**Status**: `proposed`
**Date**: 2026-09-06
**Related**: ADR-CDG-003 (node/engine seam), ADR-CDG-008 (MCP-center topology — the external control layer is one consumer of this gate), ADR-CDG-014 (frame capture discipline — the telemetry stream this gate rides on)

---

## Context

ComfyUI's latent preview mechanism (`latent_preview.py::prepare_callback`) is a fire-and-forget hook: every sampler calls it to push per-step preview images to the frontend. It interrupts the sampling loop to generate and send data, but **never waits for a response**. The executor doesn't pause; the callback is a one-way channel from sampler → UI.

This means ComfyUI's DAG executor has no native mechanism for an external system to:
1. Pause execution at step T
2. Inspect the live canvas state (latents, entropy distribution, committed positions)
3. Issue a mutation command (re-noise specific positions, apply token masks)
4. Resume execution from where it left off

The gap is structural: ComfyUI was built for image diffusion where throughput matters more than interpretability. A small artifact or misplaced pixel is "artistic flavor"; in text generation via DiffusionGemma, a single committed prestige token poisons the entire downstream context window. Text requires systemic intervention mid-loop; pixels do not.

The latent preview callback is the **one shared hook point** all samplers call — KSampler, custom samplers, DiffusionGemma's own loop. It sits above sampler-specific code and below ComfyUI's DAG executor. Hijacking it converts a passive telemetry channel into an active control plane without touching any sampler implementation or ComfyUI core.

## Decision

1. **Monkeypatch `latent_preview.prepare_callback` as the universal execution gate.** Replace the fire-and-forget callback with a wrapper that:
   - Calls the original preview function (previews still work)
   - Projects per-step telemetry to the decoupled observability layer (`obs-core.js`)
   - Checks an external orchestration lock (`threading.Event`) before returning — if clear, blocks the GPU thread until a CONTINUE signal arrives from the control layer

2. **The gate is opt-in via `execution_mode` flag.** Default: `"continuous"` (behaves identically to stock ComfyUI). When set to `"step"`, every step yields at the gate. This ensures zero impact on standard workflows — the pack ships as a research instrument, not a breaking change.

3. **The external control layer is a separate process** (the decoupled LAS web-ui or an MCP client) that communicates via HTTP POST endpoints (`/api/dgemma/next`, `/api/dgemma/mutate`) on the ComfyUI server. The gate waits on `threading.Event.wait()` with a short timeout; if no signal arrives, it times out and auto-resumes to prevent GPU thread deadlock.

4. **Mutation commands carry structured payloads** — token position masks, re-noise distributions, or probability vectors — applied to the latent state before the next step proceeds. The mutation is applied in-place on the tensor `x` (or `x0`) that the callback receives.

5. **This pattern is generalizable by design.** Any iterative ML loop inside ComfyUI that calls `prepare_callback` inherits this control plane — not just DiffusionGemma, but any future custom sampler. The monkeypatch sits above all samplers; it captures them all with one hook.

## Rationale

### Positive Consequences
- **True step-by-step interpretability.** Watch the canvas anneal frame by frame, pause right when a prestige token attempts to crystallize, observe the exact state before it becomes un-falsifiable truth.
- **One hook, all samplers.** No per-sampler instrumentation needed; `prepare_callback` is the universal seam.
- **Zero impact on stock workflows.** Default `"continuous"` mode is functionally identical to unmodified ComfyUI.
- **Bi-directional bridge.** Not just telemetry out — mutation commands flow back in through the same gate mechanism.

### Negative Consequences
- **Blocks the GPU thread when paused.** The node's Python thread holds VRAM allocation while waiting; if the external control layer crashes, the timeout prevents permanent deadlock but the step is lost.
- **Adds a dependency on the external control layer's availability.** A network hiccup between ComfyUI and the observability UI means the gate times out — acceptable behavior (auto-resume), but not silent.
- **Monkeypatching is inherently fragile.** If ComfyUI changes `prepare_callback`'s signature, the patch breaks. Mitigated by storing the original function reference and calling it through, never replacing its internals.

## Alternatives Considered

### Option A: Modify each sampler's internal loop directly
**Why rejected:** Fragile (per-sampler maintenance), doesn't generalize to future samplers, requires maintaining forks of ComfyUI core code. The `prepare_callback` hook exists precisely for this purpose — it's the intended extension point.

### Option B: Build a custom ComfyUI fork with native step-control
**Why rejected:** Massive maintenance burden; every ComfyUI update must be rebased. The monkeypatch achieves the same result without forking, and the control logic lives in the external layer where it can evolve independently.

### Option C: Use ComfyUI's existing websocket push as a two-way channel
**Why rejected:** Websocket is fire-and-forget from the server side; there's no built-in request/response pattern. HTTP POST endpoints are simpler, more debuggable, and don't require maintaining bidirectional websocket state on the ComfyUI process.

## Open Questions

- [ ] **Timeout duration for the gate.** Default should be long enough for human interaction (clicking "Step Next") but short enough to prevent indefinite GPU thread holding. **Resolution:** set initial value at 30s; tune based on first real use.
- [ ] **Should mutation commands be validated before application?** A malformed mask could corrupt the latent tensor. **Resolution:** P1 ships without validation (trust the control layer); P2 adds schema validation on incoming mutations.
- [ ] **Does this interact with ComfyUI's interrupt mechanism?** `/interrupt` sends SIGTERM to the queue worker; if a node is blocked in `Event.wait()`, does the signal propagate? **Resolution:** test during P1; may need to catch and re-raise the interrupt inside the gate.

## Supersession Relationships

**Supersedes:** none (first record on bidirectional control via preview callback).
**Superseded by:** TBD — if a future ADR introduces a native ComfyUI step-control API, this monkeypatch becomes deprecated in favor of the native path.

## Implementation Notes

| File | Change Type | Description |
|------|-------------|-------------|
| `dgemma/loop.py` | Modified | Add `execution_mode` flag; inject `threading.Event` gate into sampling loop; call `push_telemetry_to_observability()` each step |
| `latent_preview.py` (monkeypatch) | New file (`dgemma/control_patch.py`) | Store original `prepare_callback`; define wrapper with gate logic; apply patch at module import time |
| `consumers/comfyui/sampler.py` | Modified | Expose `execution_mode` widget; wire to loop |
| Server endpoints (new) | New | `/api/dgemma/next` (HTTP POST, sets Event), `/api/dgemma/mutate` (HTTP POST, applies mutation to tensor state) |
| `obs-control.js` (new) | New | Control buttons (Play/Pause/Step Next) in decoupled web-ui; sends HTTP commands to server endpoints |

## References

- ComfyUI `latent_preview.py::prepare_callback` source — the hook being monkeypatched
- ADR-CDG-014 (frame capture discipline) — the telemetry stream this gate rides on
- ADR-CDG-008 (MCP-center topology) — the external control layer is one consumer of this gate
