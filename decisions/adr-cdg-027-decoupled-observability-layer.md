# ADR-CDG-027 — Decoupled observability layer: agnostic telemetry platform over any execution engine

**Status**: `proposed`
**Date**: 2026-09-06
**Related**: ADR-CDG-026 (bidirectional control gate — the upstream producer this layer consumes), ADR-CDG-008 (MCP-center topology — the protocol surface for inter-process communication)

---

## Context

The LAS web-ui/public observability layer (`app/web-ui/public/`) was originally designed as a LangGraph-specific dashboard. It has since been decoupled from LangGraph's execution model and now operates as an agnostic telemetry platform — but this architectural choice was made implicitly through code evolution rather than recorded as a decision.

The current state: the observability layer receives events via `obs-events.js` → `obs-core.js`, renders them in `obs-graph.js`, and exposes inspection tools via `obs-inspector.js`. It has no hard dependency on LangGraph, ComfyUI, or any specific execution engine. However, without a recorded decision, future developers (or agentic coding agents) may re-introduce implicit coupling by assuming the event schema is LangGraph-specific, or by building new consumers that bypass this layer entirely.

The decoupling was completed just before the "Great Token Crunch" of February 2026 — a timing that proved fortuitous as ComfyUI-DiffusionGemma's step-level telemetry needed an independent landing zone that ComfyUI's own UI couldn't provide (ComfyUI can't resolve node traversal on timesteps).

## Decision

1. **The observability layer is a protocol-agnostic event consumer.** It accepts any JSON event stream conforming to the `ObsEvent` schema defined in `obs-events.js`. The schema has no LangGraph-specific fields and no ComfyUI-specific assumptions — it models generic execution events with:
   - `type`: event category (e.g., `"step_complete"`, `"canvas_update"`, `"control_signal"`)
   - `engine`: source identifier (e.g., `"langgraph"`, `"comfyui-dgemma"`, `"custom"`)
   - `payload`: engine-specific data, typed by `type`

2. **The event schema is the only contract.** Consumers produce events; the observability layer consumes them. There is no shared code between an execution engine and the observability layer — only the JSON schema. This means a new execution engine (e.g., a future JEPA-based sampler, a different diffusion model) can emit events to this platform without any code changes on either side.

3. **The decoupled UI runs as a separate process** from any execution engine. It connects via WebSocket or HTTP event stream to one or more engines simultaneously. This is not just a deployment choice — it's an architectural invariant: the observability layer must never block, crash, or degrade the performance of the engine it observes.

4. **The control surface (from ADR-CDG-026) uses the same decoupled UI.** The Play/Pause/Step buttons in `obs-control.js` are part of this platform — not a ComfyUI-specific feature. They send commands to any engine that exposes the corresponding API endpoint, identified by the `engine` field in the event stream.

## Rationale

### Positive Consequences
- **Engine independence.** Swap out DiffusionGemma for a JEPA-based sampler, add a second model running in parallel, or test a new sampler against the same observability — zero UI changes needed.
- **Multi-engine correlation.** Observe two different models running simultaneously on one dashboard; compare their annealing patterns, entropy trajectories, and commit fronts side by side.
- **Resilience.** The observability layer cannot crash the engine it observes (separate process, async event stream). Engine failures are visible as dropped events, not as UI freezes.
- **Future-proofing.** When you want to test ChatGPT's Astra model against your ComfyUI-DiffusionGemma setup (as discussed in design sessions), the observability layer is already ready to receive its telemetry — just define a new `ObsEvent` variant.

### Negative Consequences
- **Schema evolution requires versioning.** As new event types are added, old UI versions may not render them. Mitigated by graceful degradation (unknown event types are logged but don't crash the renderer).
- **Slightly higher latency than in-process telemetry.** Cross-process communication adds milliseconds; negligible for step-level observability (sub-second timescales) but relevant if sub-step telemetry is needed later.

## Alternatives Considered

### Option A: Embed observability directly into each execution engine's UI
**Why rejected:** ComfyUI's UI is tightly coupled to its DAG executor and cannot handle timestep-level granularity. LangGraph's web-ui was built for this purpose but creates vendor lock-in. The decoupled layer avoids both problems by being engine-agnostic.

### Option B: Use a shared message queue (Redis, RabbitMQ) as the event bus
**Why rejected:** Adds infrastructure complexity (running Redis/RabbitMQ just for telemetry) when WebSocket/HTTP streams solve the same problem with zero external dependencies. The decoupled UI already connects via WebSocket; adding a message queue layer is unnecessary indirection.

### Option C: Make the observability layer LangGraph-specific and build separate dashboards per engine
**Why rejected:** Duplicates effort across engines, fragments the developer experience, and makes multi-engine comparison impossible. The schema-agnostic approach costs almost nothing extra (one JSON envelope) and pays for itself with the first new engine integration.

## Open Questions

- [ ] **Should the `ObsEvent` schema include a `version` field?** Currently events are assumed to be latest-version; adding versioning would allow backward-compatible evolution but adds complexity. **Resolution:** defer until a second event type is added that changes the payload shape significantly.
- [ ] **What is the maximum number of simultaneous engine connections the UI should support?** Current implementation handles one comfortably; two was tested during multi-engine comparison experiments. **Resolution:** set soft limit at 4 engines based on observed use cases; add connection pool management if needed.

## Supersession Relationships

**Supersedes:** none (first record on observability layer architecture).
**Superseded by:** TBD — a future ADR on multi-model correlation or real-time model comparison may extend this decision's scope.

## Implementation Notes

| File | Change Type | Description |
|------|-------------|-------------|
| `app/web-ui/public/obs-events.js` | Documented | Add schema documentation comment at top; add `engine` field to event type definitions |
| `app/web-ui/public/obs-core.js` | Documented | Add engine-routing logic (route events by `engine` field to appropriate renderers) |
| `app/web-ui/public/obs-graph.js` | Minor | Ensure graph renderer handles multiple engines' data simultaneously |
| `ARCHITECTURE.md` | Modified | Add observability layer as a top-level component with its dependency diagram |

## References

- ADR-CDG-026 (bidirectional control gate) — the upstream producer this layer consumes
- ADR-CDG-008 (MCP-center topology) — the protocol surface for inter-process communication
- `app/web-ui/public/` — the decoupled observability layer source code
