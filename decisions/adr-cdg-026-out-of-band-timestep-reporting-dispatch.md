# ADR-CDG-026 — Out-of-band timestep reporting dispatch

**Status**: `proposed` — design only; not implemented
**Date**: 2026-09-06
**Related**: ADR-CDG-003 (node/engine seam), ADR-CDG-008 (MCP-center topology),
ADR-CDG-014 (frame-capture discipline), ADR-CDG-027 (proposed observability,
provenance, and replay work; not ratified by this record)

---

## Context

DiffusionGemma already has the right observation seam:
`_FrameCollector.on_frame` runs after capture has constructed and retained the
step's `DiffusionFrame`. Today a caller can use that seam for live reporting,
but the callback still executes synchronously in the sampler call. Parsing,
rendering, writing artifacts, or sending rich updates there extends the time
before diffusion resumes and lets an optional reporting failure escape through
the engine callback contract.

ComfyUI's graph executor is also the wrong place to schedule per-timestep report
consumers. A node class is an executor-facing adapter with declared sockets and
outputs; invoking one from a callback or worker would bypass normal graph
ordering, caching, lifecycle, and error semantics. Conversely,
`PromptServer.send_sync` gives custom nodes an official server-to-client message
path, but it only schedules delivery. It does not make report parsing, image
rendering, or file I/O background work.

The earlier proposed CDG-026 text tried to turn ComfyUI latent preview into a
blocking bidirectional control gate. That design mixed observation with
execution control, depended on a fragile global patch, and made optional UI
behavior capable of stalling generation. This revision replaces that proposal
rather than layering a second, contradictory decision on top of it.

## Decision

Adopt a **one-way, read-only, in-process reporting dispatcher**. The MVP moves
consumer work off the ComfyUI graph/node execution path and off the diffusion
hot path after a bounded handoff. It does not create a control plane.

### 1. Capture seam and lifecycle ownership boundary

`run_diffusion` owns reporting lifecycle for its core invocation. Concretely,
it either emits lifecycle events itself or constructs and closes one per-run
reporting session inside that invocation. That owner alone attempts
`RunStarted`, selects exactly one internal terminal reason, submits the matching
logical terminal item to its reserved admission/scheduling lane, and performs
the bounded drain before returning or re-raising. A surface adapter and
`_FrameCollector.on_frame` do not own run terminality.

Use the existing `_FrameCollector.on_frame` observer seam **after capture** only
for `FrameReported` snapshots. The callback performs only these sampler-thread
duties:

1. copy the minimal fields required by the selected reports into immutable,
   CPU-owned plain values (scalars, strings, tuples, bytes, or other explicitly
   bounded value objects); no view may retain mutable sampler state or GPU
   storage;
2. ask the lifecycle-owned session to assign the event's run identity and
   monotonic sequence and attempt a nonblocking `FrameReported` enqueue; and
3. return immediately to diffusion.

Snapshot creation and frame enqueue remain inside the sampler call. The
lifecycle owner performs the final bounded drain in the same core invocation.
This is **thread isolation**, not process isolation, and does not eliminate the
Python GIL. Snapshot cost and contention are therefore measured rather than
called free.

Tier-2 full distributions are not part of the default snapshot. A report that
needs them must be explicitly requested and remain subject to ADR-CDG-014's
capture budgets; the dispatcher must not make heavy capture implicit.

### 2. One managed worker, bounded per-run and aggregate state

A singleton dispatcher owns one managed in-process worker thread. Each admitted
run has a bounded FIFO so frames from that run are consumed in sequence. The
singleton also enforces independent global ceilings on active runs, total
queued/in-flight events, and total accounted resident work-item bytes (snapshots,
bounded derived payloads, and reserved control/terminal capacity). Consumer work
that cannot fit the remaining byte budget fails that report item closed rather
than creating unaccounted resident data. Per-run bounds alone are insufficient:
the sum of many individually bounded runs must never create an unbounded global
queue or per-run-state map.
The worker performs report parsing, rendering, file writes, and preparation/
delivery of small status or artifact-reference events outside normal ComfyUI
graph/node execution.

Admission is fail-closed for reporting and fail-open for generation. A
`RunStarted` request must atomically reserve one active-run slot plus its bounded
queue/accounting state. If any global budget cannot be reserved, the dispatcher
returns a disabled/degraded per-run session: it allocates no FIFO or persistent
per-run registry entry, accepts no `FrameReported` events, records only bounded
aggregate rejection counters, and never fails or delays generation. For an
admitted run, crossing either a per-run or global event/byte ceiling invokes the
declared coalesce/drop policy rather than allocating more memory. Reservations
are returned only when their accounted work is disposed. A bounded-drain timeout
seals the run, rejects new work, and discards/cancels queued work; any one
currently executing worker item remains charged to the global event/byte budget
until it exits and retains only a fixed-size sealed completion record. Thus
rejected, overflowing, abandoned, and timed-out runs cannot accumulate
unbounded dispatcher state or create unaccounted late work.

The singleton has an explicit start/ready/stop/join lifecycle; registration and
shutdown are idempotent. It accepts frame work only for a successfully admitted
run after `RunStarted` and rejects work after that run's terminal admission. It
releases a run registry entry when all charged work is disposed, using only the
bounded sealed record above after a timeout. A process shutdown requests stop,
drains only within its configured bound, records unfinished work as incomplete,
then joins within a bounded timeout. It must not leave a non-daemon thread
preventing shutdown.

This deliberately follows ComfyUI's asset seeder as the closest local managed
worker precedent—owned thread, cancellation, locking, bounded wait, explicit
shutdown—without copying its scanning-specific pause/restart behavior.

### 3. Consumers are pure logic; nodes remain adapters

Do not instantiate or invoke Comfy node classes from the worker. Extract and
reuse consumer-tier functions/classes behind report nodes. Those consumers take
plain snapshot/event values and return plain report results or artifact
references; they do not depend on the Comfy executor.

Comfy node classes remain adapters and the canonical socket/output producers
when a workflow invokes them normally. Their existing signatures and normal
final outputs do not change in Phase 1. At least one existing report consumer
must be extracted in Phase 1 to prove this separation is real rather than
aspirational.

### 4. `send_sync` is delivery, not compute

`PromptServer.instance.send_sync(...)` may deliver a small status event or an
artifact reference prepared by the worker. It is not a job queue or background
compute facility. No tensor, image array, full distribution, or large rendered
payload is sent through it. Rendering/file work completes first; the message
contains bounded metadata and, where applicable, a reference to the artifact.
Delivery exceptions are handled as that consumer's failure and never escape to
generation.

### 5. Logical event contract

The dispatcher contract has four logical event kinds:

- `RunStarted`
- `FrameReported`
- `RunFinished`
- `RunAborted`

Every event includes `run_id`, a strictly increasing per-run `sequence`, event
kind/schema version, an explicit `reporting_status`, and provenance sufficient
to attribute the model, tokenizer, effective run configuration, capture/report
specification, and producer version. `RunStarted` reports `active`;
`FrameReported` reports `reported`, `gap`, or `error` and includes
`(canvas_idx, step_idx)`; `step_idx` resets to zero for each canvas. Terminal
events distinguish generation outcome from telemetry outcome:
`generation_outcome` is `completed`, `cancelled`, or `failed`, while
`reporting_status` is `complete` or `incomplete` and carries known sequence
gaps/dropped ranges, admission or delivery failure, per-consumer errors, and
finish-barrier timeout state. `RunFinished` is used only for `completed`;
`RunAborted` is used for `cancelled` or `failed` and records the explicit
internal terminal reason (with only bounded, non-sensitive error metadata).

This distinction is required by today's core behavior: `run_diffusion` catches
`DiffusionCancelled` and converts it to the ordinary partial
`(text, CanvasState, CanvasTrace)` result shape. Its lifecycle owner must set the
internal reason to `cancelled` in that handler *before* conversion and normal
return; otherwise a return-shape observer would falsely report completion. A
normal pipeline/drive-body return sets `completed`. Any other exception sets
`failed` before it is re-raised. If Phase 1 cannot ground a finer reason at this
seam, it may emit only the grounded completed-versus-aborted distinction and
must not infer cancellation from result shape or trace length.

`run_id` is stable for one generation attempt. Sequence is the ordering source;
`(canvas_idx, step_idx)` is domain position, not a substitute for ordering.
Repeated or skipped positions therefore remain diagnosable.

### 6. Failure, overflow, and completion semantics

- **Per-consumer isolation:** each consumer runs behind its own exception
  boundary. One consumer's parse/render/write/send failure is recorded and
  other consumers continue.
- **Generation is sovereign:** no reporting enqueue, consumer, delivery,
  shutdown, or finish-barrier failure may fail an otherwise successful
  generation. The sampler callback catches reporting failures and degrades the
  reporting run to explicitly incomplete.
- **No blocking on frame enqueue:** when a per-run FIFO is full, the dispatcher
  applies its configured report policy—coalesce a replaceable live-status item
  or drop the incoming non-canonical frame report—and records the exact missing
  sequence/range and reason. It never silently drops.
- **Reserved terminal admission, not processing or delivery:** an admitted run
  reserves a terminal slot/control-lane budget that does not compete with frame
  capacity. This guarantees exactly one logical terminal item can be admitted
  and scheduled after frame overflow; it does **not** guarantee that the worker
  begins processing it, completes it, delivers it, or reaches an observer. If
  terminal processing begins and then fails, or delivery fails,
  `reporting_status=incomplete` remains in local session/accounting state.
- **Bounded finish barrier:** the sampler waits only up to a configured finish
  deadline for accepted report work and for terminal processing if the worker
  begins it before that deadline. Timeout returns generation's normal outputs
  (or preserves its exception) and marks reporting incomplete; late writes must
  follow the declared partial/atomic artifact policy.
- **Fixed terminality:** exactly one logical terminal item is admitted and
  scheduled per admitted started run. Cleanup is idempotent whether processing
  never begins, processing fails, or delivery fails. A reporting-rejected run
  has no false `RunStarted` claim; its disabled session makes one bounded local
  terminal-accounting attempt only.

A **canonical lossless run log may not use drop semantics**. Its implementation
must either prove/provision capacity for the run's maximum admitted event set,
with bounded admission established before generation, or retain the existing
post-hoc canonical writer over the returned trace. The lossy live channel is
never silently promoted to canonical evidence.

### 7. Control remains a separate decision permanently

There is no pause, mutation, resume, bidirectional command path, or Walsh–
Hadamard transform (WHT) in this ADR. Any pause/inspect/mutate/resume gate blocks
or changes execution and therefore requires a separate ADR with its own
liveness, interrupt, tenancy, tensor-ownership, and safety analysis. Mutation
operators, including WHT, require a separately versioned declarative contract;
they do not enter through this reporting dispatcher.

## Phases and exit criteria

### Phase 1 — dispatcher MVP (next alpha on the v0.5.2 release line)

Deliver only:

- dispatcher/event contracts and immutable minimal CPU snapshots;
- the singleton bounded worker and per-run FIFO;
- parity with existing live reporting/events;
- at least one extracted report consumer proving node/consumer separation; and
- lifecycle-owner, cancellation-reason, terminal-admission, per-consumer error
  isolation, per-run/global overflow and admission, aggregate byte/event/run
  bounds, and bounded-finish tests.

The enforcement seam is an injectable/fake per-run reporting session at
`run_diffusion` plus a fake frame reporter passed to `_FrameCollector`. Core
lifecycle tests record call order and assert: one start and one logical terminal
admission per admitted invocation; only `FrameReported` comes from `on_frame`;
a caught `DiffusionCancelled` records `cancelled` before the ordinary partial result is
built/returned; non-cancellation exceptions record `failed` and still propagate;
and start/enqueue/terminal/drain failures never replace generation output or its
original exception. Dispatcher tests saturate active-run, total-event, byte,
per-run, and terminal lanes and assert fixed accounting, no rejected-run map
growth, eventual reservation release, explicit gaps/incompleteness, and bounded
waits. These are design requirements for the future Phase 1 implementation; no
runtime seam is changed by this record.

**Exit criteria:** the tests above pass; measured snapshot/enqueue overhead is
reported; reporting failures cannot fail generation; existing node signatures
and normal final outputs are unchanged; no control surface or mutation operator
is present.

### Phase 2 — complete incremental report consumers (later, unversioned)

Add incremental Trace, Tally, TokenTrace, and RunLog consumers behind an
explicit `ReportSpec`; define partial-versus-atomic artifact rules per consumer;
and permit result reuse only through stable run identity, never incidental graph
cache identity.

**Exit criteria:** all four consumers have sequence-gap/error tests, artifact
commit/cleanup tests, and equivalence tests against their canonical post-hoc
results. The RunLog path proves losslessness or keeps the post-hoc writer as its
canonical path.

### Phase 3 — measure before heavier isolation (later, unversioned)

Benchmark snapshot cost, queue contention, GIL contention, rendering/file cost,
memory bounds, and finish latency. Move heavy reporting to a subprocess/service
only if those measurements warrant the operational cost. Coordinate event
schema/versioning, replay, and multi-engine work with proposed ADR-CDG-027;
this ADR does not ratify ADR-CDG-027's current claims.

Before measurement, the Phase 3 benchmark owner and architecture reviewer must
version a threshold record covering sampler handoff overhead, queue/GIL
contention, peak accounted memory, finish latency, and reporting reliability.
The trigger is benchmark-plan approval, before the first benchmark run or result
is visible; later threshold edits invalidate that measurement set for the
isolation decision and require a fresh run.

**Exit criteria:** a reproducible benchmark report cites the pre-registered
threshold record and reports results; any process/service move has its own
reviewed lifecycle/failure contract; schema and replay ownership is reconciled
with the disposition of ADR-CDG-027.

## Rationale and trade-offs

### Positive consequences

- Diffusion resumes after a bounded copy/enqueue rather than report rendering or
  file work.
- Optional reporting cannot overturn generation success.
- Pure consumers can serve ComfyUI, MCP, tests, and later transports without
  pretending a Comfy node can execute outside its executor.
- Explicit lifecycle, gaps, and terminal status make degraded telemetry honest.

### Negative consequences

- CPU snapshot creation, queue operations, final flush, and GIL contention still
  consume sampler-call time.
- A singleton worker requires careful process shutdown and cross-run fairness.
- Bounded queues force visible loss policy for live reports; they cannot provide
  lossless canonical logs without up-front capacity or the post-hoc path.
- Consumer extraction creates temporary duplication/migration work while node
  adapters are kept stable.

## Alternatives considered

### Invoke report nodes directly from `on_frame` or the worker

Rejected: this violates executor semantics and entangles reporting with ComfyUI
node lifecycle/caching. Pure consumer logic behind adapter nodes preserves reuse
without counterfeit node execution.

### Use `PromptServer.send_sync` as the background queue

Rejected: the official API is an asynchronous client-delivery path, not a
compute scheduler. Large payloads also create serialization and websocket
backpressure on the wrong boundary.

### Unbounded queue or block-on-full

Rejected: unbounded telemetry can exhaust memory; block-on-full makes reporting
latency generation latency. Bounded, explicit loss plus a reserved terminal
lane keeps liveness and makes gaps observable.

### Start with a subprocess, broker, or service

Deferred: stronger GIL/crash isolation costs serialization, deployment,
shutdown, security, and versioned protocol work. n8n's broker/runner model shows
that heavier shape, but this project adopts it only if Phase 3 measurements
justify it.

### Restore the latent-preview monkeypatch control gate

Rejected for this decision: a blocking/mutating gate has different safety and
liveness semantics. It is not an extension of read-only reporting and must be
decided separately.

## Precedents, not standards

- **ComfyUI asset seeder:** closest managed-worker lifecycle model in ComfyUI
  core. It demonstrates owned background threads, cancellation, locking, and
  bounded joins, but it is not a timestep-reporting API.
- **ComfyUI custom server messages:** the official `send_sync` path supports
  small server-to-client events; it does not promise background computation.
- **Popular custom packs:** Impact Pack and VideoHelperSuite contain useful
  partial patterns (custom events, progress/reporting, deferred media work), but
  a survey found no complete standard combining immutable timestep snapshots,
  bounded queues, terminal reservation, consumer isolation, and canonical-log
  loss semantics. They are examples, not authority.
- **Node-RED:** `send`/`done`/`close` and message cloning are adjacent precedent
  for explicit asynchronous lifecycle and ownership transfer, not a direct
  Python/ComfyUI implementation template.
- **InvokeAI:** its event service and Socket.IO boundary show events separated
  from invocation execution and application delivery.
- **n8n:** external task runners plus broker are a later heavy-isolation
  alternative, not the MVP architecture.

## Open questions

- [ ] **Exact per-run FIFO plus global active-run, event, and byte budgets, and
      live coalescing policy.** **Owner:** Phase 1 implementer.
      **Resolution trigger:** before the first implementation PR is approved, using
      worst-case Phase 1 snapshot size, concurrency, and supported maximum
      steps/canvases; all selected bounds, admission behavior, and the drop/
      coalesce matrix must appear in code constants and saturation tests.
- [ ] **Phase 1 extracted consumer.** **Owner:** Phase 1 implementer with ComfyUI
      adapter reviewer. **Resolution trigger:** implementation-plan review;
      select the lowest-risk existing live/report consumer that can demonstrate
      pure input/output equivalence without changing its node signature.
- [ ] **Finish deadline and shutdown deadline values.** **Owner:** Phase 1
      implementer. **Resolution trigger:** benchmark-gate review before merge;
      values must be justified by measured normal completion latency and covered
      by deterministic timeout tests.
- [ ] **Phase 3 benchmark threshold values.** **Owner:** Phase 3 benchmark owner
      with the architecture reviewer. **Resolution trigger:** benchmark-plan
      approval, before the first measurement is run or revealed; publish a
      versioned threshold record for handoff overhead, queue/GIL contention,
      peak accounted memory, finish latency, and reporting reliability.
- [ ] **Whether Phase 3 warrants process/service isolation.** **Owner:** Phase 3
      benchmark owner. **Resolution trigger:** the reproducible Phase 3 report
      crosses the pre-registered versioned threshold record; otherwise the
      in-process worker remains the decision.

## Supersession relationships

**Supersedes:** the earlier **proposed**, unaccepted CDG-026 text titled
“Bidirectional control gate via latent preview callback monkeypatch.” This is a
same-number correction of a proposal; it supersedes no accepted ADR.
**Superseded by:** TBD.

## Implementation notes

This change is documentation only. No dispatcher, worker, event schema, report
consumer extraction, node-signature change, runtime behavior, test, or config
change is implemented by this ADR.

## References

- ComfyUI asset seeder source:
  https://github.com/Comfy-Org/ComfyUI/blob/master/app/assets/seeder.py
- ComfyUI official custom-message documentation (`PromptServer.send_sync`):
  https://docs.comfy.org/development/comfyui-server/comms_messages
- ComfyUI Impact Pack: https://github.com/ltdrdata/ComfyUI-Impact-Pack
- ComfyUI VideoHelperSuite:
  https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite
- Node-RED node lifecycle (`send`, `done`, `close`):
  https://nodered.org/docs/creating-nodes/node-js
- Node-RED message cloning:
  https://nodered.org/docs/user-guide/writing-functions#sending-messages-asynchronously
- InvokeAI system architecture and event service:
  https://invoke-ai.github.io/InvokeAI/contributing/ARCHITECTURE/
- n8n task runners (internal/external modes and broker):
  https://docs.n8n.io/deploy/host-n8n/configure-n8n/set-up-task-runners/
- ADR-CDG-014 — `decisions/adr-cdg-014-frame-capture-discipline.md`
- ADR-CDG-027 — `decisions/adr-cdg-027-decoupled-observability-layer.md`
