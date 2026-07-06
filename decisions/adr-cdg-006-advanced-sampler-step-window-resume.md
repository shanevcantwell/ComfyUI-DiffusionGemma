# ADR-CDG-006 — `DGemmaSamplerAdvanced`: a step-windowed, CANVAS_STATE-resumable sampler

**Status**: `proposed`
**Date**: 2026-07-05
**Related**: ADR-CDG-005 (this ADR is the **first realization** of ADR-CDG-005's
resumable `CANVAS_STATE` contract — for the `EntropyBoundScheduler`, single-block
case; it moves ADR-CDG-005 from "implementation pending" toward "implemented"),
ADR-CDG-001 (`CANVAS_STATE` socket, ONE-MINT), ADR-CDG-003 (node/engine seam),
ADR-CDG-004 (Diffusers drive seam — the pipeline this node subclasses), and
`loose-ends.md`'s 2026-07-05 `DGemmaRenoise` entry (the canvas-injection pipeline
subclass this node's engine work shares a mechanism with).

---

## Context

The operator wants a second sampler node, **`DGemmaSamplerAdvanced`**, standing
*alongside* `DGemmaSampler` (`nodes/sampler.py`), not replacing it — the additive
`KSampler` / `KSamplerAdvanced` idiom. `KSamplerAdvanced` exposes
`start_at_step`/`end_at_step`/`return_with_leftover_noise` and emits a chainable
`LATENT` that a second `KSamplerAdvanced` stage resumes from. The three asks:

1. **Start/stop "timestamps"** — a partial-range denoise, the step-window analogue.
2. **A denoised, final output** — same shape as `DGemmaSampler`: `STRING` + the
   `CanvasState` validity readout (`converged`/`committed_fraction`/`steps_used`,
   `dgemma/types.py:120-212`).
3. **A "latent" output preserving the entropic state** — a chainable payload a
   downstream node can *resume or inspect*, the way `KSamplerAdvanced`'s `LATENT`
   chains into a second stage.

Three facts about the current drive seam (ADR-CDG-004), read against the installed
diffusers 0.39.0 source, constrain every part of this design and are the reason it
needs a decision record rather than a `loose-ends.md` entry:

- **The inner denoising loop has no step-window.** It is
  `for step_idx in range(predictor_steps)` (`pipeline_diffusion_gemma.py:356`).
  There is no `start_at_step`/`end_at_step` on `DiffusionGemmaPipeline.__call__`.
- **The canvas is hardcoded random-init with no injection point.**
  `canvas = torch.randint(0, vocab, (batch, canvas_length), ..., generator=generator)`
  (`pipeline_diffusion_gemma.py:346-348`) at the head of every block. There is no
  parameter to seed it from a saved canvas — the same wall `loose-ends.md`'s
  `DGemmaRenoise` entry already priced.
- **The anneal temperature is keyed on the loop counter, not on wall-clock or on
  any carried state.** `fraction = (num_inference_steps - int(timestep)) /
  num_inference_steps`, `temperature = t_min + (t_max - t_min) * fraction`
  (`scheduling_entropy_bound.py:153-154`), and the pipeline passes
  `timestep=step_idx` (the loop counter) into `scheduler.step()`
  (`pipeline_diffusion_gemma.py:378`). **The temperature at a given step is a
  function of that step's index *and* the total `num_inference_steps`.** This is
  the load-bearing constraint: you cannot get an honest partial run by shrinking
  `num_inference_steps`, because that rescales the entire anneal.

What already exists and is reused unchanged: `dgemma/loop.py`'s `_FrameCollector`
(per-step frame capture, `:103-202`), the `on_frame` live-push seam
(`nodes/sampler.py:71-118`), `derive_canvas_state` (`dgemma/loop.py:205-264`),
the thought-channel excision, and the `DGemmaPipeline` callback-allowlist subclass
(`dgemma/loop.py:62-81`).

## Decision

### 0. This is a *lighter cousin* of the deferred `DGemmaStepSampler`, not that node

`plan.md`'s "Beyond P3 — graph-driven stepping" section defers a `DGemmaStepSampler`
(`CANVAS_STATE` in → `CANVAS_STATE` out, **one step**, the *graph* drives iteration
via a For/While loop pack this checkout doesn't ship). `DGemmaSamplerAdvanced` is
**not** that node and does not unblock it:

- `DGemmaStepSampler` puts the *loop* in the graph (envelope = a loop pack). Its
  open problem is the envelope, which `plan.md` deferred.
- `DGemmaSamplerAdvanced` keeps the loop **internal to one node call** — the
  pipeline drives a *window* of steps `[start_at_step, end_at_step)` per call. The
  envelope is the node's own body, exactly as `DGemmaSampler`'s is. No loop pack is
  needed.

It is the middle rung between monolithic `DGemmaSampler` (all N steps, one call) and
the fully graph-driven `DGemmaStepSampler` (one step, graph loops): **a chunk of
steps per call, chained node-to-node through `CANVAS_STATE`.** It shares
`DGemmaStepSampler`'s *identity* contract (ADR-CDG-005's resumable `CANVAS_STATE`)
while sidestepping its *envelope* problem. If it lands, `DGemmaStepSampler` later
becomes "this, with a 1-step window and a graph loop wrapped around it" — but that
is a separate decision, still deferred.

### 1. Reuse `DGEMMA_CANVAS_STATE`; grow `CanvasState` with an optional resume payload

The chainable "latent" output (ask #3) rides the **existing `DGEMMA_CANVAS_STATE`
socket** (ADR-CDG-001 type #3). Minting a new socket type (`DGEMMA_LATENT` or
similar) for it would be a direct **ONE-MINT violation**: ADR-CDG-005 already mints
`CANVAS_STATE` as *the* canonical type for "a sufficient statistic for resuming or
branching a denoise trajectory." Two types for one identity is exactly what
ADR-CDG-001 forbids.

`CanvasState` (`dgemma/types.py:120`) today carries only the validity-readout subset
of ADR-CDG-005's contract. This ADR grows it with an **optional** `resume` field
holding the resumable sufficient statistic — the additive growth ADR-CDG-005
explicitly predicted ("the fields grow additively as phases land"). `resume is None`
is the terminal-readout state (what today's `DGemmaSampler` emits); `resume is not
None` is the chainable state (what `DGemmaSamplerAdvanced` emits when it stops with
work left, or always, TBD per Open Question). **This node is the trigger that
realizes ADR-CDG-005**, for the `EntropyBoundScheduler` single-block slice.

Proposed dataclass delta (schema only — ADR-CDG-003, no logic here):

```text
@dataclass
class ResumeState:                      # rides inside CanvasState.resume
    canvas: Any            # torch.LongTensor snapshot (CPU), the partial canvas
                           #   AFTER the last executed step — i.e. going INTO `step`
    step: int              # schedule position to resume AT (== end_at_step reached)
    canvas_idx: int        # outer-block coordinate (0 for the single-block first cut)
    num_inference_steps: int   # the FULL schedule length N — the anneal reference
    scheduler_name: str        # "EntropyBoundScheduler" (parse-at-the-door gate)
    scheduler_config: dict     # {entropy_bound, t_min, t_max, num_inference_steps}
    generator_state: Any       # torch.Generator.get_state() ByteTensor (CPU, ~5 KB)
    prompt: str                # the encoder context; a resume MUST re-encode the
                               #   same prompt or it is a different run
    thinking: bool
    gen_length: int
    confidence: float

@dataclass
class CanvasState:
    # ... all existing validity fields unchanged ...
    resume: ResumeState | None = None   # None => terminal readout; set => chainable
```

`scheduler_config` + `scheduler_name` are non-optional inside `ResumeState` for the
same reason `CanvasTrace` carries them (ADR-CDG-001 addendum, `dgemma/types.py:92-118`):
a save-state that omits which scheduler minted it is a lying payload. For
`EntropyBoundScheduler` there is **no commit-state field to carry** — it is stateless
per-step (`scheduling_entropy_bound.py:81-91`, confirmed ADR-CDG-005). That
statelessness is *why* it is the first (and, this cut, only) supported scheduler for
resume; `BlockRefinementScheduler`'s `_committed` ratchet is deferred (Open Q, and
ADR-CDG-005 Open Q 2).

### 2. Start/stop are **discrete step indices** over a fixed schedule, not `start_t`/`end_t`

Expose `start_at_step` and `end_at_step` (INT widgets), matching `KSamplerAdvanced`'s
literal naming, indexing into a schedule of fixed length `num_inference_steps`. The
node runs step indices `start_at_step .. end_at_step - 1` (Python `range` semantics);
the resumed/emitted state's `step` is `end_at_step`.

Rejected the continuous-time framing (`start_t`/`end_t` as floats) for two grounded
reasons:

- **Three-way `t` name collision.** `t_min`/`t_max` are already taken — they are the
  *temperature endpoints* of the anneal (`scheduling_entropy_bound.py:70-73`), exposed
  as widgets on `DGemmaSampler` (`nodes/sampler.py:136-137`). Separately,
  `DiffusionFrame.t` (`dgemma/types.py:73`) is the *anneal fraction* (1.0→0). A
  `start_t`/`end_t` pair would be a third, unrelated "t" on the same node —
  legibility poison in a pack whose entire thesis (ADR-CDG-001) is that payloads and
  labels mean exactly what they say.
- **The discrete axis is the one the loop and the frames actually use.** The loop
  iterates integer `step_idx` (`pipeline_diffusion_gemma.py:356`); frames are keyed on
  `step_idx` (`dgemma/types.py:73`); the live push reports `step_idx`
  (`nodes/sampler.py:106`). Step indices are the honest, already-present coordinate.

**`num_inference_steps` stays the full-schedule length N and is the anneal reference**
— the window narrows *which* steps run, never how many the anneal is normalized over.
Shrinking `num_inference_steps` to "stop early" is rejected (Alternatives, Option C):
it silently rescales the temperature trajectory.

Legibility cost, named: `DGemmaSampler` has no `start_at_step`/`end_at_step`; the two
sampler nodes therefore differ in step-vocabulary surface. This is acceptable and
mirrors `KSampler`/`KSamplerAdvanced` (the basic one hides the window; the advanced
one exposes it). It is *not* a vocabulary *inconsistency* — both use `step` as the
unit; the advanced node merely exposes a window into it.

### 3. State fidelity: **snapshot-based, in-memory handoff first**; disk persistence deferred but unblocked

`ResumeState` captures **value snapshots**, not live-object references: the canvas as
a CPU `LongTensor`, the generator via `generator.get_state()` (a ~5 KB CPU
`ByteTensor`, ADR-CDG-005 field 4), the scheduler as `(name, config dict)` to be
*reconstructed* (`EntropyBoundScheduler(**config)`), never a shared live scheduler
instance. The first cut hands this snapshot node-to-node **within one ComfyUI graph
execution** (in-memory, no disk) — which fully satisfies the operator's stated ask
(chain stage-1 → stage-2). It does **not** implement disk save/load.

Why snapshot rather than the simpler live-reference handoff:

- **Fork-safety.** A live `torch.Generator` shared by reference between two
  downstream Advanced samplers (a graph fork) would have both branches advance the
  *same* RNG object — nondeterministic cross-talk. A `get_state()` snapshot lets each
  resume `set_state()` an *independent* generator seeded to the saved point, so a fork
  produces two bit-identical branches instead of two corrupted ones.
- **No persistence rework debt.** Because the snapshot is already value-typed
  (ints, a config dict, two CPU tensors), Phase C (disk persistence) becomes "marshal
  these to a file + add save/load nodes," not a redesign. Live references would
  dead-end on serialization.

Full **cross-session disk persistence** (a `DGemmaSaveState`/`DGemmaLoadState` pair, a
versioned on-disk schema, GPU↔CPU tensor marshalling, a file location decision) is
**deferred to Phase C** — real scope the stated ask does not require, and ComfyUI has
no native "serialize an arbitrary object socket to disk" idiom to lean on. The
snapshot design makes it a clean additive follow-up, not a rewrite.

### 4. Engine: a step-windowed pipeline subclass; single-block, `EntropyBoundScheduler` only

The partial-run and resume mechanics live in `dgemma/` (ADR-CDG-003), as a new
`DiffusionGemmaPipeline` subclass (extending the existing `DGemmaPipeline`,
`dgemma/loop.py:62`) that overrides the inner block loop to accept two things the base
`__call__` cannot express:

- **An injected initial canvas** (skip the `torch.randint`, `:346-348`, when a resume
  canvas is provided) — the same mechanism `loose-ends.md`'s `DGemmaRenoise` priced at
  ~50–100 lines.
- **A step-window** — iterate `range(start_at_step, end_at_step)` instead of
  `range(predictor_steps)` (`:356`), while leaving `num_inference_steps` (hence the
  anneal, `:378` → `scheduling_entropy_bound.py:153`) at the full N.

Everything numeric stays in the reference implementation: the override still calls
`self.model(...)` (`:365`) and `self.scheduler.step(...)` (`:377`) unchanged. This is
deliberately the ADR-CDG-004 posture — *drive* the reference loop, do not
*reimplement* its numerics — applied to the loop-control layer. A new engine function
(sketch: `run_diffusion_windowed(...)`, sibling to `run_diffusion`) constructs the
subclass, threads the generator (fresh from `seed`, or `set_state()` from
`ResumeState.generator_state`), reuses `_FrameCollector` + `on_frame` untouched, and
assembles the returned `CanvasState` with its `resume` payload from the last captured
frame's canvas.

**Scope of the first cut, stated as a boundary not a silence:**

- **Single canvas block only** (`gen_length ≤ canvas_length`, i.e. `num_canvases == 1`,
  `pipeline_diffusion_gemma.py:279`). Multi-block resume needs the committed-prefix +
  one-prefill KV recompute (ADR-CDG-005 excludes the KV cache precisely because it is
  recomputable) and is deferred (Open Q).
- **`EntropyBoundScheduler` only** — stateless per-step, so its sufficient statistic is
  `(canvas, step, config, generator_state)` with no commit-mask to restore.
  `BlockRefinementScheduler` resume is deferred to the ADR-CDG-005 Open-Q-2 audit.
- **`confidence_threshold` early-stop is preserved** (`:413-426`): if the window
  converges before `end_at_step`, it stops early and the emitted state reflects the
  real stop step — honest, and consistent with `converged` semantics.

### 5. Node contract (`nodes/sampler_advanced.py`, thin per ADR-CDG-003)

```text
INPUT_TYPES:
  required:
    model               DGEMMA_MODEL
    prompt              STRING (multiline)             # ignored/validated on resume
    seed                INT                            # ignored on resume (state carries generator)
    num_inference_steps INT   default 48               # the FULL schedule length N
    start_at_step       INT   default 0                # 0 for fresh; >0 REQUIRES a resume input at that step
    end_at_step         INT   default = num_inference_steps
    t_min, t_max, entropy_bound, confidence, gen_length, thinking   # as DGemmaSampler
  optional:
    canvas_state        DGEMMA_CANVAS_STATE            # the resume input; when present, resume from it
  hidden:
    unique_id           UNIQUE_ID                      # reuses the P3 live-push routing

RETURN_TYPES = ("STRING", "DGEMMA_CANVAS_STATE", "DGEMMA_CANVAS_TRACE")
RETURN_NAMES = ("text", "canvas_state", "canvas_trace")
```

The `canvas_state` socket is **both an optional input and an output of the same type**
— the `KSamplerAdvanced` `LATENT`-in/`LATENT`-out symmetry, expressed in this pack's
native type. `sample()` stays a pure unpack → call one `dgemma.*` function → wrap-tuple
(ADR-CDG-003); all validation lives engine-side (parse-at-the-door, below).

**Parse-at-the-door contract** (engine-side, honest-failure over silent divergence —
this is the enforcement surface for "no lying save-state"):

- `start_at_step > 0` with **no** `canvas_state` input → raise (cannot start
  mid-schedule with nothing to denoise from).
- `canvas_state` present but `resume is None` (e.g. wired from basic `DGemmaSampler`)
  → raise ("this CANVAS_STATE is a terminal readout, not resumable").
- `canvas_state.resume.scheduler_name != "EntropyBoundScheduler"` → raise (unsupported
  resume scheduler this cut).
- On resume, the schedule-defining params (`num_inference_steps`, `t_min`, `t_max`,
  `entropy_bound`) and `prompt` are taken **from the state**, and a widget value that
  disagrees raises rather than silently winning — because a resumed run under a
  different N or a different prompt is a different trajectory wearing the same
  save-state (the anneal-relativity footgun made loud). See Open Q on whether to
  hard-lock (grey out) those widgets in the UI instead.

## Rationale

### Positive Consequences

- **Realizes ADR-CDG-005's contract with a bit-exact correctness anchor.** The
  resume path is testable headless (ADR-CDG-003): run `[0, N)` → `text_full`; run
  `[0, k)` → state, resume `[k, N)` → `text_resumed`; assert `canvas_ids` identical.
  That equality is the difference between a *rewind* and a *rerun* (ADR-CDG-005's own
  distinction) and is the enforcement surface that keeps the save-state honest.
- **No ONE-MINT breach, no new socket.** The chainable output is the type ADR-CDG-005
  already minted; readers wire `CANVAS_STATE → CANVAS_STATE` and it means one thing.
- **The anneal-relativity trap is closed by construction**, not by discipline:
  `num_inference_steps` is authoritative and carried in the state; a mismatched resume
  raises.
- **Reuses the whole P1–P3 spine** (`_FrameCollector`, `on_frame` live push,
  `derive_canvas_state`, excision) — the live view works on the Advanced node for free.

### Negative Consequences

- **A `__call__` override tracks upstream diffusers.** The windowed subclass duplicates
  the block-loop body (`pipeline_diffusion_gemma.py:318-435`) to change two lines of
  loop control; a diffusers 0.39.x → 0.4x change to that method must be re-reconciled.
  Priced and accepted, the same category of cost ADR-CDG-004 Open-Q-(b) accepted for
  the custom-curve `step()` override. Enforcement: a subclass-parity test that fails
  loudly if the base method's signature/shape drifts.
- **`CanvasState` now carries an optional heavy payload** (a canvas tensor + generator
  state) alongside its light validity readout. Two roles on one dataclass; mitigated by
  the explicit `resume is None` discriminator and parse-at-the-door.
- **Single-block + EntropyBound-only** is a real capability gap versus the full
  ADR-CDG-005 vision; named as Open Questions, not hidden.

## Alternatives Considered

### Option A: Mint a new `DGEMMA_LATENT` (or `DGEMMA_RESUME`) socket for the chainable output
**Why rejected:** Direct ONE-MINT violation. ADR-CDG-005 already mints `CANVAS_STATE`
as the resumable-trajectory type; a second type for the same identity is what
ADR-CDG-001 exists to forbid. The validity readout and the resume payload are two
*facets of one state*, not two types.

### Option B: Continuous `start_t`/`end_t` float inputs
**Why rejected:** Three-way `t` collision (`t_min`/`t_max` temperatures,
`DiffusionFrame.t` fraction, and now `start_t`/`end_t`) on a pack whose thesis is
label honesty; and the loop's real coordinate is the integer `step_idx`. Step indices
match `KSamplerAdvanced` and the existing frame keying.

### Option C: "Stop early" by shrinking `num_inference_steps`
**Why rejected — this is the load-bearing rejection.** The anneal temperature is
`t_min + (t_max - t_min) * (N - step)/N` (`scheduling_entropy_bound.py:153-154`).
Running `num_inference_steps = k` to stop at step `k` reruns the *entire* anneal
compressed into `k` steps — a hotter, different trajectory — not the first `k` steps of
the N-step schedule. It would silently diverge from what a resumed continuation
expects. The window must narrow *which* steps run while N (the anneal denominator)
stays fixed.

### Option D: Extract the block-loop step primitive into `dgemma/` and own the loop
**Why rejected for the first cut:** `plan.md` notes the loop body factors cleanly
(KV populate → mask build → forward → `scheduler.step`), and this is the eventual path
for a truly graph-driven `DGemmaStepSampler`. But reimplementing the KV-populate /
mask-build / block-encode *orchestration* (`pipeline_diffusion_gemma.py:318-343`) in
this pack reintroduces exactly the reference-drift risk ADR-CDG-004 eliminated by
choosing to *drive* the pipeline rather than reimplement it. The subclass override
keeps `self.model(...)`/`self.scheduler.step(...)` as the numeric authority. Revisit
Option D only if/when `DGemmaStepSampler`'s graph-driven envelope is actually built.

### Option E: Live-object handoff (share the `torch.Generator`/scheduler by reference)
**Why rejected:** Simplest, but a graph fork would have two branches mutate one live
RNG (nondeterministic), and it dead-ends on disk persistence. The `get_state()`
snapshot is ~5 KB, fork-safe, and persistence-ready (Decision §3).

### Option F: Disk persistence in the first cut
**Why rejected:** Scope creep. The stated ask (chain into a downstream Advanced
sampler) is satisfied in-memory; disk save/load needs a versioned schema, device
marshalling, save/load nodes, and a file-location decision. Deferred to Phase C, kept
cheap by the snapshot design.

### Option G: Callback-sentinel stop, no pipeline subclass
A callback raising a sentinel at `end_at_step` can *stop* a run without a subclass.
**Why rejected:** It handles only the stop side. The **resume/start** side is
impossible without injecting the saved canvas (no injection point, `:346-348`) *and*
starting the loop at `step_idx = k` with the correct annealed temperature (the loop
counter is uncontrollable from the callback). Since the subclass is needed for resume
anyway, it cleanly owns both bounds.

## Phased Roadmap

- **Phase A — Engine (headless, the keystone).** `DGemmaWindowedPipeline` subclass
  (init-canvas injection + `[start_at_step, end_at_step)` window, single-block,
  EntropyBound); `ResumeState` dataclass + `CanvasState.resume` field
  (`dgemma/types.py`); `run_diffusion_windowed` engine fn + resume reconstruction
  (`dgemma/loop.py`). **Depends on:** nothing new (reuses `_FrameCollector`).
  **Verifiable when it lands:** the bit-exact split-vs-continuous pytest
  (`[0,k)`+`[k,N)` == `[0,N)`), plus a fork-safety test (two resumes from one state →
  identical output). This is where the design's correctness is proven; touches
  `dgemma/loop.py`, `dgemma/types.py`, `tests/`.
- **Phase B — Node adapter.** `nodes/sampler_advanced.py` (thin, ADR-CDG-003), the
  `canvas_state` in/out socket, parse-at-the-door validation wired to the engine,
  `unique_id` live-push reuse; register in `__init__.py`. **Depends on:** Phase A.
  **Verifiable when it lands:** a two-stage graph (stage-1 `[0,k)` → stage-2 `[k,N)`)
  whose `text` matches a single-stage `[0,N)` run, banked as an `examples/*.api.json`
  like P2/P3 did.
- **Phase C — Cross-session persistence (deferred, gated on ask).**
  `DGemmaSaveState`/`DGemmaLoadState` nodes, versioned on-disk schema, GPU↔CPU
  marshalling. **Depends on:** Phase A's snapshot shape (already serialization-ready).
  **Verifiable:** save → new process → load → resume == in-memory resume.
- **Phase D — Multi-block + `BlockRefinementScheduler` resume (deferred).** Committed
  prefix + one-prefill KV recompute for `num_canvases > 1`; `_committed`-ratchet
  save/restore per ADR-CDG-005 Open Q 2. **Depends on:** Phase A + the ADR-CDG-005
  scheduler-audit trigger.

Phase A + B is the operator's stated ask. This design is more than one PR of work;
if the orchestrator needs sub-problem-level acceptance criteria and per-file
dependencies before dispatching build work, a `decompose-problem` pass over Phases A/B
is warranted — this ADR sequences but does not enumerate to that grain.

## Risk and Observability

- **Anneal-relativity silent divergence** (the top risk). Mitigated structurally:
  `num_inference_steps` carried in `ResumeState` and authoritative on resume; a
  mismatched widget raises (parse-at-the-door). Observed via the bit-exact Phase A
  test as the standing enforcement surface.
- **Generator aliasing on graph forks.** Mitigated by snapshot (`get_state`) +
  per-resume `set_state` (Decision §3, Option E rejection). Observability: the
  fork-safety test.
- **Partial-canvas garbage text.** Stopping at `end_at_step < N` (the
  `return_with_leftover_noise` analogue) decodes a canvas still holding renoise —
  `text` will look unfinished. This is honest and *already covered*: the
  `CanvasState` validity readout (`converged=False`, low `committed_fraction`,
  ADR-CDG-001 time-axis-lying-payload discipline) says so on the same output.
- **Subclass drift from upstream diffusers.** Mitigated by a subclass-parity test;
  degradation is a loud test failure on a version bump, not a silent numeric drift.
- **Two roles on `CanvasState`.** A downstream consumer must check `resume is not None`
  before resuming; the parse-at-the-door raise makes misuse loud rather than silent.

## Open Questions

- [ ] Should `DGemmaSamplerAdvanced` emit `resume` **always**, or only when it stopped
      with work left (`end_at_step < N` / not converged)? Emitting always makes any run
      chainable; emitting conditionally keeps "this is finished" legible in the payload.
      **Resolution trigger:** decide in Phase B against the two-stage example graph.
- [ ] On resume, **hard-lock** the schedule-defining widgets (grey them out in the
      frontend) vs. **validate-and-raise** on mismatch? Locking is friendlier but needs
      JS; raising is engine-only and ships in Phase A/B. **Resolution trigger:** Phase B
      UX pass. Default to validate-and-raise for the first cut.
- [ ] Multi-block resume (`num_canvases > 1`): committed prefix + one-prefill KV
      recompute. **Resolution trigger:** when `gen_length > canvas_length` support is
      actually requested (Phase D).
- [ ] `BlockRefinementScheduler` resume: does external step-windowing desync its
      `steps_done` quota even with `_committed` restored? **Resolution trigger:**
      inherits ADR-CDG-005 Open Q 2 verbatim — EntropyBound (stateless) is the first
      target precisely to sidestep this; revisit only after Phase A proves the
      stateless case.

## Supersession Relationships

**Supersedes:** none.
**Realizes (partial):** ADR-CDG-005 — this ADR is the first implementation of its
resumable `CANVAS_STATE` contract, for the `EntropyBoundScheduler` single-block case.
When Phase A lands, ADR-CDG-005's status should move from "accepted (implementation
pending)" toward "accepted (partially implemented — EntropyBound single-block, per
ADR-CDG-006)". *(That status edit and the `decisions/README.md` index row for
ADR-CDG-006 are follow-up writes outside this artifact — see the design note returned
with this ADR.)*
**Superseded by:** TBD (a future `DGemmaStepSampler` ADR may generalize the envelope).

## References

- `pipeline_diffusion_gemma.py:279,318-435,346-348,356,378,413-426` (installed
  diffusers 0.39.0, `.venv/.../diffusers/pipelines/diffusion_gemma/`)
- `scheduling_entropy_bound.py:81-91,153-154` (same package — statelessness + anneal)
- `dgemma/loop.py:62-81,103-202,205-264,432-590` (existing spine reused)
- `dgemma/types.py:120-212` (`CanvasState` grown here)
- `nodes/sampler.py:71-118,121-191` (basic sampler + live-push seam reused)
- `plan.md` "Beyond P3 — graph-driven stepping" (the deferred `DGemmaStepSampler`)
- `loose-ends.md` 2026-07-05 `DGemmaRenoise` (shared canvas-injection subclass mechanism)
