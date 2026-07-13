# Architecture: ComfyUI-DiffusionGemma

This document defines the **layering / single-contract** invariant of
`ComfyUI-DiffusionGemma` and makes the boundary between the core and its surfaces
enforceable. Its purpose is to distinguish a valid composition from a violation —
not to describe what the code currently does, but to state what it must do and to
name the gaps that remain.

The **decided** architecture is ADR-CDG-008's MCP-center topology: one core
(`dgemma/`), peer surfaces over it (`surfaces/comfyui/`, `surfaces/mcp/`),
analysis as a downstream consumer. Much of that target is **not yet implemented** —
every strong-register claim below carries a `path:symbol` citation of the code that
enforces it *or* the explicit `NOT-YET-IMPLEMENTED` token naming the R-item /
CDG-008 phase that will create it. The strong words describe the target; the
*Current conformance* section is what stops a reader taking them as a description of
reality.

Doctrine is included by reference, not duplicated (repo `CLAUDE.md`, opinion
locality): ground-physics invariants live in
`../operating-doctrine/ground-physics/GROUND_PHYSICS.md` and the enforceable rule
set in `../operating-doctrine/ground-physics/CODE_CONSTITUTION.md`. The handles
cited below (`ONE-DOOR`, `STATELESS-CORE`, `ONE-MINT`,
`EMIT-CANONICAL / PARSE-AT-THE-DOOR`, `IDENTITY⊥ENVELOPE`,
`CONSERVE-ACROSS-THE-DATA-BOUNDARY`) are resolved there.

---

## The invariant (read this first)

Seven rules. All seven apply simultaneously.

1. **One core, one contract.** `dgemma/` is the sole contract: its public face is
   `load_model` (`dgemma/model.py:157`) and `run_diffusion` (`dgemma/loop.py:465`,
   returning `(text, CanvasState, CanvasTrace)`). Every surface reaches the model
   *only* through these two functions; no surface accretes denoising logic. The
   perception firewall is structural: `dgemma/` imports with **zero ComfyUI
   present**, and a subprocess test asserts no `comfy.*` / `nodes.*` module leaks
   into `sys.modules` after `import dgemma` — the *absence* of a `comfy` package in
   the venv is the enforcement, not a maintained stub. This bounds *authoring*
   across the seam (a core module cannot `import comfy`), not *actuation* (a running
   surface still holds the model object). *(→ ADR-CDG-003, ADR-CDG-008 · `ONE-DOOR`)*

2. **Surfaces are peers over the core.** MCP is the base surface; the ComfyUI node
   graph is one surface among peers, no privileged position. A new surface is added
   by wrapping `load_model` + `run_diffusion` — the same move ADR-CDG-003 defined
   for nodes, generalized to `surfaces/*`. No `for`-loop-over-denoising-steps may
   appear in a surface body. *(→ ADR-CDG-008 · `ONE-DOOR`)*

3. **Analysis is a downstream consumer, not core work.** The core **emits** the
   canonical `CanvasTrace`; derived analysis **parses** it. Analysis code must live
   outside `dgemma/`'s import graph, so the base contract can be asserted to import
   no analysis. *(→ ADR-CDG-008 · `EMIT-CANONICAL / PARSE-AT-THE-DOOR`)*

4. **Socket vocabulary is minted once, surface-side; identity is core-side.** The
   `DGEMMA_*` socket strings are ComfyUI *envelope*: they belong to the ComfyUI
   surface and are declared in one mint module (target
   `surfaces/comfyui/socket_types.py`), with no inline `DGEMMA_*` literal at any
   other site. The payloads riding those sockets are the `dgemma/types.py`
   dataclasses — the *identity*. Envelope and identity are orthogonal
   (`IDENTITY⊥ENVELOPE`): the socket string may change with the surface layout; the
   dataclass may not. *(→ ADR-CDG-001, ADR-CDG-008, #35 R2 · `ONE-MINT`)*

5. **Payloads mean what they say.** A native socket type carries the real payload or
   it is a lying payload: an entropy budget is never disguised as a `SIGMAS` tensor;
   a `DISTRIBUTION` carries the real per-position distribution or it is the
   scalar-shadow trap (#14) reborn. Every socket is an
   `EMIT-CANONICAL / PARSE-AT-THE-DOOR` surface — validated at ingress, fail on
   unknown. *(→ ADR-CDG-001 · `EMIT-CANONICAL / PARSE-AT-THE-DOOR`)*

6. **The core is stateless across runs; only the model load persists.** No mutable
   run-state (a scheduler with a mutated `config`, an accumulated pin mask, a
   forward hook) survives a `run_diffusion` call. Two identical calls on one loaded
   model yield identical effective-knob telemetry. The ~53 GB model load is the
   *only* persisted object; every run constructs a fresh scheduler / canvas / run
   state. *(→ ADR-CDG-008 Correction 1, #35 R5/F4/F5 · `STATELESS-CORE`)*

7. **Step-end intervention enters as declarative payloads through one door.**
   `run_diffusion` widens only by declarative payloads (`constraints=`,
   `control_signals=`, `capture=`), each validated at ingress. A surface never
   supplies an executable participant (a closure, a hook) — the callback's
   `pipe.model` reachability is explicitly **not** a sanctioned installation path;
   that second door is foreclosed. The only executable crossing is the read-only
   `on_frame` observer, whose return is ignored. *(→ #35 delta Corrections 2/3,
   ADR-CDG-010/011 · `ONE-DOOR`)*

---

## The layers

Top (consumer) to bottom (substrate). Directory names in **target** form per
ADR-CDG-008 §Decision-2; the current on-disk names are noted where they differ.

### Consumers — derived analysis (target: `consumers/` **or** `surfaces/analysis/` — OPEN)

**What lives here:** the pure trace-analysis functions currently in
`dgemma/sampling.py` (`build_commit_heatmap`, `build_avalanche_curve`,
`corroborate_no_mask_token`, `MaskTokenCorroboration`). Their home is
`NOT-YET-IMPLEMENTED` — they currently sit inside the core's import graph and are
re-exported by `dgemma/__init__.py:26-31,49-51` (the tracked debt, CDG-008 Phase 3).

**Rules:**
- Parses an already-captured `CanvasTrace`; never re-derives what the core emitted,
  never drives the model.
- Imports the core's contract type; the core imports nothing from here.
- **OPEN:** `consumers/` vs `surfaces/analysis/` naming is unresolved (ADR-CDG-008
  Open Question #1) — `DGemmaTrace` is *also* a ComfyUI surface node wrapping the
  analysis. **Resolution trigger:** settle during the Phase-3 relocation `plan`
  pass, before any file moves.

### Surface tier — peer surfaces over the one contract (target: `surfaces/*`)

**What lives here:**
- `surfaces/comfyui/` — the ComfyUI node graph. **Currently `nodes/`** (`loader.py`,
  `sampler.py`, `trace.py`, `frames_image.py`) + top-level `web/`; the move to
  `surfaces/comfyui/` (with `web/` → `surfaces/comfyui/web/`) is
  `NOT-YET-IMPLEMENTED` (CDG-008 Phase 1).
- `surfaces/mcp/` — the base MCP surface over `load_model` + `run_diffusion`.
  `NOT-YET-IMPLEMENTED` (CDG-008 Phase 2; transcribe `semantic-kinematics-mcp` with
  the two corrections — stateless run-state, keep the automated boundary test).

**Rules:**
- Each `surfaces/*` module is a thin adapter: unpack args → call one `dgemma.*`
  function → wrap the result. No denoising-step loop in a surface body (ADR-CDG-003).
- Holds no core logic; the logic isn't in the surface, so no surface can accrete
  what the others can't reach.
- The ComfyUI surface's socket strings are minted in its own `socket_types.py`
  (rule 4); the MCP surface's state manager persists only the model load (rule 6).

### Core — the one contract (`dgemma/`, already surface-neutral)

**What lives here:** the model, the types, the denoising loop, plus the analysis math
*until Phase 3 relocates it* (the tracked debt). `dgemma/model.py` (load),
`dgemma/loop.py` (drive), `dgemma/types.py` (contract dataclasses).

**Rules:**
- Imports and runs with zero ComfyUI present (`dgemma/__init__.py`; enforced by
  `tests/test_seam.py:36-63`).
- Emits the canonical `CanvasTrace`; never parses derived analysis.
- Holds no cross-run mutable state (rule 6). The load is persisted, the run is
  stateless.
- Widens `run_diffusion` by declarative payloads only (rule 7).

---

## What the invariant does NOT govern (out of scope)

- **The published repo name `ComfyUI-DiffusionGemma`** — conserved identity
  (`IDENTITY⊥ENVELOPE`), registry-mirrored and remote-live. The internal directory
  vocabulary changes (`nodes/` → `surfaces/comfyui/`); the repo name does not.
  Renaming the repo is explicitly out of scope (ADR-CDG-008 Decision-2, Option B
  rejected). This is a scoping fact, not an exception: the layering invariant
  governs internal envelope, not the conserved external handle.

- **`CanvasTrace` (and the `dgemma/types.py` contract dataclasses) living in the
  core** — the emitted canonical type sits core-side as the contract surface both
  the core and its consumers depend on. A consumer importing it is not a layering
  violation; it is the contract being consumed at the door. *(ADR-CDG-008 Open
  Question #2, default: trace type stays in `dgemma/`, analysis functions move out.)*

- **The GGUF / llama.cpp inference backend** (ADR-CDG-007) — a
  graduation-triggered, inference-only alternate backend, not the primary
  transformers-load / diffusers-drive path. It sits beside the drive seam, not
  through it, and is not part of the current contract surface.

- **Substrate the core legitimately shares** — `torch`, `transformers`,
  `diffusers` are shared substrate beneath every layer; a surface and the core both
  importing `torch` is not a seam crossing.

---

## Diagram

```
   consumers/  (analysis: parses CanvasTrace)          -- NOT-YET-IMPLEMENTED (Phase 3)
        |  parses
        v
+-----------------------------------------------------------+
| surfaces/                                                 |
|   comfyui/  (was nodes/ + web/)   mcp/  (NEW base surface)|  -- peers
+-----------------------------------------------------------+
        |  load_model + run_diffusion  -- THE ONE CONTRACT (the door)
        v
+-----------------------------------------------------------+
| dgemma/   core -- surface-agnostic, zero ComfyUI present  |
|   model.py (load) . loop.py (drive) . types.py (contract) |
+-----------------------------------------------------------+
        |
   torch . transformers . diffusers   -- shared substrate (out of scope, beside not through)
```

The contract boundary is the `load_model` + `run_diffusion` line; every governed
surface arrow crosses it. Shared substrate sits beside the layers, not through the
door.

---

## The step-end intervention architecture (decided target, per #35)

The expansion (liquid-phase bench, #23/#28 grounding) lands **core-side of the
seam**, so every surface inherits it. Its shape is decided but
`NOT-YET-IMPLEMENTED`; it is cited to #35's R-items and ADR-CDG-010/011.

- **Engine-internal ordered composite** (R1, replaces the single hardcoded callback
  binding at `dgemma/loop.py:582`). The composite holds only engine-built
  participants — β-renoise, walker, pin, capture. Ordering is fixed:
  **capture runs before any canvas-writer** (so capture sees pre-pin,
  model-committed truth), **β-rebuild before pin**, **pin is the last writer**. The
  `pinned_mask` (model-committed vs constraint-asserted) rides each frame — else the
  trace lies (ADR-CDG-010). `NOT-YET-IMPLEMENTED` (R1).

- **Live view is not a composite participant** (#35 delta Correction 2). It stays on
  the existing engine-side `on_frame` read-only observer seam
  (`nodes/sampler.py:136-159` pattern; `run_diffusion(on_frame=…)`,
  `dgemma/loop.py:477`): receives a built `DiffusionFrame`, return ignored,
  structurally read-only, needs no position among canvas-writers. Pre-pin truth
  reaches it as *frame fields* (`pinned_mask`, effective knobs), not by observer
  ordering. This is the **only executable crossing** the surface owns. *In force
  today as a read-only observer* (`nodes/sampler.py:114-161`, `_build_on_frame`).

- **Declarative payloads on `run_diffusion`** (`constraints=`, `control_signals=`,
  `capture=`) — validated at ingress: schedule length == steps; control values
  within declared binding range; constraint ids in-vocab; fail on unknown. Foreign
  callables are rejected as a design (#35 delta Correction 3): they are
  unvalidatable at ingress and would let a surface return `{"canvas": …}` —
  surface-resident sampling logic, CDG-008's forbidden shape. Any raw-participant
  escape hatch requires its own ADR. `NOT-YET-IMPLEMENTED` (ADR-CDG-011 clauses).

- **Two-mechanism model for givens/constraints** (ADR-CDG-010, grounded in #28): a
  logit mask shapes *what commits* (a masked cell reads ~zero entropy, commits
  first — most-constrained-first propagation made literal); canvas re-assertion
  guarantees *what conditions* (rejected positions are renoised over the full vocab,
  so a given cell must be re-asserted each step or the forward pass conditions on
  garbage); givens use both. `NOT-YET-IMPLEMENTED`.

- **Forward-hook lifecycle context manager** (R5, F4): the logit mask is the
  engine-installed forward hook on `pipe.model` (the only logit door per #28 —
  callback-returned `{"logits": …}` is silently discarded). Invariant: **no hook
  survives a `run_diffusion` call**, tested clean and raising. `NOT-YET-IMPLEMENTED`.

- **Control signals as CV / LFO** (ADR-CDG-011, grounded in #23): a unitless
  per-step control signal (precomputed tensor — step count is known pre-run, so
  synth semantics survive ComfyUI's one-shot declarative executor). Units are
  declared at the **binding**, not carried by the signal (the CV principle;
  binding = parse-at-the-door). The engine walker indexes bound signals by
  `step_idx` and mutates `scheduler.config` live; `num_inference_steps` is
  non-mutable (ingress reject — #20's desync mechanism). `t_min=t_max=v` is the
  exact-per-step-temperature mechanism. Effective-knob telemetry = the values the
  scheduler actually read, riding the frame. Walker prepares the next step; capture
  records the finished step. `NOT-YET-IMPLEMENTED`.

The six bench seams the expansion factors toward (`DISTRIBUTION`, `SCHEDULE`/control
signal, pin/mask, sampling operator, `KV_CACHE`, `CANVAS_STATE`) are inventoried in
`docs/experiments/liquid-phase-decoding/concept.md`; each is a native socket under
rule 5, unbuilt except the in-callback pin (proven) and `CANVAS_STATE` (designed,
ADR-CDG-005/006).

---

## Why one core, one contract

**Why a single door.** The pack's whole point is per-step instrumentation, which has
to be developed and tested from a bare script, not from inside a live node call. A
core that imports with zero ComfyUI present is testable in isolation; a surface that
can only wrap the two contract functions cannot accrete logic the other surfaces
can't reach. One contract means a new envelope (CLI, agent tool, web API) is a
wrapper, never a fork.

**Why stateless across runs.** DiffusionGemma's ~53 GB load cannot reload per call,
so the model object persists across ComfyUI executions — which is exactly the danger:
an un-torn-down forward hook from run A shapes run B's logits (F4), and a cached
scheduler carries a prior run's mutated dims forward (the observed 25-vs-29 heatmap
frame-count mismatch, F5). Persisting *only* the immutable load and rebuilding all
run-state is what keeps two identical calls identical.

**Why analysis is a consumer, not core.** The core emits `CanvasTrace` once; analysis
parses it. Keeping analysis out of the core's import graph lets a test assert the
base contract imports no analysis — turning a prose boundary
(`dgemma/sampling.py`'s docstring claims consumer status while
`dgemma/__init__.py:26-31` contradicts it) into an enforced one.

**Why declarative payloads, not closures.** A surface-supplied callable is
unvalidatable at ingress and re-opens the door the core closed: through `pipe.model`
it could install sampling logic that belongs in the core. Declarative payloads are
checkable at the door; the forbidden shape (a surface returning a canvas) becomes
structurally unrepresentable.

---

## Current conformance (honest) — Branch B (audited)

The invariant above is the target. The code **partially** conforms: the core/surface
seam (rules 1, 2 in part, 5 in part) is in force; the MCP surface, the surface-side
naming, the analysis relocation, the mint module, the cross-run statelessness
enforcement, and the entire step-end intervention layer (rules 3, 4, 6, 7 and most
of 2) are not yet implemented.

| Violation | Why it breaks the invariant | Evidence (`path:symbol`) | Resolved by |
|-----------|----------------------------|--------------------------|-------------|
| Surface tier is named `nodes/` (a ComfyUI word) + top-level `web/`; no `surfaces/` parent, no peer MCP surface | Rule 2 — the name puts ComfyUI at the center, leaving no room for peer surfaces | `nodes/loader.py`, `nodes/sampler.py`, `nodes/trace.py`; `__init__.py:53` (`WEB_DIRECTORY = "./web"`) | `NOT-YET-IMPLEMENTED` — CDG-008 Phase 1 (R2 rides the move) |
| No MCP surface exists | Rule 2 — MCP is the decided base surface | `NOT-YET-IMPLEMENTED` — no `surfaces/mcp/` on disk | CDG-008 Phase 2 |
| Analysis lives inside the core's import graph and is re-exported by the core's public face | Rule 3 — analysis is a consumer; the core must not export it | `dgemma/__init__.py:26-31,49-51` (re-exports `build_commit_heatmap`, `build_avalanche_curve`, `corroborate_no_mask_token`); `dgemma/sampling.py` (bodies) | `NOT-YET-IMPLEMENTED` — CDG-008 Phase 3 (relocate) + Phase 4 (boundary test) |
| Socket strings re-typed as bare literals per node site; no mint module | Rule 4 — `ONE-MINT` violated; the vocabulary is authored N times | `nodes/loader.py:46` (`RETURN_TYPES = ("DGEMMA_MODEL",)`), `nodes/sampler.py:172,202`, `nodes/trace.py:80` — inline `DGEMMA_*` literals | `NOT-YET-IMPLEMENTED` — #35 R2 (mint module + grep-gate; interim `nodes/socket_types.py` → `surfaces/comfyui/socket_types.py` post-Phase-1) |
| ~~Single hardcoded callback binding; no composition / ordering / exception layer~~ **RESOLVED** | Rule 7 — five expansion participants want the slot with ordering semantics | `dgemma/composite.py:StepEndComposite` (fixed order: cancellation → capture → beta-rebuild → pin); wired at `dgemma/loop.py:step_end = StepEndComposite(capture=collector.on_step_end, should_cancel=should_cancel)` | **Resolved** — #35 R1 (PR TBD). Beta-rebuild/pin participant bodies remain `NOT-YET-IMPLEMENTED` (ADR-CDG-010 R2/R5); the composite scaffold and ordering are in force. |
| No enforcement that a forward hook is torn down after a run | Rule 6 — F4: an un-torn-down hook from run A shapes run B | `NOT-YET-IMPLEMENTED` — no lifecycle context manager exists (hook seam not yet built; #28 names `register_forward_hook` as the target door) | #35 R5 |
| Cross-run statelessness of walker/pin is incidental (fresh scheduler per run), not enforced | Rule 6 — F5: mutated `scheduler.config` + accumulated pin mask are cross-call-mutable state | `NOT-YET-IMPLEMENTED` — no same-in/same-out test; containment is today's fresh-per-run happenstance | #35 R5 / ADR-CDG-011 F5 test |
| No diffusers version guard (the transformers guard's missing twin) | Rule 5 — `anneal_temperature` re-derives the vendored formula and would silently report wrong values on a bump | Guard absent; transformers guard exists at `dgemma/model.py:78` (`_check_transformers_version`) — no diffusers analog | `NOT-YET-IMPLEMENTED` — #35 R3 |
| Declarative-payload ingress (`constraints=`, `control_signals=`, `capture=`) not present | Rule 7 — `run_diffusion` cannot yet accept validated declarative intervention | `dgemma/loop.py:465-478` (`run_diffusion` signature has `on_frame` but no `constraints`/`control_signals`/`capture`) | `NOT-YET-IMPLEMENTED` — ADR-CDG-010/011 |

**In force today (the bones survive — do not re-litigate):**

| Conforming point | Rule | Evidence (`path:symbol`) |
|------------------|------|--------------------------|
| Core imports with zero ComfyUI present; subprocess asserts no `comfy`/`nodes` leak | 1 | `tests/test_seam.py:36-63`; `dgemma/__init__.py` |
| Contract is single-entry, canonical: `run_diffusion` always returns `(text, CanvasState, CanvasTrace)`, never a bare string | 1 | `dgemma/loop.py:465,478` (return type); `load_model` at `dgemma/model.py:157` |
| Node bodies are thin adapters; no denoising-step loop in a surface body | 2 | `nodes/loader.py`, `nodes/sampler.py`, `nodes/trace.py` (ADR-CDG-003) |
| Native socket types, not `SIGMAS`/`LATENT` (no lying payload) | 5 | `nodes/sampler.py:202` (`DGEMMA_CANVAS_STATE`, `DGEMMA_CANVAS_TRACE`); ADR-CDG-001 |
| Live per-step view is a read-only observer, not a socket stream | 7 | `nodes/sampler.py:114-161` (`_build_on_frame`, `on_frame`); `run_diffusion(on_frame=…)` at `dgemma/loop.py:477` |

*Reachability note:* every row above is audited against reachable code. Rows whose
subject does not yet exist carry `NOT-YET-IMPLEMENTED` with the R-item / phase that
creates it — not an empty cell.

---

## Enforcement-surface table (one row per invariant)

The ADR-CDG-008 boundary table, extended with the review's new invariants. Each row
names its test / type / review surface and its status.

| Invariant | Enforcement surface | Status |
|-----------|---------------------|--------|
| Core imports no surface (`dgemma/` never imports `comfy.*` / `nodes.*`) | `tests/test_seam.py:36-63` (subprocess `import dgemma`, `sys.modules` leak check) | **In force.** Must extend to reject `surfaces.*` after the Phase-1 rename. |
| Core imports no analysis (base contract imports no consumer module) | Subprocess assertion (analysis not in `sys.modules` after `import dgemma`) | `NOT-YET-IMPLEMENTED` — prose-only today (`dgemma/sampling.py` docstring), contradicted by `dgemma/__init__.py:26-31`. Created by CDG-008 Phase 4 (after Phase 3 relocation). |
| Surfaces are peers over one contract (no logic in a surface body) | ADR-CDG-003's "no `for`-loop-over-steps in a surface body", generalized to `surfaces/*` | Reviewed by eye + `tests/test_trace_node.py` (`DGemmaTrace.render` purity). No mechanized cross-surface import-graph rule. Residual debt, not structural impossibility. |
| Canonical trace, parsed at the door | `run_diffusion` return-type (`dgemma/loop.py:478`); ADR-CDG-004 | **In force at the type level.** |
| Conserved repo identity (`ComfyUI-DiffusionGemma` unchanged across the rename) | Registry mirror + remote (`IDENTITY⊥ENVELOPE`); no code change touches it | **In force by omission** — the roadmap must not touch the repo name. |
| Socket vocabulary minted once (no inline `DGEMMA_*` literal outside the mint module) | Grep-gate test asserting against the module object (only the path string churns with Phase 1) | `NOT-YET-IMPLEMENTED` — #35 R2. |
| Composition ordering (capture pre-pin; β-rebuild before pin; pin last writer) | Ordered-composite test over the shared fake-pipeline fixture: `tests/test_step_end_composite.py:TestFixedOrdering`, `TestOrderingIsStructural` (`dgemma/composite.py:StepEndComposite`) | **In force** — #35 R1 (over R4's fixture). ADR-CDG-010. |
| Zero hooks after run ("no hook survives a `run_diffusion` call") | Forward-hook lifecycle context-manager test, clean + raising | `NOT-YET-IMPLEMENTED` — #35 R5 (F4). |
| Same-in/same-out walker/pin statelessness (identical calls → identical effective-knob telemetry) | Same-in/same-out test on one loaded model | `NOT-YET-IMPLEMENTED` — #35 R5 / ADR-CDG-011 F5. CDG-008 Phase-2 MCP state manager must never cache a scheduler. |
| Diffusers version guard + structural probe (scheduler kwargs, `accepted_index`, `_callback_tensor_inputs`) | Version-pin guard patterned on `dgemma/model.py:78` (`_check_transformers_version`) | `NOT-YET-IMPLEMENTED` — #35 R3. |
| Declarative payloads only into `run_diffusion` (no surface-built closures/hooks) | Ingress validation (schedule length == steps; values in binding range; ids in-vocab; fail on unknown) + the composite holding only engine-built participants | `NOT-YET-IMPLEMENTED` — ADR-CDG-010/011 ingress clauses. |
| `num_inference_steps` non-mutable mid-run | Ingress reject (guards #20's `predictor_steps`/`_num_timesteps` desync) | `NOT-YET-IMPLEMENTED` — ADR-CDG-011. |
| `DiffusionFrame` extension discipline (additive-optional, heavy-field retention policy) | Optional-with-defaults fields; retention policy for heavy `DISTRIBUTION` | `NOT-YET-IMPLEMENTED` — #35 R6 / F3 (rides research rung 4). |
| Shared fake-pipeline/scheduler fixture (N steps, mutable `config`, hook-recording model, `{"canvas":…}` application) | `tests/conftest.py:fake_pipeline_factory` (`FakeEntropyBoundScheduler`, `HookRecordingModel`, `FakeDiffusionGemmaPipeline`); self-tests in `tests/test_conftest_fake_pipeline.py` | **In force.** #35 R4. "Mutable `config`" resolved against the real `diffusers` `FrozenDict` (write-raises; mutation only via `register_to_config`, verified against the installed-wheel source) — see `tests/conftest.py`'s module docstring. |
| Frames↔images index correspondence not untagged | Per-image frame-key tag or explicit no-zip contract (`CONSERVE-DATA-BOUNDARY`) | `NOT-YET-IMPLEMENTED` — #35 F7/F9, reconciled in ADR-009 / PR #31 ratification. |

---

## What "instant fail" looks like

One row per invariant rule, each violation paired with its correct shape.

| Violation | Valid form |
|-----------|------------|
| A `nodes/*.py` body loops over denoising steps | The body unpacks args, calls `run_diffusion` once, wraps the result (ADR-CDG-003) |
| `dgemma/*.py` does `import comfy` / `from nodes import …` | The surface imports the core; the core imports nothing surface-shaped (rule 1) |
| A new analysis function added to `dgemma/sampling.py` | Added to the Phase-3 consumer home (`consumers/` / `surfaces/analysis/`), importing `CanvasTrace` (rule 3) |
| `RETURN_TYPES = ("DGEMMA_CANVAS_TRACE",)` inline at a new node site | Reference the socket string from the mint module; grep-gate rejects the inline literal (rule 4) |
| An entropy budget passed as a `SIGMAS` tensor; a `DISTRIBUTION` socket carrying only a scalar | A native `DGEMMA_*` type carrying the real payload, validated at ingress (rule 5) |
| MCP state manager caches a live scheduler across calls | Persist only `load_model`'s output; build a fresh scheduler/canvas/run-state per call (rule 6) |
| A forward hook installed in run A left registered into run B | Install via the R5 lifecycle context manager; no hook survives the call (rule 6) |
| A surface passes a closure/hook into `run_diffusion` to shape logits | Pass declarative `constraints=` / `control_signals=`; the engine installs the hook, validated at ingress (rule 7) |
| Capture ordered after pin (records constraint-asserted tokens as model-committed) | Capture runs before any canvas-writer; `pinned_mask` distinguishes the two (rule 7, ADR-CDG-010) |
| Mutating `num_inference_steps` mid-run | Reject at ingress; mutate only `scheduler.config` values `step()` reads fresh (rule 7, ADR-CDG-011) |

---

## Relation to the decision record

| ADR / decision | What it fixes | File / status |
|----------------|---------------|---------------|
| ADR-CDG-001: native socket types | Rules 4, 5 — reject lying payloads; native `DGEMMA_*` types | `decisions/adr-cdg-001-native-socket-types.md` — Accepted |
| ADR-CDG-002: transformers streamer access path | Load seam; documentary "no MASK" | `decisions/adr-cdg-002-transformers-streamer-access-path.md` — Accepted (amended by 004) |
| ADR-CDG-003: node-engine seam | Rules 1, 2 — the core/adapter split this generalizes | `decisions/adr-cdg-003-node-engine-seam.md` — Accepted |
| ADR-CDG-004: diffusers pipeline drive seam | Rule 1 — `run_diffusion` single-entry drive contract | `decisions/adr-cdg-004-diffusers-pipeline-drive-seam.md` — Accepted |
| ADR-CDG-005: `CANVAS_STATE` resumable save-state | Bench seam `CANVAS_STATE` (contract, not display) | `decisions/adr-cdg-005-canvas-state-resumable-savestate.md` — Accepted |
| ADR-CDG-006: advanced sampler step-window resume | Cross-execution resume (stand-in for the missing UI incrementer) | `decisions/adr-cdg-006-advanced-sampler-step-window-resume.md` — Proposed |
| ADR-CDG-007: GGUF backend node set | Out-of-scope inference-only backend | `decisions/adr-cdg-007-clear-alpha-gguf-backend-node-set.md` — Rejected (2026-07-06) |
| ADR-CDG-008: MCP-center multi-surface topology | Rules 1, 2, 3, 4, 6 — the decided target topology; Phases 1–5 | `decisions/adr-cdg-008-mcp-center-multi-surface-topology.md` — Accepted |
| ADR-CDG-010: constraint composite and pinned mask | Rule 7 — two-mechanism givens; composite ordering; `pinned_mask`; engine-installed hooks via R5 | `decisions/adr-cdg-010-constraint-composite-and-pinned-mask.md` — Accepted (ratified 2026-07-13, PR #43); composite-ordering clause implemented #35 R1 |
| ADR-CDG-011: control-signal CV/LFO mod matrix | Rules 6, 7 — declarative socket / closure walker split; units-at-binding; `scheduler.config`-only mutation; same-in/same-out test | `decisions/adr-cdg-011-control-signal-cv-lfo-mod-matrix.md` — Accepted (ratified 2026-07-13, PR #43) |
| Issue #35: architecture review | Findings F1–F9, refactor list R1–R6, ADR-CDG-010/011 clauses | `#35` (open, `pri:now`) |
| ADR-CDG-009: N-canvas trace display legibility (frames↔images) | Handle `CONSERVE-DATA-BOUNDARY` (F7/F9) | `decisions/adr-cdg-009-two-canvas-trace-display.md` — Proposed; PR #31 merged |

---

## Anticipated evolution — two live tracks

*This replaces the prior document's §7 "not scoped, not designed, not started" claim:
the alignment refactor is now scoped (ADR-CDG-008) and reviewed (#35).*

**Track 1 — CDG-008 alignment (Phases 1–5).** Sequenced, dependency-respecting:
Phase 1 rename `nodes/` → `surfaces/comfyui/` (+ `web/`); Phase 2 add `surfaces/mcp/`
(transcribe `semantic-kinematics-mcp` with the two corrections); Phase 3 relocate
analysis out of `dgemma/`; Phase 4 add the base-contract-imports-no-analysis test;
Phase 5 this document. Execution order is set by #35's delta: **R4 (shared fixture)
before R1 (composition layer)**, then R5; R3 anytime; R2 with/before Phase 1;
rung-4 analysis behind Phase 3.

**Track 2 — research expansion.** The liquid-phase-decoding bench
(`docs/experiments/liquid-phase-decoding/concept.md`): a six-seam inventory
(`DISTRIBUTION`, control-signal `SCHEDULE`, pin/mask, sampling operator, `KV_CACHE`,
`CANVAS_STATE`) with five falsifiable H0s. The bench principle — "every sampler
scalar is a wireable per-step field, factored so honestly the variety composes from
the bench" — is ADR-CDG-001 at scale. All six capabilities land core-side of the
seam, so CDG-008's MCP surface and a future human UI inherit them for free.
Graduation trigger: a confirmed H0 → an ADR (a socket type / scheduler seam). The
`DISTRIBUTION` socket is the gate everything else waits on (#11/#14 partial).

**Open items with resolution triggers** (do not read as decided):
- `consumers/` vs `surfaces/analysis/` naming — ADR-CDG-008 Open Question #1; settle
  in the Phase-3 relocation `plan` pass.
- Whether `CanvasTrace` moves to a shared contract module — ADR-CDG-008 Open Question
  #2; default is "stays in `dgemma/`."
- Whether the refactor needs a `plan` pass before touching `__init__.py`'s discovery
  contract — ADR-CDG-008 Open Question #3; resolution: yes.
