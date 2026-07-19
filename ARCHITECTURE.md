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

Eight rules. All eight apply simultaneously. Rules 1–7 govern the core/surface seam
(*what lives below the door and how surfaces reach through it*); rule 8 governs the tier
*above* the surfaces (*who sequences the calls the surfaces expose*). The count grew by
one when the 2026-07-16 family-congruence read (issue #92) found CDG had carried the
core/surface contract faithfully but dropped the orchestration tier both family members
name — see rule 8 and the *Orchestration / consumer plane* layer below.

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

8. **Consumers orchestrate; they do not extend.** Run sequencing — sweeps, loops,
   batteries, multi-run comparison — belongs to the tier *above* the surfaces, never
   inside a surface body and never accreted into the core. A consumer composes and
   sequences calls to already-contracted surface primitives; every capability it uses
   already exists below it as a `load_model` / `run_diffusion` wrapper. The consumers of
   this pack are humans driving the ComfyUI graph, agents driving MCP clients, scripts,
   and the E2E driver (`tests/e2e/driver.py`, the existing in-repo instance — it
   sequences whole battery scenarios over the surfaces, importing nothing from
   `dgemma`/`surfaces`/`consumers`, ADR-CDG-013 Decision 1). A consumer that reaches past
   the surface contract to touch core internals — or a code path added to the core that
   exists solely to serve one consumer's sequencing need — is the violation this rule
   names. This is sk-mcp's rule 3 transcribed to CDG's contract
   (`../semantic-kinematics-mcp/docs/ARCHITECTURE.md` §The invariant, rule 3: *"Consumers
   orchestrate; they do not extend… They exercise no novel pathways in the core"*), with
   its **EXTERNAL** framing taken deliberately.

   **⚠ Word-collision warning (adopt sk-mcp's EXTERNAL framing, reject prompt-prix's
   internal one).** "Orchestration" names two opposite sides of a contract boundary
   across the family, and CDG must not conflate them. In **sk-mcp** the *Orchestration /
   consumer plane* sits **above** the MCP contract — external consumers sequencing
   contracted calls (`../semantic-kinematics-mcp/docs/ARCHITECTURE.md` §The layers,
   *Orchestration / consumer plane*). In **prompt-prix** "ORCHESTRATION" names the
   **internal** top layer *below* its own entry points — `BatteryRunner` /
   `ConsistencyRunner` / `ComparisonSession` calling `execute_test_case()`
   (`../prompt-prix/docs/ARCHITECTURE.md` §Four-Layer Architecture, the ORCHESTRATION box
   and §Layer Import Rules). Same word, opposite side of the door. **CDG adopts sk-mcp's
   external sense**: orchestration is the plane of consumers *above* `surfaces/*`, not an
   internal runner tier the core would grow. (Precedent for issuing this warning
   explicitly rather than trusting the shared word: the repo `CLAUDE.md`'s own
   P0–P3-node-pack-phases vs. CDG-008-topology-phases note — "same word, different
   ledgers.") *(→ sk-mcp §The invariant rule 3 + §The layers · `ONE-DOOR` · rule 2 below
   the door)*

---

## The layers

Top (consumer) to bottom (substrate). Directory names in **target** form per
ADR-CDG-008 §Decision-2; the current on-disk names are noted where they differ.

### Orchestration / consumer plane — sequences the surfaces (rule 8)

**What lives here (all EXTERNAL to the pack, above `surfaces/*`):** humans driving the
ComfyUI node graph; agents and MCP clients driving `surfaces/mcp/`; scripts; and the E2E
driver (`tests/e2e/driver.py`), the one in-repo instance — it sequences whole battery
scenarios over the surfaces and, per ADR-CDG-013 Decision 1, imports nothing from
`dgemma`/`surfaces`/`consumers` (the black-box independence enforced by
`tests/e2e/test_e2e_import_guard.py`, already an *In force* row in the enforcement table).
This mirrors sk-mcp's *Orchestration / consumer plane*
(`../semantic-kinematics-mcp/docs/ARCHITECTURE.md` §The layers) — the tier that "compose[s]
and sequence[s] calls to contracted primitives" and "exercise[s] no novel pathways in the
core."

**Rules (rule 8):**
- A consumer composes and sequences already-contracted surface primitives. Sweeps, loops,
  batteries, multi-run comparison live here — never in a surface body, never accreted into
  the core.
- A consumer that bypasses the surface contract to touch core internals, or a core code
  path added to serve one consumer's sequencing, is an instant fail (rule 8 · the
  *Orchestration / consumer plane* half of sk-mcp's rules 3–4).
- **EXTERNAL framing, per the word-collision warning on rule 8:** this is sk-mcp's sense
  of "orchestration" (above the contract), not prompt-prix's internal `BatteryRunner`
  tier (below its entry points, `../prompt-prix/docs/ARCHITECTURE.md` §Four-Layer
  Architecture). CDG grows no internal runner tier; if batch sequencing needs a home, it
  is a consumer above `surfaces/*` or a new surface primitive, never core-resident logic.

### Consumers — derived analysis (`consumers/`)

**What lives here:** the pure trace-analysis functions, `consumers/analysis.py`
(`build_commit_heatmap`, `build_avalanche_curve`, `corroborate_no_mask_token`,
`MaskTokenCorroboration`). **Landed** — CDG-008 Phase 3 (issue #55), relocated
from `dgemma/sampling.py`; `dgemma/__init__.py` no longer imports or re-exports
them. ADR-CDG-008 Open Question #1 is settled to `consumers/` (see the ADR's
2026-07-13 amendment note and issue #55 §1).

**Rules:**
- Parses an already-captured `CanvasTrace`; never re-derives what the core emitted,
  never drives the model.
- Imports the core's contract type; the core imports nothing from here.
- `DGemmaTrace` (`surfaces/comfyui/trace.py`) is *also* a ComfyUI surface node
  wrapping this analysis — the split is by role, not by file: the pure
  functions are consumer-tier (`consumers/analysis.py`), the socket-wrapping
  node is surface-tier (`surfaces/comfyui/trace.py`), and the node importing
  the consumer is normal composition, not a layering inversion.

### Surface tier — peer surfaces over the one contract (target: `surfaces/*`)

**What lives here:**
- `surfaces/comfyui/` — the ComfyUI node graph (`loader.py`, `sampler.py`,
  `trace.py`, `frames_image.py`, `socket_types.py`) + `surfaces/comfyui/web/`.
  **Landed** — CDG-008 Phase 1 (issue #52), relocated from `nodes/` + top-level
  `web/`. `surfaces/__init__.py` is the empty package-marker parent this and
  `surfaces/mcp/` (below) share.
- `surfaces/mcp/` — the base MCP surface over `load_model` + `run_diffusion`
  (`server.py`, `state_manager.py`, `commands/{model,generate}.py`).
  **Landed** — CDG-008 Phase 2 (issue #52 follow-on), transcribed from
  `semantic-kinematics-mcp`'s `mcp/` layout with the two named corrections:
  the state manager persists only the loaded `DGemmaModel`
  (`surfaces/mcp/state_manager.py:StateManager`, no scheduler/canvas/run-state
  field), and the automated boundary test is kept, not regressed to
  review-only (`tests/test_mcp_surface_seam.py`).

**Rules:**
- Each `surfaces/*` module is a thin adapter: unpack args → call one `dgemma.*`
  function → wrap the result. No denoising-step loop in a surface body (ADR-CDG-003).
- Holds no core logic; the logic isn't in the surface, so no surface can accrete
  what the others can't reach.
- The ComfyUI surface's socket strings are minted in its own `socket_types.py`
  (rule 4); the MCP surface's state manager persists only the model load (rule 6).

### Core — the one contract (`dgemma/`, already surface-neutral)

**What lives here:** the model, the types, the denoising loop. `dgemma/model.py`
(load), `dgemma/loop.py` (drive), `dgemma/types.py` (contract dataclasses). The
analysis math relocated to `consumers/analysis.py` (CDG-008 Phase 3, issue #55)
— the core no longer imports or re-exports it.

**Rules:**
- Imports and runs with zero ComfyUI present (`dgemma/__init__.py`; enforced by
  `tests/test_seam.py:36-63`).
- Emits the canonical `CanvasTrace`; never parses derived analysis (enforced by
  `tests/test_seam.py::test_dgemma_does_not_import_consumers_package`, CDG-008
  Phase 4).
- Holds no cross-run mutable state (rule 6). The load is persisted, the run is
  stateless.
- Widens `run_diffusion` by declarative payloads only (rule 7).

---

## Lifecycle & tenancy — the plane CDG has not built yet (honest absence)

Rule 6 governs cross-*run* statelessness within a loaded model; it says nothing about the
*lifecycle* of the ~53 GB load itself — who starts it, who owns the process, how many
tenants share the card. Both family members name a lifecycle plane; CDG carried the
core/surface contract faithfully but left this one blank. Stated honestly as an absence,
in the same NOT-YET-IMPLEMENTED register the rest of this document uses for undesigned
target.

**Current fact (2026-07-16).** CDG loads DiffusionGemma **in-process, single-tenant**:
whatever process calls `load_model` (a ComfyUI worker, or a `surfaces/mcp/` server via
`StateManager.load`) holds the ~53 GB weights in *its own* address space for its lifetime.
The 48 GB RTX-8000 dev box fits **one** such load at a time (model-card ≥60 GB bf16; local
runs are quantized/offloaded per repo `CLAUDE.md` §Grounded facts). There is **no
lifecycle delegation**: nothing external starts, stops, swaps, or arbitrates tenancy of
the load. `StateManager` (`surfaces/mcp/state_manager.py`) persists only the model object
(rule 6 · CDG-008 Phase 2 Correction 1) — it is a *holder*, not a *lifecycle owner*. Two
surfaces cannot today share one resident load; each would load its own copy, and two
copies do not fit.

**The family's two answers (cited, neither adopted here):**

- **prompt-prix — in-process pool, "one model at a time per server."** prompt-prix keeps
  lifecycle *inside* the process via a `ServerPool` that "enforces the model-drain guard —
  one model at a time per server to prevent VRAM swap"
  (`../prompt-prix/docs/ARCHITECTURE.md` §local-inference-pool), the `current_model` drain
  guard being the enabler of its pipelined scheduling (ibid. §Battery Execution). Tenancy
  is arbitrated by an in-process component, not delegated out.
- **sk-mcp — out-of-process llauncher delegation.** sk-mcp pushes lifecycle *out of the
  core* entirely: "model-server lifecycle is delegated out of the core to llauncher"
  (`../semantic-kinematics-mcp/docs/ARCHITECTURE.md` §The invariant, rule 1), and its
  *Lifecycle plane (out of process)* states "llauncher owns the start/stop/swap/status
  lifecycle of model servers… sk-mcp tools target an already-running endpoint; they do
  not start, stop, or monitor model servers" (ibid. §The layers). Its "Why stateless"
  §externalized-lifecycle names the failure this prevents: "The moment sk-mcp holds it,
  the separation breaks and sk-mcp becomes a process manager."

**Anticipated evolution — served-engine topology (ADR-candidate, named not decided).**
The fork this section exists to name: a **served-engine** topology — one resident engine
process owns the single ~53 GB load; `surfaces/comfyui/` and `surfaces/mcp/` become
*clients* of that engine rather than in-process peers each holding their own copy;
lifecycle (start/stop/swap/status of the engine) is **llauncher-owned**, sk-mcp's answer
adopted over prompt-prix's in-process pool because CDG already runs on the llauncher
substrate (repo `CLAUDE.md` §Environment). This **amends ADR-CDG-008's in-process-peer
assumption** — CDG-008's "surfaces are peers over the core" (rule 2) tacitly assumes each
peer can hold the load in-process; a served engine makes surfaces *remote* clients of a
single resident load, which the 48 GB card's single-tenancy will eventually force. This
amendment **names the fork; it does not decide it.** The decision is a separate bracket
(operator sets requirements, per issue #92 process note).

- **Trigger** (the observation that promotes the ADR-candidate to a written ADR): a
  *second concurrent surface needing the resident model* — e.g. an MCP client and a
  ComfyUI graph both wanting the loaded weights at once, which single-tenant in-process
  loading cannot satisfy on the 48 GB card. Until that trigger fires, in-process
  single-tenant is the honest current state and the served-engine topology stays an
  ADR-candidate, not a plan.

*Enforcement surface for the tenancy fact:* today the single-tenancy is enforced only by
the *physics of the card* (two ~53 GB loads do not fit 48 GB) plus rule 6's holder-only
`StateManager` (`tests/test_mcp_statelessness.py`, which asserts no run-state is cached
but does **not** assert single-tenancy) — **known-fragile, review-only** as an
architectural invariant, pending the served-engine ADR that would give it a real
enforcement surface (an engine-client boundary test analogous to the seam tests). Named
per GROUND_PHYSICS discipline 6 (assigned enforcement): the surface is the hardware limit
plus prose, and this paragraph is that prose stating its own fragility.

---

## What the invariant does NOT govern (out of scope)

- **The published repo name `ComfyUI-DiffusionGemma`** — conserved identity
  (`IDENTITY⊥ENVELOPE`), registry-mirrored and remote-live. The internal directory
  vocabulary changed (`nodes/` → `surfaces/comfyui/`, CDG-008 Phase 1, landed);
  the repo name does not. Renaming the repo is explicitly out of scope
  (ADR-CDG-008 Decision-2, Option B rejected). This is a scoping fact, not an
  exception: the layering invariant governs internal envelope, not the
  conserved external handle.

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
+-----------------------------------------------------------+
| ORCHESTRATION / CONSUMER PLANE  (rule 8; EXTERNAL, above) |
|   humans @ ComfyUI graph . MCP clients/agents . scripts   |
|   tests/e2e/driver.py  (in-repo instance; imports nothing |
|     from dgemma/surfaces/consumers -- ADR-CDG-013 Dec.1)  |
|   -- sequences surfaces; adds no core pathway (rule 8)     |
+-----------------------------------------------------------+
        |  surface calls only (never into the core directly)
        v
   consumers/  (analysis: parses CanvasTrace)          -- landed (Phase 3)
        |  parses
        v
+-----------------------------------------------------------+
| surfaces/                                                 |
|   comfyui/  (landed, was nodes/ + web/)  mcp/  (landed)   |  -- peers
+-----------------------------------------------------------+
        |  load_model + run_diffusion  -- THE ONE CONTRACT (the door)
        v
+-----------------------------------------------------------+
| dgemma/   core -- surface-agnostic, zero ComfyUI present  |
|   model.py (load) . loop.py (drive) . types.py (contract) |
+-----------------------------------------------------------+
        |
   torch . transformers . diffusers   -- shared substrate (out of scope, beside not through)

   [lifecycle & tenancy plane -- NOT-YET-BUILT: today in-process single-tenant;
    served-engine (llauncher-owned) is an ADR-candidate, not decided -- see the
    "Lifecycle & tenancy" section above]
```

The orchestration/consumer plane sits **above** the surfaces (rule 8, EXTERNAL sense per
the word-collision warning); it reaches the core *only* through surface calls, never
laterally into `dgemma/`. The contract boundary is the `load_model` + `run_diffusion`
line; every governed
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
  (`surfaces/comfyui/sampler.py:136-159` pattern; `run_diffusion(on_frame=…)`,
  `dgemma/loop.py:477`): receives a built `DiffusionFrame`, return ignored,
  structurally read-only, needs no position among canvas-writers. Pre-pin truth
  reaches it as *frame fields* (`pinned_mask`, effective knobs), not by observer
  ordering. This is the **only executable crossing** the surface owns. *In force
  today as a read-only observer* (`surfaces/comfyui/sampler.py:114-161`, `_build_on_frame`).

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
  survives a `run_diffusion` call**, tested clean and raising. **In force**
  (`dgemma/hooks.py:install_logit_shaping_hook`, wired at
  `dgemma/loop.py:run_diffusion`'s `with install_logit_shaping_hook(...)`
  wrapping the pipeline call; `tests/test_hook_lifecycle.py`). The mask
  itself (a `constraints=`-built hook function) is still `NOT-YET-IMPLEMENTED`
  (ADR-CDG-010's own participants, R2/future scope) — R5 lands the lifecycle
  primitive every future hook installer must go through, not the mask body.

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
base contract imports no analysis — turning what was a prose-only boundary
(the pre-relocation `dgemma/sampling.py`'s docstring claimed consumer status
while `dgemma/__init__.py:26-31` contradicted it by re-exporting it) into an
enforced one: `consumers/analysis.py` (CDG-008 Phase 3) plus
`tests/test_seam.py::test_dgemma_does_not_import_consumers_package`
(CDG-008 Phase 4, issue #55).

**Why declarative payloads, not closures.** A surface-supplied callable is
unvalidatable at ingress and re-opens the door the core closed: through `pipe.model`
it could install sampling logic that belongs in the core. Declarative payloads are
checkable at the door; the forbidden shape (a surface returning a canvas) becomes
structurally unrepresentable.

---

## Current conformance (honest) — Branch B (audited)

The invariant above is the target. The code **partially** conforms: the core/surface
seam (rules 1, 2 in part, 5 in part) is in force; rule 3 (analysis is a downstream
consumer) is now in force (CDG-008 Phase 3+4, issue #55); rule 6's cross-run
statelessness enforcement is now in force for the two invariants #35 named (F4's
hook teardown, F5's fresh-scheduler same-in/same-out — #35 R5) AND for the MCP
surface's own state manager (CDG-008 Phase 2 Correction 1); the MCP surface now
exists as a real peer over the one contract (CDG-008 Phase 2). The
surface-side naming, the mint module, and most of the step-end intervention
layer (rules 4, 7 and most of 2 — the composite scaffold from R1 and the hook
lifecycle from R5 are in force, but the constraints/control-signal participant
bodies themselves are not) are not yet implemented.

| Violation | Why it breaks the invariant | Evidence (`path:symbol`) | Resolved by |
|-----------|----------------------------|--------------------------|-------------|
| ~~Surface tier is named `nodes/` (a ComfyUI word) + top-level `web/`; no `surfaces/` parent~~ **RESOLVED (naming half)** | Rule 2 — the name puts ComfyUI at the center, leaving no room for peer surfaces | `surfaces/comfyui/{loader,sampler,trace,frames_image}.py`; `surfaces/comfyui/web/live_view.js`; `__init__.py` (`WEB_DIRECTORY = "./surfaces/comfyui/web"`) | **Resolved (naming half)** — CDG-008 Phase 1 (issue #52). The MCP-peer half of this row (a second surface actually existing) is now also resolved, below. |
| ~~No MCP surface exists~~ **RESOLVED** | Rule 2 — MCP is the decided base surface | `surfaces/mcp/server.py` (`Server`, `list_tools`, `call_tool`, `main`); `surfaces/mcp/state_manager.py:StateManager` (persists only the load); `surfaces/mcp/commands/{model,generate}.py` (`load_model`/`model_status`/`generate`/`cancel_run` tools, each a thin `dgemma.*` wrap) | **Resolved** — CDG-008 Phase 2. `tests/test_mcp_surface_seam.py` (boundary, both directions), `tests/test_mcp_statelessness.py` (Correction 1, mutation-checked), `tests/test_mcp_import_guard.py` (optional-SDK guard). |
| ~~Analysis lives inside the core's import graph and is re-exported by the core's public face~~ **RESOLVED** | Rule 3 — analysis is a consumer; the core must not export it | `consumers/analysis.py` (bodies: `build_commit_heatmap`, `build_avalanche_curve`, `corroborate_no_mask_token`, `MaskTokenCorroboration`); `dgemma/__init__.py` (re-exports removed) | **Resolved** — CDG-008 Phase 3 (relocate) + Phase 4 (boundary test), issue #55. Tests: `tests/test_seam.py::test_dgemma_does_not_import_consumers_package` (+ the extended `_CHECK_SCRIPT` leak-list) and `tests/test_analysis.py`. |
| ~~Socket strings re-typed as bare literals per node site; no mint module~~ **RESOLVED** | Rule 4 — `ONE-MINT` violated; the vocabulary is authored N times | `surfaces/comfyui/socket_types.py` (the mint: `DGEMMA_MODEL`, `DGEMMA_CANVAS_STATE`, `DGEMMA_CANVAS_TRACE`); every node-site literal replaced by an import from it | **Resolved** — #35 R2 (issue #52). `tests/test_socket_mint.py` (grep-gate + round-trip, asserts against the module object). |
| ~~Single hardcoded callback binding; no composition / ordering / exception layer~~ **RESOLVED** | Rule 7 — five expansion participants want the slot with ordering semantics | `dgemma/composite.py:StepEndComposite` (fixed order: capture → cancellation → beta-rebuild → pin; ADR-CDG-010 cancellation amendment 2026-07-13); wired at `dgemma/loop.py:step_end = StepEndComposite(capture=collector.on_step_end, should_cancel=should_cancel)` | **Resolved** — #35 R1 (PR #45). Beta-rebuild/pin participant bodies remain `NOT-YET-IMPLEMENTED` (ADR-CDG-010 R2/R5); the composite scaffold and ordering are in force. |
| ~~No enforcement that a forward hook is torn down after a run~~ **RESOLVED** | Rule 6 — F4: an un-torn-down hook from run A shapes run B | `dgemma/hooks.py:install_logit_shaping_hook` (the sole `register_forward_hook` installation path, `try/finally` teardown); wired at `dgemma/loop.py:run_diffusion`'s `with install_logit_shaping_hook(dgemma_model.model, logit_hook): output = pipeline(...)` | **Resolved** — #35 R5 (F4). `tests/test_hook_lifecycle.py` (clean, cancelled, and raising paths all assert `live_hook_count == 0`). |
| ~~Cross-run statelessness of walker/pin is incidental (fresh scheduler per run), not enforced~~ **RESOLVED** | Rule 6 — F5: mutated `scheduler.config` + accumulated pin mask are cross-call-mutable state | `dgemma/loop.py:run_diffusion` constructs a fresh `EntropyBoundScheduler`/`_FrameCollector`/`StepEndComposite` every call, never caching one; `tests/test_run_diffusion_statelessness.py:TestSchedulerFreshPerCall` asserts two calls build two distinct scheduler objects | **Resolved** — #35 R5 (F5) / ADR-CDG-011 F5 test. `tests/test_run_diffusion_statelessness.py:TestSameInSameOutTelemetry` asserts identical calls yield identical telemetry and a mid-call `register_to_config` mutation never survives into the next call. |
| ~~No diffusers version guard (the transformers guard's missing twin)~~ **RESOLVED** | Rule 5 — `anneal_temperature` re-derives the vendored formula and would silently report wrong values on a bump | `dgemma/loop.py:_check_diffusers_version` (version-floor guard, twin of `dgemma/model.py:78`'s `_check_transformers_version`, adapted for the `>=0.39.0` range bound) + `dgemma/loop.py:_check_diffusers_structure` (structural probe: scheduler ctor kwargs, `EntropyBoundSchedulerOutput.accepted_index`, base `DiffusionGemmaPipeline._callback_tensor_inputs`); both invoked at module import time, `dgemma/loop.py:_check_diffusers_version(); _check_diffusers_structure()`. The probe covers names/shapes only — it cannot see the anneal formula's *body*; that slice is enforced by `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin` (pins `anneal_temperature` against the temperature the real installed `EntropyBoundScheduler.step()` actually applies, recovered from `pred_logits`) | **Resolved** — #35 R3 (formula-body slice closed per PR #48 gate finding F-1). `tests/test_diffusers_version_guard.py` (23 tests: version-floor accept/reject + fallback path, structural probe pass/fail per probed structure, formula pin across schedule points/configs). |
| ~~Declarative-payload ingress (`constraints=`, `control_signals=`, `capture=`) not present~~ **RESOLVED (ingress slice)** | Rule 7 — `run_diffusion` cannot yet accept validated declarative intervention | `dgemma/loop.py:run_diffusion` (signature accepts `constraints`/`control_signals`/`capture`, keyword-only, appended after `logit_hook`); `dgemma/payloads.py` (`Pin`, `Constraints`, `Binding`, `ControlSignals`, `MUTABLE_TARGETS`); `dgemma/ingress.py:validate_ingress` (C1–C4/V1–V6/P1/H1 with the precondition+remedy register), wired at `dgemma/loop.py:run_diffusion` before scheduler construction | **Resolved (ingress slice)** — CDG-010/011 Phase 1 (issue #64 §6, PR #65). `tests/test_ingress.py`, `tests/test_run_diffusion_ingress.py`, `tests/test_resolve_vocab_size.py`. Payloads are validated-then-ignored this phase: the **participant bodies** built from a valid payload (`PinParticipant`/`WalkerParticipant`/the `constraints=`-built logit-mask hook) remain `NOT-YET-IMPLEMENTED` — ADR-CDG-010/011 Phases 3/4. |

**In force today (the bones survive — do not re-litigate):**

| Conforming point | Rule | Evidence (`path:symbol`) |
|------------------|------|--------------------------|
| Core imports with zero ComfyUI present; subprocess asserts no `comfy`/`nodes`/`surfaces`/`consumers` leak | 1 | `tests/test_seam.py:36-63` (extended per CDG-008 Phase 1 to also reject `surfaces.*`, per CDG-008 Phase 3+4/issue #55 to also reject `consumers.*`); `dgemma/__init__.py` |
| Contract is single-entry, canonical: `run_diffusion` always returns `(text, CanvasState, CanvasTrace)`, never a bare string | 1 | `dgemma/loop.py:465,478` (return type); `load_model` at `dgemma/model.py:157` |
| Node bodies are thin adapters; no denoising-step loop in a surface body | 2 | `surfaces/comfyui/loader.py`, `surfaces/comfyui/sampler.py`, `surfaces/comfyui/trace.py` (ADR-CDG-003; relocated from `nodes/` per CDG-008 Phase 1) |
| MCP tool bodies are thin adapters over the same one contract; MCP is a peer, not the center | 2 | `surfaces/mcp/commands/model.py:load_model_tool` (wraps `StateManager.load` → `dgemma.model.load_model`), `surfaces/mcp/commands/generate.py:generate` (wraps `dgemma.loop.run_diffusion`) — CDG-008 Phase 2 |
| Analysis is a downstream consumer; the core neither imports nor re-exports it | 3 | `consumers/analysis.py` (`build_commit_heatmap`, `build_avalanche_curve`, `corroborate_no_mask_token`, `MaskTokenCorroboration`); `dgemma/__init__.py` (no analysis import/re-export) — CDG-008 Phase 3+4, issue #55 |
| Native socket types, not `SIGMAS`/`LATENT` (no lying payload) | 5 | `surfaces/comfyui/socket_types.py` (`DGEMMA_MODEL`, `DGEMMA_CANVAS_STATE`, `DGEMMA_CANVAS_TRACE`, minted once per #35 R2); `surfaces/comfyui/sampler.py` (consumes the mint); ADR-CDG-001 |
| Live per-step view is a read-only observer, not a socket stream | 7 | `surfaces/comfyui/sampler.py:114-161` (`_build_on_frame`, `on_frame`); `run_diffusion(on_frame=…)` at `dgemma/loop.py:477` |
| MCP state manager persists only the model load; every call is fresh | 6 | `surfaces/mcp/state_manager.py:StateManager` (fields: `_model`/`_repo_id`/`_quant` only — no scheduler/canvas/run-state field); CDG-008 Phase 2 Correction 1 |

*Reachability note:* every row above is audited against reachable code. Rows whose
subject does not yet exist carry `NOT-YET-IMPLEMENTED` with the R-item / phase that
creates it — not an empty cell.

---

## Enforcement-surface table (one row per invariant)

The ADR-CDG-008 boundary table, extended with the review's new invariants. Each row
names its test / type / review surface and its status.

| Invariant | Enforcement surface | Status |
|-----------|---------------------|--------|
| Core imports no surface (`dgemma/` never imports `comfy.*` / `nodes.*` / `surfaces.*` / `consumers.*`) | `tests/test_seam.py:36-63` (subprocess `import dgemma`, `sys.modules` leak check) | **In force.** Extended per CDG-008 Phase 1 (issue #52 §4) — the leak check also rejects `surfaces`/`surfaces.*`, in addition to the still-checked `nodes`/`nodes.*`. Extended again per CDG-008 Phase 3+4 (issue #55 §4) to also reject `consumers`/`consumers.*`. |
| MCP surface boundary, both directions (`dgemma` never imports `surfaces.mcp`; `surfaces.mcp` never imports `comfy`/`nodes`/`surfaces.comfyui`) | `tests/test_mcp_surface_seam.py` (subprocess, both directions — ADR-CDG-008 Phase 2 Correction 2: "keep the automated boundary test, do not regress to sk-mcp's review-only posture") | **In force** — CDG-008 Phase 2. Mutation-checked: making `dgemma/__init__.py` import `surfaces.mcp` fails this test (and `test_seam.py`) by name. |
| MCP state manager persists only the model load (never a scheduler/canvas/run-state) | Field-allowlist test on `StateManager` (structural) + same-in/same-out test on the MCP `generate` dispatch (behavioral) | **In force** — CDG-008 Phase 2 Correction 1. `tests/test_mcp_statelessness.py` (`TestStateManagerShape` — mutation-checked: an added `_scheduler` field fails by name; `TestSameInSameOutAtMCPLevel` — two identical `generate` calls on one loaded model yield identical `trace_summary`/`canvas_state`, and two distinct scheduler objects, proving the MCP adapter reintroduces no sharing on top of `run_diffusion`'s own freshness). |
| MCP SDK is an optional dependency; its absence is an actionable error, not a bare `ModuleNotFoundError`, and never blocks the ComfyUI surface | `surfaces/mcp/_mcp_sdk_guard.py:require_mcp_sdk` (called at the top of every `mcp`-dependent module) | **In force** — CDG-008 Phase 2, deliverable 5. `tests/test_mcp_import_guard.py` (subprocess, `mcp` genuinely blocked from resolution) + `tests/test_mcp_sdk_guard.py` (in-process unit test of the same branch) + a same-suite assertion that `surfaces/comfyui/*` still imports clean with `mcp` absent. |
| Core imports no analysis (base contract imports no consumer module) | Subprocess assertion (analysis not in `sys.modules` after `import dgemma`) | **In force** — CDG-008 Phase 4 (issue #55 §4), after the Phase 3 relocation. `tests/test_seam.py::test_dgemma_does_not_import_consumers_package` + the extended `_CHECK_SCRIPT` leak-list (`tests/test_seam.py::test_dgemma_imports_with_zero_comfy_present`). Mutation-checked: re-adding `from consumers.analysis import build_commit_heatmap` to `dgemma/__init__.py` fails both by name. |
| Surfaces are peers over one contract (no logic in a surface body) | ADR-CDG-003's "no `for`-loop-over-steps in a surface body", generalized to `surfaces/*` | Reviewed by eye + `tests/test_trace_node.py` (`DGemmaTrace.render` purity). No mechanized cross-surface import-graph rule. Residual debt, not structural impossibility. |
| Canonical trace, parsed at the door | `run_diffusion` return-type (`dgemma/loop.py:478`); ADR-CDG-004 | **In force at the type level.** |
| Conserved repo identity (`ComfyUI-DiffusionGemma` unchanged across the rename) | Registry mirror + remote (`IDENTITY⊥ENVELOPE`); no code change touches it | **In force by omission** — the roadmap must not touch the repo name. |
| Socket vocabulary minted once (no inline `DGEMMA_*` literal outside the mint module) | Grep-gate test asserting against the module object (only the path string churns with Phase 1) | **In force** — #35 R2 (issue #52). `tests/test_socket_mint.py` (mint-exposure check + grep-gate over `surfaces/comfyui/*.py` + live-node round-trip against the minted set). |
| Composition ordering (capture pre-pin; β-rebuild before pin; pin last writer) | Ordered-composite test over the shared fake-pipeline fixture: `tests/test_step_end_composite.py:TestFixedOrdering`, `TestOrderingIsStructural` (`dgemma/composite.py:StepEndComposite`) | **In force** — #35 R1 (over R4's fixture). ADR-CDG-010. |
| Zero hooks after run ("no hook survives a `run_diffusion` call") | Forward-hook lifecycle context-manager test, clean + raising | **In force** — #35 R5 (F4). `dgemma/hooks.py:install_logit_shaping_hook` (sole install path, `try/finally` teardown); `tests/test_hook_lifecycle.py` (clean, `DiffusionCancelled`, and raising paths, unit-level and through `run_diffusion`). |
| Same-in/same-out walker/pin statelessness (identical calls → identical effective-knob telemetry) | Same-in/same-out test on one loaded model | **In force** — #35 R5 / ADR-CDG-011 F5. `dgemma/loop.py:run_diffusion` builds a fresh scheduler/collector/composite every call; `tests/test_run_diffusion_statelessness.py` (fresh-object proof + identical-telemetry + mutation-non-survival). Now also proven at the MCP dispatch level — see the MCP state manager row above (CDG-008 Phase 2 Correction 1, `tests/test_mcp_statelessness.py`). |
| Diffusers version guard + structural probe (scheduler kwargs, `accepted_index`, `_callback_tensor_inputs`) | Range guard (`>=0.39.0`, matching `pyproject.toml`'s declared bound) patterned on `dgemma/model.py:78` (`_check_transformers_version`) + a structural probe independent of version number: `dgemma/loop.py:_check_diffusers_version`, `dgemma/loop.py:_check_diffusers_structure`; enforced by `tests/test_diffusers_version_guard.py` | **In force** — #35 R3. Names/shapes only: the probe cannot see the anneal formula's body — split out as its own row below. |
| Anneal formula fidelity (`anneal_temperature`'s re-derivation == the temperature the installed scheduler's `step()` actually applies) | Formula-pin test driving the REAL installed `EntropyBoundScheduler.step()` (expected value recovered from `pred_logits`, never from constants copied into the test): `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin` | **In force** — #35 R3 / PR #48 gate finding F-1. |
| Declarative payloads only into `run_diffusion` (no surface-built closures/hooks) | Ingress validation (schedule length == steps; values in binding range; ids in-vocab; fail on unknown) + the composite holding only engine-built participants | **Ingress validation In force** — CDG-010/011 Phase 1 (issue #64, PR #65): `dgemma/ingress.py:validate_ingress` (C1–C4/V1–V6/P1/H1, fail-on-unknown; frozen-dataclass ctor is the unknown-*key* enforcement surface) + H1 rejects two hook sources on one door; `tests/test_ingress.py`, `tests/test_run_diffusion_ingress.py`. The **composite-holding-only-engine-built-participants** half remains `NOT-YET-IMPLEMENTED` — ADR-CDG-010/011 Phases 3/4 (no participant is built from a validated payload yet). |
| `num_inference_steps` non-mutable mid-run | Ingress reject (guards #20's `predictor_steps`/`_num_timesteps` desync) | **In force** — CDG-011 Phase 1 (issue #64, PR #65). `dgemma/ingress.py:validate_control_signals` V4 (a distinct, #20-anchored reject named ahead of the generic V3 unknown-target message) + `MUTABLE_TARGETS` deliberately excludes `num_inference_steps` (`dgemma/payloads.py`); `tests/test_ingress.py::TestControlSignalsIngress::test_v4_num_inference_steps_target_rejected_with_20_anchored_message`, and reject-before-scheduler-construction proven in `tests/test_run_diffusion_ingress.py`. |
| `DiffusionFrame` extension discipline (additive-optional, heavy-field retention policy) | Optional-with-defaults fields; retention policy for heavy `DISTRIBUTION` | **In force (discipline + Tier 0 + raw ids)** — #61 Phase P-A / ADR-CDG-014. `dgemma/types.py:DiffusionFrame` (`entropy`/`top_k_ids`/`top_k_weights`/`distribution`, all optional default `None`) + `CanvasTrace.raw_canvas_ids` (optional default `None`); `dgemma/loop.py:_FrameCollector.on_step_end` derives Tier 0 `entropy` from pre-pin `logits` (capture-first ordering, ADR-CDG-010), `_build_result` populates `raw_canvas_ids` before `excise_thought_channel`. `tests/test_frame_capture_discipline.py` (additive-optional construction, Tier 0 always-on, capture-pre-pin proof), `tests/test_raw_canvas_ids.py` (pre-excision conservation, #9 EOS-in-thought-span probe). Tier 1 (`top_k`)/Tier 2 (`distribution` + budget) derivation and the `capture=` ingress knob remain `NOT-YET-IMPLEMENTED` — ADR-CDG-014 Phases P-B/P-C. |
| `KV_CACHE` payload types + ingress validation (V1-V6, fail-on-mismatch, both-token remedy messages) | `dgemma/types.py` (`KVCache`/`Provenance`/`EditOp` dataclasses, `CanvasTrace.injected_cache_provenance` additive-optional field) + `dgemma/kv_cache.py:validate_kv_cache_ingress`/`geometry_from_model`/`tokenizer_fingerprint` | **In force (types + validator only)** — ADR-CDG-012 / issue #62 Phase 1. `tests/test_kv_cache_types.py` (dataclass construction, illegal orphan state as data, `injected_cache_provenance` default), `tests/test_kv_cache_ingress.py` (V1-V6 happy path + every raise path, DV.3b both-token `match=`), `tests/conftest.py`'s `synthetic_kv_cache`/`fake_dgemma_model` fixtures (§L). No `run_diffusion` door exists yet to call this validator (`kv_cache=` param, OUT-3 population, surface nodes, socket mint, examples/workflows, DV.1/DV.2/DV.3a/DV.3c enforcement) — `NOT-YET-IMPLEMENTED`, ADR-CDG-012 Phases 2/3. |
| Shared fake-pipeline/scheduler fixture (N steps, mutable `config`, hook-recording model, `{"canvas":…}` application) | `tests/conftest.py:fake_pipeline_factory` (`FakeEntropyBoundScheduler`, `HookRecordingModel`, `FakeDiffusionGemmaPipeline`); self-tests in `tests/test_conftest_fake_pipeline.py` | **In force.** #35 R4. "Mutable `config`" resolved against the real `diffusers` `FrozenDict` (write-raises; mutation only via `register_to_config`, verified against the installed-wheel source) — see `tests/conftest.py`'s module docstring. |
| Frames↔images index correspondence not untagged | Per-image frame-key tag or explicit no-zip contract (`CONSERVE-DATA-BOUNDARY`) | `NOT-YET-IMPLEMENTED` — #35 F7/F9, reconciled in ADR-CDG-009 / PR #31 ratification. |
| E2E battery imports nothing from `dgemma`/`surfaces`/`consumers` (black-box independence, ADR-CDG-013 Decision 1) | `tests/e2e/test_e2e_import_guard.py` (subprocess `sys.modules` leak check, mirroring `tests/test_seam.py`'s shape, + a static AST scan) | **In force** — battery phase E0 (issue #59). |
| Consumers orchestrate; they do not extend (rule 8 — sweeps/loops/batteries sequence surface primitives; no consumer-serving path accretes into the core) | Review-only + the existing rule-2 no-denoising-loop-in-surface-body posture (a sequencing loop belongs *above* the surfaces, not inside one; a body that grew one would already trip the surface-body review). The one in-repo consumer, `tests/e2e/driver.py`, is additionally *mechanically* black-boxed by the E2E import guard above (ADR-CDG-013 Decision 1) — but that guard proves independence of the driver, not the general rule | **Review-only — known-fragile** (GROUND_PHYSICS discipline 6). No mechanized cross-tier rule asserts "no consumer-serving path in the core" the way `tests/test_seam.py` asserts import direction; prose + the rule-2 review gate carry it today. A mechanizable surface (e.g. a consumer-plane import-graph rule) awaits the served-engine ADR that would give the tier a code boundary. Named honestly per issue #92 scope item 3. |
| Lifecycle & tenancy of the ~53 GB load (in-process single-tenant today; no lifecycle delegation) | The physics of the 48 GB card (two ~53 GB loads do not fit) + rule 6's holder-only `StateManager` (`tests/test_mcp_statelessness.py` — asserts no run-state cached, does **not** assert single-tenancy) | **Review-only — known-fragile** (GROUND_PHYSICS discipline 6). The current single-tenancy is enforced by hardware limit + prose, not a test. The served-engine topology (ADR-candidate, §Lifecycle & tenancy) is the future home of a real enforcement surface (an engine-client boundary test) — **named, not decided**, trigger = a second concurrent surface needing the resident model. |
| Node-pack coverage measured inside the ComfyUI subprocess and merged with unit-suite data (ADR-CDG-013 Decision 4) | `[tool.coverage.run]` (`pyproject.toml`: `parallel`, `concurrency=["thread"]`, `sigterm`, `source=["dgemma","surfaces","consumers"]`) + `sitecustomize.py` (`coverage.process_startup()`, a no-op unless `COVERAGE_PROCESS_START` is set) + `coverage combine` | **In force (mechanism verified E0; not yet exercised against a real battery run — E1/live phases pending the three operator-scheduled preconditions in issue #59 §5)**. |
| Per-scenario green + combined-coverage readback banked to issue #59 / plan.md ("done" = all E2E scenarios green on the live model) | Battery run evidence, banked per ADR-CDG-013 Implementation Notes | `NOT-YET-IMPLEMENTED` — battery phases E2–E4 (issue #59). E0 landed the harness skeleton + S1; this phase (P2) adds S2 (full-knob), S3 (thinking-toggle, marked `xfail(strict=True)` against issue #9 — expected RED once live), and S4 (trace readout) to `tests/e2e/test_battery.py`/`driver.py`. All four scenarios (S1–S4) remain correctly SKIP-gated pending the three operator-scheduled live preconditions in issue #59 §5 — no scenario has run against the live model yet, so "green on the live model" is still unproven; only the harness/assertion code is built and unit-proven pre-infra. |

---

## What "instant fail" looks like

One row per invariant rule, each violation paired with its correct shape.

| Violation | Valid form |
|-----------|------------|
| A `surfaces/comfyui/*.py` body loops over denoising steps | The body unpacks args, calls `run_diffusion` once, wraps the result (ADR-CDG-003) |
| `dgemma/*.py` does `import comfy` / `from surfaces import …` | The surface imports the core; the core imports nothing surface-shaped (rule 1) |
| A new analysis function added to `dgemma/*.py` | Added to `consumers/analysis.py`, importing `CanvasTrace` from `dgemma.types` (rule 3) |
| `RETURN_TYPES = ("DGEMMA_CANVAS_TRACE",)` inline at a new node site | Reference the socket string from the mint module; grep-gate rejects the inline literal (rule 4) |
| An entropy budget passed as a `SIGMAS` tensor; a `DISTRIBUTION` socket carrying only a scalar | A native `DGEMMA_*` type carrying the real payload, validated at ingress (rule 5) |
| MCP state manager caches a live scheduler across calls | Persist only `load_model`'s output; build a fresh scheduler/canvas/run-state per call (rule 6) |
| A forward hook installed in run A left registered into run B | Install via the R5 lifecycle context manager; no hook survives the call (rule 6) |
| A surface passes a closure/hook into `run_diffusion` to shape logits | Pass declarative `constraints=` / `control_signals=`; the engine installs the hook, validated at ingress (rule 7) |
| Capture ordered after pin (records constraint-asserted tokens as model-committed) | Capture runs before any canvas-writer; `pinned_mask` distinguishes the two (rule 7, ADR-CDG-010) |
| Mutating `num_inference_steps` mid-run | Reject at ingress; mutate only `scheduler.config` values `step()` reads fresh (rule 7, ADR-CDG-011) |
| A sweep/loop/battery written *inside* a surface body (the ComfyUI node runs N configs itself) or accreted into the core to serve one surface | The sequencing lives in the orchestration/consumer plane *above* `surfaces/*`; the surface exposes one contracted primitive the consumer calls N times (rule 8, EXTERNAL sense) |
| A consumer capability that exists on only one surface — e.g. the structured run-config/run-log that today rides `surfaces/comfyui/`'s `on_frame` observer but has no `surfaces/mcp/` equivalent | Surface parity: a run-record vocabulary is core/consumer-tier identity (`IDENTITY⊥ENVELOPE`), so it is reachable from every surface, promoted when a second surface needs it (rule 8; live instance below) |

**Live instance of the drift this tier catches (2026-07-16 session).** The structured
run-log (issue #72) is presently a **ComfyUI-surface-only** capability: it rides
`surfaces/comfyui/`'s read-only `on_frame` observer (the sanctioned executable crossing)
with a post-run `consumers/` serializer, and its promotion to the MCP surface is recorded
as a **trigger on #72**, not yet built. This is the *first live instance* of the drift
class rule 8 and the lifecycle/tenancy naming exist to catch: a consumer-facing capability
settling onto one surface before the orchestration tier is named makes surface-parity
gaps invisible. Naming the tier is what turns "the MCP surface silently lacks run-logging"
from an unnoticed asymmetry into a tracked #72 trigger. The record identity is the
`dgemma/types.py` dataclasses (`IDENTITY⊥ENVELOPE`, issue #72 §Placement), so parity is
reachable — the gap is one of *which surface exposes it*, exactly the rule-8 concern.

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
| ADR-CDG-012: MITM the AR/diffusion seam — `KV_CACHE` socket + `DGemmaEncode`/`DGemmaDenoise` | `KVCache`/`Provenance`/`EditOp` types (rule 4 `IDENTITY⊥ENVELOPE`); ingress fail-on-mismatch (rule 5); declarative `kv_cache=` door (rule 7, Phase 2+) | `decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md` — Accepted (ratified 2026-07-13); Phase 1 (types + ingress, issue #62) landed here, Phases 2-5 `NOT-YET-IMPLEMENTED` |

---

## Anticipated evolution — two live tracks

*This replaces the prior document's §7 "not scoped, not designed, not started" claim:
the alignment refactor is now scoped (ADR-CDG-008) and reviewed (#35).*

**Track 1 — CDG-008 alignment (Phases 1–5).** Sequenced, dependency-respecting:
**Phase 1 (landed, issue #52) renamed `nodes/` → `surfaces/comfyui/` (+ `web/` →
`surfaces/comfyui/web/`), riding #35 R2 (the socket-type mint,
`surfaces/comfyui/socket_types.py`)**; **Phase 2 (landed) added `surfaces/mcp/`**,
transcribed from `semantic-kinematics-mcp` with the two corrections (stateless
state manager, kept boundary test — `surfaces/mcp/state_manager.py`,
`tests/test_mcp_surface_seam.py`); **Phase 3 (landed, issue #55) relocated
analysis to `consumers/analysis.py`, out of `dgemma/`**; **Phase 4 (landed,
issue #55) added the base-contract-imports-no-analysis test**
(`tests/test_seam.py::test_dgemma_does_not_import_consumers_package`); Phase 5
this document. Execution order is set by #35's delta: **R4 (shared fixture)
before R1 (composition layer)**, then R5; R3 anytime; R2 landed with Phase 1;
rung-4 analysis landed with Phase 3.

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
- ~~`consumers/` vs `surfaces/analysis/` naming~~ — ADR-CDG-008 Open Question #1,
  **settled to `consumers/`** in the Phase-3 relocation `plan` pass (issue #55 §1);
  see the ADR's 2026-07-13 amendment note.
- Whether `CanvasTrace` moves to a shared contract module — ADR-CDG-008 Open Question
  #2; default is "stays in `dgemma/`." **Confirmed** — the default held; only the
  analysis functions moved (issue #55 §2).
- Whether the refactor needs a `plan` pass before touching `__init__.py`'s discovery
  contract — ADR-CDG-008 Open Question #3; resolution: yes. **Satisfied by issue #52**
  (the plan pass) and executed by this Phase-1 branch.
