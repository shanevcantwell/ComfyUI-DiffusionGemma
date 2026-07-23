# Architecture: ComfyUI-DiffusionGemma

The layering / single-contract invariant. States what the code must do and names the gaps that remain. Strong-register claims carry a `path:symbol` citation of enforcing code or an explicit `NOT-YET-IMPLEMENTED` token naming the ADR / issue that creates it.

Doctrine by reference: ground-physics invariants in
`../operating-doctrine/ground-physics/GROUND_PHYSICS.md`. Handles cited below
(`ONE-DOOR`, `STATELESS-CORE`, `ONE-MINT`, `EMIT-CANONICAL / PARSE-AT-THE-DOOR`,
`IDENTITY⊥ENVELOPE`, `CONSERVE-ACROSS-THE-DATA-BOUNDARY`) resolve there.

---

## The invariant (read this first)

Eight rules, all simultaneous. Rules 1–7 govern the core/surface seam; rule 8 governs the tier above surfaces.

1. **One core, one contract.** `dgemma/` is the sole contract: `load_model` (`dgemma/model.py:157`) and `run_diffusion` (`dgemma/loop.py:465`, returning `(text, CanvasState, CanvasTrace)`). Every surface reaches the model *only* through these two functions. The core imports with zero ComfyUI present — enforced by subprocess test in `tests/test_seam.py`. *(→ ADR-CDG-003, ADR-CDG-008 · `ONE-DOOR`)*

2. **MCP is the canonical surface; ComfyUI consumes MCP.** The MCP surface wraps core into tools. ComfyUI calls those tools — it does not import `dgemma/` directly. No `for`-loop-over-denoising-steps in a consumer body. *(→ ADR-CDG-008 · `ONE-DOOR`)*
   **GAP (#137):** ComfyUI currently imports core directly (`surfaces/comfyui/loader.py`, `sampler.py`). Reconciling to MCP-consumer topology is tracked in issue #137.

3. **Analysis is a downstream consumer, not core work.** The core emits canonical `CanvasTrace`; derived analysis parses it from outside the core's import graph. Enforced by `tests/test_seam.py::test_dgemma_does_not_import_consumers_package`. *(→ ADR-CDG-008 · `EMIT-CANONICAL / PARSE-AT-THE-DOOR`)*

4. **Socket vocabulary is minted once, surface-side; identity is core-side.** `DGEMMA_*` socket strings live in one mint module (`surfaces/comfyui/socket_types.py`). Payloads are `dgemma/types.py` dataclasses — envelope and identity orthogonal. *(→ ADR-CDG-001, ADR-CDG-008 · `ONE-MINT`, `IDENTITY⊥ENVELOPE`)*

5. **Payloads mean what they say.** Native socket types carry real payloads or they are lying — validated at ingress, fail on unknown. *(→ ADR-CDG-001 · `EMIT-CANONICAL / PARSE-AT-THE-DOOR`)*

6. **The core is stateless across runs; only the model load persists.** No mutable run-state survives a `run_diffusion` call. Two identical calls yield identical telemetry. The ~53 GB model load is the *only* persisted object. *(→ ADR-CDG-008, #35 R5 · `STATELESS-CORE`)*

7. **Step-end intervention enters as declarative payloads through one door.** `run_diffusion` widens only by validated declarative payloads (`constraints=`, `control_signals=`, `capture=`). No surface-supplied closures or hooks — ingress rejects them. The only executable crossing is the read-only `on_frame` observer. *(→ ADR-CDG-010/011 · `ONE-DOOR`)*

8. **Consumers orchestrate; they do not extend.** Run sequencing — sweeps, loops, batteries — belongs to the tier *above* surfaces, never inside a surface body and never accreted into the core. A consumer that reaches past the surface contract to touch core internals is an instant fail. *(→ sk-mcp rule 3 transcribed · `ONE-DOOR`)*

---

## The layers

Top (consumer) to bottom (substrate).

```
+-----------------------------------------------------------+
| ORCHESTRATION / CONSUMER PLANE  (rule 8; EXTERNAL, above) |
|   humans @ ComfyUI graph . MCP clients/agents . scripts   |
|   tests/e2e/driver.py  (in-repo instance; black-box)      |
+-----------------------------------------------------------+
        |  MCP tool calls only
   consumers/  (analysis: parses CanvasTrace)
        v
+-----------------------------------------------------------+
| SURFACE TIER                                              |
|   mcp/  — canonical surface (server.py, state_manager.py) |
+-----------------------------------------------------------+
        |  load_model + run_diffusion  -- THE ONE CONTRACT
+-----------------------------------------------------------+
| dgemma/   core -- surface-agnostic, zero ComfyUI present  |
|   model.py (load) . loop.py (drive) . types.py (contract) |
+-----------------------------------------------------------+
        |
   torch . transformers . diffusers   -- shared substrate

   [lifecycle & tenancy plane -- NOT-YET-BUILT: today in-process single-tenant;
    served-engine (llauncher-owned) is an ADR-candidate, not decided]

   **GAP (#137):** `surfaces/comfyui/` currently sits beside MCP calling core directly.
   The diagram above shows the target topology: ComfyUI consumes MCP tools.
```

### Orchestration / consumer plane — sequences the surfaces (rule 8)

External to the pack. Composes and sequences already-contracted surface primitives. Sweeps, loops, batteries live here — never in a surface body, never accreted into the core. The E2E driver (`tests/e2e/driver.py`) is the one in-repo instance: it imports nothing from `dgemma`/`surfaces`/`consumers`, enforced by `tests/e2e/test_e2e_import_guard.py`.

**EXTERNAL framing:** this is sk-mcp's sense of "orchestration" (above the contract), not prompt-prix's internal runner tier (below entry points). CDG grows no internal runner tier.

### Consumers — derived analysis (`consumers/`)

Pure trace-analysis functions: `consumers/analysis.py` (`build_commit_heatmap`, `build_avalanche_curve`, `build_entropy_heatmap`, `build_token_identity_grid`). Parses an already-captured `CanvasTrace`; never re-derives what the core emitted, never drives the model. Imports the contract type; the core imports nothing from here.

### Surface tier — MCP is canonical; ComfyUI consumes it

The MCP surface wraps core into tools. Consumers call those tools — they do not import `dgemma/` directly.

- **`surfaces/mcp/`** — Canonical surface (`server.py`, `state_manager.py`, `commands/{model,generate}.py`). Thin adapter: unpack args → call one `dgemma.*` function → wrap the result. State manager persists only the loaded model (rule 6).
- **`surfaces/comfyui/`** — ComfyUI node graph (`loader.py`, `sampler.py`, `trace.py`, `token_trace.py`, `frames_image.py`, `socket_types.py`) + `web/`. **GAP (#137):** currently imports core directly instead of consuming MCP tools. Reconciling to MCP-consumer topology is tracked in issue #137.

### Core — the one contract (`dgemma/`)

The model, types, denoising loop. Imports and runs with zero ComfyUI present. Emits canonical `CanvasTrace`. Holds no cross-run mutable state. Widens `run_diffusion` by declarative payloads only.

---

## Lifecycle & tenancy — honest absence

Rule 6 governs cross-*run* statelessness; it says nothing about the *lifecycle* of the ~53 GB load itself. CDG loads **in-process, single-tenant**: whatever process calls `load_model` holds the weights for its lifetime. The 48 GB card fits one load at a time. There is no lifecycle delegation — nothing external starts, stops, swaps, or arbitrates tenancy.

**Anticipated evolution (ADR-candidate, named not decided):** a served-engine topology where one resident process owns the single load and surfaces become clients; lifecycle delegated to llauncher (sk-mcp's answer). Trigger: a second concurrent surface needing the resident model. See issue #92.

---

## The data-boundary crossing discipline

Bulk artifacts leaving one surface/process boundary for another carry a **pointer + identity sidecar**; bulk bytes travel out of band. This is `CONSERVE-ACROSS-THE-DATA-BOUNDARY` made structural. Full evidence in issue #103 and ADR-SKM-007 (`../semantic-kinematics-mcp/docs/ADRs/proposed/adr-skm-007-bulkembedder-primitive-decomposition.md`).

**The failure class:** data that type-checks but crosses mints — shape-identical, mint-incommensurable. Only carried mint identity can refuse a mismatched mix at re-entry.

### The seven primitives (four layers)

Named so consumers cite subsets by name:

- **Boundary/identity.** (1) mint-identity guard; (2) self-distrust on resume
- **Durability.** (3) append-only progress ledger; (4) typed failure markers; (5) bounded volatile head
- **Transport economics.** (6) ground-verified partitioning; (7) budgeted packing

Payload validity is vocabulary-owned, outside this discipline.

### Consumer status

| Consumer | Primitives | Status |
|----------|-----------|--------|
| Run-log emission (#72) | 1+2+3+4 | **In force** ComfyUI-side; MCP promotion pending (#103 Scope B) |
| Serialized `kv_cache` (ADR-CDG-012 tier-2) | 1 + V1–V6 door | NOT-YET-IMPLEMENTED — #103 fork, Phase 5 conditional |
| Tier-2 `DISTRIBUTION` artifacts (ADR-CDG-014) | 1+2+3+5 | Capture in-core; artifact/banking story NOT-YET-IMPLEMENTED |
| `runs/` raw-data banking (#101) | 1+3 | Proposal — draft-for-ratification |

---

## What the invariant does NOT govern (out of scope)

- **The published repo name** — conserved identity, out of scope for renaming.
- **`CanvasTrace` living in `dgemma/`** — the emitted canonical type sits core-side as the contract surface both sides depend on. A consumer importing it is not a violation. *(ADR-CDG-008 OQ2)*
- **GGUF / llama.cpp backend (ADR-CDG-007)** — inference-only alternate backend, beside the drive seam, not through it. Rejected 2026-07-06.
- **Shared substrate** — `torch`, `transformers`, `diffusers` are beneath every layer; both core and surface importing them is not a seam crossing.

---

## The step-end intervention architecture (decided target)

Expansion lands core-side of the seam, so every surface inherits it. Shape decided per #35, ADR-CDG-010/011.

- **Engine-internal ordered composite** — `dgemma/composite.py:StepEndComposite` holds engine-built participants in fixed order: capture → β-rebuild → pin → walker. Capture runs before any canvas-writer (pre-pin truth). Pin is the last writer. `pinned_mask` rides each frame. **In force.** Beta-rebuild slot built; beta-viscosity body deferred (ADR-CDG-010 OQ2).

- **Live view is not a composite participant** — stays on `on_frame` read-only observer seam (`run_diffusion(on_frame=…)`). Receives built `DiffusionFrame`, return ignored. The only executable crossing the surface owns. **In force.**

- **Declarative payloads on `run_diffusion`** — `constraints=`, `control_signals=`, `capture=` validated at ingress (schedule length == steps; values in binding range; ids in-vocab; fail on unknown). Foreign callables rejected. **In force** for ingress + pin/walker participants; beta-rebuild body deferred.

- **Two-mechanism model for givens/constraints** — logit mask shapes *what commits*; canvas re-assertion guarantees *what conditions*. NOT-YET-IMPLEMENTED (ADR-CDG-010).

- **Forward-hook lifecycle context manager** — `dgemma/hooks.py:install_logit_shaping_hook`, sole install path, `try/finally` teardown. No hook survives a `run_diffusion` call. **In force.** The mask body itself NOT-YET-IMPLEMENTED.

- **Control signals as CV / LFO** — unitless per-step control signal; units declared at binding (CV principle). Engine walker mutates `scheduler.config`; `num_inference_steps` non-mutable (ingress reject). Effective-knob telemetry rides the frame. **In force.**

---

## Why one core, one contract

- **Single door** — a core that imports with zero ComfyUI present is testable in isolation; a surface that wraps two functions cannot accrete logic other surfaces can't reach. One contract means new envelope = wrapper, not fork.
- **Stateless across runs** — the ~53 GB load persists but all run-state rebuilds per call. An un-torn-down hook from run A shapes run B's logits; a cached scheduler carries mutated dims forward. Persisting only the immutable load keeps identical calls identical.
- **Analysis is a consumer** — keeping analysis out of the core's import graph turns a prose-only boundary into an enforced one: subprocess test asserts no `consumers` in `sys.modules` after `import dgemma`.
- **Declarative payloads, not closures** — a surface-supplied callable is unvalidatable at ingress and re-opens the door. Declarative payloads are checkable; the forbidden shape becomes structurally unrepresentable.

---

## Conformance summary

All original violations from #35 resolved. Remaining gaps: beta-rebuild body (ADR-CDG-010 OQ2), frames↔images index correspondence (ADR-CDG-009), E2E battery live-run evidence (issue #59). Full enforcement-surface table and detailed conformance history in the issue tracker and ADRs.

---

## Relation to the decision record

See `decisions/adr-cdg-*.md` (18 documents). Key ADRs:

| ADR | What it fixes |
|-----|---------------|
| ADR-CDG-003 | Core/adapter split — rules 1, 2 |
| ADR-CDG-004 | `run_diffusion` single-entry drive contract — rule 1 |
| ADR-CDG-008 | MCP-center multi-surface topology — rules 1–4, 6 |
| ADR-CDG-010 | Constraint composite and pinned mask — rule 7 |
| ADR-CDG-011 | Control-signal CV/LFO mod matrix — rules 6, 7 |
| ADR-CDG-012 | `KV_CACHE` socket + encode/denoise nodes — rules 4, 5, 7 |

---

## Anticipated evolution — two live tracks

**Track 1 — CDG-008 alignment.** Phases 1–4 landed (naming, MCP surface, analysis relocation, boundary test). Phase 5 is this document.

**Track 2 — research expansion.** The liquid-phase-decoding bench (`docs/experiments/liquid-phase-decoding/concept.md`): six-seam inventory with five falsifiable H0s. All capabilities land core-side of the seam so every surface inherits them. Graduation trigger: confirmed H0 → ADR (socket type / scheduler seam).
