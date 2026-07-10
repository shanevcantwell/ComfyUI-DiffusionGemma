# ADR-CDG-008 — MCP-center topology: one core, MCP as the base surface, ComfyUI as a peer surface, analysis as a downstream consumer

**Status**: accepted
**Date**: 2026-07-08
**Related**: ADR-CDG-003 (node-engine seam — this ADR **extends** it: same
core/adapter split, generalized from "ComfyUI nodes over an engine" to "N peer
surfaces over one core"), ADR-CDG-004 (drive seam — `run_diffusion` is the
single drive entry this ADR names as one half of the base contract),
ADR-CDG-001 (native socket types — the ComfyUI-shaped face is now explicitly
*one* surface's envelope, not the core's contract)

---

## Context

The pack was built ComfyUI-first (ADR-CDG-003): a ComfyUI-agnostic engine
`dgemma/` under a `nodes/` adapter layer, with the seam enforced by a
subprocess test (`tests/test_seam.py`). That split already did the hard part —
the core imports and runs with **zero ComfyUI present** — but the *naming* and
the *topology framing* still put ComfyUI at the center: the whole surface layer
is called `nodes/` (a ComfyUI word), and `dgemma/__init__.py` re-exports
analysis functions as if they were part of the core's public face.

The operator has decided the real topology is **MCP-center**, aligning this
repo with the ecosystem's core→orchestration→surfaces layering already in force
in sibling repos (`../llauncher` runs `core/` + `operations/` + peer surfaces
`mcp_server/` / `cli/` / `agent/` / `ui/`; `../harness-tools` layers the same
way). The core's public API — `load_model()` (`dgemma/model.py:157`) and
`run_diffusion()` (`dgemma/loop.py:465`, returning
`(text, CanvasState, CanvasTrace)` per ADR-CDG-004, `loop.py:479`) — is the
**one contract**. **MCP is the primary/base surface** over that contract; the
**ComfyUI node graph is one surface among peers**, not the center.

**What is already clean** (do not re-litigate; the decision inherits it):

- `dgemma/` is surface-agnostic and tested as such. `tests/test_seam.py:36-63`
  runs `import dgemma` in a fresh subprocess and asserts no `comfy.*` / `nodes.*`
  module leaked into `sys.modules` — the absence of a `comfy` package in the
  venv *is* the enforcement, not a maintained stub (`test_seam.py:1-14`).
- The drive contract is single-entry and canonical: `run_diffusion` never
  returns a bare string, always `(text, CanvasState, CanvasTrace)`
  (`loop.py:535-536`, ADR-CDG-001 Addendum / ADR-CDG-004). `CanvasTrace` is the
  **canonical emission** a surface or a consumer parses at its door.

**What is not yet clean, and is the subject of this ADR:**

1. The surface layer is named `nodes/` — a ComfyUI term applied to the whole
   surface tier. There is no room in that name for a peer MCP surface.
2. Derived **analysis** lives *inside the core's import graph*:
   `dgemma/sampling.py` (`build_commit_heatmap`, `build_avalanche_curve`,
   `corroborate_no_mask_token`, `MaskTokenCorroboration`) is re-exported by
   `dgemma/__init__.py:26-31,49-51`, and the `DGemmaTrace` node
   (`nodes/trace.py`) consumes it. Analysis parses an already-captured
   `CanvasTrace` — it is a **downstream consumer**, not core work — yet it is
   bundled into the base contract's public surface.

## Decision

Adopt an **MCP-center, multi-surface, single-repo** topology. Three concrete
commitments:

1. **The core is the one contract; MCP is the base surface; ComfyUI is a peer.**
   `dgemma/` (`load_model` + `run_diffusion`) is the sole contract. MCP is the
   primary surface over it. The ComfyUI node graph is **one surface among
   peers**, with no privileged position. All surfaces are **siblings over the
   same core, in this one repo** — there is no separate repo for MCP.

2. **Internal naming is surface-neutral.** The top-level `nodes/` directory (a
   ComfyUI word for the whole surface tier) becomes a generic surface layout:

   ```
   dgemma/                     # core — the one contract (UNCHANGED, already neutral)
   surfaces/
     comfyui/                  # was nodes/   — one surface among peers
       web/                    # was top-level web/ — this surface's client assets
     mcp/                      # NEW peer surface over the same core
   consumers/                  # OR surfaces/analysis/ — derived analysis (see #3)
   ```

   The published **repo name `ComfyUI-DiffusionGemma` stays** — it is
   registry-mirrored, remote-published, conserved identity
   (`IDENTITY⊥ENVELOPE`: the identity handle is orthogonal to the internal
   surface envelope). Renaming the repo is explicitly **out of scope**; only
   the internal directory vocabulary changes.

3. **Analysis is out of scope for the core / base contract — enforced by
   relocation.** The core **emits** the canonical `CanvasTrace`
   (`run_diffusion`); derived analysis **parses** it. Per
   `EMIT-CANONICAL / PARSE-AT-THE-DOOR`, analysis is a downstream consumer, not
   core work. To make "out of scope" an *enforced* boundary rather than a prose
   claim, the analysis code (`dgemma/sampling.py` + its `dgemma/__init__.py`
   re-exports) is **relocated out of `dgemma/`'s import graph** into its own
   consumer module, so a test can assert the base contract does not import
   analysis. Relocation *is* the enforcement mechanism, not a cosmetic move.

## Rationale

### Positive Consequences

- **One contract, many envelopes.** With the core as the sole contract and MCP
  as the base surface, a new surface (CLI, an agent tool, a web API) is added by
  wrapping `load_model` + `run_diffusion` — the same move ADR-CDG-003 defined
  for nodes, now generalized. No surface can accrete logic the others can't
  reach, because the logic isn't in the surface.
- **The neutral naming stops re-teaching ComfyUI bias.** `surfaces/comfyui/`
  beside `surfaces/mcp/` reads as "these are peers"; `nodes/` reads as "ComfyUI
  is the thing." The layout carries the topology so a cold reader infers it
  correctly without the ADR.
- **Relocating analysis buys a real invariant.** Today `test_seam.py` proves
  `dgemma` doesn't import `comfy`/`nodes` — but it says nothing about analysis,
  which currently lives *inside* `dgemma`. Moving analysis out lets a test
  assert the base contract imports **no analysis**, closing a boundary that is
  presently prose-only (the `dgemma/sampling.py` docstring asserts consumer
  status; nothing enforces it).
- **Ecosystem alignment.** The core→surfaces shape already governs `llauncher`
  and `harness-tools`; converging this repo onto it lowers the cross-repo
  cognitive load the curator seat carries.

### Negative Consequences

- **Relocation cost, and it touches the ComfyUI discovery contract.** Moving
  `nodes/`→`surfaces/comfyui/` and `web/`→`surfaces/comfyui/web/` forces edits
  to the pack entry point: `WEB_DIRECTORY = "./web"` (`__init__.py:53`) and the
  three surface imports (`__init__.py`, both the `if __package__` and `else`
  branches) must move with the directories. The dual-context import gate
  (ADR-CDG-003's observed loader-context fix, `nodes/__init__.py`,
  `tests/test_comfyui_loader_context.py`) must keep passing across the move —
  ComfyUI puts `custom_nodes/` on `sys.path` and loads by directory-derived
  name, so the relative-import branch is load-bearing in production.
- **Test churn.** `tests/test_trace_node.py` monkeypatches
  `nodes.trace.build_commit_heatmap` (and siblings) and
  `tests/test_dual_context_import.py` asserts
  `module.build_commit_heatmap.__module__ == f"{root}.dgemma.sampling"` — both
  encode the *current* module paths and must be updated to the new analysis
  location. `tests/test_sampling.py` imports the analysis functions directly.
  These are mechanical path updates, not logic changes.
- **A migration where the sampler surface behavior does not change.** The
  ComfyUI surface's runtime behavior is **unaffected** — `DGemmaLoader →
  DGemmaSampler → DGemmaTrace` still runs identically; only its import paths and
  directory home move. This is a structural refactor, not a feature change.

## Enforcement surfaces (per boundary this ADR names)

| Boundary | Invariant | Enforcement surface | Status |
|---|---|---|---|
| Core imports no surface | `dgemma/` never imports `comfy.*` / `nodes.*` | `tests/test_seam.py:36-63` (subprocess `import dgemma`) | **In force.** Must be updated to also reject `surfaces.*` after the rename. |
| Core imports no analysis | base contract (`dgemma/`) does not import the analysis/consumer module | **NONE YET — prose-only.** `dgemma/sampling.py`'s docstring asserts consumer status; nothing tests it, and today the code contradicts it (`dgemma/__init__.py:26-31` re-exports analysis). | **Prose-only → the relocation exists to create this test.** A NEXT-phase subprocess assertion (analysis not in `sys.modules` after `import dgemma`) is the target enforcement surface. |
| Surfaces are peers over one contract | every surface only wraps `load_model` + `run_diffusion`; no surface holds logic | ADR-CDG-003's "no `for`-loop-over-steps in a surface body" one-line test, generalized to `surfaces/*` | Currently reviewed by eye + `tests/test_trace_node.py` (purity of `DGemmaTrace.render`). No mechanized cross-surface check exists; flag as prose-adjacent. |
| Canonical trace, parsed at the door | `run_diffusion` emits `CanvasTrace`; consumers parse it, never re-derive | `run_diffusion` return-type (`loop.py:479,535-536`), ADR-CDG-004 | In force at the type level. |
| Conserved repo identity | published name `ComfyUI-DiffusionGemma` unchanged across the internal rename | registry mirror + remote (`IDENTITY⊥ENVELOPE`); no code change touches it | In force by omission — the roadmap must not touch the repo name. |

**Prose-only flags to close (name-the-enforcement-surface discipline):**
the "core imports no analysis" boundary is the load-bearing one and has **no**
enforcement surface today — this ADR's relocation roadmap is what turns it from
prose into a test. The "surfaces are peers, no logic in a surface" invariant is
enforced only per-surface and by review, not by a cross-surface import-graph
rule; note it as residual debt, not a claim of structural impossibility.

## Alternatives Considered

*Recorded for the cold reader, not to reopen — the operator has decided the
topology. These are the shapes the decision rejected.*

### Option A: A separate repo for the MCP surface

**Why rejected:** Surfaces share one core with one lifecycle; splitting MCP into
its own repo would fork the core (vendored or submoduled) or invert the
dependency, and every core change would need cross-repo coordination. The
single-repo, sibling-surfaces shape keeps one core, one test suite, one version.
(Ecosystem precedent: `llauncher` keeps `mcp_server/` in-repo beside its other
surfaces, not spun out.)

### Option B: Rename the repo to something surface-neutral

**Why rejected:** The repo name is published, registry-mirrored, and remote-live
— conserved identity under `IDENTITY⊥ENVELOPE`. Renaming it to "fix" an internal
naming bias would break the conserved handle to fix an *internal envelope*
detail. The envelope (directory layout) is what carries the bias; changing it is
free, changing the identity is expensive and lossy. Neutral internals under a
conserved name is the correct split.

### Option C: Leave analysis in `dgemma/`, assert its consumer status in prose

**Why rejected:** This is the status quo, and it is exactly the failure mode the
decision closes. `dgemma/sampling.py`'s docstring already *claims* consumer
status, yet `dgemma/__init__.py:26-31` re-exports it into the core's public face
— the prose and the code disagree, and only the code runs. An "out of scope"
boundary with no enforcement surface is one refactor from gone. Relocation is
the *only* option that lets a test assert the boundary.

## Open Questions

- [ ] **Where does analysis land — `consumers/` or `surfaces/analysis/`?**
      Analysis is a consumer of the trace, but `DGemmaTrace` is *also* a ComfyUI
      surface node that wraps it. Candidate split: pure trace-analysis functions
      (the current `dgemma/sampling.py` body) → a surface-neutral `consumers/`
      (or `analysis/`) module; the ComfyUI-shaped `DGemmaTrace` adapter stays in
      `surfaces/comfyui/` and imports the consumer. **Resolution trigger:**
      settle when the relocation phase is planned (a `plan` pass over this ADR),
      before any file moves.
- [ ] **Does `CanvasTrace` (`dgemma/types.py`) stay in the core, or move to a
      shared contract module the consumers import?** It is the emitted canonical
      type, so it plausibly stays core-side as the contract surface both the core
      and its consumers depend on. **Resolution trigger:** decide alongside the
      analysis-relocation question; default is "trace type stays in `dgemma/` as
      the contract, analysis *functions* move out."
- [ ] **Is this refactor large enough to need a `decompose-problem` /
      `plan` pass before execution?** The roadmap below is sequenced but not
      step-level. **Resolution:** yes — recommend a `plan` pass over Phases 1–4
      before touching `__init__.py`'s discovery contract, given the loader-context
      fragility (ADR-CDG-003).

**Resolution plan:** all three are settled during a `plan` pass over the roadmap
below; none blocks *recording* this direction, and none should be resolved by
silently moving files ahead of that pass.

## Sequenced cleanup roadmap (recorded next-actions — NOT executed in this ADR)

Dependency-respecting. Each phase names the files it touches and what becomes
verifiable when it lands. **Nothing here is executed by this decision record.**

1. **Rename `nodes/` → `surfaces/comfyui/`, move `web/` → `surfaces/comfyui/web/`.**
   Touches: directory moves; `__init__.py` (three surface imports, both
   `__package__` branches; `WEB_DIRECTORY` from `"./web"` to the new relative
   path); `nodes/__init__.py` docstring/loader-context notes carry over.
   *Verifiable when it lands:* `tests/test_comfyui_loader_context.py` and
   `tests/test_dual_context_import.py` still pass (path strings updated); ComfyUI
   still discovers the pack and mounts the web extension.

2. **Add `surfaces/mcp/` — the base surface over `load_model` + `run_diffusion`.**
   Touches: new `surfaces/mcp/` module wrapping the two core entries
   (`dgemma/model.py:157`, `dgemma/loop.py:465`). Depends on Phase 1 only for the
   `surfaces/` parent to exist. *Verifiable when it lands:* the MCP surface
   invokes the core with zero ComfyUI import (a subprocess seam test analogous to
   `test_seam.py`, asserting `import surfaces.mcp` pulls in no `comfy`/`nodes`).

   ### Phase 2 guidance — transcribe `semantic-kinematics-mcp`, with two corrections

   The ecosystem's MCP-surface exemplar is `semantic-kinematics-mcp`
   (`/srv/dev/shanevcantwell/semantic-kinematics-mcp`); `surfaces/mcp/` should be
   a transcription of its layout, not a fresh design. Its pattern
   (under `semantic_kinematics/mcp/`):

   - `mcp/commands/*.py` — the **stateless verb layer**: each module exports a
     `get_tools()` returning tool schemas plus an async
     `handler(state_manager, args)` (e.g. `commands/model.py:15,100`,
     `commands/embeddings.py`).
   - `mcp/server.py` — `Server("name")` (`server.py:29`) with `@server.list_tools()`
     aggregating each module's `get_tools()` (`server.py:33-41`) and
     `@server.call_tool()` dispatching to the handlers and wrapping results in
     `TextContent(json.dumps(...))` (`server.py:45-93`).
   - `mcp/state_manager.py` — ephemeral lifecycle held **outside** the core.
   - `pyproject.toml [project.scripts]` entry point
     (`semantic-kinematics-mcp = "...mcp.server:main"`).

   Over CDG's existing `run_diffusion` (`dgemma/loop.py:465`) + `load_model`
   (`dgemma/model.py:157`) this is mostly transcription. Adopt it with two
   **deliberate corrections** — take `sk-mcp`'s roadmap *target*, not its current
   debt:

   - **Correction 1 — split persist-load from per-call-run state (`STATELESS-CORE`).**
     `sk-mcp`'s own ADR-003
     (`.../docs/ADRs/proposed/ADR-003-stateless-mcp-contract.md:17-25`) records that
     its `StateManager` **violates** statelessness by retaining a live `_adapter`
     and a cross-call `_embedding_cache`
     (`semantic_kinematics/mcp/state_manager.py:51-52,83-86`), with the fix
     (per-call construction, retain nothing) still on its roadmap (`ADR-003:71`).
     That cross-call-mutable-state class is the same one behind CDG's observed
     25-vs-29 heatmap frame-count mismatch (a cached scheduler carrying a prior
     run's dims forward). CDG's `surfaces/mcp/` state manager **must** persist
     **only the immutable model load** — the ~53 GB weights cannot reload per call
     (`README.md` local-run defaults; `dgemma/model.py:157`) — and construct a
     **fresh scheduler / canvas / run state per call**, carrying no prior run's
     mutable shape forward. This is the `STATELESS-CORE` invariant already cited in
     this ADR applied to the surface's lifecycle object: the *load* is persisted,
     the *run* is stateless.

   - **Correction 2 — keep the automated boundary test, do not regress to
     review-only (`ONE-DOOR`).** `sk-mcp` enforces its core↔surface boundary by
     review plus incidental tests only — no import-boundary check (its
     `docs/ARCHITECTURE.md:183-184` documents two live doors into the core). CDG
     already has `tests/test_seam.py:36-63` (subprocess import-leak assertion);
     the Phase-2 verifiable above extends it to `surfaces.mcp`, and Phase 4
     extends it to the analysis boundary. Keep the mechanized check — do not adopt
     `sk-mcp`'s review-only posture on this boundary.

   Also carry across `sk-mcp`'s **Rule #14** (no silent model default): its
   `embed_text` schema requires `model` explicitly with no default
   (`semantic_kinematics/mcp/commands/embeddings.py:39,42,82-83`), gate-tested with
   an exploding fake manager that makes a silent-default call loud
   (`tests/test_embeddings_command.py:19,24,54`). This is the same
   `EMIT-CANONICAL` / no-trust-and-degrade spine as ADR-CDG-001. CDG's MCP tools
   **must** require model + quant explicitly (no baked default) and test the gate
   the same way — a fake `load_model` that raises if called without an explicit
   model/quant, asserting the schema and handler fail loudly rather than loading a
   default.

3. **Relocate analysis out of `dgemma/`'s import graph.** Touches: move
   `dgemma/sampling.py`'s functions to the consumer module chosen in Open
   Question #1; drop the analysis re-exports from `dgemma/__init__.py:26-31,49-51`;
   repoint `surfaces/comfyui/trace.py` (was `nodes/trace.py`), `tests/test_sampling.py`,
   `tests/test_trace_node.py` (monkeypatch targets), and
   `tests/test_dual_context_import.py` (`__module__` assertion) to the new path.
   Depends on Phase 1 (surface home exists). *Verifiable when it lands:* the base
   contract no longer exports analysis.

4. **Add the boundary test: base contract imports no analysis.** Touches: extend
   `tests/test_seam.py` (or a sibling) with a subprocess assertion that after
   `import dgemma`, no analysis module is in `sys.modules`. Depends on Phase 3
   (the relocation must have happened, or the test fails by design). *Verifiable
   when it lands:* the previously prose-only "analysis out of scope" boundary now
   has an enforcement surface — the row in the table above flips from prose-only
   to in-force.

5. **Rewrite `ARCHITECTURE.md` against the governance template.** Touches: the
   existing `ARCHITECTURE.md` (top-level) currently encodes the OLD topology —
   its §3 (lines 53-60) describes `nodes/` + trace-analysis-in-`dgemma/` as
   intended. Rewrite it against
   `../harness-tools/docs/ADRs/constitution/ARCHITECTURE-GOVERNANCE-TEMPLATE.md`,
   which requires a *Current conformance* section grounded with `path:symbol`
   citations or explicit `NOT-YET-IMPLEMENTED` tokens. Depends on Phases 1–4, so
   its conformance table cites the landed, enforced state rather than an
   aspiration. *Verifiable when it lands:* every strong-register claim in
   ARCHITECTURE.md is licensed by a cited enforcement surface (template gates 1–2).

## Supersession Relationships

**Supersedes:** none (extends ADR-CDG-003's core/adapter seam and ADR-CDG-004's
drive contract; does not replace either).
**Superseded by:** TBD.

## References

- ADR-CDG-003 (`decisions/adr-cdg-003-node-engine-seam.md`) — the core/adapter
  seam this generalizes.
- ADR-CDG-004 (`decisions/adr-cdg-004-diffusers-pipeline-drive-seam.md`) —
  `run_diffusion` single-entry drive contract.
- `tests/test_seam.py` — the in-force core-imports-no-surface enforcement.
- `../llauncher` (core/ + operations/ + peer surfaces) — the ecosystem topology
  this aligns to.
- `../harness-tools/docs/ADRs/constitution/ARCHITECTURE-GOVERNANCE-TEMPLATE.md`
  — the template the forthcoming ARCHITECTURE.md rewrite (Phase 5) is filled
  against.
- GROUND_PHYSICS invariants: `ONE-DOOR`, `STATELESS-CORE`, `IDENTITY⊥ENVELOPE`,
  `EMIT-CANONICAL / PARSE-AT-THE-DOOR`
  (`../operating-doctrine/ground-physics/GROUND_PHYSICS.md`).
