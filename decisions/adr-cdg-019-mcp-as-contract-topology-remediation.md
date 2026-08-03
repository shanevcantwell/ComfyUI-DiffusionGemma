# ADR-CDG-019 — MCP as contract: topology remediation (primitives layer, directory morph)

**Status**: proposed
**Date**: 2026-07-23 (re-grounded 2026-08-03 per design-gate FAIL)
**Related**:
- ADR-CDG-008 (MCP-center topology — this **corrects** the "peer surfaces" framing that ADR-CDG-008 encoded structurally)
- ADR-CDG-018 / [Issue #129](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/129) (decompose `dgemma/loop.py`) — **sequencing dependency, gate-ruled 2026-08-03: #129 lands first.** ADR-CDG-018 Stage 1 relocates `DEFAULT_*` / `KNOB_DOCS` out of `loop.py` into `dgemma/config.py`, and Stage 4 relocates `decode_frames` into `dgemma/excision.py`. This ADR's Phase 0 primitives extraction and Phase 2 import redirection therefore import from the **post-decomposition** layout (`dgemma.config`, `dgemma.excision`), not from `dgemma.loop`. Building CDG-019 against the pre-decomposition layout would collide with #129 mid-flight. See "Sequencing" below.
- [Issue #137](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/137) (ARCHITECTURE.md bifurcation fix — this is the code-level remediation for the gap it named)
- [Issue #138](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/138) (this ADR's tracking issue — topology remediation)
- [Issue #57](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/57) (consolidate the dual-context `__package__` gates) — **folded into this ADR as a named checklist phase (Phase 4b), gate-ruled 2026-08-03.** This ADR already rewrites every gate's threshold (`>= 2` → `>= 3` across the comfyui tier); touching each gate anyway is the cheapest moment to consolidate them behind one helper. #57's issue title says "four" gates — that count is stale; the actual census is below (B3).

---

## Context

ADR-CDG-008 chose `surfaces/comfyui/` beside `surfaces/mcp/` to signal "ComfyUI isn't privileged." That directory structure reads as **peer surfaces** — two siblings reaching the same core. The design intent was MCP-center (MCP wraps core, ComfyUI consumes MCP), but the physical artifact invited the opposite: both directories import `dgemma/` directly, and there is no enforcement surface preventing a new consumer from doing the same.

Commit cd93c16 fixed ARCHITECTURE.md's prose to name MCP as canonical and flag the direct-import as GAP (#137). But prose-only discipline doesn't hold under agentic coding — an agent building a CLI or REST consumer will reach `dgemma/` directly because it's the shortest path.

**The structural problem**: there is no callable contract layer between core and consumers. `surfaces/mcp/commands/*.py` holds JSON-RPC schemas AND logic, making it hard for non-MCP consumers to import just the logic without pulling in the MCP SDK dependency. The directory structure (`surfaces/comfyui/` beside `surfaces/mcp/`) reinforces peer access visually.

**Reference model**: prompt-prix's `mcp/tools/*.py` — stateless Python functions callable directly, wrapped by both CLI and MCP server. No "surface tier" concept; just primitives + consumers. (Note: prompt-prix names its package `mcp/` because it has no installed `mcp` SDK dependency in that tree; this pack does — see the SHADOWS ruling and the package-name decision below.)

## Decision

Extract a **primitives layer** (`dgemma_mcp/primitives.py`) that is the callable contract. Move `surfaces/mcp/` to top-level `dgemma_mcp/`. Move `surfaces/comfyui/` into `consumers/comfyui/`. Delete `surfaces/`. Add an enforcement test asserting consumers don't import core directly.

### Package name: `dgemma_mcp/`, NOT `mcp/` (gate-ruled 2026-08-03 — VERDICT: SHADOWS)

The original draft placed the primitives package at top-level `mcp/`. The design gate reproduced a **fatal package-shadow** in the repo's actual venv (`/srv/dev/shanevcantwell/ComfyUI-DiffusionGemma/.venv`, MCP SDK **v1.28.1** confirmed installed at `.venv/lib/python3.12/site-packages/mcp/`), using the exact symbols `server.py` imports and the repo's real pytest invocation (`--import-mode=importlib`, `python -m pytest` from pack root, per `pyproject.toml`'s `[tool.pytest.ini_options]`):

A top-level `mcp/` package at the pack root **fully shadows the installed MCP SDK** in every context that matters (pytest, `python -m` process entry point, and plain script invocation with cwd on `sys.path`).

**Evidence — variant A (regular package, `mcp/__init__.py` present, as the draft specified):**
- `import mcp` → succeeds, but resolves to the **local** package (`mcp.__file__` points at the pack's own `mcp/__init__.py`, not site-packages).
- `from mcp.server import Server` → `ModuleNotFoundError: No module named 'mcp.server'`.
- `from mcp.server.stdio import stdio_server` → `ModuleNotFoundError: No module named 'mcp.server'`.
- `from mcp.types import TextContent, Tool` → `ModuleNotFoundError: No module named 'mcp.types'`.

This is fatal because `surfaces/mcp/server.py` imports exactly those symbols at module scope (`from mcp.server import Server`, `from mcp.server.stdio import stdio_server`, `from mcp.types import TextContent, Tool`). A top-level `mcp/` package would make the pack's own server unable to import the SDK it wraps.

**Resolution — the package is named `dgemma_mcp/`.** One paragraph of justification, as the gate required: the candidates were `mcp_surface/` and `dgemma_mcp/`. `dgemma_mcp/` wins because it is namespaced to the pack's existing identity vocabulary (`dgemma`, `DGemma*`, `DGEMMA_*` sockets), it reads as "DiffusionGemma's MCP layer" rather than "a surface" (the whole point of ADR-CDG-019 is to retire the "surface" framing, so `mcp_surface/` reintroduces the word we are deleting), and it is unambiguously not the SDK — no reader or importer confuses `dgemma_mcp` with the pip-installed `mcp`. Every checklist import below, every blast-radius row, and every test path uses `dgemma_mcp`.

**Target topology:**

```
┌───────────────────────────────────────────┐
│              ORCHESTRATION                 │
│  ComfyUI graph . MCP clients . scripts    │
│  • Calls dgemma_mcp.primitives.* or MCP   │
│  • NEVER imports dgemma/ directly         │
└───────────────────────┬───────────────────┘
                        │ dgemma_mcp.primitives.* call
   consumers/ (analysis: parses CanvasTrace)
                        v
┌───────────────────────────────────────────┐
│      MCP PRIMITIVES (dgemma_mcp/)          │
│  primitives.py — callable Python functions │
│  server.py — JSON-RPC wrapper (uses SDK)   │
│  state_manager.py — model persistence      │
│  • Stateless per-call, model load persists │
│  • Callable as Python OR over JSON-RPC     │
└───────────────────────┬───────────────────┘
                        │ load_model + run_diffusion
┌───────────────────────────────────────────┐
│              CORE (dgemma/)                │
│  model.py | loop.py | config.py | ...     │
│  • Surface-agnostic, zero ComfyUI present │
└───────────────────────┬───────────────────┘
                        │
                   torch . transformers
```

**Import rules:**

| Layer | MAY Import | MUST NOT Import |
|-------|------------|-----------------|
| **Orchestration** (`consumers/comfyui/`) | `dgemma_mcp.primitives`, `consumers.*` | `dgemma/*` directly |
| **MCP Primitives** (`dgemma_mcp/`) | `dgemma.model`, `dgemma.loop`, `dgemma.config`, `dgemma.excision`, `dgemma.types`, `dgemma.payloads`, and the installed `mcp` SDK | `comfy.*`, `consumers.comfyui.*` |
| **Core** (`dgemma/`) | `torch`, `transformers`, `diffusers` | `dgemma_mcp.*`, `consumers.*`, `comfy.*` |

---

## Sequencing (gate-ruled 2026-08-03)

**#129 (ADR-CDG-018) lands first. ADR-CDG-019 builds on the post-decomposition layout.**

ADR-CDG-018 decomposes `dgemma/loop.py` into `compat.py`, `config.py`, `capture.py`, `excision.py`, and a slimmed `loop.py`. Two of its stages move symbols this ADR depends on:

| Symbol | Pre-#129 home | Post-#129 home (#129 Stage) | Used by CDG-019 in |
|--------|---------------|-----------------------------|--------------------|
| `DEFAULT_NUM_INFERENCE_STEPS`, `DEFAULT_T_MIN`, `DEFAULT_T_MAX`, `DEFAULT_ENTROPY_BOUND`, `DEFAULT_GEN_LENGTH`, `DEFAULT_CONFIDENCE` | `dgemma/loop.py` | `dgemma/config.py` (Stage 1) | Phase 0 primitives, Phase 2 loader/sampler redirection |
| `KNOB_DOCS` | `dgemma/loop.py` | `dgemma/config.py` (Stage 1) | Phase 0 primitives, Phase 2 sampler redirection |
| `decode_frames` | `dgemma/loop.py` | `dgemma/excision.py` (Stage 4) | Phase 0 primitives; `emission.py` redirection (Phase 2) |
| `run_diffusion` | `dgemma/loop.py` | `dgemma/loop.py` (unchanged — the drive seam is preserved) | Phase 0 primitives |
| `load_model` | `dgemma/model.py` | `dgemma/model.py` (unchanged) | Phase 0 primitives |

If CDG-019 were built against `dgemma.loop` for these symbols, it would collide with #129 mid-flight (both editing the same import sites, in opposite directions). Building against the post-decomposition layout means CDG-019's Phase 0 and Phase 2 import `from dgemma.config import DEFAULT_*, KNOB_DOCS` and `from dgemma.excision import decode_frames` from the start — no rework when the two ADRs meet.

**Enforcement surface for the ordering:** #138 is blocked-by #129 (record the dependency in the issue tracker). The mechanical check is that CDG-019's Phase 0 branch does not merge while `dgemma/config.py` / `dgemma/excision.py` are absent — the primitives module's imports fail loudly if #129 has not landed.

---

## Implementation checklist

### Phase 0 — Primitives extraction (new code, judgment-bearing)

- [ ] Create `dgemma_mcp/primitives.py` extracting callable logic from `surfaces/mcp/commands/*.py`
  - `load_model(repo_id, quant, local_files_only)` — Rule #14 gate + call to `dgemma.model.load_model`
  - `generate(model, prompt, ...)` — unpack constraints/control_signals/capture payloads, call `run_diffusion`, wrap result (same logic as `commands/generate.py::generate()`)
  - `model_status(manager)` — read-only status query (from `commands/model.py::model_status_tool()`)
  - `cancel_run(run_id)` — set cancel event (from `commands/generate.py::cancel_run()`)
  - `_unpack_constraints()`, `_unpack_control_signals()`, `_unpack_capture()` — payload unpackers (reused by both primitives and server)
  - `_summarize_trace()` — CanvasTrace → JSON-safe summary
  - `_register_run()`, `_unregister_run()` — cancel event registry
- [ ] **Imports target the post-#129 layout** (see Sequencing): `from dgemma.config import DEFAULT_*, KNOB_DOCS`, `from dgemma.excision import decode_frames`, `from dgemma.loop import run_diffusion`, `from dgemma.model import load_model`.
- [ ] `dgemma_mcp/primitives.py` has **NO** dual-context import gate — it's a plain Python module, not loaded by ComfyUI's directory loader. Imports are absolute: `from dgemma.model import load_model as _load_model`.

**Why this works**: prompt-prix's `mcp/tools/*.py` are callable Python functions that both CLI/Gradio AND the MCP server import. The JSON-RPC schema is a *wrapper*, not the logic. Same pattern — `primitives.py` holds the callable contract; `server.py` wraps it with schemas.

### Phase 1a — Move `surfaces/mcp/` → `dgemma_mcp/` (directory morph)

- [ ] `git mv surfaces/mcp/__init__.py dgemma_mcp/__init__.py`
- [ ] `git mv surfaces/mcp/_mcp_sdk_guard.py dgemma_mcp/_mcp_sdk_guard.py`
- [ ] `git mv surfaces/mcp/server.py dgemma_mcp/server.py`
- [ ] `git mv surfaces/mcp/state_manager.py dgemma_mcp/state_manager.py`
- [ ] Remove `surfaces/mcp/commands/` (logic absorbed into `dgemma_mcp/primitives.py`)
  - [ ] Delete `surfaces/mcp/commands/__init__.py`
  - [ ] Delete `surfaces/mcp/commands/generate.py`
  - [ ] Delete `surfaces/mcp/commands/model.py`

**Update imports within moved files:**

- [ ] `dgemma_mcp/server.py`: replace `from surfaces.mcp.commands import generate, model` with `from dgemma_mcp.primitives import ...`; update `from surfaces.mcp._mcp_sdk_guard import require_mcp_sdk` → `from dgemma_mcp._mcp_sdk_guard import require_mcp_sdk`; update `from surfaces.mcp.state_manager import StateManager` → `from dgemma_mcp.state_manager import StateManager`. **The `from mcp.server import Server` / `from mcp.server.stdio import stdio_server` / `from mcp.types import TextContent, Tool` SDK imports are UNCHANGED** — the package rename to `dgemma_mcp/` is precisely what keeps those SDK imports resolvable (the SHADOWS fix).
- [ ] `dgemma_mcp/state_manager.py`: update the dual-context gate. Under `surfaces/mcp/` the relative climb was 2 dots (`__package__.count(".") >= 2`); under top-level `dgemma_mcp/` it is 1 dot:
  ```python
  if __package__ and __package__.count(".") >= 1:
      from ..dgemma.model import load_model
      from ..dgemma.types import DGemmaModel
  else:
      from dgemma.model import load_model
      from dgemma.types import DGemmaModel
  ```
  (Gate threshold change recorded in the Phase-4b census below.)
- [ ] `dgemma_mcp/_mcp_sdk_guard.py`: update docstring references from `surfaces/mcp/` to `dgemma_mcp/`
- [ ] `dgemma_mcp/__init__.py`: rewrite for new topology (no longer "base surface" — it's the contract layer); update all `surfaces.mcp` / `surfaces/mcp/` prose references to `dgemma_mcp`

### Phase 1b — Move `surfaces/comfyui/` → `consumers/comfyui/` (directory morph)

**Full git-mv list, re-grounded against the actual tree 2026-08-03** (`ls surfaces/comfyui/`). The original draft omitted `emission.py` and the `live_view.py`/`web/live_view.js` pair, all landed after ADR authorship (`emission.py` + `live_view.py` at commit `0613df7`, "single-mint the live-view capability", PR #206):

- [ ] `git mv surfaces/comfyui/__init__.py consumers/comfyui/__init__.py`
- [ ] `git mv surfaces/comfyui/denoise.py consumers/comfyui/denoise.py`
- [ ] `git mv surfaces/comfyui/emission.py consumers/comfyui/emission.py`  ← **added (B1)**
- [ ] `git mv surfaces/comfyui/encode.py consumers/comfyui/encode.py`
- [ ] `git mv surfaces/comfyui/frames_image.py consumers/comfyui/frames_image.py`
- [ ] `git mv surfaces/comfyui/live_view.py consumers/comfyui/live_view.py`  ← **added (I2 — moves with its .js pair below)**
- [ ] `git mv surfaces/comfyui/loader.py consumers/comfyui/loader.py`
- [ ] `git mv surfaces/comfyui/run_log_writer.py consumers/comfyui/run_log_writer.py`
- [ ] `git mv surfaces/comfyui/sampler.py consumers/comfyui/sampler.py`
- [ ] `git mv surfaces/comfyui/socket_types.py consumers/comfyui/socket_types.py`
- [ ] `git mv surfaces/comfyui/tally_audit.py consumers/comfyui/tally_audit.py`
- [ ] `git mv surfaces/comfyui/token_trace.py consumers/comfyui/token_trace.py`
- [ ] `git mv surfaces/comfyui/trace.py consumers/comfyui/trace.py`
- [ ] `git mv surfaces/comfyui/web/live_view.js consumers/comfyui/web/live_view.js`  ← **the JS half of the live-view pair (I2); move together with `live_view.py` above**

**Update imports within moved files:**

Each ComfyUI node file with a dual-context gate has its threshold raised `>= 2` → `>= 3` (one extra directory level: `consumers/comfyui/` is one deeper than `surfaces/comfyui/`). The full per-file census is in **Phase 4b** below (the gate consolidation folds #57 in here). Additionally, cross-package `dgemma.*` imports change per the redirection in Phase 2, and intra-package prose/import references to `surfaces.comfyui.*` become `.` (relative) or `consumers.comfyui.*`:

- [ ] `consumers/comfyui/denoise.py`: update `from surfaces.comfyui.emission import build_sampler_shaped_outputs` → `from .emission import ...`; `from surfaces.comfyui.live_view import LIVE_VIEW_HIDDEN_INPUT, build_on_frame` → `from .live_view import ...` (its `.emission` / `.live_view` relative branch is already 1-dot and unaffected; only the standalone-context absolute branch changes)
- [ ] `consumers/comfyui/sampler.py`: same — `surfaces.comfyui.emission` → `.emission`, `surfaces.comfyui.live_view` → `.live_view` on the absolute-context branch
- [ ] `consumers/comfyui/encode.py`, `run_log_writer.py`, `token_trace.py`, `trace.py`, `tally_audit.py`: update any `from surfaces.comfyui.<sibling>` absolute-context imports to `consumers.comfyui.<sibling>` (or relative `.<sibling>`)
- [ ] `consumers/comfyui/live_view.py`: no `dgemma.*` import and **no** dual-context gate (it lazily imports `from server import PromptServer` inside a closure) — only its docstring/prose references to `surfaces/comfyui/` update. It is NOT in the gate census.
- [ ] `consumers/comfyui/socket_types.py`: no gate, no `dgemma.*` import — prose references only (its `live_view.py` citations stay valid, same directory)

**Note**: intra-consumer relative imports (e.g., `.socket_types`, `.frames_image`, `.emission`, `.live_view`) use a 1-dot relative form and are unaffected by the directory move. Only cross-package imports and the standalone-context absolute branch of each gate change.

### Phase 1c — Clean up `surfaces/`

- [ ] Delete `surfaces/__init__.py`
- [ ] Remove empty `surfaces/` directory

### Phase 2 — Import redirection (ComfyUI → primitives)

Redirect the ComfyUI consumers' `dgemma.*` imports to `dgemma_mcp.primitives` where the primitive is the contract path. The gate threshold in these snippets is already at the Phase-4b value (`>= 3`).

**`consumers/comfyui/loader.py`** — currently imports `from ...dgemma.model import (...)` / `from dgemma.model import (...)`. Redirect to primitives:
```python
if __package__ and __package__.count(".") >= 3:
    from ...dgemma_mcp.primitives import load_model, DEFAULT_QUANT, DEFAULT_REPO_ID, _QUANT_CHOICES
else:
    from dgemma_mcp.primitives import load_model, DEFAULT_QUANT, DEFAULT_REPO_ID, _QUANT_CHOICES
```
(Note: `consumers/comfyui/` is 3 segments from pack root under the real loader; the relative climb to a top-level sibling `dgemma_mcp` is 3 dots — `...dgemma_mcp` — matching the gate threshold.)

- [ ] `consumers/comfyui/loader.py`: swap `dgemma.model` → `dgemma_mcp.primitives` for `load_model`, `_QUANT_CHOICES`, `DEFAULT_QUANT`, `DEFAULT_REPO_ID`
- [ ] `consumers/comfyui/sampler.py`: swap `dgemma.loop` → `dgemma_mcp.primitives` for `run_diffusion`, and `dgemma.config` → `dgemma_mcp.primitives` for `KNOB_DOCS` / `DEFAULT_*` (primitives re-exports these from the post-#129 `dgemma.config`); `decode_frames` via primitives (re-exported from `dgemma.excision`)
- [ ] `consumers/comfyui/emission.py`: it currently imports `from ...dgemma.loop import decode_frames` — post-#129 that symbol lives in `dgemma.excision`, and this ADR routes it through `dgemma_mcp.primitives` (`from ...dgemma_mcp.primitives import decode_frames`). Also imports `from ...consumers.run_log import RunConfig` — unchanged (already consumer-side).
- [ ] `consumers/comfyui/encode.py`: imports `from ...dgemma.kv_cache import encode_sequence`. `encode_sequence` is core-boundary kernel wiring, not a run/load primitive. **Decision (recorded):** `encode_sequence` is NOT surfaced through `primitives.py` — it is a low-level KV-cache utility with no MCP tool wrapping it. `encode.py` keeps its `dgemma.kv_cache` import (this is the one sanctioned core import remaining consumer-side, and the Phase-5 enforcement test must whitelist it — see Phase 5). If a future ADR promotes it to a primitive, redirect then; inventing a primitive with no consumer beyond one node is premature.
- [ ] `consumers/comfyui/denoise.py`: imports `from ...dgemma.loop import (...)` — redirect the run-path symbols through primitives the same way as `sampler.py`.

**`dgemma_mcp/server.py`:** replace commands dispatch with primitives:
- [ ] Replace `from surfaces.mcp.commands import generate, model` with imports from `dgemma_mcp.primitives`
- [ ] JSON-RPC schema (`Tool` definitions) stays in `server.py` (OQ1 resolution — inline until the CONSERVE-SALIENCE ceiling; see Recorded resolutions). The callable logic is in `primitives.py`.

### Phase 3 — Root entry point update

- [ ] `__init__.py`: update the 8 node imports from `.surfaces.comfyui.*` / `surfaces.comfyui.*` (both relative and absolute branches, lines 54–70) to `.consumers.comfyui.*` / `consumers.comfyui.*`
- [ ] `__init__.py`: update `WEB_DIRECTORY` from `"./surfaces/comfyui/web"` to `"./consumers/comfyui/web"` (line 96)
- [ ] `__init__.py`: update the two prose references to `surfaces/comfyui/...` (lines 11, 15, 36, 93) to `consumers/comfyui/...`

### Phase 4 — Test updates

**Tests importing `surfaces.mcp.*` (→ `dgemma_mcp.*`):**

| File | Change |
|------|--------|
| `tests/test_mcp_dual_context_import.py` | Update all `synthetic_pack_root}.surfaces.mcp.*` → `dgemma_mcp.*`; adjust dot-count assertions (state_manager gate was 2 dots under `surfaces/mcp`, now 1 under `dgemma_mcp`) |
| `tests/test_mcp_generate_command.py` | `from surfaces.mcp.commands import generate` → `from dgemma_mcp.primitives import ...`; `from surfaces.mcp.state_manager` → `from dgemma_mcp.state_manager` |
| `tests/test_mcp_model_command.py` | Same — `commands.model` → `primitives`; state_manager path update |
| `tests/test_mcp_import_guard.py` | Update all subprocess code strings from `surfaces.mcp.*` to `dgemma_mcp.*`; update ComfyUI import references from `surfaces.comfyui.*` to `consumers.comfyui.*` |
| `tests/test_mcp_sdk_guard.py` | `from surfaces.mcp._mcp_sdk_guard` → `from dgemma_mcp._mcp_sdk_guard` |
| `tests/test_mcp_server_dispatch.py` | `from surfaces.mcp import server` → `from dgemma_mcp import server`; state_manager path update |
| `tests/test_mcp_statelessness.py` | All `surfaces.mcp.*` → `dgemma_mcp.*` imports; monkeypatch paths updated |
| `tests/test_mcp_surface_seam.py` | Update all subprocess code strings from `surfaces.mcp` to `dgemma_mcp`; update docstring references. **This test is a named enforcement surface for the SHADOWS fix**: it must assert `dgemma_mcp.server` imports the real installed `mcp` SDK (i.e. `mcp.server`/`mcp.types` resolve to site-packages, not shadowed). |
| `tests/test_units_glossary_mint.py` | `from surfaces.mcp.commands import generate` → `from dgemma_mcp.primitives import ...`; `from surfaces.comfyui.sampler` → `from consumers.comfyui.sampler` |

**Tests importing `surfaces.comfyui.*` (→ `consumers.comfyui.*`):**

| File | Change |
|------|--------|
| `tests/test_comfyui_loader_context.py` | Module path references `surfaces/comfyui/` → `consumers/comfyui/`; WEB_DIRECTORY assertion updated. **Named gate-enforcement surface** — see Phase 4b. |
| `tests/test_dual_context_import.py` | All `synthetic_pack_root}.surfaces.comfyui.*` → `consumers.comfyui.*`; adjust dot-count assertions (was 2 dots under `surfaces/comfyui`, now 3 under `consumers/comfyui`). **Named gate-enforcement surface** — see Phase 4b. |
| `tests/test_frames_image.py` | `from surfaces.comfyui.frames_image` → `from consumers.comfyui.frames_image` |
| `tests/test_kv_cache_cold_wiring.py` | `surfaces.comfyui.*` → `consumers.comfyui.*` |
| `tests/test_kv_cache_nodes.py` | Same |
| `tests/test_live_seams.py` | `from surfaces.comfyui.sampler` → `from consumers.comfyui.sampler`; also covers `live_view`/`emission` if referenced |
| `tests/test_loader_contract.py` | `surfaces.comfyui.*` → `consumers.comfyui.*` |
| `tests/test_loader_folder_paths.py` | Same |
| `tests/test_run_log_writer.py` | Same |
| `tests/test_socket_mint.py` | All ComfyUI node imports `surfaces.comfyui.*` → `consumers.comfyui.*` |
| `tests/test_tally_audit_node.py` | Same |
| `tests/test_token_trace_node.py` | Same |
| `tests/test_trace_node.py` | All ComfyUI trace imports updated |

> **Audit note (do not treat this table as exhaustive without a fresh grep):** before implementation, `grep -rln "surfaces\.comfyui\|surfaces\.mcp\|surfaces/comfyui\|surfaces/mcp" tests/` and reconcile against this table. `emission.py` and `live_view.py` landed after this ADR was first drafted; any test referencing them must be caught by that grep, not by this hand-maintained list.

### Phase 4b — Consolidate the dual-context gates (folds in #57; gate-ruled 2026-08-03)

This ADR rewrites every dual-context gate's threshold anyway (the directory move changes the pack-relative depth). Touching all of them once is the cheapest moment to also close #57 (consolidate the gates behind one helper), so #57 folds in here as a **named phase**, not a vague aside.

**Gate census — re-grounded against the actual tree 2026-08-03** (`grep -rn "^if __package__ and __package__.count" surfaces/`). #57's title says "four" gates; the real count at HEAD is **twelve** real `if`-gates (nine comfyui-tier, three mcp-tier). After this ADR's `commands/*.py` deletion, ten gates remain to consolidate.

**Comfyui tier (9 gates today; all threshold `>= 2` → `>= 3` on the move):**

| File (post-move path) | Today's gate | Post-move gate | Notes |
|------|------|------|-------|
| `consumers/comfyui/loader.py` | `>= 2` | `>= 3` | + import redirect to `dgemma_mcp.primitives` (Phase 2) |
| `consumers/comfyui/sampler.py` | `>= 2` | `>= 3` | + import redirect (Phase 2) |
| `consumers/comfyui/denoise.py` | `>= 2` | `>= 3` | + import redirect (Phase 2) |
| `consumers/comfyui/encode.py` | `>= 2` | `>= 3` | keeps `dgemma.kv_cache` (whitelisted, Phase 5) |
| `consumers/comfyui/emission.py` | `>= 2` | `>= 3` | **new gate from `0613df7`**, missed by the original draft; + `decode_frames` redirect (Phase 2) |
| `consumers/comfyui/run_log_writer.py` | `>= 2` | `>= 3` | consumer-only imports |
| `consumers/comfyui/tally_audit.py` | `>= 2` | `>= 3` | consumer-only imports |
| `consumers/comfyui/token_trace.py` | `>= 2` | `>= 3` | consumer-only imports |
| `consumers/comfyui/trace.py` | `>= 2` | `>= 3` | consumer-only imports |

**MCP tier (3 gates today; two are deleted with `commands/`):**

| File | Today's gate | Fate |
|------|------|------|
| `dgemma_mcp/state_manager.py` | `>= 2` | → `>= 1` (top-level `dgemma_mcp/` is one segment shallower than `surfaces/mcp/`) |
| `surfaces/mcp/commands/model.py` | `>= 3` | **deleted** — logic absorbed into `primitives.py` (no gate; primitives has none) |
| `surfaces/mcp/commands/generate.py` | `>= 3` | **deleted** — same |

**Files with NO gate (do not add one):** `consumers/comfyui/live_view.py` (lazy `from server import PromptServer` inside a closure), `consumers/comfyui/socket_types.py`, `consumers/comfyui/frames_image.py` (relative-only imports), `dgemma_mcp/primitives.py` (OQ3 resolution — plain module, no directory-loader ambiguity), `dgemma_mcp/server.py` (single-context, plain `python -m` entry), `dgemma_mcp/__init__.py`, `dgemma_mcp/_mcp_sdk_guard.py`.

**Consolidation:**
- [ ] Extract the repeated `if __package__ and __package__.count(".") >= N: <relative> else: <absolute>` pattern into one helper (a small `_dual_context_import` utility, or a documented single-source gate constant per tier) so the depth threshold lives in ONE place per tier, not copied across nine comfyui files. The `>= 3` value is then asserted once, not nine times.
- [ ] After consolidation, the 10 remaining gates reference the shared helper rather than open-coding the `__package__.count` check.

**Enforcement surface (named):** `tests/test_dual_context_import.py` (comfyui tier) and `tests/test_comfyui_loader_context.py` (loader's real-loader `__package__` shape) enforce that the comfyui gates resolve correctly in both contexts; `tests/test_mcp_dual_context_import.py` enforces the mcp tier. These three tests are what fail if a gate threshold is wrong after the move — they are the mechanical guard for Phase 4b, and the consolidation must keep them green.

### Phase 5 — New enforcement test

- [ ] Create `tests/test_contract_seam.py`: subprocess test asserting `consumers/comfyui` does NOT import `dgemma/` directly — **with one sanctioned exception: `dgemma.kv_cache` (via `encode.py`'s `encode_sequence`)**, which is not surfaced as a primitive (Phase 2 decision). The test whitelists that single module and fails on any *other* leaked `dgemma.*` import.
  ```python
  """ComfyUI consumers must route through dgemma_mcp.primitives, not import dgemma/ directly.
  Sanctioned exception: dgemma.kv_cache (encode_sequence), a low-level KV utility
  with no MCP tool wrapping it (ADR-CDG-019 Phase 2 decision)."""
  # Subprocess: import consumers.comfyui.loader; assert no dgemma.* module leaked
  # (other than dgemma.kv_cache) that wasn't pulled in transitively through dgemma_mcp.primitives.
  ```

### Phase 6 — Documentation updates

- [ ] **ARCHITECTURE.md**: replace layer diagram with four-layer model (see Decision section above); add import rules table; remove GAP (#137) citations from the `surfaces/comfyui/` bullet (line 82) since the gap is now closed by this ADR's implementation.
- [ ] **ARCHITECTURE.md rule 4** (OQ2 resolution): update the mint citation. Rule 4 currently reads *"`DGEMMA_*` socket strings live in one mint module (`surfaces/comfyui/socket_types.py`)"* — repath to `consumers/comfyui/socket_types.py`. The mint stays consumer-side (OQ2 ruling); only the path string in the rule changes.
- [ ] **ADR-CDG-008**: add supersession note at top — "Section 1 framing ('peer surfaces') and directory structure decision (`surfaces/`) superseded by ADR-CDG-019. The core/surface seam rules (rules 1–7) remain in force."
- [ ] **ROADMAP.md**: replace the "peer surface" framing (and the `surfaces/mcp/` path in the Phase-2 done-row, line 111) with "ComfyUI consumes `dgemma_mcp.primitives` (the contract layer)" / `dgemma_mcp/`.
- [ ] **AGENTS.md**: update the architecture diagram (line 19 `surfaces/mcp/` box → `dgemma_mcp/`); update Rule 2 reference and the `test_mcp_surface_seam.py` invocation path context from `surfaces/mcp/` to `dgemma_mcp/primitives.py`.
- [ ] Update all docstring references in source files that mention `surfaces/mcp/` or `surfaces/comfyui/` paths (see blast radius below).

---

## Blast radius — every file touched (re-grounded 2026-08-03)

### Moved (git mv, content updated for new path):
| Source | Destination |
|--------|-------------|
| `surfaces/mcp/__init__.py` | `dgemma_mcp/__init__.py` |
| `surfaces/mcp/_mcp_sdk_guard.py` | `dgemma_mcp/_mcp_sdk_guard.py` |
| `surfaces/mcp/server.py` | `dgemma_mcp/server.py` |
| `surfaces/mcp/state_manager.py` | `dgemma_mcp/state_manager.py` |
| `surfaces/comfyui/__init__.py` | `consumers/comfyui/__init__.py` |
| `surfaces/comfyui/denoise.py` | `consumers/comfyui/denoise.py` |
| `surfaces/comfyui/emission.py` | `consumers/comfyui/emission.py` |
| `surfaces/comfyui/encode.py` | `consumers/comfyui/encode.py` |
| `surfaces/comfyui/frames_image.py` | `consumers/comfyui/frames_image.py` |
| `surfaces/comfyui/live_view.py` | `consumers/comfyui/live_view.py` |
| `surfaces/comfyui/loader.py` | `consumers/comfyui/loader.py` |
| `surfaces/comfyui/run_log_writer.py` | `consumers/comfyui/run_log_writer.py` |
| `surfaces/comfyui/sampler.py` | `consumers/comfyui/sampler.py` |
| `surfaces/comfyui/socket_types.py` | `consumers/comfyui/socket_types.py` |
| `surfaces/comfyui/tally_audit.py` | `consumers/comfyui/tally_audit.py` |
| `surfaces/comfyui/token_trace.py` | `consumers/comfyui/token_trace.py` |
| `surfaces/comfyui/trace.py` | `consumers/comfyui/trace.py` |
| `surfaces/comfyui/web/live_view.js` | `consumers/comfyui/web/live_view.js` |

### Deleted:
| File | Reason |
|------|--------|
| `surfaces/__init__.py` | Empty parent after move |
| `surfaces/mcp/commands/__init__.py` | Logic absorbed into `primitives.py` |
| `surfaces/mcp/commands/generate.py` | Logic extracted to `primitives.py`, schema stays in `server.py` |
| `surfaces/mcp/commands/model.py` | Same |

### New:
| File | Purpose |
|------|---------|
| `dgemma_mcp/primitives.py` | The contract layer — callable Python functions |
| `tests/test_contract_seam.py` | Enforcement test — consumers don't import core directly (except whitelisted `dgemma.kv_cache`) |

### Modified (import paths, gate thresholds, docstrings):
| File | Change type |
|------|-------------|
| `__init__.py` | 8 node import paths + WEB_DIRECTORY + prose refs |
| `consumers/__init__.py` | Docstring reference to `surfaces/__init__.py` removed |
| `tests/test_comfyui_loader_context.py` | Module path references + gate enforcement (Phase 4b) |
| `tests/test_dual_context_import.py` | All synthetic import paths (comfyui) + dot-count assertions (Phase 4b) |
| `tests/test_frames_image.py` | Import paths |
| `tests/test_kv_cache_cold_wiring.py` | Import paths |
| `tests/test_kv_cache_nodes.py` | Import paths |
| `tests/test_live_seams.py` | Import paths |
| `tests/test_loader_contract.py` | Import paths |
| `tests/test_loader_folder_paths.py` | Import paths |
| `tests/test_mcp_dual_context_import.py` | All synthetic import paths + dot-count assertions (Phase 4b, mcp tier) |
| `tests/test_mcp_generate_command.py` | Import paths |
| `tests/test_mcp_import_guard.py` | Subprocess code strings (MCP + ComfyUI) |
| `tests/test_mcp_model_command.py` | Import paths |
| `tests/test_mcp_sdk_guard.py` | Import path |
| `tests/test_mcp_server_dispatch.py` | Import paths |
| `tests/test_mcp_statelessness.py` | Import paths + monkeypatch targets |
| `tests/test_mcp_surface_seam.py` | Subprocess code strings + SHADOWS assertion (real SDK resolves) |
| `tests/test_run_log_writer.py` | Import paths |
| `tests/test_socket_mint.py` | ComfyUI node import paths |
| `tests/test_tally_audit_node.py` | Import path |
| `tests/test_token_trace_node.py` | Import path |
| `tests/test_trace_node.py` | trace import paths |
| `tests/test_units_glossary_mint.py` | Import paths (MCP + ComfyUI) |

> Test-table completeness is grep-verified at implementation time, not asserted from this list (see Phase 4 audit note).

### Documentation:
| File | Change type |
|------|-------------|
| `ARCHITECTURE.md` | Layer diagram, import rules table, GAP citations removed, **rule 4 mint path repathed to `consumers/comfyui/socket_types.py` (OQ2)** |
| `AGENTS.md` | Architecture diagram (`dgemma_mcp/`), Rule 2 reference |
| `ROADMAP.md` | "peer surface" → "consumes `dgemma_mcp.primitives`"; Phase-2 done-row path |
| `decisions/adr-cdg-008-mcp-center-multi-surface-topology.md` | Supersession note at top |

### Untouched:
| File/Dir | Reason |
|----------|--------|
| `dgemma/*` | Core — no changes needed **by this ADR** (its decomposition is #129's scope, which lands first) |
| `consumers/analysis.py` | Already in `consumers/`, no path change |
| `consumers/run_log.py` | Same |
| `consumers/tally_audit.py` | Same |

---

## Rollback plan

Every phase is a discrete git commit. Rollback = revert the last N commits:

- Phase 6 (docs): prose-only, trivially reversible
- Phase 5 (new test): delete one file
- Phase 4b (gate consolidation): revert the helper extraction — gates return to open-coded form
- Phase 4 (test updates): revert import paths — mechanical, no logic change
- Phase 3 (root entry point): revert `__init__.py`
- Phase 2 (import redirection): revert ComfyUI imports back to `dgemma.*`
- Phase 1c (delete surfaces/): recreate from git history
- Phase 1b (move comfyui → consumers): reverse git mv
- Phase 1a (move mcp → `dgemma_mcp/`): reverse git mv
- Phase 0 (primitives extraction): delete `dgemma_mcp/primitives.py`, restore `surfaces/mcp/commands/*.py`

**Recommended commit granularity**: one commit per phase. If Phase 0 (primitives) is correct, everything else is mechanical and can be batched into a single "directory morph + import update" commit if desired — but phased commits are safer for bisect.

---

## Why this is the most efficient path

1. **Primitives extraction first** isolates the only judgment-bearing step (extracting callable logic from `commands/*.py`). Everything after is mechanical: rename, update imports, run tests.
2. **The transitional GAP (#137) is honestly phased, not pretended away.** GAP #137 (consumers import `dgemma/` directly) **persists through Phase 1** — moving the files does not by itself redirect the imports. It **closes at Phase 2**, when the imports are redirected through `dgemma_mcp.primitives`. The **Phase-5 enforcement test forbids regression** — after Phase 5 lands, any consumer that reaches `dgemma/` directly (outside the sanctioned `dgemma.kv_cache` whitelist) fails a subprocess test. There is an intermediate state (Phases 1–1c, files moved but imports not yet redirected); the phasing names it rather than claiming it away, and the enforcement test is what makes the closure durable.
3. **The enforcement test (Phase 5) is the durable signal.** The directory structure reinforces it visually; the test enforces it mechanically.
4. **Modeled on prompt-prix** which has been stable under agentic coding — the primitives layer is the pattern that held there. The one deviation forced by this pack's dependency graph — the package is `dgemma_mcp/`, not `mcp/` — is recorded above (SHADOWS ruling).

---

## Recorded resolutions (formerly open questions; dispositioned by design-gate 2026-08-03)

An open question may not ride in a ratifiable contract. The three questions the draft carried are resolved here as decisions:

1. **Should `dgemma_mcp/commands/` survive as a slim schema-only module?** **Resolved: no — JSON-RPC `Tool` schema definitions stay inline in `server.py`.** The threshold to extract is CONSERVE-SALIENCE: extract to a `dgemma_mcp/commands/` (schema-only) module only if the schema block in `server.py` exceeds **~5,000 tokens / ~200 lines** (the file-comfort ceiling ADR-CDG-018 also uses). Below that ceiling, inline is cleaner than a module split. If ADR-CDG-017's remelt spec (or similar) pushes the schema over the line, extract then — as a follow-up, not in this ADR's scope.

2. **Does `consumers/comfyui/socket_types.py` stay under consumers or move to `dgemma_mcp/`?** **Resolved: it stays consumer-side.** Socket types are ComfyUI's envelope (rule 4, ONE-MINT) — consumer-specific vocabulary. The MCP server uses JSON keys, not `DGEMMA_*` socket strings. Phase 6 updates ARCHITECTURE.md rule 4's mint citation to the new path (`consumers/comfyui/socket_types.py`); the mint's *location semantics* (surface/consumer-side, one module) are unchanged — only the path string moves.

3. **Should the dual-context gate pattern be retired for `dgemma_mcp/primitives.py`?** **Resolved: `primitives.py` carries NO gate** — it is a plain Python module, never loaded by ComfyUI's directory loader, so there is no dual-context ambiguity to gate. The **consumer gates persist** (the nine comfyui-tier gates are still real: those files ARE loaded both as ComfyUI nodes and as pytest modules), and Phase 4b consolidates them behind one helper rather than retiring them. Retiring a consumer gate would reintroduce the exact bifurcation this ADR removes.
