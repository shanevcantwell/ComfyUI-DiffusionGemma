# ADR-CDG-018 — MCP as contract: topology remediation (primitives layer, directory morph)

**Status**: proposed
**Date**: 2026-07-23
**Related**: ADR-CDG-008 (MCP-center topology — this **corrects** the "peer surfaces" framing that ADR-CDG-008 encoded structurally), Issue #137 (ARCHITECTURE.md bifurcation fix — this is the code-level remediation for the gap it named)

---

## Context

ADR-CDG-008 chose `surfaces/comfyui/` beside `surfaces/mcp/` to signal "ComfyUI isn't privileged." That directory structure reads as **peer surfaces** — two siblings reaching the same core. The design intent was MCP-center (MCP wraps core, ComfyUI consumes MCP), but the physical artifact invited the opposite: both directories import `dgemma/` directly, and there is no enforcement surface preventing a new consumer from doing the same.

Commit cd93c16 fixed ARCHITECTURE.md's prose to name MCP as canonical and flag the direct-import as GAP (#137). But prose-only discipline doesn't hold under agentic coding — an agent building a CLI or REST consumer will reach `dgemma/` directly because it's the shortest path.

**The structural problem**: there is no callable contract layer between core and consumers. `surfaces/mcp/commands/*.py` holds JSON-RPC schemas AND logic, making it hard for non-MCP consumers to import just the logic without pulling in the MCP SDK dependency. The directory structure (`surfaces/comfyui/` beside `surfaces/mcp/`) reinforces peer access visually.

**Reference model**: prompt-prix's `mcp/tools/*.py` — stateless Python functions callable directly, wrapped by both CLI and MCP server. No "surface tier" concept; just primitives + consumers.

## Decision

Extract a **primitives layer** (`mcp/primitives.py`) that is the callable contract. Move `surfaces/mcp/` to top-level `mcp/`. Move `surfaces/comfyui/` into `consumers/comfyui/`. Delete `surfaces/`. Add enforcement test asserting consumers don't import core directly.

**Target topology:**

```
┌───────────────────────────────────────────┐
│              ORCHESTRATION                 │
│  ComfyUI graph . MCP clients . scripts    │
│  • Calls mcp.primitives.* or MCP tools    │
│  • NEVER imports dgemma/ directly         │
└───────────────────────┬───────────────────┘
                        │ mcp.primitives.* call
   consumers/ (analysis: parses CanvasTrace)
                        v
┌───────────────────────────────────────────┐
│           MCP PRIMITIVES (mcp/)            │
│  primitives.py — callable Python functions │
│  server.py — JSON-RPC wrapper              │
│  state_manager.py — model persistence      │
│  • Stateless per-call, model load persists │
│  • Callable as Python OR over JSON-RPC     │
└───────────────────────┬───────────────────┘
                        │ load_model + run_diffusion
┌───────────────────────────────────────────┐
│              CORE (dgemma/)                │
│  model.py | loop.py | types.py            │
│  • Surface-agnostic, zero ComfyUI present │
└───────────────────────┬───────────────────┘
                        │
                   torch . transformers
```

**Import rules:**

| Layer | MAY Import | MUST NOT Import |
|-------|------------|-----------------|
| **Orchestration** (`consumers/comfyui/`) | `mcp.primitives`, `consumers.*` | `dgemma/*` directly |
| **MCP Primitives** (`mcp/`) | `dgemma.model`, `dgemma.loop`, `dgemma.types`, `dgemma.payloads` | `comfy.*`, `consumers.comfyui.*` |
| **Core** (`dgemma/`) | `torch`, `transformers`, `diffusers` | `mcp.*`, `consumers.*`, `comfy.*` |

---

## Implementation checklist

### Phase 0 — Primitives extraction (new code, judgment-bearing)

- [ ] Create `mcp/primitives.py` extracting callable logic from `surfaces/mcp/commands/*.py`
  - `load_model(repo_id, quant, local_files_only)` — Rule #14 gate + call to `dgemma.model.load_model`
  - `generate(model, prompt, ...)` — unpack constraints/control_signals/capture payloads, call `run_diffusion`, wrap result (same logic as `commands/generate.py::generate()`)
  - `model_status(manager)` — read-only status query (from `commands/model.py::model_status_tool()`)
  - `cancel_run(run_id)` — set cancel event (from `commands/generate.py::cancel_run()`)
  - `_unpack_constraints()`, `_unpack_control_signals()`, `_unpack_capture()` — payload unpackers (reused by both primitives and server)
  - `_summarize_trace()` — CanvasTrace → JSON-safe summary
  - `_register_run()`, `_unregister_run()` — cancel event registry
- [ ] `mcp/primitives.py` has NO dual-context import gate — it's a plain Python module, not loaded by ComfyUI's directory loader. Imports are absolute: `from dgemma.model import load_model as _load_model`.

**Why this works**: prompt-prix's `mcp/tools/*.py` are callable Python functions that both CLI/Gradio AND the MCP server import. The JSON-RPC schema is a *wrapper*, not the logic. Same pattern — primitives.py holds the callable contract; server.py wraps it with schemas.

### Phase 1a — Move `surfaces/mcp/` → `mcp/` (directory morph)

- [ ] `git mv surfaces/mcp/__init__.py mcp/__init__.py`
- [ ] `git mv surfaces/mcp/_mcp_sdk_guard.py mcp/_mcp_sdk_guard.py`
- [ ] `git mv surfaces/mcp/server.py mcp/server.py`
- [ ] `git mv surfaces/mcp/state_manager.py mcp/state_manager.py`
- [ ] Remove `surfaces/mcp/commands/` (logic absorbed into `mcp/primitives.py`)
  - [ ] Delete `surfaces/mcp/commands/__init__.py`
  - [ ] Delete `surfaces/mcp/commands/generate.py`
  - [ ] Delete `surfaces/mcp/commands/model.py`

**Update imports within moved files:**

- [ ] `mcp/server.py`: replace `from surfaces.mcp.commands import generate, model` with `from mcp.primitives import ...`; update all internal references from `surfaces.mcp.*` to `mcp.*`
- [ ] `mcp/state_manager.py`: update dual-context imports — relative climb is now 2 dots (was 3 under `surfaces/mcp/`):
  ```python
  if __package__ and __package__.count(".") >= 1:
      from ...dgemma.model import load_model
      from ...dgemma.types import DGemmaModel
  else:
      from dgemma.model import load_model
      from dgemma.types import DGemmaModel
  ```
- [ ] `mcp/_mcp_sdk_guard.py`: update docstring references from `surfaces/mcp/` to `mcp/`
- [ ] `mcp/__init__.py`: rewrite for new topology (no longer "base surface" — it's the contract layer)

### Phase 1b — Move `surfaces/comfyui/` → `consumers/comfyui/` (directory morph)

- [ ] `git mv surfaces/comfyui/__init__.py consumers/comfyui/__init__.py`
- [ ] `git mv surfaces/comfyui/denoise.py consumers/comfyui/denoise.py`
- [ ] `git mv surfaces/comfyui/encode.py consumers/comfyui/encode.py`
- [ ] `git mv surfaces/comfyui/frames_image.py consumers/comfyui/frames_image.py`
- [ ] `git mv surfaces/comfyui/loader.py consumers/comfyui/loader.py`
- [ ] `git mv surfaces/comfyui/run_log_writer.py consumers/comfyui/run_log_writer.py`
- [ ] `git mv surfaces/comfyui/sampler.py consumers/comfyui/sampler.py`
- [ ] `git mv surfaces/comfyui/socket_types.py consumers/comfyui/socket_types.py`
- [ ] `git mv surfaces/comfyui/tally_audit.py consumers/comfyui/tally_audit.py`
- [ ] `git mv surfaces/comfyui/token_trace.py consumers/comfyui/token_trace.py`
- [ ] `git mv surfaces/comfyui/trace.py consumers/comfyui/trace.py`
- [ ] `git mv -f surfaces/comfyui/web/live_view.js consumers/comfyui/web/live_view.js`

**Update imports within moved files:**

Each ComfyUI node file has a dual-context import gate. The relative climb changes from 2 dots (under `surfaces/comfyui/`) to 3 dots (under `consumers/comfyui/`). Update each:

- [ ] `consumers/comfyui/loader.py`: change `__package__.count(".") >= 2` → `>= 3`; update relative imports from `...dgemma.model` to `....mcp.primitives` (Phase 2) or keep as `...dgemma.model` until Phase 2
- [ ] `consumers/comfyui/sampler.py`: same — gate threshold + import path adjustment
- [ ] `consumers/comfyui/denoise.py`: update internal imports (`from surfaces.comfyui.socket_types` → `from .socket_types`)
- [ ] `consumers/comfyui/encode.py`: same
- [ ] `consumers/comfyui/run_log_writer.py`: same
- [ ] `consumers/comfyui/token_trace.py`: same
- [ ] `consumers/comfyui/trace.py`: same

**Note**: intra-consumer imports (e.g., `.socket_types`, `.frames_image`) use relative 1-dot and are unaffected by the directory move. Only cross-package imports change.

### Phase 1c — Clean up `surfaces/`

- [ ] Delete `surfaces/__init__.py`
- [ ] Remove empty `surfaces/` directory

### Phase 2 — Import redirection (ComfyUI → primitives)

**`consumers/comfyui/loader.py`:** replace dual-context `dgemma.model` imports with:
```python
if __package__ and __package__.count(".") >= 3:
    from ....mcp.primitives import load_model, DEFAULT_QUANT, DEFAULT_REPO_ID, _QUANT_CHOICES
else:
    from mcp.primitives import load_model, DEFAULT_QUANT, DEFAULT_REPO_ID, _QUANT_CHOICES
```

- [ ] `consumers/comfyui/loader.py`: swap `dgemma.model` → `mcp.primitives` for `load_model`, `_QUANT_CHOICES`, `DEFAULT_QUANT`, `DEFAULT_REPO_ID`
- [ ] `consumers/comfyui/sampler.py`: swap `dgemma.loop` → `mcp.primitives` for `run_diffusion`, `decode_frames`, `KNOB_DOCS`, all `DEFAULT_*` constants

**`mcp/server.py`:** replace commands dispatch with primitives:
- [ ] Replace `from surfaces.mcp.commands import generate, model` with imports from `mcp.primitives`
- [ ] JSON-RPC schema (`Tool` definitions) stays in server.py or a slim `mcp/commands/` if schemas grow large. The callable logic is in primitives.

### Phase 3 — Root entry point update

- [ ] `__init__.py`: update all imports from `.surfaces.comfyui.*` to `.consumers.comfyui.*` (both relative and absolute branches)
- [ ] `__init__.py`: update `WEB_DIRECTORY` from `"./surfaces/comfyui/web"` to `"./consumers/comfyui/web"`

### Phase 4 — Test updates

**Tests importing `surfaces.mcp.*`:**

| File | Change |
|------|--------|
| `tests/test_mcp_dual_context_import.py` | Update all `synthetic_pack_root}.surfaces.mcp.*` → `mcp.*`; adjust dot-count assertions (was 3 dots under surfaces/mcp, now 2 under mcp) |
| `tests/test_mcp_generate_command.py` | `from surfaces.mcp.commands import generate` → `from mcp.primitives import ...`; `from surfaces.mcp.state_manager` → `from mcp.state_manager` |
| `tests/test_mcp_model_command.py` | Same — commands.model → primitives; state_manager path update |
| `tests/test_mcp_import_guard.py` | Update all subprocess code strings from `surfaces.mcp.*` to `mcp.*`; update ComfyUI import references from `surfaces.comfyui.*` to `consumers.comfyui.*` |
| `tests/test_mcp_sdk_guard.py` | `from surfaces.mcp._mcp_sdk_guard` → `from mcp._mcp_sdk_guard` |
| `tests/test_mcp_server_dispatch.py` | `from surfaces.mcp import server` → `from mcp import server`; state_manager path update |
| `tests/test_mcp_statelessness.py` | All `surfaces.mcp.*` → `mcp.*` imports; monkeypatch paths updated |
| `tests/test_mcp_surface_seam.py` | Update all subprocess code strings from `surfaces.mcp` to `mcp`; update docstring references |
| `tests/test_units_glossary_mint.py` | `from surfaces.mcp.commands import generate` → `from mcp.primitives import ...`; `from surfaces.comfyui.sampler` → `from consumers.comfyui.sampler` |

**Tests importing `surfaces.comfyui.*`:**

| File | Change |
|------|--------|
| `tests/test_comfyui_loader_context.py` | Update module path references from `surfaces/comfyui/` to `consumers/comfyui/`; WEB_DIRECTORY assertion updated for new path |
| `tests/test_dual_context_import.py` | All `synthetic_pack_root}.surfaces.comfyui.*` → `consumers.comfyui.*`; adjust dot-count assertions (was 2 dots under surfaces/comfyui, now 3 under consumers/comfyui) |
| `tests/test_frames_image.py` | `from surfaces.comfyui.frames_image` → `from consumers.comfyui.frames_image` |
| `tests/test_kv_cache_cold_wiring.py` | Update imports from `surfaces.comfyui.*` to `consumers.comfyui.*` |
| `tests/test_kv_cache_nodes.py` | Same |
| `tests/test_live_seams.py` | `from surfaces.comfyui.sampler` → `from consumers.comfyui.sampler` |
| `tests/test_loader_contract.py` | Update imports from `surfaces.comfyui.*` to `consumers.comfyui.*` |
| `tests/test_loader_folder_paths.py` | Same |
| `tests/test_run_log_writer.py` | Same |
| `tests/test_socket_mint.py` | All 7 ComfyUI node imports updated from `surfaces.comfyui.*` to `consumers.comfyui.*` |
| `tests/test_tally_audit_node.py` | Same |
| `tests/test_token_trace_node.py` | Same |
| `tests/test_trace_node.py` | All 4 ComfyUI trace imports updated |

### Phase 5 — New enforcement test

- [ ] Create `tests/test_contract_seam.py`: subprocess test asserting consumers/comfyui does NOT import `dgemma/` directly (only through `mcp.primitives`)
  ```python
  """ComfyUI consumers must route through mcp.primitives, not import dgemma/ directly."""
  # Subprocess: import consumers.comfyui.loader; assert no dgemma.* module leaked
  # that wasn't pulled in transitively through mcp.primitives.
  ```

### Phase 6 — Documentation updates

- [ ] **ARCHITECTURE.md**: replace layer diagram with four-layer model (see Decision section above); add import rules table; remove GAP (#137) citations since the gap is now closed by this ADR's implementation
- [ ] **ADR-CDG-008**: add supersession note at top — "Section 1 framing ('peer surfaces') and directory structure decision (`surfaces/`) superseded by ADR-CDG-018. The core/surface seam rules (rules 1–7) remain in force."
- [ ] **ROADMAP.md line 60**: replace "ComfyUI is one peer surface among others" with "ComfyUI consumes MCP primitives (the contract layer)"
- [ ] **AGENTS.md**: update architecture diagram to show `mcp/` at top level, `consumers/comfyui/` under consumers; update Rule 2 reference from "surfaces/mcp/" to "mcp/primitives.py"
- [ ] Update all docstring references in source files that mention `surfaces/mcp/` or `surfaces/comfyui/` paths (see blast radius below)

---

## Blast radius — every file touched

### Moved (git mv, content updated for new path):
| Source | Destination |
|--------|-------------|
| `surfaces/mcp/__init__.py` | `mcp/__init__.py` |
| `surfaces/mcp/_mcp_sdk_guard.py` | `mcp/_mcp_sdk_guard.py` |
| `surfaces/mcp/server.py` | `mcp/server.py` |
| `surfaces/mcp/state_manager.py` | `mcp/state_manager.py` |
| `surfaces/comfyui/__init__.py` | `consumers/comfyui/__init__.py` |
| `surfaces/comfyui/denoise.py` | `consumers/comfyui/denoise.py` |
| `surfaces/comfyui/encode.py` | `consumers/comfyui/encode.py` |
| `surfaces/comfyui/frames_image.py` | `consumers/comfyui/frames_image.py` |
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
| `surfaces/mcp/commands/__init__.py` | Logic absorbed into primitives.py |
| `surfaces/mcp/commands/generate.py` | Logic extracted to primitives.py, schema stays in server.py |
| `surfaces/mcp/commands/model.py` | Same |

### New:
| File | Purpose |
|------|---------|
| `mcp/primitives.py` | The contract layer — callable Python functions |
| `tests/test_contract_seam.py` | Enforcement test — consumers don't import core directly |

### Modified (import paths, docstrings):
| File | Change type |
|------|-------------|
| `__init__.py` | Import paths + WEB_DIRECTORY |
| `consumers/__init__.py` | Docstring reference to `surfaces/__init__.py` removed |
| `tests/test_comfyui_loader_context.py` | Module path references |
| `tests/test_dual_context_import.py` | All synthetic import paths (MCP + ComfyUI) |
| `tests/test_frames_image.py` | Import paths |
| `tests/test_kv_cache_cold_wiring.py` | Import paths |
| `tests/test_kv_cache_nodes.py` | Import paths |
| `tests/test_live_seams.py` | Import paths |
| `tests/test_loader_contract.py` | Import paths |
| `tests/test_loader_folder_paths.py` | Import paths |
| `tests/test_mcp_dual_context_import.py` | All synthetic import paths + dot-count assertions |
| `tests/test_mcp_generate_command.py` | Import paths |
| `tests/test_mcp_import_guard.py` | Subprocess code strings (MCP + ComfyUI) |
| `tests/test_mcp_model_command.py` | Import paths |
| `tests/test_mcp_sdk_guard.py` | Import path |
| `tests/test_mcp_server_dispatch.py` | Import paths |
| `tests/test_mcp_statelessness.py` | Import paths + monkeypatch targets |
| `tests/test_mcp_surface_seam.py` | Subprocess code strings + docstrings |
| `tests/test_run_log_writer.py` | Import paths |
| `tests/test_socket_mint.py` | 7 import paths |
| `tests/test_tally_audit_node.py` | Import path |
| `tests/test_token_trace_node.py` | Import path |
| `tests/test_trace_node.py` | 4 import paths |
| `tests/test_units_glossary_mint.py` | Import paths (MCP + ComfyUI) |

### Documentation:
| File | Change type |
|------|-------------|
| `ARCHITECTURE.md` | Layer diagram, import rules table, GAP citations removed |
| `AGENTS.md` | Architecture diagram, Rule 2 reference |
| `ROADMAP.md` | Line 60 — "peer surface" → "consumes MCP primitives" |
| `decisions/adr-cdg-008-mcp-center-multi-surface-topology.md` | Supersession note at top |

### Untouched:
| File/Dir | Reason |
|----------|--------|
| `dgemma/*` | Core — no changes needed |
| `consumers/analysis.py` | Already in consumers/, no path change |
| `consumers/run_log.py` | Same |
| `consumers/tally_audit.py` | Same |

---

## Rollback plan

Every phase is a discrete git commit. Rollback = revert the last N commits:

- Phase 6 (docs): prose-only, trivially reversible
- Phase 5 (new test): delete one file
- Phase 4 (test updates): revert import paths — mechanical, no logic change
- Phase 3 (root entry point): revert `__init__.py`
- Phase 2 (import redirection): revert ComfyUI imports back to `dgemma.*`
- Phase 1c (delete surfaces/): recreate from git history
- Phase 1b (move comfyui → consumers): reverse git mv
- Phase 1a (move mcp → top-level): reverse git mv
- Phase 0 (primitives extraction): delete `mcp/primitives.py`, restore `surfaces/mcp/commands/*.py`

**Recommended commit granularity**: one commit per phase. If Phase 0 (primitives) is correct, everything else is mechanical and can be batched into a single "directory morph + import update" commit if desired — but phased commits are safer for bisect.

---

## Why this is the most efficient path

1. **Primitives extraction first** isolates the only judgment-bearing step (extracting callable logic from commands/*.py). Everything after is mechanical: rename, update imports, run tests.
2. **No intermediate state where both old and new paths coexist.** Each phase moves files then updates references — no period of ambiguity where an agent could import from either path.
3. **The enforcement test (Phase 5) is the durable signal.** After this ADR lands, any future consumer that imports `dgemma/` directly fails a subprocess test. The directory structure reinforces it visually; the test enforces it mechanically.
4. **Modeled on prompt-prix** which has been stable under agentic coding — the primitives layer is the pattern that held there.

---

## Open questions

1. **Should `mcp/commands/` survive as a slim schema-only module?** Currently `server.py` imports commands and dispatches by tool name. If schemas grow large (e.g., ADR-CDG-017's remelt spec), keeping them in a separate `commands/` under `mcp/` is cleaner than embedding in `server.py`. Decision: keep schema definitions inline in `server.py` for now; extract to `mcp/commands/` only when the file exceeds ~200 lines.

2. **Does `consumers/comfyui/socket_types.py` stay under consumers or move to `mcp/`?** Socket types are ComfyUI's envelope (rule 4, ONE-MINT). They're consumer-specific vocabulary — staying in `consumers/comfyui/` is correct. The MCP server doesn't use socket types; it uses JSON keys.

3. **Should the dual-context import gate pattern be retired for `mcp/primitives.py`?** Yes — primitives is a plain Python module, not loaded by ComfyUI's directory loader. It has no dual-context gate. Only consumers/comfyui files need the gate (they're loaded both as ComfyUI nodes and as pytest modules).
