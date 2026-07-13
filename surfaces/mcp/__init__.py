"""surfaces/mcp — the base MCP surface over `dgemma`'s one contract
(`load_model` + `run_diffusion`), per ADR-CDG-008 Phase 2.

Transcribed from the ecosystem's MCP-surface exemplar,
`semantic-kinematics-mcp` (`.../semantic_kinematics/mcp/`): a
`commands/*.py` stateless-verb layer (`get_tools()` + async
`handler(state_manager, args)`), a `server.py` aggregating tool schemas and
dispatching calls, and a `state_manager.py` holding the surface's own
lifecycle object — same shape, this repo's contract underneath.

Two corrections adopted deliberately, over sk-mcp's OWN documented debt
(its ADR-003, `.../docs/ADRs/proposed/ADR-003-stateless-mcp-contract.md`),
not its current state:

1. **`STATELESS-CORE` split — persist the load, not the run.** sk-mcp's
   `StateManager` still retains a live `_adapter` plus a cross-call
   `_embedding_cache` (`state_manager.py:51-52,83-86`) — exactly the
   cross-call-mutable-state class ARCHITECTURE.md rule 6 forbids (the
   observed 25-vs-29 heatmap frame-count mismatch this repo already saw from
   a cached scheduler). `surfaces/mcp/state_manager.py` persists ONLY the
   ~53GB `DGemmaModel` load; every `generate` call builds a fresh
   scheduler/canvas/run-state by calling `dgemma.run_diffusion` itself,
   which already constructs all of that fresh per call
   (`dgemma/loop.py:run_diffusion`) — this surface adds no memoization on
   top.
2. **Keep the automated boundary test — do not regress to review-only.**
   sk-mcp's own `docs/ARCHITECTURE.md:183-184` documents TWO live doors into
   its core (MCP JSON-RPC and direct UI import) with no import-boundary
   test, review-only. This repo already has `tests/test_seam.py`'s
   subprocess import-leak assertion; `tests/test_mcp_surface_seam.py`
   extends the same discipline to `surfaces.mcp` (asserts `dgemma` does not
   import `surfaces.mcp`, mirroring `test_seam.py`'s reverse direction).

The `mcp` SDK dependency is OPTIONAL (`pyproject.toml`'s `[project.optional-
dependencies].mcp`, mirroring how this pack already treats ComfyUI itself as
absent-by-design in core tests, `tests/test_seam.py`). `commands/*.py` and
`server.py` import `mcp.types`/`mcp.server` at module scope — narrow,
structural `ImportError` guards live at the CALL sites that need the SDK
absent to matter (this `__init__.py` itself imports nothing from `mcp`, so
`import surfaces.mcp` never requires the SDK to be installed).
"""
from __future__ import annotations
