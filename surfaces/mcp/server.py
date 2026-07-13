"""surfaces/mcp/server.py — the MCP server aggregating this surface's tools.

Transcribed near-verbatim from `semantic-kinematics-mcp`'s `mcp/server.py`
shape: `Server("name")`, `@server.list_tools()` aggregating each command
module's `get_tools()`, `@server.call_tool()` dispatching by name and
wrapping every result as `TextContent(json.dumps(...))`, `stdio_server()` +
`main()` as the process entry point.

**Named deviation from the ADR-CDG-008 Phase-2 guidance's `[project.scripts]`
entry point:** this repo's `pyproject.toml` deliberately ships with
`[tool.setuptools] py-modules = []` — "no importable package to install yet
... ComfyUI loads this pack by directory, not by pip install." Adding a
console-script entry point requires setuptools to discover `dgemma`/
`surfaces` as installable packages, a bigger packaging-story change than
this phase's scope (adding `surfaces/mcp/` itself). Until that's decided,
run this server with `python -m surfaces.mcp.server` from the repo root
(with `pip install -e '.[mcp]'` or equivalent deps on `sys.path`) instead of
a minted console script — `main()` below is the same function either
entry point would call.

The `mcp` SDK is an OPTIONAL dependency (`pyproject.toml`'s
`[project.optional-dependencies].mcp` — mirrors how this pack already
treats ComfyUI itself as absent-by-design in core tests,
`tests/test_seam.py`). Importing THIS module requires `mcp` to be installed
(it constructs a `Server` at import time, same as sk-mcp's own
`server.py:29`) — that is the intended shape: a ComfyUI-only install never
imports `surfaces.mcp.server` (nothing else in this pack imports it), so the
absent SDK never blocks the ComfyUI surface. `tests/test_mcp_surface_seam.py`
guards the OTHER direction (`dgemma` never imports `surfaces.mcp`), and
`tests/test_mcp_import_guard.py` asserts `surfaces.mcp` itself (the
package init, `state_manager`, `commands/*`'s schemas) stays importable
without `mcp` present — only `server.py`'s dispatch loop needs the SDK.

No dual-context (ComfyUI-loader vs. plain) import gate here, unlike
`surfaces/comfyui/*.py` / `surfaces/mcp/commands/*.py` / `state_manager.py`:
this module is never reached through ComfyUI's directory loader at all
(nothing in the pack's root `__init__.py` imports `surfaces.mcp` —
verified, `tests/test_mcp_import_guard.py`'s ComfyUI-surface-still-loads
test) — its only real entry points are `python -m surfaces.mcp.server` and a
future console script, both always plain-package contexts. A dual-context
gate here would be untestable dead code, not a second reachable path.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from surfaces.mcp._mcp_sdk_guard import require_mcp_sdk

require_mcp_sdk()
from mcp.server import Server  # noqa: E402
from mcp.server.stdio import stdio_server  # noqa: E402
from mcp.types import TextContent, Tool  # noqa: E402

from surfaces.mcp.commands import generate, model
from surfaces.mcp.state_manager import StateManager


server = Server("comfyui-diffusiongemma")
state_manager = StateManager()

# Name -> async handler(state_manager, args). Built once, module scope —
# same "if/elif dispatch table" shape sk-mcp's server.py uses
# (`server.py:52-83`), expressed as a dict instead of the sk-mcp if/elif
# chain (fewer lines, same one-name-one-handler mapping; no behavior
# difference).
_HANDLERS = {
    "load_model": model.load_model_tool,
    "model_status": model.model_status_tool,
    "generate": generate.generate,
    "cancel_run": generate.cancel_run,
}


@server.list_tools()
async def list_tools() -> list[Tool]:
    """Aggregate every command module's `get_tools()` — same shape as
    sk-mcp's `server.py:33-42`."""
    tools: list[Tool] = []
    tools.extend(model.get_tools())
    tools.extend(generate.get_tools())
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Dispatch to the named handler, wrap the result as JSON `TextContent`
    — same shape as sk-mcp's `server.py:45-94` (including the outer
    try/except turning any handler exception into a structured `{"error":
    ...}` payload rather than letting it propagate as a transport-level
    fault)."""
    try:
        handler = _HANDLERS.get(name)
        if handler is None:
            result: dict[str, Any] = {"error": f"Unknown tool: {name}"}
        else:
            result = await handler(state_manager, arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
    except Exception as exc:  # noqa: BLE001 — deliberate breadth: never let a
        # handler bug crash the server process; report it as tool-call data
        # instead (same posture as sk-mcp's server.py:90-94).
        return [TextContent(type="text", text=json.dumps({"error": str(exc)}))]


async def run_server() -> None:
    """Run the MCP server over stdio."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    """Process entry point (`python -m surfaces.mcp.server` — see this
    module's docstring for why there is no minted console script yet)."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
