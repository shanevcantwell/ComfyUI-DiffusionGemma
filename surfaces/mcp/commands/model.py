"""surfaces/mcp/commands/model.py — model lifecycle tools: `load_model`,
`model_status`.

Transcribed from sk-mcp's `mcp/commands/model.py` shape (`get_tools()` +
async handlers taking `(manager, args)`), narrowed to this repo's ONE real
lifecycle action: load the ~53GB `DGemmaModel` (`dgemma/model.py:load_model`,
`dgemma/model.py:157`). There is no `model_unload` tool here (unlike
sk-mcp's) — see the module-level note below.

Rule #14 carried across from sk-mcp's `commands/embeddings.py:39,42,82-83`
(no silent model default): `load_model`'s MCP schema requires `repo_id` +
`quant` explicitly, with no `default` key in either property, even though
`dgemma.model.load_model` itself has grounded defaults
(`DEFAULT_REPO_ID`/`DEFAULT_QUANT`). This is a deliberate MCP-surface-level
tightening, not a core change: an agent driving this tool via a schema is
exactly the "silent-default" hazard Rule #14 names — a caller that forgets
to pass `repo_id` should get a loud schema-validation/handler error, not a
53GB download of whatever the core happens to default to. Gate-tested the
same way sk-mcp tests its own Rule #14 gate:
`tests/test_mcp_model_command.py` (an exploding fake `load_model` that
raises if reached without both args explicit).
"""
from __future__ import annotations

from typing import Any

if __package__ and __package__.count(".") >= 3:
    from .._mcp_sdk_guard import require_mcp_sdk
    from ..state_manager import StateManager
else:
    from surfaces.mcp._mcp_sdk_guard import require_mcp_sdk
    from surfaces.mcp.state_manager import StateManager

require_mcp_sdk()
from mcp.types import Tool  # noqa: E402


def get_tools() -> list[Tool]:
    """Model lifecycle tool definitions."""
    return [
        Tool(
            name="load_model",
            description=(
                "Load the DiffusionGemma model + processor into this server's "
                "persisted state (ADR-CDG-008 Phase 2: the ONLY cross-call state "
                "this MCP surface holds). Both repo_id and quant are required — "
                "no baked default — since loading is a ~53GB action a caller "
                "should always request explicitly."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": "Hugging Face repo id (e.g. 'google/diffusiongemma-26B-A4B-it')",
                    },
                    "quant": {
                        "type": "string",
                        "description": "Quantization mode; only 'none' is supported (issue #18)",
                    },
                    "local_files_only": {
                        "type": "boolean",
                        "description": "Restrict resolution to the local HF cache (default: false)",
                        "default": False,
                    },
                },
                "required": ["repo_id", "quant"],
            },
        ),
        Tool(
            name="model_status",
            description="Report whether a model is loaded, and if so which repo_id/quant/device.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


async def load_model_tool(manager: StateManager, args: dict[str, Any]) -> dict[str, Any]:
    """Load a model into `manager`. Thin adapter: unpack args, call
    `StateManager.load` (itself a thin wrap of `dgemma.model.load_model`),
    wrap the result. No retry/caching/reuse logic here — see
    `StateManager.load`'s own docstring for why a same-args call always
    reloads rather than silently short-circuiting."""
    repo_id = args.get("repo_id")
    quant = args.get("quant")
    local_files_only = args.get("local_files_only", False)

    # Rule #14 gate: fail loud, before touching the manager, rather than
    # falling through to `dgemma.model.load_model`'s own grounded defaults.
    if not repo_id:
        return {"error": "repo_id is required; no default (see load_model's schema)"}
    if not quant:
        return {"error": "quant is required; no default (see load_model's schema)"}

    model = manager.load(repo_id=repo_id, quant=quant, local_files_only=bool(local_files_only))
    return {
        "status": "loaded",
        "repo_id": repo_id,
        "quant": quant,
        "device": model.device,
        "dtype": model.dtype,
    }


async def model_status_tool(manager: StateManager, args: dict[str, Any]) -> dict[str, Any]:
    """Report `manager.status()` verbatim — a pure read, no core call."""
    return manager.status()
