"""surfaces/mcp/commands/encode.py — the `encode` tool: MCP KV-cache mint/
advance parity with ComfyUI's `DGemmaEncode` (ADR-CDG-025).

Thin adapter (ARCHITECTURE.md surface-tier rules), same posture as
`generate.py`: unpack `args`, delegate the mint/advance body to
`StateManager.encode_into_registry` (itself a thin wrap of
`dgemma.kv_cache.encode_sequence` — no new core surface, no denoising-loop
logic here), return a JSON-safe `{"kv_cache_id": ...}`.

**Why the registry lives on `StateManager`, not here** (mirrors `generate.py`'s
own `_active_runs` placement note, inverted): `_active_runs` is per-CALL
transient plumbing removed the instant its call returns, so it stays OUT of
`StateManager`. A KV-cache handle is the opposite — it is explicitly meant to
OUTLIVE the `encode` call that minted it (a caller mints once, drives
`generate` against it across multiple later calls, per ADR-CDG-025 §2's IN-3
advance parity and §Alternatives Option B's rejection). That is real
cross-call state, which is exactly what `StateManager` exists to hold
(ADR-CDG-008 Correction 1) — ADR-CDG-025 §1 amends rule 6 to say so
explicitly, rather than this module inventing a second registry outside the
one place ARCHITECTURE.md's rule-6 doctrine already governs.

**`kv_cache_id=None` (mint, IN-1) vs. `kv_cache_id=<handle>` (advance, IN-3)
is the SAME tool** — `encode_into_registry`'s own `into is None` dispatch
(delegated straight through from `dgemma.kv_cache.encode_sequence`'s
identical dispatch) — mirroring `DGemmaEncode`'s "same node body, `into is
None` dispatches mint vs. advance" shape (ADR-CDG-025 §2), transcribed onto
"same tool, `kv_cache_id` presence dispatches mint vs. advance."
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
    """`encode` tool definition."""
    return [
        Tool(
            name="encode",
            description=(
                "Mint a fresh DiffusionGemma KV-cache from raw-encoded text "
                "(context, not a chat-templated prompt/turn — no role markers, "
                "no generation-prompt suffix, no thinking mechanism), or advance "
                "an existing one with newly-committed text. Requires a model "
                "already loaded via the load_model tool. Returns an opaque "
                "kv_cache_id handle, resolvable by the generate tool's optional "
                "kv_cache_id parameter (ADR-CDG-025). The handle is held "
                "server-side only (no disk serialization) and is invalidated "
                "the moment a different model is loaded."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": (
                            "Raw-encoded CONTEXT to tokenize and commit to the cache. "
                            "On a fresh mint (kv_cache_id omitted) this is the full "
                            "context; when advancing (kv_cache_id given), this is the "
                            "newly-committed continuation."
                        ),
                    },
                    "kv_cache_id": {
                        "type": "string",
                        "description": (
                            "Omit (or null) to mint a fresh cache from prompt alone "
                            "(IN-1). Supply a handle from a prior encode call to "
                            "advance it with prompt's newly-committed text (IN-3) — "
                            "the SAME handle string is returned, now naming the "
                            "advanced cache (advance-returns-new-payload, "
                            "ADR-CDG-012 §3). An unknown or expired handle is "
                            "rejected, never silently treated as 'start fresh'."
                        ),
                    },
                },
                "required": ["prompt"],
            },
        ),
    ]


async def encode(manager: StateManager, args: dict[str, Any]) -> dict[str, Any]:
    """Thin adapter over `StateManager.encode_into_registry`: unpack, call
    once, wrap. `manager.require_model()`/`resolve_kv_cache` (both fired
    inside `encode_into_registry`) supply the same fail-loud posture
    `generate`'s missing-model / unknown-handle doors already use — a
    missing model or an unresolvable `kv_cache_id` raises here and is turned
    into a structured `{"error": ...}` by `server.py:call_tool`'s outer
    handler, never a silent fallback.
    """
    prompt = args.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    kv_cache_id = args.get("kv_cache_id")
    handle = manager.encode_into_registry(prompt, kv_cache_id=kv_cache_id)
    return {"kv_cache_id": handle}
