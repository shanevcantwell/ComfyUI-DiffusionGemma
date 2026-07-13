"""surfaces/mcp/commands — the stateless verb layer (sk-mcp pattern).

Each module exports `get_tools() -> list[Tool]` (schemas) and one or more
async `handler(state_manager, args) -> dict` functions — thin adapters only:
unpack `args`, call exactly one `dgemma.*` function (or a `StateManager`
accessor that itself wraps one), wrap the result in a plain dict. No
denoising-step loop, no scheduler construction, no analysis math lives here
(ARCHITECTURE.md surface-tier rules).
"""
from __future__ import annotations
