"""surfaces/mcp/_mcp_sdk_guard.py — the ONE place that turns an absent `mcp`
SDK into an actionable `RuntimeError` instead of a bare `ModuleNotFoundError`
(deliverable 5: "a structural import guard, not a hard requirement").

Same shape as `dgemma/model.py`'s `_check_transformers_version` /
`dgemma/loop.py`'s `_check_diffusers_version` guards: a narrow, named
precondition check at the top of every module that actually needs the
dependency, not a blanket `try/except` that would also swallow unrelated
bugs. `mcp` is declared in `pyproject.toml`'s `[project.optional-
dependencies].mcp` (NOT the core `[project] dependencies`) — a ComfyUI-only
install is never forced to have it, mirroring how this pack already treats
ComfyUI itself as absent-by-design in `dgemma/`'s own tests
(`tests/test_seam.py`).

Only `surfaces/mcp/commands/*.py` and `surfaces/mcp/server.py` call this —
`surfaces/mcp/__init__.py` and `surfaces/mcp/state_manager.py` import
NOTHING from `mcp`, so `import surfaces.mcp` / `import
surfaces.mcp.state_manager` never requires the SDK at all (this is what
`tests/test_mcp_import_guard.py` asserts).
"""
from __future__ import annotations


def require_mcp_sdk():
    """Import and return the `mcp` top-level package, or raise a `RuntimeError`
    naming the missing optional extra — never a bare `ModuleNotFoundError`
    with no context about how to fix it."""
    try:
        import mcp
    except ImportError as exc:
        raise RuntimeError(
            "surfaces/mcp requires the optional 'mcp' SDK, which is not "
            "installed in this Python environment. This is an OPTIONAL "
            "extra (ADR-CDG-008 Phase 2) — the ComfyUI surface and the "
            "dgemma core work with zero MCP dependencies present. To use "
            "the MCP surface, install it: `pip install 'comfyui-"
            "diffusiongemma[mcp]'` (or `pip install mcp>=1.0.0` directly). "
            f"Original error: {exc}"
        ) from exc
    return mcp
