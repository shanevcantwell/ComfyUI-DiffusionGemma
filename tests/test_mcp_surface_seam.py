"""Boundary enforcement for `surfaces/mcp/` (ADR-CDG-008 Phase 2, Correction
2 — "keep the automated boundary test, do not regress to review-only").

`semantic-kinematics-mcp`'s own `docs/ARCHITECTURE.md:183-184` documents TWO
live doors into its core (MCP JSON-RPC dispatch and a direct UI import) with
NO import-boundary test — review-only. This repo already has
`tests/test_seam.py`'s subprocess import-leak assertion for the
core-imports-no-surface direction; this module is the Phase-2 EXTENSION the
ADR names, adding:

1. The REVERSE direction `test_seam.py` doesn't cover: `dgemma` (the core)
   must never import `surfaces.mcp` — extending the same "core imports no
   surface" invariant `test_seam.py:36-63,78-97` already enforces for
   `surfaces`/`surfaces.comfyui` (implicitly, via the blanket `surfaces`/
   `surfaces.` prefix check) to be explicit about the new peer surface too.
2. `surfaces.mcp` imports no `comfy`/`nodes` — the MCP surface is a PEER of
   the ComfyUI surface (ARCHITECTURE.md rule 2), so it must be just as
   ComfyUI-agnostic as `dgemma` itself; a `surfaces.mcp` module reaching
   into `surfaces.comfyui` or `comfy` would be a cross-surface leak, not a
   surface-over-core wrap.

Run out-of-process (subprocess), same rationale as `test_seam.py`: a fresh
interpreter is the only way to observe what an import actually pulls in,
unmasked by another test's already-populated `sys.modules`.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_DGEMMA_NEVER_IMPORTS_MCP_SURFACE = """
import sys
import dgemma

leaked = [
    m for m in sys.modules
    if m in ("comfy", "nodes", "surfaces")
    or m.startswith(("comfy.", "nodes.", "surfaces."))
]
assert not leaked, f"unexpected modules pulled in by `import dgemma`: {leaked}"
print("OK")
"""

_MCP_SURFACE_NEVER_IMPORTS_COMFY = """
import sys
import surfaces.mcp

leaked = [
    m for m in sys.modules
    if m in ("comfy", "nodes")
    or m.startswith(("comfy.", "nodes.", "surfaces.comfyui"))
]
assert not leaked, f"unexpected modules pulled in by `import surfaces.mcp`: {leaked}"
print("OK")
"""


def _run(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_dgemma_does_not_import_mcp_surface():
    """The reverse direction: `import dgemma` must never pull in
    `surfaces.mcp` (or any `surfaces.*`) — `test_seam.py` already checks
    this via its blanket `surfaces`/`surfaces.` prefix match; this is the
    Phase-2-specific restatement the ADR's roadmap names as this phase's own
    verifiable ("the MCP surface invokes the core with zero ComfyUI import
    ... asserting `import surfaces.mcp` pulls in no `comfy`/`nodes`")
    paired with its natural other half."""
    result = _run(_DGEMMA_NEVER_IMPORTS_MCP_SURFACE)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_mcp_surface_does_not_import_comfy_or_comfyui_surface():
    """`import surfaces.mcp` (the package init only — no `mcp` SDK needed,
    see `tests/test_mcp_import_guard.py`) must pull in neither `comfy`/
    `nodes` nor the sibling `surfaces.comfyui` peer surface. Peers over one
    contract (ARCHITECTURE.md rule 2) never import each other directly."""
    result = _run(_MCP_SURFACE_NEVER_IMPORTS_COMFY)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
