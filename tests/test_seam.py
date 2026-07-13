"""dgemma imports and runs with zero ComfyUI present (ADR-CDG-003's
enforcement surface — the precondition for `dgemma/loop.py` being developable
and testable with no ComfyUI process alive at all).

Run out-of-process (a subprocess, not an in-process `sys.modules` check):
an in-process check could be fooled by a `comfy`/`nodes`/`surfaces` module
some earlier test already imported and left cached in `sys.modules`. A fresh
interpreter is the only way to observe what `import dgemma` actually pulls in
on its own.

This venv genuinely has no `comfy` package installed (ComfyUI is not a
dependency of this repo, by design), so if any `dgemma/*.py` module ever
imported `comfy`, `import dgemma` itself would raise here — the absence is
the enforcement, not a stub we have to maintain.

Extended per ADR-CDG-008 Phase 1 (issue #52 §4): after the `nodes/` ->
`surfaces/comfyui/` rename, the leak-check also rejects `surfaces`/
`surfaces.*` — the ARCHITECTURE.md "Core imports no surface" row's named
follow-up ("Must be updated to also reject `surfaces.*` after the rename").
`nodes` stays in the reject-list too: this venv still has no such package,
so its absence remains a free assertion even though the pack no longer uses
that name internally.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_CHECK_SCRIPT = """
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


def test_dgemma_imports_with_zero_comfy_present():
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"`import dgemma` failed:\n{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_dgemma_does_not_import_nodes_package():
    """Belt-and-suspenders on the same invariant, phrased as the ADR states
    it: `dgemma/` never imports from `nodes/` (only the reverse is allowed).
    `nodes/` no longer exists in this repo (ADR-CDG-008 Phase 1), but the
    absence check stays valid — nothing named `nodes` should ever leak in."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dgemma; "
            "assert not any(m == 'nodes' or m.startswith('nodes.') for m in sys.modules); "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_dgemma_does_not_import_surfaces_package():
    """ADR-CDG-008 Phase 1 (issue #52 §4): the surface layer is now named
    `surfaces/` (was `nodes/`) — `dgemma/` must never import from it, the
    same core-imports-no-surface invariant `test_seam.py` already enforces
    for the old name. Fails by design only if a `dgemma/*.py` module ever
    imports `surfaces`/`surfaces.*`, which today it does not."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dgemma; "
            "assert not any(m == 'surfaces' or m.startswith('surfaces.') for m in sys.modules); "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
