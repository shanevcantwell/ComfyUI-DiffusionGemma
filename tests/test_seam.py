"""dgemma imports and runs with zero ComfyUI present (ADR-CDG-003's
enforcement surface — the precondition for `dgemma/loop.py` being developable
and testable with no ComfyUI process alive at all).

Run out-of-process (a subprocess, not an in-process `sys.modules` check):
an in-process check could be fooled by a `comfy`/`nodes` module some earlier
test already imported and left cached in `sys.modules`. A fresh interpreter
is the only way to observe what `import dgemma` actually pulls in on its own.

This venv genuinely has no `comfy` package installed (ComfyUI is not a
dependency of this repo, by design), so if any `dgemma/*.py` module ever
imported `comfy`, `import dgemma` itself would raise here — the absence is
the enforcement, not a stub we have to maintain.
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
    if m == "comfy" or m.startswith("comfy.") or m == "nodes" or m.startswith("nodes.")
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
    it: `dgemma/` never imports from `nodes/` (only the reverse is allowed)."""
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
