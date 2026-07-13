"""dgemma imports and runs with zero ComfyUI present (ADR-CDG-003's
enforcement surface — the precondition for `dgemma/loop.py` being developable
and testable with no ComfyUI process alive at all).

Run out-of-process (a subprocess, not an in-process `sys.modules` check):
an in-process check could be fooled by a `comfy`/`nodes`/`surfaces`/`consumers`
module some earlier test already imported and left cached in `sys.modules`. A
fresh interpreter is the only way to observe what `import dgemma` actually
pulls in on its own.

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

Extended again per ADR-CDG-008 Phase 3+4 (issue #55 §4): after the analysis
relocation to `consumers/analysis.py`, the leak-check also rejects
`consumers`/`consumers.*` — the "core imports no analysis" boundary named in
ADR-CDG-008's enforcement-surface table, previously prose-only
(`dgemma/sampling.py`'s docstring claimed consumer status while
`dgemma/__init__.py:26-31` contradicted it by re-exporting it). Today
`dgemma` genuinely does not import `consumers`, so — like `nodes`/`surfaces`
before it — the absence is a free assertion, not a maintained stub.
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
    if m in ("comfy", "nodes", "surfaces", "consumers")
    or m.startswith(("comfy.", "nodes.", "surfaces.", "consumers."))
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


def test_dgemma_does_not_import_consumers_package():
    """ADR-CDG-008 Phase 4 (issue #55 §4) — the primary boundary assertion
    this phase exists to create: the base contract (`dgemma/`) must never
    import the analysis/consumer tier now that it lives in
    `consumers/analysis.py` (Phase 3). This is what flips ARCHITECTURE.md's
    "Core imports no analysis" row from prose-only to in force, and it is a
    named, single-purpose twin of `test_dgemma_does_not_import_surfaces_package`
    above so a failure here reads unambiguously as the analysis boundary
    breaking, not the surface boundary.

    Mutation check (recorded per issue #55 §4): temporarily re-adding
    `from consumers.analysis import build_commit_heatmap` to
    `dgemma/__init__.py` makes this test fail by name (`consumers` appears
    in `sys.modules` after `import dgemma`) — and independently makes
    `test_dgemma_imports_with_zero_comfy_present` above fail too, since its
    `_CHECK_SCRIPT` leak-list was extended to include `consumers` in the
    same phase. Both failing by name is the proof this assertion bites
    rather than vacuously passing (issue #55 risk R2)."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import dgemma; "
            "assert not any(m == 'consumers' or m.startswith('consumers.') for m in sys.modules); "
            "print('OK')",
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
