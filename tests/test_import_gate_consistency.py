"""Cross-gate consistency test for the hand-rolled dual-context
`__package__` import gates (issue #57).

Issue #57 found the gate population had grown past the four sites named at
filing (`__init__.py`, `consumers/analysis.py`, `surfaces/comfyui/loader.py`,
`surfaces/comfyui/trace.py`) to sixteen, each hand-authoring its own
`if __package__ ...:` predicate. The issue offered two shapes: (a) factor
the boolean into a shared helper, keeping the relative-import statements at
each site; or (b) a test asserting all gates agree. This module is shape
(b).

"Agree" does NOT mean byte-identical text — a module two directories under
the pack root genuinely needs a different dot-count threshold than one
three directories under it; collapsing that distinction would break the
gate, not consolidate it (see `surfaces/comfyui/loader.py`'s "GATE
CORRECTION" comment for the exact failure mode a wrong threshold produces).
"Agree" means every gate implements the SAME logical predicate —
`__package__` is truthy AND its dot-count is at least the module's own
nesting depth below the pack root — parameterized correctly for where that
module actually lives. This test derives the expected threshold from each
file's real filesystem depth (independent of what the source comment
*claims*) and checks the source text encodes exactly that predicate, so a
copy-paste of the wrong depth constant — the actual failure mode a helper
would not have caught either, since the depth argument is still hand-typed
at the call site under shape (a) — fails loudly here instead of only at
runtime inside a live ComfyUI process.

The site population itself is asserted against a fixed set (`GATE_SITES`)
so an added or removed gate is a visible, deliberate diff to this test, not
silent drift the issue's own history shows is easy to accumulate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The full gate population at issue #57 implementation time (16 sites) —
# grown from the four named at filing. Each entry is a repo-relative path.
# A change to this set is a deliberate statement about the gate population,
# not something that should happen as a side effect of an unrelated edit.
GATE_SITES = [
    "__init__.py",
    "consumers/analysis.py",
    "consumers/run_log.py",
    "consumers/tally_audit.py",
    "surfaces/comfyui/denoise.py",
    "surfaces/comfyui/emission.py",
    "surfaces/comfyui/encode.py",
    "surfaces/comfyui/loader.py",
    "surfaces/comfyui/run_log_writer.py",
    "surfaces/comfyui/sampler.py",
    "surfaces/comfyui/tally_audit.py",
    "surfaces/comfyui/token_trace.py",
    "surfaces/comfyui/trace.py",
    "surfaces/mcp/commands/generate.py",
    "surfaces/mcp/commands/model.py",
    "surfaces/mcp/state_manager.py",
]

# Matches the opening line of a dual-context gate, e.g.:
#   if __package__:
#   if __package__ and "." in __package__:
#   if __package__ and __package__.count(".") >= 2:
_GATE_LINE_RE = re.compile(
    r'^if __package__(?: and (?:"\." in __package__|__package__\.count\("\."\) >= (\d+)))?:\s*$'
)


def _expected_depth(relpath: str) -> int:
    """A module's nesting depth below the pack root, counted from its own
    filesystem location — e.g. `surfaces/comfyui/loader.py` sits two
    directories under the pack root (`surfaces/`, `surfaces/comfyui/`), so
    depth is 2. This is computed independently of any source comment's
    claim, so a stale or wrong comment cannot mask a wrong predicate."""
    return len(Path(relpath).parent.parts)


def _canonical_predicate(depth: int) -> str:
    """The one predicate family every gate must implement, rendered as the
    exact source line expected at that depth. Depth 0 (the pack root
    itself) collapses `count(".") >= 0` to bare truthiness, since
    `__package__` truthy already implies a dot-count of at least zero;
    depth 1 is conventionally spelled with `"." in __package__` (equivalent
    to `count(".") >= 1`) rather than the `.count()` form."""
    if depth == 0:
        return "if __package__:"
    if depth == 1:
        return 'if __package__ and "." in __package__:'
    return f'if __package__ and __package__.count(".") >= {depth}:'


def _find_gate_line(source: str) -> str | None:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("if __package__"):
            return stripped
    return None


@pytest.mark.parametrize("relpath", GATE_SITES)
def test_gate_predicate_matches_module_depth(relpath):
    """Each gate's predicate text is exactly the canonical form for its
    module's real filesystem depth — the cross-gate consistency check shape
    (b) of issue #57 asks for. A gate authored at the wrong depth threshold
    fails here instead of only inside a live ComfyUI process (the failure
    mode `loader.py`'s "GATE CORRECTION" comment documents)."""
    path = REPO_ROOT / relpath
    source = path.read_text(encoding="utf-8")

    gate_line = _find_gate_line(source)
    assert gate_line is not None, f"{relpath}: no `if __package__` gate found"

    depth = _expected_depth(relpath)
    expected = _canonical_predicate(depth)
    assert gate_line == expected, (
        f"{relpath}: gate predicate {gate_line!r} does not match the "
        f"canonical depth-{depth} predicate {expected!r} for a module at "
        f"this filesystem location"
    )


def test_gate_site_population_is_the_expected_16():
    """Guards the population itself: a `__package__` gate added anywhere in
    the tree without a corresponding `GATE_SITES` entry (or vice versa) is
    exactly the silent-drift failure mode issue #57 was filed against —
    four sites named at filing, sixteen found at implementation time with
    no test having tracked the growth in between."""
    found = set()
    for path in REPO_ROOT.rglob("*.py"):
        if ".git" in path.parts or "site-packages" in path.parts:
            continue
        relpath = path.relative_to(REPO_ROOT).as_posix()
        if relpath.startswith("tests/"):
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if _find_gate_line(source) is not None:
            found.add(relpath)

    assert found == set(GATE_SITES), (
        f"Gate population drifted from GATE_SITES.\n"
        f"  New/untracked sites: {sorted(found - set(GATE_SITES))}\n"
        f"  Missing/removed sites: {sorted(set(GATE_SITES) - found)}"
    )


def test_canonical_predicate_depth_0_matches_root_init():
    """Anchors `_canonical_predicate`'s depth-0 collapse against the actual
    root `__init__.py` gate, so the special case above is checked against
    real source, not only asserted in isolation."""
    assert _canonical_predicate(0) == "if __package__:"


def test_canonical_predicate_depth_1_matches_consumers_shape():
    """Anchors the depth-1 `"." in __package__` spelling against the
    `consumers/*.py` gates' actual shape."""
    assert _canonical_predicate(1) == 'if __package__ and "." in __package__:'
