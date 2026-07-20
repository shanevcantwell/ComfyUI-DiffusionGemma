"""tests/test_kv_cache_coverage_floor.py — ADR-CDG-012 DV.1 (issue #62 §H):
per-module 100% row-coverage floor on the `KV_CACHE` channel-crossing code,
mechanism (b) per the ratified implementation plan.

The ADR's DV.1 clause requires 100% row coverage on the channel-crossing
modules, measured in the greater-system (full mocked suite) profile — but
`coverage.py`'s `fail_under` is a single global threshold that cannot
natively express "these named files at 100%, the rest unchanged" (the ADR's
own DV.1 Open Question). Issue #62's ratified plan resolves the mechanism
question in favor of (b): a DEDICATED gate test (this file) that reads the
already-produced `.coverage` dataset and asserts each named channel module
is at 100%, failing CI **by module name** — never a `fail_under`-scoped
second coverage run (see the plan's §H for the full 5-point rationale: this
composes with the repo's existing single in-process run, doesn't risk
silently widening/narrowing an `omit` list, and reads whichever dataset is
present, including #59's combined unit+e2e dataset).

**Scope is explicit, not inflated (DV.1's own guard):** ONLY the modules this
phase's `KV_CACHE` channel code actually lives in. This test makes zero
claim about `dgemma/model.py`'s pre-existing, unrelated coverage gap (the
`from_pretrained` boundary only the `live` suite can reach) — it is not in
the allowlist below, so a regression there cannot be silently smuggled past
this gate, and this gate cannot regress model.py's gap into a false
blocker either.

**This test SKIPS (never fails) when no `.coverage` data is present** — e.g.
a bare `pytest` invocation with no `--cov`/`coverage run` wrapper. It is
meant to be run ALONGSIDE a coverage-collecting invocation (`pytest --cov`
or, per issue #50's own-run workaround, `python -m coverage run -m
pytest`), not as a replacement for one. A hard failure on absent data would
make every coverage-less test run of the suite fail this file for a reason
that has nothing to do with the invariant it guards.
"""
from __future__ import annotations

from pathlib import Path

import coverage
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# The allowlist (issue #62 plan §H "Scoped module list"): the KV_CACHE
# channel-crossing modules, positively named — not derived from an
# omit-list, so this floor cannot silently widen/narrow by editing an
# unrelated exclusion pattern (plan §H point 3).
_CHANNEL_MODULES = (
    "dgemma/kv_cache.py",
    "dgemma/types.py",
    "surfaces/comfyui/encode.py",
    "surfaces/comfyui/denoise.py",
    "surfaces/comfyui/socket_types.py",
)


def _coverage_data_available() -> coverage.CoverageData | None:
    data = coverage.CoverageData()
    try:
        data.read()
    except Exception:  # noqa: BLE001 — any read failure means "no usable data"
        return None
    if not data.measured_files():
        return None
    return data


@pytest.fixture(scope="module")
def coverage_data():
    data = _coverage_data_available()
    if data is None:
        pytest.skip(
            "no .coverage data file found — run under `pytest --cov` or "
            "`python -m coverage run -m pytest` (issue #50 own-run workaround) "
            "to exercise this gate"
        )
    return data


def _missing_lines_for(module_relpath: str) -> list[int]:
    """Analyzes `module_relpath` against the ALREADY-COLLECTED coverage data
    (module-scoped fixture already confirmed present) via `analysis2`, which
    reads the module's own source to determine total executable statements —
    exactly the per-file percent DV.1 asks for, not raw executed-line counts
    alone (`CoverageData` by itself has no notion of "lines that could have
    run")."""
    cov = coverage.Coverage()
    cov.load()
    abs_path = str(REPO_ROOT / module_relpath)
    _, _, _, missing, _ = cov.analysis2(abs_path)
    return missing


@pytest.mark.parametrize("module_relpath", _CHANNEL_MODULES)
def test_channel_module_at_100_percent_row_coverage(coverage_data, module_relpath):
    missing = _missing_lines_for(module_relpath)
    assert not missing, (
        f"DV.1 floor violated: {module_relpath} has uncovered line(s) {missing} "
        "in the full mocked-suite coverage profile — every KV_CACHE "
        "channel-crossing branch must be exercised by a test (ADR-CDG-012 DV.1)"
    )


def test_allowlist_is_not_empty():
    """Sanity: the parametrization itself must not be vacuous, or every
    assertion above would trivially pass on zero cases."""
    assert _CHANNEL_MODULES
