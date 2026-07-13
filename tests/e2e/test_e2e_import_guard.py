"""The E2E battery's import graph touches nothing from the implementation
under test (ADR-CDG-013 Decision 1, issue #59 §2) — the enforcement surface
for the operator's independence requirement, not a convention.

Mirrors `tests/test_seam.py`'s shape exactly: run out-of-process (a fresh
subprocess, not an in-process `sys.modules` check), because an in-process
check could be fooled by `dgemma`/`surfaces`/`consumers` already being
cached in `sys.modules` from an earlier test in the same pytest session. A
fresh interpreter importing only the `e2e` battery modules is the only way
to observe what those modules actually pull in on their own.

This is the black-box tier's own version of `test_seam.py`'s "core imports
no surface" guard, aimed the other way: here the invariant is "the E2E
driver imports no implementation package at all" (not even the surfaces the
`live`/mocked tiers import directly), since the whole point of this tier is
that it proves the wiring by driving ComfyUI's own API, never by reaching
into the pack.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# The implementation packages this tier must never import, by design
# (ADR-CDG-013 Decision 1's exact list).
FORBIDDEN_PACKAGES = ("dgemma", "surfaces", "consumers")

_CHECK_SCRIPT = """
import sys

# Import every e2e test/support module the same way pytest collects them,
# without running pytest itself (pytest's own plugin/collection machinery
# imports things unrelated to this invariant and would make the leak-check
# noisy). importlib.import_module mirrors how `--import-mode=importlib`
# loads each file.
import importlib.util

MODULE_PATHS = {module_paths!r}

for name, path in MODULE_PATHS.items():
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

forbidden = {forbidden!r}
leaked = [
    m for m in sys.modules
    if m in forbidden or m.startswith(tuple(f + "." for f in forbidden))
]
assert not leaked, f"e2e battery modules pulled in forbidden packages: {{leaked}}"
print("OK")
"""


def _e2e_module_paths() -> dict[str, str]:
    e2e_dir = REPO_ROOT / "tests" / "e2e"
    paths = {}
    for path in sorted(e2e_dir.glob("*.py")):
        if path.name == "test_e2e_import_guard.py":
            continue  # this file itself imports subprocess/sys/Path only; skip self-check noise
        module_name = f"_e2e_guard_check.{path.stem}"
        paths[module_name] = str(path)
    return paths


def test_e2e_battery_imports_nothing_from_the_implementation():
    module_paths = _e2e_module_paths()
    script = _CHECK_SCRIPT.format(module_paths=module_paths, forbidden=FORBIDDEN_PACKAGES)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, (
        f"e2e import-guard check failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "OK" in result.stdout


def test_e2e_battery_source_has_no_static_forbidden_imports():
    """Belt-and-suspenders, same shape as `test_seam.py`'s twin checks: a
    textual scan of every `tests/e2e/*.py` file for a static `import
    dgemma`/`from surfaces import ...`/etc. line. The subprocess check above
    is the real enforcement (it catches indirect/dynamic imports too); this
    is a fast, precise second signal when it fails — e.g. immediately naming
    which file and line introduced the forbidden import, rather than only
    "some module in the set" the subprocess check reports."""
    import ast

    e2e_dir = REPO_ROOT / "tests" / "e2e"
    offending: list[str] = []
    for path in sorted(e2e_dir.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.module else []
            else:
                continue
            for name in names:
                if name is None:
                    continue
                top_level = name.split(".", 1)[0]
                if top_level in FORBIDDEN_PACKAGES:
                    offending.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")

    assert not offending, (
        "static scan found forbidden implementation imports in tests/e2e/:\n"
        + "\n".join(offending)
    )
