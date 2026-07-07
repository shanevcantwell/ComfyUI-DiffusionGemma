"""requirements.txt <-> pyproject.toml drift guard (issue #25).

The ComfyUI registry archive ships without a build step: ComfyUI-Manager
installs dependencies via `pip install -r requirements.txt`, never by reading
pyproject.toml's `[project] dependencies`. `requirements.txt` is therefore a
DERIVED artifact (see its own header comment) — this test is the enforcement
surface that keeps it from silently drifting out of sync with the real
dependency list, which would otherwise only be caught at Manager-install
time, on a user's machine, with no CI in between.

Requires `tomllib` (stdlib, Python >=3.11) to parse pyproject.toml; skipped
on 3.10 rather than hard-erroring (repo's own `requires-python = ">=3.10"`).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 11):
    pytest.skip("tomllib is stdlib only on Python >=3.11", allow_module_level=True)

import tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# ComfyUI core already ships these — a Manager-driven pip install of any of
# them here would touch pre-built CUDA wheels (torch/torchvision) or the
# numpy ABI the rest of a ComfyUI install depends on. PEP-503 canonicalized
# (lowercase; this set has no hyphens/underscores/dots to normalize further).
CORE_PROVIDED = {"torch", "torchvision", "numpy", "pillow"}

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _canonicalize(name: str) -> str:
    """PEP 503 name canonicalization: lowercase, runs of -_. collapsed to a
    single '-'. Matches how pip/PyPI compare distribution names."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _bare_name(spec: str) -> str:
    """Strip a PEP 508-ish dependency spec down to its bare distribution
    name, for filtering/comparison — handles version specifiers (==, >=,
    etc.), extras (`pkg[extra]`), and environment markers (`pkg; marker`)."""
    spec = spec.split(";", 1)[0].strip()  # drop environment marker
    match = _NAME_RE.match(spec)
    if not match:
        raise ValueError(f"Could not parse a distribution name from spec: {spec!r}")
    return match.group(1)


def _load_pyproject_deps() -> dict[str, str]:
    """{canonical_name: verbatim_spec} for pyproject.toml's [project]
    dependencies, filtered to drop core-provided packages (torch,
    torchvision, numpy, Pillow — ComfyUI core ships all four)."""
    with open(PYPROJECT_PATH, "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    result: dict[str, str] = {}
    for spec in deps:
        canonical = _canonicalize(_bare_name(spec))
        if canonical in CORE_PROVIDED:
            continue
        result[canonical] = spec.strip()
    return result


def _load_requirements_txt() -> dict[str, str]:
    """{canonical_name: verbatim_spec} for requirements.txt, ignoring blank
    lines and '#' comments."""
    result: dict[str, str] = {}
    for line in REQUIREMENTS_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        canonical = _canonicalize(_bare_name(stripped))
        result[canonical] = stripped
    return result


def test_requirements_txt_matches_core_filtered_pyproject_deps():
    pyproject_deps = _load_pyproject_deps()
    requirements_deps = _load_requirements_txt()

    assert requirements_deps == pyproject_deps, (
        "requirements.txt is DERIVED from pyproject.toml's [project] "
        "dependencies (issue #25) and has drifted out of sync (this "
        "includes version drift — the comparison is spec-for-spec, not "
        "just name-for-name). Update pyproject.toml first, then re-emit "
        "requirements.txt to match."
    )


def test_requirements_txt_core_provided_packages_absent():
    requirements_deps = _load_requirements_txt()
    leaked = set(requirements_deps) & CORE_PROVIDED

    assert not leaked, (
        f"requirements.txt must never list core-provided packages {sorted(leaked)}: "
        "ComfyUI core already ships torch/torchvision/numpy/Pillow, and a "
        "ComfyUI-Manager-driven `pip install -r requirements.txt` touching "
        "any of them risks breaking prebuilt CUDA wheels or the numpy ABI "
        "the rest of the install depends on (issue #25)."
    )
