#!/usr/bin/env python3
"""Derive requirements.txt from pyproject.toml's [project] dependencies.

requirements.txt is a DERIVED artifact (see its own header comment and
issue #25): the ComfyUI registry archive ships without a build step, so
ComfyUI-Manager installs dependencies via `pip install -r requirements.txt`
against that file directly, never by reading pyproject.toml. pyproject.toml
stays the mint (the one place a human adds/bumps a dependency); this script
makes regenerating the derived file mechanical instead of hand-edited.

CORE_PROVIDED is the ONE-MINT definition (issue #146): ComfyUI core already
ships torch, torchvision, numpy, and Pillow, so a Manager-driven
`pip install -r requirements.txt` touching any of them risks breaking
prebuilt CUDA wheels or the numpy ABI the rest of a ComfyUI install depends
on. `tests/test_requirements_sync.py` imports this set from here rather than
redefining it, so there is exactly one place that names "what ComfyUI core
supplies."

Usage:
    python scripts/emit_requirements.py            # write requirements.txt
    python scripts/emit_requirements.py --check     # exit 1 if it would change anything (no write)

Not a commit hook (deliberately, per issue #146): drift is caught by
`tests/test_requirements_sync.py` in CI (fail-loud), and regeneration is a
deliberate, reviewed act, not an automatic one.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"

# ComfyUI core already ships these — a Manager-driven pip install of any of
# them here would touch pre-built CUDA wheels (torch/torchvision) or the
# numpy ABI the rest of a ComfyUI install depends on. PEP-503 canonicalized
# (lowercase; this set has no hyphens/underscores/dots to normalize further).
CORE_PROVIDED = {"torch", "torchvision", "numpy", "pillow"}

_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")

_HEADER = """\
# requirements.txt — DERIVED from pyproject.toml [project] dependencies.
# Do not hand-edit this file directly; edit pyproject.toml and re-emit/verify
# instead (enforced by tests/test_requirements_sync.py, which fails the build
# the moment the two drift apart).
#
# Why this file exists at all (issue #25): the ComfyUI registry archive that
# ComfyUI-Manager downloads has no build step — Manager installs dependencies
# by running `pip install -r requirements.txt` against this file, not by
# reading pyproject.toml's [project] dependencies. Without this file, the
# pack ships with zero dependencies installed on a fresh Manager install.
#
# torch, torchvision, numpy, and Pillow are intentionally OMITTED even though
# they are real [project] dependencies: ComfyUI core already ships all four,
# and a Manager-driven pip install of them here would touch pre-built CUDA
# wheels (torch/torchvision) and the numpy ABI that the rest of a ComfyUI
# install depends on — a wheel/ABI hazard, not a missing dependency. See
# tests/test_requirements_sync.py for the enforced core-filter set.
"""


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


def _load_pyproject_deps() -> list[str]:
    """Verbatim [project].dependencies specs, filtered to drop core-provided
    packages, in pyproject.toml's own declared order."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - repo's own requires-python is >=3.10
        import tomli as tomllib  # type: ignore[no-redef]

    with open(PYPROJECT_PATH, "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    result = []
    for spec in deps:
        canonical = _canonicalize(_bare_name(spec))
        if canonical in CORE_PROVIDED:
            continue
        result.append(spec.strip())
    return result


def render_requirements_txt() -> str:
    specs = _load_pyproject_deps()
    return _HEADER + "\n".join(specs) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if requirements.txt would change; do not write.",
    )
    args = parser.parse_args()

    rendered = render_requirements_txt()

    if args.check:
        current = REQUIREMENTS_PATH.read_text() if REQUIREMENTS_PATH.exists() else ""
        if current != rendered:
            print("requirements.txt is out of sync with pyproject.toml.", file=sys.stderr)
            return 1
        print("requirements.txt is in sync.")
        return 0

    REQUIREMENTS_PATH.write_text(rendered)
    print(f"Wrote {REQUIREMENTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
