"""Subprocess coverage-measurement hook (ADR-CDG-013 §3, issue #59-§3).

Python imports a module named `sitecustomize` automatically at interpreter
startup if one is importable on `sys.path` — this is the documented
mechanism `coverage.process_startup()` (coverage.py's multiprocess/subprocess
support) rides. The E2E battery's headless-ComfyUI server-launch fixture
(`tests/e2e/conftest.py`) puts this repo root on the subprocess's
`PYTHONPATH` (the same entry it adds so ComfyUI can import the pack itself)
and sets `COVERAGE_PROCESS_START` in that subprocess's env, so this file:

1. is importable the instant the server subprocess's interpreter boots —
   before ComfyUI imports the node pack — so measurement starts from the
   pack's very first line, not from whenever `dgemma`/`surfaces`/`consumers`
   happens to get imported;
2. is a genuine no-op everywhere else. If `COVERAGE_PROCESS_START` is unset
   (every other process on this box: the pytest process itself, any human
   running `python` with this repo on `PYTHONPATH`, etc.), `coverage.
   process_startup()` does nothing per coverage.py's own contract. This file
   never starts measurement on its own initiative.

Why a subprocess mechanism at all, not in-process `pytest-cov`: the
node-pack code under an `e2e` run executes inside the ComfyUI *server*
subprocess, not the pytest process — in-process `pytest-cov` would see zero
node-pack lines from a black-box run. See ADR-CDG-013 Decision 4 / Option B
(rejected) for the full reasoning, including why this sidesteps issue #50's
pytest-cov + torch C-tracer flake.
"""
from __future__ import annotations

try:
    import coverage
except ImportError:
    # coverage is a `dev` extra (pyproject.toml), not a runtime dependency of
    # the node pack — an interpreter that boots this file without it
    # installed (e.g. a plain end-user ComfyUI install with no `dev` extras)
    # must still start up cleanly. Absence here is not an error.
    pass
else:
    coverage.process_startup()
