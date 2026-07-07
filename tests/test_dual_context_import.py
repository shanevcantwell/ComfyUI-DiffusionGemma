"""Coverage closer for the dotted-package import branch of
`nodes/loader.py:19`, `nodes/sampler.py:22`, `nodes/trace.py`, and
`nodes/flipbook.py` (issue #21, same dual-context gate) —
test-coverage-plan.md Phase 2's precedent.

`tests/test_comfyui_loader_context.py` already proves this branch executes
correctly under ComfyUI's real loader mechanics — but it does so in a
subprocess (deliberately, per that file's own docstring, to guarantee a
fresh `sys.modules`), so pytest-cov — in-process only here, no
`COVERAGE_PROCESS_START` wired for subprocesses — never credits those two
lines. This test imports the same two modules a second time, in-process, as
submodules of a synthetic top-level package whose `__path__` points at the
repo root. That gives each module a genuinely dotted `__package__`
(`"<synthetic>.nodes"`) via ordinary import machinery — no subprocess, no
`exec_module` hand-rolling — so coverage.py sees the relative-import branch
run.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_PKG_NAME = "_dgemma_dual_context_probe"


@pytest.fixture
def synthetic_pack_root():
    """Registers a synthetic top-level package whose `__path__` is the repo
    root — mirroring what ComfyUI's loader effectively gives the pack,
    minus the hyphenated name — so `X.nodes.loader`/`X.dgemma.model` resolve
    via ordinary `PathFinder` lookups against `REPO_ROOT`. Tears down every
    module this pulled into `sys.modules` afterward so the probe never
    leaks into other tests' import state.
    """
    pkg = types.ModuleType(SYNTHETIC_PKG_NAME)
    pkg.__path__ = [str(REPO_ROOT)]
    sys.modules[SYNTHETIC_PKG_NAME] = pkg
    before = set(sys.modules)
    try:
        yield SYNTHETIC_PKG_NAME
    finally:
        for name in set(sys.modules) - before:
            del sys.modules[name]
        sys.modules.pop(SYNTHETIC_PKG_NAME, None)


def test_loader_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.nodes.loader")

    assert module.__package__ == f"{synthetic_pack_root}.nodes"
    assert "." in module.__package__  # the exact condition nodes/loader.py:18 gates on
    assert module.DGemmaLoader.FUNCTION == "load"
    assert module.load_model.__module__ == f"{synthetic_pack_root}.dgemma.model"


def test_sampler_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.nodes.sampler")

    assert module.__package__ == f"{synthetic_pack_root}.nodes"
    assert "." in module.__package__
    assert module.DGemmaSampler.FUNCTION == "sample"
    assert module.run_diffusion.__module__ == f"{synthetic_pack_root}.dgemma.loop"


def test_trace_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.nodes.trace")

    assert module.__package__ == f"{synthetic_pack_root}.nodes"
    assert "." in module.__package__
    assert module.DGemmaTrace.FUNCTION == "render"
    assert module.build_commit_heatmap.__module__ == f"{synthetic_pack_root}.dgemma.sampling"


def test_flipbook_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.nodes.flipbook")

    assert module.__package__ == f"{synthetic_pack_root}.nodes"
    assert "." in module.__package__
    assert module.DGemmaFlipbook.FUNCTION == "render"
    assert module.decode_frames.__module__ == f"{synthetic_pack_root}.dgemma.loop"
