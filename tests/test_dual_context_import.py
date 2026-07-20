"""Coverage closer for the dotted-package import branch of
`surfaces/comfyui/loader.py`, `surfaces/comfyui/sampler.py`,
`surfaces/comfyui/trace.py`, and `surfaces/comfyui/token_trace.py` (same
dual-context gate; relocated from `nodes/` per ADR-CDG-008 Phase 1, issue
#52) — test-coverage-plan.md Phase 2's precedent.

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
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.loader")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__  # the exact condition loader.py's gate checks
    assert module.DGemmaLoader.FUNCTION == "load"
    assert module.load_model.__module__ == f"{synthetic_pack_root}.dgemma.model"


def test_sampler_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.sampler")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaSampler.FUNCTION == "sample"
    assert module.run_diffusion.__module__ == f"{synthetic_pack_root}.dgemma.loop"


def test_trace_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.trace")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaTrace.FUNCTION == "render"
    # `consumers/` is a top-level pack-root child, exactly like `dgemma/`
    # (ADR-CDG-008 Phase 3 / Open Question #1 resolved to `consumers/`,
    # issue #55 §2) — the relative-import depth from surfaces/comfyui/ is
    # unchanged, only the middle segment moved.
    assert module.build_commit_heatmap.__module__ == f"{synthetic_pack_root}.consumers.analysis"


def test_token_trace_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `surfaces/comfyui/token_trace.py`'s dual-context
    gate (ADR-CDG-014 issue #61 P-D / issue #11) — same shape as trace.py's
    test above, proving the relative climb to
    `consumers.analysis.build_token_identity_grid` resolves under a
    genuinely dotted `__package__`."""
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.token_trace")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaTokenTrace.FUNCTION == "render"
    assert module.build_token_identity_grid.__module__ == f"{synthetic_pack_root}.consumers.analysis"


def test_tally_audit_node_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `surfaces/comfyui/tally_audit.py`'s dual-context
    gate (issue #84) — same shape as the trace.py test above, proving the
    relative climb to `consumers.tally_audit` resolves under a genuinely
    dotted `__package__`."""
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.tally_audit")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaTallyAudit.FUNCTION == "audit"
    assert module.audit_frames.__module__ == f"{synthetic_pack_root}.consumers.tally_audit"


def test_consumers_tally_audit_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `consumers/tally_audit.py`'s own dual-context
    gate (issue #84) — `consumers/` is a top-level pack-root child (same
    depth as `dgemma/`), so importing it directly (not only transitively
    via the surface node above) under a dotted package context proves its
    own relative-import branch independently."""
    module = importlib.import_module(f"{synthetic_pack_root}.consumers.tally_audit")

    assert module.__package__ == f"{synthetic_pack_root}.consumers"
    assert "." in module.__package__
    assert module.DiffusionFrame.__module__ == f"{synthetic_pack_root}.dgemma.types"


def test_consumers_run_log_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `consumers/run_log.py`'s own dual-context gate
    (issue #72), same shape as the tally_audit test above."""
    module = importlib.import_module(f"{synthetic_pack_root}.consumers.run_log")

    assert module.__package__ == f"{synthetic_pack_root}.consumers"
    assert "." in module.__package__
    assert module.CanvasTrace.__module__ == f"{synthetic_pack_root}.dgemma.types"


def test_run_log_writer_node_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `surfaces/comfyui/run_log_writer.py`'s dual-context
    gate (issue #72) — same shape as the tally_audit-node test above,
    proving the relative climb to both `consumers.run_log` and
    `socket_types` resolves under a genuinely dotted `__package__`."""
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.run_log_writer")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaRunLogWriter.FUNCTION == "write"
    assert module.build_run_log_header.__module__ == f"{synthetic_pack_root}.consumers.run_log"


def test_encode_node_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `surfaces/comfyui/encode.py`'s dual-context gate
    (ADR-CDG-012, issue #62 Phase 3) — same shape as loader.py's test above,
    proving the relative climb to `dgemma.kv_cache` resolves under a
    genuinely dotted `__package__`."""
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.encode")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaEncode.FUNCTION == "encode"
    assert module.encode_sequence.__module__ == f"{synthetic_pack_root}.dgemma.kv_cache"


def test_denoise_node_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    """Coverage closer for `surfaces/comfyui/denoise.py`'s dual-context gate
    (ADR-CDG-012, issue #62 Phase 3) — same shape as sampler.py's test above,
    proving the relative climb to `dgemma.loop` resolves under a genuinely
    dotted `__package__`."""
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.comfyui.denoise")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.comfyui"
    assert "." in module.__package__
    assert module.DGemmaDenoise.FUNCTION == "denoise"
    assert module.run_diffusion.__module__ == f"{synthetic_pack_root}.dgemma.loop"
