"""Coverage closer for the dotted-package (ComfyUI-loader-shaped) import
branch of `surfaces/mcp/state_manager.py`, `surfaces/mcp/commands/model.py`,
and `surfaces/mcp/commands/generate.py` — same technique as
`tests/test_dual_context_import.py` (that module's own docstring explains
the rationale in full: `tests/test_comfyui_loader_context.py` already proves
the real ComfyUI loader mechanics work, out-of-process; this closes the
in-process coverage gap that leaves, via a synthetic dotted parent package).

Honest scope note (unlike the ComfyUI surface's own dual-context modules):
`surfaces/mcp/` is never actually reached through ComfyUI's directory loader
in production — nothing in the pack's root `__init__.py` imports
`surfaces.mcp` (verified, `tests/test_mcp_import_guard.py`'s
ComfyUI-surface-still-loads test). The dual-context gates in these three
modules exist for STRUCTURAL CONSISTENCY with the established
`surfaces/comfyui/*.py` idiom (in case a future refactor ever does load
`surfaces/mcp/` through that path, or vendors this surface elsewhere with
the same depth), not because a live production path exercises them today.
This test proves the branch is at least correct, not that it's reachable in
today's actual deployment.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_PKG_NAME = "_dgemma_mcp_dual_context_probe"


@pytest.fixture
def synthetic_pack_root():
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


def test_state_manager_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.mcp.state_manager")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.mcp"
    assert module.__package__.count(".") >= 2  # the exact condition the gate checks
    assert module.load_model.__module__ == f"{synthetic_pack_root}.dgemma.model"


def test_commands_model_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.mcp.commands.model")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.mcp.commands"
    assert module.__package__.count(".") >= 3
    assert module.StateManager.__module__ == f"{synthetic_pack_root}.surfaces.mcp.state_manager"


def test_commands_generate_resolves_relative_import_under_dotted_package_context(synthetic_pack_root):
    module = importlib.import_module(f"{synthetic_pack_root}.surfaces.mcp.commands.generate")

    assert module.__package__ == f"{synthetic_pack_root}.surfaces.mcp.commands"
    assert module.__package__.count(".") >= 3
    assert module.run_diffusion.__module__ == f"{synthetic_pack_root}.dgemma.loop"
    assert module.StateManager.__module__ == f"{synthetic_pack_root}.surfaces.mcp.state_manager"
