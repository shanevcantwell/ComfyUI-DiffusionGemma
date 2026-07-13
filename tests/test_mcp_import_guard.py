"""`surfaces/mcp/`'s optional-dependency guard (deliverable 5: "a structural
import guard, not a hard requirement" — the ComfyUI surface must keep
loading with zero MCP deps present, mirroring how this pack already treats
ComfyUI itself as absent-by-design in core tests).

Two things this asserts, run out-of-process (subprocess) so a real absent-
`mcp` environment is observed rather than simulated by hiding an
already-imported module from `sys.modules` (which would not catch a
transitive import that runs before the hide takes effect):

1. `surfaces.mcp` (the package init) and `surfaces.mcp.state_manager` import
   with ZERO `mcp` SDK present — neither imports anything from the `mcp`
   package, so their absence-of-crash is a free assertion in THIS
   environment (the real one for CI: `mcp` is an optional extra, not a
   `[project] dependencies` entry, so a bare ComfyUI-only install genuinely
   lacks it).
2. `surfaces.mcp.commands.model` / `.generate` / `.server` (the modules that
   DO need the real SDK to function) raise an actionable `RuntimeError`
   naming the missing 'mcp' optional extra when it's absent — never a bare
   `ModuleNotFoundError` with no context, and never a silent no-op that
   would let a caller think the surface "worked" without the SDK.

This repo's own venv (used by the rest of the suite) has `mcp` installed
(needed for `tests/test_mcp_surface_seam.py` etc. to exercise the real
dispatch path) — so these tests spawn a subprocess with a THROWAWAY
`sys.path`-shadowing shim: a same-named `mcp` directory is never created;
instead the subprocess script inserts a `sitecustomize`-free blocker via
`sys.meta_path` that raises `ModuleNotFoundError` for `mcp`/`mcp.*` before
the real package (if any) is found. This proves the guard's behavior
without needing a second, mcp-less virtualenv.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_BLOCK_MCP_PREAMBLE = """
import sys, importlib.abc, importlib.machinery

class _BlockMCP(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name == "mcp" or name.startswith("mcp."):
            raise ModuleNotFoundError(f"No module named {name!r} (blocked for this test)")
        return None

sys.meta_path.insert(0, _BlockMCP())
"""


def _run(body: str) -> subprocess.CompletedProcess:
    script = _BLOCK_MCP_PREAMBLE + "\n" + body
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def test_surfaces_mcp_package_imports_with_mcp_sdk_absent():
    result = _run(
        "import surfaces.mcp\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_state_manager_imports_with_mcp_sdk_absent():
    result = _run(
        "import surfaces.mcp.state_manager as sm\n"
        "m = sm.StateManager()\n"
        "assert m.is_loaded is False\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout


def test_commands_model_raises_actionable_error_without_mcp_sdk():
    result = _run(
        "try:\n"
        "    import surfaces.mcp.commands.model\n"
        "    print('IMPORTED-UNEXPECTEDLY')\n"
        "except RuntimeError as exc:\n"
        "    assert 'mcp' in str(exc).lower()\n"
        "    assert 'optional' in str(exc).lower()\n"
        "    print('OK')\n"
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
    assert "IMPORTED-UNEXPECTEDLY" not in result.stdout


def test_commands_generate_raises_actionable_error_without_mcp_sdk():
    result = _run(
        "try:\n"
        "    import surfaces.mcp.commands.generate\n"
        "    print('IMPORTED-UNEXPECTEDLY')\n"
        "except RuntimeError as exc:\n"
        "    assert 'mcp' in str(exc).lower()\n"
        "    assert 'optional' in str(exc).lower()\n"
        "    print('OK')\n"
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
    assert "IMPORTED-UNEXPECTEDLY" not in result.stdout


def test_server_raises_actionable_error_without_mcp_sdk():
    result = _run(
        "try:\n"
        "    import surfaces.mcp.server\n"
        "    print('IMPORTED-UNEXPECTEDLY')\n"
        "except RuntimeError as exc:\n"
        "    assert 'mcp' in str(exc).lower()\n"
        "    assert 'optional' in str(exc).lower()\n"
        "    print('OK')\n"
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
    assert "IMPORTED-UNEXPECTEDLY" not in result.stdout


def test_comfyui_surface_still_loads_with_mcp_sdk_absent():
    """The precondition this whole guard exists for: `surfaces.comfyui`
    (and the pack root's own node-registration import) must be entirely
    unaffected by `mcp`'s absence — nothing in the ComfyUI surface imports
    `surfaces.mcp`."""
    result = _run(
        "import surfaces.comfyui.loader\n"
        "import surfaces.comfyui.sampler\n"
        "import surfaces.comfyui.trace\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "OK" in result.stdout
