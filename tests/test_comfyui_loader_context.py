"""Enforcement test for ComfyUI's real custom-node loader context.

Observed violation (graph smoke test, 2026-07-05, `loose-ends.md`): ComfyUI
puts `custom_nodes/` on sys.path — never the pack's own root — so a bare
`from dgemma...` import inside `nodes/*.py` raised ModuleNotFoundError under
the real loader while every in-repo test stayed green (pytest runs with the
repo root on sys.path, which masks exactly this).

This test simulates ComfyUI's actual load path, mirroring
`/srv/dev/ComfyUI/nodes.py:2226-2246` mechanics precisely:

- module name = the directory path with `.` → `_x_` (`nodes.py:2233`) — an
  absolute path string containing a hyphen (`ComfyUI-DiffusionGemma`), not an
  identifier;
- `spec_from_file_location(name, <dir>/__init__.py)` with NO explicit
  `submodule_search_locations` (`nodes.py:2241`; importlib infers
  package-ness from the `__init__.py` filename);
- registered in `sys.modules` before `exec_module` (`nodes.py:2245-2246`);
- and the load-bearing condition this suite previously lacked: the repo root
  is STRIPPED from sys.path for the duration (subprocess with a non-repo
  cwd), so any bare absolute `dgemma`/`nodes` import fails instead of being
  silently satisfied by pytest's sys.path.

Run out-of-process for the same reason as `test_seam.py`: a fresh interpreter
is the only way to guarantee no already-cached `dgemma`/`nodes` module in
`sys.modules` masks a broken import.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

_LOADER_SCRIPT = """
import importlib.util, os, sys

module_path = {module_path!r}

# The condition that catches bare absolute imports: the pack root must NOT be
# importable from sys.path (ComfyUI adds custom_nodes/, never the pack root).
repo = os.path.abspath(module_path)
sys.path[:] = [p for p in sys.path if os.path.abspath(p or os.getcwd()) != repo]

# Mirror /srv/dev/ComfyUI/nodes.py:2233,2241,2244-2246 exactly.
sys_module_name = module_path.replace(".", "_x_")
spec = importlib.util.spec_from_file_location(
    sys_module_name, os.path.join(module_path, "__init__.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules[sys_module_name] = module
spec.loader.exec_module(module)

mappings = module.NODE_CLASS_MAPPINGS
assert set(mappings) == {{"DGemmaLoader", "DGemmaSampler", "DGemmaTrace"}}, sorted(mappings)
assert all(isinstance(cls, type) for cls in mappings.values())
assert set(module.NODE_DISPLAY_NAME_MAPPINGS) == set(mappings)

# P3 (a): WEB_DIRECTORY must be present and resolve to a real directory
# relative to the pack's own root — `nodes.py:2269-2272`'s own check.
assert isinstance(module.WEB_DIRECTORY, str)
web_dir = os.path.join(module_path, module.WEB_DIRECTORY)
assert os.path.isdir(web_dir), web_dir
print("OK")
"""


def test_pack_loads_under_comfyui_loader_mechanics(tmp_path):
    """The pack's node mappings must resolve when loaded exactly the way
    ComfyUI loads a custom-node directory, with the repo root absent from
    sys.path."""
    result = subprocess.run(
        [sys.executable, "-c", _LOADER_SCRIPT.format(module_path=str(REPO_ROOT))],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # non-repo cwd: '' on sys.path must not resolve to the pack root
    )
    assert result.returncode == 0, (
        f"ComfyUI-context load failed:\n{result.stdout}\n{result.stderr}"
    )
    assert "OK" in result.stdout
