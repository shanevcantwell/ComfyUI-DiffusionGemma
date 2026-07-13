"""Direct, in-process unit test for `surfaces/mcp/_mcp_sdk_guard.require_mcp_sdk`'s
except branch — the coverage closer for `tests/test_mcp_import_guard.py`'s
subprocess-only exercise of the same code path (subprocess coverage isn't
credited by an in-process `pytest-cov` run, per that module's own docstring).

This simulates "`mcp` genuinely absent" directly (a `sys.modules` entry set
to `None`, the standard `importlib` idiom for forcing the next `import mcp`
to raise `ModuleNotFoundError` without actually uninstalling the package —
same trick a real absent-package environment triggers organically), rather
than the subprocess-level `sys.meta_path` blocker
`test_mcp_import_guard.py` uses (which proves the real cross-process
behavior; this proves the exact line-level branch cheaply).
"""
from __future__ import annotations

import sys

import pytest

from surfaces.mcp._mcp_sdk_guard import require_mcp_sdk


def test_require_mcp_sdk_raises_actionable_runtime_error_when_mcp_absent(monkeypatch):
    monkeypatch.setitem(sys.modules, "mcp", None)  # forces `import mcp` to raise ImportError

    with pytest.raises(RuntimeError) as exc_info:
        require_mcp_sdk()

    message = str(exc_info.value).lower()
    assert "mcp" in message
    assert "optional" in message
    assert exc_info.value.__cause__ is not None  # `raise ... from exc` preserved


def test_require_mcp_sdk_returns_the_real_module_when_present():
    result = require_mcp_sdk()
    assert result.__name__ == "mcp"
