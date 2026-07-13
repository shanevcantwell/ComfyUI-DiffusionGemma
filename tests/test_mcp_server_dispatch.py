"""Integration tests for `surfaces/mcp/server.py`'s dispatch layer —
`list_tools` aggregation and `call_tool` name-based routing (the sk-mcp
`server.py:33-42,45-94` shape this module transcribes).

Requires the real `mcp` SDK (an optional extra — see
`tests/test_mcp_import_guard.py` for the absent-SDK guard tests); skipped
cleanly if unavailable so the rest of the suite (which doesn't need it)
stays green in an environment without the optional extra installed.
"""
from __future__ import annotations

import asyncio
import json

import pytest

mcp = pytest.importorskip("mcp", reason="optional 'mcp' extra not installed")

from surfaces.mcp import server as server_module  # noqa: E402
from surfaces.mcp.state_manager import StateManager  # noqa: E402
from tests.test_run_diffusion_statelessness import _fake_model  # noqa: E402


def test_list_tools_returns_all_four_registered_tools():
    tools = asyncio.run(server_module.list_tools())
    names = {t.name for t in tools}
    assert names == {"load_model", "model_status", "generate", "cancel_run"}


def test_call_tool_unknown_name_returns_structured_error():
    results = asyncio.run(server_module.call_tool("no_such_tool", {}))
    assert len(results) == 1
    payload = json.loads(results[0].text)
    assert "Unknown tool" in payload["error"]


def test_call_tool_model_status_routes_to_the_model_status_handler(monkeypatch):
    """Swap the module-level `state_manager` for a fresh one with a known
    status, confirm `call_tool("model_status", {})` reports exactly that —
    proving the dispatch table actually reaches the handler, not just that
    the handler works in isolation (already covered by
    `tests/test_mcp_model_command.py`)."""
    fresh_manager = StateManager()
    monkeypatch.setattr(server_module, "state_manager", fresh_manager)

    results = asyncio.run(server_module.call_tool("model_status", {}))
    payload = json.loads(results[0].text)
    assert payload == fresh_manager.status()
    assert payload["is_loaded"] is False


def test_call_tool_load_model_missing_args_returns_error_without_crashing(monkeypatch):
    fresh_manager = StateManager()
    monkeypatch.setattr(server_module, "state_manager", fresh_manager)

    results = asyncio.run(server_module.call_tool("load_model", {}))
    payload = json.loads(results[0].text)
    assert "error" in payload
    assert fresh_manager.is_loaded is False


def test_call_tool_swallows_handler_exceptions_as_structured_error(monkeypatch):
    """A handler bug must never crash the server process (same posture as
    sk-mcp's own `server.py:90-94`) — it becomes a JSON `{"error": ...}`
    payload instead."""

    async def _boom(manager, args):
        raise ValueError("synthetic handler failure")

    monkeypatch.setitem(server_module._HANDLERS, "generate", _boom)

    results = asyncio.run(server_module.call_tool("generate", {"prompt": "hi"}))
    payload = json.loads(results[0].text)
    assert payload == {"error": "synthetic handler failure"}


def test_run_server_drives_stdio_server_and_server_run(monkeypatch):
    """`run_server` is a thin two-line wrap: enter `stdio_server()`, hand
    the streams to `server.run(...)`. Faked here (no real stdio process) to
    confirm the wiring — that `server.run` is actually invoked with the
    streams `stdio_server` yields plus the server's own init options."""
    calls = {}

    class _FakeStdioCtx:
        async def __aenter__(self):
            return ("READ", "WRITE")

        async def __aexit__(self, *exc):
            return False

    async def _fake_server_run(read_stream, write_stream, init_options):
        calls["args"] = (read_stream, write_stream, init_options)

    monkeypatch.setattr(server_module, "stdio_server", lambda: _FakeStdioCtx())
    monkeypatch.setattr(server_module.server, "run", _fake_server_run)
    monkeypatch.setattr(
        server_module.server, "create_initialization_options", lambda: "INIT_OPTS"
    )

    asyncio.run(server_module.run_server())

    assert calls["args"] == ("READ", "WRITE", "INIT_OPTS")


def test_main_drives_run_server_via_asyncio_run(monkeypatch):
    """`main()` is the process entry point — confirm it actually calls
    `run_server()` via `asyncio.run`, not just that `run_server` itself
    works in isolation (covered above)."""
    called = {}

    async def _fake_run_server():
        called["ran"] = True

    monkeypatch.setattr(server_module, "run_server", _fake_run_server)
    server_module.main()

    assert called == {"ran": True}
