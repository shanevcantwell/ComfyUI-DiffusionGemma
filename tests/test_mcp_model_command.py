"""Unit tests for `surfaces/mcp/commands/model.py` — `load_model` /
`model_status`.

Focus: Rule #14, carried across from `semantic-kinematics-mcp`'s
`commands/embeddings.py` (`tests/test_embeddings_command.py` in that repo) —
no baked-in repo_id/quant default. Invoking `load_model` without both args
explicit must NOT silently fall through to `dgemma.model.load_model`'s own
grounded defaults (`DEFAULT_REPO_ID`/`DEFAULT_QUANT`); it must return a
structured error naming what is missing, and — the loud half of the gate —
never reach the (arbitrarily expensive, ~53GB) real loader.
"""
from __future__ import annotations

import asyncio

import pytest

from surfaces.mcp.commands import model as model_module
from surfaces.mcp.state_manager import StateManager


class _ExplodingStateManager(StateManager):
    """If the handler reaches `StateManager.load` (and therefore
    `dgemma.model.load_model`) despite a missing repo_id/quant, that is the
    silent-default bug Rule #14 forbids; calling `load` makes it loud."""

    def load(self, *, repo_id, quant, local_files_only=False):
        raise AssertionError(
            "StateManager.load must NOT be called when repo_id/quant is "
            "missing (would silently load with dgemma's own default)"
        )


def _run(coro):
    return asyncio.run(coro)


def test_load_model_without_repo_id_returns_structured_error():
    result = _run(model_module.load_model_tool(_ExplodingStateManager(), {"quant": "none"}))
    assert "error" in result
    assert "repo_id" in result["error"]
    assert "status" not in result


def test_load_model_without_quant_returns_structured_error():
    result = _run(model_module.load_model_tool(_ExplodingStateManager(), {"repo_id": "org/model"}))
    assert "error" in result
    assert "quant" in result["error"]
    assert "status" not in result


def test_load_model_with_empty_repo_id_returns_structured_error():
    result = _run(
        model_module.load_model_tool(_ExplodingStateManager(), {"repo_id": "", "quant": "none"})
    )
    assert "error" in result
    assert "repo_id" in result["error"]


def test_load_model_schema_requires_repo_id_and_quant_with_no_defaults():
    tools = {t.name: t for t in model_module.get_tools()}
    schema = tools["load_model"].inputSchema
    assert set(schema["required"]) == {"repo_id", "quant"}
    assert "default" not in schema["properties"]["repo_id"]
    assert "default" not in schema["properties"]["quant"]


def test_load_model_with_both_args_reaches_state_manager(monkeypatch):
    """With both args supplied, the handler proceeds — confirmed by
    observing the (fake) StateManager.load actually gets called, and its
    result is passed through."""
    calls = []

    class _FakeModel:
        device = "cpu"
        dtype = "bfloat16"

    class _RecordingStateManager(StateManager):
        def load(self, *, repo_id, quant, local_files_only=False):
            calls.append((repo_id, quant, local_files_only))
            return _FakeModel()

    result = _run(
        model_module.load_model_tool(
            _RecordingStateManager(), {"repo_id": "org/model", "quant": "none"}
        )
    )
    assert calls == [("org/model", "none", False)]
    assert result["status"] == "loaded"
    assert result["repo_id"] == "org/model"
    assert result["quant"] == "none"
    assert result["device"] == "cpu"


def test_model_status_reports_manager_status_verbatim():
    manager = StateManager()
    result = _run(model_module.model_status_tool(manager, {}))
    assert result == manager.status()
    assert result["is_loaded"] is False
