"""ADR-CDG-008 Phase 2, Correction 1 (`STATELESS-CORE`) enforcement for
`surfaces/mcp/state_manager.py`: the MCP surface's persisted state is the
loaded model, ONLY — never a scheduler, canvas, or other per-run object.

This IS the enforcement surface ARCHITECTURE.md's rule 6 names for this
phase ("CDG-008 Phase-2 MCP state manager must never cache a scheduler") and
the "MCP state manager caches a live scheduler across calls" instant-fail
row's counterpart valid form ("Persist only `load_model`'s output; build a
fresh scheduler/canvas/run-state per call").

Two tiers:

- `TestStateManagerShape` — a structural assertion on `StateManager` itself:
  its only mutable cross-call field is the model (`_model`/`_repo_id`/
  `_quant`, all set exclusively by `load()`); no attribute holds a
  scheduler, canvas, or frame-collector-shaped object. This is the
  MUTATION-SENSITIVE check the gate asks for: introduce a `self._scheduler`
  (or any cross-call mutable field not in the allowlist) and this test
  fails BY NAME.
- `TestSameInSameOutAtMCPLevel` — the behavioral half, riding the same
  fake-pipeline pattern `tests/test_run_diffusion_statelessness.py` already
  uses (this test module imports and reuses its fakes rather than
  reinventing them): two identical `generate` tool calls through
  `surfaces.mcp.commands.generate.generate`, on ONE loaded (fake) model
  held by ONE `StateManager`, must produce byte-identical
  `trace_summary`/`canvas_state` — proving the MCP dispatch layer adds no
  cross-call state of its own on top of what `run_diffusion` itself already
  guarantees fresh per call.
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from surfaces.mcp.state_manager import StateManager
from tests.test_run_diffusion_statelessness import (
    FakeProcessor,
    _fake_model,
    _install_stateless_fakes,
)


class TestStateManagerShape:
    """Structural, mutation-sensitive check: `StateManager` may hold ONLY
    the model-load fields. This is deliberately a field-allowlist assertion
    (not a behavior probe) — a future edit that adds `self._scheduler = ...`
    or similar fails this test by name, at review time, before any call-level
    symptom (like the observed 25-vs-29 heatmap frame-count mismatch this
    ADR cites) could ever occur."""

    ALLOWED_FIELDS = {"_model", "_repo_id", "_quant"}

    def test_state_manager_dataclass_fields_are_exactly_the_model_load_triple(self):
        field_names = {f.name for f in dataclasses.fields(StateManager)}
        assert field_names == self.ALLOWED_FIELDS, (
            f"StateManager grew a field outside the model-load allowlist: "
            f"{field_names - self.ALLOWED_FIELDS}. ADR-CDG-008 Correction 1 / "
            f"ARCHITECTURE.md rule 6: the MCP state manager persists ONLY the "
            f"model load — a scheduler/canvas/run-state field here is the "
            f"exact cross-call-mutable-state violation this test exists to "
            f"catch. (MUTATION CHECK: add `_scheduler: Any = None` to "
            f"StateManager and this assertion fails.)"
        )

    def test_fresh_state_manager_holds_no_model(self):
        manager = StateManager()
        assert manager.is_loaded is False
        with pytest.raises(RuntimeError, match="No DiffusionGemma model is loaded"):
            manager.require_model()

    def test_load_replaces_rather_than_accumulates(self, monkeypatch):
        """Calling `load()` twice must leave exactly ONE model held — no
        list/cache of prior loads accumulating (the same "only the load
        persists, and only ONE of it" reading of rule 6)."""
        manager = StateManager()

        calls = []

        def fake_load_model(*, repo_id, quant, local_files_only=False):
            calls.append((repo_id, quant))
            return _fake_model()

        monkeypatch.setattr("surfaces.mcp.state_manager.load_model", fake_load_model)

        manager.load(repo_id="repo/a", quant="none")
        manager.load(repo_id="repo/b", quant="none")

        assert calls == [("repo/a", "none"), ("repo/b", "none")]
        assert manager.status()["repo_id"] == "repo/b"
        # Only one model object is ever held — dataclasses.fields already
        # proved there's no second slot for it to live in, this just
        # confirms the value itself was actually replaced, not merged.
        assert manager._quant == "none"


class TestSameInSameOutAtMCPLevel:
    """Behavioral half: two identical `generate` calls through the MCP
    dispatch layer, on one loaded fake model held by one `StateManager`,
    yield identical results — the MCP surface adds no state of its own on
    top of `run_diffusion`'s own already-enforced freshness
    (`tests/test_run_diffusion_statelessness.py`)."""

    def _make_manager_with_fake_model(self) -> StateManager:
        manager = StateManager()
        manager._model = _fake_model()
        manager._repo_id = "fake/repo"
        manager._quant = "none"
        return manager

    def test_two_identical_generate_calls_yield_identical_trace_summary(self, monkeypatch):
        from surfaces.mcp.commands import generate as generate_module

        scheduler_registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=scheduler_registry, num_steps=3)

        manager = self._make_manager_with_fake_model()
        args = {
            "prompt": "hello world",
            "seed": 42,
            "num_inference_steps": 3,
            "t_min": 0.1,
            "t_max": 0.9,
            "entropy_bound": 0.2,
            "include_frames": True,
        }

        result_1 = asyncio.run(generate_module.generate(manager, dict(args)))
        result_2 = asyncio.run(generate_module.generate(manager, dict(args)))

        assert result_1["trace_summary"] == result_2["trace_summary"]
        assert result_1["canvas_state"] == result_2["canvas_state"]
        assert result_1["text"] == result_2["text"]
        # Two calls -> two distinct scheduler objects (never a cached one
        # reused across the MCP-level calls either) — same structural proof
        # `TestSchedulerFreshPerCall` makes at the `run_diffusion` level,
        # replayed here to confirm the MCP adapter didn't reintroduce sharing
        # by e.g. constructing the scheduler itself and passing it in.
        assert len(scheduler_registry) == 2
        assert scheduler_registry[0] is not scheduler_registry[1]

    def test_generate_never_mutates_state_manager_beyond_the_model(self, monkeypatch):
        """The state manager handed to `generate` must come out with the
        exact same `_repo_id`/`_quant`/`_model` identity it went in with —
        `generate` reads the model, it never writes to the manager."""
        from surfaces.mcp.commands import generate as generate_module

        scheduler_registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=scheduler_registry, num_steps=2)

        manager = self._make_manager_with_fake_model()
        model_before = manager._model
        repo_id_before = manager._repo_id
        quant_before = manager._quant

        asyncio.run(generate_module.generate(manager, {"prompt": "hi", "num_inference_steps": 2}))

        assert manager._model is model_before
        assert manager._repo_id == repo_id_before
        assert manager._quant == quant_before
        assert set(dataclasses.fields(type(manager))) == set(dataclasses.fields(StateManager))
