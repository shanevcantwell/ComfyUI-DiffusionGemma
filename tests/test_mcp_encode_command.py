"""Unit tests for `surfaces/mcp/commands/encode.py` — the `encode` tool
(ADR-CDG-025).

Two altitudes, mirroring `tests/test_mcp_generate_command.py`'s own split:

- `TestEncodeThinAdapter`/`TestEncodeSchema` — the tool body itself, against
  `dgemma_model_factory` (`tests/conftest.py` §L), the SAME fixture
  `tests/test_kv_cache_ingress.py`/this module's sibling
  `tests/test_mcp_statelessness.py::TestKVCacheRegistryLifecycle` already use
  to drive a real (cheap, CPU-only) `dgemma.kv_cache.encode_sequence` call —
  no re-hosted fixture, no real weights.
- `TestEncodeGenerateHandleRoundTrip` — the cross-module contract: a handle
  minted by `encode` is resolvable by `generate`'s `kv_cache_id` door, and
  the EXACT `KVCache` object `encode` stored is what reaches
  `run_diffusion(kv_cache=...)`. `run_diffusion` itself is monkeypatched to a
  recording fake here (not the heavier decode-capable fixture
  `tests/test_kv_cache_drive_body.py` owns) — this module's job is proving
  the MCP-layer plumbing hands the resolved object through unmutated, not
  re-proving the engine's own with-cache drive body, which that other
  module's fixtures already cover.
"""
from __future__ import annotations

import asyncio

import pytest

from surfaces.mcp.commands import encode as encode_module
from surfaces.mcp.commands import generate as generate_module
from surfaces.mcp.state_manager import StateManager


def _manager_with(model) -> StateManager:
    manager = StateManager()
    manager._model = model
    manager._repo_id = model.repo_id
    manager._quant = "none"
    return manager


class TestEncodeSchema:
    def test_encode_schema_requires_only_prompt(self):
        tools = {t.name: t for t in encode_module.get_tools()}
        schema = tools["encode"].inputSchema
        assert schema["required"] == ["prompt"]
        assert set(schema["properties"]) == {"prompt", "kv_cache_id"}


class TestEncodeThinAdapter:
    def test_encode_without_prompt_returns_structured_error(self, dgemma_model_factory):
        manager = _manager_with(dgemma_model_factory())
        result = asyncio.run(encode_module.encode(manager, {}))
        assert "error" in result
        assert "prompt" in result["error"]

    def test_encode_without_loaded_model_raises_actionable_error(self):
        manager = StateManager()  # no model loaded
        with pytest.raises(RuntimeError, match="No DiffusionGemma model is loaded"):
            asyncio.run(encode_module.encode(manager, {"prompt": "hi"}))

    def test_encode_fresh_mint_returns_a_kv_cache_id(self, dgemma_model_factory):
        manager = _manager_with(dgemma_model_factory())
        result = asyncio.run(encode_module.encode(manager, {"prompt": "hello"}))
        assert set(result) == {"kv_cache_id"}
        assert isinstance(result["kv_cache_id"], str) and result["kv_cache_id"]
        assert result["kv_cache_id"] in manager._kv_cache_registry

    def test_encode_advance_reuses_the_same_handle(self, dgemma_model_factory):
        manager = _manager_with(dgemma_model_factory())
        mint = asyncio.run(encode_module.encode(manager, {"prompt": "first"}))

        advance = asyncio.run(
            encode_module.encode(manager, {"prompt": "second", "kv_cache_id": mint["kv_cache_id"]})
        )

        assert advance["kv_cache_id"] == mint["kv_cache_id"]
        assert len(manager._kv_cache_registry) == 1

    def test_encode_with_unknown_kv_cache_id_raises_value_error(self, dgemma_model_factory):
        manager = _manager_with(dgemma_model_factory())
        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            asyncio.run(encode_module.encode(manager, {"prompt": "hi", "kv_cache_id": "bogus"}))

    def test_encode_invalid_kv_cache_id_surfaces_through_server_call_tool_as_structured_error(
        self, monkeypatch, dgemma_model_factory
    ):
        """Same posture `test_mcp_generate_command.py`'s
        `TestInvalidPayloadSurfacesCoreIngressError` proves for `generate`:
        `surfaces.mcp.server.call_tool`'s outer try/except turns the
        `ValueError` into a structured `{"error": ...}` payload, never an
        unhandled exception at the transport layer."""
        import json

        pytest.importorskip("mcp", reason="optional 'mcp' extra not installed")
        from surfaces.mcp import server as server_module

        manager = _manager_with(dgemma_model_factory())
        monkeypatch.setattr(server_module, "state_manager", manager)

        result = asyncio.run(
            server_module.call_tool("encode", {"prompt": "hi", "kv_cache_id": "bogus"})
        )
        payload = json.loads(result[0].text)
        assert "error" in payload
        assert "Unknown or expired kv_cache_id" in payload["error"]


class TestEncodeGenerateHandleRoundTrip:
    """The cross-tool contract ADR-CDG-025 exists to enable: `encode`'s
    output handle is exactly what `generate`'s `kv_cache_id` door resolves
    and hands to `run_diffusion(kv_cache=...)`."""

    def _install_recording_run_diffusion(self, monkeypatch):
        calls: list[dict] = []

        def fake_run_diffusion(dgemma_model, prompt, **kwargs):
            calls.append({"prompt": prompt, "kv_cache": kwargs.get("kv_cache")})

            class _FakeCanvasState:
                converged = True
                committed_fraction = 1.0
                steps_used = 1
                thought = None
                stray_thought_delimiter = None
                turn_closed = True
                answer_tokens = 1
                finished_honestly = True

            class _FakeCanvasTrace:
                scheduler_name = "fake"
                scheduler_config = {}
                frames: list = []

            return "TEXT", _FakeCanvasState(), _FakeCanvasTrace()

        monkeypatch.setattr(generate_module, "run_diffusion", fake_run_diffusion)
        return calls

    def test_generate_resolves_the_handle_encode_minted_and_passes_the_same_object(
        self, monkeypatch, dgemma_model_factory
    ):
        model = dgemma_model_factory()
        manager = _manager_with(model)
        calls = self._install_recording_run_diffusion(monkeypatch)

        mint = asyncio.run(encode_module.encode(manager, {"prompt": "donor context"}))
        handle = mint["kv_cache_id"]
        stored_cache = manager._kv_cache_registry[handle]

        result = asyncio.run(
            generate_module.generate(manager, {"prompt": "current turn", "kv_cache_id": handle})
        )

        assert "error" not in result
        assert len(calls) == 1
        assert calls[0]["kv_cache"] is stored_cache
        assert calls[0]["prompt"] == "current turn"

    def test_generate_without_kv_cache_id_passes_kv_cache_none(self, monkeypatch, dgemma_model_factory):
        """Omitting kv_cache_id must be byte-identical to before this
        widening: `run_diffusion` still receives `kv_cache=None`."""
        manager = _manager_with(dgemma_model_factory())
        calls = self._install_recording_run_diffusion(monkeypatch)

        asyncio.run(generate_module.generate(manager, {"prompt": "hi"}))

        assert len(calls) == 1
        assert calls[0]["kv_cache"] is None

    def test_generate_with_unknown_kv_cache_id_raises_before_run_diffusion_is_called(
        self, monkeypatch, dgemma_model_factory
    ):
        manager = _manager_with(dgemma_model_factory())
        calls = self._install_recording_run_diffusion(monkeypatch)

        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            asyncio.run(
                generate_module.generate(manager, {"prompt": "hi", "kv_cache_id": "no-such-handle"})
            )

        # The handle-resolve door fires BEFORE run_diffusion is reached
        # (ADR-CDG-025 §3: "a new, earlier door... not a replacement for the
        # existing one") — a rejected handle must never tie up a scheduler/
        # pipeline construction the way a bad payload at the later door
        # would.
        assert calls == []

    def test_generate_with_a_handle_minted_under_a_different_model_raises(
        self, monkeypatch, dgemma_model_factory
    ):
        """§4: a handle surviving a model swap is the anticipated failure —
        `load()` clears the registry wholesale, so the handle is simply gone
        (not a stale-but-present entry) by the time `generate` resolves it."""
        manager = _manager_with(dgemma_model_factory(repo_id="fake/model-a"))
        self._install_recording_run_diffusion(monkeypatch)

        mint = asyncio.run(encode_module.encode(manager, {"prompt": "donor context"}))
        handle = mint["kv_cache_id"]

        # Simulate a model swap WITHOUT going through StateManager.load()
        # (which would clear the registry) — directly swap `_model`, the
        # same "future code path" scenario resolve_kv_cache's belt-and-
        # suspenders identity check exists for (ADR-CDG-025 §4).
        manager._model = dgemma_model_factory(repo_id="fake/model-b")
        manager._repo_id = "fake/model-b"

        with pytest.raises(ValueError, match="minted under a different model"):
            asyncio.run(
                generate_module.generate(manager, {"prompt": "hi", "kv_cache_id": handle})
            )
