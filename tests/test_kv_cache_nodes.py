"""tests/test_kv_cache_nodes.py — ADR-CDG-012 Phase 3 (issue #62):
`DGemmaEncode`/`DGemmaDenoise` thin-adapter purity + socket wiring.

Scope: these are ADR-CDG-003 thin adapters — unpack widget inputs, call one
`dgemma.*` function, wrap the result. No step-loop/denoising logic may live
in either node body; these tests assert the wiring is correct (INPUT_TYPES,
RETURN_TYPES, the one-call body), not step-loop behavior (that is
`dgemma.loop.run_diffusion`'s/`dgemma.kv_cache.encode_sequence`'s own test
suite's job).
"""
from __future__ import annotations

import inspect
import sys

from dgemma.types import KVCache
from surfaces.comfyui.denoise import DGemmaDenoise
from surfaces.comfyui.encode import DGemmaEncode
from surfaces.comfyui.socket_types import DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, DGEMMA_KV_CACHE, DGEMMA_MODEL


class _StubModel:
    processor = object()


class _StubTrace:
    frames = ()


class _FakeFrame:
    """Minimal stand-in for `dgemma.types.DiffusionFrame` — only the fields
    `_build_on_frame`'s closure actually reads (same shape as
    `tests/test_loader_contract.py`'s own `_FakeFrame`)."""

    canvas_idx = 0
    step_idx = 3
    t = 0.2
    temperature = 0.5
    committed_fraction = 1.0


class TestDGemmaEncodeContract:
    def test_input_types_shape(self):
        spec = DGemmaEncode.INPUT_TYPES()
        assert spec["required"]["model"][0] == DGEMMA_MODEL
        assert spec["required"]["text"][0] == "STRING"
        assert spec["optional"]["kv_cache"][0] == DGEMMA_KV_CACHE

    def test_return_types(self):
        assert DGemmaEncode.RETURN_TYPES == (DGEMMA_KV_CACHE,)
        assert DGemmaEncode.RETURN_NAMES == ("kv_cache",)
        assert DGemmaEncode.FUNCTION == "encode"

    def test_body_is_a_thin_adapter_no_step_loop(self):
        """ADR-CDG-003 rule 2: no `for`/`while` loop over denoising steps may
        live in the node body — the mint/advance loop belongs entirely to
        `dgemma.kv_cache.encode_sequence`."""
        source = inspect.getsource(DGemmaEncode.encode)
        assert "for " not in source
        assert "while " not in source


class TestDGemmaEncodeMintAndAdvance:
    def test_mints_a_fresh_kv_cache_from_text(self, dgemma_model_factory):
        model = dgemma_model_factory()
        node = DGemmaEncode()

        (cache,) = node.encode(model, "hello")

        assert isinstance(cache, KVCache)
        assert cache.provenance.minting_sequence is not None
        assert cache.provenance.model_repo_id == model.repo_id

    def test_advances_an_existing_kv_cache_when_supplied(self, dgemma_model_factory):
        model = dgemma_model_factory()
        node = DGemmaEncode()

        (first,) = node.encode(model, "hello")
        (second,) = node.encode(model, " world", kv_cache=first)

        assert second.provenance.minting_sequence[: len(first.provenance.minting_sequence)] == (
            first.provenance.minting_sequence
        )
        assert sum(second.cumulative_length) > sum(first.cumulative_length)

    def test_kv_cache_none_default_mints_fresh(self, dgemma_model_factory):
        model = dgemma_model_factory()
        node = DGemmaEncode()

        (explicit_none,) = node.encode(model, "hi", kv_cache=None)
        (omitted,) = node.encode(model, "hi")

        assert explicit_none.provenance.minting_sequence == omitted.provenance.minting_sequence


class TestDGemmaDenoiseContract:
    def test_input_types_shape(self):
        spec = DGemmaDenoise.INPUT_TYPES()
        assert spec["required"]["model"][0] == DGEMMA_MODEL
        assert spec["optional"]["kv_cache"][0] == DGEMMA_KV_CACHE
        assert "unique_id" in spec["hidden"]

    def test_return_types(self):
        assert DGemmaDenoise.RETURN_TYPES == ("STRING", DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE)
        assert DGemmaDenoise.RETURN_NAMES == ("text", "canvas_state", "canvas_trace")
        assert DGemmaDenoise.FUNCTION == "denoise"

    def test_body_is_a_thin_adapter_no_step_loop(self):
        """ADR-CDG-003 rule 2: the denoising loop lives entirely inside
        `dgemma.loop.run_diffusion` — this adapter body has no loop of its
        own, matching `DGemmaSampler.sample`'s existing shape."""
        source = inspect.getsource(DGemmaDenoise.denoise)
        assert "for " not in source
        assert "while " not in source


class TestDGemmaDenoiseThreadsKVCacheThrough:
    """`DGemmaDenoise.denoise` forwards `kv_cache=` to `run_diffusion`
    unchanged — the thin-adapter contract for IN-2, verified by
    monkeypatching the exact `dgemma.*` call this node makes (same pattern
    `tests/test_loader_contract.py` uses for `DGemmaSampler`)."""

    def test_forwards_kv_cache_to_run_diffusion(self, monkeypatch):
        captured = {}

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            captured["kv_cache"] = kwargs.get("kv_cache")
            return ("text", "state", _StubTrace())

        monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)

        node = DGemmaDenoise()
        sentinel_cache = object()
        node.denoise(
            _StubModel(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            kv_cache=sentinel_cache,
            unique_id="42",
        )

        assert captured["kv_cache"] is sentinel_cache

    def test_kv_cache_omitted_forwards_none(self, monkeypatch):
        captured = {}

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            captured["kv_cache"] = kwargs.get("kv_cache")
            return ("text", "state", _StubTrace())

        monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)

        node = DGemmaDenoise()
        node.denoise(
            _StubModel(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
        )

        assert captured["kv_cache"] is None


class TestDGemmaDenoiseLiveFramePush:
    """Coverage closer for `_build_on_frame`'s live-push closure (identical
    shape to `DGemmaSampler`'s own — see `tests/test_loader_contract.py`'s
    `TestLiveFramePush` for the precedent this mirrors), scoped to
    `denoise.py`'s own module path/event name."""

    def test_denoise_succeeds_unchanged_when_promptserver_unavailable(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "server", raising=False)
        trace_stub = _StubTrace()

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())
            return ("text", "state", trace_stub)

        monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)

        node = DGemmaDenoise()
        text, state, trace = node.denoise(
            _StubModel(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            unique_id="42",
        )

        assert (text, state, trace) == ("text", "state", trace_stub)

    def test_on_frame_pushes_via_send_sync_when_promptserver_available(self, monkeypatch):
        captured_calls = []

        class FakeInstance:
            def send_sync(self, event, data, sid=None):
                captured_calls.append((event, data, sid))

        class FakePromptServer:
            instance = FakeInstance()

        fake_server_module = type(sys)("server")
        fake_server_module.PromptServer = FakePromptServer
        monkeypatch.setitem(sys.modules, "server", fake_server_module)

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())
            return ("text", "state", _StubTrace())

        monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)

        node = DGemmaDenoise()
        node.denoise(
            _StubModel(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            unique_id="42",
        )

        assert len(captured_calls) == 1
        event, data, sid = captured_calls[0]
        assert event == "dgemma.denoise.step"
        assert data["node"] == "42"
        assert data["canvas_idx"] == 0
        assert data["step_idx"] == 3
        assert data["committed_fraction"] == 1.0

    def test_on_frame_is_a_no_op_when_promptserver_instance_is_none(self, monkeypatch):
        class FakePromptServer:
            instance = None

        fake_server_module = type(sys)("server")
        fake_server_module.PromptServer = FakePromptServer
        monkeypatch.setitem(sys.modules, "server", fake_server_module)

        trace_stub = _StubTrace()

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())  # must not raise
            return ("text", "state", trace_stub)

        monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)

        node = DGemmaDenoise()
        result = node.denoise(
            _StubModel(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            unique_id="42",
        )

        assert result == ("text", "state", trace_stub)

    def test_send_sync_failure_does_not_kill_the_run(self, monkeypatch, caplog):
        class ExplodingInstance:
            def send_sync(self, event, data, sid=None):
                raise RuntimeError("websocket dropped mid-run")

        class FakePromptServer:
            instance = ExplodingInstance()

        fake_server_module = type(sys)("server")
        fake_server_module.PromptServer = FakePromptServer
        monkeypatch.setitem(sys.modules, "server", fake_server_module)

        trace_stub = _StubTrace()

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())
            return ("text", "state", trace_stub)

        monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)

        node = DGemmaDenoise()
        with caplog.at_level("WARNING"):
            result = node.denoise(
                _StubModel(),
                prompt="hi",
                seed=1,
                num_inference_steps=1,
                t_min=0.1,
                t_max=0.5,
                entropy_bound=0.1,
                confidence=0.1,
                gen_length=8,
                unique_id="42",
            )

        assert result == ("text", "state", trace_stub)
        assert any("live push failed" in record.message for record in caplog.records)
