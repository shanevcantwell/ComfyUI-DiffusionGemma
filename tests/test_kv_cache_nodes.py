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

from dgemma.types import KVCache
from surfaces.comfyui.denoise import DGemmaDenoise
from surfaces.comfyui.encode import DGemmaEncode
from surfaces.comfyui.socket_types import DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, DGEMMA_KV_CACHE, DGEMMA_MODEL


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
