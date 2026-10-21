"""BASE-02 source subtests; the group runner separately enforces BLOCKED gate inputs."""
from __future__ import annotations

import ast
import importlib.metadata
import math
from pathlib import Path
import subprocess
import types
import unittest
from unittest import mock

import torch

from i0_testlib import (
    ENGINE_DESIGN,
    FIXTURES,
    REPO_ROOT,
    checked_distribution_bytes,
    decoder_visible_positions,
    entropy_bound_accept,
    inverse_cdf,
    load_authenticated_cdg_package,
    read_authenticated_cdg_source,
    read_evidence_artifact,
    read_json,
    retained_prefix_positions,
    sha256,
    stable_softmax,
    top_k_contract,
    verify_base02_gate,
    verify_evidence_scope,
)


def class_source(source: str, class_name: str) -> str:
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError(f"no source segment for {class_name}")
    return segment


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return __import__("hashlib").sha1(header + data).hexdigest()


class Base02BaselineDefectReproductions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        cls.pack, cls.pack_loader = cls.enterClassContext(
            load_authenticated_cdg_package(cls.lock, verify_oracle_bindings=True)
        )
        cls.by_id = {case["id"]: case for case in read_json(FIXTURES / "baseline-defects.json")["cases"]}

    def test_cache_advance_mutates_input_object(self):
        kv_module = self.pack_loader.import_module("dgemma.kv_cache")
        types_module = self.pack_loader.import_module("dgemma.types")
        KVCache, Provenance = types_module.KVCache, types_module.Provenance

        class Cache:
            def __init__(self, length=2): self.length, self.layers = length, [object()]
            def get_seq_length(self, layer_idx=0): return self.length
            def append(self, count): self.length += count

        class Encoder:
            def __init__(self): self.weight = torch.nn.Parameter(torch.zeros(1))
            def parameters(self): yield self.weight
            def __call__(self, *, input_ids, past_key_values, position_ids):
                past_key_values.append(input_ids.shape[-1])
                return types.SimpleNamespace(past_key_values=past_key_values)

        encoder = Encoder()
        model = types.SimpleNamespace(model=types.SimpleNamespace(model=types.SimpleNamespace(encoder=encoder)), repo_id="model")
        payload = KVCache(Cache(), (2,), {}, Provenance((1, 2), (), "model", "tokenizer"))
        with mock.patch.object(kv_module, "geometry_from_model", return_value={"num_hidden_layers": 1}), mock.patch.object(
            kv_module, "tokenizer_fingerprint", return_value="tokenizer"
        ):
            result = kv_module.encode_sequence(model, [3, 4], into=payload)
        self.assertIs(result.cache, payload.cache)
        self.assertEqual(payload.cache.length, 4)
        self.assertEqual(self.by_id["CACHE-ADVANCE-MUTATION"]["classification"], "defect-to-repair")

    def test_issue_265_prefill_alias_then_second_ingress_rejects_v7(self):
        kv_module = self.pack_loader.import_module("dgemma.kv_cache")
        types_module = self.pack_loader.import_module("dgemma.types")
        prefill_templated_turn = kv_module.prefill_templated_turn
        validate_kv_cache_ingress = kv_module.validate_kv_cache_ingress
        KVCache, Provenance = types_module.KVCache, types_module.Provenance

        class Layer:
            def __init__(self, length): self.keys = torch.zeros((1, 1, length, 1), dtype=torch.float32)

        class Cache:
            def __init__(self, length=2): self.length, self.layers = length, [Layer(length)]
            def __len__(self): return len(self.layers)
            def get_seq_length(self, layer_idx=0): return self.length
            def append(self, count):
                self.length += count
                self.layers[0].keys = torch.zeros((1, 1, self.length, 1), dtype=torch.float32)

        class Encoder:
            def __init__(self): self.weight = torch.nn.Parameter(torch.zeros(1))
            def parameters(self): yield self.weight
            def __call__(self, *, input_ids, past_key_values, position_ids):
                past_key_values.append(input_ids.shape[-1])
                return types.SimpleNamespace(past_key_values=past_key_values)

        text_config = types.SimpleNamespace(
            num_hidden_layers=1,
            layer_types=["sliding_attention"],
            sliding_window=4,
            rope_parameters={"sliding_attention": {"rope_type": "default"}},
        )
        config = types.SimpleNamespace(get_text_config=lambda: text_config)
        encoder = Encoder()
        cache = Cache()
        dgemma_model = types.SimpleNamespace(
            model=types.SimpleNamespace(config=config, model=types.SimpleNamespace(encoder=encoder)),
            processor=types.SimpleNamespace(tokenizer=types.SimpleNamespace(vocab_size=8)),
            repo_id="model",
            dtype="float32",
            device="cpu",
        )
        geometry = {
            "num_hidden_layers": 1,
            "layer_types": ("sliding_attention",),
            "sliding_window": 4,
            "rope_parameters": {"sliding_attention": {"rope_type": "default"}},
        }
        payload = KVCache(cache, (2,), geometry, Provenance((1, 2), (), "model", "model:8"))
        pipeline = types.SimpleNamespace(
            processor=types.SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: {"input_ids": torch.tensor([[3, 4, 5]])}
            ),
            model=types.SimpleNamespace(model=types.SimpleNamespace(encoder=encoder)),
        )
        self.assertIs(prefill_templated_turn(pipeline, cache, {"prompt": "turn"}), cache)
        self.assertEqual(cache.length, 5)
        with self.assertRaisesRegex(ValueError, "V7") as caught:
            validate_kv_cache_ingress(payload, dgemma_model)
        self.assertIn("#265", str(caught.exception))
        self.assertIn("re-run DGemmaEncode", str(caught.exception))

    def test_issue_263_offset_shape_rejected_at_door(self):
        ingress_module = self.pack_loader.import_module("dgemma.ingress")
        reject_multi_block_composed_prefill = ingress_module.reject_multi_block_composed_prefill

        cache = object()
        reject_multi_block_composed_prefill("turn", cache, gen_length=256, canvas_length=256)
        reject_multi_block_composed_prefill("", cache, gen_length=512, canvas_length=256)
        with self.assertRaisesRegex(ValueError, "#263"):
            reject_multi_block_composed_prefill("turn", cache, gen_length=257, canvas_length=256)

    def test_denoise_omits_cancellation_while_sampler_wires_it(self):
        denoise_module = self.pack_loader.import_module("surfaces.comfyui.denoise")
        sampler_module = self.pack_loader.import_module("surfaces.comfyui.sampler")

        result = ("text", object(), types.SimpleNamespace(frames=[]))
        params = dict(model=object(), prompt="p", seed=1, num_inference_steps=2, t_min=0.4, t_max=0.8, entropy_bound=0.1, confidence=0.005, gen_length=256, thinking=False, unique_id="1")
        with mock.patch.object(denoise_module, "run_diffusion", return_value=result) as denoise_run, mock.patch.object(
            denoise_module, "build_sampler_shaped_outputs", return_value=("denoise",)
        ):
            denoise_module.DGemmaDenoise().denoise(**params, kv_cache=None)
        with mock.patch.object(sampler_module, "run_diffusion", return_value=result) as sampler_run, mock.patch.object(
            sampler_module, "build_sampler_shaped_outputs", return_value=("sampler",)
        ):
            sampler_module.DGemmaSampler().sample(**params)
        self.assertNotIn("should_cancel", denoise_run.call_args.kwargs)
        self.assertTrue(callable(sampler_run.call_args.kwargs["should_cancel"]))

    def test_autoround_quantlinear_missing_weight_reproduces_failure(self):
        model_module = self.pack_loader.import_module("dgemma.model")
        _assert_tie_integrity = model_module._assert_tie_integrity

        quant_linear = types.SimpleNamespace(qweight=torch.ones(1))
        layer = types.SimpleNamespace(self_attn=types.SimpleNamespace(q_proj=quant_linear))
        text_config = types.SimpleNamespace(num_hidden_layers=1, num_attention_heads=1, hidden_size=1, head_dim=1, global_head_dim=1, layer_types=["sliding_attention"])
        inner = types.SimpleNamespace(encoder=types.SimpleNamespace(language_model=types.SimpleNamespace(layers=[layer])), decoder=types.SimpleNamespace(layers=[layer]))
        model = types.SimpleNamespace(config=types.SimpleNamespace(text_config=text_config), model=inner)
        with self.assertRaisesRegex(AttributeError, "weight"):
            _assert_tie_integrity(model)
        self.assertIsNone(self.lock["autoround_derivative"]["license"]["spdx"])
        self.assertEqual(self.by_id["AUTOROUND-DECLARED-BROKEN"]["classification"], "declared-but-broken-mode")


if __name__ == "__main__":
    unittest.main(verbosity=2)
