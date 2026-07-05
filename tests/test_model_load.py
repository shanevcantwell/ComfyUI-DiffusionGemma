"""dgemma/model.py unit coverage (test-coverage-plan.md Phase 1 — the 30%
module; no weights, no GPU, no ComfyUI).

`_quantization_config` and `_resolve_device` take plain inputs/fakes, so they
are exercised directly with no monkeypatching (test-coverage-plan.md: "don't
mock what you can test directly"). `_device_map` needs `torch.cuda.is_available`
monkeypatched (there is no real CUDA on this box's test runner). `load_model`
needs `DiffusionGemmaForBlockDiffusion.from_pretrained` +
`AutoProcessor.from_pretrained` monkeypatched — those are the one real
external seam this module has.
"""
from __future__ import annotations

import pytest
import torch
from transformers import BitsAndBytesConfig

from dgemma.model import (
    DEFAULT_QUANT,
    DEFAULT_REPO_ID,
    _device_map,
    _quantization_config,
    _resolve_device,
    load_model,
)
from dgemma.types import DGemmaModel


class TestQuantizationConfig:
    def test_nf4_is_4bit_with_float16_compute_dtype(self):
        """The sm_75 grounded fact (CLAUDE.md): compute dtype is float16, not
        bfloat16 — this dev box's Turing RTX-8000 has no native bf16
        tensor-core support."""
        config = _quantization_config("nf4")
        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
        assert config.bnb_4bit_compute_dtype == torch.float16

    def test_int8_is_8bit(self):
        config = _quantization_config("int8")
        assert isinstance(config, BitsAndBytesConfig)
        assert config.load_in_8bit is True
        assert config.load_in_4bit is False

    def test_none_returns_none(self):
        assert _quantization_config("none") is None

    def test_invalid_quant_raises_value_error(self):
        with pytest.raises(ValueError, match="quant must be one of"):
            _quantization_config("fp8")


class TestDeviceMap:
    @pytest.mark.parametrize("quant", ["nf4", "int8"])
    def test_quantized_with_cuda_pins_to_gpu_0(self, monkeypatch, quant):
        """Observed-failure rationale (model.py docstring): accelerate's
        "auto" placement estimate spills modules the bnb 4-bit quantizer then
        rejects outright — pinning to {"": 0} skips the estimator entirely."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _device_map(quant) == {"": 0}

    @pytest.mark.parametrize("quant", ["nf4", "int8"])
    def test_quantized_without_cuda_falls_back_to_auto(self, monkeypatch, quant):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _device_map(quant) == "auto"

    def test_none_with_cuda_is_still_auto(self, monkeypatch):
        """The unquantized 26B path may need multi-GPU sharding or CPU
        offload — "auto" earns its keep here, unlike the quantized paths."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _device_map("none") == "auto"

    def test_none_without_cuda_is_auto(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _device_map("none") == "auto"


class _FakeParam:
    def __init__(self, device: str):
        self.device = device


class FakeHfModel:
    """Stands in for a loaded `DiffusionGemmaForBlockDiffusion`: only
    `hf_device_map` and `parameters()` matter to `_resolve_device`."""

    def __init__(self, hf_device_map=None, first_param_device="cpu"):
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map
        self._first_param_device = first_param_device

    def parameters(self):
        yield _FakeParam(self._first_param_device)


class TestResolveDevice:
    def test_int_gpu_entry_resolves_to_cuda_n(self):
        model = FakeHfModel(hf_device_map={"model.layers.0": 0})
        assert _resolve_device(model) == "cuda:0"

    def test_non_int_non_cpu_disk_entry_is_returned_as_is(self):
        """A device_map value that is neither a bare int (accelerate's GPU
        encoding) nor "cpu"/"disk" — e.g. an mps/other accelerator string —
        still counts as the accelerator and is returned verbatim."""
        model = FakeHfModel(hf_device_map={"model.embed": "mps"})
        assert _resolve_device(model) == "mps"

    def test_cpu_spill_map_still_finds_the_accelerator(self):
        """First parameter off-GPU, later entry is the int accelerator —
        execution device, not first-parameter device, is what must resolve."""
        model = FakeHfModel(
            hf_device_map={"model.embed": "cpu", "model.layers.10": 1},
            first_param_device="cpu",
        )
        assert _resolve_device(model) == "cuda:1"

    def test_all_cpu_map_falls_back_to_first_parameter_device(self):
        model = FakeHfModel(
            hf_device_map={"model.embed": "cpu", "model.layers.0": "disk"},
            first_param_device="cpu",
        )
        assert _resolve_device(model) == "cpu"

    def test_no_hf_device_map_falls_back_to_first_parameter_device(self):
        model = FakeHfModel(hf_device_map=None, first_param_device="cuda:0")
        assert _resolve_device(model) == "cuda:0"


class FakeProcessor:
    """Stands in for `AutoProcessor.from_pretrained`'s return value —
    `load_model` never inspects it beyond storing it on `DGemmaModel`."""


class TestLoadModel:
    def _install_fakes(self, monkeypatch, captured: dict, hf_device_map=None):
        def fake_from_pretrained(repo_id, **kwargs):
            captured["repo_id"] = repo_id
            captured["kwargs"] = kwargs
            return FakeHfModel(hf_device_map=hf_device_map, first_param_device="cpu")

        def fake_processor_from_pretrained(repo_id):
            captured["processor_repo_id"] = repo_id
            return FakeProcessor()

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr("dgemma.model.AutoProcessor.from_pretrained", fake_processor_from_pretrained)

    @pytest.mark.parametrize(
        "quant,expect_quant_kwarg,expect_dtype_kwarg",
        [
            ("nf4", True, False),
            ("int8", True, False),
            ("none", False, True),
        ],
    )
    def test_load_kwargs_shape_per_quant(self, monkeypatch, quant, expect_quant_kwarg, expect_dtype_kwarg):
        """quantization_config present iff quantized; dtype=bfloat16 iff
        unquantized — the two are mutually exclusive load_kwargs."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        result = load_model(repo_id="fake/repo", quant=quant)

        kwargs = captured["kwargs"]
        assert ("quantization_config" in kwargs) is expect_quant_kwarg
        assert ("dtype" in kwargs) is expect_dtype_kwarg
        if expect_dtype_kwarg:
            assert kwargs["dtype"] == torch.bfloat16
        if expect_quant_kwarg:
            assert isinstance(kwargs["quantization_config"], BitsAndBytesConfig)
        assert kwargs["device_map"] == _device_map(quant)
        assert isinstance(result, DGemmaModel)

    def test_returned_dgemma_model_fields(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 2})
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        result = load_model(repo_id="google/diffusiongemma-26B-A4B-it", quant="nf4")

        assert result.repo_id == "google/diffusiongemma-26B-A4B-it"
        assert result.dtype == "float16"  # _QUANT_DTYPE_LABELS["nf4"]
        assert result.device == "cuda:2"  # from _resolve_device via hf_device_map
        assert result.quant == "nf4"
        assert captured["processor_repo_id"] == "google/diffusiongemma-26B-A4B-it"

    def test_none_quant_dtype_label_is_bfloat16(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map=None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        result = load_model(repo_id="fake/repo", quant="none")

        assert result.dtype == "bfloat16"

    def test_invalid_quant_raises_before_touching_from_pretrained(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured)

        with pytest.raises(ValueError, match="quant must be one of"):
            load_model(repo_id="fake/repo", quant="fp8")

        assert "kwargs" not in captured  # never got far enough to call from_pretrained

    def test_defaults_are_the_one_mint_module_constants(self, monkeypatch):
        """DEFAULT_REPO_ID/DEFAULT_QUANT source `load_model`'s own defaults
        and the loader widget default (model.py's ONE-MINT comment) — calling
        with no args must actually use them, not silently diverge."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map=None)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        load_model()

        assert captured["repo_id"] == DEFAULT_REPO_ID
        assert DEFAULT_QUANT == "none"
        assert "dtype" in captured["kwargs"]  # DEFAULT_QUANT == "none" path
