"""dgemma/model.py unit coverage (test-coverage-plan.md Phase 1 — the 30%
module; no weights, no GPU, no ComfyUI).

`_resolve_device` takes plain fakes, so it is exercised directly with no
monkeypatching (test-coverage-plan.md: "don't mock what you can test
directly"). `load_model` needs
`DiffusionGemmaForBlockDiffusion.from_pretrained` +
`AutoProcessor.from_pretrained` monkeypatched — those are the one real
external seam this module has.

Issue #18 removed the bnb nf4/int8 quant paths (bitsandbytes cannot quantize
DiffusionGemma's fused 3D MoE experts, so both were misleading on any
hardware for this architecture) — `quant` now only accepts `"none"`, and
this module no longer has `_quantization_config`/`_device_map` branches to
cover.

`TestTransformersVersionGuard` covers issue #25's front-door guard: the
`installed` parameter on `_check_transformers_version` exists precisely so
this is testable without monkeypatching `sys.modules["transformers"]`.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.model import (
    DEFAULT_QUANT,
    DEFAULT_REPO_ID,
    REQUIRED_TRANSFORMERS_VERSION,
    _check_transformers_version,
    _resolve_device,
    load_model,
)
from dgemma.types import DGemmaModel


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
    def _install_fakes(self, monkeypatch, captured: dict, hf_device_map=None, raise_on=None):
        def fake_from_pretrained(repo_id, **kwargs):
            if raise_on == "model":
                raise OSError(f"{repo_id} is not a local folder and is not a valid model identifier")
            captured["repo_id"] = repo_id
            captured["kwargs"] = kwargs
            return FakeHfModel(hf_device_map=hf_device_map, first_param_device="cpu")

        def fake_processor_from_pretrained(repo_id, **kwargs):
            if raise_on == "processor":
                raise OSError(f"{repo_id} is not a local folder and is not a valid model identifier")
            captured["processor_repo_id"] = repo_id
            captured["processor_kwargs"] = kwargs
            return FakeProcessor()

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr("dgemma.model.AutoProcessor.from_pretrained", fake_processor_from_pretrained)

    def test_load_kwargs_shape(self, monkeypatch):
        """quant="none" is the only path left: device_map="auto",
        dtype=bfloat16, no quantization_config."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id="fake/repo", quant="none")

        kwargs = captured["kwargs"]
        assert kwargs["device_map"] == "auto"
        assert kwargs["dtype"] == torch.bfloat16
        assert "quantization_config" not in kwargs
        assert isinstance(result, DGemmaModel)

    def test_returned_dgemma_model_fields(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 2})

        result = load_model(repo_id="google/diffusiongemma-26B-A4B-it", quant="none")

        assert result.repo_id == "google/diffusiongemma-26B-A4B-it"
        assert result.dtype == "bfloat16"
        assert result.device == "cuda:2"  # from _resolve_device via hf_device_map
        assert result.quant == "none"
        assert captured["processor_repo_id"] == "google/diffusiongemma-26B-A4B-it"

    def test_invalid_quant_raises_before_touching_from_pretrained(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured)

        with pytest.raises(ValueError, match="quant must be one of"):
            load_model(repo_id="fake/repo", quant="nf4")

        assert "kwargs" not in captured  # never got far enough to call from_pretrained

    def test_defaults_are_the_one_mint_module_constants(self, monkeypatch):
        """DEFAULT_REPO_ID/DEFAULT_QUANT source `load_model`'s own defaults
        and the loader widget default (model.py's ONE-MINT comment) — calling
        with no args must actually use them, not silently diverge."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map=None)

        load_model()

        assert captured["repo_id"] == DEFAULT_REPO_ID
        assert DEFAULT_QUANT == "none"
        assert captured["kwargs"]["dtype"] == torch.bfloat16

    def test_local_files_only_defaults_false_and_threads_into_both_calls(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map=None)

        load_model(repo_id="fake/repo")

        assert captured["kwargs"]["local_files_only"] is False
        assert captured["processor_kwargs"]["local_files_only"] is False

    def test_local_files_only_true_threads_into_both_calls(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map=None)

        load_model(repo_id="fake/repo", local_files_only=True)

        assert captured["kwargs"]["local_files_only"] is True
        assert captured["processor_kwargs"]["local_files_only"] is True

    def test_unresolvable_repo_raises_clean_runtime_error_not_raw_oserror(self, monkeypatch):
        """The from_pretrained OSError (typo'd repo_id / no network / not
        cached under local_files_only=True) must surface as an actionable
        RuntimeError naming the repo_id, not a raw transformers/HF stack
        trace."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, raise_on="model")

        with pytest.raises(RuntimeError, match="fake/nonexistent-repo") as excinfo:
            load_model(repo_id="fake/nonexistent-repo", local_files_only=True)

        assert "local_files_only=True" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, OSError)  # original error is chained, not swallowed

    def test_unresolvable_repo_without_local_files_only_names_network_cause(self, monkeypatch):
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, raise_on="model")

        with pytest.raises(RuntimeError, match="network") as excinfo:
            load_model(repo_id="fake/nonexistent-repo", local_files_only=False)

        assert isinstance(excinfo.value.__cause__, OSError)

    def test_processor_load_failure_also_raises_clean_error(self, monkeypatch):
        """The processor's from_pretrained is a separate call to the same
        unresolvable repo_id — its failure must be wrapped the same way as
        the model's, not left as a raw OSError."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, raise_on="processor")

        with pytest.raises(RuntimeError, match="fake/nonexistent-repo"):
            load_model(repo_id="fake/nonexistent-repo")

    def test_unrelated_value_error_is_not_swallowed(self, monkeypatch):
        """The narrow `except OSError` must not catch unrelated bugs — a
        ValueError raised inside from_pretrained (e.g. a real config bug)
        must propagate as itself, not get relabeled as a load-resolution
        error."""

        def raising_from_pretrained(repo_id, **kwargs):
            raise ValueError("unrelated config bug")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", raising_from_pretrained
        )

        with pytest.raises(ValueError, match="unrelated config bug"):
            load_model(repo_id="fake/repo")


class TestTransformersVersionGuard:
    """issue #25 front-door guard: ComfyUI-Manager silently skips a
    requirements.txt pin that would downgrade an already-installed package,
    so this repo's env can end up holding a transformers other than the one
    it targets. `_check_transformers_version` must turn that into one
    actionable RuntimeError instead of a raw import/attribute traceback."""

    def test_matching_version_is_a_no_op(self):
        _check_transformers_version(REQUIRED_TRANSFORMERS_VERSION)  # must not raise

    def test_older_version_raises_actionable_runtime_error(self):
        with pytest.raises(RuntimeError) as excinfo:
            _check_transformers_version("5.12.0")

        message = str(excinfo.value)
        assert REQUIRED_TRANSFORMERS_VERSION in message  # names the required version
        assert "5.12.0" in message  # names what's actually installed
        assert "pip install transformers==" in message  # concrete fix
        assert "issue #25" in message.lower() or "#25" in message

    def test_newer_version_also_raises(self):
        """The pin is exact (`==`), not a floor — a newer transformers is
        just as much a mismatch as an older one."""
        with pytest.raises(RuntimeError, match=REQUIRED_TRANSFORMERS_VERSION):
            _check_transformers_version("5.14.0")

    def test_message_explains_manager_downgrade_skip_behavior(self):
        """The actionable message must explain *why* the env can be wrong
        even after a normal ComfyUI-Manager install — not just state the
        required version."""
        with pytest.raises(RuntimeError) as excinfo:
            _check_transformers_version("5.12.0")

        assert "downgrade" in str(excinfo.value).lower()

    def test_installed_none_reads_the_real_transformers_version(self):
        """Default (no `installed` arg) path: reads the real, currently
        importable `transformers.__version__` — exercised here as a no-op
        because the dev/test environment is pinned to the required version."""
        _check_transformers_version()  # must not raise in this repo's own env
