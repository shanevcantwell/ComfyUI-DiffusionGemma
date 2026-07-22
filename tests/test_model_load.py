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


class FakeConfig:
    """Stands in for `AutoConfig.from_pretrained`'s return value — `load_model`
    only forwards this to `_resolve_placement` (fix #119), which this test
    class stubs out (`_install_fakes` monkeypatches `_resolve_placement` to a
    fixed `"auto"` return) so these kwargs-shape/error-wrapping tests stay
    about `load_model`'s own request-building and error-mapping behavior —
    not about placement policy or tie-integrity, which get their own real
    (non-mocked) coverage against a genuine shrunk checkpoint in
    `test_tie_integrity_guard.py`."""


class TestLoadModel:
    def _install_fakes(
        self, monkeypatch, captured: dict, hf_device_map=None, raise_on=None, skip_tie_guard=True
    ):
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

        def fake_config_from_pretrained(repo_id, **kwargs):
            if raise_on == "config":
                raise OSError(f"{repo_id} is not a local folder and is not a valid model identifier")
            captured["config_repo_id"] = repo_id
            captured["config_kwargs"] = kwargs
            return FakeConfig()

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr("dgemma.model.AutoProcessor.from_pretrained", fake_processor_from_pretrained)
        monkeypatch.setattr("dgemma.model.AutoConfig.from_pretrained", fake_config_from_pretrained)
        # This class exercises load_model's own kwargs-building/error-mapping
        # behavior — _resolve_placement (fix #119 placement policy) and
        # _assert_tie_integrity (fix #119 guard) are covered on their own
        # merits, against real transformers objects, in
        # test_tie_integrity_guard.py. Stubbing them here keeps a FakeHfModel
        # (no real config/module tree) from tripping the guard's attribute
        # access on a concern this class isn't testing.
        if skip_tie_guard:
            monkeypatch.setattr("dgemma.model._resolve_placement", lambda config, **kw: "auto")
            monkeypatch.setattr("dgemma.model._assert_tie_integrity", lambda model: None)

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
        captured: dict = {}
        self._install_fakes(monkeypatch, captured)

        def raising_from_pretrained(repo_id, **kwargs):
            raise ValueError("unrelated config bug")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", raising_from_pretrained
        )

        with pytest.raises(ValueError, match="unrelated config bug"):
            load_model(repo_id="fake/repo")

    def test_explicit_device_map_used_verbatim_and_skips_placement_policy(self, monkeypatch):
        """fix #119: a caller-supplied `device_map` is never second-guessed —
        `_resolve_placement` must not even be called, and `AutoConfig.from_pretrained`
        (only needed to *decide* a placement) is skipped entirely."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, skip_tie_guard=True)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("_resolve_placement must not be called when device_map is explicit")

        monkeypatch.setattr("dgemma.model._resolve_placement", fail_if_called)

        explicit_map = {"": "cpu"}
        load_model(repo_id="fake/repo", device_map=explicit_map)

        assert captured["kwargs"]["device_map"] == explicit_map
        assert "config_repo_id" not in captured  # AutoConfig.from_pretrained never called

    def test_no_explicit_device_map_calls_resolve_placement_with_fetched_config(self, monkeypatch):
        """fix #119: the default (no explicit device_map) path fetches the
        repo's config and hands it to `_resolve_placement` — this is the new
        seam `_resolve_placement`'s own tests (test_tie_integrity_guard.py)
        exercise directly; here we only check `load_model` wires it up."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, skip_tie_guard=True)

        seen = {}

        def fake_resolve_placement(config, **kwargs):
            seen["config"] = config
            return "auto"

        monkeypatch.setattr("dgemma.model._resolve_placement", fake_resolve_placement)

        load_model(repo_id="fake/repo")

        assert captured["config_repo_id"] == "fake/repo"
        assert isinstance(seen["config"], FakeConfig)
        assert captured["kwargs"]["device_map"] == "auto"

    def test_config_load_failure_raises_clean_runtime_error(self, monkeypatch):
        """The config fetch (fix #119, needed only to decide placement) fails
        the same way the model/processor fetches do — an actionable
        RuntimeError, not a raw OSError."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, raise_on="config")

        with pytest.raises(RuntimeError, match="fake/nonexistent-repo"):
            load_model(repo_id="fake/nonexistent-repo")

    def test_tie_integrity_guard_is_called_before_returning(self, monkeypatch):
        """fix #119: `_assert_tie_integrity` runs on every load (unless a
        test explicitly stubs it) — a real corruption must be visible to a
        caller of `load_model`, not just to whoever calls the guard directly."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, skip_tie_guard=False)

        def raising_guard(model):
            raise RuntimeError("tie-integrity guard fired")

        monkeypatch.setattr("dgemma.model._assert_tie_integrity", raising_guard)
        monkeypatch.setattr("dgemma.model._resolve_placement", lambda config, **kw: "auto")

        with pytest.raises(RuntimeError, match="tie-integrity guard fired"):
            load_model(repo_id="fake/repo")


class TestTransformersVersionGuard:
    """issue #25 front-door guard: ComfyUI-Manager silently skips a
    requirements.txt pin that would downgrade an already-installed package,
    so this repo's env can end up holding a transformers other than the one
    it targets. `_check_transformers_version` must turn that into one
    actionable RuntimeError instead of a raw import/attribute traceback.

    Patch-tolerant (coordinator follow-up): the guard accepts the pinned
    major.minor series (`5.13.x`) and flags only a different minor or major —
    a working patch bump is a bugfix on the same tested API surface, while a
    minor/major bump is untested surface."""

    # ACCEPTED: the exact pin and any patch within the series must not raise.
    @pytest.mark.parametrize("version", ["5.13.0", "5.13.1", "5.13.99"])
    def test_patch_within_pinned_series_is_accepted(self, version):
        _check_transformers_version(version)  # must not raise

    # REJECTED: a different minor, a different major, or a clearly-old
    # version must all raise the actionable error.
    @pytest.mark.parametrize("version", ["5.12.0", "5.14.0", "6.0.0", "4.50.0"])
    def test_out_of_series_version_raises_actionable_runtime_error(self, version):
        with pytest.raises(RuntimeError) as excinfo:
            _check_transformers_version(version)

        message = str(excinfo.value)
        assert REQUIRED_TRANSFORMERS_VERSION in message  # names the required version
        assert version in message  # names what's actually installed
        assert "pip install transformers==" in message  # concrete fix
        assert "issue #25" in message.lower() or "#25" in message

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
        because the dev/test environment is on the pinned major.minor series."""
        _check_transformers_version()  # must not raise in this repo's own env
