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
    LoadInterrupted,
    _check_transformers_version,
    _resolve_device,
    load_model,
)
from dgemma.types import DGemmaModel


class _FakeDevice:
    """Minimal stand-in for `torch.device` — `.type` is what
    `_assert_no_meta_tensors` inspects; `str()` is what `_resolve_device` and
    the old device-string assertions compare against."""

    def __init__(self, device_str: str):
        self._device_str = device_str
        # torch.device("cuda:0").type == "cuda"; torch.device("cpu").type == "cpu";
        # torch.device("meta").type == "meta" — same split here.
        self.type = device_str.split(":")[0]

    def __str__(self):
        return self._device_str


class _FakeParam:
    def __init__(self, device: str):
        self.device = _FakeDevice(device)


class FakeHfModel:
    """Stands in for a loaded `DiffusionGemmaForBlockDiffusion`: `hf_device_map`
    and `parameters()` matter to `_resolve_device`; `.to()` + `named_parameters()`
    + `named_buffers()` are what `load_model`'s post-load meta-tensor assertion
    (issue #142) and the final `.to("cuda")` call touch."""

    def __init__(self, hf_device_map=None, first_param_device="cpu"):
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map
        self._first_param_device = first_param_device

    def parameters(self):
        yield _FakeParam(self._first_param_device)

    def named_parameters(self):
        yield "fake.weight", _FakeParam(self._first_param_device)

    def named_buffers(self):
        return iter(())

    def to(self, *args, **kwargs):
        return self


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
        """Also pins `torch.cuda.is_available()` True (#159 gate finding):
        these fakes drive `load_model()` to its real `.to("cuda")` call,
        which is a no-op against `FakeHfModel.to` — but the no-CUDA guard
        upstream of it (issue #143) is a real host check, so without this
        pin these tests pass on a CUDA host and fail on CPU-only CI for a
        reason unrelated to what they're testing.
        `test_no_cuda_raises_intentional_runtime_error_not_unbound_local`
        overrides this back to False right after calling this helper —
        monkeypatch's later-wins semantics keep that override intact."""

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
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)

    def test_load_kwargs_shape(self, monkeypatch):
        """quant="none": dtype=bfloat16, low_cpu_mem_usage=False (forces real
        CPU tensors so meta tensors can't slip past to .to("cuda") — dd2767c
        dropped device_map="auto" entirely, so no such key is passed)."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id="fake/repo", quant="none")

        kwargs = captured["kwargs"]
        assert "device_map" not in kwargs
        assert kwargs["low_cpu_mem_usage"] is False
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

    def test_no_cuda_raises_intentional_runtime_error_not_unbound_local(self, monkeypatch):
        """issue #143: the no-CUDA else-branch used to reference a `device`
        variable removed by dd2767c, raising an opaque UnboundLocalError
        instead of the intended message. Must now raise a house-register
        RuntimeError naming the missing precondition, with no dead reference."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: False)

        with pytest.raises(RuntimeError, match="CUDA"):
            load_model(repo_id="fake/repo", quant="none")

    def test_meta_resident_tensor_after_load_raises_before_to_cuda(self, monkeypatch):
        """issue #142's enforcement surface: any parameter/buffer still on
        meta after load (and after the autoround re-tie attempt) must raise,
        naming the tensor, instead of reaching .to("cuda") and surfacing an
        opaque 'Cannot copy out of meta tensor' error downstream."""
        captured: dict = {}

        def fake_from_pretrained(repo_id, **kwargs):
            captured["kwargs"] = kwargs
            return FakeHfModel(hf_device_map={"model.layers.0": 0}, first_param_device="meta")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr(
            "dgemma.model.AutoProcessor.from_pretrained", lambda repo_id, **kw: FakeProcessor()
        )

        with pytest.raises(RuntimeError, match="meta-resident"):
            load_model(repo_id="fake/repo", quant="none")

    def test_to_cuda_called_when_cuda_available(self, monkeypatch):
        """The real device-move: model.to("cuda") is invoked (not skipped)
        once CUDA is available and no meta tensors remain."""
        captured: dict = {}
        calls: list = []

        class _TrackedFakeHfModel(FakeHfModel):
            def to(self, *args, **kwargs):
                calls.append(args)
                return self

        def fake_from_pretrained(repo_id, **kwargs):
            captured["kwargs"] = kwargs
            return _TrackedFakeHfModel(hf_device_map={"model.layers.0": 0}, first_param_device="cpu")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr(
            "dgemma.model.AutoProcessor.from_pretrained", lambda repo_id, **kw: FakeProcessor()
        )
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)

        load_model(repo_id="fake/repo", quant="none")

        assert calls == [("cuda",)]


class TestCheckInterrupted:
    """issue #140 loader half: `load_model`'s optional `check_interrupted`
    predicate, polled at each phase boundary (quant/checkpoint pre-flight,
    model `from_pretrained`, processor `from_pretrained`, `.to("cuda")`
    device move). `dgemma/model.py` stays ComfyUI-agnostic (ADR-CDG-003) —
    `check_interrupted` is a plain zero-arg callable, never a `comfy`
    import; the surface-side translation into ComfyUI's own
    `InterruptProcessingException` lives in `surfaces/comfyui/loader.py`
    (see `tests/test_loader_interrupt.py`)."""

    def _install_fakes(self, monkeypatch, captured: dict, hf_device_map=None):
        def fake_from_pretrained(repo_id, **kwargs):
            captured.setdefault("from_pretrained_calls", []).append(repo_id)
            return FakeHfModel(hf_device_map=hf_device_map, first_param_device="cpu")

        def fake_processor_from_pretrained(repo_id, **kwargs):
            captured.setdefault("processor_calls", []).append(repo_id)
            return FakeProcessor()

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr("dgemma.model.AutoProcessor.from_pretrained", fake_processor_from_pretrained)
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)

    def test_none_check_interrupted_never_interrupts(self, monkeypatch):
        """The default (`None`, every non-ComfyUI caller — tests, MCP, direct
        script use): load proceeds exactly as before this parameter existed."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id="fake/repo", quant="none", check_interrupted=None)

        assert isinstance(result, DGemmaModel)
        assert captured["from_pretrained_calls"] == ["fake/repo"]
        assert captured["processor_calls"] == ["fake/repo"]

    def test_predicate_reporting_false_never_interrupts(self, monkeypatch):
        """An always-False predicate (polled repeatedly across all four
        boundaries) must behave identically to `None` — load completes."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        poll_count = {"n": 0}

        def never_interrupt():
            poll_count["n"] += 1
            return False

        result = load_model(repo_id="fake/repo", quant="none", check_interrupted=never_interrupt)

        assert isinstance(result, DGemmaModel)
        # Polled at all four phase boundaries (pre-flight, model, processor, device move).
        assert poll_count["n"] == 4

    def test_interrupted_on_first_poll_raises_before_any_blocking_call(self, monkeypatch):
        """A predicate reporting True on its FIRST poll (the quant/checkpoint
        pre-flight boundary, the earliest phase) must raise `LoadInterrupted`
        before the blocking model `from_pretrained` call ever starts — the
        whole point of polling at phase boundaries rather than only at the
        end."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        with pytest.raises(LoadInterrupted, match="pre-flight"):
            load_model(repo_id="fake/repo", quant="none", check_interrupted=lambda: True)

        assert "from_pretrained_calls" not in captured
        assert "processor_calls" not in captured

    def test_interrupted_before_model_from_pretrained_skips_it(self, monkeypatch):
        """A predicate that passes the pre-flight poll but reports True at
        the next boundary must stop before the model `from_pretrained` call
        — the pre-flight guard's own (best-effort, real-network) config read
        must have already happened, but nothing further."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        calls = {"n": 0}

        def interrupt_on_second_poll():
            calls["n"] += 1
            return calls["n"] >= 2  # poll 1: pre-flight (False); poll 2: model (True)

        with pytest.raises(LoadInterrupted, match="model from_pretrained"):
            load_model(repo_id="fake/repo", quant="none", check_interrupted=interrupt_on_second_poll)

        assert "from_pretrained_calls" not in captured
        assert "processor_calls" not in captured

    def test_interrupted_between_model_and_processor_load_skips_processor(self, monkeypatch):
        """A predicate that flips True only after the model `from_pretrained`
        call has already returned must stop BEFORE the processor
        `from_pretrained` call — proving the poll is genuinely per-boundary,
        not just a single check at entry."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        calls = {"n": 0}

        def interrupt_after_first_two_polls():
            calls["n"] += 1
            # Poll 1: pre-flight (False). Poll 2: before model from_pretrained
            # (False, let the model load start). Poll 3: before processor
            # load (True, stop here).
            return calls["n"] >= 3

        with pytest.raises(LoadInterrupted, match="processor from_pretrained"):
            load_model(
                repo_id="fake/repo", quant="none", check_interrupted=interrupt_after_first_two_polls
            )

        assert captured["from_pretrained_calls"] == ["fake/repo"]  # model DID load
        assert "processor_calls" not in captured  # processor load never started

    def test_interrupted_before_device_move_skips_to_cuda(self, monkeypatch):
        """A predicate that flips True only at the final boundary must stop
        before `.to("cuda")` — model + processor both already loaded, but the
        device move itself must never be reached."""
        captured: dict = {}
        moved_to_cuda = []

        class _TrackedFakeHfModel(FakeHfModel):
            def to(self, *args, **kwargs):
                moved_to_cuda.append(args)
                return self

        def fake_from_pretrained(repo_id, **kwargs):
            return _TrackedFakeHfModel(hf_device_map={"model.layers.0": 0}, first_param_device="cpu")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained", fake_from_pretrained
        )
        monkeypatch.setattr(
            "dgemma.model.AutoProcessor.from_pretrained", lambda repo_id, **kw: FakeProcessor()
        )
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)

        calls = {"n": 0}

        def interrupt_on_fourth_poll():
            calls["n"] += 1
            return calls["n"] >= 4  # pre-flight, model, processor all pass; device move stops

        with pytest.raises(LoadInterrupted):
            load_model(repo_id="fake/repo", quant="none", check_interrupted=interrupt_on_fourth_poll)

        assert moved_to_cuda == []

    def test_load_interrupted_names_the_device_move_phase(self, monkeypatch):
        """The raised message names which phase boundary the interrupt was
        caught at — actionable for anyone reading a log, not just a bare
        'interrupted'. Exercised at the LAST boundary (`.to("cuda")`) as the
        distinct case from the first-poll test above, so the message-naming
        behavior is proven for an interior/final phase too, not only phase 1."""
        captured: dict = {}
        self._install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        calls = {"n": 0}

        def interrupt_on_fourth_poll():
            calls["n"] += 1
            return calls["n"] >= 4

        with pytest.raises(LoadInterrupted, match=r'device move \(\.to\("cuda"\)\)'):
            load_model(repo_id="fake/repo", quant="none", check_interrupted=interrupt_on_fourth_poll)


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
