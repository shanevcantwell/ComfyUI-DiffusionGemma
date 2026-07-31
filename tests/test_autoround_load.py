"""AutoRound INT4 load-path unit coverage (quant="autoround").

No weights, no GPU, no auto-round runtime needed — monkeypatches
`DiffusionGemmaForBlockDiffusion.from_pretrained` + `AutoProcessor.from_pretrained`
to verify the kwargs shape, patch application, and repo_id auto-selection.

The autoround path was added in commit 193edd3 (issue #128) but had no test
coverage because it requires a pre-quantized checkpoint + GPU to exercise live.
These tests verify the load-path mechanics (patches applied, correct kwargs,
repo_id resolution) without needing the actual INT4 weights.

Cross-references:
- dgemma/model.py `_apply_autoround_patches()` — three patches for transformers/auto-round
- dgemma/model.py `load_model()` — repo_id auto-selection per quant mode
- handoff 2026-07-23-int4-autoround-loaded.md — verified load on RTX-8000
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from dgemma.model import (
    AUTOROUND_REPO_ID,
    DEFAULT_QUANT,
    DEFAULT_REPO_ID,
    _apply_autoround_patches,
    load_model,
)


# ---------------------------------------------------------------------------
# Fakes — stand in for transformers' loaded model + processor
# ---------------------------------------------------------------------------

class _FakeDevice:
    """Minimal stand-in for `torch.device` — `.type` is what
    `_assert_no_meta_tensors` inspects; `str()` is what `_resolve_device`
    compares against."""

    def __init__(self, device_str: str):
        self._device_str = device_str
        self.type = device_str.split(":")[0]

    def __str__(self):
        return self._device_str


class _FakeParam:
    def __init__(self, device: str):
        self.device = _FakeDevice(device)


class FakeHfModel:
    """Stands in for a loaded `DiffusionGemmaForBlockDiffusion`. `.to()` +
    `named_parameters()` + `named_buffers()` are what `load_model`'s
    post-load meta-tensor assertion (issue #142) and the final
    `.to("cuda")` call touch — no `lm_head` attribute, so `_retie_lm_head`'s
    autoround re-tie step no-ops on these fakes (getattr(model, "lm_head",
    None) is None) rather than needing a full tied-weight fake."""

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


class FakeProcessor:
    """Stands in for `AutoProcessor.from_pretrained`'s return value."""


class _FakeNvmlProcess:
    """Stands in for the per-process struct
    `nvmlDeviceGetComputeRunningProcesses` returns: `.pid` + `.usedGpuMemory`
    (bytes) are the two fields `_gpu_memory_holders` (dgemma/model.py,
    issue #191) reads off each entry."""

    def __init__(self, pid: int, used_gpu_memory: int):
        self.pid = pid
        self.usedGpuMemory = used_gpu_memory


class FakePynvmlModule:
    """A faked `pynvml` module (issue #191's "faked NVML layer" test
    requirement) — installed into `sys.modules["pynvml"]` so
    `_gpu_memory_holders`'s local `import pynvml` resolves to this fake
    instead of touching real hardware. `procs`: the list of `_FakeNvmlProcess`
    `nvmlDeviceGetComputeRunningProcesses` should report; `names`: a
    `{pid: name}` map `nvmlSystemGetProcessName` looks up.
    """

    def __init__(self, procs: "list[_FakeNvmlProcess]", names: "dict[int, str]"):
        self._procs = procs
        self._names = names
        self.init_called = False
        self.shutdown_called = False

    def nvmlInit(self):
        self.init_called = True

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetHandleByIndex(self, index: int):
        return index

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        return self._procs

    def nvmlSystemGetProcessName(self, pid: int):
        return self._names[pid]


class RaisingFakePynvmlModule:
    """A faked `pynvml` module whose `nvmlInit`/enumeration calls raise —
    stands in for a real driver-not-loaded/enumeration failure so the guard's
    honest-degradation path (issue #191: "occupants unmeasurable: <reason>")
    is exercised without needing an actual broken driver."""

    def __init__(self, raise_at: str, error: Exception):
        self._raise_at = raise_at
        self._error = error

    def nvmlInit(self):
        if self._raise_at == "init":
            raise self._error

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index: int):
        if self._raise_at == "handle":
            raise self._error
        return index

    def nvmlDeviceGetComputeRunningProcesses(self, handle):
        if self._raise_at == "enumerate":
            raise self._error
        return []

    def nvmlSystemGetProcessName(self, pid: int):
        return "unused"


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _install_fakes(monkeypatch, captured: dict, hf_device_map=None, raise_on=None):
    """Monkeypatch transformers' from_pretrained calls to capture kwargs.

    Also pins `torch.cuda.is_available()` True (#159 gate finding): the
    no-CUDA guard in `load_model` (issue #143) is a real host check, so
    without this pin these tests pass on a CUDA host and fail on CPU-only
    CI for a reason that has nothing to do with what they're testing. Tests
    that specifically exercise the no-CUDA branch override this back to
    False after calling this helper — monkeypatch's later-wins semantics
    keep that override intact.

    Also pins `torch.cuda.mem_get_info` to a roomy fake (48 GiB free): the
    autoround pre-load VRAM precondition (issue #183, split-fails probe
    outcome) reads it on CUDA-available hosts, and on CPU-only CI the
    `is_available` pin above would otherwise send the precondition into a
    real `mem_get_info` call that raises. Tests that specifically exercise
    the precondition override this pin with their own values.

    Also pins `torch.cuda.current_device` to `0` (issue #191): the
    precondition's failure path reads this to pick which NVML device index
    to enumerate holders for; a real CUDA host's actual index doesn't matter
    to any test here, only that the call doesn't hit real hardware state."""

    def fake_from_pretrained(repo_id, **kwargs):
        if raise_on == "model":
            raise OSError(f"{repo_id} is not a valid model identifier")
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return FakeHfModel(hf_device_map=hf_device_map, first_param_device="cpu")

    def fake_processor_from_pretrained(repo_id, **kwargs):
        if raise_on == "processor":
            raise OSError(f"{repo_id} is not a valid model identifier")
        captured["processor_repo_id"] = repo_id
        captured["processor_kwargs"] = kwargs
        return FakeProcessor()

    monkeypatch.setattr(
        "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
        fake_from_pretrained,
    )
    monkeypatch.setattr(
        "dgemma.model.AutoProcessor.from_pretrained",
        fake_processor_from_pretrained,
    )
    monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "dgemma.model.torch.cuda.mem_get_info",
        lambda *a, **k: (48 * 1024**3, 48 * 1024**3),
    )
    monkeypatch.setattr("dgemma.model.torch.cuda.current_device", lambda: 0)


# ---------------------------------------------------------------------------
# Test: repo_id auto-selection
# ---------------------------------------------------------------------------

class TestRepoIdAutoSelection:
    """load_model() auto-selects the checkpoint matching quant mode when
    no explicit repo_id is given (repo_id=None)."""

    def test_none_repo_with_quant_none_uses_default_repo(self, monkeypatch):
        """quant="none" + repo_id=None → DEFAULT_REPO_ID (Google bf16)."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model(repo_id=None, quant="none")

        assert captured["repo_id"] == DEFAULT_REPO_ID

    def test_none_repo_with_quant_autoround_uses_intel_repo(self, monkeypatch):
        """quant="autoround" + repo_id=None → AUTOROUND_REPO_ID (Intel INT4)."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model(repo_id=None, quant="autoround")

        assert captured["repo_id"] == AUTOROUND_REPO_ID
        assert "Intel" in captured["repo_id"]
        assert "AutoRound" in captured["repo_id"]

    def test_explicit_repo_overrides_auto_selection(self, monkeypatch):
        """An explicit repo_id is never overridden by auto-selection."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        custom_repo = "my-org/my-custom-checkpoint"
        load_model(repo_id=custom_repo, quant="autoround")

        assert captured["repo_id"] == custom_repo

    def test_default_call_uses_bf16_path(self, monkeypatch):
        """Calling load_model() with no args uses DEFAULT_REPO_ID + bf16."""
        import torch

        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model()

        assert captured["repo_id"] == DEFAULT_REPO_ID
        assert captured["kwargs"]["dtype"] == torch.bfloat16


# ---------------------------------------------------------------------------
# Test: autoround kwargs shape
# ---------------------------------------------------------------------------

class TestAutoroundKwargsShape:
    """quant="autoround" produces the correct load_kwargs for transformers."""

    def test_autoround_uses_dtype_auto(self, monkeypatch):
        """dtype="auto" lets transformers read the quantization config from
        the checkpoint's config.json — required for AutoRound W4A16 loading."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model(repo_id=None, quant="autoround")

        assert captured["kwargs"]["dtype"] == "auto"

    def test_autoround_returns_int4_dtype_label(self, monkeypatch):
        """The returned DGemmaModel has dtype='int4' for autoround loads."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=None, quant="autoround")

        assert result.dtype == "int4"
        assert result.quant == "autoround"


# ---------------------------------------------------------------------------
# Test: placement UNIFORMITY across quant modes (issue #183)
# ---------------------------------------------------------------------------

class TestPlacementUniformity:
    """UNIFORMITY mutation guards (issue #183) — the retargeted successors of
    the per-quant guards `test_autoround_does_not_use_device_map` and
    `test_autoround_calls_to_cuda` (which pinned the divergent placement
    branch #183's field report falsified).

    There is ONE load path: `load_kwargs` for every quant mode is identical
    except `dtype` (checkpoint identity), and NO quant mode calls
    `.to("cuda")` — accelerate places everything during `from_pretrained`.

    Mutation-verified by construction: both quant values run through the
    SAME parametrized assertion body, and the exact-key-set assertion means
    reintroducing ANY quant-conditional placement kwarg (for these quants or
    a future third one wired through this parametrize list) reddens
    `test_placement_kwargs_uniform` by name; re-adding a `.to("cuda")` for
    either quant reddens `test_neither_quant_calls_to_cuda` by name."""

    # The one legitimate per-quant residue: dtype, a checkpoint-identity
    # fact ("auto" reads the INT4 checkpoint's own quantization_config).
    EXPECTED_DTYPE = {"autoround": "auto", "none": None}  # None → torch.bfloat16

    @pytest.mark.parametrize("quant", ["autoround", "none"])
    def test_placement_kwargs_uniform(self, monkeypatch, quant):
        """SAME assertion body for both quants: `device_map="auto"`, no
        `low_cpu_mem_usage`, `local_files_only` threaded — and NOTHING
        else besides `dtype`. The exact-key-set assertion is the strong
        pin: a quant-special kwarg cannot be added for either mode (or a
        future third) without failing here."""
        import torch

        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model(repo_id=None, quant=quant)

        kwargs = captured["kwargs"]
        assert kwargs["device_map"] == "auto"
        assert "low_cpu_mem_usage" not in kwargs
        assert kwargs["local_files_only"] is False
        assert set(kwargs) == {"device_map", "dtype", "local_files_only"}
        expected_dtype = self.EXPECTED_DTYPE[quant] or torch.bfloat16
        assert kwargs["dtype"] == expected_dtype

    def test_autoround_and_none_receive_identical_placement_kwargs(
        self, monkeypatch
    ):
        """Direct pairwise pin: capture both quants' load_kwargs and assert
        they are EQUAL once `dtype` (the one checkpoint-identity residue) is
        set aside — placement never varies by quant."""
        captured_autoround: dict = {}
        captured_none: dict = {}
        _install_fakes(
            monkeypatch, captured_autoround, hf_device_map={"model.layers.0": 0}
        )
        load_model(repo_id=None, quant="autoround")
        _install_fakes(
            monkeypatch, captured_none, hf_device_map={"model.layers.0": 0}
        )
        load_model(repo_id=None, quant="none")

        kwargs_autoround = dict(captured_autoround["kwargs"])
        kwargs_none = dict(captured_none["kwargs"])
        assert kwargs_autoround.pop("dtype") != kwargs_none.pop("dtype")
        assert kwargs_autoround == kwargs_none

    @pytest.mark.parametrize("quant", ["autoround", "none"])
    def test_neither_quant_calls_to_cuda(self, monkeypatch, quant):
        """NO quant mode calls `.to("cuda")` (issue #183): accelerate has
        already placed every tensor under `device_map="auto"`; a whole-model
        `.to("cuda")` would pull CPU-spilled weights back onto the card
        (the OOM mechanism PR #177 fixed, issue #173). Mutation guard:
        re-adding `model.to("cuda")` for EITHER quant leaves `calls`
        non-empty and fails this test by name."""
        calls: list = []

        class _TrackedFakeHfModel(FakeHfModel):
            def to(self, *args, **kwargs):
                calls.append(args)
                return self

        def fake_from_pretrained(repo_id, **kwargs):
            return _TrackedFakeHfModel(
                hf_device_map={"model.layers.0": 0}, first_param_device="cpu"
            )

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            fake_from_pretrained,
        )
        monkeypatch.setattr(
            "dgemma.model.AutoProcessor.from_pretrained",
            lambda repo_id, **kw: FakeProcessor(),
        )
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (48 * 1024**3, 48 * 1024**3),
        )

        load_model(repo_id=None, quant=quant)

        assert calls == []


# ---------------------------------------------------------------------------
# Test: autoround pre-load VRAM precondition (issue #183, split-fails arm)
# ---------------------------------------------------------------------------

class TestAutoroundVramPrecondition:
    """The ONE legitimate remaining quant-conditional besides dtype: the
    fail-loud pre-load VRAM check for quant='autoround' (issue #183 —
    shipped because the forced-split probe proved a CPU/GPU split of the
    INT4 checkpoint crashes inside from_pretrained; see docs/experiments/
    2026-07-30-autoround-unified-path-split-check/).

    Issue #191: the failure message reports the ONE measured condition
    (required floor, measured free, measured per-process holder list) with
    at most one remedy line derived from that measured state — never a
    menu of hypothetical causes/remedies. These tests fake the NVML layer
    (`sys.modules["pynvml"]`) rather than touching real hardware."""

    def test_autoround_precheck_rejects_insufficient_vram(self, monkeypatch):
        """With too little free VRAM, quant='autoround' raises RuntimeError
        BEFORE from_pretrained (never dispatches the split that cannot
        load), naming required and available bytes."""
        from dgemma.model import AUTOROUND_MIN_FREE_VRAM_BYTES

        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        # Override the roomy default pin: 8 GiB free < the ~30 GiB floor.
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (8 * 1024**3, 48 * 1024**3),
        )
        fake_pynvml = FakePynvmlModule(procs=[], names={})
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        message = str(excinfo.value)
        # Raised BEFORE from_pretrained: the fake never captured a call.
        assert "kwargs" not in captured
        # Names both numbers...
        assert f"{AUTOROUND_MIN_FREE_VRAM_BYTES / 1024**3:.1f}" in message
        assert "8.0" in message
        # ...and exactly one remedy line, not a menu of hypothetical fixes.
        assert message.count("Remedy:") == 1
        assert "no per-process GPU memory holders reported" in message
        # No speculative cause-prose survives (the pre-#191 message named a
        # CPU/GPU split crash and issue #183 by number as a *possible*
        # cause; the new message states only the measured condition).
        assert "docs/experiments" not in message
        assert "#183" not in message

    def test_autoround_precheck_reports_measured_state(self, monkeypatch):
        """The message states the measured floor/free numbers and the
        measured holder report — built from the faked NVML layer's actual
        return values, not prose."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (8 * 1024**3, 48 * 1024**3),
        )
        fake_pynvml = FakePynvmlModule(
            procs=[_FakeNvmlProcess(pid=4242, used_gpu_memory=1402 * 1024 * 1024)],
            names={4242: "/usr/bin/llama-server"},
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        message = str(excinfo.value)
        assert fake_pynvml.init_called
        assert fake_pynvml.shutdown_called
        assert "4242" in message
        assert "llama-server" in message
        assert "1402" in message

    def test_autoround_precheck_identifies_self_vs_foreign(self, monkeypatch):
        """A holder whose pid matches this process's own (`os.getpid()`) is
        reported as this process; a different pid is reported as foreign —
        the guard's own prior in-process load is distinguished from a
        foreign tenant, per the issue's acceptance criteria."""
        import os

        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (8 * 1024**3, 48 * 1024**3),
        )
        own_pid = os.getpid()
        foreign_pid = own_pid + 1
        fake_pynvml = FakePynvmlModule(
            procs=[
                _FakeNvmlProcess(pid=own_pid, used_gpu_memory=5 * 1024 * 1024 * 1024),
                _FakeNvmlProcess(pid=foreign_pid, used_gpu_memory=2 * 1024 * 1024 * 1024),
            ],
            names={own_pid: "python3", foreign_pid: "some-other-process"},
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        message = str(excinfo.value)
        assert f"pid {own_pid}" in message
        assert "this process" in message
        assert f"pid {foreign_pid}" in message
        assert "foreign process" in message
        # A foreign tenant is present: the derived remedy names unloading it,
        # not the generic "free VRAM" line.
        assert "unload the foreign process" in message

    def test_autoround_precheck_degrades_honestly_without_nvml(self, monkeypatch):
        """When `pynvml` is not importable, the message says occupants are
        unmeasurable and names why — it does not fall back to the old
        hypothetical cause/remedy prose."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (8 * 1024**3, 48 * 1024**3),
        )
        monkeypatch.setitem(sys.modules, "pynvml", None)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        message = str(excinfo.value)
        assert "occupants unmeasurable" in message
        assert "docs/experiments" not in message
        assert "CPU/GPU split" not in message

    def test_autoround_precheck_degrades_honestly_on_nvml_init_failure(self, monkeypatch):
        """A `pynvml` that imports but fails during `nvmlInit` (e.g. driver
        not loaded) also reports unmeasurable-with-reason, not a crash and
        not a fallback to hypothetical prose."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (8 * 1024**3, 48 * 1024**3),
        )
        fake_pynvml = RaisingFakePynvmlModule(
            raise_at="init", error=RuntimeError("driver not loaded")
        )
        monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        message = str(excinfo.value)
        assert "occupants unmeasurable" in message
        assert "driver not loaded" in message

    def test_autoround_precheck_passes_with_sufficient_vram(self, monkeypatch):
        """With enough free VRAM the precondition is silent and the load
        proceeds normally."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=None, quant="autoround")

        assert captured["kwargs"]["device_map"] == "auto"
        assert result.quant == "autoround"

    def test_quant_none_has_no_vram_precondition(self, monkeypatch):
        """bf16 (quant='none') loads regardless of free VRAM — its CPU-spill
        path is field-proven (docs/experiments/bf16-fit-mechanism/), so the
        precondition must never gate it."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (8 * 1024**3, 48 * 1024**3),
        )

        result = load_model(repo_id=None, quant="none")

        assert captured["kwargs"]["device_map"] == "auto"
        assert result.quant == "none"


# ---------------------------------------------------------------------------
# Test: _apply_autoround_patches mechanics
# ---------------------------------------------------------------------------

class TestApplyAutoroundPatches:
    """Verify the three patches are applied to the correct targets, and (H2
    — issue #142's standing recommendation) that `_apply_autoround_patches`
    is a scoped context manager: active only inside the `with` block,
    restored on exit — not a permanent global monkeypatch."""

    def test_patches_kv_cache_warmup(self):
        """Patch 2: caching_allocator_warmup is replaced with a no-op to
        prevent bf16-sized buffer pre-allocation (~46GB) before knowing
        weights are INT4 (~30GB)."""
        from transformers import modeling_utils

        original = modeling_utils.caching_allocator_warmup
        with _apply_autoround_patches():
            patched = modeling_utils.caching_allocator_warmup
            assert patched is not original
            # Calling it should not raise and should return None
            assert patched() is None

    def test_patches_mark_tied_weights_as_initialized(self):
        """Patch 3a: mark_tied_weights_as_initialized is wrapped to handle
        quantized modules that have .qweight instead of .weight."""
        from transformers import modeling_utils

        original = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized
        with _apply_autoround_patches():
            patched = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized
            assert patched is not original

    def test_patches_tie_weights(self):
        """Patch 3b: tie_weights is wrapped to handle quantized modules."""
        from transformers import modeling_utils

        original = modeling_utils.PreTrainedModel.tie_weights
        with _apply_autoround_patches():
            patched = modeling_utils.PreTrainedModel.tie_weights
            assert patched is not original

    def test_patches_are_restored_on_clean_exit(self):
        """H2: the patches are load-time-only hooks — they must not survive
        past the `with` block, so a caller outside the autoround load path
        is never affected by them."""
        from transformers import modeling_utils

        original_warmup = modeling_utils.caching_allocator_warmup
        original_mark = modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized
        original_tie = modeling_utils.PreTrainedModel.tie_weights

        with _apply_autoround_patches():
            assert modeling_utils.caching_allocator_warmup is not original_warmup

        assert modeling_utils.caching_allocator_warmup is original_warmup
        assert modeling_utils.PreTrainedModel.mark_tied_weights_as_initialized is original_mark
        assert modeling_utils.PreTrainedModel.tie_weights is original_tie

    def test_patches_are_restored_even_if_body_raises(self):
        """Restore-on-exit must be exception-safe (try/finally), not just
        the happy path — a from_pretrained failure inside the with block
        must not leave transformers permanently patched."""
        from transformers import modeling_utils

        original_warmup = modeling_utils.caching_allocator_warmup

        with pytest.raises(ValueError):
            with _apply_autoround_patches():
                assert modeling_utils.caching_allocator_warmup is not original_warmup
                raise ValueError("simulated from_pretrained failure")

        assert modeling_utils.caching_allocator_warmup is original_warmup

    def test_nested_application_is_guarded_not_double_wrapped(self):
        """Reentrancy guard: a nested `with _apply_autoround_patches():`
        (e.g. a caller wrapping load_model, which also applies the patches)
        must not double-wrap, and the inner exit must not restore originals
        that the outer scope still needs active."""
        from transformers import modeling_utils

        original_warmup = modeling_utils.caching_allocator_warmup

        with _apply_autoround_patches():
            outer_patched = modeling_utils.caching_allocator_warmup
            assert outer_patched is not original_warmup

            with _apply_autoround_patches():
                # Inner scope sees the same patch, not a re-wrap
                assert modeling_utils.caching_allocator_warmup is outer_patched

            # Inner exit must not have restored the original — outer is
            # still inside its own `with` block.
            assert modeling_utils.caching_allocator_warmup is outer_patched

        # Outer exit does restore.
        assert modeling_utils.caching_allocator_warmup is original_warmup


# ---------------------------------------------------------------------------
# Test: auto-round missing error messaging
# ---------------------------------------------------------------------------

class TestAutoroundMissingError:
    """When quant='autoround' but auto-round is not installed, surface an
    actionable RuntimeError instead of a raw ImportError from deep in
    transformers/accelerate."""

    def test_import_error_during_autoround_load_is_wrapped(self, monkeypatch):
        """If from_pretrained raises ImportError (auto-round not installed),
        load_model wraps it with an actionable message naming the fix command."""
        def raising_from_pretrained(repo_id, **kwargs):
            raise ImportError("No module named 'auto_round'")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            raising_from_pretrained,
        )
        # Issue #135 hermeticity fix: pin the autoround pre-load VRAM
        # precondition's own CUDA touchpoints (`_assert_autoround_vram_
        # precondition`, issue #183) so this test exercises the
        # from_pretrained ImportError path unconditionally, not whatever
        # `torch.cuda.mem_get_info()` happens to report on the host running
        # the suite (real host state under GPU-tenant contention; a raw
        # RuntimeError on CPU-only CI once `is_available` is no longer
        # pinned True elsewhere).
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (48 * 1024**3, 48 * 1024**3),
        )

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        message = str(excinfo.value)
        assert "auto-round" in message.lower() or "autoround" in message.lower()
        assert "pip install" in message
        assert "[quant]" in message or "auto-round" in message

    def test_import_error_during_autoround_is_chained(self, monkeypatch):
        """The original ImportError is chained as __cause__ for debugging."""
        def raising_from_pretrained(repo_id, **kwargs):
            raise ImportError("No module named 'auto_round'")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            raising_from_pretrained,
        )
        # Issue #135 hermeticity fix: see sibling test above — pin the VRAM
        # precondition's CUDA touchpoints so this test isn't at the mercy of
        # host VRAM occupancy or CUDA availability.
        monkeypatch.setattr("dgemma.model.torch.cuda.is_available", lambda: True)
        monkeypatch.setattr(
            "dgemma.model.torch.cuda.mem_get_info",
            lambda *a, **k: (48 * 1024**3, 48 * 1024**3),
        )

        with pytest.raises(RuntimeError) as excinfo:
            load_model(repo_id=None, quant="autoround")

        assert isinstance(excinfo.value.__cause__, ImportError)

    def test_import_error_with_quant_none_propagates_as_is(self, monkeypatch):
        """An ImportError during quant='none' (not autoround) is NOT wrapped —
        it's a genuine bug, not a missing optional dependency."""
        def raising_from_pretrained(repo_id, **kwargs):
            raise ImportError("some unrelated import issue")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            raising_from_pretrained,
        )

        with pytest.raises(ImportError, match="some unrelated import issue"):
            load_model(repo_id=None, quant="none")


# ---------------------------------------------------------------------------
# Test: autoround end-to-end (mocked)
# ---------------------------------------------------------------------------

class TestAutoroundEndToEnd:
    """Full load_model() flow for quant='autoround' with mocked transformers.
    
    Verifies the complete path: patches applied → correct kwargs → 
    Intel checkpoint selected → DGemmaModel returned with int4 dtype."""

    def test_full_autoround_load_flow(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=None, quant="autoround")

        # Repo auto-selected to Intel INT4 checkpoint
        assert captured["repo_id"] == AUTOROUND_REPO_ID
        # dtype="auto" for transformers to read quantization config
        assert captured["kwargs"]["dtype"] == "auto"
        # Unified placement (issue #183): same device_map="auto" as bf16
        assert captured["kwargs"]["device_map"] == "auto"
        # Processor called with same repo
        assert captured["processor_repo_id"] == AUTOROUND_REPO_ID
        # Result has correct dtype label
        assert result.dtype == "int4"
        assert result.quant == "autoround"
        assert result.repo_id == AUTOROUND_REPO_ID

    def test_autoround_with_local_files_only(self, monkeypatch):
        """local_files_only threads through both model and processor calls."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model(repo_id=None, quant="autoround", local_files_only=True)

        assert captured["kwargs"]["local_files_only"] is True
        assert captured["processor_kwargs"]["local_files_only"] is True


# ---------------------------------------------------------------------------
# Test: INT4 tied-weight retie + spill exemption under the unified path
# (issue #183 steps 6-7 — #119's corruption CLASS on the INT4 path, distinct
# from #119's own unrelated sampler-side shape bug; NOT a #119 closure claim)
# ---------------------------------------------------------------------------

class _FakeTensorParam:
    """A parameter-like fake with `.device` — identity-comparable so the
    retie test can assert lm_head and embed_tokens share the SAME object."""

    def __init__(self, device: str):
        self.device = _FakeDevice(device)


class _FakeLmHead:
    def __init__(self, weight):
        self.weight = weight


class FakeTiedModel:
    """Fake with a real tied-weight surface: `lm_head.weight` (possibly
    meta) + `get_parameter('model.decoder.embed_tokens.weight')` — the
    exact pair `_retie_lm_head` repairs (issue #142), now exercised on its
    real branch (the prior FakeHfModel had no lm_head, so retie always
    no-opped — the untested-branch gap all three plan passes named)."""

    def __init__(self, lm_head_device="meta", embed_device="cpu", hf_device_map=None):
        self._embed_param = _FakeTensorParam(embed_device)
        self.lm_head = _FakeLmHead(_FakeTensorParam(lm_head_device))
        if hf_device_map is not None:
            self.hf_device_map = hf_device_map

    def get_parameter(self, name):
        assert name == "model.decoder.embed_tokens.weight"
        return self._embed_param

    def named_parameters(self):
        yield "lm_head.weight", self.lm_head.weight
        yield "model.decoder.embed_tokens.weight", self._embed_param

    def named_buffers(self):
        return iter(())


class TestRetieUnderUnifiedPlacement:
    """`_retie_lm_head` + `_assert_no_meta_tensors` on the INT4 path under
    the unified device_map="auto" placement (issue #183 step 6). Covers the
    offload-exempt case regardless of whether natural spill or a forced
    split triggers it in practice."""

    def test_retie_points_meta_lm_head_at_real_embed_tokens(self):
        """meta lm_head.weight + real embed_tokens.weight → after retie,
        both names resolve to the SAME object (a genuine tie shares
        storage, never a copy)."""
        from dgemma.model import _retie_lm_head

        model = FakeTiedModel(lm_head_device="meta", embed_device="cpu")
        assert model.lm_head.weight is not model._embed_param

        _retie_lm_head(model)

        assert model.lm_head.weight is model._embed_param

    def test_retie_noop_when_lm_head_already_real(self):
        """A real (non-meta) lm_head.weight is left untouched — retie is a
        targeted repair, not a general rebinder."""
        from dgemma.model import _retie_lm_head

        model = FakeTiedModel(lm_head_device="cuda:0", embed_device="cuda:0")
        original = model.lm_head.weight

        _retie_lm_head(model)

        assert model.lm_head.weight is original

    def test_retie_leaves_meta_when_source_also_meta(self):
        """Both sides meta → retie declines (pointing one meta tensor at
        another fixes nothing); `_assert_no_meta_tensors` then raises, by
        name, as the enforcement surface."""
        from dgemma.model import _assert_no_meta_tensors, _retie_lm_head

        model = FakeTiedModel(lm_head_device="meta", embed_device="meta")

        _retie_lm_head(model)

        assert model.lm_head.weight is not model._embed_param
        with pytest.raises(RuntimeError, match="lm_head.weight"):
            _assert_no_meta_tensors(model)

    def test_offload_exempt_meta_lm_head_does_not_raise(self):
        """A meta lm_head whose owning module hf_device_map places on 'cpu'
        is accelerate's legitimate mmap spill (PR #177's exemption) — no
        raise, proven to extend to the INT4 path (issue #183 step 7)."""
        from dgemma.model import _assert_no_meta_tensors

        model = FakeTiedModel(
            lm_head_device="meta",
            embed_device="meta",
            hf_device_map={"lm_head": "cpu", "model.decoder.embed_tokens": "cpu"},
        )

        _assert_no_meta_tensors(model)  # must not raise

    def test_stranded_meta_still_raises_for_autoround_load(self, monkeypatch):
        """Through the full load_model(quant='autoround') flow: a meta param
        whose device-map entry is an ACCELERATOR (not cpu/disk) is stranded,
        not spill — still raises (issue #183 step 7's still-raises half)."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"fake": 0})

        def fake_from_pretrained(repo_id, **kwargs):
            return FakeHfModel(hf_device_map={"fake": 0}, first_param_device="meta")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            fake_from_pretrained,
        )

        with pytest.raises(RuntimeError, match="meta-resident"):
            load_model(repo_id=None, quant="autoround")

    def test_offload_meta_exempt_through_autoround_load(self, monkeypatch):
        """Through the full load_model(quant='autoround') flow: a meta param
        whose owning module the device map sends to 'cpu' is exempt — the
        load succeeds (the same exemption bf16 already gets via PR #177,
        inherited by autoround now that both share device_map='auto')."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured)

        def fake_from_pretrained(repo_id, **kwargs):
            return FakeHfModel(
                hf_device_map={"fake": "cpu", "other": 0}, first_param_device="meta"
            )

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            fake_from_pretrained,
        )

        result = load_model(repo_id=None, quant="autoround")

        assert result.quant == "autoround"
        assert result.device == "cuda:0"
