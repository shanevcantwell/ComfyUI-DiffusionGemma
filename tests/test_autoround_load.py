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


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _install_fakes(monkeypatch, captured: dict, hf_device_map=None, raise_on=None):
    """Monkeypatch transformers' from_pretrained calls to capture kwargs.

    Also pins `torch.cuda.is_available()` True (#159 gate finding): these
    fakes drive `load_model()` to its real `.to("cuda")` call, which is a
    no-op against `FakeHfModel.to` — but the no-CUDA guard upstream of it
    (issue #143) is a real host check, so without this pin these tests pass
    on a CUDA host and fail on CPU-only CI for a reason that has nothing to
    do with what they're testing. Tests that specifically exercise the
    no-CUDA branch override this back to False after calling this helper —
    monkeypatch's later-wins semantics keep that override intact."""

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

    def test_autoround_does_not_use_device_map(self, monkeypatch):
        """dd2767c dropped device_map="auto" entirely (tied params crash
        under CPU spill) — both quant modes now load with low_cpu_mem_usage
        =False and a single explicit .to("cuda") after, not accelerate
        dispatch. No device_map key is passed to from_pretrained."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        load_model(repo_id=None, quant="autoround")

        assert "device_map" not in captured["kwargs"]
        assert captured["kwargs"]["low_cpu_mem_usage"] is False

    def test_autoround_returns_int4_dtype_label(self, monkeypatch):
        """The returned DGemmaModel has dtype='int4' for autoround loads."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=None, quant="autoround")

        assert result.dtype == "int4"
        assert result.quant == "autoround"


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
        assert "device_map" not in captured["kwargs"]
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
