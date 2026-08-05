"""PRE-FLIGHT quant/checkpoint mismatch guard coverage (issue #141).

Root cause under test: selecting an AutoRound INT4 checkpoint with
`quant="none"` used to hang permanently in `from_pretrained` — transformers
silently deserialized INT4 data (`qweight`/`qzeros`) as bf16 tensors, with
no error surface at all. The fix is `dgemma.model._check_quant_checkpoint_match`,
a lightweight `AutoConfig.from_pretrained` read of the checkpoint's
`quantization_config.quant_method`, called BEFORE the blocking
`DiffusionGemmaForBlockDiffusion.from_pretrained` in `load_model()`.

These tests never let a mismatch reach `from_pretrained` — the guard raises
first, so the module-level fakes below are never asked to complete a load
for the raise-direction tests. `AutoConfig.from_pretrained` is monkeypatched
directly (it is the guard's own dependency), independent from the
`FakeHfModel`/`_install_fakes` pattern already in `test_autoround_load.py`,
which this file also reuses (with a local `.to()` stub) for the
pass-through paths that DO reach `from_pretrained`.

Cross-references:
- dgemma/model.py `_checkpoint_quant_method` / `_check_quant_checkpoint_match` — the guard
- dgemma/model.py `load_model()` — guard is called after repo_id resolution,
  before `_apply_autoround_patches()` / `from_pretrained`
- tests/test_autoround_load.py — existing FakeHfModel/_install_fakes pattern
  (note: that fake lacks `.to()`, so several of ITS tests already fail on a
  CUDA host independent of this change — not this file's concern, see #141
  task scope)
"""
from __future__ import annotations

import pytest

from dgemma.model import (
    AUTOROUND_REPO_ID,
    DEFAULT_REPO_ID,
    _check_quant_checkpoint_match,
    load_model,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeQuantConfig:
    """Stands in for a `PretrainedConfig.quantization_config` object (some
    transformers versions return an object with a `.quant_method` attribute
    rather than a plain dict)."""

    def __init__(self, quant_method: str):
        self.quant_method = quant_method


class FakeAutoConfig:
    """Stands in for `AutoConfig.from_pretrained`'s return value."""

    def __init__(self, quantization_config=None):
        self.quantization_config = quantization_config


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
    """Stands in for a loaded `DiffusionGemmaForBlockDiffusion`.

    Adds `.to()` on top of test_autoround_load.py's FakeHfModel (which
    lacks it) so pass-through tests here can exercise load_model() to
    completion on a CUDA-present host without touching the real device
    block at model.py:317-329 (that region is #142 territory, left as-is).
    `named_parameters()`/`named_buffers()` back the post-load meta-tensor
    assertion (issue #142), which now runs on every quant path."""

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


def _install_load_fakes(monkeypatch, captured: dict, hf_device_map=None):
    """Monkeypatch transformers' from_pretrained calls to capture kwargs,
    for the pass-through (non-raising) guard tests that reach load_model()'s
    actual load. Mirrors test_autoround_load.py's _install_fakes, plus the
    `.to()` stub on FakeHfModel above.

    Also pins `torch.cuda.is_available()` True (#159 gate finding): these
    tests drive `load_model()` to its real `.to("cuda")` call, a no-op
    against `FakeHfModel.to` — but the no-CUDA guard upstream of it (issue
    #143) is a real host check, so without this pin these tests pass on a
    CUDA host and fail on CPU-only CI for a reason unrelated to what they're
    testing.

    Also pins `torch.cuda.mem_get_info` to a roomy fake (48 GiB free)
    (issue #135 — hermeticity fix, mirroring test_autoround_load.py's
    `_install_fakes`): the autoround pre-load VRAM precondition
    (`_assert_autoround_vram_precondition`, issue #183) reads it whenever
    a test here reaches `load_model(quant="autoround")`. Without this pin,
    the `is_available` True pin above sends that precondition into a REAL
    `torch.cuda.mem_get_info()` call — which raises `RuntimeError: No CUDA
    GPUs are available` on CPU-only CI (the `is_available` pin lies about
    device presence) and reads live host VRAM occupancy on a real CUDA host
    (flaking these tests under GPU-tenant contention). Neither failure mode
    has anything to do with what these tests verify."""

    def fake_from_pretrained(repo_id, **kwargs):
        captured["repo_id"] = repo_id
        captured["kwargs"] = kwargs
        return FakeHfModel(hf_device_map=hf_device_map, first_param_device="cpu")

    def fake_processor_from_pretrained(repo_id, **kwargs):
        captured["processor_repo_id"] = repo_id
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


def _patch_autoconfig(monkeypatch, *, quantization_config=None, raise_exc=None):
    """Monkeypatch transformers.AutoConfig.from_pretrained, the guard's own
    lightweight dependency — independent of the from_pretrained fakes above,
    since the guard runs before from_pretrained is ever called."""

    def fake_autoconfig_from_pretrained(repo_id, **kwargs):
        if raise_exc is not None:
            raise raise_exc
        return FakeAutoConfig(quantization_config=quantization_config)

    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained", fake_autoconfig_from_pretrained
    )


# ---------------------------------------------------------------------------
# Mismatch direction 1: checkpoint quantized, quant='none' passed
# ---------------------------------------------------------------------------

class TestQuantizedCheckpointWithQuantNone:
    """Checkpoint declares a quant_method but caller passed quant='none' —
    this is the exact hang scenario from issue #141 and must raise, not hang."""

    def test_dict_quantization_config_raises(self, monkeypatch):
        _patch_autoconfig(
            monkeypatch, quantization_config={"quant_method": "auto-round"}
        )

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match(AUTOROUND_REPO_ID, "none", False)

        message = str(excinfo.value)
        assert "auto-round" in message
        assert "quant='none'" in message
        assert "quant='autoround'" in message  # remedy names the fix

    def test_object_quantization_config_raises(self, monkeypatch):
        """Some transformers versions expose quantization_config as an
        object with .quant_method rather than a plain dict."""
        _patch_autoconfig(
            monkeypatch, quantization_config=_FakeQuantConfig("auto-round")
        )

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match(AUTOROUND_REPO_ID, "none", False)

        assert "auto-round" in str(excinfo.value)

    def test_message_names_both_tokens_and_remedy(self, monkeypatch):
        """House style (kv_cache.py V1-V6): name BOTH the violated
        precondition and the actionable remedy in one message."""
        _patch_autoconfig(
            monkeypatch, quantization_config={"quant_method": "auto-round"}
        )

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match("some/checkpoint", "none", False)

        message = str(excinfo.value)
        assert "some/checkpoint" in message
        assert "quant_method" in message
        assert "Remedy" in message

    def test_full_load_model_raises_before_from_pretrained(self, monkeypatch):
        """End-to-end: load_model() itself raises via the guard, and never
        reaches from_pretrained (which would hang on real INT4 data)."""
        _patch_autoconfig(
            monkeypatch, quantization_config={"quant_method": "auto-round"}
        )

        def unreachable_from_pretrained(repo_id, **kwargs):
            pytest.fail("from_pretrained must not be called past a guard raise")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            unreachable_from_pretrained,
        )

        with pytest.raises(RuntimeError, match="quant='none'"):
            load_model(repo_id=AUTOROUND_REPO_ID, quant="none")


# ---------------------------------------------------------------------------
# Mismatch direction 2: quant='autoround' passed, checkpoint unquantized
# ---------------------------------------------------------------------------

class TestQuantAutoroundWithUnquantizedCheckpoint:
    """Caller passed quant='autoround' but the checkpoint has no
    quantization_config at all — the autoround patches target INT4 weights
    that don't exist here."""

    def test_no_quantization_config_raises(self, monkeypatch):
        _patch_autoconfig(monkeypatch, quantization_config=None)

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match(DEFAULT_REPO_ID, "autoround", False)

        message = str(excinfo.value)
        assert "quant='autoround'" in message
        assert "quant='none'" in message  # remedy names the fix
        assert DEFAULT_REPO_ID in message

    def test_empty_dict_quantization_config_raises(self, monkeypatch):
        """An empty dict is falsy — treated the same as no config at all."""
        _patch_autoconfig(monkeypatch, quantization_config={})

        with pytest.raises(RuntimeError, match="quant='autoround'"):
            _check_quant_checkpoint_match(DEFAULT_REPO_ID, "autoround", False)

    def test_full_load_model_raises_before_from_pretrained(self, monkeypatch):
        _patch_autoconfig(monkeypatch, quantization_config=None)

        def unreachable_from_pretrained(repo_id, **kwargs):
            pytest.fail("from_pretrained must not be called past a guard raise")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            unreachable_from_pretrained,
        )

        with pytest.raises(RuntimeError, match="quant='autoround'"):
            load_model(repo_id=DEFAULT_REPO_ID, quant="autoround")


# ---------------------------------------------------------------------------
# Mismatch direction 3 (issue #210): checkpoint declares a DIFFERENT
# quant backend than the one requested — not just "quantized vs not".
# ---------------------------------------------------------------------------

class TestBackendAwareMismatch:
    """The pre-#210 guard only tested `quant_method is None` vs not-None.
    A checkpoint declaring a non-AutoRound `quant_method` (e.g. "gptq" —
    note AutoRound itself can also export GPTQ-format checkpoints) passed
    the old guard under quant='autoround' and crashed unhandled deep in
    from_pretrained. The guard must compare the declared method against the
    accepted set for the requested mode, not just check for presence."""

    def test_gptq_checkpoint_under_quant_autoround_raises(self, monkeypatch):
        """A GPTQ-format checkpoint is not an AutoRound checkpoint — the
        autoround patch path does not know how to deserialize it."""
        _patch_autoconfig(monkeypatch, quantization_config={"quant_method": "gptq"})

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match("some/gptq-checkpoint", "autoround", False)

        message = str(excinfo.value)
        assert "gptq" in message
        assert "quant='autoround'" in message
        assert "some/gptq-checkpoint" in message

    def test_gptq_checkpoint_under_quant_autoround_names_accepted_set(
        self, monkeypatch
    ):
        _patch_autoconfig(monkeypatch, quantization_config={"quant_method": "gptq"})

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match(DEFAULT_REPO_ID, "autoround", False)

        message = str(excinfo.value)
        assert "auto-round" in message  # the accepted set is named

    def test_gptq_checkpoint_under_quant_autoround_via_object_config(
        self, monkeypatch
    ):
        """Object-shaped quantization_config (some transformers versions),
        mirroring TestQuantizedCheckpointWithQuantNone's coverage."""
        _patch_autoconfig(monkeypatch, quantization_config=_FakeQuantConfig("gptq"))

        with pytest.raises(RuntimeError, match="gptq"):
            _check_quant_checkpoint_match(DEFAULT_REPO_ID, "autoround", False)

    def test_autoround_checkpoint_under_non_matching_quant_none_raises(
        self, monkeypatch
    ):
        """The mirror direction: an AutoRound (or any quantized) checkpoint
        loaded with a non-matching request — here quant='none' — still
        raises via direction 1, which is already backend-agnostic (any
        non-None declared_method + quant='none' is a mismatch). Named
        explicitly for #210's "both directions" acceptance criterion."""
        _patch_autoconfig(
            monkeypatch, quantization_config={"quant_method": "auto-round"}
        )

        with pytest.raises(RuntimeError) as excinfo:
            _check_quant_checkpoint_match(AUTOROUND_REPO_ID, "none", False)

        message = str(excinfo.value)
        assert "auto-round" in message
        assert "quant='none'" in message

    def test_full_load_model_raises_before_from_pretrained(self, monkeypatch):
        """End-to-end: load_model() itself raises via the guard on a
        backend mismatch, never reaching from_pretrained (issue #210's
        named failure mode: an unhandled crash deep inside the load)."""
        _patch_autoconfig(monkeypatch, quantization_config={"quant_method": "gptq"})

        def unreachable_from_pretrained(repo_id, **kwargs):
            pytest.fail("from_pretrained must not be called past a guard raise")

        monkeypatch.setattr(
            "dgemma.model.DiffusionGemmaForBlockDiffusion.from_pretrained",
            unreachable_from_pretrained,
        )

        with pytest.raises(RuntimeError, match="gptq"):
            load_model(repo_id="some/gptq-checkpoint", quant="autoround")


# ---------------------------------------------------------------------------
# Matched configs: pass-through, no raise
# ---------------------------------------------------------------------------

class TestMatchedConfigPassesThrough:
    """quant='none' + unquantized checkpoint, and quant='autoround' +
    quantized checkpoint, are both matches — the guard must not raise, and
    load_model() must proceed to (the faked) from_pretrained."""

    def test_none_quant_with_unquantized_checkpoint_does_not_raise(self, monkeypatch):
        _patch_autoconfig(monkeypatch, quantization_config=None)

        _check_quant_checkpoint_match(DEFAULT_REPO_ID, "none", False)  # no raise

    def test_autoround_quant_with_quantized_checkpoint_does_not_raise(
        self, monkeypatch
    ):
        _patch_autoconfig(
            monkeypatch, quantization_config={"quant_method": "auto-round"}
        )

        _check_quant_checkpoint_match(AUTOROUND_REPO_ID, "autoround", False)  # no raise

    def test_full_load_model_quant_none_matched_reaches_from_pretrained(
        self, monkeypatch
    ):
        _patch_autoconfig(monkeypatch, quantization_config=None)
        captured: dict = {}
        _install_load_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=DEFAULT_REPO_ID, quant="none")

        assert captured["repo_id"] == DEFAULT_REPO_ID
        assert result.quant == "none"

    def test_full_load_model_quant_autoround_matched_reaches_from_pretrained(
        self, monkeypatch
    ):
        _patch_autoconfig(
            monkeypatch, quantization_config={"quant_method": "auto-round"}
        )
        captured: dict = {}
        _install_load_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=AUTOROUND_REPO_ID, quant="autoround")

        assert captured["repo_id"] == AUTOROUND_REPO_ID
        assert result.quant == "autoround"


# ---------------------------------------------------------------------------
# Unreadable config: best-effort fall-through, never a new failure door
# ---------------------------------------------------------------------------

class TestUnreadableConfigFallsThrough:
    """If AutoConfig.from_pretrained itself fails (network trouble, repo
    not cached under local_files_only, malformed config, ...), the guard
    must NOT raise or block — it logs a warning and lets the normal load
    path proceed (requirement 1: best-effort, not a new failure door)."""

    def test_autoconfig_raising_falls_through_without_raising(self, monkeypatch):
        _patch_autoconfig(monkeypatch, raise_exc=OSError("repo not found"))

        _check_quant_checkpoint_match(
            "some/nonexistent-repo", "autoround", False
        )  # must not raise despite quant='autoround' + no confirmed config

    def test_autoconfig_raising_logs_warning(self, monkeypatch, capsys):
        _patch_autoconfig(monkeypatch, raise_exc=OSError("repo not found"))

        _check_quant_checkpoint_match("some/nonexistent-repo", "none", False)

        captured_out = capsys.readouterr().out
        assert "[WARN]" in captured_out
        assert "some/nonexistent-repo" in captured_out

    def test_full_load_model_falls_through_to_from_pretrained(self, monkeypatch):
        """With an unreadable config, load_model() proceeds past the guard
        and reaches the (faked) from_pretrained — no new hang, no new raise."""
        _patch_autoconfig(monkeypatch, raise_exc=OSError("network unreachable"))
        captured: dict = {}
        _install_load_fakes(monkeypatch, captured, hf_device_map={"model.layers.0": 0})

        result = load_model(repo_id=DEFAULT_REPO_ID, quant="none")

        assert captured["repo_id"] == DEFAULT_REPO_ID
        assert result.quant == "none"

    def test_local_files_only_is_forwarded_to_autoconfig(self, monkeypatch):
        """Requirement 1: the guard must respect local_files_only."""
        seen_kwargs: dict = {}

        def fake_autoconfig_from_pretrained(repo_id, **kwargs):
            seen_kwargs.update(kwargs)
            return FakeAutoConfig(quantization_config=None)

        monkeypatch.setattr(
            "transformers.AutoConfig.from_pretrained", fake_autoconfig_from_pretrained
        )

        _check_quant_checkpoint_match(DEFAULT_REPO_ID, "none", True)

        assert seen_kwargs.get("local_files_only") is True
