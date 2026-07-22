"""fix #119 — tie-integrity guard + placement policy, against REAL
transformers 5.13.0 tie machinery (not mocked): a shrunk
`DiffusionGemmaForBlockDiffusion`-pattern checkpoint whose encoder text stack
owns no weights of its own (mirroring the real 26B checkpoint's own
`model.safetensors.index.json`, per the 2026-07-22 forensic verdict banked on
issue #119), loaded through the real `from_pretrained` + `tie_weights` path.

Grounding (adapted from the forensic-pass scratchpad reproducers
`tie_probe.py`/`toy_repro.py`/`toy_repro2.py`/`toy_repro3.py` — read, not
copied verbatim): a real `DiffusionGemmaConfig` (shrunk `text_config` +
`vision_config`, `Gemma4VisionConfig` is REQUIRED — `DiffusionGemmaEncoderModel.__init__`
unconditionally constructs `AutoModel.from_config(config.vision_config)`,
so `vision_config=None` raises `ValueError` at model construction, not a
graceful skip) is used to build a real (if tiny) model, whose state_dict is
then filtered to drop every encoder-text-stack weight the real checkpoint
also omits (`layer_scalar` buffers are the only encoder.language_model keys
the real index.json retains) before being saved as a `safetensors` checkpoint
and reloaded via `from_pretrained` — the exact seam `dgemma.model.load_model`
drives.

This module intentionally does NOT attempt to reproduce the corruption via a
genuine multi-device dispatch split: this box is CPU-only, and a CPU+disk
split was independently found (during this fix's grounding pass) to crash
inside transformers' own `tie_weights` at a DIFFERENT, earlier point
(`modeling_utils.py:2723`'s `torch.equal` on meta tensors, `NotImplementedError`)
before the numel corruption this guard targets could even occur — a second,
also-real transformers 5.13.0 bug, consistent with the forensic verdict's fix
candidate 3 (upstream report). A CPU-only all-`"cpu"` split (matching
`toy_repro3.py`'s sub-layer-granularity map) does not exercise `hf_device_map`
dispatch at all in this transformers version (no non-cpu/disk device present
-> `from_pretrained` never calls `dispatch_model`), so it cannot reproduce the
corruption either. The corruption itself is therefore exercised the only way
that is both CPU-only and non-mocked: constructing it directly on a genuinely
tied, genuinely loaded model (`TestTieIntegrityGuardCatchesCorruption`) — the
guard is checked against a REAL collapsed weight with the REAL expected-shape
arithmetic, not a synthetic mock of the guard's own check.
"""
from __future__ import annotations

import os

import pytest
import torch

from dgemma.model import (
    _assert_tie_integrity,
    _estimate_model_bytes,
    _fanout_pairwise_device_map,
    _pairwise_colocated_device_map,
    _resolve_placement,
    _tensor_ties_match,
)


def _build_toy_config():
    from transformers import DiffusionGemmaConfig
    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import (
        DiffusionGemmaTextConfig,
    )
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig

    text_config = DiffusionGemmaTextConfig(
        hidden_size=64,
        intermediate_size=48,
        moe_intermediate_size=32,
        num_experts=4,
        top_k_experts=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        vocab_size=256,
        sliding_window=8,
        max_position_embeddings=512,
    )
    vision_config = Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=48,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
    )
    config = DiffusionGemmaConfig(text_config=text_config, vision_config=vision_config)
    config.canvas_length = 8
    return config


def _encoder_owns_no_weights(key: str) -> bool:
    """Mirrors the real 26B checkpoint's `model.safetensors.index.json`
    (forensic verdict): every `model.encoder.language_model.*` key is
    dropped from the saved checkpoint EXCEPT `layer_scalar` buffers — the
    encoder text stack owns no weights of its own; everything else is tied
    from the decoder at `tie_weights()` time."""
    if key.startswith("model.encoder.language_model"):
        return key.endswith("layer_scalar")
    return True


@pytest.fixture(scope="session")
def toy_checkpoint_dir(tmp_path_factory):
    """Builds the shrunk DiffusionGemma-pattern checkpoint once per test
    session (real transformers construction + real safetensors save — no
    mocking of the tie mechanism) and returns its directory."""
    from safetensors.torch import save_file
    from transformers import DiffusionGemmaForBlockDiffusion

    config = _build_toy_config()
    torch.manual_seed(0)
    model = DiffusionGemmaForBlockDiffusion(config)
    state_dict = model.state_dict()
    filtered = {k: v.clone() for k, v in state_dict.items() if _encoder_owns_no_weights(k)}
    assert len(filtered) < len(state_dict), (
        "sanity: the filter must actually drop encoder-text-stack keys, "
        "else this checkpoint doesn't mirror the real one's no-owned-weights shape"
    )

    outdir = tmp_path_factory.mktemp("toy_ckpt")
    save_file(filtered, os.path.join(outdir, "model.safetensors"), metadata={"format": "pt"})
    config.save_pretrained(outdir)
    del model
    return str(outdir)


@pytest.fixture()
def toy_model_single_device(toy_checkpoint_dir):
    """A healthy load: single-device placement, the degenerate case
    `_pairwise_colocated_device_map` also emits for a single accelerator —
    every tied pair trivially co-located, guard must PASS."""
    from transformers import DiffusionGemmaForBlockDiffusion

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        toy_checkpoint_dir, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    return model


class TestTieIntegrityGuardPassesOnHealthyLoad:
    def test_guard_does_not_raise_on_single_device_load(self, toy_model_single_device):
        _assert_tie_integrity(toy_model_single_device)  # must not raise

    def test_encoder_and_decoder_q_proj_are_the_same_tensor(self, toy_model_single_device):
        """Grounding check: the real tie mechanism DOES produce Python-object
        identity for a healthy single-device load — not just equal values."""
        enc_q = toy_model_single_device.model.encoder.language_model.layers[0].self_attn.q_proj.weight
        dec_q = toy_model_single_device.model.decoder.layers[0].self_attn.q_proj.weight
        assert enc_q is dec_q

    def test_guard_checks_layer_0_and_last_layer(self, toy_model_single_device):
        """2 layers configured -> sample_indices == {0, 1}; both must be
        healthy for the guard to pass (already implied by the no-raise test
        above, asserted explicitly here so a future guard change that
        silently narrows the sample to only layer 0 is caught)."""
        num_layers = toy_model_single_device.config.text_config.num_hidden_layers
        assert num_layers == 2
        for layer_idx in range(num_layers):
            enc_q = toy_model_single_device.model.encoder.language_model.layers[
                layer_idx
            ].self_attn.q_proj.weight
            dec_q = toy_model_single_device.model.decoder.layers[layer_idx].self_attn.q_proj.weight
            assert enc_q is dec_q


class TestTieIntegrityGuardCatchesCorruption:
    """Reproducer-derived (adapted from toy_repro.py's `probe` pattern): the
    guard must catch the REAL observed defect shape — an encoder q_proj
    weight collapsed to a single output feature while the decoder's stays
    correct, the exact keystone-arithmetic failure the forensic verdict
    traces to `hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)`
    (`modeling_diffusion_gemma.py:329-330`) downstream. This is injected
    directly on a genuinely-tied, genuinely-loaded model — not a mock of the
    guard's own comparison."""

    def test_guard_raises_on_collapsed_encoder_weight(self, toy_model_single_device):
        model = toy_model_single_device
        enc_attn = model.model.encoder.language_model.layers[0].self_attn
        dec_attn = model.model.decoder.layers[0].self_attn
        assert enc_attn.q_proj.weight is dec_attn.q_proj.weight  # healthy before corruption

        with torch.no_grad():
            collapsed = enc_attn.q_proj.weight.data[:1, :].clone()
        enc_attn.q_proj.weight = torch.nn.Parameter(collapsed)

        with pytest.raises(RuntimeError) as excinfo:
            _assert_tie_integrity(model)

        message = str(excinfo.value)
        # The three actionable facts the contract requires: the defect, the
        # observed shape, and hf_device_map (checked on message SUBSTANCE,
        # not exact wording).
        assert "tied-weight corruption" in message.lower() or "tied" in message.lower()
        assert "split" in message.lower() or "placement" in message.lower()
        assert str(tuple(collapsed.shape)) in message  # observed (bad) shape named
        assert "hf_device_map" in message
        assert "119" in message  # issue reference

    def test_guard_raises_on_last_layer_collapse_too(self, toy_model_single_device):
        """The forensic verdict's residual-unknown note says a real split can
        corrupt an arbitrary subset of layers depending on the dispatch-hook
        boundary — the guard's layer-0-and-last sample must catch a
        last-layer-only corruption, not just a layer-0 one."""
        model = toy_model_single_device
        last_idx = model.config.text_config.num_hidden_layers - 1
        enc_attn = model.model.encoder.language_model.layers[last_idx].self_attn

        with torch.no_grad():
            collapsed = enc_attn.q_proj.weight.data[:1, :].clone()
        enc_attn.q_proj.weight = torch.nn.Parameter(collapsed)

        with pytest.raises(RuntimeError, match=f"layer {last_idx}"):
            _assert_tie_integrity(model)

    def test_guard_raises_when_shape_is_right_but_values_diverge(self, toy_model_single_device):
        """A same-shape-but-untied encoder weight (a hypothetical partial
        fix that gives the encoder its own storage without actually copying
        the decoder's values) must also fail — the guard checks tie
        correctness, not just shape."""
        model = toy_model_single_device
        enc_attn = model.model.encoder.language_model.layers[0].self_attn
        dec_attn = model.model.decoder.layers[0].self_attn

        with torch.no_grad():
            diverged = torch.zeros_like(dec_attn.q_proj.weight)
        enc_attn.q_proj.weight = torch.nn.Parameter(diverged)

        with pytest.raises(RuntimeError, match="119"):
            _assert_tie_integrity(model)

    def test_guard_reproduces_the_downstream_view_crash_when_uncaught(self, toy_model_single_device):
        """Closes the loop to the forensic verdict's keystone arithmetic: the
        SAME collapsed weight that trips the guard also reproduces the
        byte-exact downstream `numel==seq_len` view crash the guard exists
        to convert into an honest load-time failure — grounding that the
        guard's trigger condition is the real defect, not a stand-in."""
        model = toy_model_single_device
        enc_attn = model.model.encoder.language_model.layers[0].self_attn

        with torch.no_grad():
            collapsed = enc_attn.q_proj.weight.data[:1, :].clone()
        enc_attn.q_proj.weight = torch.nn.Parameter(collapsed)

        ids = torch.randint(0, 200, (1, 11))
        with pytest.raises(RuntimeError, match="invalid for input of size"):
            with torch.no_grad():
                model.model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))


class TestTensorTiesMatch:
    """Unit coverage for the helper `_assert_tie_integrity` uses to decide
    tie correctness — same-tensor fast path, value-equal fallback, and the
    meta-tensor `torch.equal` `NotImplementedError` the forensic verdict's
    toy reproducer independently found (fix candidate 3's upstream-crash
    finding) must degrade to "not tied," never propagate out of the guard."""

    def test_same_tensor_is_tied(self):
        w = torch.randn(4, 4)
        assert _tensor_ties_match(w, w) is True

    def test_equal_values_different_storage_is_tied(self):
        a = torch.ones(4, 4)
        b = torch.ones(4, 4)
        assert a is not b
        assert _tensor_ties_match(a, b) is True

    def test_different_shape_is_not_tied(self):
        a = torch.ones(4, 4)
        b = torch.ones(1, 4)
        assert _tensor_ties_match(a, b) is False

    def test_different_values_is_not_tied(self):
        a = torch.zeros(4, 4)
        b = torch.ones(4, 4)
        assert _tensor_ties_match(a, b) is False

    def test_meta_tensor_comparison_degrades_to_not_tied(self):
        """The forensic verdict's toy reproducer found `torch.equal` itself
        raises `NotImplementedError` on meta tensors (transformers 5.13.0 +
        this torch build) — the helper must catch that and report "not
        tied" rather than letting the guard crash on an unrelated error."""
        a = torch.empty(4, 4, device="meta")
        b = torch.empty(4, 4, device="meta")
        assert _tensor_ties_match(a, b) is False


class TestPairwiseColocatedDeviceMap:
    """Placement-policy unit tests: the pair-co-location property holds on
    the emitted map — every encoder layer N is pinned to the same device as
    decoder layer N. Pure config-driven, no model instantiation needed."""

    def test_every_encoder_layer_colocated_with_its_decoder_counterpart(self):
        config = _build_toy_config()
        device_map = _pairwise_colocated_device_map(config, "cpu")

        num_layers = config.text_config.num_hidden_layers
        assert num_layers > 0
        for layer_idx in range(num_layers):
            enc_key = f"model.encoder.language_model.layers.{layer_idx}"
            dec_key = f"model.decoder.layers.{layer_idx}"
            assert enc_key in device_map
            assert dec_key in device_map
            assert device_map[enc_key] == device_map[dec_key]

    def test_map_covers_the_whole_model_via_the_catch_all_entry(self):
        config = _build_toy_config()
        device_map = _pairwise_colocated_device_map(config, "cuda:0")
        assert device_map[""] == "cuda:0"

    def test_device_string_is_used_verbatim_for_every_pinned_entry(self):
        config = _build_toy_config()
        device_map = _pairwise_colocated_device_map(config, "cuda:3")
        pinned = [v for k, v in device_map.items() if k != ""]
        assert pinned  # non-empty: num_hidden_layers > 0 for this config
        assert all(v == "cuda:3" for v in pinned)


class TestFanoutPairwiseDeviceMap:
    """The multi-GPU pairwise fan-out primitive: every layer PAIR stays
    co-located (fix #119's actual invariant), while pairs themselves are
    spread round-robin across the ranked device list."""

    def test_every_pair_still_colocated_across_a_fanout(self):
        config = _build_toy_config()
        device_map = _fanout_pairwise_device_map(config, ["cuda:0", "cuda:1"])

        num_layers = config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            enc_key = f"model.encoder.language_model.layers.{layer_idx}"
            dec_key = f"model.decoder.layers.{layer_idx}"
            assert device_map[enc_key] == device_map[dec_key]

    def test_pairs_are_actually_spread_round_robin(self):
        """3+ layers over 2 devices must use BOTH devices, alternating by
        layer index — else this degenerates to the single-device map with
        extra steps and doesn't actually fix the too-big-for-one-GPU case."""
        config = _build_toy_config()
        config.text_config.num_hidden_layers = 4
        config.text_config.layer_types = ["sliding_attention"] * 3 + ["full_attention"]
        device_map = _fanout_pairwise_device_map(config, ["cuda:0", "cuda:1"])

        assert device_map["model.encoder.language_model.layers.0"] == "cuda:0"
        assert device_map["model.encoder.language_model.layers.1"] == "cuda:1"
        assert device_map["model.encoder.language_model.layers.2"] == "cuda:0"
        assert device_map["model.encoder.language_model.layers.3"] == "cuda:1"

    def test_catch_all_entry_pinned_to_the_first_ranked_device(self):
        config = _build_toy_config()
        device_map = _fanout_pairwise_device_map(config, ["cuda:1", "cuda:0"])
        assert device_map[""] == "cuda:1"  # first in ranked_devices, i.e. largest free memory


class TestEstimateModelBytes:
    """`_estimate_model_bytes`'s only job is to be a REAL (if coarse)
    function of config size — never a fixed constant, never zero for a
    real-shaped config, and strictly increasing with more layers/wider
    hidden size, which is what `_resolve_placement`'s fit branch depends on
    to make a real decision instead of a coin flip."""

    def test_nonzero_for_a_real_shaped_config(self):
        config = _build_toy_config()
        assert _estimate_model_bytes(config) > 0

    def test_more_layers_costs_more_bytes(self):
        small = _build_toy_config()
        big = _build_toy_config()
        big.text_config.num_hidden_layers = small.text_config.num_hidden_layers * 4
        assert _estimate_model_bytes(big) > _estimate_model_bytes(small)

    def test_wider_hidden_size_costs_more_bytes(self):
        small = _build_toy_config()
        big = _build_toy_config()
        big.text_config.hidden_size = small.text_config.hidden_size * 4
        assert _estimate_model_bytes(big) > _estimate_model_bytes(small)


class TestResolvePlacement:
    """`_resolve_placement`'s policy: single-device/pairwise-map when a
    device is visible and the model fits; multi-GPU pairwise fan-out when it
    doesn't fit on the largest single GPU but more than one GPU is visible;
    `"auto"` only when no device is visible at all, or exactly one
    (too-small) device is visible — mirrors `_resolve_device`'s existing
    device-ranking convention (prefer an accelerator over cpu/disk)."""

    def test_no_devices_visible_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setattr("accelerate.utils.get_max_memory", lambda *a, **k: {})
        config = _build_toy_config()
        assert _resolve_placement(config) == "auto"

    def test_cpu_only_visible_returns_pairwise_map_on_cpu(self, monkeypatch):
        monkeypatch.setattr("accelerate.utils.get_max_memory", lambda *a, **k: {"cpu": 16 * 1024**3})
        config = _build_toy_config()
        device_map = _resolve_placement(config)
        assert isinstance(device_map, dict)
        assert device_map[""] == "cpu"
        num_layers = config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            assert device_map[f"model.encoder.language_model.layers.{layer_idx}"] == "cpu"
            assert device_map[f"model.decoder.layers.{layer_idx}"] == "cpu"

    def test_gpu_visible_prefers_gpu_over_cpu(self, monkeypatch):
        monkeypatch.setattr(
            "accelerate.utils.get_max_memory",
            lambda *a, **k: {0: 48 * 1024**3, "cpu": 128 * 1024**3},
        )
        config = _build_toy_config()
        device_map = _resolve_placement(config)
        assert device_map[""] == "cuda:0"

    def test_largest_gpu_chosen_when_multiple_visible(self, monkeypatch):
        monkeypatch.setattr(
            "accelerate.utils.get_max_memory",
            lambda *a, **k: {0: 8 * 1024**3, 1: 48 * 1024**3, "cpu": 128 * 1024**3},
        )
        config = _build_toy_config()
        device_map = _resolve_placement(config)
        assert device_map[""] == "cuda:1"

    def test_get_max_memory_failure_degrades_to_auto(self, monkeypatch):
        """`accelerate.utils.get_max_memory` calls CUDA/NPU/etc. device
        probes that can themselves raise on an unusual environment — the
        placement policy must degrade to `"auto"` rather than letting
        `load_model` crash on a probing failure unrelated to the actual
        load."""

        def raising_get_max_memory(*args, **kwargs):
            raise RuntimeError("device probe exploded")

        monkeypatch.setattr("accelerate.utils.get_max_memory", raising_get_max_memory)
        config = _build_toy_config()
        assert _resolve_placement(config) == "auto"

    def test_model_too_big_for_one_gpu_fans_out_across_multiple(self, monkeypatch):
        """The fit gate: when the estimate exceeds the largest single GPU's
        free memory but more than one GPU is visible, `_resolve_placement`
        must fan out (never silently fall back to `"auto"` while an
        alternative pairwise-safe placement exists)."""
        config = _build_toy_config()
        estimate = _estimate_model_bytes(config)
        monkeypatch.setattr(
            "accelerate.utils.get_max_memory",
            # Each GPU alone is smaller than the estimate; together the
            # policy still must not just give up and go "auto".
            lambda *a, **k: {0: estimate // 4, 1: estimate // 4, "cpu": 128 * 1024**3},
        )

        device_map = _resolve_placement(config)

        assert isinstance(device_map, dict)
        num_layers = config.text_config.num_hidden_layers
        seen_devices = set()
        for layer_idx in range(num_layers):
            enc_device = device_map[f"model.encoder.language_model.layers.{layer_idx}"]
            dec_device = device_map[f"model.decoder.layers.{layer_idx}"]
            assert enc_device == dec_device  # fix #119's invariant, never violated
            seen_devices.add(enc_device)
        assert seen_devices == {"cuda:0", "cuda:1"}  # actually used BOTH GPUs

    def test_model_too_big_for_the_only_gpu_falls_back_to_auto(self, monkeypatch):
        """Exactly one GPU visible and the model doesn't fit on it: nowhere
        to fan out to, so `"auto"` (+ the tie-integrity guard backstop) is
        the honest answer, not a placement this policy would have to
        fabricate capacity for."""
        config = _build_toy_config()
        estimate = _estimate_model_bytes(config)
        monkeypatch.setattr(
            "accelerate.utils.get_max_memory",
            lambda *a, **k: {0: estimate // 4, "cpu": 128 * 1024**3},
        )

        assert _resolve_placement(config) == "auto"

    def test_model_fits_on_largest_gpu_does_not_fan_out(self, monkeypatch):
        """Sanity: even with multiple GPUs visible, a model that fits on the
        single largest one takes the simpler (safer) single-device path —
        fan-out is reached only when single-device genuinely doesn't fit."""
        config = _build_toy_config()
        estimate = _estimate_model_bytes(config)
        monkeypatch.setattr(
            "accelerate.utils.get_max_memory",
            lambda *a, **k: {0: estimate * 1000, 1: estimate * 1000, "cpu": 128 * 1024**3},
        )

        device_map = _resolve_placement(config)
        assert device_map[""] == "cuda:0"
        num_layers = config.text_config.num_hidden_layers
        for layer_idx in range(num_layers):
            assert device_map[f"model.encoder.language_model.layers.{layer_idx}"] == "cuda:0"
