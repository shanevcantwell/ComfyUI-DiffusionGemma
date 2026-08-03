"""issue #119 — tied-weight corruption under split `device_map="auto"`:
`_assert_tie_integrity` + its helpers (`_tensor_ties_match`,
`_resolve_meta_weight`), against REAL transformers 5.13.0 tie machinery (not
mocked): a shrunk `DiffusionGemmaForBlockDiffusion`-pattern checkpoint whose
encoder text stack owns no weights of its own (mirroring the real 26B
checkpoint's own `model.safetensors.index.json`, per the 2026-07-22 forensic
verdict banked on issue #119), loaded through the real `from_pretrained` +
`tie_weights` path.

Grounding (salvaged from the closed PR #121's grounding pass —
`fix/119-tied-weights-device-map-guard` @ `07eb764`, read and adapted, not
copied verbatim): a real `DiffusionGemmaConfig` (shrunk `text_config` +
`vision_config`, `Gemma4VisionConfig` is REQUIRED —
`DiffusionGemmaEncoderModel.__init__` unconditionally constructs
`AutoModel.from_config(config.vision_config)`, so `vision_config=None` raises
`ValueError` at model construction, not a graceful skip) is used to build a
real (if tiny) model, whose state_dict is then filtered to drop every
encoder-text-stack weight the real checkpoint also omits (`layer_scalar`
buffers are the only `encoder.language_model` keys the real index.json
retains) before being saved as a `safetensors` checkpoint and reloaded via
`from_pretrained` — the exact seam `dgemma.model.load_model` drives.

This module intentionally does NOT attempt to reproduce the corruption via a
genuine multi-device dispatch split: that needs multiple visible accelerators
or a CUDA host, neither guaranteed in CI. The corruption is instead exercised
the way that is both hardware-independent and non-mocked: constructing it
directly on a genuinely tied, genuinely loaded model
(`TestTieIntegrityGuardCatchesCorruption`) — the guard is checked against a
REAL collapsed weight with the REAL expected-shape arithmetic, not a
synthetic mock of the guard's own comparison. The cpu-offload false-positive
that killed the closed PR #121's first live-verify attempt (guard tripped on
a healthy tied pair under real `AlignDevicesHook(offload=True)` dispatch,
PR #121 comment 5044645329) is reproduced via REAL accelerate offload
machinery (`accelerate.hooks.attach_align_device_hook`) in
`TestTensorTiesMatchUnderRealOffload` — CPU-only, no CUDA required (the
hook's `execution_device` is never actually dispatched to in these tests).

Placement-policy scope note: this module does NOT test a
`_resolve_placement`/pairwise-co-located `device_map` override — that half of
PR #121 is deliberately not ported (see `dgemma/model.py`'s module docstring
and this fix's PR body): issues #173/#183 hardened `device_map="auto"` as
this pack's field-verified, release-blocking-protected default placement for
`quant="none"`, and an untested placement override belongs behind its own
live-GPU verification cycle, not folded into this guard-only fix.
"""
from __future__ import annotations

import os

import pytest
import torch

from dgemma.model import (
    _assert_tie_integrity,
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
    """A healthy load: single-device placement — every tied pair trivially
    co-located, guard must PASS."""
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
    """Reproducer-derived: the guard must catch the REAL observed defect
    shape — an encoder q_proj weight collapsed to a single output feature
    while the decoder's stays correct, the exact keystone-arithmetic failure
    the forensic verdict traces to
    `hidden_shape = (*hidden_states.shape[:-1], -1, head_dim)`
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
        assert "tied" in message.lower()
        assert "split" in message.lower() or "placement" in message.lower()
        assert str(tuple(collapsed.shape)) in message  # observed (bad) shape named
        assert "hf_device_map" in message
        assert "119" in message  # issue reference

    def test_guard_raises_on_last_layer_collapse_too(self, toy_model_single_device):
        """A real split can corrupt an arbitrary subset of layers depending
        on the dispatch-hook boundary — the guard's layer-0-and-last sample
        must catch a last-layer-only corruption, not just a layer-0 one."""
        model = toy_model_single_device
        last_idx = model.config.text_config.num_hidden_layers - 1
        enc_attn = model.model.encoder.language_model.layers[last_idx].self_attn

        with torch.no_grad():
            collapsed = enc_attn.q_proj.weight.data[:1, :].clone()
        enc_attn.q_proj.weight = torch.nn.Parameter(collapsed)

        with pytest.raises(RuntimeError, match=f"layer {last_idx}"):
            _assert_tie_integrity(model)

    def test_guard_raises_when_shape_is_right_but_values_diverge(self, toy_model_single_device):
        """A same-shape-but-untied encoder weight (a hypothetical partial fix
        that gives the encoder its own storage without actually copying the
        decoder's values) must also fail — the guard checks tie correctness,
        not just shape."""
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
        byte-exact downstream `numel==seq_len` view crash the guard exists to
        convert into an honest load-time failure — grounding that the
        guard's trigger condition is the real defect, not a stand-in."""
        model = toy_model_single_device
        enc_attn = model.model.encoder.language_model.layers[0].self_attn

        with torch.no_grad():
            collapsed = enc_attn.q_proj.weight.data[:1, :].clone()
        enc_attn.q_proj.weight = torch.nn.Parameter(collapsed)

        ids = torch.randint(0, 200, (1, 11))
        with pytest.raises(RuntimeError, match="invalid for input of size"), torch.no_grad():
            model.model.encoder(input_ids=ids, attention_mask=torch.ones_like(ids))


class _ModuleStub:
    """Bare stand-in for `nn.Linear`-shaped `q_proj` modules: just a
    `.weight` attribute, no `_hf_hook` (mirrors an un-dispatched/
    non-offloaded module). `_tensor_ties_match`'s contract takes MODULES, not
    bare tensors, so unit coverage on the non-offload paths needs something
    with a `.weight` to pass in place of a real `nn.Linear`."""

    def __init__(self, weight):
        self.weight = weight


class TestTensorTiesMatch:
    """Unit coverage for the helper `_assert_tie_integrity` uses to decide
    tie correctness — same-tensor fast path, value-equal fallback, and the
    meta-tensor case must degrade to "not tied," never propagate an
    unhandled exception out of the guard.

    These cases exercise the non-offload paths (no `_hf_hook` on the module —
    `_resolve_meta_weight` returns the weight unchanged when it isn't meta).
    The offload-aware `weights_map` resolution path — the false positive that
    killed the closed PR #121's first live-verify attempt — is covered
    separately by `TestTensorTiesMatchUnderRealOffload` below, against real
    accelerate dispatch machinery."""

    def test_same_tensor_is_tied(self):
        w = torch.randn(4, 4)
        assert _tensor_ties_match(_ModuleStub(w), _ModuleStub(w)) is True

    def test_equal_values_different_storage_is_tied(self):
        a = torch.ones(4, 4)
        b = torch.ones(4, 4)
        assert a is not b
        assert _tensor_ties_match(_ModuleStub(a), _ModuleStub(b)) is True

    def test_different_shape_is_not_tied(self):
        a = torch.ones(4, 4)
        b = torch.ones(1, 4)
        assert _tensor_ties_match(_ModuleStub(a), _ModuleStub(b)) is False

    def test_different_values_is_not_tied(self):
        a = torch.zeros(4, 4)
        b = torch.ones(4, 4)
        assert _tensor_ties_match(_ModuleStub(a), _ModuleStub(b)) is False

    def test_meta_tensor_with_no_hook_is_not_tied(self):
        """A meta weight with no `_hf_hook` at all (never dispatched through
        accelerate) has no resolvable `weights_map` — `_resolve_meta_weight`
        must return `None` and the guard must report "not tied" rather than
        raising `AttributeError` or silently passing an unverifiable tie."""
        a = torch.empty(4, 4, device="meta")
        b = torch.empty(4, 4, device="meta")
        assert _tensor_ties_match(_ModuleStub(a), _ModuleStub(b)) is False


@pytest.fixture()
def toy_model_cpu_offloaded(toy_checkpoint_dir):
    """Dispatches the shrunk tied model through REAL accelerate offload
    machinery (`attach_align_device_hook`, the primitive `accelerate.cpu_offload`
    and `dispatch_model` themselves call) so every module ends up meta-at-rest
    with a genuine `AlignDevicesHook(offload=True)` + `weights_map` attached —
    mirroring what `device_map="auto"` + CPU-spill produces in production, and
    the exact state the 2026-07-22 discriminating probe found tripping the
    closed PR #121's pre-amendment guard into a false positive (issue #119,
    PR #121 comment 5044645329) on this pack's actual dev box.

    `execution_device` is set but never dispatched to in these CPU-only
    tests — the hook's presence (and its `weights_map`) is what
    `_resolve_meta_weight` needs, not an actual forward pass through CUDA.

    Loaded single-device first (mirrors this repo's own `toy_model_single_device`
    fixture) so the tie is genuine Python-object identity before offload
    reshapes storage into the hook's `weights_map`."""
    from accelerate.hooks import attach_align_device_hook
    from transformers import DiffusionGemmaForBlockDiffusion

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(
        toy_checkpoint_dir, dtype=torch.bfloat16, device_map={"": "cpu"}
    )
    state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}
    attach_align_device_hook(
        model,
        execution_device=torch.device("cpu"),
        offload=True,
        weights_map=state_dict,
    )
    return model, state_dict


class TestTensorTiesMatchUnderRealOffload:
    """The false-positive the closed PR #121 hit on real live-verify
    (2026-07-22, this pack's dev box, single-GPU `device_map="auto"` +
    CPU-spill): the pre-amendment guard swallowed meta tensors'
    `torch.equal` `NotImplementedError` into an unconditional `False`, so ANY
    cpu-offloaded sampled layer reported a broken tie even when — per the
    discriminating probe's three independent checks (shared `weights_map`
    storage, hook-materialized `data_ptr` equality, coherent generation with
    the guard bypassed) — the tie was fully intact. These tests drive the
    REAL accelerate dispatch machinery (`attach_align_device_hook`), not a
    mock of the guard's own comparison — this is the CPU-reproducer for the
    exact regression that killed #121's first live-verify pass."""

    def test_healthy_offloaded_tie_passes(self, toy_model_cpu_offloaded):
        model, _ = toy_model_cpu_offloaded
        enc_q = model.model.encoder.language_model.layers[1].self_attn.q_proj
        dec_q = model.model.decoder.layers[1].self_attn.q_proj

        assert enc_q.weight.is_meta and dec_q.weight.is_meta  # sanity: genuinely offloaded
        assert _tensor_ties_match(enc_q, dec_q) is True

        # And the full guard, exercised end-to-end, must not raise either —
        # this is the exact scenario (healthy tie, cpu-offloaded, sampled at
        # the last layer) that raised a false-positive RuntimeError on #121's
        # live-verify run.
        _assert_tie_integrity(model)

    def test_healthy_offloaded_tie_passes_at_last_layer(self, toy_model_cpu_offloaded):
        """The guard samples layer 0 and the last layer — the false positive
        was found specifically at the last layer in the live-verify run
        (layer 29 of the real 26-layer checkpoint), so this repeats the
        healthy-tie check at this toy config's last layer explicitly."""
        model, _ = toy_model_cpu_offloaded
        last_idx = model.config.text_config.num_hidden_layers - 1
        enc_q = model.model.encoder.language_model.layers[last_idx].self_attn.q_proj
        dec_q = model.model.decoder.layers[last_idx].self_attn.q_proj

        assert enc_q.weight.is_meta and dec_q.weight.is_meta
        assert _tensor_ties_match(enc_q, dec_q) is True

    def test_perturbed_weights_map_entry_is_detected(self, toy_checkpoint_dir):
        """Perturb ONE side's weights_map entry to genuinely different
        storage/values (not just a different Python object) before attaching
        the offload hooks — the resolved-tensor comparison must still catch
        the mismatch and the full guard must raise with the actionable
        message, not silently pass because both sides are meta."""
        from accelerate.hooks import attach_align_device_hook
        from transformers import DiffusionGemmaForBlockDiffusion

        model = DiffusionGemmaForBlockDiffusion.from_pretrained(
            toy_checkpoint_dir, dtype=torch.bfloat16, device_map={"": "cpu"}
        )
        state_dict = {n: p.to("cpu") for n, p in model.state_dict().items()}
        perturbed_key = "model.encoder.language_model.layers.0.self_attn.q_proj.weight"
        assert perturbed_key in state_dict  # sanity: key actually present pre-perturbation
        state_dict[perturbed_key] = torch.zeros_like(state_dict[perturbed_key])

        attach_align_device_hook(
            model,
            execution_device=torch.device("cpu"),
            offload=True,
            weights_map=state_dict,
        )

        enc_q = model.model.encoder.language_model.layers[0].self_attn.q_proj
        dec_q = model.model.decoder.layers[0].self_attn.q_proj
        assert enc_q.weight.is_meta and dec_q.weight.is_meta  # still genuinely offloaded
        assert _tensor_ties_match(enc_q, dec_q) is False

        with pytest.raises(RuntimeError) as excinfo:
            _assert_tie_integrity(model)
        message = str(excinfo.value)
        assert "tied" in message.lower()
        assert "119" in message

    def test_meta_weight_with_hook_stripped_is_not_tied(self, toy_model_cpu_offloaded):
        """A module whose weight is meta but whose `_hf_hook` has been
        removed (e.g. a partial/broken dispatch state) has no resolvable
        `weights_map` — `_resolve_meta_weight` must return `None` and the
        guard must report "not tied," not silently pass an unverifiable
        tie."""
        model, _ = toy_model_cpu_offloaded
        enc_q = model.model.encoder.language_model.layers[0].self_attn.q_proj
        dec_q = model.model.decoder.layers[0].self_attn.q_proj
        assert enc_q.weight.is_meta  # still meta after the hook is stripped below

        del enc_q._hf_hook

        assert _tensor_ties_match(enc_q, dec_q) is False

        with pytest.raises(RuntimeError) as excinfo:
            _assert_tie_integrity(model)
        assert "119" in str(excinfo.value)

    def test_meta_weight_with_missing_weights_map_entry_is_not_tied(self, toy_model_cpu_offloaded):
        """A module whose `_hf_hook.weights_map` is present but genuinely
        missing the `"weight"` key (a corrupted/incomplete offload store,
        distinct from `_hf_hook` being absent entirely) must degrade to "not
        tied" via `_resolve_meta_weight`'s `KeyError` branch, not raise
        `KeyError` out of the guard."""
        model, _ = toy_model_cpu_offloaded
        enc_q = model.model.encoder.language_model.layers[0].self_attn.q_proj
        dec_q = model.model.decoder.layers[0].self_attn.q_proj
        assert enc_q.weight.is_meta

        weights_map = enc_q._hf_hook.weights_map
        key = weights_map.prefix + "weight"
        assert key in weights_map.dataset  # sanity: entry present before deletion
        del weights_map.dataset[key]

        assert _tensor_ties_match(enc_q, dec_q) is False

        with pytest.raises(RuntimeError) as excinfo:
            _assert_tie_integrity(model)
        assert "119" in str(excinfo.value)
