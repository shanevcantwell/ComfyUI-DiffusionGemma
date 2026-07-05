"""nodes/*.py adapt without logic (ADR-CDG-003): unpack kwargs -> call one
`dgemma.*` function -> wrap the result in a tuple, nothing more. Verified by
monkeypatching the exact `dgemma.*` call each node makes and asserting the
node method is pure pass-through/wrap.

ComfyUI-absent-safe import strategy: `nodes/loader.py` and `nodes/sampler.py`
import nothing from `comfy` at module level (this venv has no `comfy` package
at all, so any such import would already have raised at collection time —
the second test below is belt-and-suspenders on the same invariant
`tests/test_seam.py` checks from the `dgemma/` side).
"""
from __future__ import annotations

import sys

import pytest

from nodes.loader import DGemmaLoader
from nodes.sampler import DGemmaSampler


def test_nodes_modules_do_not_import_comfy():
    assert not any(m == "comfy" or m.startswith("comfy.") for m in sys.modules)


def test_loader_input_types_declares_repo_id_and_quant():
    spec = DGemmaLoader.INPUT_TYPES()
    assert "repo_id" in spec["required"]
    assert "quant" in spec["required"]
    assert spec["required"]["quant"][0] == ["nf4", "int8", "none"]


def test_loader_quant_default_is_none_not_nf4():
    """Issue #4: nf4 OOMs structurally on the 48GB dev box (bnb can't
    quantize the fused 3D MoE experts); "none" (bf16 CPU-spill) is the one
    with verified PASSes. The widget default must reflect that, not the
    stale nf4 default."""
    spec = DGemmaLoader.INPUT_TYPES()
    assert spec["required"]["quant"][1]["default"] == "none"


def test_loader_declarations():
    assert DGemmaLoader.RETURN_TYPES == ("DGEMMA_MODEL",)
    assert DGemmaLoader.FUNCTION == "load"
    assert DGemmaLoader.CATEGORY == "DiffusionGemma"


def test_loader_calls_load_model_and_wraps_tuple(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_load_model(repo_id, quant):
        captured["repo_id"] = repo_id
        captured["quant"] = quant
        return sentinel

    monkeypatch.setattr("nodes.loader.load_model", fake_load_model)

    node = DGemmaLoader()
    result = node.load(repo_id="google/diffusiongemma-26B-A4B-it", quant="nf4")

    assert result == (sentinel,)
    assert captured == {"repo_id": "google/diffusiongemma-26B-A4B-it", "quant": "nf4"}


def test_sampler_declarations():
    assert DGemmaSampler.RETURN_TYPES == ("STRING", "DGEMMA_CANVAS_STATE")
    assert DGemmaSampler.FUNCTION == "sample"
    assert DGemmaSampler.CATEGORY == "DiffusionGemma"


def test_sampler_input_types_declares_all_p2_widgets():
    spec = DGemmaSampler.INPUT_TYPES()
    assert set(spec["required"]) == {
        "model",
        "prompt",
        "seed",
        "num_inference_steps",
        "t_min",
        "t_max",
        "entropy_bound",
        "confidence",
        "gen_length",
        "thinking",
    }
    assert spec["required"]["model"] == ("DGEMMA_MODEL",)
    assert spec["required"]["thinking"] == ("BOOLEAN", {"default": False})
    # Grounded defaults (plan.md Phase 2 / CLAUDE.md).
    assert spec["required"]["num_inference_steps"][1]["default"] == 48
    assert spec["required"]["t_min"][1]["default"] == pytest.approx(0.4)
    assert spec["required"]["t_max"][1]["default"] == pytest.approx(0.8)
    assert spec["required"]["entropy_bound"][1]["default"] == pytest.approx(0.1)
    assert spec["required"]["confidence"][1]["default"] == pytest.approx(0.005)
    assert spec["required"]["gen_length"][1]["default"] == 256


def test_sampler_calls_run_diffusion_and_wraps_tuple(monkeypatch):
    sentinel_model = object()
    captured = {}

    def fake_run_diffusion(model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return ("decoded text", "canvas-state-stub")

    monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

    node = DGemmaSampler()
    result = node.sample(
        model=sentinel_model,
        prompt="hello",
        seed=7,
        num_inference_steps=48,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=0.1,
        confidence=0.005,
        gen_length=256,
        thinking=False,
    )

    assert result == ("decoded text", "canvas-state-stub")
    assert captured["model"] is sentinel_model
    assert captured["prompt"] == "hello"
    assert captured["kwargs"]["seed"] == 7
    # P2: assert the node actually forwards every widget value rather than
    # silently dropping one to some hardcoded default.
    assert captured["kwargs"]["entropy_bound"] == pytest.approx(0.1)
    assert captured["kwargs"]["t_min"] == pytest.approx(0.4)
    assert captured["kwargs"]["t_max"] == pytest.approx(0.8)
    assert captured["kwargs"]["num_inference_steps"] == 48
    assert captured["kwargs"]["gen_length"] == 256
    assert captured["kwargs"]["confidence"] == pytest.approx(0.005)
    assert captured["kwargs"]["thinking"] is False


def test_sampler_forwards_non_default_thinking_and_confidence(monkeypatch):
    """Distinct-from-default values must actually thread through — a node
    that silently forwarded its own hardcoded constants instead of the
    passed-in widget values would pass the test above (defaults match) but
    fail this one."""
    captured = {}

    def fake_run_diffusion(model, prompt, **kwargs):
        captured["kwargs"] = kwargs
        return ("text", "state")

    monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

    node = DGemmaSampler()
    node.sample(
        model=object(),
        prompt="hi",
        seed=1,
        num_inference_steps=12,
        t_min=0.1,
        t_max=0.5,
        entropy_bound=0.25,
        confidence=0.9,
        gen_length=64,
        thinking=True,
    )

    assert captured["kwargs"]["num_inference_steps"] == 12
    assert captured["kwargs"]["t_min"] == pytest.approx(0.1)
    assert captured["kwargs"]["t_max"] == pytest.approx(0.5)
    assert captured["kwargs"]["entropy_bound"] == pytest.approx(0.25)
    assert captured["kwargs"]["confidence"] == pytest.approx(0.9)
    assert captured["kwargs"]["gen_length"] == 64
    assert captured["kwargs"]["thinking"] is True
