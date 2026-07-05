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
    assert DGemmaSampler.RETURN_TYPES == ("STRING", "DGEMMA_CANVAS_STATE", "DGEMMA_CANVAS_TRACE")
    assert DGemmaSampler.RETURN_NAMES == ("text", "canvas_state", "canvas_trace")
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
    # P3: `unique_id` is a hidden input (standard ComfyUI idiom), not a widget —
    # routes the live per-step push (plan.md Phase 3 (a)) to the right node.
    assert spec["hidden"] == {"unique_id": "UNIQUE_ID"}
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
        return ("decoded text", "canvas-state-stub", "canvas-trace-stub")

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

    assert result == ("decoded text", "canvas-state-stub", "canvas-trace-stub")
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
    # P3: an `on_frame` callable is always forwarded (unconditionally built
    # by the node, regardless of whether a live ComfyUI process exists to
    # actually consume it — see the guarded-import tests below).
    assert callable(captured["kwargs"]["on_frame"])


def test_sampler_forwards_non_default_thinking_and_confidence(monkeypatch):
    """Distinct-from-default values must actually thread through — a node
    that silently forwarded its own hardcoded constants instead of the
    passed-in widget values would pass the test above (defaults match) but
    fail this one."""
    captured = {}

    def fake_run_diffusion(model, prompt, **kwargs):
        captured["kwargs"] = kwargs
        return ("text", "state", "trace")

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


class TestLiveFramePush:
    """P3 step 5's own regression coverage: the `PromptServer` import surface
    risk (plan.md Risks — first time `nodes/` imports live server
    infrastructure). `server`/`comfy` are not installed in this venv at all
    (`tests/test_seam.py`'s own grounding), so the "absent" branch below is
    the actual condition every other test in this suite already runs under;
    the "present" branch is exercised by injecting a fake `server` module
    into `sys.modules` — no real ComfyUI process needed for either."""

    def test_sample_succeeds_unchanged_when_promptserver_unavailable(self, monkeypatch):
        """The concrete regression test for the guarded-import risk
        (plan.md step 5's own Verifies): `PromptServer` absent (the normal
        pytest condition) must not raise, and `text`/`canvas_state` must be
        unaffected."""
        monkeypatch.delitem(sys.modules, "server", raising=False)

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())  # the collector always calls on_frame per step
            return ("text", "state", "trace")

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        result = node.sample(
            model=object(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            thinking=False,
            unique_id="42",
        )

        assert result == ("text", "state", "trace")

    def test_on_frame_pushes_via_send_sync_when_promptserver_available(self, monkeypatch):
        """The "present" branch, exercised without a real ComfyUI process:
        inject a fake `server` module carrying a fake `PromptServer.instance`
        and assert the closure calls `send_sync` with the pack's own event
        name and a payload keyed to the right node — never
        `comfy.utils.ProgressBar`'s image-typed `preview=` slot (plan.md
        Risks' named trap)."""
        captured_calls = []

        class FakeInstance:
            def send_sync(self, event, data, sid=None):
                captured_calls.append((event, data, sid))

        class FakePromptServer:
            instance = FakeInstance()

        fake_server_module = type(sys)("server")
        fake_server_module.PromptServer = FakePromptServer
        monkeypatch.setitem(sys.modules, "server", fake_server_module)

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())
            return ("text", "state", "trace")

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        node.sample(
            model=object(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            thinking=False,
            unique_id="42",
        )

        assert len(captured_calls) == 1
        event, data, sid = captured_calls[0]
        assert event == "dgemma.sampler.step"
        assert data["node"] == "42"
        assert data["canvas_idx"] == 0
        assert data["step_idx"] == 3
        assert data["committed_fraction"] == pytest.approx(1.0)

    def test_on_frame_is_a_no_op_when_promptserver_instance_is_none(self, monkeypatch):
        """`PromptServer.instance` is `None` before the server has fully
        started — the closure must degrade to a no-op here too, not just on
        `ImportError`."""

        class FakePromptServer:
            instance = None

        fake_server_module = type(sys)("server")
        fake_server_module.PromptServer = FakePromptServer
        monkeypatch.setitem(sys.modules, "server", fake_server_module)

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())  # must not raise
            return ("text", "state", "trace")

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        result = node.sample(
            model=object(),
            prompt="hi",
            seed=1,
            num_inference_steps=1,
            t_min=0.1,
            t_max=0.5,
            entropy_bound=0.1,
            confidence=0.1,
            gen_length=8,
            thinking=False,
            unique_id="42",
        )

        assert result == ("text", "state", "trace")

    def test_send_sync_failure_does_not_kill_the_run(self, monkeypatch, caplog):
        """Display must never kill generation (review finding, 2026-07-05):
        a `send_sync` failure (serialization error, dropped websocket) is
        logged and swallowed by the closure — the run completes and the
        outputs are intact. The guard lives in this node-layer closure by
        deliberate choice; the engine's `on_frame` contract propagates
        callback exceptions (`dgemma/loop.py`, `_FrameCollector`'s
        docstring)."""

        class ExplodingInstance:
            def send_sync(self, event, data, sid=None):
                raise RuntimeError("websocket dropped mid-run")

        class FakePromptServer:
            instance = ExplodingInstance()

        fake_server_module = type(sys)("server")
        fake_server_module.PromptServer = FakePromptServer
        monkeypatch.setitem(sys.modules, "server", fake_server_module)

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            # The engine invokes the callback per step and does NOT guard it
            # (engine contract) — if the closure's own guard were missing,
            # this raise would propagate and abort the "run".
            on_frame(_FakeFrame())
            return ("text", "state", "trace")

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        with caplog.at_level("WARNING"):
            result = node.sample(
                model=object(),
                prompt="hi",
                seed=1,
                num_inference_steps=1,
                t_min=0.1,
                t_max=0.5,
                entropy_bound=0.1,
                confidence=0.1,
                gen_length=8,
                thinking=False,
                unique_id="42",
            )

        assert result == ("text", "state", "trace")  # output intact, run completed
        assert any("live push failed" in record.message for record in caplog.records)


class _FakeFrame:
    """Minimal stand-in for `dgemma.types.DiffusionFrame` — only the fields
    `_build_on_frame`'s closure actually reads."""

    canvas_idx = 0
    step_idx = 3
    t = 0.2
    temperature = 0.5
    committed_fraction = 1.0
