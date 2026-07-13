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
import torch

from nodes.loader import DGemmaLoader
from nodes.sampler import DGemmaSampler


class _StubModel:
    """Minimal `DGEMMA_MODEL` stand-in exposing only `.processor` — the one
    attribute `DGemmaSampler.sample()` reads (P3's `frames` output) to call
    `decode_frames(model.processor, ...)`. A bare `object()` has no
    `.processor` and would raise `AttributeError` the moment `sample()`
    reaches that call."""

    processor = object()


class _StubTrace:
    """Minimal `DGEMMA_CANVAS_TRACE` stand-in exposing only `.frames` — see
    `_StubModel`. Empty by default so the real (unmocked) `decode_frames`
    degrades to `[]` rather than needing a real tokenizer."""

    frames = ()


def test_nodes_modules_do_not_import_comfy():
    assert not any(m == "comfy" or m.startswith("comfy.") for m in sys.modules)


def test_loader_input_types_declares_repo_id_and_quant():
    spec = DGemmaLoader.INPUT_TYPES()
    assert "repo_id" in spec["required"]
    assert "quant" in spec["required"]
    assert spec["required"]["quant"][0] == ["none"]


def test_loader_quant_selector_does_not_offer_nf4_or_int8():
    """Issue #18 — the door-hardening test: bitsandbytes cannot quantize
    DiffusionGemma's fused 3D MoE experts, so nf4/int8 were misleading on any
    hardware for this architecture and must not be offered at all, not just
    de-defaulted."""
    spec = DGemmaLoader.INPUT_TYPES()
    assert "nf4" not in spec["required"]["quant"][0]
    assert "int8" not in spec["required"]["quant"][0]


def test_loader_quant_default_is_none_not_nf4():
    """Issue #4: nf4 OOMs structurally on the 48GB dev box (bnb can't
    quantize the fused 3D MoE experts); "none" (bf16 CPU-spill) is the one
    with verified PASSes. The widget default must reflect that, not the
    stale nf4 default."""
    spec = DGemmaLoader.INPUT_TYPES()
    assert spec["required"]["quant"][1]["default"] == "none"


def test_loader_input_types_declares_local_files_only():
    """local_files_only stays off (network download) by default — only an
    explicit opt-in restricts resolution to the local HF cache."""
    spec = DGemmaLoader.INPUT_TYPES()
    assert spec["required"]["local_files_only"] == ("BOOLEAN", {"default": False})


def test_loader_declarations():
    assert DGemmaLoader.RETURN_TYPES == ("DGEMMA_MODEL",)
    assert DGemmaLoader.FUNCTION == "load"
    assert DGemmaLoader.CATEGORY == "DiffusionGemma"


def test_loader_calls_load_model_and_wraps_tuple(monkeypatch):
    sentinel = object()
    captured = {}

    def fake_load_model(repo_id, quant, local_files_only):
        captured["repo_id"] = repo_id
        captured["quant"] = quant
        captured["local_files_only"] = local_files_only
        return sentinel

    monkeypatch.setattr("nodes.loader.load_model", fake_load_model)

    node = DGemmaLoader()
    result = node.load(repo_id="google/diffusiongemma-26B-A4B-it", quant="none", local_files_only=True)

    assert result == (sentinel,)
    assert captured == {
        "repo_id": "google/diffusiongemma-26B-A4B-it",
        "quant": "none",
        "local_files_only": True,
    }


def test_loader_local_files_only_defaults_to_false_when_omitted(monkeypatch):
    """A graph built before this widget existed (or a caller that only passes
    the required kwargs) must still get the download-permitting default, not
    a TypeError."""
    captured = {}

    def fake_load_model(repo_id, quant, local_files_only):
        captured["local_files_only"] = local_files_only
        return object()

    monkeypatch.setattr("nodes.loader.load_model", fake_load_model)

    node = DGemmaLoader()
    node.load(repo_id="google/diffusiongemma-26B-A4B-it", quant="none")

    assert captured["local_files_only"] is False


def test_sampler_declarations():
    assert DGemmaSampler.RETURN_TYPES == (
        "STRING",
        "DGEMMA_CANVAS_STATE",
        "DGEMMA_CANVAS_TRACE",
        "STRING",
        "IMAGE",
    )
    assert DGemmaSampler.RETURN_NAMES == (
        "text",
        "canvas_state",
        "canvas_trace",
        "frames",
        "images",
    )
    # `frames` (P3, the per-step flipbook STRING list) is the only
    # `OUTPUT_IS_LIST=True` output. `frames_image` (issue #21 rework) is a
    # single stacked (N, H, W, 3) batch tensor, NOT a list — a wrong flag
    # here would make ComfyUI fan out per-frame and break
    # PreviewImage/SaveAnimatedWEBP/VHS, which all expect one batch tensor.
    assert DGemmaSampler.OUTPUT_IS_LIST == (False, False, False, True, False)
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
    # Issue #22 honesty finding: `thinking` carries an on-widget (hover)
    # tooltip marking it experimental — the injected-message path is
    # pinned one token short of native `enable_thinking=True`
    # (tests/test_chat_template_thinking.py) and behavioral impact is
    # unverified. Default must stay unchanged; the tooltip is additive.
    assert spec["required"]["thinking"][0] == "BOOLEAN"
    assert spec["required"]["thinking"][1]["default"] is False
    assert "EXPERIMENTAL" in spec["required"]["thinking"][1]["tooltip"]
    # Grounded defaults (plan.md Phase 2 / CLAUDE.md).
    assert spec["required"]["num_inference_steps"][1]["default"] == 48
    assert spec["required"]["t_min"][1]["default"] == pytest.approx(0.4)
    assert spec["required"]["t_max"][1]["default"] == pytest.approx(0.8)
    assert spec["required"]["entropy_bound"][1]["default"] == pytest.approx(0.1)
    assert spec["required"]["confidence"][1]["default"] == pytest.approx(0.005)
    assert spec["required"]["gen_length"][1]["default"] == 256


def test_sampler_calls_run_diffusion_and_wraps_tuple(monkeypatch):
    sentinel_model = _StubModel()
    sentinel_trace = _StubTrace()
    sentinel_image_batch = object()
    captured = {}

    def fake_run_diffusion(model, prompt, **kwargs):
        captured["model"] = model
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return ("decoded text", "canvas-state-stub", sentinel_trace)

    def fake_decode_frames(processor, frames):
        captured["decode_processor"] = processor
        captured["decode_frames"] = frames
        return ["frame 0", "frame 1"]

    def fake_render_frames_to_image_batch(frames, **kwargs):
        captured["render_frames"] = frames
        captured["render_kwargs"] = kwargs
        return sentinel_image_batch

    monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)
    monkeypatch.setattr("nodes.sampler.decode_frames", fake_decode_frames)
    monkeypatch.setattr("nodes.sampler.render_frames_to_image_batch", fake_render_frames_to_image_batch)

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

    assert result == (
        "decoded text",
        "canvas-state-stub",
        sentinel_trace,
        ["frame 0", "frame 1"],
        sentinel_image_batch,
    )
    assert captured["model"] is sentinel_model
    # `decode_frames` is called with the model's processor and the trace's
    # own frames — one helper call, no logic of its own (ADR-CDG-003).
    assert captured["decode_processor"] is sentinel_model.processor
    assert captured["decode_frames"] is sentinel_trace.frames
    # `render_frames_to_image_batch` reuses the SAME decoded strings
    # `decode_frames` just produced — one decode, two renderings, never a
    # second re-decode of `canvas_trace.frames`.
    assert captured["render_frames"] == ["frame 0", "frame 1"]
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


def test_sampler_frames_image_output_is_a_stacked_batch_tensor_not_a_list(monkeypatch):
    """The 5th output contract (issue #21 rework): `frames_image` is a
    SINGLE stacked `(N, H, W, 3)` float32 `[0, 1]` `IMAGE` batch tensor, with
    `N == len(frames)` — never a list of N single-frame tensors. A list here
    would fan out per-frame under ComfyUI's `OUTPUT_IS_LIST` machinery and
    break `PreviewImage`'s scrubber, `SaveAnimatedWEBP`, and VHS, all of
    which expect one batch tensor (`nodes/sampler.py`'s own docstring).

    `run_diffusion`/`decode_frames` are mocked (as every other test in this
    file mocks them) but `render_frames_to_image_batch` runs FOR REAL here —
    this is the one test in this file proving the actual rendering, not just
    that it was called.
    """
    sentinel_trace = _StubTrace()
    decoded_frames = ["noise noise noise", "partial coherent text", "the sky is blue"]

    def fake_run_diffusion(model, prompt, **kwargs):
        return ("decoded text", "canvas-state-stub", sentinel_trace)

    def fake_decode_frames(processor, frames):
        return decoded_frames

    monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)
    monkeypatch.setattr("nodes.sampler.decode_frames", fake_decode_frames)

    node = DGemmaSampler()
    result = node.sample(
        model=_StubModel(),
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

    assert len(result) == 5
    _text, _canvas_state, _canvas_trace, frames, frames_image = result
    assert frames == decoded_frames

    assert isinstance(frames_image, torch.Tensor)
    assert not isinstance(frames_image, list)
    assert frames_image.dtype == torch.float32
    assert frames_image.dim() == 4  # (N, H, W, 3)
    num_steps, _height, _width, channels = frames_image.shape
    assert num_steps == len(decoded_frames)
    assert channels == 3
    assert torch.all(frames_image >= 0.0)
    assert torch.all(frames_image <= 1.0)


def test_sampler_forwards_non_default_thinking_and_confidence(monkeypatch):
    """Distinct-from-default values must actually thread through — a node
    that silently forwarded its own hardcoded constants instead of the
    passed-in widget values would pass the test above (defaults match) but
    fail this one."""
    captured = {}

    def fake_run_diffusion(model, prompt, **kwargs):
        captured["kwargs"] = kwargs
        return ("text", "state", _StubTrace())

    monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

    node = DGemmaSampler()
    node.sample(
        model=_StubModel(),
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


def _assert_result_with_no_captured_frames(result, *, text: str, state: str, trace) -> None:
    """Shared assertion for `TestLiveFramePush`'s fixtures below: every one
    uses a `_StubTrace()` with `frames=()`, so `frames` decodes to `[]` and
    `frames_image` (real, unmocked `render_frames_to_image_batch`) renders
    the honest empty `(0, 1, 1, 3)` batch — see
    `tests/test_frames_image.py::TestRenderOutputContract::test_no_frames_yields_empty_batch_not_a_crash`
    for that helper's own degenerate-input contract."""
    got_text, got_state, got_trace, got_frames, got_image = result
    assert got_text == text
    assert got_state == state
    assert got_trace is trace
    assert got_frames == []
    assert got_image.shape == (0, 1, 1, 3)


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
        trace_stub = _StubTrace()

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())  # the collector always calls on_frame per step
            return ("text", "state", trace_stub)

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        result = node.sample(
            model=_StubModel(),
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

        _assert_result_with_no_captured_frames(result, text="text", state="state", trace=trace_stub)

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
            return ("text", "state", _StubTrace())

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        node.sample(
            model=_StubModel(),
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

        trace_stub = _StubTrace()

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            on_frame(_FakeFrame())  # must not raise
            return ("text", "state", trace_stub)

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        result = node.sample(
            model=_StubModel(),
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

        _assert_result_with_no_captured_frames(result, text="text", state="state", trace=trace_stub)

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

        trace_stub = _StubTrace()

        def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
            # The engine invokes the callback per step and does NOT guard it
            # (engine contract) — if the closure's own guard were missing,
            # this raise would propagate and abort the "run".
            on_frame(_FakeFrame())
            return ("text", "state", trace_stub)

        monkeypatch.setattr("nodes.sampler.run_diffusion", fake_run_diffusion)

        node = DGemmaSampler()
        with caplog.at_level("WARNING"):
            result = node.sample(
                model=_StubModel(),
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

        _assert_result_with_no_captured_frames(result, text="text", state="state", trace=trace_stub)  # output intact, run completed
        assert any("live push failed" in record.message for record in caplog.records)


class _FakeFrame:
    """Minimal stand-in for `dgemma.types.DiffusionFrame` — only the fields
    `_build_on_frame`'s closure actually reads."""

    canvas_idx = 0
    step_idx = 3
    t = 0.2
    temperature = 0.5
    committed_fraction = 1.0
