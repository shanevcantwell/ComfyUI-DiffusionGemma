"""surfaces/comfyui/sampler.py's `_build_should_cancel` closure (issue #140
sampler half, closes #38's surface-wiring gap).

The engine-side cancellation seam (`dgemma.composite._CancellationParticipant`,
`dgemma.loop.run_diffusion`'s `should_cancel` param + `DiffusionCancelled`
partial-return handling) already exists and is already tested in isolation
(`tests/test_run_diffusion_cancel.py`, `tests/test_step_end_composite.py`).
This file covers ONLY the surface connection: does the sampler's closure
correctly read `comfy.model_management.processing_interrupted()`, and does
`DGemmaSampler.sample()` pass that closure into `run_diffusion` as
`should_cancel`. No new cancellation semantics are exercised here — that
would duplicate the engine-side suite.

`comfy` is genuinely absent from this venv (ComfyUI is not a dependency of
this repo, by design — same posture `tests/test_seam.py` and
`tests/test_loader_folder_paths.py` already document for `folder_paths`).
`_build_should_cancel` imports `comfy.model_management` lazily, INSIDE the
returned closure — the same shape `live_view.build_on_frame` already uses for
`from server import PromptServer` — so:

- the "no interrupt" case (a) needs no fake at all: the real `ImportError`
  from a genuinely-missing `comfy` package IS the no-comfy path, exercised
  directly (mirrors `test_sampler_node_frames_real`'s `unique_id=None`
  headless-path note in `tests/test_live_seams.py`: the absence is the
  normal condition, not a gap to paper over).
- the "faked comfy says True" case (b) needs a real, importable
  `comfy.model_management` submodule injected via `sys.modules` (a bare
  attribute stub on a `sys.modules["comfy"]` object would NOT satisfy
  `import comfy.model_management as model_management`, which walks the
  import system, not attribute access) — the same throwaway-module-injection
  idea `tests/test_mcp_import_guard.py` uses for the inverse (blocking a
  present package), applied here to construct an absent one.

Fixture teardown always pops both `comfy` and `comfy.model_management` from
`sys.modules` so this fake never leaks into another test module (same
discipline as `test_dual_context_import.py`'s `synthetic_pack_root` fixture).
"""
from __future__ import annotations

import sys
import types

import pytest

import surfaces.comfyui.sampler as sampler_module


@pytest.fixture
def fake_comfy_model_management():
    """Injects an importable `comfy.model_management` module into
    `sys.modules` with a controllable `processing_interrupted()`. Yields the
    fake submodule so a test can flip its return value; tears down both
    `sys.modules` entries afterward regardless of what else ran."""

    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.__path__ = []  # marks it as a package so submodule import resolves
    model_management = types.ModuleType("comfy.model_management")
    model_management.processing_interrupted = lambda: False  # default: not interrupted

    sys.modules["comfy"] = comfy_pkg
    sys.modules["comfy.model_management"] = model_management
    try:
        yield model_management
    finally:
        sys.modules.pop("comfy.model_management", None)
        sys.modules.pop("comfy", None)


class TestBuildShouldCancel:
    def test_returns_false_when_comfy_is_absent(self):
        """(a) No ComfyUI present (the real condition in this venv, no fake
        needed) — the closure must degrade to "never cancel", never raise."""
        assert "comfy" not in sys.modules  # sanity: genuinely absent here

        should_cancel = sampler_module._build_should_cancel()

        assert should_cancel() is False

    def test_returns_false_when_comfy_reports_not_interrupted(self, fake_comfy_model_management):
        fake_comfy_model_management.processing_interrupted = lambda: False

        should_cancel = sampler_module._build_should_cancel()

        assert should_cancel() is False

    def test_returns_true_when_comfy_reports_interrupted(self, fake_comfy_model_management):
        """(b) The faked `processing_interrupted()` says True — the closure
        must surface exactly that, unmodified."""
        fake_comfy_model_management.processing_interrupted = lambda: True

        should_cancel = sampler_module._build_should_cancel()

        assert should_cancel() is True

    def test_swallows_unexpected_exceptions_and_reports_not_cancelled(
        self, fake_comfy_model_management
    ):
        """Display/interrupt plumbing must never kill generation (the same
        non-negotiable rule `live_view.build_on_frame`'s docstring states for the live
        push) — a `processing_interrupted()` that itself raises degrades to
        "not cancelled", logged, not propagated."""

        def _boom():
            raise RuntimeError("simulated comfy internals failure")

        fake_comfy_model_management.processing_interrupted = _boom

        should_cancel = sampler_module._build_should_cancel()

        assert should_cancel() is False

    def test_returns_a_fresh_closure_each_call(self):
        """Not load-bearing behavior, just documents the shape: each call
        builds an independent closure (mirrors `live_view.build_on_frame`'s pattern),
        not a shared singleton predicate."""
        first = sampler_module._build_should_cancel()
        second = sampler_module._build_should_cancel()

        assert first is not second


class TestSamplerNodeWiresShouldCancel:
    def test_sample_passes_should_cancel_into_run_diffusion(self, monkeypatch):
        """(c) `DGemmaSampler.sample()` must pass a `should_cancel=` kwarg
        into `run_diffusion` — captured here via a monkeypatched
        `run_diffusion` stand-in, so this test proves the WIRING (the sampler
        forwards a callable seam), not the engine-side cancellation behavior
        itself (covered by `tests/test_run_diffusion_cancel.py`)."""
        captured_kwargs = {}

        class FakeTokenizer:
            eos_token_id = 999

            def decode(self, ids, skip_special_tokens=True):
                return "fake decoded text"

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        class FakeModel:
            processor = FakeProcessor()
            repo_id = "fake/repo"
            quant = "none"
            device = "cpu"
            dtype = "bfloat16"

        class FakeCanvasState:
            pass

        class FakeCanvasTrace:
            frames = []

        def fake_run_diffusion(model, prompt, **kwargs):
            captured_kwargs.update(kwargs)
            return ("text", FakeCanvasState(), FakeCanvasTrace())

        def fake_decode_frames(processor, frames):
            return []

        def fake_render_frames_to_image_batch(frames, **kwargs):
            return None

        monkeypatch.setattr(sampler_module, "run_diffusion", fake_run_diffusion)
        # Issue #162/#166 (PR #168): decode_frames/render_frames_to_image_batch
        # calls moved into the shared surfaces.comfyui.emission helper
        # (build_sampler_shaped_outputs) — patched at their new home, not on
        # sampler_module (which no longer imports them directly).
        monkeypatch.setattr("surfaces.comfyui.emission.decode_frames", fake_decode_frames)
        monkeypatch.setattr(
            "surfaces.comfyui.emission.render_frames_to_image_batch",
            fake_render_frames_to_image_batch,
        )

        node = sampler_module.DGemmaSampler()
        node.sample(
            FakeModel(),
            "hi",
            seed=0,
            num_inference_steps=4,
            t_min=0.4,
            t_max=0.8,
            entropy_bound=0.1,
            confidence=0.005,
            gen_length=64,
            thinking=False,
            unique_id=None,
        )

        assert "should_cancel" in captured_kwargs
        assert callable(captured_kwargs["should_cancel"])
        # And it's the real closure's shape — degrades to False with no comfy
        # present, not a stub that always raises or always cancels.
        assert captured_kwargs["should_cancel"]() is False


class TestZeroComfyImportSafety:
    def test_sampler_module_still_importable_with_zero_comfy_present(self):
        """(d) Extends the existing dual-context/seam coverage
        (`tests/test_seam.py`, `tests/test_dual_context_import.py`): the
        cancellation wiring must not change the module's own import-time
        behavior. `surfaces.comfyui.sampler` is already imported (this test
        file imports it at module scope, alongside every other test module in
        the suite) with no `comfy` package installed in this venv — this
        assertion just pins that fact explicitly for THIS module after the
        `should_cancel` addition, the same way `test_seam.py` pins it for
        `dgemma` as a whole."""
        assert "comfy" not in sys.modules
        assert hasattr(sampler_module, "_build_should_cancel")
        assert hasattr(sampler_module, "DGemmaSampler")

    def test_build_should_cancel_does_not_import_comfy_until_the_closure_is_called(self):
        """The lazy-import discipline itself: building the closure must not
        trigger the `comfy` import — only CALLING it does (same shape as
        `live_view.build_on_frame`, whose `from server import PromptServer` line
        lives inside `on_frame`, not inside `build_on_frame` itself)."""
        assert "comfy" not in sys.modules

        should_cancel = sampler_module._build_should_cancel()

        assert "comfy" not in sys.modules  # building the closure alone imports nothing

        should_cancel()  # now it may attempt the import (and fail -> False)

        assert "comfy" not in sys.modules  # and still absent afterward: ImportError, not a stub
