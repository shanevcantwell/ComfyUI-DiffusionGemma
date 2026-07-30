"""surfaces/comfyui/loader.py's `_build_check_interrupted` closure + the
`LoadInterrupted` -> `comfy.model_management.InterruptProcessingException`
translation at the node boundary (issue #140 loader half).

The engine-side interrupt seam (`dgemma.model.load_model`'s `check_interrupted`
kwarg + phase-boundary polling + `LoadInterrupted`) already exists and is
already tested in isolation (`tests/test_model_load.py::TestCheckInterrupted`).
This file covers ONLY the surface connection: does the loader's closure
correctly read `comfy.model_management.processing_interrupted()`, does
`DGemmaLoader.load()` pass that closure into `load_model` as
`check_interrupted`, and does a raised `LoadInterrupted` get translated into
the exact exception type ComfyUI's own executor special-cases for a clean
"Processing interrupted" outcome (`execution.py`'s
`except comfy.model_management.InterruptProcessingException`) rather than the
generic error/traceback path. No new interrupt semantics are exercised here —
that would duplicate the engine-side suite. Structure mirrors
`tests/test_sampler_cancellation.py` throughout (same lazy-import discipline,
same fake-module-injection technique, same zero-comfy-present baseline).

`comfy` is genuinely absent from this venv (ComfyUI is not a dependency of
this repo, by design — same posture `tests/test_seam.py` and
`tests/test_loader_folder_paths.py` already document for `folder_paths`).
Both `_build_check_interrupted` and the translation in `load()` import
`comfy.*` lazily, INSIDE their respective call sites — never at module top —
so:

- the "no interrupt" case needs no fake at all: the real `ImportError` from a
  genuinely-missing `comfy` package IS the no-comfy path.
- the "faked comfy says True" case needs a real, importable
  `comfy.model_management` submodule injected via `sys.modules` (a bare
  attribute stub on a `sys.modules["comfy"]` object would NOT satisfy
  `import comfy.model_management as model_management`, which walks the
  import system, not attribute access) — same technique
  `test_sampler_cancellation.py` uses.

Fixture teardown always pops both `comfy` and `comfy.model_management` from
`sys.modules` so this fake never leaks into another test module.
"""
from __future__ import annotations

import sys
import types

import pytest

import surfaces.comfyui.loader as loader_module
from dgemma.model import LoadInterrupted


@pytest.fixture
def fake_comfy_model_management():
    """Injects an importable `comfy.model_management` module into
    `sys.modules` with a controllable `processing_interrupted()` and a real
    `InterruptProcessingException` class (mirrors the real one's shape: a
    plain exception ComfyUI's executor pattern-matches by type, not
    message). Yields the fake submodule so a test can flip its return value
    or assert on the exception class; tears down both `sys.modules` entries
    afterward regardless of what else ran."""

    comfy_pkg = types.ModuleType("comfy")
    comfy_pkg.__path__ = []  # marks it as a package so submodule import resolves
    model_management = types.ModuleType("comfy.model_management")
    model_management.processing_interrupted = lambda: False  # default: not interrupted

    class _FakeInterruptProcessingException(BaseException):
        pass

    model_management.InterruptProcessingException = _FakeInterruptProcessingException

    sys.modules["comfy"] = comfy_pkg
    sys.modules["comfy.model_management"] = model_management
    try:
        yield model_management
    finally:
        sys.modules.pop("comfy.model_management", None)
        sys.modules.pop("comfy", None)


class TestBuildCheckInterrupted:
    def test_returns_false_when_comfy_is_absent(self):
        """No ComfyUI present (the real condition in this venv, no fake
        needed) — the closure must degrade to "never interrupt", never raise."""
        assert "comfy" not in sys.modules  # sanity: genuinely absent here

        check_interrupted = loader_module._build_check_interrupted()

        assert check_interrupted() is False

    def test_returns_false_when_comfy_reports_not_interrupted(self, fake_comfy_model_management):
        fake_comfy_model_management.processing_interrupted = lambda: False

        check_interrupted = loader_module._build_check_interrupted()

        assert check_interrupted() is False

    def test_returns_true_when_comfy_reports_interrupted(self, fake_comfy_model_management):
        """The faked `processing_interrupted()` says True — the closure must
        surface exactly that, unmodified."""
        fake_comfy_model_management.processing_interrupted = lambda: True

        check_interrupted = loader_module._build_check_interrupted()

        assert check_interrupted() is True

    def test_swallows_unexpected_exceptions_and_reports_not_interrupted(
        self, fake_comfy_model_management
    ):
        """Interrupt-plumbing must never itself crash the load (the same
        non-negotiable rule the sampler's `_build_should_cancel` docstring
        states for its own predicate) — a `processing_interrupted()` that
        itself raises degrades to "not interrupted", logged, not propagated."""

        def _boom():
            raise RuntimeError("simulated comfy internals failure")

        fake_comfy_model_management.processing_interrupted = _boom

        check_interrupted = loader_module._build_check_interrupted()

        assert check_interrupted() is False

    def test_returns_a_fresh_closure_each_call(self):
        """Not load-bearing behavior, just documents the shape: each call
        builds an independent closure (mirrors `_build_should_cancel`'s
        pattern), not a shared singleton predicate."""
        first = loader_module._build_check_interrupted()
        second = loader_module._build_check_interrupted()

        assert first is not second


class TestLoaderNodeWiresCheckInterrupted:
    def test_load_passes_check_interrupted_into_load_model_hf_path(self, monkeypatch):
        """`DGemmaLoader.load()` must pass a `check_interrupted=` kwarg into
        `load_model` on the primary HF-identifier path — captured via a
        monkeypatched `load_model` stand-in, proving the WIRING (the loader
        forwards a callable seam), not the engine-side interrupt behavior
        itself (covered by `tests/test_model_load.py::TestCheckInterrupted`)."""
        captured = {}

        def fake_load_model(repo_id, quant, local_files_only, check_interrupted=None):
            captured["check_interrupted"] = check_interrupted
            return object()

        monkeypatch.setattr(loader_module, "load_model", fake_load_model)

        loader_module.DGemmaLoader().load(quant="none")

        assert callable(captured["check_interrupted"])
        # It's the real closure's shape — degrades to False with no comfy
        # present, not a stub that always raises or always interrupts.
        assert captured["check_interrupted"]() is False

    def test_load_passes_check_interrupted_into_load_model_local_folders_path(self, monkeypatch):
        """The same wiring on the local-folders/dropdown path (issue #17) —
        a resolved local directory still gets the interrupt seam, not just
        the primary HF-identifier flow."""
        captured = {}

        monkeypatch.setattr(loader_module, "_LOCAL_FOLDERS_ENABLED", True)
        monkeypatch.setattr(
            loader_module, "resolve_local_model_dir", lambda name: "/models/diffusion_models/" + name
        )

        def fake_load_model(repo_id, quant, local_files_only, check_interrupted=None):
            captured["check_interrupted"] = check_interrupted
            return object()

        monkeypatch.setattr(loader_module, "load_model", fake_load_model)

        loader_module.DGemmaLoader().load(quant="none", local_model_dir="some-local-model")

        assert callable(captured["check_interrupted"])


class TestLoadInterruptedTranslation:
    """`LoadInterrupted` (raised by `load_model` when `check_interrupted`
    reports `True`) must be translated, at the node boundary, into
    `comfy.model_management.InterruptProcessingException` — the exact type
    ComfyUI's `execution.py` pattern-matches for a clean "Processing
    interrupted" outcome instead of the generic error/traceback path every
    other exception here takes."""

    def test_load_interrupted_becomes_comfy_interrupt_exception(self, fake_comfy_model_management, monkeypatch):
        def fake_load_model(repo_id, quant, local_files_only, check_interrupted=None):
            raise LoadInterrupted("DiffusionGemma load interrupted before phase: model from_pretrained.")

        monkeypatch.setattr(loader_module, "load_model", fake_load_model)

        with pytest.raises(fake_comfy_model_management.InterruptProcessingException):
            loader_module.DGemmaLoader().load(quant="none")

    def test_load_interrupted_translation_chains_the_original_exception(
        self, fake_comfy_model_management, monkeypatch
    ):
        """The translated exception keeps the original `LoadInterrupted` as
        its `__cause__` (an explicit `raise ... from exc`) — the phase-naming
        detail stays reachable for anyone inspecting the traceback, rather
        than being discarded at the translation boundary."""
        original = LoadInterrupted("DiffusionGemma load interrupted before phase: device move.")

        def fake_load_model(repo_id, quant, local_files_only, check_interrupted=None):
            raise original

        monkeypatch.setattr(loader_module, "load_model", fake_load_model)

        with pytest.raises(fake_comfy_model_management.InterruptProcessingException) as excinfo:
            loader_module.DGemmaLoader().load(quant="none")

        assert excinfo.value.__cause__ is original

    def test_non_interrupt_exceptions_are_not_translated(self, fake_comfy_model_management, monkeypatch):
        """A different exception (e.g. the unresolvable-repo `RuntimeError`)
        must propagate as itself — the translation is narrowly scoped to
        `LoadInterrupted`, never a blanket catch that would also swallow or
        relabel unrelated load failures."""

        def fake_load_model(repo_id, quant, local_files_only, check_interrupted=None):
            raise RuntimeError("unrelated load failure")

        monkeypatch.setattr(loader_module, "load_model", fake_load_model)

        with pytest.raises(RuntimeError, match="unrelated load failure"):
            loader_module.DGemmaLoader().load(quant="none")


class TestZeroComfyImportSafety:
    def test_loader_module_still_importable_with_zero_comfy_present(self):
        """Extends the existing dual-context/seam coverage
        (`tests/test_seam.py`, `tests/test_dual_context_import.py`): the
        interrupt wiring must not change the module's own import-time
        behavior. `surfaces.comfyui.loader` is already imported (this test
        file imports it at module scope, alongside every other test module
        in the suite) with no `comfy` package installed in this venv."""
        assert "comfy" not in sys.modules
        assert hasattr(loader_module, "_build_check_interrupted")
        assert hasattr(loader_module, "DGemmaLoader")

    def test_build_check_interrupted_does_not_import_comfy_until_called(self):
        """The lazy-import discipline itself: building the closure must not
        trigger the `comfy` import — only CALLING it does (same shape as
        `_build_should_cancel`)."""
        assert "comfy" not in sys.modules

        check_interrupted = loader_module._build_check_interrupted()

        assert "comfy" not in sys.modules  # building the closure alone imports nothing

        check_interrupted()  # now it may attempt the import (and fail -> False)

        assert "comfy" not in sys.modules  # and still absent afterward: ImportError, not a stub
