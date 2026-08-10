"""Enforcement tests for `dgemma.hooks.install_logit_shaping_hook` (#35 R5,
F4) — the engine's sole sanctioned forward-hook installation path
(ADR-CDG-010 Decision 5).

This IS the enforcement surface ARCHITECTURE.md's enforcement-surface table
names as "Zero hooks after run ('no hook survives a `run_diffusion` call')
| Forward-hook lifecycle context-manager test, clean + raising".

Two tiers, per the task brief:

- `TestInstallLogitShapingHookUnit` — the context manager in isolation,
  against `tests/conftest.py`'s `HookRecordingModel` (#35 R4), covering
  clean/raising/no-op paths directly and cheaply.
- `TestRunDiffusionHookLifecycle` — the `run_diffusion`-level no-op path
  (`logit_hook=None`) plus the rule-7 rejection of a bare caller-supplied
  `logit_hook=` (issue #221 — see `dgemma.ingress.
  reject_conflicting_hook_sources`).
- `TestRunDiffusionHookLifecycleWithConstraints` — the full
  `run_diffusion`-level enforcement: a real `run_diffusion` call (fakes
  installed the same way `tests/test_run_diffusion_knobs.py`/
  `test_run_diffusion_cancel.py` do) driving a `HookRecordingModel`
  through the clean, cancelled, and exception-raising paths via the
  engine-internal `constraints=` -> hook path (the only path left that
  installs a real hook through this door since #221), asserting
  `live_hook_count == 0` after each — the actual invariant a caller
  relies on, not just the primitive's own contract.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.composite import DiffusionCancelled
from dgemma.hooks import install_logit_shaping_hook
from dgemma.loop import run_diffusion
from dgemma.payloads import Constraints, Pin
from dgemma.types import DGemmaModel


class TestInstallLogitShapingHookUnit:
    """`install_logit_shaping_hook` in isolation, against the R4
    hook-recording model — no `run_diffusion` involved."""

    def test_clean_exit_removes_the_hook(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        with install_logit_shaping_hook(built.model, lambda mod, inp, out: None):
            assert built.model.live_hook_count == 1
        assert built.model.live_hook_count == 0

    def test_hook_fires_during_the_block(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        calls = []
        with install_logit_shaping_hook(built.model, lambda mod, inp, out: calls.append(out.logits.shape)):
            built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))
        assert calls == [(1, 4, built.model.vocab_size)]

    def test_exception_inside_the_block_still_removes_the_hook(self, fake_pipeline_factory):
        """The raising path: an arbitrary exception mid-block must not skip
        teardown — `try/finally`, not a bare `try/except`."""
        built = fake_pipeline_factory()
        with pytest.raises(RuntimeError, match="boom"):
            with install_logit_shaping_hook(built.model, lambda mod, inp, out: None):
                assert built.model.live_hook_count == 1
                raise RuntimeError("boom")
        assert built.model.live_hook_count == 0

    def test_diffusion_cancelled_inside_the_block_still_removes_the_hook(self, fake_pipeline_factory):
        """The specific exception `run_diffusion` distinguishes for partial
        return must be no exception to hook teardown — it propagates through
        this context manager exactly like any other."""
        built = fake_pipeline_factory()
        with pytest.raises(DiffusionCancelled):
            with install_logit_shaping_hook(built.model, lambda mod, inp, out: None):
                assert built.model.live_hook_count == 1
                raise DiffusionCancelled(step_idx=1)
        assert built.model.live_hook_count == 0

    def test_none_hook_fn_installs_nothing(self, fake_pipeline_factory):
        """The default, unwired shape (today's only real `run_diffusion`
        call): no `constraints=` payload means no hook function, and this
        context manager must not call `register_forward_hook` at all —
        zero cost, zero installed hooks, trivially satisfying the
        invariant rather than installing-then-immediately-removing a
        pass-through hook."""
        built = fake_pipeline_factory()
        with install_logit_shaping_hook(built.model, None):
            assert built.model.live_hook_count == 0
        assert built.model.live_hook_count == 0
        assert built.model.install_log == []

    def test_nested_hooks_each_torn_down_independently(self, fake_pipeline_factory):
        """Not a load-bearing usage today (one hook per run), but pins that
        the context manager doesn't assume it is the only hook ever
        installed on the model — nesting tears down cleanly inner-then-outer."""
        built = fake_pipeline_factory()
        with install_logit_shaping_hook(built.model, lambda mod, inp, out: None):
            assert built.model.live_hook_count == 1
            with install_logit_shaping_hook(built.model, lambda mod, inp, out: None):
                assert built.model.live_hook_count == 2
            assert built.model.live_hook_count == 1
        assert built.model.live_hook_count == 0


# --- run_diffusion-level enforcement -----------------------------------


class FakeSchedulerOutput:
    def __init__(self, accepted: list[list[bool]]):
        self.accepted_index = torch.tensor(accepted, dtype=torch.bool)


class FakePipelineOutput:
    def __init__(self, sequences: list[torch.Tensor]):
        self.sequences = sequences
        self.texts = ["<<unused>>"]


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _model_with_hook_recording(hook_model) -> DGemmaModel:
    return DGemmaModel(
        model=hook_model, processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


def _install_hook_lifecycle_fakes(monkeypatch, *, num_steps: int, raise_on_step: int | None = None):
    """Mirrors `tests/test_run_diffusion_cancel.py`'s `_install_multistep_fakes`,
    but the pipeline never touches `pipe.model` itself (the real pipeline
    doesn't either — `run_diffusion` reaches the model only via
    `dgemma_model.model`, which is what `install_logit_shaping_hook` wraps).
    `raise_on_step`, when given, raises a plain `RuntimeError` from inside the
    pipeline call at that step index — simulating a participant defect or a
    real model error mid-run, the "any other exception" path."""

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.num_inference_steps = kwargs["num_inference_steps"]

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self.model = model
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            for step_idx in range(num_steps):
                if raise_on_step is not None and step_idx == raise_on_step:
                    raise RuntimeError("simulated mid-run failure")
                # Real forward pass through `pipe.model` — proves a
                # `logit_hook` installed via `install_logit_shaping_hook`
                # actually fires on the same object `run_diffusion` reaches,
                # not a stand-in the context manager never touches.
                self.model(decoder_input_ids=torch.zeros((1, 2), dtype=torch.long))
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True]]),
                    "canvas": torch.tensor([[step_idx]]),
                }
                callback(self, step_idx, step_idx, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor([num_steps], dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestRunDiffusionHookLifecycle:
    """The invariant as `run_diffusion`'s own caller experiences it: whatever
    path the call takes, zero hooks remain on the model afterward.

    A caller-supplied `logit_hook=` is now rejected outright at ingress
    (rule-7 hardening, issue #221) — `TestRunDiffusionHookLifecycleWithConstraints`
    below covers the same clean/cancelled/raising lifecycle for the one
    remaining way a hook reaches `install_logit_shaping_hook`: the engine's
    own internal build from `constraints=`. This class now covers the
    `logit_hook=None` no-op path plus the new bare-closure rejection."""

    def test_clean_run_with_no_logit_hook_leaves_zero_hooks(self, monkeypatch, fake_pipeline_factory):
        """Today's only real call shape (`logit_hook=None`): confirms the
        no-op path costs nothing and leaves nothing installed."""
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=3)

        run_diffusion(_model_with_hook_recording(built.model), "hi")

        assert built.model.live_hook_count == 0

    def test_bare_logit_hook_rejected_before_any_hook_installs(self, monkeypatch, fake_pipeline_factory):
        """Rule-7 hole (#221): a caller-supplied `logit_hook=` — with no
        `constraints=` in the picture — must raise at ingress with a named
        remedy, BEFORE the scheduler/pipeline are constructed and before
        `install_logit_shaping_hook` ever runs. Zero hooks touch the model
        on this reject path."""
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=3)

        with pytest.raises(ValueError, match="logit_hook= is engine-internal only"):
            run_diffusion(
                _model_with_hook_recording(built.model),
                "hi",
                logit_hook=lambda mod, inp, out: None,
            )

        assert built.model.install_log == []
        assert built.model.live_hook_count == 0


class TestRunDiffusionHookLifecycleWithConstraints:
    """Issue #64 Phase 3 extension: the SAME zero-hooks-after-run invariant,
    now exercised with a `constraints=` payload (the ENGINE-BUILT hook
    `dgemma.constraints_hook.build_logit_mask_hook`, installed through the
    exact same `install_logit_shaping_hook` path). Since issue #221 rejects
    every caller-supplied `logit_hook=` at ingress, this class is now the
    ONLY place a real hook is installed and torn down through a full
    `run_diffusion` call — proving the clean/cancelled/raising lifecycle
    for the one remaining path a hook reaches this door, matching the task
    brief's "extend test_hook_lifecycle.py — zero hooks after
    clean/cancelled/raising runs WITH a constraints payload"."""

    def test_clean_run_with_constraints_installs_and_tears_down(self, monkeypatch, fake_pipeline_factory):
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=3)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        run_diffusion(_model_with_hook_recording(built.model), "hi", constraints=constraints)

        # The engine-built hook actually installed (proves constraints=
        # really drives the hook path, not a vacuous no-op) and tore down
        # cleanly.
        assert built.model.install_log == [0]
        assert built.model.live_hook_count == 0

    def test_cancelled_run_with_constraints_still_tears_down_the_hook(self, monkeypatch, fake_pipeline_factory):
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=5)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        text, canvas_state, canvas_trace = run_diffusion(
            _model_with_hook_recording(built.model),
            "hi",
            should_cancel=lambda: True,
            constraints=constraints,
        )

        assert canvas_trace.frames  # cancelled but evidence-bearing, per #38
        assert built.model.install_log == [0]
        assert built.model.live_hook_count == 0

    def test_raising_run_with_constraints_still_tears_down_the_hook(self, monkeypatch, fake_pipeline_factory):
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=5, raise_on_step=2)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        with pytest.raises(RuntimeError, match="simulated mid-run failure"):
            run_diffusion(
                _model_with_hook_recording(built.model),
                "hi",
                constraints=constraints,
            )

        assert built.model.install_log == [0]
        assert built.model.live_hook_count == 0

    def test_constraints_and_logit_hook_together_still_rejected_at_ingress(
        self, monkeypatch, fake_pipeline_factory
    ):
        """H1 (ADR-CDG-010 D5) still applies even though `constraints=`
        builds its own hook internally: the two-source-on-one-door reject
        is unconditional, checked BEFORE any hook is installed — zero hooks
        on this reject path too."""
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=3)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        with pytest.raises(ValueError, match="cannot both be given"):
            run_diffusion(
                _model_with_hook_recording(built.model),
                "hi",
                constraints=constraints,
                logit_hook=lambda mod, inp, out: None,
            )

        assert built.model.install_log == []
        assert built.model.live_hook_count == 0

    def test_two_sequential_constrained_runs_on_one_model_each_leave_zero_hooks(
        self, monkeypatch, fake_pipeline_factory
    ):
        """The F4 failure mode named directly: run A's engine-built hook
        must not survive into run B. Two back-to-back `constraints=` calls
        on the SAME model object each leave `live_hook_count == 0` — two
        independent install/remove cycles, not one hook reused."""
        built = fake_pipeline_factory()
        _install_hook_lifecycle_fakes(monkeypatch, num_steps=2)
        model = _model_with_hook_recording(built.model)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        run_diffusion(model, "hi", constraints=constraints)
        assert built.model.live_hook_count == 0

        run_diffusion(model, "hi", constraints=constraints)
        assert built.model.live_hook_count == 0
        assert built.model.install_log == [0, 1]
        assert built.model.removal_log == [0, 1]
