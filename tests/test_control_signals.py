"""End-to-end `control_signals=` tests (ADR-CDG-011, issue #64 Phase 4) — the
control-signal walker LIVE through `run_diffusion`, driven against the R4
fake-pipeline fixture (`tests/conftest.py`) so the walker's
`scheduler.register_to_config` write is exercised against a real
`FakeEntropyBoundScheduler`/`FakeFrozenConfig` pair (the genuine
write-raises-except-through-`register_to_config` mutation path), not a
looser hand-rolled fake.

Same `_wire_fake_pipeline` idiom as `tests/test_constraints.py` (issue #64
Phase 3's end-to-end test module for the sibling `constraints=` payload):
monkeypatches `dgemma.loop.EntropyBoundScheduler`/`dgemma.loop.DGemmaPipeline`
to the CLASSES of a `fake_pipeline_factory()`-built instance, so a real
`run_diffusion` call drives the real canvas/config-threading fixture end to
end (per `tests/test_conftest_fake_pipeline.py`'s import-mode caveat: never a
bare `from tests.conftest import ...` of a fixture class).

This IS the enforcement surface ADR-CDG-011's enforcement-surface table names
for clauses 4-8 (walker write mechanism, exact-temperature mechanism, walker
timing, effective-knob telemetry, same-in/same-out statelessness) at the
`run_diffusion` level; `tests/test_participants.py` covers `WalkerParticipant`
in isolation, and `tests/test_run_diffusion_statelessness.py::
TestWalkerStatePerRun` covers the cross-call containment half.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.payloads import Binding, ControlSignals
from dgemma.types import DGemmaModel


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0
    vocab_size = 8

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _fake_model_with(model) -> DGemmaModel:
    return DGemmaModel(
        model=model, processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


def _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, *, accepted=None, **factory_kwargs):
    """See `tests/test_constraints.py`'s identically named helper for the
    full rationale (re-applying `accepted=` to every scheduler
    `run_diffusion` constructs, since `run_diffusion` always builds its OWN
    fresh scheduler instance — `TestSchedulerFreshPerCall`'s invariant)."""
    built = fake_pipeline_factory(**factory_kwargs)
    scheduler_cls = type(built.scheduler)
    pipeline_cls = type(built.pipeline)

    def _scheduler_factory(**kwargs):
        scheduler = scheduler_cls(**kwargs)
        if accepted is not None:
            scheduler._accepted_source = accepted
        return scheduler

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)
    return built


class TestWalkerWritesThroughToNextFrame:
    """ADR-CDG-011 clause 6/7: the walker's write at step k's callback
    governs step k+1's forward pass, so it must be visible in step k+1's
    `effective_*` telemetry (captured BEFORE the walker runs that step, per
    the fixed `capture -> ... -> walker` order) — never in step k's own
    frame, which was already captured before the walker ran."""

    def test_binding_written_at_step_k_is_visible_in_frame_k_plus_1(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(
            monkeypatch, fake_pipeline_factory, num_inference_steps=3, entropy_bound=0.1, t_min=0.4, t_max=0.8
        )
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0), low=0.0, high=1.0),)
        )

        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            control_signals=control_signals,
        )

        assert len(trace.frames) == 3
        # Step 0's frame reflects the CTOR entropy_bound (0.1) — signal[0] is
        # never applied by the walker (gate ruling O1).
        assert trace.frames[0].effective_entropy_bound == pytest.approx(0.1)
        # Step 1's frame reflects the walker's write made at the end of step
        # 0's callback: signal[1] = 0.5 mapped into [0.0, 1.0] == 0.5.
        assert trace.frames[1].effective_entropy_bound == pytest.approx(0.5)
        # Step 2's frame reflects the write made at the end of step 1's
        # callback: signal[2] = 1.0 mapped into [0.0, 1.0] == 1.0.
        assert trace.frames[2].effective_entropy_bound == pytest.approx(1.0)

    def test_binding_range_mapping_is_low_plus_range_times_raw(self, monkeypatch, fake_pipeline_factory):
        """Decision 4's mapping formula, `value = low + (high - low) *
        signal[k]`, exercised with a non-[0,1] declared range so a bug that
        dropped the affine mapping (e.g. writing the raw [0,1] sample
        directly) would be caught."""
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2)
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.25), low=0.02, high=0.3),)
        )

        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            control_signals=control_signals,
        )

        # step 1's frame: signal[1] = 0.25 -> 0.02 + (0.3 - 0.02) * 0.25 = 0.09
        assert trace.frames[1].effective_entropy_bound == pytest.approx(0.09)


class TestStepZeroNotPreSeeded:
    """Gate ruling O1 (issue #64, 2026-07-13): `signal[0]` is never applied by
    the walker — step 0 always runs under the ctor-supplied config, the
    walker's first write happens at the END of step 0's callback (preparing
    step 1), never before step 0's own forward pass."""

    def test_first_frame_reflects_ctor_config_not_signal_zero(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2, t_min=0.4, t_max=0.8)
        # signal[0] maps to a wildly different t_min/t_max than the ctor's,
        # so a step-0-preseeding bug would be caught by frame 0 NOT matching
        # the ctor values.
        control_signals = ControlSignals(
            bindings=(
                Binding(target="t_min", signal=(1.0, 0.0), low=0.0, high=1.0),
                Binding(target="t_max", signal=(1.0, 0.0), low=0.0, high=1.0),
            )
        )

        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            control_signals=control_signals,
        )

        assert trace.frames[0].effective_t_min == pytest.approx(0.4)
        assert trace.frames[0].effective_t_max == pytest.approx(0.8)


class TestFinalStepIsNoOp:
    """The walker must not attempt to write `signal[N]` at the last step's
    callback (`step_idx == N - 1`) — there is no step N in the schedule, and
    the signal has exactly N samples (ingress V1: length == num_inference_steps),
    so `signal[N]` would be out of bounds."""

    def test_last_step_callback_does_not_raise_and_writes_nothing_further(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2)
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 1.0), low=0.0, high=1.0),)
        )

        # Must not raise (no IndexError from an out-of-bounds signal[2]).
        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            control_signals=control_signals,
        )
        assert len(trace.frames) == 2
        # Step 1 (the last step) still reflects step 0's write (signal[1] = 1.0
        # mapped into [0,1] == 1.0) — the no-op is "don't write signal[2]",
        # not "undo the prior write."
        assert trace.frames[1].effective_entropy_bound == pytest.approx(1.0)


class TestExactPerStepTemperature:
    """ADR-CDG-011 clause 5: binding one signal to BOTH `t_min` and `t_max`
    with the same per-step value collapses `anneal_temperature`'s formula
    (`t_min + (t_max - t_min) * t`) to that exact value every step — the
    sanctioned exact-temperature mechanism, not a workaround."""

    def test_t_min_eq_t_max_binding_yields_exact_temperature_every_step(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=3, t_min=0.4, t_max=0.8)
        # Both bindings share the SAME signal, mapped into the SAME [low, high]
        # range -> t_min == t_max at every governed step.
        shared_signal = (0.0, 0.3, 0.9)
        control_signals = ControlSignals(
            bindings=(
                Binding(target="t_min", signal=shared_signal, low=0.1, high=0.6),
                Binding(target="t_max", signal=shared_signal, low=0.1, high=0.6),
            )
        )

        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            control_signals=control_signals,
        )

        # Step 0 is ctor config (0.4, 0.8) -- t_min != t_max there, expected
        # (O1: signal[0] never applied).
        assert trace.frames[0].effective_t_min != trace.frames[0].effective_t_max
        # Steps 1-2 governed by the walker: t_min == t_max exactly, and
        # temperature == that exact value with zero anneal drift.
        for i in (1, 2):
            frame = trace.frames[i]
            assert frame.effective_t_min == pytest.approx(frame.effective_t_max)
            assert frame.temperature == pytest.approx(frame.effective_t_min)


class TestWalkerSkippedOnCancellation:
    """A cancelled step's walker never runs — the composite's cancellation
    check (position 2, after capture, before every writer AND the walker)
    raises before the writer loop or the walker are reached
    (`dgemma/composite.py`)."""

    def test_cancelled_run_never_writes_through_the_walker(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=5)
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 1.0, 1.0, 1.0, 1.0), low=0.0, high=1.0),)
        )
        def should_cancel() -> bool:
            return True  # cancel on the very first check (step 0's callback)

        # run_diffusion itself catches DiffusionCancelled and returns the
        # partial trace/text — it does not propagate to this test.
        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=5,
            control_signals=control_signals,
            should_cancel=should_cancel,
        )

        # Cancelled after step 0's frame was captured; step 1 never ran, so
        # the walker's step-0 write into signal[1] never had a chance to be
        # observed by a later frame (there is no later frame).
        assert len(trace.frames) == 1
        assert trace.frames[0].effective_entropy_bound == pytest.approx(0.1)


class TestMultipleBindingsOneRegisterCall:
    """Decision 4's fold-in: two bindings governing the same step are
    written in ONE `register_to_config` call (a single whole-dict rebuild),
    not two separate calls that could observably clobber each other."""

    def test_two_bindings_for_the_same_step_both_land(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2)
        control_signals = ControlSignals(
            bindings=(
                Binding(target="entropy_bound", signal=(0.0, 1.0), low=0.0, high=1.0),
                Binding(target="t_min", signal=(0.0, 0.5), low=0.0, high=1.0),
            )
        )

        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            control_signals=control_signals,
        )

        assert trace.frames[1].effective_entropy_bound == pytest.approx(1.0)
        assert trace.frames[1].effective_t_min == pytest.approx(0.5)


class TestNoBindingsIsANoOp:
    """`ControlSignals(bindings=())`/`None` builds no walker at all — a run
    is byte-identical to today's no-`control_signals=` behavior (empty ==
    no-op, `dgemma/payloads.py`)."""

    def test_empty_bindings_produces_identical_trace_to_none(self, monkeypatch, fake_pipeline_factory):
        built_a = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2)
        _, _, trace_a = run_diffusion(
            _fake_model_with(built_a.model), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )

        built_b = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2)
        _, _, trace_b = run_diffusion(
            _fake_model_with(built_b.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            control_signals=ControlSignals(bindings=()),
        )

        telemetry_a = [(f.effective_entropy_bound, f.effective_t_min, f.effective_t_max) for f in trace_a.frames]
        telemetry_b = [(f.effective_entropy_bound, f.effective_t_min, f.effective_t_max) for f in trace_b.frames]
        assert telemetry_a == telemetry_b
