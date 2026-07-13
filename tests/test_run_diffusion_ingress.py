"""`run_diffusion`-level ingress wiring tests (ADR-CDG-010/011, issue #64
Phase 1 — "payloads validated-then-ignored beyond the reject paths").

Phase 1 done-criteria (issue #64 §6): a valid `constraints=`/
`control_signals=`/`capture=` payload passes ingress and the run behaves
IDENTICALLY to a run with no payload at all (no participant is built from it
yet — that's Phases 3/4). An invalid payload still raises at `run_diffusion`
entry, before the scheduler/pipeline is even constructed. This module proves
both halves at the `run_diffusion` boundary, not just at the `dgemma.ingress`
unit level (`tests/test_ingress.py` covers that).

Uses the same lightweight fake-pipeline idiom as
`tests/test_run_diffusion_knobs.py` (not the heavier R4 fixture in
`tests/conftest.py`, which Phase 1 doesn't need — no participant/hook body
exists yet to drive through it).
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.payloads import Binding, Constraints, ControlSignals, Pin
from dgemma.types import DGemmaModel


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


def _fake_model() -> DGemmaModel:
    return DGemmaModel(
        model=object(), processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


def _install_fakes(monkeypatch, *, num_steps: int = 2, scheduler_kwargs_out: dict | None = None):
    class FakeScheduler:
        def __init__(self, **kwargs):
            if scheduler_kwargs_out is not None:
                scheduler_kwargs_out.update(kwargs)
            self.num_inference_steps = kwargs["num_inference_steps"]

        @property
        def config(self):
            return self

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self.eos_token_id = 999

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            for step_idx in range(num_steps):
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True]]),
                    "canvas": torch.tensor([[step_idx]]),
                }
                callback(self, step_idx, step_idx, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor([num_steps], dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestValidPayloadsAreIgnoredBehaviorally:
    """Phase 1's central done-criterion: a valid payload changes NOTHING
    about the run's output (participants aren't wired yet) — only an
    invalid one is observable, via the reject path."""

    def test_valid_constraints_payload_produces_identical_trace_to_none(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=3)
        text_a, state_a, trace_a = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=3
        )

        _install_fakes(monkeypatch, num_steps=3)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))
        text_b, state_b, trace_b = run_diffusion(
            _fake_model(),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            constraints=constraints,
        )

        assert text_a == text_b
        assert state_a.committed_fraction == state_b.committed_fraction
        assert trace_a.scheduler_config == trace_b.scheduler_config
        telemetry_a = [(f.t, f.temperature, f.committed_fraction_per_example) for f in trace_a.frames]
        telemetry_b = [(f.t, f.temperature, f.committed_fraction_per_example) for f in trace_b.frames]
        assert telemetry_a == telemetry_b

    def test_valid_control_signals_payload_produces_identical_trace_to_none(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=3)
        text_a, state_a, trace_a = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=3
        )

        _install_fakes(monkeypatch, num_steps=3)
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0), low=0.02, high=0.3),)
        )
        text_b, state_b, trace_b = run_diffusion(
            _fake_model(),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            control_signals=control_signals,
        )

        # No walker exists yet (Phase 4) — the scheduler's config is never
        # mutated mid-run by this payload in Phase 1, so the trace is
        # byte-identical to a run with no control_signals at all.
        assert text_a == text_b
        assert trace_a.scheduler_config == trace_b.scheduler_config

    class _FakeCaptureSpec:
        def __init__(self, keep_frames="all"):
            self.keep_frames = keep_frames

    def test_valid_capture_payload_produces_identical_trace_to_none(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=2)
        text_a, _state_a, trace_a = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )

        _install_fakes(monkeypatch, num_steps=2)
        text_b, _state_b, trace_b = run_diffusion(
            _fake_model(),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            capture=self._FakeCaptureSpec(keep_frames="all"),
        )

        assert text_a == text_b
        assert len(trace_a.frames) == len(trace_b.frames)


class TestRejectPathsAreLiveAtRunDiffusion:
    """The reject half is NOT deferred — an invalid payload raises at
    `run_diffusion` entry, before the scheduler is even constructed
    (`scheduler_kwargs_out` stays empty), matching issue #64 §6's "reject
    paths live, no participant behavior" framing."""

    def test_invalid_constraints_raises_before_scheduler_construction(self, monkeypatch):
        scheduler_kwargs: dict = {}
        _install_fakes(monkeypatch, scheduler_kwargs_out=scheduler_kwargs)
        constraints = Constraints(pins=(Pin(position=999, token_id=1),))

        with pytest.raises(ValueError, match="out of range"):
            run_diffusion(
                _fake_model(),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                gen_length=10,
                constraints=constraints,
            )
        assert scheduler_kwargs == {}, "scheduler must not be constructed when ingress rejects"

    def test_invalid_control_signals_raises_before_scheduler_construction(self, monkeypatch):
        scheduler_kwargs: dict = {}
        _install_fakes(monkeypatch, scheduler_kwargs_out=scheduler_kwargs)
        control_signals = ControlSignals(
            bindings=(Binding(target="num_inference_steps", signal=(0.0, 1.0), low=1.0, high=10.0),)
        )

        with pytest.raises(ValueError, match="'num_inference_steps' is rejected"):
            run_diffusion(
                _fake_model(),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                control_signals=control_signals,
            )
        assert scheduler_kwargs == {}

    def test_conflicting_hook_sources_raises_before_scheduler_construction(self, monkeypatch):
        scheduler_kwargs: dict = {}
        _install_fakes(monkeypatch, scheduler_kwargs_out=scheduler_kwargs)

        def a_hook(module, args, output):
            return output

        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        with pytest.raises(ValueError, match="cannot both be given"):
            run_diffusion(
                _fake_model(),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                constraints=constraints,
                logit_hook=a_hook,
            )
        assert scheduler_kwargs == {}

    def test_invalid_capture_raises_before_scheduler_construction(self, monkeypatch):
        scheduler_kwargs: dict = {}
        _install_fakes(monkeypatch, scheduler_kwargs_out=scheduler_kwargs)

        class BadCaptureSpec:
            keep_frames = "sometimes"

        with pytest.raises(ValueError, match="keep_frames must be 'last' or 'all'"):
            run_diffusion(
                _fake_model(),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                capture=BadCaptureSpec(),
            )
        assert scheduler_kwargs == {}


class TestNonePayloadsRegressionFloor:
    """No behavior change for `None` payloads (issue #64 §6 regression
    floor) — a bare call with none of the three new kwargs given at all
    still works exactly as before."""

    def test_bare_call_with_no_new_kwargs_unaffected(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=2)
        text, state, trace = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )
        assert text == "TEXT:2"
        assert state.steps_used == 2
        assert len(trace.frames) == 2
