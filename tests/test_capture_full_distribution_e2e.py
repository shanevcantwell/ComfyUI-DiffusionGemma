"""End-to-end `capture=` Tier-2 full-distribution tests (ADR-CDG-014
Decision 3/5, issue #61 P-C) — driven through `run_diffusion` against the R4
fake-pipeline fixture (`tests/conftest.py`), mirroring
`tests/test_capture_top_k_e2e.py`'s Tier-1 e2e shape.

This module proves the `run_diffusion` -> `CaptureSpec.capture_full_distribution`
/`max_full_distribution_steps` -> `_FrameCollector` wiring works at the real
call boundary, including the ingress budget-reject path firing BEFORE any
scheduler is constructed (unit coverage of `_FrameCollector` itself and of
`validate_capture`'s Tier-2 reject register already live in
`tests/test_frame_capture_discipline.py` and `tests/test_ingress.py`
respectively).
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.payloads import CaptureSpec
from dgemma.types import DGemmaModel


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0
    vocab_size = 6

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


def _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, **factory_kwargs):
    """Same monkeypatch idiom as `tests/test_capture_top_k_e2e.py`'s helper."""
    built = fake_pipeline_factory(**factory_kwargs)
    scheduler_cls = type(built.scheduler)
    pipeline_cls = type(built.pipeline)
    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", scheduler_cls)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)
    return built


class TestCaptureFullDistributionEndToEnd:
    """ADR-CDG-014 Decision 3/5 Tier 2, issue #61 P-C: `run_diffusion(capture=
    CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=N))`
    populates the first N captured frames' `distribution` from the real
    per-step `logits`, budget-gated at ingress."""

    def test_full_distribution_populates_frames_within_budget_through_run_diffusion(
        self, monkeypatch, fake_pipeline_factory
    ):
        vocab = 6
        canvas_shape = (1, 4)
        built = _wire_fake_pipeline(
            monkeypatch,
            fake_pipeline_factory,
            num_inference_steps=3,
            vocab_size=vocab,
            canvas_shape=canvas_shape,
        )
        logits = torch.tensor(
            [[[0.0, 3.0, -1.0, 2.0, 0.5, -2.0]] * canvas_shape[1]], dtype=torch.float32
        )
        built.model.forward = lambda decoder_input_ids, **_ignored: type(
            "Out", (), {"logits": logits}
        )()

        _, _state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            gen_length=canvas_shape[1],
            keep_frames="all",
            capture=CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=2),
        )

        assert len(trace.frames) == 3
        expected = torch.softmax(logits[0], dim=-1)
        # Budget=2: first two captured frames retain distribution, the third
        # does not (ADR-CDG-014 Decision 5 — budget caps retention regardless
        # of keep_frames="all").
        for frame in trace.frames[:2]:
            assert frame.distribution is not None
            assert frame.distribution.shape == (canvas_shape[1], vocab)
            assert torch.allclose(frame.distribution, expected, atol=1e-5)
        assert trace.frames[2].distribution is None
        # Tier 0 rides alongside Tier 2 unaffected, on every frame including
        # the one past budget.
        for frame in trace.frames:
            assert frame.entropy is not None

    def test_capture_none_leaves_distribution_absent(self, monkeypatch, fake_pipeline_factory):
        """Regression floor: no `capture=` at all is byte-identical to
        before this phase — `distribution` stays `None` on every frame."""
        built = _wire_fake_pipeline(
            monkeypatch, fake_pipeline_factory, num_inference_steps=2, vocab_size=6, canvas_shape=(1, 4)
        )

        _, _state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            gen_length=4,
            keep_frames="all",
        )

        for frame in trace.frames:
            assert frame.distribution is None
            assert frame.entropy is not None

    def test_capture_full_distribution_false_leaves_distribution_absent(self, monkeypatch, fake_pipeline_factory):
        """`CaptureSpec(capture_full_distribution=False)` (the field's own
        default) is the same as no `capture=` at all for Tier 2's purposes —
        a budget given alongside `False` is inert, never enables capture."""
        built = _wire_fake_pipeline(
            monkeypatch, fake_pipeline_factory, num_inference_steps=2, vocab_size=6, canvas_shape=(1, 4)
        )

        _, _state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            gen_length=4,
            keep_frames="all",
            capture=CaptureSpec(capture_full_distribution=False, max_full_distribution_steps=4),
        )

        for frame in trace.frames:
            assert frame.distribution is None

    def test_unbounded_tier2_request_rejected_before_scheduler_construction(
        self, monkeypatch, fake_pipeline_factory
    ):
        """The load-bearing reject (ADR-CDG-014 Decision 3 Tier 2) is live at
        `run_diffusion`, matching every other payload's ingress contract: a
        `capture_full_distribution=True` with no budget raises before any
        scheduler is built — never a silent OOM path."""
        scheduler_kwargs: dict = {}

        built = fake_pipeline_factory(num_inference_steps=2, vocab_size=6, canvas_shape=(1, 4))
        scheduler_cls = type(built.scheduler)
        pipeline_cls = type(built.pipeline)

        def _scheduler_factory(**kwargs):
            scheduler_kwargs.update(kwargs)
            return scheduler_cls(**kwargs)

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)

        with pytest.raises(ValueError, match=r"capture_full_distribution=True requires max_full_distribution_steps"):
            run_diffusion(
                _fake_model_with(built.model),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                gen_length=4,
                capture=CaptureSpec(capture_full_distribution=True),
            )
        assert scheduler_kwargs == {}, "scheduler must not be constructed when ingress rejects"

    def test_zero_budget_rejected_before_scheduler_construction(self, monkeypatch, fake_pipeline_factory):
        scheduler_kwargs: dict = {}

        built = fake_pipeline_factory(num_inference_steps=2, vocab_size=6, canvas_shape=(1, 4))
        scheduler_cls = type(built.scheduler)
        pipeline_cls = type(built.pipeline)

        def _scheduler_factory(**kwargs):
            scheduler_kwargs.update(kwargs)
            return scheduler_cls(**kwargs)

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)

        with pytest.raises(ValueError, match=r"max_full_distribution_steps must be > 0, got 0"):
            run_diffusion(
                _fake_model_with(built.model),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                gen_length=4,
                capture=CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=0),
            )
        assert scheduler_kwargs == {}

    def test_budget_wider_than_run_retains_every_frame(self, monkeypatch, fake_pipeline_factory):
        """A budget >= the run's actual step count is a legitimate,
        non-degenerate request — every captured frame retains
        `distribution` (the budget simply never binds)."""
        vocab = 5
        # The fake pipeline's canvas length is hardcoded to 4 regardless of
        # `gen_length` (`tests/conftest.py:381`) — match it here so the
        # requested `gen_length` doesn't silently diverge from the actual
        # canvas shape the fixture produces.
        canvas_shape = (1, 4)
        built = _wire_fake_pipeline(
            monkeypatch, fake_pipeline_factory, num_inference_steps=2, vocab_size=vocab, canvas_shape=canvas_shape
        )

        _, _state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            gen_length=canvas_shape[1],
            keep_frames="all",
            capture=CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=100),
        )

        assert len(trace.frames) == 2
        for frame in trace.frames:
            assert frame.distribution is not None
            assert frame.distribution.shape == (canvas_shape[1], vocab)

    def test_tier1_and_tier2_compose_through_run_diffusion(self, monkeypatch, fake_pipeline_factory):
        """Both retention-tier knobs on the same `capture=` payload derive
        from the same pre-pin logits without interfering with each other."""
        vocab = 6
        canvas_shape = (1, 4)
        k = 3
        built = _wire_fake_pipeline(
            monkeypatch, fake_pipeline_factory, num_inference_steps=1, vocab_size=vocab, canvas_shape=canvas_shape
        )
        logits = torch.tensor(
            [[[0.0, 3.0, -1.0, 2.0, 0.5, -2.0]] * canvas_shape[1]], dtype=torch.float32
        )
        built.model.forward = lambda decoder_input_ids, **_ignored: type(
            "Out", (), {"logits": logits}
        )()

        _, _state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=1,
            gen_length=canvas_shape[1],
            keep_frames="all",
            capture=CaptureSpec(top_k=k, capture_full_distribution=True, max_full_distribution_steps=1),
        )

        frame = trace.frames[0]
        assert frame.entropy is not None
        expected_values, expected_ids = logits[0].topk(k, dim=-1)
        assert torch.equal(frame.top_k_ids, expected_ids)
        expected_dist = torch.softmax(logits[0], dim=-1)
        assert torch.allclose(frame.distribution, expected_dist, atol=1e-5)
