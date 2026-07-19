"""End-to-end `capture=` Tier-1 top-k tests (ADR-CDG-014 Decision 3, issue
#61 P-B) — driven through `run_diffusion` against the R4 fake-pipeline
fixture (`tests/conftest.py`), the way `tests/test_constraints.py` drives
`constraints=` end to end, rather than the lighter hand-rolled fakes
`tests/test_run_diffusion_ingress.py` uses for ingress-only assertions (those
never emit `logits` in `callback_kwargs`, so they cannot exercise Tier 1's
actual derivation — see that module's `TestValidPayloadsAreIgnoredBehaviorally`
docstring for the same gap noted against `constraints=`).

This module proves the `run_diffusion` -> `CaptureSpec.top_k` -> `_FrameCollector`
wiring works at the real call boundary (unit coverage of `_FrameCollector`
itself already lives in `tests/test_frame_capture_discipline.py`).
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
    """Same monkeypatch idiom as `tests/test_constraints.py`'s
    `_wire_fake_pipeline` (import-mode caveat: never `from tests.conftest
    import ...` by name — reach the real classes only through the fixture)."""
    built = fake_pipeline_factory(**factory_kwargs)
    scheduler_cls = type(built.scheduler)
    pipeline_cls = type(built.pipeline)
    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", scheduler_cls)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)
    return built


class TestCaptureTopKEndToEnd:
    """ADR-CDG-014 Decision 3 Tier 1, issue #61 P-B: `run_diffusion(capture=
    CaptureSpec(top_k=k))` populates every captured frame's `top_k_ids`/
    `top_k_weights` from the real per-step `logits`."""

    def test_top_k_populates_every_frame_through_run_diffusion(self, monkeypatch, fake_pipeline_factory):
        vocab = 6
        canvas_shape = (1, 4)
        k = 3
        built = _wire_fake_pipeline(
            monkeypatch,
            fake_pipeline_factory,
            num_inference_steps=2,
            vocab_size=vocab,
            canvas_shape=canvas_shape,
        )
        # Distinct non-uniform logits so top-k selection is meaningful, not
        # the zeros-everywhere default (which would tie every candidate).
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
            num_inference_steps=2,
            gen_length=canvas_shape[1],
            keep_frames="all",
            capture=CaptureSpec(top_k=k),
        )

        assert len(trace.frames) == 2
        expected_values, expected_ids = logits[0].topk(k, dim=-1)
        expected_weights = torch.softmax(expected_values, dim=-1)
        for frame in trace.frames:
            assert frame.top_k_ids is not None
            assert frame.top_k_weights is not None
            assert frame.top_k_ids.shape == (canvas_shape[1], k)
            assert torch.equal(frame.top_k_ids, expected_ids)
            assert torch.allclose(frame.top_k_weights, expected_weights, atol=1e-5)
            # Tier 0 rides alongside Tier 1 unaffected (both derive from the
            # same pre-pin logits, ADR-CDG-014 Decision 4).
            assert frame.entropy is not None

    def test_capture_none_leaves_top_k_fields_absent(self, monkeypatch, fake_pipeline_factory):
        """Regression floor: no `capture=` at all is byte-identical to
        before this phase — Tier 1 fields stay `None` on every frame."""
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
            assert frame.top_k_ids is None
            assert frame.top_k_weights is None
            assert frame.entropy is not None

    def test_capture_top_k_zero_leaves_fields_absent(self, monkeypatch, fake_pipeline_factory):
        """`CaptureSpec(top_k=0)` (the field's own default) is the same
        as no `capture=` at all for Tier 1's purposes."""
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
            capture=CaptureSpec(top_k=0),
        )

        for frame in trace.frames:
            assert frame.top_k_ids is None
            assert frame.top_k_weights is None

    def test_invalid_top_k_raises_before_scheduler_construction(self, monkeypatch, fake_pipeline_factory):
        """The reject half is live at `run_diffusion`, matching every other
        payload's ingress contract (`tests/test_run_diffusion_ingress.py`'s
        `TestRejectPathsAreLiveAtRunDiffusion`) — a negative `top_k` raises
        before any scheduler is built."""
        scheduler_kwargs: dict = {}

        built = fake_pipeline_factory(num_inference_steps=2, vocab_size=6, canvas_shape=(1, 4))
        scheduler_cls = type(built.scheduler)
        pipeline_cls = type(built.pipeline)

        def _scheduler_factory(**kwargs):
            scheduler_kwargs.update(kwargs)
            return scheduler_cls(**kwargs)

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)

        with pytest.raises(ValueError, match=r"top_k must be >= 0"):
            run_diffusion(
                _fake_model_with(built.model),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                gen_length=4,
                capture=CaptureSpec(top_k=-1),
            )
        assert scheduler_kwargs == {}, "scheduler must not be constructed when ingress rejects"

    def test_top_k_exceeding_vocab_size_raises_before_scheduler_construction(
        self, monkeypatch, fake_pipeline_factory
    ):
        """The vocab-ceiling reject (ADR-CDG-014 rule 5) is also live at
        `run_diffusion` when a real tokenizer-derived `vocab_size` is
        resolvable — `FakeTokenizer.vocab_size = 6` here."""
        scheduler_kwargs: dict = {}

        built = fake_pipeline_factory(num_inference_steps=2, vocab_size=6, canvas_shape=(1, 4))
        scheduler_cls = type(built.scheduler)
        pipeline_cls = type(built.pipeline)

        def _scheduler_factory(**kwargs):
            scheduler_kwargs.update(kwargs)
            return scheduler_cls(**kwargs)

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)

        with pytest.raises(ValueError, match=r"top_k=999 exceeds vocab_size=6"):
            run_diffusion(
                _fake_model_with(built.model),
                "hi",
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                gen_length=4,
                capture=CaptureSpec(top_k=999),
            )
        assert scheduler_kwargs == {}
