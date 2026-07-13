"""Enforcement tests for the `DiffusionFrame`/`CanvasTrace` capture discipline
(ADR-CDG-014, issue #61 Phase P-A) — the R6 additive-optional field
discipline, Tier 0's always-on per-position entropy capture, and the
capture-pre-pin ordering guarantee (ADR-CDG-010).

Scope (P-A only): additive-optional field discipline (Decision 1), Tier 0
entropy (Decision 3's always-on row), and capture-pre-pin ordering
(Decision 4). Tier 1 (`top_k`)/Tier 2 (`distribution` + budget) knobs are
NOT implemented here (P-B/P-C) — this suite only asserts their FIELDS exist
and default to `None` under the same additive-optional discipline, not their
derivation.
"""
from __future__ import annotations

import math

import pytest
import torch

from dgemma.composite import StepEndComposite
from dgemma.loop import _FrameCollector
from dgemma.types import CanvasTrace, DiffusionFrame


class TestAdditiveOptionalFieldDiscipline:
    """ADR-CDG-014 Decision 1 (R6): every new field is optional with a
    default; the pre-existing positional fields never move and never gain a
    required sibling. Enforcement: construct with ONLY the pre-R6 positional
    args and assert success — a required new field breaks this by name."""

    def test_diffusion_frame_constructs_with_only_pre_r6_positional_args(self):
        frame = DiffusionFrame(
            canvas_idx=0,
            step_idx=0,
            t=1.0,
            temperature=0.8,
            committed_fraction_per_example=(1.0,),
            canvas=torch.tensor([1, 2, 3]),
        )
        assert frame.entropy is None
        assert frame.top_k_ids is None
        assert frame.top_k_weights is None
        assert frame.distribution is None

    def test_canvas_trace_constructs_with_only_pre_r6_positional_args(self):
        frame = DiffusionFrame(
            canvas_idx=0, step_idx=0, t=1.0, temperature=0.8,
            committed_fraction_per_example=(1.0,), canvas=torch.tensor([1]),
        )
        trace = CanvasTrace(frames=[frame], scheduler_name="EntropyBoundScheduler", scheduler_config={})
        assert trace.raw_canvas_ids is None

    def test_new_fields_are_keyword_only_additions_not_positional(self):
        """A required new field would break the pre-R6 positional-only
        construction above by name (TypeError: missing argument) — this
        test asserts the reverse-engineering safeguard directly: every R6
        field has a declared default, so it can never be a required
        positional slot regardless of call-site ordering."""
        import dataclasses

        frame_fields = {f.name: f for f in dataclasses.fields(DiffusionFrame)}
        for name in ("entropy", "top_k_ids", "top_k_weights", "distribution"):
            assert frame_fields[name].default is None

        trace_fields = {f.name: f for f in dataclasses.fields(CanvasTrace)}
        assert trace_fields["raw_canvas_ids"].default is None


class TestDefaultSemanticsAbsenceNotEmpty:
    """ADR-CDG-014 Decision 2: `None` means "not captured," never "captured
    empty." A consumer must be able to distinguish a `None` entropy field
    (tier off) from a real all-zero entropy vector (a legitimate degenerate
    capture) — this test pins that the two are represented differently."""

    def test_none_entropy_is_distinct_from_zero_entropy_tensor(self):
        no_capture = DiffusionFrame(
            canvas_idx=0, step_idx=0, t=1.0, temperature=0.8,
            committed_fraction_per_example=(1.0,), canvas=torch.tensor([1]),
        )
        zero_capture = DiffusionFrame(
            canvas_idx=0, step_idx=0, t=1.0, temperature=0.8,
            committed_fraction_per_example=(1.0,), canvas=torch.tensor([1]),
            entropy=torch.zeros(4),
        )
        assert no_capture.entropy is None
        assert zero_capture.entropy is not None
        assert torch.all(zero_capture.entropy == 0.0)
        # The guard a consumer must apply: never coerce None into a
        # zero-valued reading.
        assert not (no_capture.entropy is None and torch.equal(zero_capture.entropy, torch.zeros(4))
                    and no_capture.entropy == zero_capture.entropy)  # would raise/lie if entropy were 0 not None


def _fake_scheduler(num_inference_steps: int = 4):
    from dataclasses import dataclass

    @dataclass
    class FakeScheduler:
        num_inference_steps: int

    return FakeScheduler(num_inference_steps)


def _callback_kwargs(accepted: list[list[bool]], canvas_value: int = 0, logits: torch.Tensor | None = None) -> dict:
    from dataclasses import dataclass

    @dataclass
    class FakeSchedulerOutput:
        accepted_index: torch.Tensor

    accepted_tensor = torch.tensor(accepted, dtype=torch.bool)
    kwargs = {
        "scheduler_output": FakeSchedulerOutput(accepted_index=accepted_tensor),
        "canvas": torch.full(accepted_tensor.shape, canvas_value, dtype=torch.long),
    }
    if logits is not None:
        kwargs["logits"] = logits
    return kwargs


class TestTier0AlwaysOnEntropyCapture:
    """ADR-CDG-014 Decision 3's Tier 0 row: per-position entropy
    `float32[canvas_len]`, always captured when `logits` is reachable."""

    def test_every_frame_carries_entropy_when_logits_present(self):
        collector = _FrameCollector(scheduler=_fake_scheduler(3), t_min=0.4, t_max=0.8, keep_frames="all")
        vocab = 6
        canvas_len = 4
        for step_idx in range(3):
            logits = torch.randn(1, canvas_len, vocab)
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * canvas_len], logits=logits))

        assert len(collector.frames) == 3
        for frame in collector.frames:
            assert frame.entropy is not None
            assert frame.entropy.shape == (canvas_len,)

    def test_entropy_absent_when_logits_not_requested(self):
        """Additive-optional/absence discipline (Decision 1/2): a caller
        driving the collector directly without `logits` in callback_kwargs
        (e.g. a pre-R6-shaped call site) gets `entropy=None`, not a crash
        and not a fabricated zero vector."""
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        assert collector.frames[0].entropy is None

    def test_entropy_math_uniform_logits_gives_max_entropy(self):
        """Hand-constructed logits, no real weights needed: a uniform
        (all-equal) logit row over vocab V has entropy exactly ln(V) at
        every position — the maximum-entropy case."""
        vocab = 8
        canvas_len = 3
        logits = torch.zeros(1, canvas_len, vocab)  # uniform over vocab
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        entropy = collector.frames[0].entropy
        assert entropy.shape == (canvas_len,)
        expected = math.log(vocab)
        for value in entropy.tolist():
            assert value == pytest.approx(expected, abs=1e-5)

    def test_entropy_math_one_hot_logits_gives_near_zero_entropy(self):
        """A one-hot (single overwhelmingly large logit) row is
        near-deterministic — entropy approaches 0."""
        vocab = 8
        canvas_len = 2
        logits = torch.full((1, canvas_len, vocab), -1e9)
        logits[:, :, 0] = 1e9  # position argmax overwhelmingly certain
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        entropy = collector.frames[0].entropy
        for value in entropy.tolist():
            assert value == pytest.approx(0.0, abs=1e-4)

    def test_entropy_math_matches_hand_computed_categorical_entropy(self):
        """A non-degenerate, hand-picked logit row: verify the collector's
        entropy reading matches an independently hand-computed Shannon
        entropy over the softmax of those exact logits, not just the two
        extreme cases above."""
        raw_logits = [0.0, 1.0, 2.0, -1.0]
        logits = torch.tensor([raw_logits], dtype=torch.float32).unsqueeze(0)  # [1, 1, vocab]
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True]], logits=logits))

        probs = torch.softmax(torch.tensor(raw_logits), dim=-1)
        expected_entropy = -(probs * probs.log()).sum().item()
        assert collector.frames[0].entropy[0].item() == pytest.approx(expected_entropy, abs=1e-5)

    def test_2d_logits_without_batch_dim_supported(self):
        """Some fixtures/call sites may hand the collector already-squeezed
        `[canvas_len, vocab]` logits (no batch dim) — the collector must not
        assume a batch dim is always present."""
        vocab = 5
        canvas_len = 3
        logits = torch.randn(canvas_len, vocab)  # no leading batch dim
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        assert collector.frames[0].entropy.shape == (canvas_len,)


class TestCapturePrePinOrdering:
    """ADR-CDG-014 Decision 4 (ADR-CDG-010 ordering): capture is the
    composite's FIRST participant, so entropy derives from the step's
    pre-pin `logits` — never from anything a later pin/beta-rebuild
    participant does to the canvas. Proven by scripting a pin participant
    that rewrites the canvas and asserting the captured entropy still
    matches the model's logits, not the pinned canvas."""

    def test_entropy_reflects_model_logits_not_pin_rewrite(self, fake_pipeline_factory):
        vocab = 6
        canvas_shape = (1, 4)
        # Distinct, non-uniform logits per position so entropy is
        # meaningfully different from the degenerate all-equal case.
        logits = torch.tensor(
            [[[0.0, 1.0, 2.0, -1.0, 0.5, 3.0]] * canvas_shape[1]], dtype=torch.float32
        )

        built = fake_pipeline_factory(
            num_inference_steps=1, vocab_size=vocab, canvas_shape=canvas_shape,
        )
        # Force the model's forward to return our hand-picked logits instead
        # of the fixture's default zeros, so we have a known ground truth to
        # assert entropy against.
        built.model.forward = lambda decoder_input_ids, **_ignored: type(
            "Out", (), {"logits": logits}
        )()

        collector = _FrameCollector(scheduler=built.scheduler, t_min=0.4, t_max=0.8, keep_frames="all")

        def pin(pipe, global_step, step_idx, callback_kwargs):
            # Canvas-writer that overwrites every position with a fixed
            # value — if capture ran AFTER this (or read this output), the
            # captured `canvas` field would show 999 rather than the
            # scheduler's own committed value. Entropy must be unaffected
            # either way, since it derives from `logits`, not `canvas`.
            return {"canvas": torch.full_like(callback_kwargs["canvas"], 999)}

        step_end = StepEndComposite(capture=collector.on_step_end, pin=(pin,))

        result = built.pipeline(
            num_inference_steps=1,
            callback_on_step_end=step_end,
            callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
        )

        # Pin's rewrite did reach the pipeline's output (proves the pin
        # participant actually ran and would have contaminated a
        # post-pin capture).
        assert torch.all(result.sequences == 999)

        # But the captured frame's canvas is the PRE-pin committed value
        # (capture ran first) ...
        assert not torch.all(collector.frames[0].canvas == 999)

        # ... and its entropy matches the known logits exactly, independent
        # of the pin rewrite.
        expected_entropy = torch.distributions.Categorical(logits=logits[0]).entropy()
        assert torch.allclose(collector.frames[0].entropy, expected_entropy, atol=1e-5)
