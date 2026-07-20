"""Enforcement tests for the `DiffusionFrame`/`CanvasTrace` capture discipline
(ADR-CDG-014, issue #61 Phases P-A/P-B/P-C) — the R6 additive-optional field
discipline, Tier 0's always-on per-position entropy capture, Tier 1's
on-request top-k capture, Tier 2's explicit-opt-in-with-budget full
distribution capture, and the capture-pre-pin ordering guarantee
(ADR-CDG-010).

Scope: additive-optional field discipline (Decision 1), Tier 0 entropy
(Decision 3's always-on row, P-A), Tier 1 top-k (Decision 3's on-request row,
P-B), Tier 2 full distribution + retention budget (Decision 3/5's
explicit-opt-in row, P-C), and capture-pre-pin ordering (Decision 4).

**Issue #64 Phase 2 addition:** `pinned_mask` (ADR-CDG-010 Decision 4) and
`effective_entropy_bound`/`effective_t_min`/`effective_t_max` (ADR-CDG-011
clause 7) join the same additive-optional discipline this module already
enforces — `TestAdditiveOptionalFieldDiscipline` is extended to cover all
four new fields, and two new classes (`TestPinnedMask`,
`TestEffectiveKnobTelemetry`) pin their derivation per the plan's §5 test
design (issue #64, gate-ratified 2026-07-13).

**Issue #61 P-B addition:** `TestTier1TopKCapture` pins the plan's test (c)
— `top_k=0` yields `top_k_ids is None`, `top_k=8` yields shape
`[canvas_len, 8]` — plus the capture-pre-pin ordering guarantee extended to
Tier 1 (top-k derives from the same pre-pin `logits` entropy does, never a
post-pin artifact).

**Issue #61 P-C addition:** `TestTier2FullDistributionCapture` pins the
plan's test (d) — an unbounded full-dist request raises at ingress (covered
in `tests/test_ingress.py`); here, `_FrameCollector`-level unit coverage of
the budget mechanics: off by default, on-with-budget populates
`distribution` for exactly the first N captured steps, absent thereafter,
independent of Tier 0/1, and (extended) capture-pre-pin ordering for Tier 2.
"""
from __future__ import annotations

import math

import pytest
import torch

from dgemma.composite import StepEndComposite
from dgemma.loop import _FrameCollector
from dgemma.payloads import Constraints, Pin
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
        # Issue #64 Phase 2 additions — same additive-optional discipline.
        assert frame.pinned_mask is None
        assert frame.effective_entropy_bound is None
        assert frame.effective_t_min is None
        assert frame.effective_t_max is None

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
        for name in (
            "entropy", "top_k_ids", "top_k_weights", "distribution",
            "pinned_mask", "effective_entropy_bound", "effective_t_min", "effective_t_max",
        ):
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


class TestTier1TopKCapture:
    """ADR-CDG-014 Decision 3's Tier 1 row (issue #61 P-B): per-position
    top-k candidate ids + weights, on request only (`top_k` knob, default
    0/off). Plan's test (c): `top_k=0` -> `None`, `top_k=8` ->
    `[canvas_len, 8]`."""

    def test_top_k_zero_leaves_fields_none(self):
        """Default off — byte-identical to every pre-P-B run: no `top_k=`
        threaded through means Tier 1 stays absent even though `logits` is
        present (Tier 0 still fires)."""
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)
        vocab, canvas_len = 6, 4
        logits = torch.randn(1, canvas_len, vocab)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        frame = collector.frames[0]
        assert frame.top_k_ids is None
        assert frame.top_k_weights is None
        assert frame.entropy is not None  # Tier 0 unaffected by Tier 1 being off

    def test_top_k_requested_yields_correct_shape(self):
        vocab, canvas_len, k = 10, 4, 8
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, top_k=k)
        logits = torch.randn(1, canvas_len, vocab)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        frame = collector.frames[0]
        assert frame.top_k_ids is not None
        assert frame.top_k_weights is not None
        assert frame.top_k_ids.shape == (canvas_len, k)
        assert frame.top_k_weights.shape == (canvas_len, k)

    def test_top_k_absent_when_logits_not_reachable(self):
        """Additive-optional/absence discipline: requesting top_k without
        `logits` in callback_kwargs (e.g. a pre-R6-shaped call site) must not
        crash and must not fabricate a zero/empty top-k — both fields stay
        `None`, same absence semantics `entropy` already gets."""
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, top_k=8)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        assert collector.frames[0].top_k_ids is None
        assert collector.frames[0].top_k_weights is None

    def test_top_k_ids_match_manual_topk_over_same_logits(self):
        """Independently hand-verify the top-k ids against `torch.topk`
        directly over the exact logits handed in — not just a shape check."""
        vocab, canvas_len, k = 12, 3, 5
        logits = torch.randn(1, canvas_len, vocab)
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, top_k=k)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        expected_values, expected_ids = logits[0].topk(k, dim=-1)
        assert torch.equal(collector.frames[0].top_k_ids, expected_ids)
        expected_weights = torch.softmax(expected_values, dim=-1)
        assert torch.allclose(collector.frames[0].top_k_weights, expected_weights, atol=1e-6)

    def test_top_k_weights_sum_to_one_per_position(self):
        """The Tier-1 weights are a renormalization over just the k selected
        candidates (a per-position softmax restricted to that slice, not an
        approximation of the full-vocab distribution) — each position's row
        must sum to 1."""
        vocab, canvas_len, k = 20, 5, 6
        logits = torch.randn(1, canvas_len, vocab)
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, top_k=k)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        sums = collector.frames[0].top_k_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(canvas_len), atol=1e-5)

    def test_2d_logits_without_batch_dim_supported_for_top_k(self):
        """Mirrors entropy's own no-batch-dim support test — top_k must not
        assume a leading batch dim is always present either."""
        vocab, canvas_len, k = 7, 3, 4
        logits = torch.randn(canvas_len, vocab)  # no leading batch dim
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, top_k=k)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        assert collector.frames[0].top_k_ids.shape == (canvas_len, k)
        assert collector.frames[0].top_k_weights.shape == (canvas_len, k)


class TestTier2FullDistributionCapture:
    """ADR-CDG-014 Decision 3/5's Tier 2 row (issue #61 P-C): full
    per-position distribution, explicit opt-in WITH A BUDGET. Off by
    default; on-with-budget populates `distribution` for exactly the first
    `max_full_distribution_steps` captured steps (in step order), absent
    thereafter regardless of `keep_frames`; Tier 0/1 fields are unaffected
    by Tier 2's budget running out."""

    def test_distribution_none_by_default(self):
        """`capture_full_distribution=False` (the field's own default)
        leaves `distribution` `None` even though `logits` is present (Tier
        0 still fires) — byte-identical to every pre-P-C run."""
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)
        vocab, canvas_len = 6, 4
        logits = torch.randn(1, canvas_len, vocab)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        frame = collector.frames[0]
        assert frame.distribution is None
        assert frame.entropy is not None  # Tier 0 unaffected by Tier 2 being off

    def test_distribution_absent_when_logits_not_reachable(self):
        """Additive-optional/absence discipline: requesting Tier 2 without
        `logits` in callback_kwargs must not crash and must not fabricate a
        zero/empty distribution — stays `None`, same absence semantics
        entropy/top-k already get."""
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8,
            capture_full_distribution=True, max_full_distribution_steps=5,
        )
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        assert collector.frames[0].distribution is None

    def test_distribution_populated_within_budget(self):
        vocab, canvas_len = 10, 4
        collector = _FrameCollector(
            scheduler=_fake_scheduler(3), t_min=0.4, t_max=0.8, keep_frames="all",
            capture_full_distribution=True, max_full_distribution_steps=3,
        )
        logits = torch.randn(1, canvas_len, vocab)

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        frame = collector.frames[0]
        assert frame.distribution is not None
        assert frame.distribution.shape == (canvas_len, vocab)

    def test_distribution_matches_manual_softmax_over_same_logits(self):
        vocab, canvas_len = 7, 3
        logits = torch.randn(1, canvas_len, vocab)
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8,
            capture_full_distribution=True, max_full_distribution_steps=1,
        )

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        expected = torch.softmax(logits[0], dim=-1)
        assert torch.allclose(collector.frames[0].distribution, expected, atol=1e-6)

    def test_distribution_absent_past_the_budget(self):
        """The budget caps RETAINED frames — steps beyond
        `max_full_distribution_steps` carry `distribution=None`, regardless
        of `keep_frames="all"` retaining the frame itself (Decision 5)."""
        vocab, canvas_len, budget = 8, 3, 2
        collector = _FrameCollector(
            scheduler=_fake_scheduler(4), t_min=0.4, t_max=0.8, keep_frames="all",
            capture_full_distribution=True, max_full_distribution_steps=budget,
        )
        for step_idx in range(4):
            logits = torch.randn(1, canvas_len, vocab)
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * canvas_len], logits=logits))

        assert len(collector.frames) == 4
        for frame in collector.frames[:budget]:
            assert frame.distribution is not None
        for frame in collector.frames[budget:]:
            assert frame.distribution is None

    def test_tier0_and_tier1_unaffected_by_tier2_budget_exhaustion(self):
        """Tier 0 (entropy)/Tier 1 (top-k) keep their own independent
        policies — Tier 2's budget running out must not suppress them."""
        vocab, canvas_len, k, budget = 6, 3, 4, 1
        collector = _FrameCollector(
            scheduler=_fake_scheduler(2), t_min=0.4, t_max=0.8, keep_frames="all",
            top_k=k, capture_full_distribution=True, max_full_distribution_steps=budget,
        )
        for step_idx in range(2):
            logits = torch.randn(1, canvas_len, vocab)
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * canvas_len], logits=logits))

        first, second = collector.frames
        assert first.distribution is not None
        assert second.distribution is None
        # Tier 0/1 unaffected on BOTH frames, including the one past budget.
        for frame in (first, second):
            assert frame.entropy is not None
            assert frame.top_k_ids is not None
            assert frame.top_k_weights is not None

    def test_distribution_rows_sum_to_one(self):
        """The full distribution is a real per-position probability
        distribution, not a raw softmax-shaped tensor with drift — each
        position's row sums to 1."""
        vocab, canvas_len = 9, 5
        logits = torch.randn(1, canvas_len, vocab)
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8,
            capture_full_distribution=True, max_full_distribution_steps=1,
        )

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        sums = collector.frames[0].distribution.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(canvas_len), atol=1e-5)

    def test_2d_logits_without_batch_dim_supported_for_distribution(self):
        """Mirrors entropy/top-k's own no-batch-dim support test."""
        vocab, canvas_len = 5, 3
        logits = torch.randn(canvas_len, vocab)  # no leading batch dim
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8,
            capture_full_distribution=True, max_full_distribution_steps=1,
        )

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True] * canvas_len], logits=logits))

        assert collector.frames[0].distribution.shape == (canvas_len, vocab)

    def test_budget_none_with_flag_true_means_unbounded_in_direct_use(self):
        """Ingress always pairs `capture_full_distribution=True` with a
        budget through `run_diffusion`, but a caller driving the collector
        directly (bypassing ingress) with `max_full_distribution_steps=None`
        gets the documented degradation: no cap, every step populated —
        never a crash."""
        vocab, canvas_len = 6, 3
        collector = _FrameCollector(
            scheduler=_fake_scheduler(3), t_min=0.4, t_max=0.8, keep_frames="all",
            capture_full_distribution=True, max_full_distribution_steps=None,
        )
        for step_idx in range(3):
            logits = torch.randn(1, canvas_len, vocab)
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * canvas_len], logits=logits))

        for frame in collector.frames:
            assert frame.distribution is not None


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

    def test_top_k_reflects_model_logits_not_pin_rewrite(self, fake_pipeline_factory):
        """Same ordering proof as entropy above, extended to Tier 1 (issue
        #61 P-B): top-k derives from the pre-pin `logits`, never from a
        canvas a later pin participant rewrote."""
        vocab = 6
        canvas_shape = (1, 4)
        logits = torch.tensor(
            [[[0.0, 1.0, 2.0, -1.0, 0.5, 3.0]] * canvas_shape[1]], dtype=torch.float32
        )
        k = 3

        built = fake_pipeline_factory(
            num_inference_steps=1, vocab_size=vocab, canvas_shape=canvas_shape,
        )
        built.model.forward = lambda decoder_input_ids, **_ignored: type(
            "Out", (), {"logits": logits}
        )()

        collector = _FrameCollector(
            scheduler=built.scheduler, t_min=0.4, t_max=0.8, keep_frames="all", top_k=k
        )

        def pin(pipe, global_step, step_idx, callback_kwargs):
            return {"canvas": torch.full_like(callback_kwargs["canvas"], 999)}

        step_end = StepEndComposite(capture=collector.on_step_end, pin=(pin,))

        result = built.pipeline(
            num_inference_steps=1,
            callback_on_step_end=step_end,
            callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
        )

        assert torch.all(result.sequences == 999)
        assert not torch.all(collector.frames[0].canvas == 999)

        expected_values, expected_ids = logits[0].topk(k, dim=-1)
        assert torch.equal(collector.frames[0].top_k_ids, expected_ids)
        expected_weights = torch.softmax(expected_values, dim=-1)
        assert torch.allclose(collector.frames[0].top_k_weights, expected_weights, atol=1e-5)

    def test_distribution_reflects_model_logits_not_pin_rewrite(self, fake_pipeline_factory):
        """Same ordering proof as entropy/top-k above, extended to Tier 2
        (issue #61 P-C): the full distribution derives from the pre-pin
        `logits`, never from a canvas a later pin participant rewrote."""
        vocab = 6
        canvas_shape = (1, 4)
        logits = torch.tensor(
            [[[0.0, 1.0, 2.0, -1.0, 0.5, 3.0]] * canvas_shape[1]], dtype=torch.float32
        )

        built = fake_pipeline_factory(
            num_inference_steps=1, vocab_size=vocab, canvas_shape=canvas_shape,
        )
        built.model.forward = lambda decoder_input_ids, **_ignored: type(
            "Out", (), {"logits": logits}
        )()

        collector = _FrameCollector(
            scheduler=built.scheduler, t_min=0.4, t_max=0.8, keep_frames="all",
            capture_full_distribution=True, max_full_distribution_steps=1,
        )

        def pin(pipe, global_step, step_idx, callback_kwargs):
            return {"canvas": torch.full_like(callback_kwargs["canvas"], 999)}

        step_end = StepEndComposite(capture=collector.on_step_end, pin=(pin,))

        result = built.pipeline(
            num_inference_steps=1,
            callback_on_step_end=step_end,
            callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
        )

        assert torch.all(result.sequences == 999)
        assert not torch.all(collector.frames[0].canvas == 999)

        expected = torch.softmax(logits[0], dim=-1)
        assert torch.allclose(collector.frames[0].distribution, expected, atol=1e-5)


def _fake_scheduler_with_config(
    num_inference_steps: int = 4, entropy_bound: float = 0.1, t_min: float = 0.4, t_max: float = 0.8
):
    """A `.config`-bearing fake scheduler (mirrors `tests/conftest.py`'s
    `FakeFrozenConfig`/`FakeEntropyBoundScheduler` shape at the exact surface
    `_FrameCollector`'s effective-knob telemetry reads: `.config.
    entropy_bound`/`.t_min`/`.t_max`, mutable only via `register_to_config`).
    """
    from dataclasses import dataclass, field

    @dataclass
    class FakeConfig:
        entropy_bound: float
        t_min: float
        t_max: float

    @dataclass
    class FakeScheduler:
        num_inference_steps: int
        _config: FakeConfig = field(default=None)

        def __post_init__(self):
            if self._config is None:
                self._config = FakeConfig(entropy_bound=entropy_bound, t_min=t_min, t_max=t_max)

        @property
        def config(self):
            return self._config

        def register_to_config(self, **kwargs):
            merged = {
                "entropy_bound": self._config.entropy_bound,
                "t_min": self._config.t_min,
                "t_max": self._config.t_max,
            }
            merged.update(kwargs)
            self._config = FakeConfig(**merged)

    return FakeScheduler(num_inference_steps)


class TestPinnedMask:
    """ADR-CDG-010 Decision 4, issue #64 Phase 2 (gate correction A1):
    `pinned_mask` is derived from a supplied `Constraints` payload's pin
    positions — `True` at every pinned position, `None` when no constraints
    were given. No pin participant exists yet (Phase 3): this is the
    validated-then-ignored payload's positions read directly, not an
    observed per-step write, and the mask is constant across every frame in
    the run (the D6 hard-pin, position-static invariant this phase's
    computation is scoped to — see `DiffusionFrame.pinned_mask`'s docstring
    for the scope guard)."""

    def test_pinned_mask_none_when_no_constraints_supplied(self):
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        assert collector.frames[0].pinned_mask is None

    def test_pinned_mask_none_when_constraints_has_no_pins(self):
        """`Constraints()`'s default empty-tuple pins is a no-op, matching
        `None` (ADR-CDG-010's `Constraints` docstring) — the mask stays
        absent, never an all-`False` tensor standing in for "no pins"."""
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, constraints=Constraints(pins=())
        )
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        assert collector.frames[0].pinned_mask is None

    def test_pinned_mask_true_at_every_pinned_position(self):
        constraints = Constraints(pins=(Pin(position=0, token_id=5), Pin(position=2, token_id=9)))
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, constraints=constraints
        )
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True, True, True]]))

        mask = collector.frames[0].pinned_mask
        assert mask is not None
        assert mask.tolist() == [True, False, True, False]

    def test_pinned_mask_true_regardless_of_scheduler_commit_reading(self):
        """D4 trace-honesty test: a pinned cell's `pinned_mask` is `True`
        even on a step where the scheduler's own `accepted_index` says that
        position was NOT committed — `pinned_mask` reports the constraint
        layer's claim, independent of the model's commit reading."""
        constraints = Constraints(pins=(Pin(position=1, token_id=3),))
        collector = _FrameCollector(
            scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8, constraints=constraints
        )
        # Position 1 (the pinned one) reads as NOT accepted by the scheduler.
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, False, True]]))

        assert collector.frames[0].pinned_mask.tolist() == [False, True, False]

    def test_pinned_mask_constant_across_every_frame_in_the_run(self):
        """The D6 position-static invariant this phase's computation relies
        on: the same mask rides every frame of one run, not just the first
        callback."""
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))
        collector = _FrameCollector(
            scheduler=_fake_scheduler(3), t_min=0.4, t_max=0.8, keep_frames="all", constraints=constraints
        )
        for step_idx in range(3):
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True, True]]))

        for frame in collector.frames:
            assert frame.pinned_mask.tolist() == [True, False]


class TestEffectiveKnobTelemetry:
    """ADR-CDG-011 clause 7, issue #64 Phase 2: `DiffusionFrame`'s
    `effective_entropy_bound`/`effective_t_min`/`effective_t_max` reflect
    `scheduler.config`'s value AT THAT CALLBACK — never a static ctor
    snapshot or a binding's declared curve. No walker exists yet (Phase 4);
    this pins the read-path honesty a future walker's writes will surface
    through, proven here via a direct `register_to_config` mutation (the
    same mechanism a walker uses, ADR-CDG-011 Decision 4)."""

    def test_effective_fields_reflect_ctor_config_with_no_mutation(self):
        scheduler = _fake_scheduler_with_config(entropy_bound=0.1, t_min=0.4, t_max=0.8)
        collector = _FrameCollector(scheduler=scheduler, t_min=0.4, t_max=0.8)
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        frame = collector.frames[0]
        assert frame.effective_entropy_bound == pytest.approx(0.1)
        assert frame.effective_t_min == pytest.approx(0.4)
        assert frame.effective_t_max == pytest.approx(0.8)

    def test_effective_fields_reflect_mid_run_config_mutation(self):
        """A mid-run `register_to_config` mutation (the exact write
        mechanism a future walker performs, ADR-CDG-011 Decision 4) must
        show up in the NEXT captured frame's `effective_*` fields — the
        telemetry-honesty enforcement the plan's `TestEffectiveKnobTelemetry`
        names: a walker bug that silently fails to write through would be
        invisible if this read path fell back to the binding's static
        curve instead of the scheduler's actually-read value."""
        scheduler = _fake_scheduler_with_config(
            num_inference_steps=3, entropy_bound=0.1, t_min=0.4, t_max=0.8
        )
        collector = _FrameCollector(scheduler=scheduler, t_min=0.4, t_max=0.8, keep_frames="all")

        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))
        scheduler.register_to_config(entropy_bound=0.05, t_min=0.2, t_max=0.6)
        collector.on_step_end(None, 1, 1, _callback_kwargs([[True, True]]))

        first, second = collector.frames
        assert first.effective_entropy_bound == pytest.approx(0.1)
        assert first.effective_t_min == pytest.approx(0.4)
        assert first.effective_t_max == pytest.approx(0.8)

        assert second.effective_entropy_bound == pytest.approx(0.05)
        assert second.effective_t_min == pytest.approx(0.2)
        assert second.effective_t_max == pytest.approx(0.6)

    def test_t_and_temperature_reflect_live_t_min_t_max_after_mutation(self):
        """`t`/`temperature` (the pre-existing anneal fields) must be
        consistent with the mutated `effective_t_min`/`effective_t_max`, not
        the original ctor values — a walker-mutated anneal range is one
        honest reading, not two disagreeing ones."""
        scheduler = _fake_scheduler_with_config(
            num_inference_steps=2, entropy_bound=0.1, t_min=0.4, t_max=0.8
        )
        collector = _FrameCollector(scheduler=scheduler, t_min=0.4, t_max=0.8, keep_frames="all")

        scheduler.register_to_config(t_min=0.1, t_max=0.3)
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        frame = collector.frames[0]
        from dgemma.loop import anneal_temperature

        expected_t, expected_temperature = anneal_temperature(
            step_idx=0, num_inference_steps=2, t_min=0.1, t_max=0.3
        )
        assert frame.t == pytest.approx(expected_t)
        assert frame.temperature == pytest.approx(expected_temperature)
        assert frame.effective_t_min == pytest.approx(0.1)
        assert frame.effective_t_max == pytest.approx(0.3)

    def test_effective_entropy_bound_none_when_scheduler_has_no_config(self):
        """Named degradation (mirrors `resolve_vocab_size`'s stub fallback):
        a bare pre-R4-style test double exposing only `.num_inference_steps`
        (no `.config` at all) must not crash — `effective_entropy_bound`
        reads `None` (no ctor fallback exists for it), while `effective_t_min`/
        `effective_t_max` fall back to the ctor `t_min`/`t_max` this
        collector was constructed with."""
        collector = _FrameCollector(scheduler=_fake_scheduler(1), t_min=0.4, t_max=0.8)
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True]]))

        frame = collector.frames[0]
        assert frame.effective_entropy_bound is None
        assert frame.effective_t_min == pytest.approx(0.4)
        assert frame.effective_t_max == pytest.approx(0.8)
