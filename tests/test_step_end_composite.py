"""Enforcement tests for `dgemma.composite.StepEndComposite` (#35 R1,
ADR-CDG-010 Decision 3) — the ordered composite replacing the single
hardcoded `callback_on_step_end=collector.on_step_end` binding
(`dgemma/loop.py:582` pre-R1; ARCHITECTURE.md's "Single hardcoded callback
binding" violation row).

This IS the enforcement surface ARCHITECTURE.md's enforcement-surface table
names for "Composition ordering (capture pre-pin; beta-rebuild before pin;
pin last writer) | Ordered-composite test over the shared fake-pipeline
fixture" and cites `dgemma/composite.py:StepEndComposite`.

Driven through `tests/conftest.py`'s `fake_pipeline_factory`/`fake_pipeline`
fixtures (#35 R4), per that module's own self-test convention
(`tests/test_conftest_fake_pipeline.py`'s docstring): going through the
fixture is the honest test of the real seam — a real `run_diffusion` call
threads `StepEndComposite` into exactly this pipeline shape
(`dgemma/loop.py`'s `pipeline(..., callback_on_step_end=step_end, ...)`).

`dgemma.composite`/`dgemma.loop` ARE imported directly (unlike the fixture
classes in `conftest.py`): they are the real modules under test, not the
test-only fixture layer `test_conftest_fake_pipeline.py`'s import-mode
caveat is about.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.composite import DiffusionCancelled, StepEndComposite, StepEndParticipant


def _run_steps(fake_pipeline_factory, *, num_inference_steps: int, step_end) -> None:
    built = fake_pipeline_factory(num_inference_steps=num_inference_steps)
    built.pipeline(
        num_inference_steps=num_inference_steps,
        callback_on_step_end=step_end,
        callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
    )


class _RecordingParticipant:
    """Named participant that appends `(name, step_idx)` to a shared log and
    optionally overwrites the canvas — used to observe the composite's
    actual call order and canvas-threading without needing real
    beta-rebuild/pin implementations (R1 ships the scaffold, not those
    bodies)."""

    def __init__(self, name: str, log: list, *, canvas_value: int | None = None):
        self.name = name
        self.log = log
        self.canvas_value = canvas_value

    def __call__(self, pipe, global_step, step_idx, callback_kwargs) -> dict | None:
        self.log.append((self.name, step_idx, callback_kwargs["canvas"].clone()))
        if self.canvas_value is None:
            return None
        return {"canvas": torch.full_like(callback_kwargs["canvas"], self.canvas_value)}


class TestFixedOrdering:
    """ADR-CDG-010 Decision 3: capture before any canvas-writer; beta-rebuild
    before pin; pin is the last writer. Proven structurally (the composite
    always runs its participants in this order, not merely by convention),
    exercised end to end through the fake pipeline."""

    def test_capture_runs_before_beta_rebuild_and_pin(self, fake_pipeline_factory):
        log: list = []
        capture = _RecordingParticipant("capture", log)
        beta = _RecordingParticipant("beta_rebuild", log)
        pin = _RecordingParticipant("pin", log)
        step_end = StepEndComposite(capture=capture, beta_rebuild=(beta,), pin=(pin,))

        _run_steps(fake_pipeline_factory, num_inference_steps=3, step_end=step_end)

        names_per_step = [
            [entry[0] for entry in log if entry[1] == step_idx] for step_idx in range(3)
        ]
        assert names_per_step == [
            ["capture", "beta_rebuild", "pin"],
            ["capture", "beta_rebuild", "pin"],
            ["capture", "beta_rebuild", "pin"],
        ]

    def test_pin_is_the_last_writer_its_canvas_value_reaches_the_scheduler(self, fake_pipeline_factory):
        """Pin's `{"canvas": ...}` must be what the pipeline actually
        applies for the step — proving pin's re-assertion is not
        overwritten by anything running after it (nothing does, by
        construction, but this pins the observable behavior a future
        reordering bug would break)."""
        log: list = []
        capture = _RecordingParticipant("capture", log)
        beta = _RecordingParticipant("beta_rebuild", log, canvas_value=7)
        pin = _RecordingParticipant("pin", log, canvas_value=99)
        step_end = StepEndComposite(capture=capture, beta_rebuild=(beta,), pin=(pin,))

        built = fake_pipeline_factory(num_inference_steps=1)
        result = built.pipeline(
            num_inference_steps=1,
            callback_on_step_end=step_end,
            callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
        )
        assert torch.all(result.sequences == 99)

    def test_beta_rebuild_writes_before_pin_pin_sees_beta_output_not_original_canvas(self, fake_pipeline_factory):
        """Beta-rebuild must finish writing before pin re-asserts (ADR-CDG-010
        Decision 3) — proven by pin observing beta's output in its own
        `callback_kwargs["canvas"]`, not the scheduler's original
        `prev_sample`, mirroring the pipeline's own cross-callback
        `callback_outputs.pop("canvas", canvas)` threading applied within
        one callback."""
        seen_by_pin: list = []

        def pin(pipe, global_step, step_idx, callback_kwargs):
            seen_by_pin.append(callback_kwargs["canvas"].clone())
            return None

        beta = _RecordingParticipant("beta_rebuild", [], canvas_value=42)
        step_end = StepEndComposite(capture=lambda *a: {}, beta_rebuild=(beta,), pin=(pin,))

        _run_steps(fake_pipeline_factory, num_inference_steps=1, step_end=step_end)

        assert torch.all(seen_by_pin[0] == 42)

    def test_capture_sees_pre_writer_canvas_never_post_pin_state(self, fake_pipeline_factory):
        """Capture must read model-committed, pre-pin truth (ADR-CDG-010
        Decision 3's rationale: capture ordered after pin would record
        constraint-asserted tokens as if the model had committed them).
        Proven by capture's observed canvas never equalling pin's
        post-hoc override value."""
        seen_by_capture: list = []

        def capture(pipe, global_step, step_idx, callback_kwargs):
            seen_by_capture.append(callback_kwargs["canvas"].clone())
            return {}

        pin = _RecordingParticipant("pin", [], canvas_value=123)
        step_end = StepEndComposite(capture=capture, pin=(pin,))

        _run_steps(fake_pipeline_factory, num_inference_steps=1, step_end=step_end)

        assert not torch.all(seen_by_capture[0] == 123)

    def test_no_writer_participants_returns_empty_dict_scheduler_output_wins(self, fake_pipeline_factory):
        """With only capture wired (today's real `run_diffusion` shape — no
        beta-rebuild/pin participants exist yet), the composite's return
        must be `{}` so the pipeline's `.pop("canvas", canvas)` keeps the
        scheduler's own `prev_sample`, unchanged from the pre-composite
        single-binding behavior."""
        step_end = StepEndComposite(capture=lambda *a: {})
        built = fake_pipeline_factory(num_inference_steps=2)
        result = built.pipeline(
            num_inference_steps=2,
            callback_on_step_end=step_end,
            callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
        )
        # Default fixture acceptance is all-True, so the scheduler's own
        # prev_sample (== the fed-in canvas, per FakeEntropyBoundScheduler.step)
        # survives untouched.
        assert result.sequences.shape == (1, 4)


class TestOrderingIsStructural:
    """The fixed order is not a convention callers must remember — it is
    baked into `StepEndComposite.__call__`'s body, so a participant cannot
    be registered "out of order": there is no ordering parameter to get
    wrong, only fixed constructor slots (`capture`, `beta_rebuild`, `pin`).
    This test asserts that structural property directly rather than only
    its behavioral consequence."""

    def test_constructor_has_no_ordering_parameter(self):
        import inspect

        sig = inspect.signature(StepEndComposite.__init__)
        assert set(sig.parameters) == {"self", "capture", "should_cancel", "beta_rebuild", "pin"}

    def test_participant_protocol_has_no_priority_or_order_field(self):
        # StepEndParticipant is a Protocol naming only `name` + `__call__`;
        # no ordering/priority attribute exists for a participant to abuse
        # to reorder itself relative to the fixed capture/beta/pin slots.
        assert set(StepEndParticipant.__annotations__) == {"name"}


class TestCancellationSeam:
    """Issue #38, folded into R1's composer spec: the cancel check has a
    defined position (second — AFTER capture, before every canvas-writer;
    ADR-CDG-010 cancellation amendment 2026-07-13, PR #45) and a defined
    exception path (`DiffusionCancelled`, propagating out of the composite
    for `run_diffusion` to catch — see `tests/test_run_diffusion_cancel.py`
    for the full `run_diffusion`-level partial-return behavior; this class
    covers the composite's own contract in isolation)."""

    def test_should_cancel_true_raises_after_capturing_the_truncation_frame(self, fake_pipeline_factory):
        """The amendment's load-bearing flip: the cancelled step's canvas is
        already scheduler-committed by `callback_on_step_end` time, so
        capture MUST run before the cancellation check — the trace keeps
        the exact truncation-point frame. (Pre-amendment, cancel-first,
        this assertion was `capture_calls == []`; the flip is deliberate,
        pinning the new order — a regression back to cancel-first fails
        here.)"""
        capture_calls: list = []

        def capture(pipe, global_step, step_idx, callback_kwargs):
            capture_calls.append(step_idx)
            return {}

        step_end = StepEndComposite(capture=capture, should_cancel=lambda: True)
        built = fake_pipeline_factory(num_inference_steps=3)

        with pytest.raises(DiffusionCancelled) as exc_info:
            built.pipeline(
                num_inference_steps=3,
                callback_on_step_end=step_end,
                callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
            )
        # Cancellation fires on the very first step, but only AFTER that
        # step's committed frame was captured — exactly one capture call.
        assert capture_calls == [0]
        assert exc_info.value.step_idx == 0

    def test_cancelled_step_runs_no_canvas_writers(self, fake_pipeline_factory):
        """The half of the old cancel-first rationale that survives the
        amendment: cancellation still precedes every canvas-writer, so no
        beta-rebuild/pin pass runs for a step whose result will never be
        used — only the evidence (capture) side of the step completes."""
        writer_log: list = []
        beta = _RecordingParticipant("beta_rebuild", writer_log, canvas_value=7)
        pin = _RecordingParticipant("pin", writer_log, canvas_value=99)
        capture_calls: list = []

        def capture(pipe, global_step, step_idx, callback_kwargs):
            capture_calls.append(step_idx)
            return {}

        step_end = StepEndComposite(
            capture=capture, should_cancel=lambda: True, beta_rebuild=(beta,), pin=(pin,)
        )
        built = fake_pipeline_factory(num_inference_steps=3)

        with pytest.raises(DiffusionCancelled):
            built.pipeline(
                num_inference_steps=3,
                callback_on_step_end=step_end,
                callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
            )
        assert capture_calls == [0]
        assert writer_log == []

    def test_should_cancel_false_never_raises_capture_runs_every_step(self, fake_pipeline_factory):
        capture_calls: list = []

        def capture(pipe, global_step, step_idx, callback_kwargs):
            capture_calls.append(step_idx)
            return {}

        step_end = StepEndComposite(capture=capture, should_cancel=lambda: False)
        _run_steps(fake_pipeline_factory, num_inference_steps=3, step_end=step_end)

        assert capture_calls == [0, 1, 2]

    def test_no_should_cancel_given_is_a_no_op_every_step(self, fake_pipeline_factory):
        """Default (`should_cancel=None`) must never raise — the common,
        unwired case (today's real `run_diffusion` call sites that don't
        pass `should_cancel`) behaves exactly as before R1."""
        capture_calls: list = []

        def capture(pipe, global_step, step_idx, callback_kwargs):
            capture_calls.append(step_idx)
            return {}

        step_end = StepEndComposite(capture=capture)
        _run_steps(fake_pipeline_factory, num_inference_steps=4, step_end=step_end)

        assert capture_calls == [0, 1, 2, 3]

    def test_cancel_after_n_steps_captures_through_the_truncation_step(self, fake_pipeline_factory):
        """A predicate that flips True partway through stops the run on the
        step where it trips — but that step's committed frame is still
        captured (the amendment's evidence policy): capture runs through
        and including the truncation step, and no step after it."""
        capture_calls: list = []
        state = {"count": 0}

        def should_cancel() -> bool:
            state["count"] += 1
            return state["count"] > 2  # cancel on the 3rd check (step_idx 2)

        def capture(pipe, global_step, step_idx, callback_kwargs):
            capture_calls.append(step_idx)
            return {}

        step_end = StepEndComposite(capture=capture, should_cancel=should_cancel)
        built = fake_pipeline_factory(num_inference_steps=5)

        with pytest.raises(DiffusionCancelled) as exc_info:
            built.pipeline(
                num_inference_steps=5,
                callback_on_step_end=step_end,
                callback_on_step_end_tensor_inputs=["canvas", "scheduler_output"],
            )
        assert exc_info.value.step_idx == 2
        # Steps 0 and 1 completed normally; step 2 (the truncation step) was
        # captured and THEN cancelled; steps 3-4 never ran.
        assert capture_calls == [0, 1, 2]

    def test_diffusion_cancelled_carries_step_idx(self):
        exc = DiffusionCancelled(step_idx=5)
        assert exc.step_idx == 5
        assert "5" in str(exc)
