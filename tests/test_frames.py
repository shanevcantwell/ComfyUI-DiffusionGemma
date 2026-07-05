"""Frame / CanvasState contract tests — `dgemma/loop.py`'s per-step collection
seam (plan.md: frames are the loop's native contract from day one, not
something Phase 3 invents).

Uses a FAKE `scheduler_output` and a stub pipeline call (nothing here touches
`diffusers.DiffusionGemmaPipeline` or a real model): `_FrameCollector.on_step_end`
and `derive_canvas_state` are pure functions of the callback contract, exactly
so they're unit-testable this way (ADR-CDG-003).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from dgemma.loop import _FrameCollector, anneal_temperature, derive_canvas_state
from dgemma.types import CanvasState, DiffusionFrame


@dataclass
class FakeSchedulerOutput:
    """Mimics `EntropyBoundSchedulerOutput` (diffusers 0.39.0,
    `scheduling_entropy_bound.py`) — only the field the collector reads."""

    accepted_index: torch.Tensor


def _callback_kwargs(accepted: list[list[bool]], canvas_value: int = 0) -> dict:
    """Build callback kwargs from a per-example acceptance matrix
    (`accepted[i][j]` = example i, block position j)."""
    accepted_tensor = torch.tensor(accepted, dtype=torch.bool)
    return {
        "scheduler_output": FakeSchedulerOutput(accepted_index=accepted_tensor),
        "canvas": torch.full(accepted_tensor.shape, canvas_value, dtype=torch.long),
    }


def _frame(**overrides) -> DiffusionFrame:
    defaults = dict(
        canvas_idx=0,
        step_idx=3,
        t=0.1,
        temperature=0.42,
        committed_fraction_per_example=(1.0,),
        canvas=None,
    )
    defaults.update(overrides)
    return DiffusionFrame(**defaults)


class TestAnnealTemperature:
    """Spot-checks against `scheduling_entropy_bound.py:153-155`'s inlined formula."""

    def test_step_0_is_hottest(self):
        t, temperature = anneal_temperature(step_idx=0, num_inference_steps=48, t_min=0.4, t_max=0.8)
        assert t == pytest.approx(1.0)
        assert temperature == pytest.approx(0.8)

    def test_mid_schedule_step(self):
        t, temperature = anneal_temperature(step_idx=24, num_inference_steps=48, t_min=0.4, t_max=0.8)
        assert t == pytest.approx((48 - 24) / 48)
        assert temperature == pytest.approx(0.4 + 0.4 * ((48 - 24) / 48))

    def test_last_step_approaches_but_does_not_equal_t_min(self):
        t, temperature = anneal_temperature(step_idx=47, num_inference_steps=48, t_min=0.4, t_max=0.8)
        assert t == pytest.approx(1 / 48)
        assert temperature == pytest.approx(0.4 + 0.4 * (1 / 48))
        assert temperature > 0.4


class TestFrameCollector:
    def test_committed_fraction_math(self):
        collector = _FrameCollector(num_inference_steps=4, t_min=0.4, t_max=0.8, keep_frames="all")
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True, True, False, False]]))
        assert collector.frames[0].committed_fraction == pytest.approx(0.5)
        assert collector.frames[0].committed_fraction_per_example == pytest.approx((0.5,))

    def test_committed_fraction_batched_is_per_example(self):
        """Review finding (2026-07-05): mean over the block dim ONLY. A
        batch-blended scalar would silently claim 0.5 for a batch where one
        example committed everything and the other nothing."""
        collector = _FrameCollector(num_inference_steps=4, t_min=0.4, t_max=0.8, keep_frames="all")
        collector.on_step_end(
            None, 0, 0,
            _callback_kwargs([[True, True, True, True], [False, False, False, False]]),
        )

        frame = collector.frames[0]
        assert frame.committed_fraction_per_example == pytest.approx((1.0, 0.0))
        # The scalar convenience refuses to blend examples rather than lying.
        with pytest.raises(ValueError, match="batch_size==1"):
            _ = frame.committed_fraction

    def test_degenerate_zero_length_block_raises(self):
        """Review finding (2026-07-05): block length 0 would make the mean
        NaN, and NaN reads as not-converged downstream — pin the honest
        behavior: surface degenerate input, never launder it into a validity
        field."""
        collector = _FrameCollector(num_inference_steps=4, t_min=0.4, t_max=0.8)
        kwargs = {
            "scheduler_output": FakeSchedulerOutput(accepted_index=torch.empty((1, 0), dtype=torch.bool)),
            "canvas": torch.empty((1, 0), dtype=torch.long),
        }
        with pytest.raises(ValueError, match="block length 0"):
            collector.on_step_end(None, 0, 0, kwargs)

    def test_absolute_t_keying_not_loop_index(self):
        """A mid-schedule-start run (variation/renoise, `loose-ends.md`): the
        loop-relative counter (`global_step`) starts at 0 but the
        scheduler-relative `step_idx` values don't. Frames must key off
        `step_idx` (and the `t`/`temperature` derived from it), not the
        0-based loop position, or cross-run traces would be silently
        incomparable."""
        collector = _FrameCollector(num_inference_steps=48, t_min=0.4, t_max=0.8, keep_frames="all")
        collector.on_step_end(None, 0, 20, _callback_kwargs([[True] * 4]))
        collector.on_step_end(None, 1, 21, _callback_kwargs([[True] * 4]))

        assert collector.frames[0].step_idx == 20
        assert collector.frames[1].step_idx == 21
        assert collector.frames[0].t != collector.frames[1].t
        assert collector.frames[0].temperature != collector.frames[1].temperature

    def test_multi_canvas_frame_identity(self):
        """Review finding (2026-07-05): the pipeline's `step_idx` resets per
        canvas (`pipeline_diffusion_gemma.py:356`, nested in the canvas loop
        at `:318`), so once gen_length > canvas_length two frames can share
        `(step_idx, t, temperature)`. `canvas_idx` must distinguish them."""
        collector = _FrameCollector(num_inference_steps=3, t_min=0.4, t_max=0.8, keep_frames="all")
        # Canvas 0: steps 0,1,2 — then canvas 1: steps 0,1 (early-stopped).
        for global_step, step_idx in enumerate([0, 1, 2, 0, 1]):
            collector.on_step_end(None, global_step, step_idx, _callback_kwargs([[True] * 4]))

        assert [f.canvas_idx for f in collector.frames] == [0, 0, 0, 1, 1]
        step0_frames = [f for f in collector.frames if f.step_idx == 0]
        assert len(step0_frames) == 2
        # Same schedule position, distinct identity via canvas_idx.
        assert step0_frames[0].t == step0_frames[1].t
        assert step0_frames[0].canvas_idx != step0_frames[1].canvas_idx

    def test_keep_frames_last_retains_only_latest(self):
        collector = _FrameCollector(num_inference_steps=4, t_min=0.4, t_max=0.8, keep_frames="last")
        for step_idx in range(4):
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * 4]))

        assert len(collector.frames) == 1
        assert collector.frames[0].step_idx == 3
        assert collector.steps_used == 4  # counted regardless of retention policy

    def test_keep_frames_all_retains_every_step(self):
        collector = _FrameCollector(num_inference_steps=4, t_min=0.4, t_max=0.8, keep_frames="all")
        for step_idx in range(4):
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * 4]))

        assert len(collector.frames) == 4
        assert [f.step_idx for f in collector.frames] == [0, 1, 2, 3]

    def test_on_frame_invoked_once_per_captured_step_with_right_frame(self):
        """P3 step 2: `on_frame` is the seam that lets `nodes/sampler.py`
        push a live view without `dgemma/loop.py` ever importing ComfyUI
        (ADR-CDG-003) — this pins the pure-Python half of that contract:
        invoked exactly once per step, with the same `DiffusionFrame` that
        got appended to `collector.frames`."""
        seen: list[DiffusionFrame] = []
        collector = _FrameCollector(
            num_inference_steps=3, t_min=0.4, t_max=0.8, keep_frames="all", on_frame=seen.append
        )
        for step_idx in range(3):
            collector.on_step_end(None, step_idx, step_idx, _callback_kwargs([[True] * 4]))

        assert len(seen) == 3
        assert seen == collector.frames

    def test_on_frame_exception_propagates_engine_contract(self):
        """Engine contract (review finding, 2026-07-05): the collector does
        NOT swallow a caller's callback exception — silently eating a user's
        analysis-callback error at the engine layer would be its own
        dishonesty. Display-only callbacks guard themselves at their own
        layer (`nodes/sampler.py`'s closure). The frame is still retained
        before the callback runs, so nothing is lost by the raise."""
        def exploding(frame):
            raise RuntimeError("user callback bug")

        collector = _FrameCollector(
            num_inference_steps=1, t_min=0.4, t_max=0.8, keep_frames="all", on_frame=exploding
        )
        with pytest.raises(RuntimeError, match="user callback bug"):
            collector.on_step_end(None, 0, 0, _callback_kwargs([[True]]))
        assert len(collector.frames) == 1  # retention happened before the callback raised

    def test_on_frame_none_is_a_silent_no_op(self):
        """Default (`on_frame=None`): capture proceeds exactly as before —
        no live-push consumer is required for the collector to work."""
        collector = _FrameCollector(num_inference_steps=1, t_min=0.4, t_max=0.8)
        collector.on_step_end(None, 0, 0, _callback_kwargs([[True]]))  # must not raise
        assert collector.steps_used == 1

    def test_on_step_end_is_pure_capture_no_canvas_override(self):
        """P1 is pure capture (ADR-CDG-004): the collector must return `{}` so
        `callback_outputs.pop("canvas", canvas)` at the pipeline call site
        leaves the canvas untouched. Constraint injection (P5) is a different
        callback."""
        collector = _FrameCollector(num_inference_steps=1, t_min=0.4, t_max=0.8)
        result = collector.on_step_end(None, 0, 0, _callback_kwargs([[True]]))
        assert result == {}


class TestDeriveCanvasState:
    def test_converged_when_last_frame_fully_committed(self):
        state = derive_canvas_state(
            text="hello", canvas_ids=None,
            frames=[_frame(committed_fraction_per_example=(1.0,))], steps_used=4,
        )

        assert isinstance(state, CanvasState)
        assert state.text == "hello"
        assert state.converged is True
        assert state.committed_fraction == pytest.approx(1.0)
        assert state.steps_used == 4

    def test_not_converged_when_partial_commit(self):
        state = derive_canvas_state(
            text="hello", canvas_ids=None,
            frames=[_frame(committed_fraction_per_example=(0.5,))], steps_used=4,
        )

        assert state.converged is False
        assert state.committed_fraction == pytest.approx(0.5)

    def test_raises_on_no_frames(self):
        with pytest.raises(RuntimeError):
            derive_canvas_state(text="", canvas_ids=None, frames=[], steps_used=0)

    def test_raises_on_batched_frame_rather_than_blending(self):
        """CanvasState is single-example scope (P1 drives one prompt);
        deriving from a batched frame must refuse, not blend."""
        with pytest.raises(ValueError, match="batch_size==1"):
            derive_canvas_state(
                text="x", canvas_ids=None,
                frames=[_frame(committed_fraction_per_example=(1.0, 0.0))], steps_used=1,
            )

    def test_turn_closed_and_answer_tokens_default_when_omitted(self):
        """[SEVERABLE RIDER — issue #9] Additive-only: existing call shape
        (no `eos_token_id`, `canvas_ids=None`) must keep working, with the
        new fields defaulting honestly (no evidence of EOS -> not closed;
        no canvas -> 0 answer tokens) rather than requiring every existing
        call site to be touched."""
        state = derive_canvas_state(
            text="hello", canvas_ids=None,
            frames=[_frame(committed_fraction_per_example=(1.0,))], steps_used=4,
        )

        assert state.turn_closed is False
        assert state.answer_tokens == 0

    def test_turn_closed_true_when_eos_present_in_canvas_ids(self):
        """`answer_tokens` counts pre-EOS content only (review finding,
        2026-07-05): the EOS is the stop signal, not answer content, so
        `[1, 2, 999, 3]` honestly counts 2 — never 4 (which would include
        the EOS and the post-EOS fill)."""
        state = derive_canvas_state(
            text="hello", canvas_ids=[1, 2, 999, 3],
            frames=[_frame(committed_fraction_per_example=(1.0,))], steps_used=4,
            eos_token_id=999,
        )

        assert state.turn_closed is True
        assert state.answer_tokens == 2

    def test_answer_tokens_excludes_trailing_eos_fill_padding(self):
        """The converged-run padding case at the unit level: content + EOS +
        N×EOS fill (the live-observed ~30-token tail) counts only the
        content, not the padding a bare `len()` would inflate by."""
        state = derive_canvas_state(
            text="hello", canvas_ids=[7, 8, 9, 999, 999, 999, 999],
            frames=[_frame(committed_fraction_per_example=(1.0,))], steps_used=4,
            eos_token_id=999,
        )

        assert state.turn_closed is True
        assert state.answer_tokens == 3

    def test_turn_closed_false_when_eos_absent_despite_full_commit(self):
        """Specimen (ii)'s shape at the `derive_canvas_state` unit level:
        `committed_fraction == 1.0` (full commit this step) does NOT imply
        `turn_closed` — the canvas can be entirely full of plausible,
        non-EOS filler."""
        state = derive_canvas_state(
            text="hello", canvas_ids=[1, 2, 3, 4],
            frames=[_frame(committed_fraction_per_example=(1.0,))], steps_used=4,
            eos_token_id=999,
        )

        assert state.converged is True
        assert state.turn_closed is False
        assert state.answer_tokens == 4
