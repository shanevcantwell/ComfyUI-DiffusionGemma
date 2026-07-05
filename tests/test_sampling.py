"""`dgemma/sampling.py` — pure functions over a synthetic `CanvasTrace` (P3
steps 3-4). No ComfyUI, no torch-autograd/pipeline dependency (ADR-CDG-003):
everything here is hand-computable against small, explicit frames.
"""
from __future__ import annotations

import torch

from dgemma.sampling import (
    MaskTokenCorroboration,
    build_avalanche_curve,
    build_commit_heatmap,
    corroborate_no_mask_token,
)
from dgemma.types import CanvasTrace, DiffusionFrame


def _frame(canvas_idx: int, step_idx: int, canvas_row: list[int], committed_fraction: float) -> DiffusionFrame:
    return DiffusionFrame(
        canvas_idx=canvas_idx,
        step_idx=step_idx,
        t=1.0 - step_idx * 0.1,
        temperature=0.5,
        committed_fraction_per_example=(committed_fraction,),
        canvas=torch.tensor([canvas_row], dtype=torch.long),
    )


class TestCanvasTrace:
    """P3 step 1: `CanvasTrace` must carry the scheduler identity that
    minted its frames' commit readings alongside the frames themselves
    (ADR-CDG-001 addendum) — both present and queryable, not a bare list."""

    def test_carries_frames_and_scheduler_identity(self):
        frames = [_frame(0, 0, [1, 2], 1.0)]
        config = {"entropy_bound": 0.1, "t_min": 0.4, "t_max": 0.8, "num_inference_steps": 48}

        trace = CanvasTrace(frames=frames, scheduler_name="EntropyBoundScheduler", scheduler_config=config)

        assert trace.frames == frames
        assert trace.scheduler_name == "EntropyBoundScheduler"
        assert trace.scheduler_config == config


class TestBuildCommitHeatmap:
    def test_shape_and_hand_computed_values(self):
        frames = [
            _frame(0, 0, [1, 2, 3], 0.0),
            _frame(0, 1, [1, 5, 3], 0.33),  # position 1 changed
            _frame(0, 2, [9, 5, 3], 0.66),  # position 0 changed
        ]
        trace = CanvasTrace(frames=frames, scheduler_name="TestScheduler", scheduler_config={})

        heatmap = build_commit_heatmap(trace)

        assert len(heatmap) == 3
        assert all(len(row) == 3 for row in heatmap)
        assert heatmap[0] == [1, 1, 1]  # first frame of the block: nothing locked in yet
        assert heatmap[1] == [0, 1, 0]
        assert heatmap[2] == [1, 0, 0]

    def test_new_canvas_idx_resets_the_diff_baseline(self):
        """A new block (`canvas_idx` bump) has no same-block prior frame to
        diff against — reports all-changed, same as the very first frame."""
        frames = [
            _frame(0, 0, [1, 2], 1.0),
            _frame(0, 1, [1, 2], 1.0),  # stable within block 0
            _frame(1, 0, [1, 2], 0.0),  # new block, identical values but must still read as "changed"
        ]
        trace = CanvasTrace(frames=frames, scheduler_name="TestScheduler", scheduler_config={})

        heatmap = build_commit_heatmap(trace)

        assert heatmap[0] == [1, 1]
        assert heatmap[1] == [0, 0]
        assert heatmap[2] == [1, 1]  # block boundary, not a real "changed" reading


class TestBuildAvalancheCurve:
    def test_returns_committed_fraction_per_frame_in_order(self):
        frames = [
            _frame(0, 0, [1, 2], 0.1),
            _frame(0, 1, [1, 2], 0.4),
            _frame(0, 2, [1, 2], 1.0),
        ]
        trace = CanvasTrace(frames=frames, scheduler_name="TestScheduler", scheduler_config={})

        curve = build_avalanche_curve(trace)

        assert curve == [0.1, 0.4, 1.0]


class TestCorroborateNoMaskToken:
    def test_visibly_varying_uncommitted_positions_report_no_fixed_sentinel(self):
        """Uniform-state renoise signature: the pre-transition values a
        still-unaccepted position held vary across steps."""
        frames = [
            _frame(0, 0, [10, 20, 30, 40], 0.0),
            _frame(0, 1, [11, 20, 31, 40], 0.5),  # positions 0,2 changed FROM 10,30
            _frame(0, 2, [12, 20, 32, 40], 1.0),  # positions 0,2 changed again FROM 11,31
        ]
        trace = CanvasTrace(frames=frames, scheduler_name="TestScheduler", scheduler_config={})

        result = corroborate_no_mask_token(trace)

        assert isinstance(result, MaskTokenCorroboration)
        assert result.no_fixed_sentinel is True
        assert result.candidate_sentinel_id is None

    def test_positions_pinned_to_one_constant_flag_a_candidate_sentinel(self):
        """Absorbing-MASK signature: uncommitted positions hold one fixed id
        (99) until their own one-time reveal — the pre-transition value is
        always 99, never anything else."""
        frames = [
            _frame(0, 0, [99, 20, 99, 40], 0.5),
            _frame(0, 1, [99, 20, 7, 40], 0.75),  # position 2 revealed FROM the sentinel 99
            _frame(0, 2, [5, 20, 7, 40], 1.0),  # position 0 revealed FROM the sentinel 99
        ]
        trace = CanvasTrace(frames=frames, scheduler_name="TestScheduler", scheduler_config={})

        result = corroborate_no_mask_token(trace)

        assert result.no_fixed_sentinel is False
        assert result.candidate_sentinel_id == 99

    def test_no_changes_observed_is_vacuously_no_fixed_sentinel(self):
        """Nothing ever changed (degenerate/synthetic single-frame or fully
        static trace) — no evidence to contradict the no-mask hypothesis."""
        frames = [_frame(0, 0, [1, 2, 3], 1.0)]
        trace = CanvasTrace(frames=frames, scheduler_name="TestScheduler", scheduler_config={})

        result = corroborate_no_mask_token(trace)

        assert result.no_fixed_sentinel is True
        assert result.candidate_sentinel_id is None
