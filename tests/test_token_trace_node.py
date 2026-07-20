"""surfaces/comfyui/token_trace.py adapts without logic (ADR-CDG-003): unpack
-> call `consumers.analysis.build_token_identity_grid` -> wrap into `STRING`.
Mirrors `tests/test_trace_node.py`'s thin-wrapper test shape (issue #61 P-D /
issue #11).
"""
from __future__ import annotations

import torch

from dgemma.types import CanvasTrace, DiffusionFrame
from surfaces.comfyui.token_trace import DGemmaTokenTrace


def test_declarations():
    spec = DGemmaTokenTrace.INPUT_TYPES()
    assert set(spec["required"]) == {"canvas_trace"}
    assert spec["required"]["canvas_trace"] == ("DGEMMA_CANVAS_TRACE",)
    assert DGemmaTokenTrace.RETURN_TYPES == ("STRING",)
    assert DGemmaTokenTrace.RETURN_NAMES == ("token_report",)
    assert DGemmaTokenTrace.FUNCTION == "render"
    assert DGemmaTokenTrace.CATEGORY == "DiffusionGemma"


def _frame(canvas_idx, step_idx, canvas_row, t=1.0, temperature=0.5):
    return DiffusionFrame(
        canvas_idx=canvas_idx,
        step_idx=step_idx,
        t=t,
        temperature=temperature,
        committed_fraction_per_example=(1.0,),
        canvas=torch.tensor([canvas_row], dtype=torch.long),
    )


def test_render_calls_build_token_identity_grid_and_wraps_result(monkeypatch):
    captured = {}

    def fake_build_token_identity_grid(trace):
        captured["trace"] = trace
        return [[1, 2, 3]]

    monkeypatch.setattr(
        "surfaces.comfyui.token_trace.build_token_identity_grid", fake_build_token_identity_grid
    )

    trace = CanvasTrace(
        frames=[_frame(0, 0, [1, 2, 3])],
        scheduler_name="EntropyBoundScheduler",
        scheduler_config={"entropy_bound": 0.1},
        raw_canvas_ids=[9, 8, 7],
    )

    node = DGemmaTokenTrace()
    (report,) = node.render(canvas_trace=trace)

    assert captured["trace"] is trace
    assert isinstance(report, str)
    assert "raw_canvas_ids (3 tokens): [9, 8, 7]" in report
    assert "scheduler=EntropyBoundScheduler" in report
    assert "canvas_idx=0 step_idx=0" in report
    assert "[1, 2, 3]" in report


def test_render_reports_raw_canvas_ids_absent_honestly(monkeypatch):
    """ADR-CDG-014 Decision 1/2: `raw_canvas_ids is None` (legacy/no-capture
    trace) must be reported as "not captured", never rendered as an empty
    list standing in for "no ids"."""
    monkeypatch.setattr(
        "surfaces.comfyui.token_trace.build_token_identity_grid", lambda trace: []
    )

    trace = CanvasTrace(frames=[], scheduler_name="EntropyBoundScheduler", scheduler_config={})

    node = DGemmaTokenTrace()
    (report,) = node.render(canvas_trace=trace)

    assert "raw_canvas_ids: not captured (legacy/no-capture trace)" in report
    assert "raw_canvas_ids (0 tokens)" not in report


def test_render_end_to_end_real_function_no_mocking():
    """One unmocked pass (real `build_token_identity_grid`) over a
    hand-built multi-frame trace."""
    frames = [
        _frame(0, 0, [10, 20, 30], t=1.0, temperature=0.8),
        _frame(0, 1, [10, 25, 30], t=0.5, temperature=0.6),
    ]
    trace = CanvasTrace(
        frames=frames,
        scheduler_name="EntropyBoundScheduler",
        scheduler_config={},
        raw_canvas_ids=[10, 25, 30],
    )

    node = DGemmaTokenTrace()
    (report,) = node.render(canvas_trace=trace)

    assert "raw_canvas_ids (3 tokens): [10, 25, 30]" in report
    assert "per-step token-id grid (2 steps):" in report
    assert "canvas_idx=0 step_idx=0 t=1.0000 temperature=0.8000: [10, 20, 30]" in report
    assert "canvas_idx=0 step_idx=1 t=0.5000 temperature=0.6000: [10, 25, 30]" in report
