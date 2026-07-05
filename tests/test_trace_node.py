"""nodes/trace.py adapts without logic (ADR-CDG-003): unpack -> call the
`dgemma.sampling` functions -> wrap into `IMAGE`/`STRING`. Mirrors
`tests/test_loader_contract.py`'s thin-wrapper test shape: monkeypatch the
exact `dgemma.sampling` calls and assert `DGemmaTrace.render` is a pure
pass-through/wrap around their outputs, plus a small set of direct
tensor-shape assertions for the one piece of real (ComfyUI-sanctioned, see
nodes/trace.py's own docstring) wrapping logic in this file.
"""
from __future__ import annotations

import torch

from dgemma.sampling import MaskTokenCorroboration
from nodes.trace import DGemmaTrace


def test_declarations():
    assert DGemmaTrace.INPUT_TYPES() == {"required": {"canvas_trace": ("DGEMMA_CANVAS_TRACE",)}}
    assert DGemmaTrace.RETURN_TYPES == ("IMAGE", "STRING")
    assert DGemmaTrace.RETURN_NAMES == ("heatmap", "summary")
    assert DGemmaTrace.FUNCTION == "render"
    assert DGemmaTrace.CATEGORY == "DiffusionGemma"


class _FakeTrace:
    scheduler_name = "EntropyBoundScheduler"
    scheduler_config = {"entropy_bound": 0.1}


def test_render_calls_sampling_functions_and_wraps_results(monkeypatch):
    sentinel_trace = _FakeTrace()
    captured = {}

    def fake_build_commit_heatmap(trace):
        captured["heatmap_trace"] = trace
        return [[1, 0], [0, 1]]

    def fake_build_avalanche_curve(trace):
        captured["curve_trace"] = trace
        return [0.5, 1.0]

    def fake_corroborate_no_mask_token(trace):
        captured["corroboration_trace"] = trace
        return MaskTokenCorroboration(no_fixed_sentinel=True)

    monkeypatch.setattr("nodes.trace.build_commit_heatmap", fake_build_commit_heatmap)
    monkeypatch.setattr("nodes.trace.build_avalanche_curve", fake_build_avalanche_curve)
    monkeypatch.setattr("nodes.trace.corroborate_no_mask_token", fake_corroborate_no_mask_token)

    node = DGemmaTrace()
    image, summary = node.render(canvas_trace=sentinel_trace)

    # No logic of its own: every `dgemma.sampling` call received the same
    # trace object, unmodified.
    assert captured["heatmap_trace"] is sentinel_trace
    assert captured["curve_trace"] is sentinel_trace
    assert captured["corroboration_trace"] is sentinel_trace

    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 2, 2, 3)
    assert image.dtype == torch.float32
    assert torch.equal(image[0, :, :, 0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    assert "steps=2" in summary
    assert "0.5000" in summary and "1.0000" in summary
    assert "no fixed sentinel" in summary


def test_render_reports_fixed_sentinel_candidate_in_summary(monkeypatch):
    monkeypatch.setattr("nodes.trace.build_commit_heatmap", lambda trace: [[1]])
    monkeypatch.setattr("nodes.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "nodes.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(no_fixed_sentinel=False, candidate_sentinel_id=99),
    )

    node = DGemmaTrace()
    _, summary = node.render(canvas_trace=_FakeTrace())

    assert "FIXED SENTINEL CANDIDATE id=99" in summary


def test_heatmap_to_image_degenerate_empty_heatmap_does_not_raise(monkeypatch):
    """Defensive edge case: an empty trace's heatmap must not crash tensor
    construction — degrade to a minimal placeholder image instead."""
    monkeypatch.setattr("nodes.trace.build_commit_heatmap", lambda trace: [])
    monkeypatch.setattr("nodes.trace.build_avalanche_curve", lambda trace: [])
    monkeypatch.setattr(
        "nodes.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(no_fixed_sentinel=True),
    )

    node = DGemmaTrace()
    image, summary = node.render(canvas_trace=_FakeTrace())

    assert image.shape == (1, 1, 1, 3)
    assert "steps=0" in summary
