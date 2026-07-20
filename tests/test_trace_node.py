"""surfaces/comfyui/trace.py adapts without logic (ADR-CDG-003): unpack ->
call the `consumers.analysis` functions -> wrap into `IMAGE`/`STRING`.
Mirrors `tests/test_loader_contract.py`'s thin-wrapper test shape:
monkeypatch the exact `consumers.analysis` calls (as imported into
`surfaces.comfyui.trace`) and assert `DGemmaTrace.render` is a pure
pass-through/wrap around their outputs, plus a small set of direct
tensor-shape assertions for the one piece of real (ComfyUI-sanctioned, see
surfaces/comfyui/trace.py's own docstring) wrapping logic in this file.
"""
from __future__ import annotations

import pytest
import torch

from consumers.analysis import MaskTokenCorroboration
from surfaces.comfyui.trace import DGemmaTrace


def test_declarations():
    spec = DGemmaTrace.INPUT_TYPES()
    assert set(spec["required"]) == {"canvas_trace", "cell_px", "mode"}
    assert spec["required"]["canvas_trace"] == ("DGEMMA_CANVAS_TRACE",)
    # cell_px (operator finding, 2026-07-05): nearest-neighbor upscale
    # widget — a raw steps×positions heatmap (256×11 observed live) is
    # unreadably small as pixels.
    assert spec["required"]["cell_px"] == ("INT", {"default": 6, "min": 1, "max": 32})
    # mode (ADR-CDG-014 issue #61 P-D): default "commit" is byte-identical
    # to every pre-P-D graph.
    assert spec["required"]["mode"] == (["commit", "entropy"], {"default": "commit"})
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

    def fake_build_commit_heatmap(trace, scale=1):
        captured["heatmap_trace"] = trace
        captured["scale"] = scale
        return [[1, 0], [0, 1]]

    def fake_build_avalanche_curve(trace):
        captured["curve_trace"] = trace
        return [0.5, 1.0]

    def fake_corroborate_no_mask_token(trace):
        captured["corroboration_trace"] = trace
        return MaskTokenCorroboration(verdict="evidence_against_sentinel")

    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", fake_build_commit_heatmap)
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", fake_build_avalanche_curve)
    monkeypatch.setattr("surfaces.comfyui.trace.corroborate_no_mask_token", fake_corroborate_no_mask_token)

    node = DGemmaTrace()
    image, summary = node.render(canvas_trace=sentinel_trace, cell_px=4)

    # No logic of its own: every `consumers.analysis` call received the same
    # trace object, unmodified — and `cell_px` threads straight through as
    # `scale` (the scaling math is engine-side, ADR-CDG-003).
    assert captured["heatmap_trace"] is sentinel_trace
    assert captured["scale"] == 4
    assert captured["curve_trace"] is sentinel_trace
    assert captured["corroboration_trace"] is sentinel_trace

    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 2, 2, 3)
    assert image.dtype == torch.float32
    assert torch.equal(image[0, :, :, 0], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    assert "steps=2" in summary
    assert "0.5000" in summary and "1.0000" in summary
    assert "no fixed sentinel" in summary


def test_render_defaults_cell_px_to_6(monkeypatch):
    """The widget default and the Python-signature default must agree —
    a graph built without the widget (older banked graphs) and a fresh GUI
    node must render identically."""
    captured = {}

    def fake_build_commit_heatmap(trace, scale=1):
        captured["scale"] = scale
        return [[1]]

    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", fake_build_commit_heatmap)
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="evidence_against_sentinel"),
    )

    node = DGemmaTrace()
    node.render(canvas_trace=_FakeTrace())

    assert captured["scale"] == 6
    assert DGemmaTrace.INPUT_TYPES()["required"]["cell_px"][1]["default"] == 6


def test_render_end_to_end_scaled_image_dimensions():
    """One unmocked pass (real `consumers.analysis` functions) pinning the
    scaled IMAGE dimensions: a 2-frame, 3-position trace at cell_px=4 must
    produce a (1, 2*4, 3*4, 3) tensor — the operator-visible fix."""
    from dgemma.types import CanvasTrace, DiffusionFrame

    frames = [
        DiffusionFrame(
            canvas_idx=0, step_idx=0, t=1.0, temperature=0.8,
            committed_fraction_per_example=(0.0,), canvas=torch.tensor([[1, 2, 3]]),
        ),
        DiffusionFrame(
            canvas_idx=0, step_idx=1, t=0.5, temperature=0.6,
            committed_fraction_per_example=(1.0,), canvas=torch.tensor([[1, 5, 3]]),
        ),
    ]
    trace = CanvasTrace(frames=frames, scheduler_name="EntropyBoundScheduler", scheduler_config={})

    node = DGemmaTrace()
    image, summary = node.render(canvas_trace=trace, cell_px=4)

    assert image.shape == (1, 2 * 4, 3 * 4, 3)


def test_render_labels_committed_fraction_as_block_local(monkeypatch):
    """ADR-CDG-009 / issue #26: the summary must say `committed_fraction` is
    block-local (resets at each canvas boundary), not just print bare
    numbers a reader could misread as a global-progress sawtooth/re-melt."""
    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", lambda trace, scale=1: [[1]])
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0, 0.0, 1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="evidence_against_sentinel"),
    )

    node = DGemmaTrace()
    _, summary = node.render(canvas_trace=_FakeTrace())

    assert "block-local" in summary
    assert "canvas/block boundary" in summary
    assert "1.0000, 0.0000, 1.0000" in summary


def test_render_reports_fixed_sentinel_candidate_in_summary(monkeypatch):
    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", lambda trace, scale=1: [[1]])
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="sentinel_found", candidate_sentinel_id=99),
    )

    node = DGemmaTrace()
    _, summary = node.render(canvas_trace=_FakeTrace())

    assert "FIXED SENTINEL CANDIDATE id=99" in summary


def test_render_reports_vacuous_verdict_distinctly_in_summary(monkeypatch):
    """Issue #22: zero observed transitions must NOT print the same
    "supported" wording genuine evidence earns — the vacuous case gets its
    own line, so the summary can't overclaim corroboration on zero
    evidence."""
    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", lambda trace, scale=1: [[1]])
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="vacuous"),
    )

    node = DGemmaTrace()
    _, summary = node.render(canvas_trace=_FakeTrace())

    assert "vacuous" in summary
    assert "supported" not in summary
    assert "FIXED SENTINEL CANDIDATE" not in summary


def test_heatmap_to_image_degenerate_empty_heatmap_does_not_raise(monkeypatch):
    """Defensive edge case: an empty trace's heatmap must not crash tensor
    construction — degrade to a minimal placeholder image instead."""
    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", lambda trace, scale=1: [])
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="evidence_against_sentinel"),
    )

    node = DGemmaTrace()
    image, summary = node.render(canvas_trace=_FakeTrace())

    assert image.shape == (1, 1, 1, 3)
    assert "steps=0" in summary


# ---------------------------------------------------------------------------
# ADR-CDG-014 issue #61 P-D: mode widget ("commit" default | "entropy")
# ---------------------------------------------------------------------------


def test_render_defaults_mode_to_commit_and_does_not_call_entropy_heatmap(monkeypatch):
    """The widget default and the Python-signature default must agree
    (mirrors `cell_px`'s own default-parity test) — and, since default
    mode is "commit", `build_entropy_heatmap` must never be called."""
    captured = {}

    def fake_build_commit_heatmap(trace, scale=1):
        captured["called_commit"] = True
        return [[1]]

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_entropy_heatmap must not be called when mode='commit'")

    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", fake_build_commit_heatmap)
    monkeypatch.setattr("surfaces.comfyui.trace.build_entropy_heatmap", fail_if_called)
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="evidence_against_sentinel"),
    )

    node = DGemmaTrace()
    node.render(canvas_trace=_FakeTrace())

    assert captured["called_commit"] is True
    assert DGemmaTrace.INPUT_TYPES()["required"]["mode"][1]["default"] == "commit"


def test_render_mode_entropy_calls_entropy_heatmap_not_commit_heatmap(monkeypatch):
    captured = {}

    def fail_if_called(*args, **kwargs):
        raise AssertionError("build_commit_heatmap must not be called when mode='entropy'")

    def fake_build_entropy_heatmap(trace, scale=1):
        captured["entropy_trace"] = trace
        captured["scale"] = scale
        return [[0.1, 0.9], [0.5, 0.5]]

    monkeypatch.setattr("surfaces.comfyui.trace.build_commit_heatmap", fail_if_called)
    monkeypatch.setattr("surfaces.comfyui.trace.build_entropy_heatmap", fake_build_entropy_heatmap)
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="evidence_against_sentinel"),
    )

    sentinel_trace = _FakeTrace()
    node = DGemmaTrace()
    image, _ = node.render(canvas_trace=sentinel_trace, cell_px=3, mode="entropy")

    assert captured["entropy_trace"] is sentinel_trace
    assert captured["scale"] == 3
    assert isinstance(image, torch.Tensor)
    assert image.shape == (1, 2, 2, 3)


def test_render_unknown_mode_raises(monkeypatch):
    monkeypatch.setattr("surfaces.comfyui.trace.build_avalanche_curve", lambda trace: [1.0])
    monkeypatch.setattr(
        "surfaces.comfyui.trace.corroborate_no_mask_token",
        lambda trace: MaskTokenCorroboration(verdict="evidence_against_sentinel"),
    )

    node = DGemmaTrace()
    with pytest.raises(ValueError, match="unknown mode"):
        node.render(canvas_trace=_FakeTrace(), mode="bogus")


def test_entropy_heatmap_to_image_min_max_normalizes_per_render():
    """`_entropy_heatmap_to_image` min-max normalizes to [0, 1] — a 2x2
    grid with min=0.1, max=0.9 must map 0.1->0.0, 0.9->1.0, and the
    midpoint 0.5 -> 0.5."""
    from surfaces.comfyui.trace import _entropy_heatmap_to_image

    image = _entropy_heatmap_to_image([[0.1, 0.9], [0.5, 0.1]])

    assert image.shape == (1, 2, 2, 3)
    assert image[0, 0, 0, 0].item() == pytest.approx(0.0)
    assert image[0, 0, 1, 0].item() == pytest.approx(1.0)
    assert image[0, 1, 0, 0].item() == pytest.approx(0.5)
    assert image[0, 1, 1, 0].item() == pytest.approx(0.0)


def test_entropy_heatmap_to_image_degenerate_all_equal_grid_normalizes_to_zero():
    """A grid where every cell is equal (max == min) must not divide by
    zero — normalizes to all-0 rather than raising/NaN-ing (e.g. a
    single-frame trace, or a fully-converged run with zero entropy
    spread)."""
    from surfaces.comfyui.trace import _entropy_heatmap_to_image

    image = _entropy_heatmap_to_image([[0.42, 0.42], [0.42, 0.42]])

    assert not torch.isnan(image).any()
    assert torch.equal(image[..., 0], torch.zeros(1, 2, 2))


def test_entropy_heatmap_to_image_degenerate_empty_heatmap_does_not_raise():
    from surfaces.comfyui.trace import _entropy_heatmap_to_image

    image = _entropy_heatmap_to_image([])

    assert image.shape == (1, 1, 1, 3)


def test_render_end_to_end_entropy_mode_real_functions():
    """One unmocked pass (real `consumers.analysis.build_entropy_heatmap`)
    over a hand-built trace with real `entropy` tensors — pins that the
    mode="entropy" path produces a correctly-shaped, normalized IMAGE with
    no mocking involved."""
    from dgemma.types import CanvasTrace, DiffusionFrame

    frames = [
        DiffusionFrame(
            canvas_idx=0, step_idx=0, t=1.0, temperature=0.8,
            committed_fraction_per_example=(0.0,), canvas=torch.tensor([[1, 2, 3]]),
            entropy=torch.tensor([0.1, 0.5, 0.9]),
        ),
        DiffusionFrame(
            canvas_idx=0, step_idx=1, t=0.5, temperature=0.6,
            committed_fraction_per_example=(1.0,), canvas=torch.tensor([[1, 5, 3]]),
            entropy=torch.tensor([0.2, 0.4, 0.6]),
        ),
    ]
    trace = CanvasTrace(frames=frames, scheduler_name="EntropyBoundScheduler", scheduler_config={})

    node = DGemmaTrace()
    image, summary = node.render(canvas_trace=trace, cell_px=2, mode="entropy")

    assert image.shape == (1, 2 * 2, 3 * 2, 3)
    assert image.min().item() == pytest.approx(0.0)
    assert image.max().item() == pytest.approx(1.0)
    assert "steps=2" in summary
