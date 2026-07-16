"""surfaces/comfyui/tally_audit.py adapts without logic (ADR-CDG-003):
unpack -> call `consumers.tally_audit.audit_frames` -> wrap into `STRING`.
Mirrors `tests/test_trace_node.py`'s thin-wrapper test shape: monkeypatch
the exact `consumers.tally_audit` call (as imported into
`surfaces.comfyui.tally_audit`) and assert `DGemmaTallyAudit.audit` is a
pure pass-through/wrap around its output, plus real end-to-end passes
against the fixture-derived frames (issue #84 AC#4).
"""
from __future__ import annotations

from pathlib import Path

from consumers.tally_audit import (
    FrameAuditResult,
    NumeralCellResult,
    RevisionEvent,
    TallyAudit,
    audit_frames,
    extract_decoded_frames_from_composite_blob,
)
from surfaces.comfyui.tally_audit import DGemmaTallyAudit

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_declarations():
    spec = DGemmaTallyAudit.INPUT_TYPES()
    assert set(spec["required"]) == {"frames"}
    assert spec["required"]["frames"] == ("STRING", {"forceInput": True})
    assert DGemmaTallyAudit.INPUT_IS_LIST is True
    assert DGemmaTallyAudit.RETURN_TYPES == ("STRING",)
    assert DGemmaTallyAudit.RETURN_NAMES == ("audit_report",)
    assert DGemmaTallyAudit.FUNCTION == "audit"
    assert DGemmaTallyAudit.CATEGORY == "DiffusionGemma"


def test_audit_calls_audit_frames_and_wraps_result(monkeypatch):
    captured = {}
    sentinel_result = TallyAudit(
        frame_results=[FrameAuditResult(frame_idx=0, parse_status="ok", format_name="inline_list")],
        revisions=[],
        final_frame_arithmetically_consistent=True,
    )

    def fake_audit_frames(frames):
        captured["frames"] = frames
        return sentinel_result

    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", fake_audit_frames)

    node = DGemmaTallyAudit()
    (report,) = node.audit(frames=["frame one", "frame two"])

    # No logic of its own: the exact frames list passed straight through.
    assert captured["frames"] == ["frame one", "frame two"]
    assert "frames=1" in report


def test_audit_accepts_a_tuple_input_from_comfyui_list_wrapping(monkeypatch):
    """ComfyUI's `INPUT_IS_LIST` convention may hand this node a `list` or a
    `tuple` depending on the upstream node's own collection type — the
    adapter must not assume `list` specifically."""
    captured = {}

    def fake_audit_frames(frames):
        captured["frames"] = frames
        return TallyAudit([], [], None)

    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", fake_audit_frames)

    node = DGemmaTallyAudit()
    node.audit(frames=("frame one", "frame two"))

    assert captured["frames"] == ["frame one", "frame two"]


def test_report_lists_per_step_parse_status():
    node = DGemmaTallyAudit()
    (report,) = node.audit(
        frames=[
            "no structure here at all",
            "*   **0:** 1 time\n*   **1:** 1 time\n*   **2:** 1 time\n*   **3:** 1 time\n"
            "*   **4:** 1 time\n*   **5:** 1 time\n*   **6:** 1 time\n*   **7:** 1 time\n"
            "*   **8:** 1 time\n*   **9:** 1 time",
        ]
    )

    assert "step 0: unrecognized" in report
    assert "step 1: ok (inline_list)" in report


def test_report_lists_partial_steps_with_unparsed_numerals():
    node = DGemmaTallyAudit()
    (report,) = node.audit(
        frames=[
            "*   **0:** 1 time\n*   **1:**  garbage token\n*   **2:** 1 time\n*   **3:** 1 time\n"
            "*   **4:** 1 time\n*   **5:** 1 time\n*   **6:** 1 time\n*   **7:** 1 time\n"
            "*   **8:** 1 time\n*   **9:** 1 time",
        ]
    )

    assert "partial (inline_list)" in report
    assert "unparsed numerals=[1]" in report


def test_report_lists_revisions_when_present(monkeypatch):
    sentinel_result = TallyAudit(
        frame_results=[],
        revisions=[RevisionEvent(numeral=4, from_frame_idx=1, to_frame_idx=2, from_value=2, to_value=3)],
        final_frame_arithmetically_consistent=None,
    )
    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", lambda frames: sentinel_result)

    node = DGemmaTallyAudit()
    (report,) = node.audit(frames=[])

    assert "numeral 4: 2 -> 3 (step 1 -> 2)" in report


def test_report_says_none_observed_when_no_revisions(monkeypatch):
    sentinel_result = TallyAudit(frame_results=[], revisions=[], final_frame_arithmetically_consistent=None)
    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", lambda frames: sentinel_result)

    node = DGemmaTallyAudit()
    (report,) = node.audit(frames=[])

    assert "revisions: none observed" in report


def test_report_final_consistency_true(monkeypatch):
    sentinel_result = TallyAudit(frame_results=[], revisions=[], final_frame_arithmetically_consistent=True)
    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", lambda frames: sentinel_result)

    node = DGemmaTallyAudit()
    (report,) = node.audit(frames=[])

    assert "arithmetically consistent" in report
    assert "INCONSISTENT" not in report


def test_report_final_consistency_false(monkeypatch):
    sentinel_result = TallyAudit(frame_results=[], revisions=[], final_frame_arithmetically_consistent=False)
    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", lambda frames: sentinel_result)

    node = DGemmaTallyAudit()
    (report,) = node.audit(frames=[])

    assert "INCONSISTENT" in report


def test_report_final_consistency_none(monkeypatch):
    sentinel_result = TallyAudit(frame_results=[], revisions=[], final_frame_arithmetically_consistent=None)
    monkeypatch.setattr("surfaces.comfyui.tally_audit.audit_frames", lambda frames: sentinel_result)

    node = DGemmaTallyAudit()
    (report,) = node.audit(frames=[])

    assert "no parseable claim" in report


class TestEndToEndRealFixtures:
    """One unmocked pass per real fixture (mirrors
    `test_trace_node.py::test_render_end_to_end_scaled_image_dimensions`):
    real `consumers.tally_audit.audit_frames` output, real extracted
    frames, through the actual node body."""

    def test_run1_end_to_end_reports_consistent_final_tally(self):
        blob = (FIXTURES_DIR / "count_numerals_2026-07-15T23-57-39_0000.txt").read_text(encoding="utf-8")
        frames = extract_decoded_frames_from_composite_blob(blob)

        node = DGemmaTallyAudit()
        (report,) = node.audit(frames=frames)

        assert "frames=12" in report
        assert "arithmetically consistent" in report

    def test_run2_end_to_end_reports_the_real_revision_and_consistent_final_tally(self):
        blob = (FIXTURES_DIR / "count_numerals_2026-07-15T23-59-14_0000.txt").read_text(encoding="utf-8")
        frames = extract_decoded_frames_from_composite_blob(blob)

        node = DGemmaTallyAudit()
        (report,) = node.audit(frames=frames)

        assert "frames=17" in report
        assert "numeral 3: 1 -> 2" in report
        assert "arithmetically consistent" in report
