"""`consumers/tally_audit.py` — issue #84: audit a "count the numerals" task's
decoded frames against the model's own restated evidence. No ComfyUI, no
torch-autograd/pipeline dependency (ADR-CDG-003): the extractor and matchers
operate on plain strings; only `dgemma.types` is imported from the core.

Fixtures (`tests/fixtures/count_numerals_*.txt`) are byte-identical real run
output (issue #84's non-mocked-fixtures convention; see
`tests/fixtures/README.md` for provenance) — composite blobs, not
`decode_frames()` output directly (DECISION F-1), so every fixture-driven
test below extracts first.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from consumers.tally_audit import (
    CompositeBlobExtractionError,
    FrameAuditResult,
    NumeralCellResult,
    RevisionEvent,
    TallyAudit,
    audit_frames,
    count_evidence_numerals,
    extract_decoded_frames_from_composite_blob,
    parse_tally_frame,
    watch_revisions,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
RUN1_PATH = FIXTURES_DIR / "count_numerals_2026-07-15T23-57-39_0000.txt"  # inline bold-markdown list
RUN2_PATH = FIXTURES_DIR / "count_numerals_2026-07-15T23-59-14_0000.txt"  # GFM pipe table


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# F-1: composite-blob extraction
# ---------------------------------------------------------------------------


class TestExtractDecodedFramesFromCompositeBlob:
    """DECISION F-1: the on-disk fixtures are composite trace+escaped-newline
    blobs, not `decode_frames()` output — the extractor must split header
    from frames and fail honestly on a shape it doesn't recognize."""

    def test_run1_extracts_twelve_frames_matching_steps_header(self):
        """`steps=12` in the header; splitting must yield exactly 12 frames
        (the header itself is dropped, not counted as frame 0)."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN1_PATH))
        assert len(frames) == 12

    def test_run2_extracts_seventeen_frames_matching_steps_header(self):
        """`steps=17` in the header; same shape, second real fixture."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN2_PATH))
        assert len(frames) == 17

    def test_extracted_frames_do_not_contain_the_header(self):
        """The header's own scheduler/steps/committed_fraction lines must not
        leak into frame 0 — proves the split point, not just the count."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN1_PATH))
        assert "scheduler=EntropyBoundScheduler" not in frames[0]
        assert "committed_fraction per step" not in frames[0]

    def test_last_frame_ends_with_the_real_final_tally(self):
        """Sanity-anchors the split against ground truth read directly off
        the fixture: run 1's final frame is the fully-resolved 13-numeral
        answer."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN1_PATH))
        assert "*(Total count: 13)*" in frames[-1]

    def test_no_delimiter_at_all_raises_honestly(self):
        """A blob with no `thought`-frame delimiter is not "zero frames
        captured" — it's an unrecognized shape (DECISION F-1's honest-failure
        clause)."""
        with pytest.raises(CompositeBlobExtractionError, match="no frame delimiter"):
            extract_decoded_frames_from_composite_blob("just some plain text, no structure at all")

    def test_empty_string_raises_honestly(self):
        with pytest.raises(CompositeBlobExtractionError):
            extract_decoded_frames_from_composite_blob("")

    def test_header_only_blob_yields_one_empty_frame_not_a_crash(self):
        """The delimiter present with nothing following it: `str.split`
        guarantees at least one (empty-string) part after the header — this
        is a degenerate but honest result (an empty frame the caller's own
        parser will separately report as `unrecognized`), not a second
        failure mode distinct from "delimiter absent entirely"."""
        blob = "some header text\\n\\nthought\n"
        frames = extract_decoded_frames_from_composite_blob(blob)
        assert frames == [""]


# ---------------------------------------------------------------------------
# F-2: format matchers — structure-keyed, per-numeral-cell granularity
# ---------------------------------------------------------------------------


class TestParseInlineListFormat:
    """Run 1's shape: `*   **N:** k time(s)` bullets."""

    def test_well_formed_inline_list_parses_ok(self):
        text = (
            "Here is a set of 13 individual numerals:\n\n"
            "**4, 7, 2, 4, 9, 1, 7, 4, 5, 2, 0, 7, 6**\n\n"
            "**Sum of appearances for each numeral:**\n\n"
            "*   **0:** 1 time\n*   **1:** 1 time\n*   **2:** 2 times\n"
            "*   **3:** 0 times\n*   **4:** 3 times\n*   **5:** 1 time\n"
            "*   **6:** 1 time\n*   **7:** 3 times\n*   **8:** 0 times\n"
            "*   **9:** 1 time\n\n*(Total count: 13)*"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.parse_status == "ok"
        assert result.format_name == "inline_list"
        assert result.claimed_counts() == {
            0: 1, 1: 1, 2: 2, 3: 0, 4: 3, 5: 1, 6: 1, 7: 3, 8: 0, 9: 1,
        }
        assert result.claimed_total == 13

    def test_singular_time_at_k_equals_1_still_parses(self):
        """DECISION F-2: the literal shape is `*   **N:** k time(s)` — "time"
        singular at k=1, not always "times" — the matcher must not assume
        the plural form."""
        text = "*   **1:** 1 time"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.cells[1].claimed == 1

    def test_garbage_value_cell_demotes_only_that_cell_to_partial(self):
        """DECISION F-2's per-numeral-cell granularity: one garbage value
        (`luc times` instead of an int) must not fail the whole frame."""
        text = (
            "*   **0:** 1 time\n*   **1:** 1 time\n*   **2:** 2 times\n"
            "*   **3:** 0 times\n*   **4:** 3 times\n*   **5:** 1 time\n"
            "*   **6:** 1 time\n*   **7:**  MożDropout\n*   **8:** 0 times\n"
            "*   **9:** 1 time"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.parse_status == "partial"
        assert result.format_name == "inline_list"
        assert result.cells[7].claimed is None
        assert result.cells[7].raw_value == "MożDropout"
        # Every other cell still parsed cleanly.
        assert result.cells[0].claimed == 1
        assert result.cells[9].claimed == 1

    def test_missing_numerals_is_partial_not_ok(self):
        """An in-progress frame that hasn't reached every numeral yet (e.g.
        early in the run) is `partial`, not `ok` — `ok` requires all ten."""
        text = "*   **0:** 1 time\n*   **1:** 1 time"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.parse_status == "partial"
        assert set(result.cells) == {0, 1}


class TestParsePipeTableFormat:
    """Run 2's shape: `| Numeral | <label> |` GFM table — matchers key on
    structure, never the header label (DECISION F-2)."""

    def test_well_formed_pipe_table_parses_ok(self):
        text = (
            "### Sum of Appearances:\n\n"
            "| Numeral | Frequency |\n| :--- | :--- |\n"
            "| 0 | 2 |\n| 1 | 3 |\n| 2 | 3 |\n| 3 | 2 |\n| 4 | 3 |\n"
            "| 5 | 3 |\n| 6 | 2 |\n| 7 | 3 |\n| 8 | 2 |\n| 9 | 3 |\n"
            "| **Total** | **26** |"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.parse_status == "ok"
        assert result.format_name == "pipe_table"
        assert result.claimed_counts() == {
            0: 2, 1: 3, 2: 3, 3: 2, 4: 3, 5: 3, 6: 2, 7: 3, 8: 2, 9: 3,
        }
        assert result.claimed_total == 26

    def test_garbage_header_label_does_not_cause_unrecognized(self):
        """DECISION F-2's central precision finding: the header cell itself
        is frequently garbage (`ratings`/`Crum` observed live) — a matcher
        keyed on the label string would spuriously reject this well-formed
        table. Structure (pipe-delimited + separator row) is what matters."""
        text = (
            "| Numeral | ratings |\n| :--- | :--- |\n"
            "| 0 |  insign |\n| 1 |  maggio |\n| 2 | hältnis |\n| 3 | wo |\n"
            "| 4 |  scène |\n| 5 |  гла |\n| 6 | ອກ |\n| 7 |  conveniently |\n"
            "| 8 |  Royalty |\n| 9 | aqua |"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.format_name == "pipe_table"  # recognized as a table at all
        assert result.parse_status != "unrecognized"

    def test_garbage_value_cell_demotes_only_that_cell_to_partial(self):
        """Mirrors the inline-list per-cell test: a garbage VALUE cell
        (`| 0 |  التس |`) is a `partial` cell, not a whole-frame rejection —
        the row/table STRUCTURE around it is still legible."""
        text = (
            "| Numeral | Frequency |\n| :--- | :--- |\n"
            "| 0 |  التس |\n| 1 | 3 |\n| 2 | домо |\n| 3 | 1 |\n| 4 | ద్ |\n"
            "| 5 | stå |\n| 6 | 制造 |\n| 7 | August |\n| 8 | apsingToolbar |\n"
            "| 9 | 不禁 |"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.parse_status == "partial"
        assert result.format_name == "pipe_table"
        assert result.cells[0].claimed is None
        assert result.cells[0].raw_value == "التس"
        assert result.cells[1].claimed == 3

    def test_garbage_numeral_cell_is_skipped_not_recorded(self):
        """DECISION F-2: a numeral cell that is itself garbage (e.g. `章`
        instead of a digit) is not a numeral-tally row at all — skipped,
        not recorded as an unparseable numeral (there is no numeral key to
        record it under)."""
        text = (
            "| Numeral | ratings |\n| :--- | :--- |\n"
            "| 0 |  insign |\n|  章 |  scène |\n|  ă |  гла |\n| 6 | ອກ |"
        )
        result = parse_tally_frame(text, frame_idx=0)
        assert set(result.cells) == {0, 6}

    def test_junk_before_first_pipe_does_not_break_row_matching(self):
        """Observed live: heading text runs straight into the first `|` with
        no space (`### Sum of Appearances:пиона| Numeral | Crum |`) — the
        row matcher must match the pipe-delimited cells regardless of what
        precedes them on the line."""
        text = (
            "### Sum of Appearances:пиона| Numeral | Crum |\n"
            "| :--- | :--- |\n"
            "| 0 | RequestBody |\n| 1 |  دھو |\n| 2 | 𒄈 |\n| 3 | ເພ |\n"
            "| 4 |  पहनने |\n| 5 |  Nc |\n| 6 |  Faktoren |\n| 7 |  মস্ক |\n"
            "| 8 | bigcup |\n| 9 | कार्ट |\n| **Total** | **26** | दिसंबर"
        )
        result = parse_tally_frame(text, frame_idx=0)
        assert result.format_name == "pipe_table"
        assert result.cells[0].claimed is None  # "RequestBody" is garbage
        assert result.cells[3].claimed is None  # "ເພ" is garbage


class TestUnrecognizedFormat:
    """AC#2: an artificial unknown-format frame yields `unrecognized` +
    excerpt, never a fabricated count."""

    def test_prose_with_no_structural_anchor_is_unrecognized(self):
        text = "The model thinks about numerals but never actually tallies anything here."
        result = parse_tally_frame(text, frame_idx=3)

        assert result.parse_status == "unrecognized"
        assert result.format_name is None
        assert result.cells == {}
        assert result.claimed_counts() == {}
        assert result.raw_excerpt  # the raw text is kept, not silently dropped

    def test_unrecognized_frame_never_reports_a_fabricated_zero(self):
        """The central honesty guarantee: an unrecognized frame's
        `claimed_counts()` is empty, never all-zero (a `0` for every numeral
        would be indistinguishable from ten genuine zero-claims)."""
        result = parse_tally_frame("completely unrelated text", frame_idx=0)
        assert result.claimed_counts() == {}

    def test_a_bare_table_like_prose_with_no_separator_row_is_unrecognized(self):
        """Pipe characters alone (no `| :--- | :--- |` structural marker) are
        not enough to call this a table — avoids false-positiving on
        incidental `|` usage in prose."""
        text = "some | random | text | with | pipes | but no separator row"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.parse_status == "unrecognized"

    def test_random_bullets_with_no_bold_numeral_are_unrecognized(self):
        text = "* first item\n* second item\n* third item, no numerals here"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.parse_status == "unrecognized"


# ---------------------------------------------------------------------------
# Evidence counter
# ---------------------------------------------------------------------------


class TestCountEvidenceNumerals:
    """Operator requirement (c): procedurally count numerals in the model's
    own restated evidence, not trust the claimed tally."""

    def test_run1_style_single_bold_list(self):
        text = "**4, 7, 2, 4, 9, 1, 7, 4, 5, 2, 0, 7, 6**"
        counts = count_evidence_numerals(text)
        assert counts == {0: 1, 1: 1, 2: 2, 3: 0, 4: 3, 5: 1, 6: 1, 7: 3, 8: 0, 9: 1}

    def test_run2_style_two_row_labels(self):
        text = (
            "**Row 1:** 4, 7, 2, 9, 0, 5, 4, 8, 2, 7, 1, 6, 9\n"
            "**Row 2:** 5, 3, 8, 1, 7, 2, 6, 4, 9, 1, 5, 3, 0"
        )
        counts = count_evidence_numerals(text)
        assert counts == {0: 2, 1: 3, 2: 3, 3: 2, 4: 3, 5: 3, 6: 2, 7: 3, 8: 2, 9: 3}

    def test_garbage_tokens_between_commas_are_not_counted_as_numerals(self):
        """A multi-character or non-digit token (decode noise) is honestly
        NOT a numeral — never coerced into one just because it sits in a
        comma-separated evidence list."""
        text = "**4, hely, க்கவும், iprop, Bamboo**"
        counts = count_evidence_numerals(text)
        assert counts[4] == 1
        assert sum(counts.values()) == 1

    def test_a_total_row_does_not_pollute_the_row_labeled_evidence_count(self):
        """When a `**Row k:**`-labeled evidence line is present, a stray bold
        span elsewhere in the same frame (the pipe table's own
        `| **Total** | **26** |` row) must not be double-counted as a second
        evidence list — regression test for the bug this function's first
        draft actually had."""
        text = (
            "**Row 1:** 4, 7, 2, 9, 0, 5, 4, 8, 2, 7, 1, 6, 9\n"
            "**Row 2:** 5, 3, 8, 1, 7, 2, 6, 4, 9, 1, 5, 3, 0\n\n"
            "| Numeral | Frequency |\n| :--- | :--- |\n"
            "| 0 | 2 |\n| **Total** | **26** |"
        )
        counts = count_evidence_numerals(text)
        assert sum(counts.values()) == 26  # not inflated by the Total row


# ---------------------------------------------------------------------------
# Revision watcher
# ---------------------------------------------------------------------------


class TestWatchRevisions:
    def _result(self, frame_idx: int, claimed: dict[int, int]) -> FrameAuditResult:
        return FrameAuditResult(
            frame_idx=frame_idx,
            parse_status="ok",
            format_name="pipe_table",
            cells={n: NumeralCellResult(numeral=n, claimed=v, raw_value=str(v)) for n, v in claimed.items()},
        )

    def test_synthetic_4_2_to_3_event_reproduced(self):
        """AC#3: the known `4: 2→3`-shaped event (docs/experiments/
        2026-07-15-dg-numeral-counts-update-in-response) on a synthetic
        fixture."""
        frames = [
            self._result(0, {4: 2, 7: 3}),
            self._result(1, {4: 2, 7: 3}),
            self._result(2, {4: 3, 7: 3}),
        ]
        events = watch_revisions(frames)

        assert len(events) == 1
        assert events[0] == RevisionEvent(numeral=4, from_frame_idx=1, to_frame_idx=2, from_value=2, to_value=3)

    def test_no_change_yields_no_events(self):
        frames = [self._result(0, {0: 1}), self._result(1, {0: 1}), self._result(2, {0: 1})]
        assert watch_revisions(frames) == []

    def test_unrecognized_frames_are_skipped_not_diffed(self):
        """An unrecognized frame contributes no claimed values — it must
        neither start nor end a revision comparison."""
        frames = [
            self._result(0, {4: 2}),
            FrameAuditResult(frame_idx=1, parse_status="unrecognized", format_name=None),
            self._result(2, {4: 3}),
        ]
        events = watch_revisions(frames)
        # frame 1 is skipped entirely, so the comparison is 0 -> 2 directly.
        assert events == [RevisionEvent(numeral=4, from_frame_idx=0, to_frame_idx=2, from_value=2, to_value=3)]

    def test_a_numeral_absent_from_one_side_is_not_a_revision(self):
        """A numeral only present in one of the two frames (not yet reached,
        or a garbage cell in the other) must not register as a revision
        to/from a missing value."""
        frames = [self._result(0, {4: 2}), self._result(1, {7: 3})]  # disjoint numerals
        assert watch_revisions(frames) == []

    def test_real_run2_fixture_revision_3_1_to_2(self):
        """DECISION F-3 (design-gate ratification): the naturally occurring
        revision the real run-2 fixture actually contains — `3: 1→2` at the
        final step — proven against real data, not only the synthetic
        `4:2→3` fixture above."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN2_PATH))
        audit = audit_frames(frames)

        matching = [e for e in audit.revisions if e.numeral == 3]
        assert len(matching) == 1
        assert matching[0].from_value == 1
        assert matching[0].to_value == 2
        assert matching[0].to_frame_idx == len(frames) - 1  # the final captured step


# ---------------------------------------------------------------------------
# Top-level `audit_frames` — including the real fixtures end-to-end
# ---------------------------------------------------------------------------


class TestAuditFramesRealFixtures:
    """AC#1: both real-run fixtures parse `ok` at final step and are judged
    arithmetically consistent."""

    def test_run1_final_frame_parses_ok_and_is_consistent(self):
        frames = extract_decoded_frames_from_composite_blob(_read(RUN1_PATH))
        audit = audit_frames(frames)

        assert audit.frame_results[-1].parse_status == "ok"
        assert audit.final_frame_arithmetically_consistent is True

    def test_run2_final_frame_parses_ok_and_is_consistent(self):
        frames = extract_decoded_frames_from_composite_blob(_read(RUN2_PATH))
        audit = audit_frames(frames)

        assert audit.frame_results[-1].parse_status == "ok"
        assert audit.final_frame_arithmetically_consistent is True

    def test_run1_early_frames_are_unrecognized_not_fabricated(self):
        """Early frames (pure noise, no tally structure yet) must be
        `unrecognized`, never a fabricated all-zero claim."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN1_PATH))
        audit = audit_frames(frames)

        assert audit.frame_results[0].parse_status == "unrecognized"
        assert audit.frame_results[0].claimed_counts() == {}

    def test_run1_frame_results_length_matches_frame_count(self):
        frames = extract_decoded_frames_from_composite_blob(_read(RUN1_PATH))
        audit = audit_frames(frames)
        assert len(audit.frame_results) == len(frames) == 12

    def test_run2_frame_results_length_matches_frame_count(self):
        frames = extract_decoded_frames_from_composite_blob(_read(RUN2_PATH))
        audit = audit_frames(frames)
        assert len(audit.frame_results) == len(frames) == 17


class TestAuditFramesDegenerateInput:
    def test_empty_list_yields_empty_audit_not_a_crash(self):
        audit = audit_frames([])

        assert audit == TallyAudit(frame_results=[], revisions=[], final_frame_arithmetically_consistent=None)

    def test_all_unrecognized_frames_yield_none_consistency_not_false(self):
        """`None` (no claim to check) is a different condition from `False`
        (checked and found inconsistent) — an all-unrecognized run must
        report the former."""
        audit = audit_frames(["no structure here", "still nothing recognizable"])
        assert audit.final_frame_arithmetically_consistent is None

    def test_final_frame_partial_with_no_parsed_cells_yields_none(self):
        """A final frame that matched a format's structural anchor but has
        zero cleanly-parsed cells (e.g. every value garbage) has nothing to
        check consistency against either — `None`, not `False`."""
        audit = audit_frames(["* first item\n* second item, no numerals here"])
        assert audit.frame_results[-1].parse_status == "unrecognized"
        assert audit.final_frame_arithmetically_consistent is None

    def test_inconsistent_final_claim_is_reported_false(self):
        """A final frame whose claimed count disagrees with the true
        evidence-derived count must report `False`, not silently pass."""
        text = (
            "**4, 7**\n\n"
            "*   **0:** 0 times\n*   **1:** 0 times\n*   **2:** 0 times\n"
            "*   **3:** 0 times\n*   **4:** 5 times\n*   **5:** 0 times\n"
            "*   **6:** 0 times\n*   **7:** 5 times\n*   **8:** 0 times\n"
            "*   **9:** 0 times"
        )
        audit = audit_frames([text])
        assert audit.frame_results[-1].parse_status == "ok"
        assert audit.final_frame_arithmetically_consistent is False
