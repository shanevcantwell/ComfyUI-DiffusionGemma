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
# Issue #86's third format: plain dash-bullet list. Two files from the same
# 2026-07-16 sweep — 0000 has a bolded `**Sum of appearances:**` header,
# 0009 has the same header unbolded; both have plain (unbolded) `Row N:`
# evidence labels. 0009 is selected specifically for its one real revision
# event (numeral 3, claimed 4->3, frames 6->8 — see tests/fixtures/README.md).
RUN3_PATH = FIXTURES_DIR / "count_numerals_2026-07-16T00-36-18_0000.txt"
RUN3_REVISION_PATH = FIXTURES_DIR / "count_numerals_2026-07-16T00-36-18_0009.txt"


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

    def test_run3_extracts_ten_frames_matching_steps_header(self):
        """`steps=10` in the header; third real fixture (issue #86's
        dash-bullet format, file 0000)."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_PATH))
        assert len(frames) == 10

    def test_run3_revision_fixture_extracts_ten_frames_matching_steps_header(self):
        """`steps=10` in the header; the revision-bearing sibling (file
        0009) from the same sweep."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_REVISION_PATH))
        assert len(frames) == 10

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


class TestParseDashBulletListFormat:
    """Issue #86's third observed shape: plain dash-bullets, one numeral
    per line, `- N: v` — no `**` bold wrap anywhere, no pipe table."""

    def test_well_formed_dash_bullet_list_parses_ok(self):
        text = (
            "Sum of appearances:\n"
            "- 0: 2\n- 1: 3\n- 2: 3\n- 3: 3\n- 4: 3\n"
            "- 5: 3\n- 6: 3\n- 7: 2\n- 8: 2\n- 9: 2"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.parse_status == "ok"
        assert result.format_name == "dash_bullet_list"
        assert result.claimed_counts() == {
            0: 2, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 2, 8: 2, 9: 2,
        }

    def test_bolded_header_above_the_list_does_not_prevent_dash_matching(self):
        """File 0000's real shape: the section header IS bolded
        (`**Sum of appearances:**`) even though the bullets themselves are
        never `**`-wrapped — the matcher must key on the bullet structure
        alone, not on the header's own bolding."""
        text = (
            "**Sum of appearances:**\n"
            "- 0: 2\n- 1: 3\n- 2: 3\n- 3: 3\n- 4: 3\n"
            "- 5: 3\n- 6: 3\n- 7: 2\n- 8: 2\n- 9: 2"
        )
        result = parse_tally_frame(text, frame_idx=0)
        assert result.parse_status == "ok"
        assert result.format_name == "dash_bullet_list"

    def test_garbage_value_cell_demotes_only_that_cell_to_partial(self):
        """DECISION F-2's per-numeral-cell granularity, third format:
        mirrors the inline-list/pipe-table garbage-value tests."""
        text = (
            "- 0: 2\n- 1: 3\n- 2: 3\n- 3:  ratings\n- 4: 3\n"
            "- 5: 3\n- 6: 3\n- 7: 2\n- 8: 2\n- 9: 2"
        )
        result = parse_tally_frame(text, frame_idx=0)

        assert result.parse_status == "partial"
        assert result.format_name == "dash_bullet_list"
        assert result.cells[3].claimed is None
        assert result.cells[3].raw_value == "ratings"
        assert result.cells[0].claimed == 2
        assert result.cells[9].claimed == 2

    def test_digit_embedded_in_a_garbage_token_is_not_read_as_the_value(self):
        """Observed live: `- 6: <unused4981>` is a special-token literal,
        not a claimed value of 4981 — the leading-integer anchor must not
        pick a digit out of the MIDDLE of an unrelated garbage token (this
        is what distinguishes the dash-bullet matcher's value extraction
        from a bare anywhere-in-string digit search)."""
        text = (
            "- 0: 2\n- 1: 3\n- 2:  Demons\n- 3:  ratings\n- 4:  nếu\n"
            "- 5:  repeater\n- 6: <unused4981>\n- 7: 2\n- 8: 2\n- 9: 2"
        )
        result = parse_tally_frame(text, frame_idx=0)
        assert result.cells[6].claimed is None
        assert result.cells[6].raw_value == "<unused4981>"

    def test_trailing_noise_after_a_clean_value_still_parses(self):
        """Observed live: the final bullet in a frame frequently runs
        straight into the next fragment's decode noise with no separator
        (`- 9: 2 然而DONE wikip`) — `2` is the model's real claimed value;
        the leading-integer anchor must not demote it to garbage purely
        because of what follows on the same (delimiter-less) line."""
        text = (
            "- 0: 2\n- 1: 3\n- 2: 3\n- 3: 3\n- 4: 3\n"
            "- 5: 3\n- 6: 3\n- 7: 2\n- 8: 2\n- 9: 2 然而DONE wikip"
        )
        result = parse_tally_frame(text, frame_idx=0)
        assert result.cells[9].claimed == 2
        assert result.parse_status == "ok"

    def test_garbage_numeral_token_is_skipped_not_recorded(self):
        """A `- <token>: <value>` line whose numeral token isn't a bare
        digit (e.g. `- isHidden:  organiser`, observed live) is not a
        numeral-tally row at all — skipped, mirroring the other two
        matchers' garbage-numeral handling."""
        text = "- 0: 2\n- isHidden:  organiser\n- 7: 2"
        result = parse_tally_frame(text, frame_idx=0)
        assert set(result.cells) == {0, 7}

    def test_missing_numerals_is_partial_not_ok(self):
        text = "- 0: 2\n- 1: 3"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.parse_status == "partial"
        assert set(result.cells) == {0, 1}

    def test_no_claimed_total_field_for_this_format(self):
        """No `Total`-shaped line has been observed in the dash-bullet
        format (unlike the other two) — `claimed_total` is honestly `None`,
        never inferred from the ten cells."""
        text = "- 0: 2\n- 1: 3\n- 2: 3\n- 3: 3\n- 4: 3\n- 5: 3\n- 6: 3\n- 7: 2\n- 8: 2\n- 9: 2"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.claimed_total is None

    def test_dash_bullet_matcher_never_shadows_inline_list_or_pipe_table(self):
        """Registry-ordering regression: the inline-list's `*   **N:**`
        bold-bullet anchor and the pipe-table's `| N | v |` structure must
        still win over the dash-bullet matcher when present, since matcher
        order in `parse_tally_frame` tries inline-list and pipe-table
        first."""
        inline_text = "*   **0:** 1 time\n*   **1:** 1 time"
        assert parse_tally_frame(inline_text, frame_idx=0).format_name == "inline_list"

        table_text = "| Numeral | Frequency |\n| :--- | :--- |\n| 0 | 2 |\n| 1 | 3 |"
        assert parse_tally_frame(table_text, frame_idx=0).format_name == "pipe_table"


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

    def test_random_dash_bullets_with_no_numeral_colon_shape_are_unrecognized(self):
        """Dash bullets alone are not enough — they must also carry the
        `<token>: <value>` shape the dash-bullet matcher anchors on."""
        text = "- first item, no colon shape\n- second item same\n- third, still no match"
        result = parse_tally_frame(text, frame_idx=0)
        assert result.parse_status == "unrecognized"

    def test_an_artificial_fourth_format_is_unrecognized(self):
        """AC#2/AC#4: a fourth structural shape this module has never seen
        (neither bold-bullet, nor pipe-table, nor dash-bullet) must still
        return `unrecognized` + the raw excerpt — the honest-failure path
        is retained, not narrowed away by adding the third matcher."""
        text = (
            "Tally results (semicolon-separated, a shape none of the three "
            "registered matchers recognize):\n"
            "0=2; 1=3; 2=3; 3=3; 4=3; 5=3; 6=3; 7=2; 8=2; 9=2"
        )
        result = parse_tally_frame(text, frame_idx=5)

        assert result.parse_status == "unrecognized"
        assert result.format_name is None
        assert result.cells == {}
        assert result.claimed_counts() == {}
        assert result.raw_excerpt == text[:500]


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

    def test_issue_86_style_unbolded_row_labels(self):
        """Issue #86's third format: `Row k:` labels with NO `**` wrap at
        all (unlike run 2's `**Row k:**`) — the same label-then-plain-list
        shape, just unbolded end to end."""
        text = (
            "Row 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3\n"
            "Row 2: 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6"
        )
        counts = count_evidence_numerals(text)
        assert counts == {0: 2, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 2, 8: 2, 9: 2}

    def test_unbolded_row_label_with_dash_bullet_total_does_not_double_count(self):
        """Mirrors the bolded-Total regression test above, for the third
        format: a `- **Total**: N`-shaped stray bold span (not itself
        observed, but the same class of risk) must not double-count
        alongside unbolded `Row k:` evidence lines."""
        text = (
            "Row 1: 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3\n"
            "Row 2: 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6\n\n"
            "Sum of appearances:\n- 0: 2\n- 1: 3\n"
        )
        counts = count_evidence_numerals(text)
        assert sum(counts.values()) == 26

    def test_bolded_row_label_still_matches_after_making_the_wrap_optional(self):
        """Regression: loosening `_EVIDENCE_ROW_LABEL_RE` to tolerate an
        unbolded label must not break the original bolded run-2 shape."""
        text = "**Row 1:** 4, 7, 2, 9, 0, 5, 4, 8, 2, 7, 1, 6, 9"
        counts = count_evidence_numerals(text)
        assert counts == {0: 1, 1: 1, 2: 2, 3: 0, 4: 2, 5: 1, 6: 1, 7: 2, 8: 1, 9: 2}


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

    def test_real_run3_fixture_revision_3_4_to_3_bridges_the_garbage_frame(self):
        """Issue #86's grounding datum: file 0009 contains the sweep's only
        revision — numeral 3, claimed 4 (frame 6) then a garbage/
        unparseable value cell (frame 7, `玖`) then re-parses cleanly as 3
        (frame 8). Per DECISION F-2's per-numeral-cell granularity, the
        watcher must skip frame 7's unparseable cell-3 and bridge the
        comparison 6->8 directly — not report a spurious 6->7 or 7->8
        event, and not miss the revision entirely."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_REVISION_PATH))
        audit = audit_frames(frames)

        matching = [e for e in audit.revisions if e.numeral == 3]
        assert len(matching) == 1
        assert matching[0].from_frame_idx == 6
        assert matching[0].to_frame_idx == 8
        assert matching[0].from_value == 4
        assert matching[0].to_value == 3
        # Frame 7's cell 3 genuinely failed to parse — confirms the bridge
        # is happening BECAUSE of the per-cell skip, not by coincidence.
        assert audit.frame_results[7].cells[3].claimed is None


# ---------------------------------------------------------------------------
# Top-level `audit_frames` — including the real fixtures end-to-end
# ---------------------------------------------------------------------------


class TestAuditFramesRealFixtures:
    """AC#1: all real-run fixtures parse `ok` at final step and are judged
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

    def test_run3_final_frame_parses_ok_and_is_consistent(self):
        """Issue #86 AC: the new dash-bullet-format fixture (bolded
        `**Sum of appearances:**` header, file 0000) parses `ok` at its
        final step and is arithmetically consistent."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_PATH))
        audit = audit_frames(frames)

        assert audit.frame_results[-1].format_name == "dash_bullet_list"
        assert audit.frame_results[-1].parse_status == "ok"
        assert audit.final_frame_arithmetically_consistent is True

    def test_run3_revision_fixture_final_frame_parses_ok_and_is_consistent(self):
        """Issue #86 AC: the revision-bearing fixture (file 0009, unbolded
        `Sum of appearances:` header) also parses `ok` at its final step."""
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_REVISION_PATH))
        audit = audit_frames(frames)

        assert audit.frame_results[-1].format_name == "dash_bullet_list"
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

    def test_run3_frame_results_length_matches_frame_count(self):
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_PATH))
        audit = audit_frames(frames)
        assert len(audit.frame_results) == len(frames) == 10

    def test_run3_revision_fixture_frame_results_length_matches_frame_count(self):
        frames = extract_decoded_frames_from_composite_blob(_read(RUN3_REVISION_PATH))
        audit = audit_frames(frames)
        assert len(audit.frame_results) == len(frames) == 10


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
