"""consumers/tally_audit.py — pure functions auditing a "count the numerals"
task's decoded frames against the model's own restated evidence (issue #84).

Consumer tier per ADR-CDG-008 Open Question #1 (settled `consumers/`) and the
same discipline as `consumers/analysis.py`: this module parses already-decoded
per-step strings — it never wraps `load_model`/`run_diffusion`, never drives
the model, and imports only `dgemma.types` from the core (nothing else). It
lives outside `dgemma/`'s import graph (`tests/test_seam.py`'s subprocess
leak-check covers `consumers/*`, this module included by name).

**What this module does NOT do:** infer a claimed count from a format it does
not recognize. `EMIT-CANONICAL / PARSE-AT-THE-DOOR` (ARCHITECTURE.md rule 5)
applied to model-generated markdown, not a data-plane socket: an unrecognized
frame format is REPORTED (`parse_status="unrecognized"` + the raw excerpt),
never fabricated as a plausible-looking zero or blank. The two observed
formats (issue #84's grounding — the two `count_numerals_*` runs under
`/srv/dev/ComfyUI/output/`, format differed between consecutive runs) are:

1. **Inline bold-markdown list** (run 1, 2026-07-15T23-57-39): repeated
   `*   **N:** k time(s)` lines (the literal text says "time" singular at
   k=1, "times" plural otherwise — but the matcher below keys on structure,
   not the label word, per the design-gate's DECISION F-2).
2. **GFM pipe table** (run 2, 2026-07-15T23-59-14): `| Numeral | <label> |`
   header (the header's own label cell is frequently a garbage token —
   `Frequency`, `ratings`, `Crum` all observed live — so matchers must NOT
   key on that label string), a `| :--- | :--- |` separator, ten `| N | v |`
   body rows, and an optional `| **Total** | **T** |` row.

Per DECISION F-2 (design-gate ratification, issue #84): matchers key on
*structure* (pipe-delimited rows in table form; bullet-prefixed
bold-numeral lines in list form), not on header/label strings, and grant
per-numeral-cell granularity — a single garbage VALUE cell (e.g.
`| 0 |  التس |`) demotes only that cell to unparsed, not the whole frame to
`unrecognized`, since the row/list structure around it is still legible.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# Dual-context import, same discipline and same depth as
# `consumers/analysis.py` (see that module's docstring for the full
# rationale) — `consumers/` sits one level under the pack root, same depth
# as `dgemma/` itself, so the relative climb is exactly two dots.
if __package__ and "." in __package__:
    from ..dgemma.types import DiffusionFrame
else:
    from dgemma.types import DiffusionFrame


# ---------------------------------------------------------------------------
# F-1: composite-blob extraction (the fixture shape, design-gate DECISION F-1)
# ---------------------------------------------------------------------------

# The literal per-frame delimiter `decode_frames`'s raw (non-excised) decode
# leaves behind: an escaped `\n\n` (two literal backslash-n sequences — the
# on-disk artifact's own header/caption join convention, NOT a real newline)
# followed by the `thought` channel-label token and one REAL newline before
# the next frame's decoded text begins. See tests/fixtures/README.md for the
# byte-level provenance that grounds this literal.
_FRAME_DELIMITER = "\\n\\nthought\n"


class CompositeBlobExtractionError(ValueError):
    """Raised by `extract_decoded_frames_from_composite_blob` when `blob`
    does not contain the expected header/frame-delimiter shape — the
    honest-failure path DECISION F-1 requires: a malformed blob must fail
    naming what it could not split, never silently mis-split into wrong
    frame boundaries."""


def extract_decoded_frames_from_composite_blob(blob: str) -> list[str]:
    """Reverse a `DGemmaTrace`-style composite `.txt` artifact into the
    `list[str]` shape `audit_frames` (below) consumes.

    DECISION F-1 (design-gate ratification, issue #84): the on-disk
    `count_numerals_*.txt` step-logs are NOT `decode_frames()` output
    directly — each is a composite blob: a `_format_summary`-shaped header
    (timestamp+prompt joined by literal `\\n\\n` escapes, then
    `scheduler=…`/`steps=…`/`committed_fraction per step…`/`mask-token
    corroboration…` lines joined by real newlines) followed by the per-frame
    decoded texts `decode_frames` produced, each frame boundary marked by
    the literal delimiter this module names `_FRAME_DELIMITER`.

    Splitting on that delimiter and dropping part 0 (the header) yields
    exactly the per-frame `list[str]` `audit_frames` expects — this
    extractor is legacy-txt-format support (issue #72 will supersede it
    once the forward schema'd-JSONL emission path lands; this module is
    what #72 demotes txt-scraping to, not a competing parser).

    Raises `CompositeBlobExtractionError` (parse-at-the-door, honest
    failure) when the delimiter never appears at all — a blob with no
    frame boundary is not "zero frames captured", it is a shape this
    extractor does not recognize, and inventing an empty result would be
    exactly the lying-payload trap ADR-CDG-001 forbids applied to this
    legacy-format reader. (`str.split` on a delimiter that IS present
    always yields at least one part after the header, so "delimiter found
    but zero frames resulted" cannot occur separately from this check —
    there is one failure mode here, not two.)
    """
    if _FRAME_DELIMITER not in blob:
        raise CompositeBlobExtractionError(
            "extract_decoded_frames_from_composite_blob: no frame delimiter "
            f"({_FRAME_DELIMITER!r}) found in blob — not a recognized "
            "composite trace+escaped-newline artifact (DECISION F-1)."
        )
    parts = blob.split(_FRAME_DELIMITER)
    return parts[1:]  # part 0 is the header; drop it.


# ---------------------------------------------------------------------------
# F-2: format-matcher registry, per-numeral-cell granularity
# ---------------------------------------------------------------------------

# Structure-keyed, not label-keyed (DECISION F-2): the pipe-table header's
# own label cell is frequently garbage (`Frequency`/`ratings`/`Crum` all
# observed live in the run-2 fixture) — matching on `Numeral`/`Frequency`
# text would spuriously reject a well-formed table with a garbled header.
_INLINE_ROW_RE = re.compile(
    r"\*\s*\*\*\s*(?P<numeral>\S)\s*:?\s*\*\*:?\s*(?P<value>\S+)?\s*(?:times?|Total)?",
)
# A pipe-table body row: `| <numeral-cell> | <value-cell> |` — the leading
# `|` may be preceded by arbitrary garbage (the ### heading text frequently
# runs straight into the first `|` with no space, e.g.
# `### Sum of Appearances:пиона| Numeral | Crum |`), so this matches only
# the two-cell pipe form itself, not what precedes it on the line.
_TABLE_ROW_RE = re.compile(r"\|\s*(?P<numeral>[^|]*?)\s*\|\s*(?P<value>[^|]*?)\s*\|")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|\s*:?-+:?\s*\|\s*:?-+:?\s*\|\s*$")

# A numeral cell is "recognized" iff, after stripping markdown bold markers,
# it is exactly one ASCII digit 0-9 (garbage tokens like "章 "/" ă " never
# match) — or the literal `**Total**` marker.
_NUMERAL_CELL_RE = re.compile(r"^\*{0,2}([0-9])\*{0,2}$")
_TOTAL_CELL_RE = re.compile(r"^\*{0,2}Total\*{0,2}$", re.IGNORECASE)
# A value cell is "recognized" iff it reduces to a bare integer (markdown
# bold markers stripped) — garbage tokens (" التس ", " домо ") never match.
_VALUE_CELL_RE = re.compile(r"^\*{0,2}(-?[0-9]+)\*{0,2}$")


@dataclass
class NumeralCellResult:
    """One numeral's claimed-count cell, per-cell granular (DECISION F-2):
    a single garbage value cell must not fail the whole frame."""

    numeral: int
    """Which digit 0-9 this cell reports on."""

    claimed: int | None
    """The parsed claimed count, or `None` if this cell's value token did
    not reduce to an integer (garbage token — e.g. `| 0 |  التس |`)."""

    raw_value: str
    """The raw (unparsed) value token, always kept — even when `claimed`
    parsed cleanly — so a caller can display what the model actually wrote,
    not just the parsed int."""


@dataclass
class FrameAuditResult:
    """One frame's per-numeral tally read, plus the frame-level
    `parse_status` DECISION F-2 requires: `"ok"` (every numeral cell 0-9
    parsed AND ten distinct numerals were found), `"partial"` (the
    table/list structure was recognized but at least one numeral cell
    failed to parse, or fewer than ten distinct numerals were found),
    `"unrecognized"` (neither the inline-list nor pipe-table structure was
    found at all — REPORTED, never inferred, per ARCHITECTURE.md rule 5)."""

    frame_idx: int
    parse_status: Literal["ok", "partial", "unrecognized"]
    format_name: Literal["inline_list", "pipe_table", None]
    cells: dict[int, NumeralCellResult] = field(default_factory=dict)
    """Keyed by numeral 0-9 — only numerals actually found in this frame's
    tally are present (a frame that never reaches numeral `9` in its list
    yet legitimately has no `9` entry, not a `9: None` placeholder)."""

    claimed_total: int | None = None
    """The claimed `**Total**`/`(Total count: N)` value, if this frame
    printed one — `None` when absent (both formats sometimes omit it, e.g.
    an in-progress inline list that never reaches the `(Total count: …)`
    line, DECISION F-2's cell-vs-frame distinction extended to the total)."""

    raw_excerpt: str = ""
    """The raw frame text (or a bounded excerpt of it) — always kept on
    non-`"ok"` status so a caller can see what wasn't recognized, per the
    honest-failure discipline (never silently drop the evidence of why a
    frame parsed the way it did)."""

    def claimed_counts(self) -> dict[int, int]:
        """Numerals whose cell parsed to a real int — `None`-valued cells
        (garbage tokens) are excluded, not coerced to `0` (a `0` would be a
        fabricated count, not an honest read of "couldn't parse")."""
        return {n: cell.claimed for n, cell in self.cells.items() if cell.claimed is not None}


def _match_inline_list(frame_text: str) -> dict[int, NumeralCellResult] | None:
    """Match the run-1 shape: repeated `*   **N:** k time(s)` lines
    (possibly newline- or run-on-joined by decode noise — the fixture shows
    both). Returns `None` (not matched at all) only if zero bullet-numeral
    pairs are found; DECISION F-2 grants per-cell garbage tolerance within
    an otherwise-recognized list."""
    # Loosened structural anchor: `*` bullet, `**`-wrapped single-char
    # numeral-ish token, `:**`, then a value run before the next `*` bullet
    # or end of string. Garbage numeral tokens ("Ju", "ড্ড") are legitimately
    # not single digits — DECISION F-2 says garbage VALUE cells demote to
    # partial, but a garbage NUMERAL token means this bullet isn't a
    # numeral-tally row at all, so it is skipped rather than recorded.
    pattern = re.compile(
        r"\*\s*\*\*\s*([^*:]{1,20}?)\s*:?\*\*:?\s*([^*\n]{0,40})", re.MULTILINE
    )
    cells: dict[int, NumeralCellResult] = {}
    for match in pattern.finditer(frame_text):
        numeral_token, value_token = match.group(1).strip(), match.group(2).strip()
        numeral_match = _NUMERAL_CELL_RE.match(numeral_token)
        if not numeral_match:
            continue  # Not a numeral-keyed bullet at all (garbage or "Total").
        numeral = int(numeral_match.group(1))
        value_match = re.search(r"-?\d+", value_token)
        claimed = int(value_match.group(0)) if value_match else None
        cells[numeral] = NumeralCellResult(numeral=numeral, claimed=claimed, raw_value=value_token)
    return cells or None


def _match_pipe_table(frame_text: str) -> tuple[dict[int, NumeralCellResult], int | None] | None:
    """Match the run-2 shape: pipe-table body rows `| N | v |`, keyed on
    the two-pipe-cell structure alone (DECISION F-2 — never on the header
    label, which is frequently garbage). Requires at least one
    `| :--- | :--- |`-shaped separator row to be present anywhere in the
    frame (the structural marker of "this is a table", distinguishing a
    real table from stray `|` characters in prose) before scanning body
    rows; returns `None` if no separator is found at all."""
    lines = frame_text.split("\n")
    has_separator = any(_TABLE_SEPARATOR_RE.match(line) for line in lines)
    if not has_separator:
        return None

    cells: dict[int, NumeralCellResult] = {}
    claimed_total: int | None = None
    for line in lines:
        if _TABLE_SEPARATOR_RE.match(line):
            continue
        for match in _TABLE_ROW_RE.finditer(line):
            numeral_token, value_token = match.group("numeral").strip(), match.group("value").strip()
            numeral_match = _NUMERAL_CELL_RE.match(numeral_token)
            if numeral_match:
                numeral = int(numeral_match.group(1))
                value_match = _VALUE_CELL_RE.match(value_token)
                claimed = int(value_match.group(1)) if value_match else None
                cells[numeral] = NumeralCellResult(numeral=numeral, claimed=claimed, raw_value=value_token)
                continue
            if _TOTAL_CELL_RE.match(numeral_token):
                value_match = _VALUE_CELL_RE.match(value_token)
                if value_match:
                    claimed_total = int(value_match.group(1))
    return cells, claimed_total


def parse_tally_frame(frame_text: str, frame_idx: int) -> FrameAuditResult:
    """Parse one decoded frame's tally claim, trying each registered
    matcher in turn (DECISION F-2's format-matcher registry). Frame-level
    `parse_status`:

    - `"unrecognized"`: neither matcher found its structural anchor at all
      (no bullet-numeral pairs, no pipe-table separator) — REPORTED with
      the raw excerpt, never inferred as an empty/zero tally.
    - `"partial"`: a matcher's structure was found, but fewer than all ten
      numerals 0-9 parsed cleanly (either missing entirely — an
      in-progress frame that hasn't reached that numeral yet — or present
      with a garbage value cell).
    - `"ok"`: all ten numerals 0-9 present with a cleanly parsed int value.
    """
    inline_cells = _match_inline_list(frame_text)
    if inline_cells is not None:
        format_name: Literal["inline_list", "pipe_table"] = "inline_list"
        cells = inline_cells
        claimed_total = _extract_inline_total(frame_text)
    else:
        table_result = _match_pipe_table(frame_text)
        if table_result is None:
            return FrameAuditResult(
                frame_idx=frame_idx,
                parse_status="unrecognized",
                format_name=None,
                raw_excerpt=frame_text[:500],
            )
        format_name = "pipe_table"
        cells, claimed_total = table_result

    all_ten_present = all(n in cells for n in range(10))
    all_parsed = all(cell.claimed is not None for cell in cells.values())
    parse_status: Literal["ok", "partial"] = "ok" if (all_ten_present and all_parsed) else "partial"

    return FrameAuditResult(
        frame_idx=frame_idx,
        parse_status=parse_status,
        format_name=format_name,
        cells=cells,
        claimed_total=claimed_total,
        raw_excerpt="" if parse_status == "ok" else frame_text[:500],
    )


_INLINE_TOTAL_RE = re.compile(r"Total count:\s*(-?\d+)", re.IGNORECASE)


def _extract_inline_total(frame_text: str) -> int | None:
    """The run-1 format's total line, `*(Total count: 13)*` — a distinct
    shape from the pipe table's `| **Total** | **26** |` row, so it gets
    its own small extractor rather than overloading `_match_inline_list`'s
    per-bullet pattern."""
    match = _INLINE_TOTAL_RE.search(frame_text)
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Evidence counter — procedurally count the model's own restated numerals
# ---------------------------------------------------------------------------

# The evidence line(s): two observed shapes.
# Run 1: a single bold-wrapped comma list, the numerals themselves INSIDE
# the bold span — `**4, 7, 2, 4, 9, 1, 7, 4, 5, 2, 0, 7, 6**`.
# Run 2: a bold `**Row k:**` LABEL only, with the comma list as plain text
# after it to end-of-line — `**Row 1:** 4, 7, 2, 9, 0, 5, 4, 8, 2, 7, 1, 6, 9`
# (the numerals are NOT inside the bold span here, unlike run 1). Two
# distinct regexes rather than one, since the label-then-plain-list shape
# and the all-bold shape aren't the same span structure — trying to force
# one pattern to cover both would either miss run 2 (as an earlier version
# of this function did) or over-match unrelated bold spans in run 1.
_EVIDENCE_BOLD_LIST_RE = re.compile(r"\*\*([0-9](?:\s*,\s*[^*\n]+)*)\*\*")
_EVIDENCE_ROW_LABEL_RE = re.compile(r"\*\*Row\s*\d+:\*\*\s*([^\n]+)")


def count_evidence_numerals(frame_text: str) -> dict[int, int]:
    """Procedurally count numerals 0-9 in the model's own restated
    evidence rows (the bold `**4, 7, 2, ...**` / `**Row k:** ...` lines) —
    the "true" count this frame's tally claim is checked against (operator
    requirement (c): count-and-compare claims vs. the model's own drawn
    evidence).

    Deliberately narrow: only single-ASCII-digit tokens count. A
    multi-character or non-digit token between commas (garbage decode
    noise, frequent in both fixtures' early frames) is real evidence of
    "the canvas hasn't resolved this position yet", not a numeral — never
    coerced into one."""
    counts: dict[int, int] = {n: 0 for n in range(10)}

    def _tally(comma_list: str) -> None:
        for token in comma_list.split(","):
            token = token.strip().strip("*").strip()
            if re.fullmatch(r"[0-9]", token):
                counts[int(token)] += 1

    for evidence_match in _EVIDENCE_ROW_LABEL_RE.finditer(frame_text):
        _tally(evidence_match.group(1))
    if not _EVIDENCE_ROW_LABEL_RE.search(frame_text):
        # Only fall back to the bold-list shape when no `**Row k:**` label
        # was found at all — otherwise a bare `**Total**`/`**26**` bold
        # span elsewhere in the same frame (the pipe table's total row)
        # would be misread as a second evidence list and double-count.
        for evidence_match in _EVIDENCE_BOLD_LIST_RE.finditer(frame_text):
            _tally(evidence_match.group(1))

    return counts


# ---------------------------------------------------------------------------
# Revision watcher — frame-over-frame diff of claimed values
# ---------------------------------------------------------------------------

@dataclass
class RevisionEvent:
    """One numeral's claimed count changing between two consecutive
    *parseable* frames (mechanizes the observed `4: 2→3` datum,
    docs/experiments/2026-07-15-dg-numeral-counts-update-in-response)."""

    numeral: int
    from_frame_idx: int
    to_frame_idx: int
    from_value: int
    to_value: int


def watch_revisions(frame_results: list[FrameAuditResult]) -> list[RevisionEvent]:
    """Frame-over-frame diff of each numeral's claimed count, across
    frames in order, skipping `"unrecognized"` frames (nothing to diff
    against — an unrecognized frame contributes no claimed values at all,
    so it neither starts nor ends a revision pair) but including
    `"partial"` frames' successfully-parsed cells (DECISION F-2's per-cell
    granularity extended to the watcher: a `"partial"` frame's cleanly
    parsed numerals are still real evidence of what the model claimed at
    that step, even though other cells in the same frame failed to
    parse).

    A numeral absent from a frame's `claimed_counts()` (not yet reached,
    or a garbage cell) does not count as "revised to/from missing" — the
    comparison is only ever between two frames that BOTH have a parsed
    value for that numeral, mirroring `corroborate_no_mask_token`'s
    same-shape "only compare when both sides are observable" discipline
    in `consumers/analysis.py`."""
    events: list[RevisionEvent] = []
    last_claimed: dict[int, tuple[int, int]] = {}  # numeral -> (frame_idx, value)
    for result in frame_results:
        if result.parse_status == "unrecognized":
            continue
        for numeral, value in result.claimed_counts().items():
            if numeral in last_claimed:
                prev_idx, prev_value = last_claimed[numeral]
                if prev_value != value:
                    events.append(
                        RevisionEvent(
                            numeral=numeral,
                            from_frame_idx=prev_idx,
                            to_frame_idx=result.frame_idx,
                            from_value=prev_value,
                            to_value=value,
                        )
                    )
            last_claimed[numeral] = (result.frame_idx, value)
    return events


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

@dataclass
class TallyAudit:
    """The complete per-step audit record `audit_frames` returns: one
    `FrameAuditResult` per input frame, the revision events observed
    across the whole run, and — for the FINAL frame only — whether the
    claimed tally matches the true evidence-derived count for every
    numeral (arithmetic consistency, AC#1)."""

    frame_results: list[FrameAuditResult]
    revisions: list[RevisionEvent]
    final_frame_arithmetically_consistent: bool | None
    """`True` iff the final frame parsed `"ok"` or `"partial"` (at least
    some claimed cells) AND every claimed cell equals
    `count_evidence_numerals`'s true count for that numeral. `None` when
    the final frame is `"unrecognized"` — there is no claim to check
    consistency against, which is a different condition from "checked and
    found inconsistent" (`False`)."""


def audit_frames(decoded_frames: list[str]) -> TallyAudit:
    """Top-level consumer entry point (ARCHITECTURE.md rule 3): parse each
    decoded frame's tally claim, count each frame's own restated evidence,
    watch for frame-over-frame revisions, and check the final frame's
    claim against its own evidence for arithmetic consistency.

    `decoded_frames` is the `list[str]` shape `dgemma.loop.decode_frames`
    produces (or `extract_decoded_frames_from_composite_blob` recovers from
    a legacy composite `.txt` artifact) — this function takes the already-
    decoded strings directly, never a `CanvasTrace`/`DiffusionFrame`
    object, so it needs no tokenizer and no core-type import beyond what a
    caller already has in hand.

    `[]` input yields an empty `TallyAudit` (no frames, no revisions, no
    final-frame verdict) rather than raising — an honest empty result,
    mirroring `consumers/analysis.py`'s degenerate-input handling."""
    frame_results = [parse_tally_frame(text, idx) for idx, text in enumerate(decoded_frames)]
    revisions = watch_revisions(frame_results)

    if not frame_results:
        return TallyAudit(frame_results=[], revisions=[], final_frame_arithmetically_consistent=None)

    final = frame_results[-1]
    final_text = decoded_frames[-1]
    if final.parse_status == "unrecognized" or not final.claimed_counts():
        consistency: bool | None = None
    else:
        true_counts = count_evidence_numerals(final_text)
        consistency = all(
            claimed == true_counts.get(numeral, 0) for numeral, claimed in final.claimed_counts().items()
        )

    return TallyAudit(
        frame_results=frame_results,
        revisions=revisions,
        final_frame_arithmetically_consistent=consistency,
    )
