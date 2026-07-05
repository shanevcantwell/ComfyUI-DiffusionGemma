"""Thought-channel excision tests (issue #8).

The model emits `<|channel>thought\\n<channel|>` (empty channel) or
`<|channel>...content...<channel|>` (non-empty, thinking on) at turn start;
upstream `skip_special_tokens=True` decode strips only the id-100/id-101
delimiters, leaking the plain-text frame. `excise_thought_channel` is a pure
id-level function (ADR-CDG-001 payload-contamination discipline: id-span
excision over decoded-string regex) — no tokenizer/model/ComfyUI needed to
exercise it.
"""
from __future__ import annotations

import torch

from dgemma.loop import (
    THOUGHT_CHANNEL_END_ID,
    THOUGHT_CHANNEL_END_TOKEN,
    THOUGHT_CHANNEL_START_ID,
    THOUGHT_CHANNEL_START_TOKEN,
    excise_thought_channel,
    resolve_thought_channel_ids,
)

START = THOUGHT_CHANNEL_START_ID  # 100, "<|channel>"
END = THOUGHT_CHANNEL_END_ID  # 101, "<channel|>"
THOUGHT_WORD_ID = 45518  # ordinary vocab id for "thought" (issue #8 grounding)
NEWLINE_ID = 107


class TestExciseThoughtChannel:
    def test_empty_channel_is_stripped(self):
        """(a) Empty thinking channel — expected even with thinking off, per
        the model card ("an empty thinking channel might still be emitted").
        Grounded ids from issue #8: [100, 45518, 107, 101, ...answer...]."""
        canvas_ids = [START, THOUGHT_WORD_ID, NEWLINE_ID, END, 9, 10, 11]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [9, 10, 11]
        assert result.thought_spans == [[THOUGHT_WORD_ID, NEWLINE_ID]]  # the empty frame's own label
        assert result.stray_start_delimiter is False

    def test_non_empty_channel_content_is_separated_not_leaked(self):
        """(b) Thinking on, real chain-of-thought content between the
        delimiters: separated out as a thought span, never left in
        `remaining_ids` (which feeds the answer `STRING`)."""
        thought_content = [55, 56, 57, 58]
        canvas_ids = [START, *thought_content, END, 1, 2, 3]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [1, 2, 3]
        assert result.thought_spans == [thought_content]

    def test_no_channel_present_is_a_no_op(self):
        """(c) No channel delimiters anywhere — nothing excised, nothing
        reported as a thought."""
        canvas_ids = [1, 2, 3, 4, 5]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == canvas_ids
        assert result.thought_spans == []
        assert result.stray_start_delimiter is False

    def test_ordinary_thought_token_without_delimiters_is_not_false_stripped(self):
        """(d) The literal "thought" vocab id appears in the canvas with NO
        surrounding 100/101 delimiters — must not be mistaken for a channel
        frame and stripped."""
        canvas_ids = [THOUGHT_WORD_ID, NEWLINE_ID, 5, 6, THOUGHT_WORD_ID]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == canvas_ids
        assert result.thought_spans == []

    def test_accepts_torch_tensor_and_returns_plain_ints(self):
        canvas_ids = torch.tensor([START, 1, END, 2, 3], dtype=torch.long)
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [2, 3]
        assert result.thought_spans == [[1]]
        assert all(isinstance(x, int) for x in result.remaining_ids)
        assert all(isinstance(x, int) for span in result.thought_spans for x in span)

    def test_two_spans_are_both_excised(self):
        """Multi-span (review finding, 2026-07-05): a second id-100..id-101
        frame leaking its plain-text content onto the STRING payload is the
        same ADR-CDG-001 breach as the first — ALL well-formed spans are
        excised, each surfacing as its own thought span."""
        canvas_ids = [START, 11, END, 1, 2, START, 22, 23, END, 3]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [1, 2, 3]
        assert result.thought_spans == [[11], [22, 23]]
        assert result.stray_start_delimiter is False

    def test_adjacent_spans_are_both_excised(self):
        """Back-to-back frames with no answer tokens between them."""
        canvas_ids = [START, 11, END, START, 22, END, 7, 8]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [7, 8]
        assert result.thought_spans == [[11], [22]]

    def test_unmatched_start_at_head_excises_truncated_frame_to_end(self):
        """Turn-START truncated frame (no closing delimiter, start_id at
        index 0 — the documented frame position): excise to end. No answer
        text can precede index 0, so nothing is lost but the broken frame."""
        canvas_ids = [START, 7, 8, 9]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == []
        assert result.thought_spans == [[7, 8, 9]]
        assert result.stray_start_delimiter is False

    def test_stray_unmatched_start_mid_canvas_never_truncates_answer(self):
        """Review finding (2026-07-05): a stray unmatched start_id PAST the
        head must NOT be treated as thought-to-end-of-canvas — that would
        silently destroy answer text. It is left in place (the raw delimiter
        stays in remaining_ids; skip_special_tokens decode drops just the
        delimiter) and the anomaly is flagged for the validity side."""
        canvas_ids = [1, 2, START, 7, 8, 9]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [1, 2, START, 7, 8, 9]  # answer tokens 7,8,9 kept
        assert result.thought_spans == []
        assert result.stray_start_delimiter is True

    def test_wellformed_span_then_stray_start_keeps_tail_and_flags(self):
        """A valid turn-start frame followed later by a stray unmatched
        start: the frame is excised normally, the tail after the stray
        delimiter survives, and the anomaly is flagged."""
        canvas_ids = [START, 11, END, 1, 2, START, 7, 8]
        result = excise_thought_channel(canvas_ids, START, END)

        assert result.remaining_ids == [1, 2, START, 7, 8]
        assert result.thought_spans == [[11]]
        assert result.stray_start_delimiter is True

    def test_default_ids_match_module_constants(self):
        """Called with no explicit ids, the defaults are the ONE-MINT
        fallback constants (100/101), not some other pair."""
        canvas_ids = [START, 1, END, 2]
        result = excise_thought_channel(canvas_ids)

        assert result.remaining_ids == [2]
        assert result.thought_spans == [[1]]


class TestResolveThoughtChannelIds:
    def test_falls_back_to_constants_when_processor_has_no_tokenizer(self):
        start_id, end_id = resolve_thought_channel_ids(processor=object())
        assert (start_id, end_id) == (THOUGHT_CHANNEL_START_ID, THOUGHT_CHANNEL_END_ID)

    def test_reads_ids_from_tokenizer_vocab_when_available(self):
        """A tokenizer that actually knows the special tokens (e.g. a
        checkpoint that renumbered them) is preferred over the hardcoded
        fallback."""

        class FakeTokenizer:
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return {THOUGHT_CHANNEL_START_TOKEN: 200, THOUGHT_CHANNEL_END_TOKEN: 201}[token]

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        start_id, end_id = resolve_thought_channel_ids(FakeProcessor())
        assert (start_id, end_id) == (200, 201)

    def test_falls_back_when_tokenizer_returns_unk(self):
        """If the tokenizer doesn't actually have these strings in its
        vocab, `convert_tokens_to_ids` degrades to `unk_token_id` — must not
        be trusted as a real id pair."""

        class FakeTokenizer:
            unk_token_id = 999

            def convert_tokens_to_ids(self, token):
                return 999

        class FakeProcessor:
            tokenizer = FakeTokenizer()

        start_id, end_id = resolve_thought_channel_ids(FakeProcessor())
        assert (start_id, end_id) == (THOUGHT_CHANNEL_START_ID, THOUGHT_CHANNEL_END_ID)
