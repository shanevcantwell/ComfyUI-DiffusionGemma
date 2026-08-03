"""dgemma/excision.py — thought-channel excision + the decode/derive family
(ADR-CDG-018 Stage 4, issue #129's `dgemma/loop.py` decomposition).

`ThoughtChannelExcision`, `excise_thought_channel`, `_decode_ids`,
`_extract_thought_text`, `decode_frames`, `resolve_vocab_size`,
`resolve_thought_channel_ids`, `derive_canvas_state` — extracted verbatim
from `dgemma/loop.py` (no behavior change — see
`tests/test_loop_golden_trace.py`, the decomposition's no-behavior-change
oracle). All post-run/no-loop-state: this cluster processes a completed run's
canvas ids and captured frames, never touches the scheduler/pipeline mid-run.

Ratified plan (issue #129, "excision.py membership" ruling): `derive_canvas_
state`/`resolve_vocab_size`/`resolve_thought_channel_ids` join the thought-
excision symbols here — the ADR's Stage 4 name list didn't enumerate them
explicitly, but they are the same decode/derive family (`resolve_vocab_size`
is called only by `run_diffusion`'s ingress; `resolve_thought_channel_ids`/
`derive_canvas_state` only by `_build_result`) and depend on `dgemma/config.py`'s
`THOUGHT_CHANNEL_*` constants (Phase 1's dependency, realized here).

`loop.py` re-imports every name, so every existing
`from dgemma.loop import decode_frames` / `resolve_vocab_size` / etc. import
site keeps working unchanged (the ratified plan's facade ruling, issue #129)
— including `surfaces/comfyui/emission.py`/`denoise.py`, which import
`decode_frames` from `dgemma.loop` in both dual-context arms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import (
    THOUGHT_CHANNEL_END_ID,
    THOUGHT_CHANNEL_END_TOKEN,
    THOUGHT_CHANNEL_LABEL,
    THOUGHT_CHANNEL_START_ID,
    THOUGHT_CHANNEL_START_TOKEN,
)
from .types import CanvasState, DiffusionFrame


def derive_canvas_state(
    *,
    text: str,
    canvas_ids: Any,
    frames: list[DiffusionFrame],
    steps_used: int,
    thought: str | None = None,
    stray_thought_delimiter: bool = False,
    eos_token_id: int | None = None,
) -> CanvasState:
    """Derive `CanvasState`'s validity fields from the captured frames.

    See `CanvasState.converged`'s docstring for what "converged" honestly
    does and does not claim. `thought` and `stray_thought_delimiter`
    (issue #8) are passed through unmodified — the excised thought-channel
    content (or `None`) and the stray-delimiter anomaly flag from
    `excise_thought_channel`.

    `turn_closed`/`answer_tokens` (issue #9, severable rider): reuses the
    excision/decode machinery already in `run_diffusion` (`dgemma/loop.py`)
    rather than capturing anything new. `turn_closed` is `True` iff
    `eos_token_id` is given and appears somewhere in `canvas_ids`: EOS was
    actually committed inside the generated region, as opposed to the canvas
    simply running out (`gen_length` reached with no EOS ever emitted) — the
    exact honesty gap issue #9 named, independent of `converged` (a run can
    converge on non-EOS filler once the canvas is full).

    `answer_tokens` counts the (thought-excised) ids **before the first
    EOS**, mirroring `_decode_ids`'s own trim: `canvas_ids` is not
    eos-trimmed, and a converged run pads the rest of the canvas with a
    trailing EOS/renoise fill run (observed live, ~30 tokens), so a bare
    `len(canvas_ids)` would inflate the count by that padding — defeating
    the honesty purpose the field exists for (review finding, 2026-07-05).
    The EOS token itself is deliberately NOT counted: it is the stop signal,
    not answer content. When no EOS is present the full (thought-excised)
    length is the honest count — every id is content the budget-truncated
    canvas actually holds. `0` when `canvas_ids` is `None` (the existing
    unit-test call shape, which never asserts on this field).
    """
    if not frames:
        raise RuntimeError("No frames captured — the denoising callback never fired.")
    last = frames[-1]
    if canvas_ids is not None:
        ids = [int(x) for x in canvas_ids]
        turn_closed = eos_token_id is not None and eos_token_id in ids
        answer_tokens = ids.index(eos_token_id) if turn_closed else len(ids)
    else:
        turn_closed = False
        answer_tokens = 0
    return CanvasState(
        text=text,
        canvas_ids=canvas_ids,
        converged=last.committed_fraction >= 1.0,
        committed_fraction=last.committed_fraction,
        steps_used=steps_used,
        thought=thought,
        stray_thought_delimiter=stray_thought_delimiter,
        turn_closed=bool(turn_closed),
        answer_tokens=answer_tokens,
    )


def resolve_vocab_size(processor: Any) -> int | None:
    """Resolve a vocab size for `dgemma.ingress.validate_constraints`'s C3
    check (issue #64 §3.4), off `processor`'s tokenizer.

    Same tokenizer-unwrap path `resolve_thought_channel_ids` uses
    (`getattr(processor, "tokenizer", processor)`). Tries `len(tokenizer)`
    first (the usual `PreTrainedTokenizerBase.__len__`), then
    `tokenizer.vocab_size`. Returns `None` — a named degradation, not a
    raise — when neither is available (e.g. a bare stub in a unit test that
    exposes no vocab at all), mirroring `resolve_thought_channel_ids`'s own
    stub fallback: C3 is skipped rather than this resolver inventing a size.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return len(tokenizer)
    except TypeError:
        pass
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    return None


def resolve_thought_channel_ids(processor: Any) -> tuple[int, int]:
    """Resolve the (start, end) thought-channel delimiter ids from `processor`.

    Prefers reading them off the tokenizer's own vocab
    (`convert_tokens_to_ids`) so a checkpoint swap that renumbers special
    tokens can't silently desync from a hardcoded pair; falls back to the
    module-level `THOUGHT_CHANNEL_START_ID`/`THOUGHT_CHANNEL_END_ID`
    constants (`dgemma/config.py`; provenance: `tokenizer_config.json`, see
    the comment above their definition) when `processor` doesn't expose a
    usable tokenizer — e.g. a bare stub in a unit test, or an `unk_token`
    fallback signaling the strings aren't in this vocab at all.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return THOUGHT_CHANNEL_START_ID, THOUGHT_CHANNEL_END_ID

    start_id = convert(THOUGHT_CHANNEL_START_TOKEN)
    end_id = convert(THOUGHT_CHANNEL_END_TOKEN)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        start_id is None
        or end_id is None
        or (unk_id is not None and (start_id == unk_id or end_id == unk_id))
    ):
        return THOUGHT_CHANNEL_START_ID, THOUGHT_CHANNEL_END_ID
    return start_id, end_id


@dataclass
class ThoughtChannelExcision:
    """Result of `excise_thought_channel` — named fields instead of a
    positional tuple, because the excision reports three independent things
    (the cleaned ids, zero-or-more excised spans, a stray-delimiter anomaly
    flag) and a growing anonymous tuple is how call sites silently misread
    which position means what."""

    remaining_ids: list[int]
    """Canvas ids with every well-formed thought-channel span (and, when
    applicable, the truncated turn-start frame) removed. Feeds the answer
    `STRING`."""

    thought_spans: list[list[int]]
    """Delimiter-exclusive content ids of each excised span, in canvas
    order. Empty list when no channel was present; an individual span may
    itself be `[]` (a zero-content frame)."""

    stray_start_delimiter: bool = False
    """`True` iff an unmatched `start_id` was found PAST the head of the
    generated region and therefore left in place rather than excised
    (excising it would silently destroy answer text — see
    `excise_thought_channel`). Surfaced so `CanvasState` can report the
    anomaly instead of the payload absorbing it invisibly."""


def excise_thought_channel(
    canvas_ids: Any,
    start_id: int = THOUGHT_CHANNEL_START_ID,
    end_id: int = THOUGHT_CHANNEL_END_ID,
) -> ThoughtChannelExcision:
    """Excise every thought-channel span from a canvas id sequence (issue #8).

    Pure id-level operation (ADR-CDG-001 payload-contamination discipline:
    id-span excision over decoded-string regex). The model emits
    `<|channel>thought\\n<channel|>` (empty channel — expected even with
    thinking off, per the model card) or `<|channel>...content...<channel|>`
    (non-empty, thinking on) at turn start; upstream
    `batch_decode(..., skip_special_tokens=True)` strips only the id-100/
    id-101 delimiters themselves, leaving `thought`/`\\n`/content — ordinary
    vocab tokens — to survive into the decoded string.

    Accepts a `torch.LongTensor`, a `list[int]`, or any 1-D iterable of ids;
    `remaining_ids`/`thought_spans` hold plain Python ints (never torch
    scalars), so downstream `tokenizer.decode` calls get plain id lists.

    Behavior, by case:
    - No `start_id` anywhere -> nothing excised, `thought_spans == []` —
      the false-strip guard: content that merely *mentions* "thought" as
      ordinary vocab is left untouched.
    - Each well-formed `start_id ... end_id` pair -> both delimiters and
      everything between them are removed from `remaining_ids`; the
      delimiter-exclusive content (possibly `[]`) is appended to
      `thought_spans`. ALL well-formed spans are excised, not just the
      first — a second leaked frame is the same ADR-CDG-001 breach as the
      first (review finding, 2026-07-05).
    - Unmatched `start_id` (no `end_id` anywhere after it) **at the head of
      the generated region** (index 0 — the documented turn-start frame
      position) -> treated as a truncated frame: excised through the end of
      the sequence, the tail going to `thought_spans`. No answer text can
      precede index 0, so nothing is lost but the broken frame.
    - Unmatched `start_id` **past the head** -> left in place untouched,
      along with everything after it — never silently truncate answer text.
      The raw delimiter stays in `remaining_ids` (where a
      `skip_special_tokens=True` decode drops the delimiter itself but keeps
      all surrounding answer text), and `stray_start_delimiter=True` is set
      so the anomaly surfaces on the `CanvasState` validity side rather
      than vanishing.
    """
    ids = [int(x) for x in canvas_ids]
    remaining: list[int] = []
    thought_spans: list[list[int]] = []
    stray_start_delimiter = False

    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            remaining.append(ids[i])
            i += 1
            continue
        try:
            end = ids.index(end_id, i + 1)
        except ValueError:
            if i == 0:
                # Truncated turn-start frame: excise-to-end loses nothing
                # but the broken frame.
                thought_spans.append(ids[1:])
            else:
                # Stray mid-canvas start delimiter: keep it and everything
                # after it — answer text is never silently dropped.
                stray_start_delimiter = True
                remaining.extend(ids[i:])
            break
        thought_spans.append(ids[i + 1 : end])
        i = end + 1

    return ThoughtChannelExcision(
        remaining_ids=remaining,
        thought_spans=thought_spans,
        stray_start_delimiter=stray_start_delimiter,
    )


def _decode_ids(processor: Any, ids: list[int], eos_token_id: int | None) -> str:
    """Decode `ids` the way the pipeline decodes `texts[0]`
    (`pipeline_diffusion_gemma.py:437-453`): trim at the first `eos_token_id`
    (inclusive) so post-EOS canvas-fill/renoise-garbage tokens don't leak in,
    then `skip_special_tokens=True`.

    Duplicated here rather than trusting the pipeline's own `output.texts[0]`
    because that value was decoded from the un-excised ids and still carries
    the thought-channel leak `excise_thought_channel` exists to remove; this
    re-derives the visible text from the corrected ids instead.
    """
    if eos_token_id is not None and eos_token_id in ids:
        ids = ids[: ids.index(eos_token_id) + 1]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(ids, skip_special_tokens=True)


def _extract_thought_text(decoded_channel: str) -> str | None:
    """Strip the chat template's fixed `"thought\\n"` channel-name label
    (provenance: see `THOUGHT_CHANNEL_LABEL`, `dgemma/config.py`) from a
    decoded between-delimiter span, returning `None` when nothing real
    remains.

    The canonical empty channel decodes to exactly `"thought\\n"` — label,
    no content — which must surface as "no thought", not as a `CanvasState`
    field containing the literal word "thought".
    """
    stripped = decoded_channel
    if stripped.startswith(THOUGHT_CHANNEL_LABEL):
        stripped = stripped[len(THOUGHT_CHANNEL_LABEL) :]
    stripped = stripped.strip()
    return stripped or None


def decode_frames(processor: Any, frames: list[DiffusionFrame]) -> list[str]:
    """Decode each captured `DiffusionFrame.canvas` to a string, in frame
    order — the "flipbook" series (noise -> coherent text), the raw per-step
    view `tools/flipbook/flipbook.py` renders from the GGUF CLI, exposed here
    for the transformers backend (plan.md P3, node-level `frames` output).

    Deliberately RAW, unlike `_decode_ids`: `skip_special_tokens=True`, but
    NO eos-trim and NO thought-channel excision. Early frames are mostly
    noise and transient thought-channel delimiters — that IS the intended
    view; trimming or excising here would hide the evolution the flipbook
    exists to show. (Contrast `_decode_ids`, which trims at EOS and is fed
    post-excision ids — that's the *answer* text, a different concern.)

    `canvas` may be a 1-D `[canvas_len]` tensor or a 2-D `[batch, canvas_len]`
    tensor (`run_diffusion` is single-example/batch-1 today) — example 0 is
    decoded for a 2-D tensor. Ids are moved off-device and converted to a
    plain `list[int]` (`.tolist()`) before `tokenizer.decode`, so this works
    identically for a CPU/GPU tensor or a plain list/tuple already in test
    fixtures.

    `[]` when `frames` is empty.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    texts: list[str] = []
    for frame in frames:
        canvas = frame.canvas
        if hasattr(canvas, "dim") and canvas.dim() == 2:
            canvas = canvas[0]
        ids = canvas.tolist() if hasattr(canvas, "tolist") else list(canvas)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts
