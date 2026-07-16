"""nodes/frames_image.py — `render_frames_to_image_batch`: renders a list of
already-decoded per-step strings into a ComfyUI `IMAGE` batch (issue #21,
reworked to be a `DGemmaSampler` output rather than a standalone node).

`DGemmaSampler` already decodes each captured step to a string for its
`frames` `STRING` output (`dgemma.loop.decode_frames` over
`canvas_trace.frames`) — this module renders that SAME list of strings as a
`(N, H, W, 3)` float32 `[0, 1]` image batch, so the "watch it reason" series
is watchable/shareable (e.g. via `SaveAnimatedWEBP`/VHS downstream) and not
just inspectable as text, with the sampler's own strings passed straight in
rather than re-decoded (one decode, two renderings).

Tensor/PIL construction here — rasterizing each string to a fixed-size RGB
canvas, then stacking one canvas per step on the batch dim — is the
ADR-CDG-003-sanctioned adapter-layer exception, the same shape as
`nodes/trace.py`'s own `_heatmap_to_image`: no denoising-loop logic, no
re-derivation of what `decode_frames` returns, just wrapping plain strings
into a ComfyUI-native `IMAGE` tensor. This module is intentionally
ComfyUI-agnostic itself (no socket types, no `INPUT_TYPES`/`RETURN_TYPES`) —
it is a plain rendering helper the node layer (`nodes/sampler.py`) calls,
not a node.

**Metadata banner (issue #84, DECISION S-1):** `FrameMetadata` + the
`frame_metadata=` parameter below thread `DiffusionFrame`'s per-step
telemetry (`t`, `temperature`, `committed_fraction`, mean Tier-0 `entropy`)
into the rendered batch, drawn as a top-left banner line — a SEPARATE
per-image key from `canvas_indices`, threaded the identical way
(length-checked parallel-to-`frames` list, additive-optional per field,
`None` renders `—`). `FrameMetadata` is a plain surface-side dataclass, not
a `dgemma/types.py` addition: it is a rendering-time repackaging of fields
that already live on `DiffusionFrame` (ADR-CDG-014), built by the caller
(`surfaces/comfyui/sampler.py`) from `canvas_trace.frames` — no new core
type, no new socket, nothing crosses the `dgemma`/`surfaces` seam that
doesn't already (rule 3/5 unaffected)."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Headless-safe font search (issue #21's hard constraint): a short list of
# common monospace TTF install paths, tried in order. None of these are
# guaranteed to exist on a given box (dev container, CI runner) — every
# lookup is guarded, and exhausting the list falls back to PIL's own bundled
# bitmap default font, which always loads. Never raises on a missing font.
_MONOSPACE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Debian/Ubuntu
    "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",  # Fedora/RHEL
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",  # Arch
    "/Library/Fonts/Courier New.ttf",  # macOS
    "C:\\Windows\\Fonts\\consola.ttf",  # Windows (Consolas)
)

# Dark ComfyUI-canvas-matching palette (issue #21: "match the ComfyUI canvas").
_BG_COLOR = (0x1E, 0x1E, 0x1E)
_FG_COLOR = (0xD4, 0xD4, 0xD4)
_CAPTION_COLOR = (0x90, 0x90, 0x90)
_MARGIN = 8


def _load_font(font_size: int):
    """Best-effort monospace TTF load; degrades to `ImageFont.load_default()`
    on any failure (missing file, unreadable, corrupt) — never raises. The
    bitmap default ignores `font_size` (a PIL limitation, not a bug here):
    legibility-over-failure is the point of this fallback chain."""
    for path in _MONOSPACE_FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, font_size)
        except OSError:
            continue
    logging.info(
        "render_frames_to_image_batch: no monospace TTF found among candidates, "
        "falling back to PIL's bundled default bitmap font."
    )
    return ImageFont.load_default()


def _wrap_text(text: str, font, max_width: int) -> list[str]:
    """Word-wrap `text` to `max_width` pixels using `font`'s own metrics.

    Pure with respect to the `font` object handed in — no file I/O, no
    module-level state — so it is unit-testable with any loaded PIL font,
    real or default-bitmap. Existing newlines are preserved as hard breaks
    (each source paragraph wraps independently); an empty paragraph yields a
    blank line rather than vanishing, so vertical spacing in the raster
    matches the source text.
    """
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if not current or font.getlength(candidate) <= max_width:
                current = candidate
            else:
                lines.append(current)
                current = word
        lines.append(current)
    return lines


def _line_height(font) -> int:
    """Pixel line height for `font`, derived from its own bounding box
    (ascender to descender of a representative glyph pair) plus a small
    leading margin — works identically for a real TTF or the bitmap
    default, neither of which expose a line-height property directly."""
    top, bottom = font.getbbox("Ag")[1], font.getbbox("Ag")[3]
    return max(bottom - top, 1) + 4


def _rasterize_frame(
    text: str, *, width: int, height: int, font, caption: str | None, banner: str | None = None
) -> Image.Image:
    """Render one decoded step to a fixed `(width, height)` dark-background
    RGB canvas: an optional metadata `banner` line top-left (issue #84,
    DECISION S-1), then word-wrapped body text below it, optional small
    step-index `caption` bottom-right. `height` is precomputed by the
    caller from the tallest wrapped frame in the batch PLUS one banner line
    when `frame_metadata` was supplied, so every frame shares one canvas
    size (`IMAGE` batching requires uniform H/W) — this function never
    resizes itself, it just draws into the size it's given."""
    image = Image.new("RGB", (width, height), _BG_COLOR)
    draw = ImageDraw.Draw(image)
    line_h = _line_height(font)
    y = _MARGIN
    if banner:
        draw.text((_MARGIN, y), banner, font=font, fill=_CAPTION_COLOR)
        y += line_h
    lines = _wrap_text(text, font, max_width=width - 2 * _MARGIN)
    for line in lines:
        draw.text((_MARGIN, y), line, font=font, fill=_FG_COLOR)
        y += line_h
    if caption:
        caption_width = font.getlength(caption)
        draw.text(
            (max(width - _MARGIN - caption_width, _MARGIN), max(height - _MARGIN - line_h, 0)),
            caption,
            font=font,
            fill=_CAPTION_COLOR,
        )
    return image


@dataclass
class FrameMetadata:
    """One frame's banner-relevant telemetry (issue #84, DECISION S-1) — a
    surface-side repackaging of `DiffusionFrame` fields (ADR-CDG-014), not a
    new core type. Every field beyond `step_idx`/`total_steps` is
    additive-optional (`None` renders `—`, never a fabricated value) —
    mirrors `DiffusionFrame.entropy`'s own "`None` means not captured this
    run, never zero" discipline (dgemma/types.py)."""

    step_idx: int
    """0-based running index of this frame within the whole captured
    series (matches `DiffusionFrame.step_idx`'s per-block numbering — this
    banner does not attempt N-canvas block-local renumbering, that is
    `_block_local_captions`' own job for the bottom-right caption)."""

    total_steps: int
    """Total frame count in this series — `i/M` denominator."""

    t: float | None = None
    temperature: float | None = None
    committed_fraction: float | None = None

    mean_entropy: float | None = None
    """Mean of `DiffusionFrame.entropy` (Tier 0, ADR-CDG-014) across canvas
    positions — a scalar reduction of the per-position tensor, computed by
    the caller (`surfaces/comfyui/sampler.py`) before this dataclass is
    built; `None` when `entropy` was not captured this run (mirrors that
    field's own `None` semantics, ADR-CDG-014 Decision 3)."""


def _format_number(value: float | None, fmt: str) -> str:
    """`—` for `None` (additive-optional discipline: absence renders
    honestly, never as a fabricated `0.0000`), `fmt` otherwise."""
    return "—" if value is None else format(value, fmt)


def _format_banner(metadata: "FrameMetadata") -> str:
    """`step i/M · t=... · temperature=... · committed%=... · entropy=...`
    (issue #84 operator requirement (a)). 1-based `i` for display (matches
    `_block_local_captions`' own 1-based convention); absent optional
    fields render `—`."""
    return (
        f"step {metadata.step_idx + 1}/{metadata.total_steps} · "
        f"t={_format_number(metadata.t, '.3f')} · "
        f"temperature={_format_number(metadata.temperature, '.3f')} · "
        f"committed%={_format_number(metadata.committed_fraction, '.1%')} · "
        f"entropy={_format_number(metadata.mean_entropy, '.3f')}"
    )


def _block_local_captions(canvas_indices: list[int]) -> list[str]:
    """Given a per-frame `canvas_idx` key (one entry per rendered frame, in
    order), produce the N-canvas caption for each frame: `"canvas k/N · step
    i/M"`, where `k` is the 1-based canvas number, `N` the total distinct
    canvas count in this run, `i` the 1-based block-local step within that
    canvas, and `M` that canvas's own step count (ADR-CDG-009 §2).

    N-ary by construction — the caption is derived from the per-image canvas
    key, never from a hardcoded boundary count. The degenerate N=1 case (every
    entry `0`) reads `"canvas 1/1 · step i/M"` with **no boundary treatment**;
    the general N≥2 case increments `k` at each `canvas_idx` change. Boundaries
    are inferred generically the same way `_FrameCollector` infers them
    (a non-increasing `canvas_idx` cannot occur here since the key is monotone
    non-decreasing by capture order, so a *change* in value is a boundary),
    with no assumption that there is exactly one.

    Robust to a `canvas_indices` that is not zero-based or contiguous (e.g. a
    future mid-schedule start): `N` and `k` are computed over the *distinct*
    canvas values actually present, ranked in first-seen order, so the caption
    stays honest even if the raw indices skip.
    """
    # Rank distinct canvas ids by first appearance → 1-based canvas number `k`.
    order: dict[int, int] = {}
    for cidx in canvas_indices:
        if cidx not in order:
            order[cidx] = len(order) + 1
    total_canvases = len(order)

    # Per-canvas step count `M`, and running per-canvas step position `i`.
    per_canvas_len: dict[int, int] = {}
    for cidx in canvas_indices:
        per_canvas_len[cidx] = per_canvas_len.get(cidx, 0) + 1

    captions: list[str] = []
    seen: dict[int, int] = {}
    for cidx in canvas_indices:
        seen[cidx] = seen.get(cidx, 0) + 1
        k = order[cidx]
        i = seen[cidx]
        m = per_canvas_len[cidx]
        captions.append(f"canvas {k}/{total_canvases} · step {i}/{m}")
    return captions


def render_frames_to_image_batch(
    frames: list[str],
    width: int = 512,
    font_size: int = 20,
    caption_step_index: bool = True,
    canvas_indices: list[int] | None = None,
    frame_metadata: "list[FrameMetadata] | None" = None,
) -> torch.Tensor:
    """Rasterize `frames` (one already-decoded step each, in order — e.g.
    `DGemmaSampler`'s own `frames` `STRING` output) into a ComfyUI `IMAGE`
    tensor: `(N, H, W, 3)` float32 in `[0, 1]`, channels-last, N ==
    `len(frames)`. All wrapping/height math runs once up front so every
    frame shares identical `(H, W)` — the batch-dim requirement — rather
    than each frame picking its own size and needing a later resize pass.

    `canvas_indices` (ADR-CDG-009 §2, N-canvas reframe): an optional per-image
    key carrying each frame's `canvas_idx` (parallel to `frames`, one entry
    each). When provided, the caption becomes the N-canvas form
    `"canvas k/N · step i/M"` — block-local numbering keyed to the canvas index
    *per image* (the #35 F7 `CONSERVE-DATA-BOUNDARY` move: the image↔canvas
    correspondence is carried explicitly, not reconstructed by a fragile 1:1
    positional zip). This is N-ary by construction: N=1 (every entry `0`) reads
    `"canvas 1/1 · step i/M"` with no boundary treatment, N≥2 numbers each
    block; no code path assumes exactly one boundary. When `canvas_indices` is
    `None` the caption falls back to the flat `"step idx/total"` form (callers
    without frame metadata, and existing behavior). `caption_step_index=False`
    suppresses captions entirely, regardless of `canvas_indices`.

    `frame_metadata` (issue #84, DECISION S-1): an optional per-image
    `FrameMetadata` key, threaded the SAME way as `canvas_indices` — parallel
    to `frames`, length-checked, one entry each. When provided, a top-left
    banner line (`_format_banner`: `"step i/M · t=... · temperature=... ·
    committed%=... · entropy=..."`) is drawn on every frame, and the batch
    height grows by one line to make room for it. `None` (the default)
    renders no banner at all — existing callers/tests are byte-for-byte
    unaffected (no layout regression for the banner-off default).

    NOTE (ADR-CDG-009 §2, held design): synthetic **divider frames** between
    canvases are deliberately NOT inserted here — that encoding (background
    color, whether a non-denoising frame belongs in the `images` batch) is an
    open ratification question. This function ships the caption/per-image-key
    slice only; batch length stays `len(frames)` (no `+num_transitions` yet).

    `[]` input (no captured frames) yields a `(0, 1, 1, 3)` tensor rather
    than fabricating a placeholder frame — an honest empty batch, mirroring
    `nodes/trace.py`'s degenerate-input handling.
    """
    if not frames:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)

    if canvas_indices is not None and len(canvas_indices) != len(frames):
        raise ValueError(
            "render_frames_to_image_batch: canvas_indices must be parallel to "
            f"frames (got {len(canvas_indices)} indices for {len(frames)} frames)."
        )

    if frame_metadata is not None and len(frame_metadata) != len(frames):
        raise ValueError(
            "render_frames_to_image_batch: frame_metadata must be parallel to "
            f"frames (got {len(frame_metadata)} entries for {len(frames)} frames)."
        )

    font = _load_font(font_size)
    total = len(frames)
    wrapped = [_wrap_text(text, font, max_width=width - 2 * _MARGIN) for text in frames]
    max_lines = max(len(lines) for lines in wrapped)
    line_h = _line_height(font)
    banner_lines = 1 if frame_metadata is not None else 0
    height = 2 * _MARGIN + (max_lines + banner_lines) * line_h

    if canvas_indices is not None:
        block_captions = _block_local_captions(canvas_indices)
    else:
        block_captions = None

    images = []
    for idx, text in enumerate(frames):
        if not caption_step_index:
            caption = None
        elif block_captions is not None:
            caption = block_captions[idx]
        else:
            caption = f"step {idx + 1}/{total}"
        banner = _format_banner(frame_metadata[idx]) if frame_metadata is not None else None
        image = _rasterize_frame(text, width=width, height=height, font=font, caption=caption, banner=banner)
        images.append(np.asarray(image, dtype=np.uint8))

    batch = np.stack(images, axis=0)  # (N, H, W, 3) uint8
    return torch.from_numpy(batch).to(torch.float32) / 255.0
