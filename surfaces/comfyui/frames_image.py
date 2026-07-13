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
"""
from __future__ import annotations

import logging

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
    text: str, *, width: int, height: int, font, caption: str | None
) -> Image.Image:
    """Render one decoded step to a fixed `(width, height)` dark-background
    RGB canvas: word-wrapped body text top-left, optional small step-index
    caption bottom-right. `height` is precomputed by the caller from the
    tallest wrapped frame in the batch, so every frame shares one canvas
    size (`IMAGE` batching requires uniform H/W) — this function never
    resizes itself, it just draws into the size it's given."""
    image = Image.new("RGB", (width, height), _BG_COLOR)
    draw = ImageDraw.Draw(image)
    lines = _wrap_text(text, font, max_width=width - 2 * _MARGIN)
    line_h = _line_height(font)
    y = _MARGIN
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

    font = _load_font(font_size)
    total = len(frames)
    wrapped = [_wrap_text(text, font, max_width=width - 2 * _MARGIN) for text in frames]
    max_lines = max(len(lines) for lines in wrapped)
    line_h = _line_height(font)
    height = 2 * _MARGIN + max_lines * line_h

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
        image = _rasterize_frame(text, width=width, height=height, font=font, caption=caption)
        images.append(np.asarray(image, dtype=np.uint8))

    batch = np.stack(images, axis=0)  # (N, H, W, 3) uint8
    return torch.from_numpy(batch).to(torch.float32) / 255.0
