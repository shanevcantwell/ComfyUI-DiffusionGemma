"""nodes/flipbook.py — DGemmaFlipbook: thin ComfyUI adapter (ADR-CDG-003).

Renders the SAME per-step series `nodes/sampler.py`'s `frames` STRING output
already carries — `dgemma.loop.decode_frames` over `canvas_trace.frames` — as
an `IMAGE` batch instead of a list of strings, so the "watch it reason"
series is watchable/shareable (e.g. via `SaveAnimatedWEBP`/VHS downstream)
and not just inspectable as text. Purely additive (issue #21): this module
never touches `nodes/sampler.py`'s STRING output or calls `decode_frames`
differently — same decoded strings, alternate rendering, so the two outputs
always agree on content.

Tensor/PIL construction here — rasterizing each decoded step to a
fixed-size RGB canvas, then stacking one canvas per step on the batch dim —
is the ADR-CDG-003-sanctioned adapter-layer exception, the same shape as
`nodes/trace.py`'s own `_heatmap_to_image`: no denoising-loop logic, no
re-derivation of what `decode_frames` returns, just wrapping plain strings
into a ComfyUI-native `IMAGE` tensor.

**INPUT_TYPES is `canvas_trace` only (issue #21's stated signature,
mirroring `DGemmaTrace`), which surfaced a real gap this file does NOT paper
over:** `decode_frames` needs a processor (tokenizer) to turn a frame's raw
canvas ids into text, and `CanvasTrace` previously carried no such reference
(`DGemmaTrace`'s own analysis never decodes text, so it never needed one).
Rather than smuggle a second `DGEMMA_MODEL` input onto this node — which
would diverge from the issue's `canvas_trace`-only signature and force every
existing graph to rewire a second edge — `dgemma.types.CanvasTrace` gained
one new optional field, `processor` (default `None`, populated by
`dgemma.loop.run_diffusion` from `dgemma_model.processor`, the same object
`DGEMMA_MODEL` already points at — pointer-based transfer, not a copy). This
is the only change outside this file; it is additive (existing keyword-arg
`CanvasTrace(...)` constructions across the test suite are unaffected) and
does not touch `nodes/sampler.py`'s STRING `frames` output or its own call
to `decode_frames`.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Dual-context import, explicit package-depth gate — see nodes/loader.py for
# the full rationale (ComfyUI loader context vs. pytest/standalone).
if __package__ and "." in __package__:
    from ..dgemma.loop import decode_frames
else:
    from dgemma.loop import decode_frames

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
        "DGemmaFlipbook: no monospace TTF found among candidates, "
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


def _frames_to_image_batch(
    texts: list[str], *, width: int, font_size: int, caption_step_index: bool
) -> torch.Tensor:
    """Rasterize `texts` (one decoded step each, in order) into a ComfyUI
    `IMAGE` tensor: `(N, H, W, 3)` float32 in `[0, 1]`, channels-last, N ==
    `len(texts)`. All wrapping/height math runs once up front so every
    frame shares identical `(H, W)` — the batch-dim requirement — rather
    than each frame picking its own size and needing a later resize pass.

    `[]` input (a trace with no captured frames) yields a `(0, 1, 1, 3)`
    tensor rather than fabricating a placeholder frame — an honest empty
    batch, mirroring `nodes/trace.py`'s degenerate-input handling.
    """
    if not texts:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)

    font = _load_font(font_size)
    total = len(texts)
    wrapped = [_wrap_text(text, font, max_width=width - 2 * _MARGIN) for text in texts]
    max_lines = max(len(lines) for lines in wrapped)
    line_h = _line_height(font)
    height = 2 * _MARGIN + max_lines * line_h

    frames = []
    for idx, text in enumerate(texts):
        caption = f"step {idx + 1}/{total}" if caption_step_index else None
        image = _rasterize_frame(text, width=width, height=height, font=font, caption=caption)
        frames.append(np.asarray(image, dtype=np.uint8))

    batch = np.stack(frames, axis=0)  # (N, H, W, 3) uint8
    return torch.from_numpy(batch).to(torch.float32) / 255.0


class DGemmaFlipbook:
    """Renders a complete `CANVAS_TRACE` as a watchable `IMAGE` batch: one
    frame per captured denoising step, the decoded canvas text (via the same
    `dgemma.loop.decode_frames` the sampler's `frames` STRING output uses)
    rasterized onto a fixed-size dark canvas and stacked on the batch dim —
    `(num_steps, H, W, 3)` float32 in `[0, 1]`, channels-last. Feed the
    output to `SaveAnimatedWEBP` or a Video Helper Suite node to play the
    noise-to-coherent-text series back as an animation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_trace": ("DGEMMA_CANVAS_TRACE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "font_size": ("INT", {"default": 20, "min": 6, "max": 128}),
                "caption_step_index": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "render"
    CATEGORY = "DiffusionGemma"

    def render(self, canvas_trace, width: int = 512, font_size: int = 20, caption_step_index: bool = True):
        if canvas_trace.frames and canvas_trace.processor is None:
            raise ValueError(
                "DGemmaFlipbook: canvas_trace.processor is None, but frames are "
                "present — cannot decode canvas ids to text without a tokenizer. "
                "This CANVAS_TRACE was likely hand-constructed (e.g. a test "
                "fixture) rather than produced by DGemmaSampler/run_diffusion, "
                "which always populates it."
            )
        texts = decode_frames(canvas_trace.processor, canvas_trace.frames)
        images = _frames_to_image_batch(
            texts, width=width, font_size=font_size, caption_step_index=caption_step_index
        )
        return (images,)
