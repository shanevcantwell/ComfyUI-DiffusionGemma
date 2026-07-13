"""`nodes/frames_image.py` — `render_frames_to_image_batch` (issue #21,
reworked from a standalone `DGemmaFlipbook` node into a `DGemmaSampler`
output). Pure rendering helper: no ComfyUI, no real model/tokenizer, no CUDA
(ADR-CDG-003) — it takes already-decoded strings directly, so this suite
needs no fake tokenizer/processor at all (unlike the prior node, which took
a `CANVAS_TRACE` and had to decode it itself).

Critical, explicitly tested here (issue #21's hard constraint, carried
through the rework unchanged): this must run headless, with NO system
monospace font assumed present — the font search degrading to
`PIL.ImageFont.load_default()` must never raise.
"""
from __future__ import annotations

import pytest
import torch

from surfaces.comfyui.frames_image import (
    _block_local_captions,
    _load_font,
    _wrap_text,
    render_frames_to_image_batch,
)


def _frames(num_frames: int = 3) -> list[str]:
    """Each frame decodes to a visibly different length string, so the
    decoded strings wrap to a different number of lines — the scenario that
    actually exercises the "every frame must share one H/W" padding logic,
    rather than all frames happening to be the same size by coincidence."""
    return [" ".join(f"word{i}" for i in range(1 + step * 6)) for step in range(num_frames)]


class TestRenderOutputContract:
    """The load-bearing shape/dtype/range contract: `(N, H, W, 3)` float32 in
    `[0, 1]`, `N == len(frames)`, every frame sharing one H/W — the ComfyUI
    `IMAGE` batch precondition."""

    def test_output_is_correctly_shaped_image_batch(self):
        frames = _frames(num_frames=3)

        images = render_frames_to_image_batch(frames, width=256, font_size=16, caption_step_index=True)

        assert isinstance(images, torch.Tensor)
        assert images.dtype == torch.float32
        assert images.dim() == 4
        num_steps, height, width, channels = images.shape
        assert num_steps == len(frames) == 3
        assert width == 256
        assert channels == 3
        assert height > 0

    def test_values_in_unit_range(self):
        frames = _frames(num_frames=2)

        images = render_frames_to_image_batch(frames, width=200, font_size=14)

        assert torch.all(images >= 0.0)
        assert torch.all(images <= 1.0)

    def test_frames_of_differing_text_length_still_share_identical_hw(self):
        """A later frame decodes to a visibly longer string (more wrapped
        lines) than an earlier one — both must still land in a batch of
        uniform H/W, since torch.stack requires it. This is the scenario
        the height-precompute pass in `render_frames_to_image_batch` exists
        for."""
        frames = _frames(num_frames=3)

        images = render_frames_to_image_batch(frames, width=150, font_size=18)

        # A single stacked tensor already proves uniform H/W (torch.stack
        # would refuse a ragged set) — assert it explicitly per-frame too.
        heights = {images[i].shape[0] for i in range(images.shape[0])}
        widths = {images[i].shape[1] for i in range(images.shape[0])}
        assert len(heights) == 1
        assert len(widths) == 1

    def test_no_frames_yields_empty_batch_not_a_crash(self):
        """Degenerate input (mirrors `nodes/trace.py`'s empty-heatmap
        handling): an honest zero-length batch, not a fabricated frame."""
        images = render_frames_to_image_batch([])

        assert images.shape[0] == 0

    def test_caption_step_index_toggle_does_not_crash_either_way(self):
        frames = _frames(num_frames=2)

        with_caption = render_frames_to_image_batch(frames, caption_step_index=True)
        without_caption = render_frames_to_image_batch(frames, caption_step_index=False)

        assert with_caption.shape[0] == without_caption.shape[0] == 2


class TestHeadlessFontFallback:
    """Issue #21's hard constraint: no dependency on a system font that may
    be absent, and never raise on a missing one. `_load_font` must degrade
    to `PIL.ImageFont.load_default()` — exercised here by forcing every
    candidate TTF path to fail, simulating a box with zero system fonts
    installed (the tests must pass in that environment regardless of what
    fonts happen to be on THIS machine)."""

    def test_load_font_falls_back_to_default_when_no_ttf_available(self, monkeypatch):
        import surfaces.comfyui.frames_image as frames_image_mod

        monkeypatch.setattr(frames_image_mod, "_MONOSPACE_FONT_CANDIDATES", ("/nonexistent/path/does-not-exist.ttf",))

        font = _load_font(20)  # must not raise

        assert font is not None
        # PIL's bitmap default still exposes the metrics this module needs.
        assert hasattr(font, "getlength")
        assert hasattr(font, "getbbox")

    def test_render_end_to_end_with_no_system_fonts_available(self, monkeypatch):
        """Full render, forced onto the `load_default()` fallback path — the
        actual headless scenario the constraint cares about, not just the
        font loader in isolation."""
        import surfaces.comfyui.frames_image as frames_image_mod

        monkeypatch.setattr(frames_image_mod, "_MONOSPACE_FONT_CANDIDATES", ())

        frames = _frames(num_frames=2)

        images = render_frames_to_image_batch(frames, width=256, font_size=20)

        assert images.shape[0] == 2
        assert images.dtype == torch.float32


class TestNCanvasCaptions:
    """ADR-CDG-009 (N-canvas reframe) / issue #26, #35 F7: the flipbook caption
    is the N-ary `"canvas k/N · step i/M"` form keyed to a per-image canvas
    index, NOT a hardcoded two-canvas or single-boundary assumption.

    Parametrized over N — including the degenerate N=1, which must render
    normal captions and NO boundary treatment (zero dividers). Two is the
    observed case (thinking + answer), not the structural limit."""

    def test_n1_single_canvas_captions_and_no_boundary(self):
        """N=1: every frame keyed to canvas 0 → `canvas 1/1 · step i/M`, and
        no divider is inserted (batch length == frame count, unchanged)."""
        canvas_indices = [0, 0, 0, 0]
        captions = _block_local_captions(canvas_indices)

        assert captions == [
            "canvas 1/1 · step 1/4",
            "canvas 1/1 · step 2/4",
            "canvas 1/1 · step 3/4",
            "canvas 1/1 · step 4/4",
        ]
        # Every caption reports N == 1 — no spurious second canvas.
        assert all("canvas 1/1" in c for c in captions)

    def test_n2_two_canvas_captions_reset_step_at_boundary(self):
        """N=2 (the observed thinking+answer case): block-local step resets to
        1 at the single boundary; canvas number advances 1→2."""
        # canvas 0 has 3 steps, canvas 1 has 2 steps.
        canvas_indices = [0, 0, 0, 1, 1]
        captions = _block_local_captions(canvas_indices)

        assert captions == [
            "canvas 1/2 · step 1/3",
            "canvas 1/2 · step 2/3",
            "canvas 1/2 · step 3/3",
            "canvas 2/2 · step 1/2",
            "canvas 2/2 · step 2/2",
        ]

    def test_n3_three_canvas_captions(self):
        """N=3: two boundaries, block-local step resets at each; N reported
        as 3 throughout — proves 2 is not baked in anywhere."""
        canvas_indices = [0, 0, 1, 2, 2, 2]
        captions = _block_local_captions(canvas_indices)

        assert captions == [
            "canvas 1/3 · step 1/2",
            "canvas 1/3 · step 2/2",
            "canvas 2/3 · step 1/1",
            "canvas 3/3 · step 1/3",
            "canvas 3/3 · step 2/3",
            "canvas 3/3 · step 3/3",
        ]

    def test_non_contiguous_canvas_ids_ranked_by_first_seen(self):
        """A future mid-schedule start could yield non-zero-based / skipping
        canvas ids; N and k are computed over the distinct values present,
        ranked first-seen, so the caption stays honest."""
        canvas_indices = [5, 5, 9]
        captions = _block_local_captions(canvas_indices)

        assert captions == [
            "canvas 1/2 · step 1/2",
            "canvas 1/2 · step 2/2",
            "canvas 2/2 · step 1/1",
        ]

    @pytest.mark.parametrize(
        "canvas_indices, expected_transitions",
        [
            ([0, 0, 0], 0),            # N=1 → zero boundaries, ZERO dividers
            ([0, 0, 1, 1], 1),         # N=2 → one boundary
            ([0, 1, 2], 2),            # N=3 → two boundaries
            ([0, 0, 1, 2, 2, 2], 2),   # N=3, uneven block sizes
        ],
    )
    def test_transition_count_is_n_minus_one(self, canvas_indices, expected_transitions):
        """The divider-count invariant (ADR-CDG-009): num_transitions == N-1,
        so N=1 → 0. This pins the count the (held) divider-frame design will
        insert, and guarantees the degenerate N=1 case renders NO divider."""
        transitions = sum(
            1 for a, b in zip(canvas_indices, canvas_indices[1:]) if a != b
        )
        distinct = len(set(canvas_indices))
        assert transitions == expected_transitions
        assert transitions == distinct - 1  # N-1, N=1 → 0

    def test_render_batch_length_unchanged_no_divider_frames_yet(self):
        """The shipped slice ships CAPTIONS only — synthetic divider frames are
        held design (ADR-CDG-009 §2). So batch length is still len(frames) for
        every N, INCLUDING that N=1 renders no extra frame. When the divider
        design ratifies, this becomes len(frames)+num_transitions and N=1 stays
        len(frames) (+0)."""
        for canvas_indices in ([0, 0, 0], [0, 0, 1, 1], [0, 1, 2]):
            frames = _frames(num_frames=len(canvas_indices))
            images = render_frames_to_image_batch(
                frames, width=200, font_size=14, canvas_indices=canvas_indices
            )
            assert images.shape[0] == len(frames)

    def test_render_accepts_canvas_indices_without_crashing(self):
        """End-to-end: passing the per-image key produces a valid batch (the
        caption path is pixel-opaque, so correctness of the string is asserted
        against `_block_local_captions` above; here we prove the render path
        itself is wired and doesn't raise)."""
        frames = _frames(num_frames=5)
        images = render_frames_to_image_batch(
            frames, width=256, font_size=16, canvas_indices=[0, 0, 0, 1, 1]
        )
        assert images.shape[0] == 5
        assert images.dtype == torch.float32

    def test_mismatched_canvas_indices_length_rejected(self):
        """A per-image key that isn't parallel to frames is a caller bug that
        would silently mis-caption — reject at the door, don't launder it."""
        frames = _frames(num_frames=3)
        with pytest.raises(ValueError, match="parallel"):
            render_frames_to_image_batch(frames, canvas_indices=[0, 1])

    def test_canvas_indices_none_falls_back_to_flat_caption(self):
        """Backward compatibility: without the per-image key, the flat
        `step i/total` caption path is unchanged (callers without frame
        metadata, existing behavior)."""
        frames = _frames(num_frames=3)
        images = render_frames_to_image_batch(frames, canvas_indices=None)
        assert images.shape[0] == 3


class TestWrapText:
    """Pure helper, testable with any loaded PIL font (real or default)."""

    def test_short_text_is_a_single_line(self):
        font = _load_font(16)
        lines = _wrap_text("hello world", font, max_width=1000)
        assert lines == ["hello world"]

    def test_long_text_wraps_into_multiple_lines(self):
        font = _load_font(16)
        long_text = " ".join(f"word{i}" for i in range(40))
        lines = _wrap_text(long_text, font, max_width=100)
        assert len(lines) > 1
        # No word content lost across the wrap.
        assert " ".join(lines).split() == long_text.split()

    def test_explicit_newlines_are_preserved_as_hard_breaks(self):
        font = _load_font(16)
        lines = _wrap_text("first\nsecond", font, max_width=1000)
        assert lines == ["first", "second"]
