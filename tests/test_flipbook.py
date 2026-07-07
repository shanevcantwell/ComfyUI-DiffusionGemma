"""`nodes/flipbook.py` — `DGemmaFlipbook`: renders a `CANVAS_TRACE` as a
watchable `IMAGE` batch (issue #21). No ComfyUI, no real model/tokenizer, no
CUDA (ADR-CDG-003) — a fake tokenizer stands in for `canvas_trace.processor`
(mirrors `tests/test_frames.py`'s own `_FakeTokenizer`/`_FakeProcessor`
pattern for `decode_frames`), and `dgemma.types.CanvasTrace`/`DiffusionFrame`
are constructed directly, the same way `tests/test_trace_node.py` and
`tests/test_sampling.py` build synthetic traces.

Critical, explicitly tested here (issue #21's hard constraint): this must
run headless, with NO system monospace font assumed present — the font
search degrading to `PIL.ImageFont.load_default()` must never raise.
"""
from __future__ import annotations

import torch

from dgemma.types import CanvasTrace, DiffusionFrame
from nodes.flipbook import DGemmaFlipbook, _load_font, _wrap_text


class _FakeTokenizer:
    """Deterministic stand-in for a real tokenizer: `decode` joins ids as
    space-separated vocab words, so a test can control exactly how long
    (and how many wrapped lines) each frame's decoded text is."""

    VOCAB = {i: f"word{i}" for i in range(1, 30)}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(self.VOCAB.get(i, f"id{i}") for i in ids)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


def _frame(step_idx: int, canvas_row: list[int]) -> DiffusionFrame:
    return DiffusionFrame(
        canvas_idx=0,
        step_idx=step_idx,
        t=1.0 - step_idx * 0.1,
        temperature=0.5,
        committed_fraction_per_example=(step_idx / 3,),
        canvas=torch.tensor([canvas_row], dtype=torch.long),
    )


_UNSET = object()


def _canvas_trace(num_frames: int = 3, *, processor=_UNSET) -> CanvasTrace:
    """A small synthetic trace: each frame's canvas row grows longer, so the
    decoded strings have visibly different lengths (and wrap to a different
    number of lines) — the scenario that actually exercises the "every
    frame must share one H/W" padding logic, rather than all frames
    happening to be the same size by coincidence.

    `processor` defaults to a fake tokenizer stand-in; pass `processor=None`
    explicitly to test the no-processor-with-frames error path (a sentinel,
    not `is not None`, so an explicit `None` isn't silently overridden)."""
    frames = [_frame(i, list(range(1, 2 + i * 6))) for i in range(num_frames)]
    return CanvasTrace(
        frames=frames,
        scheduler_name="EntropyBoundScheduler",
        scheduler_config={"entropy_bound": 0.1},
        processor=_FakeProcessor() if processor is _UNSET else processor,
    )


def test_registered_in_node_class_mappings():
    from __init__ import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    assert NODE_CLASS_MAPPINGS["DGemmaFlipbook"] is DGemmaFlipbook
    assert NODE_DISPLAY_NAME_MAPPINGS["DGemmaFlipbook"] == "DiffusionGemma Flipbook"


def test_declarations():
    spec = DGemmaFlipbook.INPUT_TYPES()
    assert set(spec["required"]) == {"canvas_trace", "width", "font_size", "caption_step_index"}
    assert spec["required"]["canvas_trace"] == ("DGEMMA_CANVAS_TRACE",)
    assert spec["required"]["width"] == ("INT", {"default": 512, "min": 64, "max": 4096})
    assert spec["required"]["font_size"] == ("INT", {"default": 20, "min": 6, "max": 128})
    assert spec["required"]["caption_step_index"] == ("BOOLEAN", {"default": True})
    assert DGemmaFlipbook.RETURN_TYPES == ("IMAGE",)
    assert DGemmaFlipbook.RETURN_NAMES == ("frames",)
    assert DGemmaFlipbook.FUNCTION == "render"
    assert DGemmaFlipbook.CATEGORY == "DiffusionGemma"


class TestRenderOutputContract:
    """The load-bearing shape/dtype/range contract: `(N, H, W, 3)` float32 in
    `[0, 1]`, `N == len(canvas_trace.frames)`, every frame sharing one H/W —
    the ComfyUI `IMAGE` batch precondition."""

    def test_output_is_correctly_shaped_image_batch(self):
        trace = _canvas_trace(num_frames=3)
        node = DGemmaFlipbook()

        (images,) = node.render(trace, width=256, font_size=16, caption_step_index=True)

        assert isinstance(images, torch.Tensor)
        assert images.dtype == torch.float32
        assert images.dim() == 4
        num_steps, height, width, channels = images.shape
        assert num_steps == len(trace.frames) == 3
        assert width == 256
        assert channels == 3
        assert height > 0

    def test_values_in_unit_range(self):
        trace = _canvas_trace(num_frames=2)
        node = DGemmaFlipbook()

        (images,) = node.render(trace, width=200, font_size=14)

        assert torch.all(images >= 0.0)
        assert torch.all(images <= 1.0)

    def test_frames_of_differing_text_length_still_share_identical_hw(self):
        """The frame at step 2 decodes to a visibly longer string (more
        wrapped lines) than step 0 — both must still land in a batch of
        uniform H/W, since torch.stack requires it. This is the scenario
        the height-precompute pass in `_frames_to_image_batch` exists for."""
        trace = _canvas_trace(num_frames=3)
        node = DGemmaFlipbook()

        (images,) = node.render(trace, width=150, font_size=18)

        # A single stacked tensor already proves uniform H/W (torch.stack
        # would refuse a ragged set) — assert it explicitly per-frame too.
        heights = {images[i].shape[0] for i in range(images.shape[0])}
        widths = {images[i].shape[1] for i in range(images.shape[0])}
        assert len(heights) == 1
        assert len(widths) == 1

    def test_no_captured_frames_yields_empty_batch_not_a_crash(self):
        """Degenerate input (mirrors `nodes/trace.py`'s empty-heatmap
        handling): an honest zero-length batch, not a fabricated frame."""
        trace = _canvas_trace(num_frames=0)
        node = DGemmaFlipbook()

        (images,) = node.render(trace)

        assert images.shape[0] == 0

    def test_processor_none_with_frames_present_raises_clearly(self):
        """A hand-built CANVAS_TRACE with frames but no processor (the field
        defaults to None) cannot be decoded — this must fail with a clear,
        attributed error rather than an opaque AttributeError deep inside
        `decode_frames`."""
        trace = _canvas_trace(num_frames=1, processor=None)
        node = DGemmaFlipbook()

        try:
            node.render(trace)
        except ValueError as exc:
            assert "processor" in str(exc)
        else:
            raise AssertionError("expected ValueError for processor=None with frames present")

    def test_caption_step_index_toggle_does_not_crash_either_way(self):
        trace = _canvas_trace(num_frames=2)
        node = DGemmaFlipbook()

        (with_caption,) = node.render(trace, caption_step_index=True)
        (without_caption,) = node.render(trace, caption_step_index=False)

        assert with_caption.shape[0] == without_caption.shape[0] == 2


class TestHeadlessFontFallback:
    """Issue #21's hard constraint: no dependency on a system font that may
    be absent, and never raise on a missing one. `_load_font` must degrade
    to `PIL.ImageFont.load_default()` — exercised here by forcing every
    candidate TTF path to fail, simulating a box with zero system fonts
    installed (the tests must pass in that environment regardless of what
    fonts happen to be on THIS machine)."""

    def test_load_font_falls_back_to_default_when_no_ttf_available(self, monkeypatch):
        import nodes.flipbook as flipbook_mod

        monkeypatch.setattr(flipbook_mod, "_MONOSPACE_FONT_CANDIDATES", ("/nonexistent/path/does-not-exist.ttf",))

        font = _load_font(20)  # must not raise

        assert font is not None
        # PIL's bitmap default still exposes the metrics this module needs.
        assert hasattr(font, "getlength")
        assert hasattr(font, "getbbox")

    def test_render_end_to_end_with_no_system_fonts_available(self, monkeypatch):
        """Full node render, forced onto the `load_default()` fallback path
        — the actual headless scenario the constraint cares about, not just
        the font loader in isolation."""
        import nodes.flipbook as flipbook_mod

        monkeypatch.setattr(flipbook_mod, "_MONOSPACE_FONT_CANDIDATES", ())

        trace = _canvas_trace(num_frames=2)
        node = DGemmaFlipbook()

        (images,) = node.render(trace, width=256, font_size=20)

        assert images.shape[0] == 2
        assert images.dtype == torch.float32


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
