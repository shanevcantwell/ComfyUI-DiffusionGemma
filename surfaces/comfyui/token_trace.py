"""surfaces/comfyui/token_trace.py — DGemmaTokenTrace: thin ComfyUI adapter
(ADR-CDG-003), issue #61 P-D / issue #11.

The debug-node half of #11's design (ADR-CDG-014 Decision 6, ratified plan
issue #61: "`raw_canvas_ids` — trace-field first, debug node in P-D"): a
small surface node exposing `CanvasTrace.raw_canvas_ids` (the final,
pre-excision token id sequence) and `consumers.analysis.
build_token_identity_grid` (the per-step raw token-id grid) as a `STRING`
report — mirrors `surfaces/comfyui/trace.py`/`tally_audit.py`'s composition
pattern: pure functions in `consumers/`, thin socket-unwrap/wrap here.

Deliberately renders raw integer ids, not decoded text: decoding needs a
tokenizer this node does not have (it takes `canvas_trace`, not a model) —
same reasoning `consumers/analysis.py`'s own docstring gives for staying
tokenizer-free. A decoded-token rendering is a real follow-up (would need a
`DGEMMA_MODEL` input purely for its processor), out of this phase's scope.
"""
from __future__ import annotations

# Dual-context import, same depth/gate as surfaces/comfyui/trace.py.
if __package__ and __package__.count(".") >= 2:
    from ...consumers.analysis import build_token_identity_grid
    from .socket_types import DGEMMA_CANVAS_TRACE
else:
    from consumers.analysis import build_token_identity_grid
    from surfaces.comfyui.socket_types import DGEMMA_CANVAS_TRACE


def _format_report(trace) -> str:
    """Cheapest-correct STRING rendering (mirrors `trace.py`'s
    `_format_summary` / `tally_audit.py`'s `_format_report`): the
    pre-excision final sequence (issue #11's headline gap this closes) plus
    a per-step token-id grid, one row per frame.

    `raw_canvas_ids is None` (additive-optional absence, ADR-CDG-014
    Decision 1/2 — a legacy/no-capture trace) is reported honestly as
    "not captured", never rendered as an empty sequence — the same
    absence-vs-empty discipline `build_entropy_heatmap` enforces for
    entropy."""
    lines: list[str] = [f"scheduler={trace.scheduler_name} config={trace.scheduler_config}"]

    if trace.raw_canvas_ids is None:
        lines.append("raw_canvas_ids: not captured (legacy/no-capture trace)")
    else:
        lines.append(f"raw_canvas_ids ({len(trace.raw_canvas_ids)} tokens): {trace.raw_canvas_ids}")

    grid = build_token_identity_grid(trace)
    lines.append(f"per-step token-id grid ({len(grid)} steps):")
    for frame, row in zip(trace.frames, grid):
        lines.append(
            f"  canvas_idx={frame.canvas_idx} step_idx={frame.step_idx} "
            f"t={frame.t:.4f} temperature={frame.temperature:.4f}: {row}"
        )
    return "\n".join(lines)


class DGemmaTokenTrace:
    """Debug node (issue #11 / ADR-CDG-014 Decision 6): renders the
    pre-excision final canvas ids (`CanvasTrace.raw_canvas_ids`) and the
    per-step raw token-identity grid as a `STRING` report."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_trace": (DGEMMA_CANVAS_TRACE,),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("token_report",)
    FUNCTION = "render"
    CATEGORY = "DiffusionGemma"

    def render(self, canvas_trace):
        return (_format_report(canvas_trace),)
