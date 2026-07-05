"""nodes/trace.py — DGemmaTrace: thin ComfyUI adapter (ADR-CDG-003).

Post-hoc analysis over a complete `DGEMMA_CANVAS_TRACE` socket (plan.md Phase
3 (b) — unaffected by the (a) live-view split; see `nodes/sampler.py`'s
docstring). Calls the pure `dgemma.sampling` functions (steps 3-4: heatmap,
avalanche curve, mask-token corroboration) and wraps their plain-list/
dataclass results into ComfyUI-native socket types.

The one piece of non-trivial code in this file — building an `IMAGE` tensor
from a plain `list[list[int]]` — is the ADR-CDG-003-sanctioned exception, not
a violation: `dgemma/sampling.py`'s own docstring says explicitly that
ComfyUI-shaped tensor construction does NOT belong there, and plan.md step 6
assigns it here ("a real tensor built here, in the adapter layer, from the
plain array `dgemma/sampling.py` returned"). No denoising-loop logic, no
`for` loop over steps of its own — the heatmap/curve/corroboration
computation itself is entirely `dgemma.sampling`'s.
"""
from __future__ import annotations

import torch

# Dual-context import, explicit package-depth gate — see nodes/loader.py for
# the full rationale.
if __package__ and "." in __package__:
    from ..dgemma.sampling import (
        build_avalanche_curve,
        build_commit_heatmap,
        corroborate_no_mask_token,
    )
else:
    from dgemma.sampling import (
        build_avalanche_curve,
        build_commit_heatmap,
        corroborate_no_mask_token,
    )


def _heatmap_to_image(heatmap: list[list[int]]) -> torch.Tensor:
    """Wrap a plain `step x canvas-position` 0/1 array into a ComfyUI
    `IMAGE` tensor: `(batch, H, W, C)` float32 in `[0, 1]`, channels-last.
    Grayscale broadcast to 3 channels (an RGB `PreviewImage` consumer is the
    expected downstream, not a dedicated grayscale socket)."""
    if not heatmap or not heatmap[0]:
        return torch.zeros((1, 1, 1, 3), dtype=torch.float32)
    grid = torch.tensor(heatmap, dtype=torch.float32)  # (H, W)
    return grid.unsqueeze(0).unsqueeze(-1).expand(1, *grid.shape, 3).contiguous()


def _format_summary(trace, curve: list[float], corroboration) -> str:
    """Cheapest-correct rendering of the avalanche/commit curve (plan.md
    step 6: "a STRING summary ... implementer's call") plus item (c)'s
    mask-token corroboration verdict, so the empirical check (dgemma/
    sampling.py's own docstring on ADR-CDG-004's documentary "no MASK"
    confirmation) is actually visible on the graph, not just in a unit
    test."""
    lines = [
        f"scheduler={trace.scheduler_name} config={trace.scheduler_config}",
        f"steps={len(curve)}",
        "committed_fraction per step: " + ", ".join(f"{value:.4f}" for value in curve),
    ]
    if corroboration.no_fixed_sentinel:
        lines.append("mask-token corroboration: no fixed sentinel (uniform-state renoise supported)")
    else:
        lines.append(
            "mask-token corroboration: FIXED SENTINEL CANDIDATE id="
            f"{corroboration.candidate_sentinel_id} (absorbing-MASK signature)"
        )
    return "\n".join(lines)


class DGemmaTrace:
    """Renders a complete `CANVAS_TRACE` as a commit-state heatmap `IMAGE`
    plus a `STRING` summary of the avalanche curve and mask-token
    corroboration."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"canvas_trace": ("DGEMMA_CANVAS_TRACE",)}}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("heatmap", "summary")
    FUNCTION = "render"
    CATEGORY = "DiffusionGemma"

    def render(self, canvas_trace):
        heatmap = build_commit_heatmap(canvas_trace)
        curve = build_avalanche_curve(canvas_trace)
        corroboration = corroborate_no_mask_token(canvas_trace)

        image = _heatmap_to_image(heatmap)
        summary = _format_summary(canvas_trace, curve, corroboration)
        return (image, summary)
