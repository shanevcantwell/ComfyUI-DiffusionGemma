"""surfaces/comfyui/trace.py — DGemmaTrace: thin ComfyUI adapter (ADR-CDG-003).

Post-hoc analysis over a complete `DGEMMA_CANVAS_TRACE` socket (plan.md Phase
3 (b) — unaffected by the (a) live-view split; see `surfaces/comfyui/sampler.py`'s
docstring). Calls the pure `dgemma.sampling` functions (steps 3-4: heatmap,
avalanche curve, mask-token corroboration) and wraps their plain-list/
dataclass results into ComfyUI-native socket types. `dgemma/sampling.py`
itself is NOT relocated by this phase — ADR-CDG-008 Open Question #1
(analysis's eventual `consumers/`/`surfaces/analysis/` home) is Phase 3,
explicitly out of scope here.

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

# Dual-context import, explicit package-depth gate — see
# surfaces/comfyui/loader.py for the full rationale. This module lives two
# levels under the pack root (surfaces/comfyui/), so the relative climb to
# dgemma/ is THREE dots (ADR-CDG-008 Phase 1 / issue #52 risk R-1).
# dgemma.sampling itself is NOT relocated by this phase (Phase 3 is out of
# scope, Open Question #1 unresolved) — only the import depth changes.
if __package__ and "." in __package__:
    from ...dgemma.sampling import (
        build_avalanche_curve,
        build_commit_heatmap,
        corroborate_no_mask_token,
    )
    from .socket_types import DGEMMA_CANVAS_TRACE
else:
    from dgemma.sampling import (
        build_avalanche_curve,
        build_commit_heatmap,
        corroborate_no_mask_token,
    )
    from surfaces.comfyui.socket_types import DGEMMA_CANVAS_TRACE


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
    test.

    The `committed_fraction` label is explicit about block-local scope
    (ADR-CDG-009, issue #26): the value is `DiffusionFrame.committed_fraction`,
    which is a per-block reading (`accepted_index.float().mean(dim=-1)` over
    the ACTIVE block only, dgemma/loop.py's `_FrameCollector`) — it resets
    toward 0 at every `canvas_idx` boundary, not just at the start of the
    whole run. Labeling it plainly "committed_fraction" reads as a global
    progress metric and makes the block-boundary reset look like the canvas
    re-melted, which is exactly the misreading issue #26 reports. This is a
    caption fix only — the underlying value and its docstring
    (dgemma/types.py `DiffusionFrame.committed_fraction`) were already
    correct; this propagates that existing meaning to the one operator-facing
    text surface that didn't say it.

    Tri-state verdict (issue #22): the vacuous case (zero observed
    transitions) prints its own line, distinct from genuine "evidence
    against a fixed sentinel" — printing the same "supported" wording on
    zero evidence is exactly the overclaim ADR-CDG-001 forbids."""
    lines = [
        f"scheduler={trace.scheduler_name} config={trace.scheduler_config}",
        f"steps={len(curve)}",
        "committed_fraction per step (block-local — resets near 0 at each "
        "canvas/block boundary; this is block advancement, not re-melt): "
        + ", ".join(f"{value:.4f}" for value in curve),
    ]
    if corroboration.verdict == "evidence_against_sentinel":
        lines.append("mask-token corroboration: no fixed sentinel (uniform-state renoise supported)")
    elif corroboration.verdict == "vacuous":
        lines.append(
            "mask-token corroboration: vacuous (no mid-renoise transitions observed — "
            "neither supports nor contradicts uniform-state renoise)"
        )
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
        return {
            "required": {
                "canvas_trace": (DGEMMA_CANVAS_TRACE,),
                # Nearest-neighbor upscale factor (operator finding,
                # 2026-07-05: a raw steps×positions map — 256×11 observed —
                # is unreadably small). Threads straight through to
                # `build_commit_heatmap(scale=...)`; the scaling math is
                # engine-side (ADR-CDG-003), this widget is pure unpack.
                "cell_px": ("INT", {"default": 6, "min": 1, "max": 32}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("heatmap", "summary")
    FUNCTION = "render"
    CATEGORY = "DiffusionGemma"

    def render(self, canvas_trace, cell_px: int = 6):
        heatmap = build_commit_heatmap(canvas_trace, scale=cell_px)
        curve = build_avalanche_curve(canvas_trace)
        corroboration = corroborate_no_mask_token(canvas_trace)

        image = _heatmap_to_image(heatmap)
        summary = _format_summary(canvas_trace, curve, corroboration)
        return (image, summary)
