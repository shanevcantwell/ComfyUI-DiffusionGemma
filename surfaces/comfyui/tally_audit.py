"""surfaces/comfyui/tally_audit.py — DGemmaTallyAudit: thin ComfyUI adapter
(ADR-CDG-003), issue #84.

Wraps the pure `consumers.tally_audit.audit_frames` the same way
`surfaces/comfyui/trace.py` wraps `consumers.analysis`'s functions (the
composition pattern named in ARCHITECTURE.md's Consumers section: "the
pure functions are consumer-tier, the socket-wrapping node is
surface-tier, and the node importing the consumer is normal composition,
not a layering inversion").

**Why this node takes `frames` (the sampler's already-decoded STRING list),
not `canvas_trace`:** unlike `DGemmaTrace` (whose `consumers.analysis`
functions read `DiffusionFrame.canvas` tensors directly, no decode needed),
`consumers.tally_audit.audit_frames` operates on already-decoded per-step
text — the exact shape `DGemmaSampler`'s `frames` output already is
(`dgemma.loop.decode_frames` over `canvas_trace.frames`, one decode, reused
here as a third rendering rather than decoding a second time). Taking
`canvas_trace` instead would force this node to re-decode with a tokenizer
it would have to additionally carry — the same "payload-purity smell"
`surfaces/comfyui/sampler.py`'s docstring names for why `frames_image` is a
sampler output rather than a standalone `CANVAS_TRACE`-input node.

`INPUT_IS_LIST = True` (ComfyUI's per-input-list convention, not
`OUTPUT_IS_LIST`'s per-output twin): `frames` is `DGemmaSampler`'s
`OUTPUT_IS_LIST=True` STRING output, and this node needs the WHOLE ordered
list at once (frame-over-frame revision watching), not one call per
list element — the ComfyUI execution model runs a node once per item
unless `INPUT_IS_LIST` says otherwise. No denoising-step loop is added by
this: the list is already fully captured by the time it reaches this
node; `audit_frames` (consumer-tier) does the per-frame iteration, not this
adapter body.
"""
from __future__ import annotations

# Dual-context import, same depth/gate as surfaces/comfyui/trace.py (see
# that module's own comment for the full rationale) — this module lives at
# the same surfaces/comfyui/ depth, so the relative climb to consumers/ is
# the same three dots. No `DGEMMA_*` socket import needed here (unlike
# trace.py): this node's only input is a plain `STRING` list, and its
# output is a plain `STRING` — no native socket type crosses this node's
# boundary (rule 4 only applies where a `DGEMMA_*` mint entry is actually
# needed).
if __package__ and __package__.count(".") >= 2:
    from ...consumers.tally_audit import audit_frames
else:
    from consumers.tally_audit import audit_frames


def _format_report(audit) -> str:
    """Cheapest-correct STRING rendering (mirrors `trace.py`'s
    `_format_summary`): per-frame parse status line, the revision events
    observed, and the final-frame arithmetic-consistency verdict — the
    three pieces of evidence operator requirement (b)/(c)/(d) ask this
    node to surface, made visible on the graph rather than only in a unit
    test."""
    lines: list[str] = [f"frames={len(audit.frame_results)}"]
    for result in audit.frame_results:
        if result.parse_status == "ok":
            lines.append(f"  step {result.frame_idx}: ok ({result.format_name})")
        elif result.parse_status == "partial":
            unparsed = [n for n, cell in result.cells.items() if cell.claimed is None]
            lines.append(
                f"  step {result.frame_idx}: partial ({result.format_name}), "
                f"unparsed numerals={sorted(unparsed)}"
            )
        else:
            lines.append(f"  step {result.frame_idx}: unrecognized")

    if audit.revisions:
        lines.append("revisions:")
        for event in audit.revisions:
            lines.append(
                f"  numeral {event.numeral}: {event.from_value} -> {event.to_value} "
                f"(step {event.from_frame_idx} -> {event.to_frame_idx})"
            )
    else:
        lines.append("revisions: none observed")

    if audit.final_frame_arithmetically_consistent is None:
        lines.append("final tally: no parseable claim to check for arithmetic consistency")
    elif audit.final_frame_arithmetically_consistent:
        lines.append("final tally: arithmetically consistent with the model's own restated evidence")
    else:
        lines.append(
            "final tally: INCONSISTENT — claimed counts disagree with the model's own restated evidence"
        )
    return "\n".join(lines)


class DGemmaTallyAudit:
    """Audits a "count the numerals" run's decoded frames against the
    model's own restated evidence: per-step parse status, frame-over-frame
    revision events, and final-tally arithmetic consistency (issue #84)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("STRING", {"forceInput": True}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audit_report",)
    FUNCTION = "audit"
    CATEGORY = "DiffusionGemma"

    def audit(self, frames: list[str]):
        audit_result = audit_frames(list(frames))
        return (_format_report(audit_result),)
