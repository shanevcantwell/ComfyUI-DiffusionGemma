"""nodes/loader.py — DGemmaLoader: thin ComfyUI adapter (ADR-CDG-003).

Unpacks widget inputs, calls one `dgemma.*` function, wraps the result in a
tuple. No logic lives here — if a `for` loop or a loading decision ever
creeps into this file, it belongs in `dgemma/model.py`, not here.
"""
from __future__ import annotations

# Dual-context import, explicit package-depth gate (same discipline as the
# root __init__.py — no blanket try/except, which masks real failures).
# ComfyUI loads the pack as a package named after its directory path
# (`/srv/dev/ComfyUI/nodes.py:2233,2241`) and never puts the pack root on
# sys.path, so this module's __package__ is "<pack>.nodes" (dotted) and only
# the relative `..dgemma` can resolve. Under pytest/standalone the repo root
# is on sys.path and this module is top-level "nodes" (no dot), so only the
# absolute form can resolve. Observed violation: graph smoke test 2026-07-05
# (`loose-ends.md`); enforcement: tests/test_comfyui_loader_context.py.
if __package__ and "." in __package__:
    from ..dgemma.model import DEFAULT_QUANT, DEFAULT_REPO_ID, load_model
else:
    from dgemma.model import DEFAULT_QUANT, DEFAULT_REPO_ID, load_model


class DGemmaLoader:
    """Loads a DiffusionGemma model + processor onto the `DGEMMA_MODEL` socket."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": DEFAULT_REPO_ID}),
                # Default "none" (not "nf4"), issue #4: nf4 OOMs structurally
                # on this box (bnb can't quantize the fused 3D MoE experts) —
                # see dgemma/model.py's DEFAULT_QUANT provenance comment.
                "quant": (["nf4", "int8", "none"], {"default": DEFAULT_QUANT}),
            }
        }

    RETURN_TYPES = ("DGEMMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DiffusionGemma"

    def load(self, repo_id: str, quant: str):
        return (load_model(repo_id=repo_id, quant=quant),)
