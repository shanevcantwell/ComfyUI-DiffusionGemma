"""nodes/loader.py — DGemmaLoader: thin ComfyUI adapter (ADR-CDG-003).

Unpacks widget inputs, calls one `dgemma.*` function, wraps the result in a
tuple. No logic lives here — if a `for` loop or a loading decision ever
creeps into this file, it belongs in `dgemma/model.py`, not here.
"""
from __future__ import annotations

from dgemma.model import DEFAULT_REPO_ID, load_model


class DGemmaLoader:
    """Loads a DiffusionGemma model + processor onto the `DGEMMA_MODEL` socket."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"default": DEFAULT_REPO_ID}),
                "quant": (["nf4", "int8", "none"], {"default": "nf4"}),
            }
        }

    RETURN_TYPES = ("DGEMMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DiffusionGemma"

    def load(self, repo_id: str, quant: str):
        return (load_model(repo_id=repo_id, quant=quant),)
