"""surfaces/comfyui/encode.py — DGemmaEncode: thin ComfyUI adapter (ADR-CDG-003).

ADR-CDG-012 (issue #62 Phase 3): the `KV_CACHE` seam's mint/advance node.
Unpacks widget inputs (a text prompt, tokenized here — the one non-trivial
step, and still not denoising-loop logic), calls one `dgemma.*` function
(`dgemma.kv_cache.encode_sequence`), wraps the result on the `DGEMMA_KV_CACHE`
socket. No cache-advance logic lives here — the mint/advance body is entirely
`dgemma.kv_cache.encode_sequence`'s (rule 2).

IN-1 (fresh mint, no `kv_cache` input wired) / IN-3 (advance, `kv_cache`
wired) are the SAME node body: `encode_sequence(model, ids, into=kv_cache)`
already dispatches on `into is None` vs. not (ADR-CDG-012 §D.1). This mirrors
`DGemmaLoader`'s single-`load()`-body-handles-both-paths shape rather than
splitting into two node classes for what is one call with an optional arg.
"""
from __future__ import annotations

# Dual-context import, explicit package-depth gate — see
# surfaces/comfyui/loader.py for the full rationale (ComfyUI loader context
# vs. pytest/standalone). This module lives two levels under the pack root
# (surfaces/comfyui/), so the relative climb to dgemma/ is THREE dots
# (ADR-CDG-008 Phase 1 / issue #52 risk R-1). Gate is
# `__package__.count(".") >= 2`, not bare dot-presence — see loader.py's
# "GATE CORRECTION" comment. Issue #62 implementation plan §M: this file is a
# new consumer of the existing depth-2 predicate, not a fourth gate variant.
if __package__ and __package__.count(".") >= 2:
    from ...dgemma.kv_cache import encode_sequence
    from .socket_types import DGEMMA_KV_CACHE, DGEMMA_MODEL
else:
    from dgemma.kv_cache import encode_sequence
    from surfaces.comfyui.socket_types import DGEMMA_KV_CACHE, DGEMMA_MODEL


class DGemmaEncode:
    """Mints a fresh `DGEMMA_KV_CACHE` from a text prompt (IN-1), or advances
    an existing one with newly-committed text (IN-3, `kv_cache` wired)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (DGEMMA_MODEL,),
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "kv_cache": (DGEMMA_KV_CACHE,),
            },
        }

    RETURN_TYPES = (DGEMMA_KV_CACHE,)
    RETURN_NAMES = ("kv_cache",)
    FUNCTION = "encode"
    CATEGORY = "DiffusionGemma"

    def encode(self, model, text: str, kv_cache=None):
        # `PreTrainedTokenizerBase.encode(text) -> list[int]` (grounded
        # against the installed transformers 5.13.0) — the same
        # `getattr(processor, "tokenizer", processor)` unwrap
        # `dgemma.loop.resolve_vocab_size`/`resolve_thought_channel_ids` use.
        tokenizer = getattr(model.processor, "tokenizer", model.processor)
        token_ids = tokenizer.encode(text)
        return (encode_sequence(model, token_ids, into=kv_cache),)
