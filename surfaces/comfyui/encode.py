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

    DESCRIPTION = (
        "Mints or advances a DiffusionGemma KV-cache from raw-encoded "
        "text (context, not a chat-templated prompt/turn). The optional "
        "kv_cache input is a dual door: leave it UNWIRED to mint a fresh "
        "cache from text alone (IN-1); WIRE an existing kv_cache in to "
        "ADVANCE it with the newly-committed text (IN-3). Same node, one "
        "call — the presence of a wired cache is the only thing that "
        "switches mint vs. advance. Feed the output cache into "
        "DGemmaDenoise's kv_cache input to condition a run on it — the "
        "turn text itself belongs on DGemmaDenoise's prompt widget. "
        "Caution: if DGemmaDenoise's decoder composes a prompt onto this "
        "node's output cache (ADR-CDG-024), the cache object is grown in "
        "place; ComfyUI's node-result caching can then silently hand a "
        "later run the already-grown cache. If you see the wrong/stale "
        "turn echoed back, invalidate this node (change its input, or "
        "force re-execution) rather than reusing its cached output (#265)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    DGEMMA_MODEL,
                    {"tooltip": "Loaded DiffusionGemma model (from DGemmaLoader)."},
                ),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Raw-encoded CONTEXT — NOT a prompt slot. "
                            "Tokenized directly (no chat role markers, no "
                            "generation-prompt suffix, no thinking "
                            "mechanism) and conditions as background; it "
                            "cannot restrict output the way a prompt does. "
                            "The current-turn text belongs on DGemmaDenoise's "
                            "prompt widget, not here. On a fresh mint this "
                            "is the full context; when advancing a wired "
                            "cache, this is the newly-committed "
                            "continuation."
                        ),
                    },
                ),
            },
            "optional": {
                "kv_cache": (
                    DGEMMA_KV_CACHE,
                    {
                        "tooltip": (
                            "UNWIRED = mint a fresh cache from text. WIRED "
                            "= advance the incoming cache with text. This "
                            "one connection is what switches mint vs. "
                            "advance — there is no separate mode widget."
                        )
                    },
                ),
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
