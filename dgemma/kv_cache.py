"""dgemma/kv_cache.py — `KV_CACHE` ingress validation + mint/advance helpers
(ADR-CDG-012, issue #62 Phases 1 + 3).

Engine-side, ComfyUI-agnostic (ADR-CDG-003) — the twin of `dgemma/hooks.py`:
one engine concern, one file, at `dgemma/` depth 1 (no new package, so the
existing depth-1/no-dual-context-gate story is unchanged; #57 blast-radius
unaffected on the core side — issue #62 implementation plan §M).

**Phase 1** (issue #62 ratification, Q-1: tier-2 OUT of first implementation)
landed:

- `geometry_from_model` / `tokenizer_fingerprint` — the fingerprint
  derivations `validate_kv_cache_ingress`'s V2/V4 checks compare a `KVCache`
  payload against.
- `validate_kv_cache_ingress` — the V1–V6 door validator (ADR-CDG-012 §D.3),
  fired at every `KV_CACHE` ingress (IN-2/IN-3/IN-4). Fail-on-mismatch, never
  trust-and-degrade (`EMIT-CANONICAL / PARSE-AT-THE-DOOR`).

**Phase 3** (issue #62 §A/§N, this module's new addition):

- `encode_sequence` — the mint/advance body IN-1 (fresh mint, `into=None`) /
  IN-3 (advance an existing cache, `into=<KVCache>`) feeds. A near-wrapper
  over the separately-callable encoder (ADR-CDG-012 Context: `model.model.
  encoder(input_ids=..., past_key_values=cache, position_ids=...)` is
  directly callable today — grounded against the installed transformers
  5.13.0 `DiffusionGemmaForBlockDiffusion.model.encoder` call path, verified
  this pass at `modeling_diffusion_gemma.py:1010-1160` (`DiffusionGemmaEncoderModel.
  forward`) and `:1495-1504` (`self.encoder = DiffusionGemmaEncoderModel(...)`)).
  This is the encoder-mint half of the seam — unlike `DGemmaDenoise`'s
  decoder-drive body, `encode_sequence` is NOT gated on the ADR's real-weights
  de-risk smoke test (issue #62 Q-2): that Open Question is scoped to "the
  **decoder** driven with a caller-built cache," and the encoder call this
  function wraps is already the pipeline's own unmodified first-encode path
  (`pipeline_diffusion_gemma.py`'s own per-block encode), not a novel drive
  shape.

**Explicitly NOT in this module yet** (later phases, not silently folded in):
`save_kv_cache`/`load_kv_cache` (IN-4's disk crossing) and any tier-2 surgery
op (`dgemma/kv_surgery.py`) are both Phase 5, conditional on operator scope
per issue #62 Q-1.
"""
from __future__ import annotations

from typing import Any

import torch

from .types import KVCache, Provenance


def geometry_from_model(dgemma_model: Any) -> dict:
    """Derive the geometry fingerprint (ADR-CDG-012 §2) a `KVCache.geometry`
    is validated against (V2). Reads the loaded model's config — the
    installed `transformers` `DiffusionGemmaTextConfig` fields grounded
    against the real class (`configuration_diffusion_gemma.py`):
    `num_hidden_layers`, `layer_types` (list of `"full_attention"` /
    `"sliding_attention"` per layer), `sliding_window`, `rope_parameters`
    (dict keyed by layer-type name).

    Returns a plain dict, comparable by `==` to a `KVCache.geometry` payload
    — this is deliberately the same shape a synthetic test fixture
    constructs (`tests/conftest.py`'s `synthetic_kv_cache`), so V2 is a
    structural equality check, not a bespoke per-field comparison that could
    silently skip a field neither side updated.
    """
    config = dgemma_model.model.config
    return {
        "num_hidden_layers": config.num_hidden_layers,
        "layer_types": tuple(config.layer_types),
        "sliding_window": config.sliding_window,
        "rope_parameters": dict(config.rope_parameters),
    }


def tokenizer_fingerprint(dgemma_model: Any) -> str:
    """Derive the vocab-alignment fingerprint (ADR-CDG-012 §D.0) a
    `Provenance.tokenizer_fingerprint` is validated against (V4).

    Combines `repo_id` (which checkpoint minted the tokenizer) with the
    tokenizer's own `vocab_size` — both cheap, always-present attributes
    that don't require loading anything beyond what `load_model` already
    loaded. This is a vocab-*alignment* check, not a cryptographic identity:
    it catches "wrong model's tokenizer" (different repo_id) and "same repo,
    differently configured/truncated vocab" (different vocab_size), which is
    exactly the failure ADR-CDG-012 §D.0 names ("a cache minted under a
    different tokenizer conditioning the canvas on token ids that mean
    something else").
    """
    tokenizer = getattr(dgemma_model.processor, "tokenizer", dgemma_model.processor)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    return f"{dgemma_model.repo_id}:{vocab_size}"


def validate_kv_cache_ingress(payload: KVCache, dgemma_model: Any) -> None:
    """Fire at every `KV_CACHE` ingress door (IN-2, IN-3, IN-4) before the
    payload is used. Fail-on-mismatch (rule 5, `EMIT-CANONICAL /
    PARSE-AT-THE-DOOR`) — raises `ValueError` on the first failing check,
    never silently degrades. Returns `None` on full pass.

    Ordering (ADR-CDG-012 §D.3 / issue #62 implementation plan §C): V1
    (layer count) -> V2 (geometry) -> V4 (vocab) -> V3 (cumulative_length)
    -> V6 (dtype/device) -> V5 (orphan). V5 is checked last here even
    though it is model-independent, so a caller always sees the
    model-alignment failures (if any) before the payload's own internal
    consistency failure — either order is defensible per the plan; this
    module picks one and is consistent about it.

    Every raise names BOTH the violated precondition AND the actionable
    remedy in one message (DV.3b, issue #62 implementation plan §C) — a
    cold user who mis-wires around the type system is told what is wrong
    and what to do, not handed a bare assertion.
    """
    model_config = dgemma_model.model.config

    # V1 — layer count of `cache` == loaded model's decoder-layer count.
    # Failure this prevents: a cache from a differently-sized model
    # attaching with a truncated/over-long layer set — silent wrong-geometry
    # attention (ADR-CDG-012 §D.3).
    cache_layer_count = len(payload.cache.key_cache)
    expected_layer_count = model_config.num_hidden_layers
    if cache_layer_count != expected_layer_count:
        raise ValueError(
            f"KV_CACHE ingress V1 failed: cache has {cache_layer_count} layers, "
            f"model expects {expected_layer_count}. "
            "Remedy: re-mint this cache with the loaded model (DGemmaEncode), "
            "or load the model that actually minted this cache."
        )

    # V2 — geometry.layer_types / sliding_window / RoPE params ==
    # model.config derivation. Failure this prevents: a cache built against
    # one layer-type pattern fed to another produces wrong masks with no
    # crash (the Neg-Consequences "silent geometry mismatch").
    expected_geometry = geometry_from_model(dgemma_model)
    payload_geometry = {
        "num_hidden_layers": payload.geometry.get("num_hidden_layers"),
        "layer_types": tuple(payload.geometry.get("layer_types") or ()),
        "sliding_window": payload.geometry.get("sliding_window"),
        "rope_parameters": dict(payload.geometry.get("rope_parameters") or {}),
    }
    if payload_geometry != expected_geometry:
        raise ValueError(
            f"KV_CACHE ingress V2 failed: cache geometry {payload_geometry} != "
            f"model geometry {expected_geometry} (layer_types/sliding_window/RoPE "
            "mismatch). "
            "Remedy: re-mint against this model; geometry is fixed by the loaded "
            "model, not by the cache."
        )

    # V4 — provenance.tokenizer_fingerprint / model_repo_id match the loaded
    # model. Failure this prevents: vocab misalignment — a cache minted
    # under a different tokenizer conditioning the canvas on token ids that
    # mean something else (orphan-provenance poisoning, vocab flavor).
    expected_fingerprint = tokenizer_fingerprint(dgemma_model)
    if payload.provenance.model_repo_id != dgemma_model.repo_id or (
        payload.provenance.tokenizer_fingerprint != expected_fingerprint
    ):
        raise ValueError(
            "KV_CACHE ingress V4 failed: cache minted under tokenizer "
            f"{payload.provenance.tokenizer_fingerprint!r} / repo "
            f"{payload.provenance.model_repo_id!r}, model loaded is "
            f"{expected_fingerprint!r} / {dgemma_model.repo_id!r}. "
            "Remedy: re-mint the cache with the matching model, or load the "
            "model that actually minted this cache."
        )

    # V3 — cumulative_length present, one entry per layer, all non-negative.
    # Failure this prevents: the ranked-#1 blocker — a stale/uninitialized
    # cumulative_length silently corrupting mask offsets
    # (`cache_utils.py:254,270`) — plausible-but-wrong mask, not a crash.
    cumulative_length = payload.cumulative_length
    if cumulative_length is None or len(cumulative_length) != expected_layer_count:
        got_len = 0 if cumulative_length is None else len(cumulative_length)
        raise ValueError(
            f"KV_CACHE ingress V3 failed: cumulative_length ragged/missing "
            f"(got len {got_len}, expected {expected_layer_count}). "
            "Remedy: re-encode via DGemmaEncode, which fills cumulative_length "
            "for every layer — never hand-track it."
        )
    if any(length < 0 for length in cumulative_length):
        raise ValueError(
            f"KV_CACHE ingress V3 failed: cumulative_length has a negative entry "
            f"{cumulative_length!r}. "
            "Remedy: re-encode via DGemmaEncode, which fills cumulative_length "
            "for every layer — never hand-track it."
        )

    # V6 — cache dtype/device match the loaded model. Failure this prevents:
    # a CPU-loaded or fp32 deserialized cache (IN-4) attaching to a
    # bf16-on-GPU model — device/dtype drift that would error deep in
    # attention rather than at the door.
    cache_tensor = payload.cache.key_cache[0] if cache_layer_count else None
    if cache_tensor is not None:
        cache_dtype = str(cache_tensor.dtype)
        cache_device = str(cache_tensor.device)
        model_dtype = dgemma_model.dtype
        model_device = dgemma_model.device
        dtype_ok = cache_dtype == model_dtype or cache_dtype.endswith(model_dtype)
        device_ok = cache_device == model_device or cache_device.startswith(model_device)
        if not (dtype_ok and device_ok):
            raise ValueError(
                f"KV_CACHE ingress V6 failed: cache dtype/device "
                f"{cache_dtype}/{cache_device} != model {model_dtype}/{model_device} "
                "(e.g. fp32-on-CPU vs bf16-on-GPU). "
                "Remedy: move/cast the cache to the model's device/dtype, or "
                "re-mint it on the loaded model."
            )

    # V5 — provenance non-orphan: NOT (minting_sequence is None AND
    # edit_script == ()). Failure this prevents: a cache with no
    # reproduction path at all — unreproducible, unauditable experimental
    # input (§D.0 illegal state).
    provenance = payload.provenance
    if provenance.minting_sequence is None and tuple(provenance.edit_script) == ():
        raise ValueError(
            "KV_CACHE ingress V5 failed: orphan cache — no minting_sequence and "
            "an empty edit_script, so there is no reproduction path. "
            "Remedy: supply the minting sequence (tier 1) or the edit-script "
            "(tier 2) that produced this cache."
        )


def encode_sequence(
    dgemma_model: Any,
    token_ids: "list[int] | tuple[int, ...]",
    *,
    into: "KVCache | None" = None,
) -> KVCache:
    """Mint (`into=None`, IN-1) or advance (`into=<KVCache>`, IN-3) a
    `KVCache` by running `token_ids` through the loaded model's encoder —
    the sole cache writer (ADR-CDG-012 Context: `modeling_diffusion_gemma.py`
    `DiffusionGemmaEncoderModel.forward`, `:1082-1160`, the only
    `past_key_values.update()` call path in the architecture).

    A near-wrapper (ADR-CDG-003's node/engine seam — no denoising-loop logic
    lives here, just the encoder call + provenance bookkeeping): calls
    `dgemma_model.model.model.encoder(input_ids=..., past_key_values=...,
    position_ids=...)` and returns a **new** `KVCache` (§3 advance-returns-
    new-payload — `into`'s own `cache`/`provenance` objects are never mutated
    in place; the encoder's `past_key_values.update()` mutates the
    `DynamicCache` object itself in transformers' own implementation, but
    this function still treats `into` as logically read-only from the
    caller's perspective by re-deriving every `KVCache` field fresh on the
    object the encoder call actually advanced, and by never handing back the
    SAME `Provenance`/`geometry` dict identity `into` held).

    `position_ids` continue from `into`'s `cumulative_length` (one shared
    running position, matching the encoder's own `past_seen_tokens` derivation
    at `:1131`) when advancing; start at 0 for a fresh mint.

    Provenance (§1, IN-1/IN-3):
    - fresh mint (`into=None`): `provenance.minting_sequence = tuple(token_ids)`,
      `edit_script = ()`, `model_repo_id`/`tokenizer_fingerprint` stamped from
      `dgemma_model` (IN-1).
    - advance (`into=<KVCache>`): `into`'s ingress is NOT re-validated here —
      the caller (`DGemmaEncode`'s IN-3 optional input) is responsible for
      having a valid `KVCache` already; `encode_sequence` extends
      `minting_sequence` by `token_ids` when `into.provenance.minting_sequence`
      is non-`None` (tier 1 stays tier 1, IN-3), and leaves a tier-2 cache's
      `None` minting_sequence untouched (advancing a tier-2 cache does not
      retroactively invent a tier-1 history) while still deep-copying
      `edit_script`/other provenance fields forward unchanged.

    `cumulative_length` (D.0 ranked-#1 blocker): derived fresh from the
    encoder-advanced cache via `cache.get_seq_length(layer_idx=i)` per layer
    — never hand-tracked, never copied from `into` (the encoder call itself
    is what advances it).

    Not gated on the ADR's real-weights de-risk smoke test (issue #62 Q-2):
    that Open Question is scoped to the **decoder** driven with a
    caller-built cache (`DGemmaDenoise`'s live drive body, Phase 4); this
    function wraps the encoder's own unmodified first-encode call path,
    already exercised by every existing `run_diffusion` call today.
    """
    num_layers = geometry_from_model(dgemma_model)["num_hidden_layers"]

    if into is None:
        cache = None
        start_position = 0
    else:
        cache = into.cache
        start_position = into.cumulative_length[0] if into.cumulative_length else 0

    ids_tensor = torch.as_tensor(list(token_ids), dtype=torch.long)
    if ids_tensor.dim() == 1:
        ids_tensor = ids_tensor.unsqueeze(0)
    position_ids = torch.arange(ids_tensor.shape[-1]) + start_position
    position_ids = position_ids.unsqueeze(0)

    encoder = dgemma_model.model.model.encoder
    outputs = encoder(input_ids=ids_tensor, past_key_values=cache, position_ids=position_ids)
    advanced_cache = outputs.past_key_values

    cumulative_length = tuple(advanced_cache.get_seq_length(layer_idx=i) for i in range(num_layers))
    geometry = geometry_from_model(dgemma_model)

    if into is None:
        provenance = Provenance(
            minting_sequence=tuple(token_ids),
            edit_script=(),
            model_repo_id=dgemma_model.repo_id,
            tokenizer_fingerprint=tokenizer_fingerprint(dgemma_model),
        )
    else:
        prior_minting_sequence = into.provenance.minting_sequence
        new_minting_sequence = (
            None if prior_minting_sequence is None else tuple(prior_minting_sequence) + tuple(token_ids)
        )
        provenance = Provenance(
            minting_sequence=new_minting_sequence,
            edit_script=tuple(into.provenance.edit_script),
            model_repo_id=into.provenance.model_repo_id,
            tokenizer_fingerprint=into.provenance.tokenizer_fingerprint,
        )

    return KVCache(
        cache=advanced_cache,
        cumulative_length=cumulative_length,
        geometry=geometry,
        provenance=provenance,
    )
