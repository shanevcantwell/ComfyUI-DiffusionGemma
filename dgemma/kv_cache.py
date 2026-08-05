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

**ADR-CDG-024** (issue #257, prompt-under-injection composition):

- `prefill_templated_turn` — chat-templates a `prompt_kwargs` dict (the SAME
  dict `dgemma/loop.py`'s no-cache path builds) and prefills the resulting
  turn onto an already-injected cache, generalizing `encode_sequence`'s
  encoder-call shape to a templated turn instead of raw token ids. Consumed
  by `_run_pipeline_with_injected_cache`'s composed branch (`dgemma/loop.py`)
  when a non-empty `prompt` accompanies `kv_cache=`.

**Interim ingress guards (issues #263/#265, 2026-08-05):** `validate_kv_cache_
ingress`'s V7 check (new) rejects a cache grown in place by a prior composed
run since it was minted (#265 aliasing hazard). A second interim guard
against composed multi-block runs (#263's block>0 splice-offset defect) lives
in `dgemma/ingress.py`'s `reject_multi_block_composed_prefill` — see that
function's docstring. Both are named INTERIM in code and retire once their
underlying issue's root-cause fix lands; neither is a permanent invariant.

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

    `DiffusionGemmaConfig` is a **composite** config (issue #162): none of
    these four fields live top-level — all four live one level down, on the
    text sub-config. `config.get_text_config()`
    (`configuration_utils.py:1240`, always populated by the real class's
    `__post_init__`) resolves that sub-config. No `getattr(..., default)`
    fallback chain here (PARSE-AT-THE-DOOR) — a config shape missing one of
    these fields fails loud, rather than silently degrading to a fabricated
    default.
    """
    text_config = dgemma_model.model.config.get_text_config()
    return {
        "num_hidden_layers": text_config.num_hidden_layers,
        "layer_types": tuple(text_config.layer_types),
        "sliding_window": text_config.sliding_window,
        "rope_parameters": dict(text_config.rope_parameters),
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
    -> V6 (dtype/device) -> V5 (orphan) -> V7 (aliasing, issue #265). V5 is
    checked before V7 (not last) even though both are model-independent, so
    the orphan-provenance failure (a cache with no reproduction path at all)
    is reported before the aliasing failure (a cache whose reproduction path
    no longer matches its live tensors) — either order is defensible per the
    plan; this module picks one and is consistent about it.

    **V7 (issue #265, interim guard — not the root-cause fix):**
    `prefill_templated_turn` grows a `KVCache.cache` object IN PLACE (its own
    docstring below: "the SAME object, grown in place") — a composed run
    (`prompt=` + `kv_cache=`, ADR-CDG-024) mutates the live `DynamicCache`
    tensors the caller's `KVCache` payload still points at, but the
    payload's own `cumulative_length` (stamped at mint/advance time by
    `encode_sequence`) is never updated to match. A caller — most commonly
    ComfyUI's node-result cache reusing an unchanged `DGemmaEncode` output
    across two runs — can then hand `run_diffusion` a `KVCache` whose
    `payload.cache` has ALREADY been grown by a prior composed run's
    prefill, while `payload.cumulative_length` still reports the
    pre-growth length. Decoding against that cache silently inherits the
    prior run's prefilled turn — content this run never submitted (the live
    failure PR #262's acceptance run observed). V7 catches the mismatch at
    the door: if the cache's actual `get_seq_length()` exceeds what
    `cumulative_length` recorded at mint/advance time, the payload is
    stale — rejected, not silently decoded against. Retires when #265's
    root-cause fix (prefill onto a copy, or an equivalent non-mutating
    shape) lands; tracked as an interim invariant, not a permanent one,
    mirroring the retired issue #248 exclusivity guard's own precedent
    (`dgemma/ingress.py`'s `reject_prompt_and_kv_cache` tombstone).

    Every raise names BOTH the violated precondition AND the actionable
    remedy in one message (DV.3b, issue #62 implementation plan §C) — a
    cold user who mis-wires around the type system is told what is wrong
    and what to do, not handed a bare assertion.
    """
    # Composite config (issue #162): `num_hidden_layers` lives on the text
    # sub-config, not top-level — see `geometry_from_model`'s docstring.
    text_config = dgemma_model.model.config.get_text_config()

    # V1 — layer count of `cache` == loaded model's decoder-layer count.
    # Failure this prevents: a cache from a differently-sized model
    # attaching with a truncated/over-long layer set — silent wrong-geometry
    # attention (ADR-CDG-012 §D.3).
    #
    # transformers==5.13.0's real `DynamicCache` (grounded live against the
    # pinned install, issue #178): per-layer state lives on `.layers`, a
    # list of `DynamicLayer` objects (`cache_utils.py:1499-1603`), NOT on a
    # `.key_cache`/`.value_cache` list attribute (that shape was removed
    # upstream). `len(cache)` == `len(cache.layers)`
    # (`Cache.__len__`, `cache_utils.py:1143-1149`) — used directly here,
    # single pinned-API code path, no hasattr fallback (PARSE-AT-THE-DOOR).
    cache_layer_count = len(payload.cache)
    expected_layer_count = text_config.num_hidden_layers
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
    #
    # Real `DynamicLayer.keys` (`cache_utils.py:132-168`) — per-layer key
    # tensor, populated by `lazy_initialization`/`update`. Same `.layers`
    # surface as V1 above.
    cache_tensor = payload.cache.layers[0].keys if cache_layer_count else None
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

    # V7 (issue #265, interim guard) — the payload's own live cache must not
    # have grown PAST what its own cumulative_length recorded at mint/advance
    # time. `prefill_templated_turn` (ADR-CDG-024) grows `payload.cache` in
    # place without updating `payload.cumulative_length` to match — see this
    # function's own docstring above for the full failure-mode grounding.
    # `get_seq_length()` (no `layer_idx=`, defaulting to 0) matches the same
    # no-arg convention `dgemma/loop.py` already uses at every other
    # `past_key_values.get_seq_length()` call site on this path. Compared
    # against `cumulative_length[0]` — V3 above has already confirmed
    # `cumulative_length` is present and one-entry-per-layer, so index 0 is
    # always safe here (V1/V3 both precede V7 in this function's ordering).
    # A cache with zero layers (the legal V6-skip degenerate case) has
    # nothing to alias — `get_seq_length()` on an empty-`.layers` cache
    # returns 0, so this check is a no-op there rather than a spurious
    # reject.
    if cache_layer_count:
        actual_seq_length = payload.cache.get_seq_length()
        minted_length = cumulative_length[0]
        if actual_seq_length > minted_length:
            raise ValueError(
                f"KV_CACHE ingress V7 failed: cache has been grown in place "
                f"since it was minted — actual length {actual_seq_length} "
                f"exceeds the minted length {minted_length} recorded on this "
                "payload (issue #265). This is the everyday shape of "
                "ComfyUI reusing a cached DGemmaEncode node output that a "
                "PRIOR composed run (prompt + kv_cache, ADR-CDG-024) already "
                "prefilled and grew — decoding against it now would silently "
                "inherit that prior run's turn, content this run never "
                "submitted. Remedy: re-run DGemmaEncode to mint a fresh "
                "cache (change its input, or invalidate/bypass the node's "
                "cached result) rather than reusing this cache object across "
                "composed runs. Interim guard pending #265's root-cause fix "
                "(prefill onto a copy)."
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

    Device pinning (issue #187): `ids_tensor`/`position_ids` are minted
    directly on the encoder's own parameter device (`next(encoder.
    parameters()).device`) — parse-at-the-door for device identity, not a
    reliance on ambient accelerate hooks moving CPU-default tensors for us.
    Under bf16 spill, accelerate's `AlignDevicesHooks` happened to paper
    over CPU-minted inputs; under a whole-fit INT4 load (post-#183) no such
    hook exists, and a CPU `ids_tensor` reaching a CUDA-resident embedding
    weight crashed (`RuntimeError: ... index is on cpu, different from
    other tensors on cuda:0`). Resolving the device from the encoder itself
    (not `dgemma_model.device`/`_resolve_device`) holds correctly under all
    three regimes this function runs in: whole-fit (this fix's target),
    bf16 CPU spill (unaffected — the encoder's own parameters already carry
    the accelerator device attention needs), and CPU-only test fakes
    (encoder parameters report `cpu`, so minted tensors stay `cpu` too).

    Ingress validation (issue #227): `token_ids` empty (`len(token_ids) ==
    0`) is rejected here, at the door, before any tensor is minted or the
    encoder is touched. Left unguarded, an empty sequence reaches
    transformers' `find_packed_sequence_indices` (called deep inside the
    encoder forward) and raises an uncaught `IndexError` — a substrate
    crash instead of a typed ingress rejection (`EMIT-CANONICAL /
    PARSE-AT-THE-DOOR`). The empty-string case (`tokenizer("",
    add_special_tokens=False)["input_ids"] == []`) is the concrete producer
    of this shape (issue #227's title case); the check is on `token_ids`
    itself so it also catches a caller handing an empty list/tuple directly,
    not just the text-shaped path.

    OOM hardening (issue #226 hardening slice, NOT the root-cause fix — see
    `dgemma/kv_cache.py`'s `encode_sequence` OOM re-raise below): this is the
    **bare transformers lane** — the encoder is called directly, not through
    `DiffusionGemmaPipeline`'s own internal encode call, and OOM is a
    possible outcome here that the pipeline's own equivalent call has not
    been observed to hit under the same VRAM state (#226/#229). The
    `torch.OutOfMemoryError` re-raise below only wraps that lane's own
    forward call; it changes no happy-path behavior.

    Live-proof provenance (#228 pt.1): live-proven under bf16 CPU-spill via
    the ComfyUI server lane (gate run 4 S-B, 2026-07-30, `a68e29d`; #145) —
    regressed 2026-08-04, BARE lane only (#226). Standing surface: `tests/e2e/test_battery.py::test_encode_live`.
    """
    if len(token_ids) == 0:
        raise ValueError(
            "encode_sequence door rejected: token_ids is empty (len(token_ids) == 0). "
            "Remedy: encode_sequence requires at least one token id — a caller "
            "tokenizing an empty/whitespace-only string (e.g. "
            'tokenizer("", add_special_tokens=False)) must reject that text before '
            "calling encode_sequence, rather than minting a cache from zero tokens."
        )

    num_layers = geometry_from_model(dgemma_model)["num_hidden_layers"]

    if into is None:
        cache = None
        start_position = 0
    else:
        cache = into.cache
        start_position = into.cumulative_length[0] if into.cumulative_length else 0

    encoder = dgemma_model.model.model.encoder
    encoder_device = next(encoder.parameters()).device

    ids_tensor = torch.as_tensor(list(token_ids), dtype=torch.long, device=encoder_device)
    if ids_tensor.dim() == 1:
        ids_tensor = ids_tensor.unsqueeze(0)
    position_ids = torch.arange(ids_tensor.shape[-1], device=encoder_device) + start_position
    position_ids = position_ids.unsqueeze(0)

    try:
        outputs = encoder(input_ids=ids_tensor, past_key_values=cache, position_ids=position_ids)
    except torch.OutOfMemoryError as e:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            mem_info = f"{free_bytes / (1024 ** 3):.2f}/{total_bytes / (1024 ** 3):.2f} GiB free/total (cuda.mem_get_info)"
        else:
            mem_info = "torch.cuda.is_available() is False — no CUDA mem_get_info readback possible"
        raise torch.OutOfMemoryError(
            "encode_sequence: OutOfMemoryError in the bare transformers lane — "
            "the encoder is called directly here, not through "
            "DiffusionGemmaPipeline's own internal encode call, and OOM is a "
            "possible outcome specific to this lane (see #226/#229 — root cause "
            f"is a separate, still-open fix, not addressed by this re-raise). {mem_info}."
        ) from e
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


def prefill_templated_turn(
    pipeline: Any,
    cache: Any,
    prompt_kwargs: dict,
    *,
    add_generation_prompt: bool = True,
):
    """ADR-CDG-024 §1: chat-template `prompt_kwargs` exactly as the no-cache
    path does (`dgemma/loop.py`'s `prompt_kwargs` construction — the SAME
    dict, handed here instead of to `pipeline(...)`) and prefill the
    resulting turn onto an already-injected `cache` (a live `DynamicCache`,
    e.g. `KVCache.cache`), in place of `_run_pipeline_with_injected_cache`'s
    skipped "no `prompt` re-encode" (IN-2).

    Takes `pipeline` (a `DiffusionGemmaPipeline`/`DGemmaPipeline` instance,
    NOT the bare `DGemmaModel` wrapper `encode_sequence` above takes) because
    that is what `_run_pipeline_with_injected_cache` already has in scope —
    `pipeline.processor`/`pipeline.model` are the same objects
    `register_modules` wired at pipeline-construction time from the loaded
    `DGemmaModel`'s own `.processor`/`.model`.

    Mirrors the diffusers reference shape this composition generalizes:
    `pipeline_diffusion_gemma.py`'s own `_prepare_inputs` (`:119-125`) —
    `processor.apply_chat_template(messages, add_generation_prompt=...,
    tokenize=True, return_tensors="pt", return_dict=True)` — and
    `encode_sequence`'s own `encoder(input_ids=..., past_key_values=...,
    position_ids=...)` call shape (above), generalized to run once, before
    block 0's decode, for the templated turn instead of a committed canvas
    block (mechanically the same as the block>0 re-encode at
    `dgemma/loop.py:261-267`).

    `position_ids` continue from `cache`'s CURRENT length
    (`cache.get_seq_length()`, read once, immediately before this call) —
    same "continue from the cache's own advanced length" discipline
    `encode_sequence` uses for its `into=<KVCache>` advance case.

    Returns the advanced `cache` (the SAME object, grown in place by the
    real `DynamicCache.update`, matching `encode_sequence`'s
    `outputs.past_key_values` identity — mutated, not replaced). The caller
    (`_run_pipeline_with_injected_cache`) re-derives the templated-turn
    length via `cache.get_seq_length()` AFTER this call, per ADR-CDG-024 §4's
    named failure-mode prevention (position-id drift at the splice) — this
    function deliberately does not also return a separate token count, so
    there is no second length-tracking value that could drift from the
    cache's own state.

    `prompt_kwargs` is never re-derived or re-templated here — it is the
    exact dict `dgemma/loop.py`'s `run_diffusion` already builds once
    (`{"messages": [...]}` under `thinking=True`, `{"prompt": prompt}`
    otherwise) for the no-cache path, per ADR-CDG-024 §4's "one
    template-construction site, two consumers" failure-mode prevention
    (double-templating / silent divergence from the no-cache template
    shape).

    Normalization to the processor call (issue #257 live-failure fix,
    2026-08-05): `prompt_kwargs` is the PIPELINE-shaped dict (`prompt=` /
    `messages=` are `DiffusionGemmaPipeline.__call__` parameter names, not
    `apply_chat_template`'s — the real
    `ProcessorMixin.apply_chat_template(conversation, ...)` takes a required
    POSITIONAL `conversation` and would silently swallow a `prompt=` kwarg
    into `**kwargs` while raising on the missing positional). This function
    therefore mirrors the pipeline's own `_prepare_inputs` normalization
    verbatim (`pipeline_diffusion_gemma.py:112-125`): a bare `prompt` string
    is wrapped as a single user turn (`[{"role": "user", "content":
    prompt}]`, the image-free arm of `:117`) when `messages` is absent, and
    the resulting `messages` list is passed POSITIONALLY as `conversation`
    (`:119-125`). The batch (`isinstance(prompt, list)`) and image arms of
    `_prepare_inputs` are deliberately not mirrored: this path is
    batch-size-1 by contract (the drive body's own ingress assert) and
    `run_diffusion` never builds an image into `prompt_kwargs`."""
    prompt = prompt_kwargs.get("prompt")
    messages = prompt_kwargs.get("messages")
    if messages is None:
        # Same single-user-turn wrap as `_prepare_inputs`
        # (`pipeline_diffusion_gemma.py:117`, image-free arm).
        messages = [{"role": "user", "content": prompt}]

    encoded = pipeline.processor.apply_chat_template(
        messages,  # positional `conversation` — the real signature's required arg
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    input_ids = encoded["input_ids"]

    encoder = pipeline.model.model.encoder
    encoder_device = next(encoder.parameters()).device
    input_ids = input_ids.to(device=encoder_device)

    start_position = cache.get_seq_length()
    position_ids = torch.arange(
        start_position, start_position + input_ids.shape[-1], device=encoder_device
    ).unsqueeze(0)

    outputs = encoder(input_ids=input_ids, past_key_values=cache, position_ids=position_ids)
    return outputs.past_key_values
