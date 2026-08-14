"""surfaces/mcp/state_manager.py — this surface's cross-call state: the
loaded `DGemmaModel` (ADR-CDG-008 Phase 2 Correction 1, `STATELESS-CORE`),
and, per ADR-CDG-025's rule-6 amendment, a bounded KV-cache handle registry
scoped to that model.

Transcribed from `semantic-kinematics-mcp`'s `mcp/state_manager.py` shape (a
dataclass the server constructs once and hands to every command handler),
deliberately NARROWED against that source's own documented debt: sk-mcp's
`StateManager` retains a live `_adapter` **and** a cross-call
`_embedding_cache` (`.../mcp/state_manager.py:51-52,83-86`), which its own
ADR-SKM-0009 names as the statelessness violation still on its roadmap to fix
(`ADR-SKM-0009:71`). This `StateManager` deliberately holds nothing else: no
scheduler, no canvas, no run-state, no cache keyed on prompt/knobs. The
model load is the one object this pack's own doctrine says must persist
(the ~53GB weights, `README.md` local-run defaults) — every `generate` call
still goes through `dgemma.run_diffusion`, which builds its own fresh
`EntropyBoundScheduler` / `_FrameCollector` / `StepEndComposite` internally
(`dgemma/loop.py:run_diffusion`) — this surface adds no memoization of any
of that on top.

**ADR-CDG-025 §1 (rule-6 amendment): a second, explicitly bounded persisted
class.** `StateManager` now also holds a KV-cache registry — entries keyed
by an opaque server-minted handle, evicted wholesale on every `load()` call
(§4: "no partial survival... a cache is scoped to the model *object* that
minted it, not the repo_id string"), and checked for model-identity at
resolve time (`resolve_kv_cache`'s `Provenance` comparison), not just at
mint time. This is NOT the `STATELESS-CORE` violation rule 6 forecloses in
general — it is the ONE additional persisted class ADR-CDG-025 explicitly
sanctions, with its own lifecycle contract (bounded count, evict-on-reload,
identity-checked-at-resolve). A future third persisted class still needs its
own ADR amendment (ADR-CDG-025 §1: "this is not an open door").

`load_model`/`is_loaded`/`model_status`/`encode_into_registry`/
`resolve_kv_cache` are the whole surface: nothing here memoizes a scheduler,
a partial canvas, or a prompt/knobs-keyed run result. A future addition that
stores anything ELSE (a scheduler, a canvas, a last-prompt cache) is exactly
the ARCHITECTURE.md rule-6 violation this module exists to foreclose — see
`tests/test_mcp_statelessness.py`, which mutation-checks this file directly
(asserts a hypothetical cached-scheduler shape, or an unbounded/unevicted
third persisted class, would be caught).
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

if __package__ and __package__.count(".") >= 2:
    from ...dgemma.kv_cache import encode_sequence, tokenizer_fingerprint
    from ...dgemma.model import load_model
    from ...dgemma.types import DGemmaModel, KVCache
else:
    from dgemma.kv_cache import encode_sequence, tokenizer_fingerprint
    from dgemma.model import load_model
    from dgemma.types import DGemmaModel, KVCache

# ADR-CDG-025 §4 "Bounded registry — no unbounded growth": a small fixed
# slot count with LRU eviction (the open question's "implementer's call" —
# resolved here as fixed-count-plus-LRU rather than a byte-size ceiling,
# since a `KVCache`'s per-layer tensor footprint isn't cheaply knowable from
# this module without reaching into the live `DynamicCache`, while a slot
# count is a plain, auditable constant). 8 concurrent handles mirrors the
# ADR's own example figure (§4).
DEFAULT_KV_CACHE_REGISTRY_CAPACITY = 8


@dataclass
class StateManager:
    """Holds the loaded model plus (ADR-CDG-025) a bounded, model-scoped
    KV-cache handle registry. Constructed once by `server.py`, passed to
    every command handler — same shape as sk-mcp's `StateManager`, with the
    embedding-cache / live-adapter cross-call state (that source's own named
    debt) not transcribed at all.
    """

    _model: Optional[DGemmaModel] = field(default=None, repr=False)
    _repo_id: Optional[str] = None
    _quant: Optional[str] = None
    # ADR-CDG-025 §4: an `OrderedDict` gives LRU-on-capacity eviction for
    # free (move-to-end on every touch, popitem(last=False) evicts the
    # least-recently-used entry) without a second bookkeeping structure.
    # `repr=False` — a live `KVCache` holds `transformers.DynamicCache`
    # tensors; letting `repr()`/dataclass equality walk into those would be
    # both slow and useless for debugging output.
    _kv_cache_registry: "OrderedDict[str, KVCache]" = field(default_factory=OrderedDict, repr=False)
    _kv_cache_registry_capacity: int = DEFAULT_KV_CACHE_REGISTRY_CAPACITY

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, *, repo_id: str, quant: str, local_files_only: bool = False) -> DGemmaModel:
        """(Re)load the model, replacing whatever was previously held.

        No implicit reuse-if-same-args short-circuit: a caller that wants to
        avoid a reload checks `model_status` first (`is_loaded` +
        `repo_id`/`quant`) and skips the call itself — this method always
        does what it's told, so "did a real load just happen" never has to
        be inferred from a cache-hit side effect.

        ADR-CDG-025 §4 eviction trigger: every `load()` call clears the
        ENTIRE KV-cache registry first — "no partial survival, no 'same
        repo_id so keep the caches' heuristic... a cache is scoped to the
        model *object* that minted it, not the repo_id string." This runs
        even when the load below fails/raises: a caller that just asked for
        a (possibly different) model must never have a stale handle resolve
        against whatever was previously loaded.
        """
        self._kv_cache_registry.clear()
        self._model = load_model(repo_id=repo_id, quant=quant, local_files_only=local_files_only)
        self._repo_id = repo_id
        self._quant = quant
        return self._model

    def require_model(self) -> DGemmaModel:
        """The model, or a loud `RuntimeError` naming the missing precondition
        — never a silent `None` handed on to `dgemma.run_diffusion` (which
        would fail with a confusing attribute error deep inside the engine
        instead of a clear message at the door)."""
        if self._model is None:
            raise RuntimeError(
                "No DiffusionGemma model is loaded. Call the 'load_model' tool "
                "first (with an explicit repo_id + quant) before 'generate'."
            )
        return self._model

    def status(self) -> dict:
        return {
            "is_loaded": self.is_loaded,
            "repo_id": self._repo_id,
            "quant": self._quant,
            "device": self._model.device if self._model is not None else None,
        }

    def _store_kv_cache(self, kv_cache: KVCache) -> str:
        """Mint a fresh opaque handle (ADR-CDG-025 §Open Questions: "a
        server-minted opaque handle satisfies the design as written") and
        store `kv_cache` under it, evicting the least-recently-used entry
        first if the registry is already at capacity."""
        while len(self._kv_cache_registry) >= self._kv_cache_registry_capacity:
            self._kv_cache_registry.popitem(last=False)
        handle = uuid.uuid4().hex
        self._kv_cache_registry[handle] = kv_cache
        return handle

    def encode_into_registry(self, prompt: str, kv_cache_id: Optional[str] = None) -> str:
        """ADR-CDG-025 §2: `encode` tool body. Resolves `kv_cache_id` (if
        given) against the registry with the SAME fail-on-unknown posture
        `resolve_kv_cache` uses, tokenizes `prompt` with the loaded model's
        tokenizer (raw `tokenizer.encode`, no chat template — mirrors
        `DGemmaEncode`'s `surfaces/comfyui/encode.py:104-111` deliberately
        raw-encode contract), calls `dgemma.kv_cache.encode_sequence` to
        mint (`into=None`) or advance (`into=<resolved cache>`), and stores
        the result.

        Advance-returns-new-payload (ADR-CDG-012 §3, this ADR's §2): when
        `kv_cache_id` names an existing handle, that handle's registry entry
        is REPLACED by the new `KVCache` `encode_sequence` returns — same
        handle string, new value — rather than minting a second handle per
        advance.
        """
        model = self.require_model()
        into = self.resolve_kv_cache(kv_cache_id) if kv_cache_id is not None else None

        tokenizer = getattr(model.processor, "tokenizer", model.processor)
        token_ids = tokenizer.encode(prompt)
        new_cache = encode_sequence(model, token_ids, into=into)

        if kv_cache_id is not None:
            # Replace the input handle's entry with the output — mirrors
            # DGemmaEncode's "new payload, not a mutation" contract without
            # inventing a second handle per advance (ADR-CDG-025 §2).
            self._kv_cache_registry[kv_cache_id] = new_cache
            self._kv_cache_registry.move_to_end(kv_cache_id)
            return kv_cache_id
        return self._store_kv_cache(new_cache)

    def resolve_kv_cache(self, kv_cache_id: str) -> KVCache:
        """The stored `KVCache` for `kv_cache_id`, or a loud `ValueError`
        naming the handle and why it's unresolvable — never a silent `None`
        handed on to `run_diffusion` (ADR-CDG-025 §3: "the same fail-loud
        posture `state_manager.require_model()` already takes for a missing
        model load").

        Two failure classes, both fail-loud (§3):
        - unknown/dangling handle (not in the registry at all — evicted by
          capacity, evicted by a model reload, or simply never minted);
        - identity mismatch (§4 "belt-and-suspenders... a mint-identity
          guard applied to a live object"): the entry's `Provenance.
          model_repo_id`/`tokenizer_fingerprint` no longer match the
          CURRENTLY loaded model. `load()` already clears the whole registry
          on every reload, so this should be unreachable in practice; the
          check makes that foreclosure structural rather than
          sequencing-dependent (a future code path that swaps `_model`
          without going through `load()` would still be caught here).

        A resolve touch moves the entry to most-recently-used (LRU
        eviction, §4) — reading a handle keeps it alive, the same as
        `encode`'s advance path already does via `move_to_end`.
        """
        model = self.require_model()
        if kv_cache_id not in self._kv_cache_registry:
            raise ValueError(
                f"Unknown or expired kv_cache_id: {kv_cache_id!r}. It may have been "
                "evicted (registry capacity, or a model reload since it was minted) "
                "or never returned by 'encode'. Call 'encode' again to mint a fresh "
                "handle — never silently substituted with an unconditioned cache."
            )
        cache = self._kv_cache_registry[kv_cache_id]
        if (
            cache.provenance.model_repo_id != model.repo_id
            or cache.provenance.tokenizer_fingerprint != tokenizer_fingerprint(model)
        ):
            raise ValueError(
                f"kv_cache_id {kv_cache_id!r} was minted under a different model "
                f"(cache.provenance.model_repo_id={cache.provenance.model_repo_id!r}) than "
                f"the currently loaded {model.repo_id!r}. A cache never survives its "
                "minting model (ADR-CDG-025 §4) — call 'encode' again against the "
                "currently loaded model."
            )
        self._kv_cache_registry.move_to_end(kv_cache_id)
        return cache
