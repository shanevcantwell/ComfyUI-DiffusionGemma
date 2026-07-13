"""tests/test_kv_cache_types.py — ADR-CDG-012 Phase 1 (issue #62): `KVCache`/
`Provenance`/`EditOp` construction, the illegal orphan state as pure data
(§D.0), and `CanvasTrace.injected_cache_provenance`'s additive-optional
default.

Pure-data tests — no ingress validation here (that's
`test_kv_cache_ingress.py`); this file only pins the dataclass shapes
transcribed from ADR-CDG-012 §D.0/§B.
"""
from __future__ import annotations

from dgemma.types import CanvasTrace, DiffusionFrame, EditOp, KVCache, Provenance


class TestEditOp:
    def test_construction(self):
        op = EditOp(op="ablate", params={"layer_indices": [0, 6]})
        assert op.op == "ablate"
        assert op.params == {"layer_indices": [0, 6]}


class TestProvenance:
    def test_tier1_shape(self):
        prov = Provenance(
            minting_sequence=(1, 2, 3),
            edit_script=(),
            model_repo_id="google/diffusiongemma-26B-A4B-it",
            tokenizer_fingerprint="google/diffusiongemma-26B-A4B-it:262144",
        )
        assert prov.minting_sequence == (1, 2, 3)
        assert prov.edit_script == ()

    def test_tier2_shape(self):
        edit = EditOp(op="scale", params={"factor": 0.5})
        prov = Provenance(
            minting_sequence=None,
            edit_script=(edit,),
            model_repo_id="google/diffusiongemma-26B-A4B-it",
            tokenizer_fingerprint="google/diffusiongemma-26B-A4B-it:262144",
        )
        assert prov.minting_sequence is None
        assert prov.edit_script == (edit,)

    def test_illegal_orphan_state_is_constructible_as_data(self):
        """§D.0: `minting_sequence is None and edit_script == ()` is the
        ILLEGAL orphan state — but as a dataclass, construction itself does
        not raise (Python dataclasses don't self-validate). Rejecting it is
        `validate_kv_cache_ingress`'s V5 job (test_kv_cache_ingress.py), not
        the dataclass's. This test pins that the *type layer* permits
        constructing the state (so the ingress validator has something to
        reject), not that it is a valid *use*.
        """
        prov = Provenance(
            minting_sequence=None,
            edit_script=(),
            model_repo_id="google/diffusiongemma-26B-A4B-it",
            tokenizer_fingerprint="google/diffusiongemma-26B-A4B-it:262144",
        )
        assert prov.minting_sequence is None
        assert prov.edit_script == ()


class TestKVCache:
    def test_construction(self):
        prov = Provenance(
            minting_sequence=(1, 2, 3),
            edit_script=(),
            model_repo_id="google/diffusiongemma-26B-A4B-it",
            tokenizer_fingerprint="google/diffusiongemma-26B-A4B-it:262144",
        )
        cache = KVCache(
            cache=object(),
            cumulative_length=(3, 3, 3),
            geometry={"num_hidden_layers": 3},
            provenance=prov,
        )
        assert cache.cumulative_length == (3, 3, 3)
        assert cache.provenance is prov


class TestCanvasTraceInjectedCacheProvenance:
    """OUT-3 (ADR-CDG-012 §D.2): additive-optional field, default None,
    identity only."""

    def _frame(self) -> DiffusionFrame:
        return DiffusionFrame(
            canvas_idx=0,
            step_idx=0,
            t=1.0,
            temperature=0.8,
            committed_fraction_per_example=(1.0,),
            canvas=None,
        )

    def test_default_is_none(self):
        trace = CanvasTrace(frames=[self._frame()], scheduler_name="EntropyBoundScheduler", scheduler_config={})
        assert trace.injected_cache_provenance is None

    def test_populated_carries_identity_not_tensors(self):
        prov = Provenance(
            minting_sequence=(1, 2, 3),
            edit_script=(),
            model_repo_id="google/diffusiongemma-26B-A4B-it",
            tokenizer_fingerprint="google/diffusiongemma-26B-A4B-it:262144",
        )
        trace = CanvasTrace(
            frames=[self._frame()],
            scheduler_name="EntropyBoundScheduler",
            scheduler_config={},
            injected_cache_provenance=prov,
        )
        assert trace.injected_cache_provenance is prov
        # OUT-3 carries identity only — no tensor field on Provenance itself.
        assert not hasattr(trace.injected_cache_provenance, "cache")
