"""tests/test_kv_cache_ingress.py — ADR-CDG-012 Phase 1 (issue #62):
`dgemma.kv_cache.validate_kv_cache_ingress`'s V1-V6 branches, happy path plus
every raise path, each asserting DV.3b's both-token message contract
(precondition token AND remedy token, not a bare assertion).

Uses the `synthetic_kv_cache_factory` fixture (`tests/conftest.py`, §L) —
no real weights, every check exercised against a small fake model/cache
pair. `geometry_from_model`/`tokenizer_fingerprint` are exercised
incidentally (every V2/V4 check calls them); this file is also their only
direct coverage in Phase 1.
"""
from __future__ import annotations

import pytest

from dgemma.kv_cache import (
    encode_sequence,
    geometry_from_model,
    tokenizer_fingerprint,
    validate_kv_cache_ingress,
)


class TestHappyPath:
    def test_matching_tier1_cache_passes(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory()
        assert validate_kv_cache_ingress(cache, model) is None

    def test_matching_tier2_cache_passes(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(tier=2)
        assert validate_kv_cache_ingress(cache, model) is None

    def test_fake_dynamic_cache_get_seq_length_mirrors_real_surface(self, synthetic_kv_cache_factory):
        """`FakeDynamicCache.get_seq_length()` (`tests/conftest.py`, §L) mirrors
        `transformers.DynamicCache`'s surface a future phase's `encode_sequence`
        will read (`cached_len = past_key_values.get_seq_length()`, ADR-CDG-012
        grounding). Not called by `validate_kv_cache_ingress` today (it reads
        `key_cache`/`value_cache` directly) — this pins the fixture's own
        self-consistency now, so the method is exercised the moment it exists."""
        _, cache = synthetic_kv_cache_factory()
        assert cache.cache.get_seq_length() == cache.cache.key_cache[0].shape[2]


class TestGeometryFromModel:
    def test_derives_expected_fields(self, dgemma_model_factory):
        model = dgemma_model_factory(num_hidden_layers=6, sliding_window=16)
        geometry = geometry_from_model(model)
        assert geometry["num_hidden_layers"] == 6
        assert geometry["sliding_window"] == 16
        assert len(geometry["layer_types"]) == 6
        assert "rope_parameters" in geometry


class TestTokenizerFingerprint:
    def test_combines_repo_id_and_vocab_size(self, dgemma_model_factory):
        model = dgemma_model_factory(repo_id="fake/dgemma-test", vocab_size=32)
        fingerprint = tokenizer_fingerprint(model)
        assert "fake/dgemma-test" in fingerprint
        assert "32" in fingerprint


class TestV1LayerCountMismatch:
    def test_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="layer_count")
        with pytest.raises(ValueError, match="V1") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "layers" in message
        assert "re-mint" in message or "load the model" in message


class TestV2GeometryFingerprintMismatch:
    def test_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="geometry")
        with pytest.raises(ValueError, match="V2") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "geometry" in message
        assert "re-mint" in message


class TestV3MissingOrRaggedCumulativeLength:
    def test_ragged_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="cumulative_length_ragged")
        with pytest.raises(ValueError, match="V3") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "cumulative_length" in message
        assert "DGemmaEncode" in message

    def test_negative_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="cumulative_length_negative")
        with pytest.raises(ValueError, match="V3") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "cumulative_length" in message
        assert "DGemmaEncode" in message

    def test_none_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory()
        cache.cumulative_length = None
        with pytest.raises(ValueError, match="V3") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "cumulative_length" in message
        assert "DGemmaEncode" in message


class TestV4VocabMismatch:
    def test_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="vocab")
        with pytest.raises(ValueError, match="V4") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "tokenizer" in message or "repo" in message
        assert "re-mint" in message or "load the model" in message


class TestV5OrphanProvenance:
    def test_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="orphan")
        with pytest.raises(ValueError, match="V5") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "orphan" in message
        assert "minting sequence" in message or "edit-script" in message


class TestV6DtypeDeviceMismatch:
    def test_raises(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="dtype_device")
        with pytest.raises(ValueError, match="V6") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "dtype" in message or "device" in message
        assert "move/cast" in message or "re-mint" in message


class TestV6SkippedOnEmptyLayerCache:
    """The legal degenerate case: a 0-hidden-layer model (`num_hidden_layers=0`)
    paired with a matching 0-layer cache passes V1 (`cache_layer_count ==
    expected_layer_count == 0`), which makes `cache_tensor` (line 191)
    `None` — `payload.cache.key_cache[0] if cache_layer_count else None`
    short-circuits on the falsy `cache_layer_count`, never indexing an empty
    `key_cache` list. V6's dtype/device block (192->212) is then skipped
    entirely rather than raising or erroring, and ingress still completes
    (V5's orphan check still runs and still passes for a non-orphan
    provenance) — this is the documented skip-side behavior for the
    zero-layer edge, not an accident of a 0-length list happening not to
    crash."""

    def test_zero_layer_cache_skips_v6_and_passes(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(model_kwargs={"num_hidden_layers": 0})
        assert len(cache.cache.key_cache) == 0
        assert validate_kv_cache_ingress(cache, model) is None


class TestEncodeSequenceAlreadyBatchedIds:
    """`encode_sequence`'s `ids_tensor.dim() == 1` check (`dgemma/kv_cache.py`)
    guards against a caller passing already-batched (2-D) `token_ids` — the
    branch's false side (`284->286`, skipping the `unsqueeze(0)`) has no
    exerciser today: the sole real caller, `DGemmaEncode.encode`, always
    hands `encode_sequence` a flat `list[int]` straight from
    `tokenizer.encode(text)` (1-D by construction). This test constructs the
    2-D input directly to close the branch, and is honestly framed
    (repo coverage-residual convention, issue #75) as covering a defensive
    guard for a shape no current caller produces, not a real call path.
    """

    def test_2d_token_ids_skip_the_unsqueeze(self, dgemma_model_factory):
        model = dgemma_model_factory()
        # A list-of-one-list is already (batch=1, seq_len) shaped once
        # `torch.as_tensor` sees it — `ids_tensor.dim() == 1` is False, so
        # the `unsqueeze(0)` on the next line must NOT fire.
        cache = encode_sequence(model, [[1, 2, 3]], into=None)
        assert cache.cache.key_cache[0].shape[2] == 3


class TestOrderingIsDeterministic:
    """V1 fires before V2/V4/V3/V6/V5 when multiple checks would fail —
    pins the ordering the module docstring commits to, so a future edit that
    reorders checks changes this test deliberately, not silently."""

    def test_layer_count_mismatch_reported_before_geometry(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(mismatch="layer_count")
        # Also corrupt geometry — V1 must still fire first.
        cache.geometry["sliding_window"] += 1
        with pytest.raises(ValueError, match="V1"):
            validate_kv_cache_ingress(cache, model)
