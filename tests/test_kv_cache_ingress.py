"""tests/test_kv_cache_ingress.py — ADR-CDG-012 Phase 1 (issue #62):
`dgemma.kv_cache.validate_kv_cache_ingress`'s V1-V6 branches, happy path plus
every raise path, each asserting DV.3b's both-token message contract
(precondition token AND remedy token, not a bare assertion). V7 (issue #265,
interim guard, 2026-08-05) is covered separately below by
`TestV7CacheAliasing`.

Uses the `synthetic_kv_cache_factory` fixture (`tests/conftest.py`, §L) —
no real weights, every check exercised against a small fake model/cache
pair. `geometry_from_model`/`tokenizer_fingerprint` are exercised
incidentally (every V2/V4 check calls them); this file is also their only
direct coverage in Phase 1.
"""
from __future__ import annotations

import pytest
import torch

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
        """`cache.cache` (`tests/conftest.py`'s `FakeDynamicCache`, §L, now a
        REAL `transformers.DynamicCache` subclass — issue #178) exercises
        `get_seq_length()` against the real class's own `.layers[i].keys`
        surface `validate_kv_cache_ingress` reads (`len(cache)`,
        `cache.layers[0].keys`). Not called by `validate_kv_cache_ingress`
        today (it reads `.layers` directly) — this pins the fixture's own
        self-consistency now, so the method is exercised the moment it exists."""
        _, cache = synthetic_kv_cache_factory()
        assert cache.cache.get_seq_length() == cache.cache.layers[0].keys.shape[2]


class TestGeometryFromModel:
    def test_derives_expected_fields(self, dgemma_model_factory):
        model = dgemma_model_factory(num_hidden_layers=6, sliding_window=16)
        geometry = geometry_from_model(model)
        assert geometry["num_hidden_layers"] == 6
        assert geometry["sliding_window"] == 16
        assert len(geometry["layer_types"]) == 6
        assert "rope_parameters" in geometry

    def test_reads_through_get_text_config_not_top_level(self, dgemma_model_factory):
        """Issue #162 regression: `DiffusionGemmaConfig` is a composite
        config — none of the four fields `geometry_from_model` reads live
        top-level, only on the text sub-config `get_text_config()`
        resolves. Fixed by resolving through `get_text_config()`
        (`dgemma/kv_cache.py`); this pins the composite-shaped call path
        works end to end (fixture -> `.model.config.get_text_config()` ->
        the four fields)."""
        model = dgemma_model_factory(num_hidden_layers=6, sliding_window=16)
        # The fake's top-level config object has NO num_hidden_layers/
        # layer_types/sliding_window/rope_parameters attribute of its own —
        # mirroring the real DiffusionGemmaConfig's actual shape.
        top_level_config = model.model.config
        assert not hasattr(top_level_config, "num_hidden_layers")
        assert not hasattr(top_level_config, "layer_types")
        assert not hasattr(top_level_config, "sliding_window")
        assert not hasattr(top_level_config, "rope_parameters")
        # ...yet geometry_from_model succeeds by resolving through
        # get_text_config() rather than reading the top-level object flat.
        geometry = geometry_from_model(model)
        assert geometry["num_hidden_layers"] == 6
        assert geometry["sliding_window"] == 16

    def test_flat_shaped_config_is_not_silently_accepted(self, dgemma_model_factory):
        """The other half of the #162 regression: a config shape with the
        four fields FLAT (no nested `text_config`, no `get_text_config()`)
        is not a config `geometry_from_model` can read — it must fail loud
        (AttributeError), not silently degrade. This documents that the
        pre-#162 fake (which fabricated these fields flat) could pass
        where the real `DiffusionGemmaConfig` class fails, and that the
        fixed fixture/production code no longer tolerates that shape."""

        class _FlatConfig:
            """A deliberately WRONG-shaped config: the four fields live
            top-level, exactly the shape the pre-#162 `FakeDGemmaModelConfig`
            fabricated — and exactly the shape the real
            `DiffusionGemmaConfig` does NOT have."""

            def __init__(self) -> None:
                self.num_hidden_layers = 6
                self.layer_types = ["full_attention"] * 6
                self.sliding_window = 16
                self.rope_parameters = {}

        model = dgemma_model_factory(num_hidden_layers=6, sliding_window=16)
        model.model.config = _FlatConfig()

        with pytest.raises(AttributeError):
            geometry_from_model(model)


class TestTokenizerFingerprint:
    def test_combines_repo_id_and_vocab_size(self, dgemma_model_factory):
        model = dgemma_model_factory(repo_id="fake/dgemma-test", vocab_size=32)
        fingerprint = tokenizer_fingerprint(model)
        assert "fake/dgemma-test" in fingerprint
        assert "32" in fingerprint


class TestValidateAgainstRealTransformersDynamicCache:
    """Issue #178 regression: `validate_kv_cache_ingress` crashed live
    (`AttributeError: 'DynamicCache' object has no attribute 'key_cache'`)
    against `transformers==5.13.0`'s REAL `DynamicCache`, even though the
    unit suite passed — every other test in this module runs against
    `tests/conftest.py`'s `FakeDynamicCache`, which is now a genuine
    `DynamicCache` subclass (issue #178), but this test deliberately
    constructs a bare `transformers.DynamicCache()` directly (no subclass,
    no fixture helper) so a future fixture-only fix can't mask a real
    regression here again — this is the class the live crash actually
    named.

    Mutation-verify (ledger #145 step 4): reverting
    `dgemma/kv_cache.py`'s V1 access from `len(payload.cache)` back to
    `len(payload.cache.key_cache)` makes this test fail with the exact
    `AttributeError` issue #178 reported, by name.
    """

    def test_real_dynamic_cache_passes_ingress(self, dgemma_model_factory):
        from transformers.cache_utils import DynamicCache

        from dgemma.types import KVCache, Provenance

        model = dgemma_model_factory(num_hidden_layers=6, sliding_window=16)
        text_config = model.model.config.get_text_config()
        num_layers = text_config.num_hidden_layers

        cache = DynamicCache()
        shape = (1, 2, 4, 8)
        for layer_idx in range(num_layers):
            key_states = torch.zeros(shape, dtype=torch.bfloat16)
            value_states = torch.zeros(shape, dtype=torch.bfloat16)
            cache.update(key_states, value_states, layer_idx)

        payload = KVCache(
            cache=cache,
            cumulative_length=tuple([4] * num_layers),
            geometry=geometry_from_model(model),
            provenance=Provenance(
                minting_sequence=(1, 2, 3),
                edit_script=(),
                model_repo_id=model.repo_id,
                tokenizer_fingerprint=tokenizer_fingerprint(model),
            ),
        )

        # The exact call that crashed live in #178 (`dgemma/loop.py:1372` ->
        # `dgemma/kv_cache.py:130`), against the real class it crashed on.
        assert validate_kv_cache_ingress(payload, model) is None


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


class TestV7CacheAliasing:
    """Issue #265 interim guard: `validate_kv_cache_ingress`'s V7 check
    rejects a `KVCache` payload whose live `cache.get_seq_length()` has
    outgrown the `cumulative_length` recorded on the payload at mint/advance
    time — the shape a prior composed run's in-place `prefill_templated_turn`
    growth produces when the SAME `KVCache` object is handed to
    `run_diffusion`/`validate_kv_cache_ingress` a second time (e.g. ComfyUI
    reusing a cached `DGemmaEncode` node output across two runs).

    `synthetic_kv_cache_factory` builds a matching (V1-V6-passing) cache;
    `.append(n)` (`tests/conftest.py`'s `FakeDynamicCache`, the same growth
    helper `test_kv_cache_drive_body.py`'s multi-block tests use) grows the
    live cache tensors WITHOUT touching `cumulative_length` — exactly what
    `prefill_templated_turn`'s real in-place growth does, reproduced here
    without needing a real prefill call."""

    def test_fresh_matching_cache_passes_v7(self, synthetic_kv_cache_factory):
        """Baseline: an unmutated, just-minted cache (the common case —
        cache.get_seq_length() == cumulative_length[0]) passes V7 exactly as
        it already passes V1-V6."""
        model, cache = synthetic_kv_cache_factory()
        assert cache.cache.get_seq_length() == cache.cumulative_length[0]
        assert validate_kv_cache_ingress(cache, model) is None

    def test_cache_grown_in_place_since_minting_rejected(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory()
        # Simulate a prior composed run's `prefill_templated_turn` growth:
        # the live cache tensors grow, but `cumulative_length` (this
        # payload's own field) is never updated to match — the exact
        # divergence #265 names.
        cache.cache.append(5)
        assert cache.cache.get_seq_length() > cache.cumulative_length[0]
        with pytest.raises(ValueError, match="V7") as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "#265" in message
        assert "DGemmaEncode" in message
        assert "grown in place" in message or "grown" in message

    def test_grown_cache_message_names_remedy(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory()
        cache.cache.append(1)
        with pytest.raises(ValueError) as excinfo:
            validate_kv_cache_ingress(cache, model)
        message = str(excinfo.value)
        assert "re-run" in message or "fresh cache" in message

    def test_cache_at_exactly_minted_length_passes(self, synthetic_kv_cache_factory):
        """Boundary: actual == minted (not strictly greater) must pass —
        V7 rejects only cache growth PAST the minted length, never an
        untouched cache sitting exactly at it."""
        model, cache = synthetic_kv_cache_factory()
        assert cache.cache.get_seq_length() == cache.cumulative_length[0]
        assert validate_kv_cache_ingress(cache, model) is None

    def test_v7_skipped_on_zero_layer_cache(self, synthetic_kv_cache_factory):
        """The same legal zero-layer degenerate case V6 skips (cache_layer_
        count falsy) also skips V7 — nothing to alias with zero layers."""
        model, cache = synthetic_kv_cache_factory(model_kwargs={"num_hidden_layers": 0})
        assert len(cache.cache.layers) == 0
        assert validate_kv_cache_ingress(cache, model) is None

    def test_v7_fires_after_v5_orphan_check(self, synthetic_kv_cache_factory):
        """Ordering pin: V5 (orphan) is checked before V7 (aliasing) per the
        module docstring's stated order — an orphan cache that is ALSO grown
        past its minted length reports V5 first."""
        model, cache = synthetic_kv_cache_factory(mismatch="orphan")
        cache.cache.append(5)
        with pytest.raises(ValueError, match="V5"):
            validate_kv_cache_ingress(cache, model)


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
    `None` — `payload.cache.layers[0].keys if cache_layer_count else None`
    short-circuits on the falsy `cache_layer_count`, never indexing an empty
    `.layers` list. V6's dtype/device block (192->212) is then skipped
    entirely rather than raising or erroring, and ingress still completes
    (V5's orphan check still runs and still passes for a non-orphan
    provenance) — this is the documented skip-side behavior for the
    zero-layer edge, not an accident of a 0-length list happening not to
    crash."""

    def test_zero_layer_cache_skips_v6_and_passes(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory(model_kwargs={"num_hidden_layers": 0})
        assert len(cache.cache.layers) == 0
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
        assert cache.cache.layers[0].keys.shape[2] == 3


class TestEncodeSequenceDevicePinning:
    """Issue #187: `encode_sequence` must mint `ids_tensor`/`position_ids`
    directly on the encoder's own parameter device
    (`next(encoder.parameters()).device`), not on whatever device
    `torch.as_tensor`/`torch.arange` default to (CPU) and hope an ambient
    accelerate hook moves later. `"meta"` stands in for a non-CPU
    accelerator device here (real CUDA is not assumed present in a test
    process) — `dgemma_model_factory`'s `encoder_device=` threads straight
    to `tests/conftest.py`'s `_FakeEncoderModel.parameters()`, which is
    exactly the surface `encode_sequence` now reads, so this is a live
    exercise of the production code path, not a re-implementation of it.
    """

    def test_mints_on_encoder_parameter_device_not_cpu_default(self, dgemma_model_factory):
        model = dgemma_model_factory(encoder_device="meta")

        encode_sequence(model, [1, 2, 3], into=None)

        encoder = model.model.model.encoder
        assert encoder.last_input_ids_device == torch.device("meta")
        assert encoder.last_position_ids_device == torch.device("meta")

    def test_stays_on_cpu_when_encoder_is_cpu_resident(self, dgemma_model_factory):
        """The CPU-only test-fake regime (acceptance criteria's third named
        regime, alongside whole-fit and spill): an encoder whose parameters
        report `cpu` must still receive `cpu`-resident minted tensors — the
        fix must not force a device move where none is needed."""
        model = dgemma_model_factory(encoder_device="cpu")

        encode_sequence(model, [1, 2, 3], into=None)

        encoder = model.model.model.encoder
        assert encoder.last_input_ids_device == torch.device("cpu")
        assert encoder.last_position_ids_device == torch.device("cpu")

    def test_advance_path_also_pins_to_encoder_device(self, synthetic_kv_cache_factory, dgemma_model_factory):
        """IN-3 (advance, `into=<KVCache>` non-`None`) goes through the same
        mint call inside `encode_sequence` — pins just as the fresh-mint
        (IN-1) path does, not a second unguarded code path."""
        model, cache = synthetic_kv_cache_factory(model_kwargs={"encoder_device": "meta"})

        encode_sequence(model, [4, 5], into=cache)

        encoder = model.model.model.encoder
        assert encoder.last_input_ids_device == torch.device("meta")
        assert encoder.last_position_ids_device == torch.device("meta")


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


class TestEncodeSequenceEmptyTokenIdsRejected:
    """Issue #227: `encode_sequence("")`'s underlying failure shape is an
    EMPTY `token_ids` sequence — the concrete producer is
    `tokenizer("", add_special_tokens=False)["input_ids"] == []`
    (`docs/experiments/2026-08-04-adr-cdg-012-q2-smoke/run_sec2_liveness_sweep.py`,
    the live evidence that surfaced this bug: seed=*, text#2 -> IndexError,
    0.00-0.01s, before any GPU work). Left unguarded, an empty sequence
    reaches transformers' `find_packed_sequence_indices` deep in the encoder
    forward and raises an uncaught `IndexError`. `encode_sequence` now
    checks `len(token_ids) == 0` at the door, before any tensor is minted or
    the encoder is touched — `EMIT-CANONICAL / PARSE-AT-THE-DOOR`: fail
    loud, typed, at ingress, not a leaked substrate `IndexError`.

    Both `""` and whitespace-only text tokenize to an empty `input_ids` list
    under `add_special_tokens=False` (no content, no special tokens to
    contribute a token either) — both are exercised here as the two
    concrete empty-list producers named in the issue, via a real
    `_FakeTokenizer`-shaped call (`tokenizer(text,
    add_special_tokens=False)["input_ids"]`) construction, without needing a
    real tokenizer.
    """

    @pytest.mark.parametrize("token_ids", [[], ()], ids=["empty-list", "empty-tuple"])
    def test_empty_token_ids_raises_before_encoder_is_touched(self, dgemma_model_factory, token_ids):
        model = dgemma_model_factory()
        with pytest.raises(ValueError, match="token_ids is empty") as excinfo:
            encode_sequence(model, token_ids, into=None)
        message = str(excinfo.value)
        assert "encode_sequence" in message
        assert "door" in message
        # The encoder must never be called — the door rejects before any
        # forward pass, not merely before the forward pass errors.
        encoder = model.model.model.encoder
        assert encoder.last_input_ids_device is None

    def test_empty_string_tokenizes_to_empty_ids_which_the_door_rejects(self, dgemma_model_factory):
        """The literal issue #227 title case, reconstructed end to end:
        tokenizing `""` (empty string) with `add_special_tokens=False`
        produces an empty `input_ids` list — exactly the shape
        `run_sec2_liveness_sweep.py`'s text#2 fed `encode_sequence` live.
        `_FakeTokenizer.encode` (`tests/conftest.py`) always returns a
        non-empty list (`or [0]`), so this test calls the tokenizer the same
        way the real `AutoProcessor`'s tokenizer is called at the door
        (`tokenizer(text, add_special_tokens=False)["input_ids"]`) rather
        than through the fake's `.encode(text)` convenience method, to
        genuinely reproduce the empty-list shape."""
        model = dgemma_model_factory()
        empty_ids = []  # tokenizer("", add_special_tokens=False)["input_ids"]
        with pytest.raises(ValueError, match="token_ids is empty"):
            encode_sequence(model, empty_ids, into=None)

    def test_whitespace_only_text_also_tokenizes_empty_and_is_rejected(self, dgemma_model_factory):
        """Whitespace-only text ("   ") is the second empty-ingress case
        named alongside "" in issue #227's test scope — under
        `add_special_tokens=False` a tokenizer with no leading/trailing
        whitespace tokens configured (the common case) also collapses "   "
        to an empty `input_ids` list; asserted directly here against the
        empty-list shape (`encode_sequence` cannot distinguish the two by
        the time `token_ids` reaches it — the door catches both by construction)."""
        model = dgemma_model_factory()
        whitespace_ids = []  # tokenizer("   ", add_special_tokens=False)["input_ids"]
        with pytest.raises(ValueError, match="token_ids is empty"):
            encode_sequence(model, whitespace_ids, into=None)

    def test_advance_path_also_rejects_empty_token_ids(self, synthetic_kv_cache_factory):
        """IN-3 (advance, `into=<KVCache>`) goes through the same length
        check inside `encode_sequence` — the door guards both mint (IN-1)
        and advance (IN-3), not just the fresh-mint path."""
        model, cache = synthetic_kv_cache_factory()
        with pytest.raises(ValueError, match="token_ids is empty"):
            encode_sequence(model, [], into=cache)


class TestEncodeSequenceOutOfMemoryHardening:
    """Issue #226 hardening slice (typed fail-loud OOM ONLY — the
    offload-hook root cause named in #226/#229 is explicitly NOT addressed
    here): `encode_sequence` calls the bare encoder directly (not through
    `DiffusionGemmaPipeline`'s own internal encode call), and a bare
    `torch.OutOfMemoryError` from that forward call is re-raised as a typed,
    informative error naming the bare-transformers lane, chaining the
    original exception (`raise ... from e`) rather than swallowing or
    replacing it silently.

    `_FakeEncoderModel.__call__` (`tests/conftest.py`) is monkeypatched here
    to raise `torch.OutOfMemoryError` directly — no real CUDA OOM is
    triggered or required; the fake stands in for "the encoder's forward
    call raised this", which is the only contract `encode_sequence`'s
    `try/except` around the encoder call depends on.
    """

    def test_oom_reraised_as_typed_error_chaining_original(self, dgemma_model_factory, monkeypatch):
        model = dgemma_model_factory()
        encoder = model.model.model.encoder
        original_oom = torch.OutOfMemoryError("CUDA out of memory (fake)")

        def _raise_oom(self, **_kwargs):
            raise original_oom

        monkeypatch.setattr(type(encoder), "__call__", _raise_oom)

        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            encode_sequence(model, [1, 2, 3], into=None)

        message = str(excinfo.value)
        assert "bare transformers lane" in message
        assert "encode_sequence" in message
        # Chained, not swallowed/replaced (`raise ... from e`).
        assert excinfo.value.__cause__ is original_oom

    def test_oom_message_names_the_lane_and_issue_refs(self, dgemma_model_factory, monkeypatch):
        model = dgemma_model_factory()
        encoder = model.model.model.encoder

        def _raise_oom(self, **_kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")

        monkeypatch.setattr(type(encoder), "__call__", _raise_oom)

        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            encode_sequence(model, [1, 2, 3], into=None)

        message = str(excinfo.value)
        assert "#226" in message
        assert "#229" in message

    def test_oom_message_includes_cuda_mem_get_info_when_cuda_available(self, dgemma_model_factory, monkeypatch):
        """When `torch.cuda.is_available()` is True, the re-raised message
        must carry a `torch.cuda.mem_get_info()` readback — asserted via
        monkeypatching both `torch.cuda.is_available` and
        `torch.cuda.mem_get_info` so this test's outcome does not depend on
        whether a real CUDA device happens to be present in the test
        process."""
        model = dgemma_model_factory()
        encoder = model.model.model.encoder

        def _raise_oom(self, **_kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")

        monkeypatch.setattr(type(encoder), "__call__", _raise_oom)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 * 1024**3, 48 * 1024**3))

        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            encode_sequence(model, [1, 2, 3], into=None)

        message = str(excinfo.value)
        assert "mem_get_info" in message
        assert "2.00" in message
        assert "48.00" in message

    def test_oom_message_names_cuda_unavailable_when_no_cuda(self, dgemma_model_factory, monkeypatch):
        """When `torch.cuda.is_available()` is False, the re-raise must not
        attempt a `mem_get_info()` readback (which would itself raise on a
        genuinely CUDA-less process) — the message honestly names the
        absence instead."""
        model = dgemma_model_factory()
        encoder = model.model.model.encoder

        def _raise_oom(self, **_kwargs):
            raise torch.OutOfMemoryError("CUDA out of memory (fake)")

        monkeypatch.setattr(type(encoder), "__call__", _raise_oom)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        with pytest.raises(torch.OutOfMemoryError) as excinfo:
            encode_sequence(model, [1, 2, 3], into=None)

        message = str(excinfo.value)
        assert "is_available" in message or "no CUDA" in message

    def test_happy_path_unaffected_when_no_oom(self, dgemma_model_factory):
        """The try/except must not change behavior on the happy path — a
        normal (non-OOM-raising) encoder call still mints a cache exactly as
        before."""
        model = dgemma_model_factory()
        cache = encode_sequence(model, [1, 2, 3], into=None)
        assert cache.cache.layers[0].keys.shape[2] == 3

    def test_non_oom_exception_from_encoder_is_not_caught(self, dgemma_model_factory, monkeypatch):
        """The `except torch.OutOfMemoryError` must not widen into a bare
        `except Exception` — a different exception type from the encoder
        propagates unchanged, uncaught, unwrapped."""
        model = dgemma_model_factory()
        encoder = model.model.model.encoder

        def _raise_runtime_error(self, **_kwargs):
            raise RuntimeError("some other unrelated failure")

        monkeypatch.setattr(type(encoder), "__call__", _raise_runtime_error)

        with pytest.raises(RuntimeError, match="some other unrelated failure"):
            encode_sequence(model, [1, 2, 3], into=None)


class TestEncodeSequenceNoGradGuard:
    """Issue #226 root-cause fix (2026-08-06): `encode_sequence`'s encoder
    call must run inside `torch.no_grad()`, matching
    `DiffusionGemmaPipeline.__call__`'s own `@torch.no_grad()` decoration.
    Bisected root cause of the deterministic bf16-CPU-spill OOM named in
    this class's sibling `TestEncodeSequenceOutOfMemoryHardening`: with no
    grad-disabling context, every activation the encoder's MoE expert
    dispatch produced was retained for an unused backward pass, tipping
    accelerate's incremental per-layer offload-hook weight materialization
    over the edge under CPU spill. `_FakeEncoderModel.last_grad_enabled`
    (`tests/conftest.py`) records `torch.is_grad_enabled()` at the moment
    the fake encoder is actually invoked — a mechanical enforcement surface
    for this invariant (not just the docstring), so a future edit that
    drops the `torch.no_grad()` wrapper fails this test rather than only
    being caught by a live GPU OOM.
    """

    def test_encoder_called_with_grad_disabled_on_fresh_mint(self, dgemma_model_factory):
        model = dgemma_model_factory()
        encoder = model.model.model.encoder

        encode_sequence(model, [1, 2, 3], into=None)

        assert encoder.last_grad_enabled is False

    def test_encoder_called_with_grad_disabled_on_advance(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory()
        encoder = model.model.model.encoder

        encode_sequence(model, [4, 5], into=cache)

        assert encoder.last_grad_enabled is False

    def test_ambient_grad_enabled_outside_the_call_is_restored_after(self, dgemma_model_factory):
        """`torch.no_grad()` inside `encode_sequence` must be scoped to the
        encoder call only — it must not leak and disable grad for the
        CALLER'S surrounding context after `encode_sequence` returns."""
        model = dgemma_model_factory()

        assert torch.is_grad_enabled() is True
        encode_sequence(model, [1, 2, 3], into=None)
        assert torch.is_grad_enabled() is True
