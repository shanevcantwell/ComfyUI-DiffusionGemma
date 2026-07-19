"""tests/test_kv_cache_run_diffusion.py — ADR-CDG-012 Phase 2 (issue #62):
`run_diffusion`'s `kv_cache=` door (IN-2), fired against the fake pipeline
(no real weights).

Scope, per the ratified Phase 2 plan (issue #62 §N): `kv_cache=None` is
byte-identical to today; `kv_cache=<valid synthetic>` ingress-validates
(`dgemma.kv_cache.validate_kv_cache_ingress`) BEFORE the scheduler/pipeline
are constructed, stamps `CanvasTrace.injected_cache_provenance` (OUT-3), and
never mutates the input `KVCache` payload (§3 advance-returns-new-payload —
Phase 2 has no OUT-1/OUT-2 emit path of its own; that is the Phase-3 node
bodies'). The live decoder-drive body stays a skeleton (Phase 4, gated on the
ADR's real-weights de-risk smoke test) — these tests do not, and must not,
assert anything about the decoder actually consuming the injected cache's
tensors.

Fixture composition note: `dgemma_model_factory`/`synthetic_kv_cache_factory`
(`tests/conftest.py` §L) build a `DGemmaModel` whose `.processor` is the bare
`_FakeProcessor`/`_FakeTokenizer` pair — sufficient for
`validate_kv_cache_ingress` (which only reads `.model.config` and
`.processor.tokenizer.vocab_size`) but NOT for driving `run_diffusion` to a
built `CanvasState`/`CanvasTrace` (`_build_result` needs a real `.decode`/
`.eos_token_id`). This module therefore builds its own decode-capable
`DGemmaModel` (mirroring `tests/test_run_diffusion_statelessness.py`'s
`_fake_model()`), but with a REAL `FakeDGemmaModelConfig`-backed `.model.config`
so `synthetic_kv_cache`'s geometry/fingerprint fixtures line up — see
`_kv_capable_fake_model()` below.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.kv_cache import geometry_from_model, tokenizer_fingerprint
from dgemma.loop import run_diffusion
from dgemma.types import DGemmaModel, EditOp, KVCache, Provenance
from tests.conftest import FakeDGemmaModelConfig, FakeDynamicCache


class _FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0
    vocab_size = 32

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _FakeInnerModel:
    def __init__(self, config: FakeDGemmaModelConfig) -> None:
        self.config = config


def _kv_capable_fake_model(*, repo_id: str = "fake/dgemma-test") -> DGemmaModel:
    """A `DGemmaModel` decode-capable enough to drive `run_diffusion` to
    completion (unlike `tests/conftest.py`'s bare `fake_dgemma_model`), while
    still exposing the real `FakeDGemmaModelConfig`-shaped `.model.config`
    `synthetic_kv_cache`'s geometry/fingerprint derivation reads. No
    `constraints=`/`logit_hook=` is ever passed in this module, so
    `install_logit_shaping_hook` never calls `register_forward_hook` — this
    `.model` need not support it (mirrors `_fake_model()`'s reasoning in
    `tests/test_run_diffusion_statelessness.py`, adapted for a config-bearing
    inner model instead of a hook-capable one)."""
    config = FakeDGemmaModelConfig(num_hidden_layers=6, sliding_window=16)
    return DGemmaModel(
        model=_FakeInnerModel(config),
        processor=_FakeProcessor(),
        device="cpu",
        dtype="bfloat16",
        repo_id=repo_id,
        quant="none",
    )


def _matching_kv_cache(model: DGemmaModel, *, minting_sequence=(1, 2, 3)) -> KVCache:
    """Builds a `KVCache` whose geometry/fingerprint match `model` exactly —
    the local twin of `tests/conftest.py`'s `synthetic_kv_cache`, reused
    directly here (same shape) so this module doesn't duplicate the mismatch
    matrix ingress already owns (Phase 1's `test_kv_cache_ingress.py`)."""
    config = model.model.config
    cache = FakeDynamicCache(num_layers=config.num_hidden_layers)
    geometry = geometry_from_model(model)
    return KVCache(
        cache=cache,
        cumulative_length=tuple([0] * config.num_hidden_layers),
        geometry=geometry,
        provenance=Provenance(
            minting_sequence=minting_sequence,
            edit_script=(),
            model_repo_id=model.repo_id,
            tokenizer_fingerprint=tokenizer_fingerprint(model),
        ),
    )


def _install_fakes(monkeypatch, *, num_steps: int = 2):
    """Installs the same shape of `EntropyBoundScheduler`/`DGemmaPipeline`
    fakes `tests/test_run_diffusion_statelessness.py` uses — a scheduler
    exposing `.config`/`.num_inference_steps`/`register_to_config`, and a
    pipeline whose `__call__` drives the callback `num_steps` times then
    returns a fixed-length sequence. No cache-aware behavior: Phase 2's fake
    pipeline does not need to consume `kv_cache` at all, because the live
    drive body is Phase 4's — these tests exercise the ingress+trace-stamp
    skeleton, not decoder behavior."""

    class FakeSchedulerOutput:
        def __init__(self, accepted):
            self.accepted_index = torch.tensor(accepted, dtype=torch.bool)

    class FakePipelineOutput:
        def __init__(self, sequences):
            self.sequences = sequences
            self.texts = ["<<unused>>"]

    class _RecordingFrozenConfig:
        def __init__(self, **kwargs):
            object.__setattr__(self, "_values", dict(kwargs))

        def __getattr__(self, name):
            values = object.__getattribute__(self, "_values")
            if name in values:
                return values[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            raise AttributeError(f"frozen — use register_to_config, not direct set of {name!r}")

    class FakeScheduler:
        def __init__(self, *, entropy_bound, t_max, t_min, num_inference_steps):
            self._config = _RecordingFrozenConfig(
                entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
            )
            self.num_inference_steps = num_inference_steps

        @property
        def config(self):
            return self._config

        def register_to_config(self, **kwargs):
            merged = dict(object.__getattribute__(self._config, "_values"))
            merged.update(kwargs)
            self._config = _RecordingFrozenConfig(**merged)

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self._scheduler = scheduler
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            for step_idx in range(num_steps):
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True]]),
                    "canvas": torch.tensor([[step_idx]]),
                }
                callback(self, step_idx, step_idx, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor([num_steps], dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestKVCacheNoneUnchanged:
    """`kv_cache=None` (the default) is byte-identical to `run_diffusion`'s
    pre-Phase-2 behavior — the additive-optional discipline's core claim,
    made an assertion rather than left as an unexercised default."""

    def test_default_omits_kv_cache_produces_no_injected_provenance(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=2)
        model = _kv_capable_fake_model()

        _, _, trace = run_diffusion(model, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2)

        assert trace.injected_cache_provenance is None

    def test_explicit_none_matches_omitted_default(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=2)
        model_a = _kv_capable_fake_model()
        model_b = _kv_capable_fake_model()

        _, state_a, trace_a = run_diffusion(
            model_a, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )
        _, state_b, trace_b = run_diffusion(
            model_b, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=None
        )

        assert trace_a.scheduler_config == trace_b.scheduler_config
        assert state_a.committed_fraction == state_b.committed_fraction
        assert trace_a.injected_cache_provenance == trace_b.injected_cache_provenance is None


class TestKVCacheValidIngress:
    """`kv_cache=<valid synthetic>` ingress-validates and stamps OUT-3 on the
    fake pipeline — Phase 2's positive path."""

    def test_valid_cache_stamps_injected_cache_provenance(self, monkeypatch):
        _install_fakes(monkeypatch, num_steps=2)
        model = _kv_capable_fake_model()
        cache = _matching_kv_cache(model, minting_sequence=(4, 5, 6))

        _, _, trace = run_diffusion(
            model, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=cache
        )

        assert trace.injected_cache_provenance is not None
        assert trace.injected_cache_provenance == cache.provenance
        assert trace.injected_cache_provenance.minting_sequence == (4, 5, 6)
        assert trace.injected_cache_provenance.model_repo_id == model.repo_id

    def test_valid_cache_does_not_mutate_input_payload(self, monkeypatch):
        """§3 advance-returns-new-payload, Phase-2 slice: `run_diffusion`
        reads the injected payload (to validate + stamp OUT-3) but never
        writes through it — the cache object, cumulative_length, geometry,
        and provenance identity are all the exact objects/values supplied,
        untouched, after the call returns."""
        _install_fakes(monkeypatch, num_steps=2)
        model = _kv_capable_fake_model()
        cache = _matching_kv_cache(model)
        original_cumulative_length = cache.cumulative_length
        original_provenance = cache.provenance
        original_cache_obj = cache.cache

        run_diffusion(model, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=cache)

        assert cache.cache is original_cache_obj
        assert cache.cumulative_length == original_cumulative_length
        assert cache.provenance is original_provenance
        assert cache.provenance.minting_sequence == (1, 2, 3)

    def test_two_identical_kv_cache_calls_yield_identical_provenance_stamp(self, monkeypatch):
        """Same-in/same-out rider (statelessness, rule 6): two independent
        `run_diffusion(kv_cache=...)` calls with equal-but-distinct payloads
        produce equal (not aliased) `injected_cache_provenance` stamps, and
        neither call's scheduler config leaks into the other's."""
        _install_fakes(monkeypatch, num_steps=2)
        model_a = _kv_capable_fake_model()
        model_b = _kv_capable_fake_model()
        cache_a = _matching_kv_cache(model_a)
        cache_b = _matching_kv_cache(model_b)

        _, _, trace_a = run_diffusion(
            model_a, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=cache_a
        )
        _, _, trace_b = run_diffusion(
            model_b, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=cache_b
        )

        assert trace_a.injected_cache_provenance == trace_b.injected_cache_provenance
        assert trace_a.injected_cache_provenance is not trace_b.injected_cache_provenance


class TestKVCacheInvalidIngressRejectedBeforeSchedulerConstruction:
    """Ingress fires BEFORE the scheduler/pipeline are constructed (rule 5,
    EMIT-CANONICAL / PARSE-AT-THE-DOOR) — a bad injected cache never ties up
    either resource. Proven by asserting the monkeypatched
    scheduler/pipeline classes are never instantiated when ingress rejects."""

    def test_layer_count_mismatch_raises_before_scheduler_built(self, monkeypatch):
        constructed: list = []

        class TrackingScheduler:
            def __init__(self, **kwargs):
                constructed.append("scheduler")

        class TrackingPipeline:
            def __init__(self, **kwargs):
                constructed.append("pipeline")

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", TrackingScheduler)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", TrackingPipeline)

        model = _kv_capable_fake_model()
        good_cache = _matching_kv_cache(model)
        # Defeat V1 (layer count): drop the last cache layer.
        bad_cache = KVCache(
            cache=FakeDynamicCache(num_layers=model.model.config.num_hidden_layers - 1),
            cumulative_length=good_cache.cumulative_length[:-1],
            geometry=good_cache.geometry,
            provenance=good_cache.provenance,
        )

        with pytest.raises(ValueError, match="V1"):
            run_diffusion(
                model, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=bad_cache
            )

        assert constructed == []

    def test_orphan_provenance_raises_before_scheduler_built(self, monkeypatch):
        constructed: list = []

        class TrackingScheduler:
            def __init__(self, **kwargs):
                constructed.append("scheduler")

        class TrackingPipeline:
            def __init__(self, **kwargs):
                constructed.append("pipeline")

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", TrackingScheduler)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", TrackingPipeline)

        model = _kv_capable_fake_model()
        good_cache = _matching_kv_cache(model)
        orphan_cache = KVCache(
            cache=good_cache.cache,
            cumulative_length=good_cache.cumulative_length,
            geometry=good_cache.geometry,
            provenance=Provenance(
                minting_sequence=None,
                edit_script=(),
                model_repo_id=model.repo_id,
                tokenizer_fingerprint=tokenizer_fingerprint(model),
            ),
        )

        with pytest.raises(ValueError, match="V5"):
            run_diffusion(
                model, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=orphan_cache
            )

        assert constructed == []

    def test_tier2_cache_with_edit_script_passes_ingress(self, monkeypatch):
        """Non-orphan tier-2 shape (`minting_sequence=None`,
        non-empty `edit_script`) is legal input to the ingress door even
        though no tier-2 surgery op exists yet (Phase 5, out of scope) — the
        `Provenance` dataclass shape alone is what V5 checks."""
        _install_fakes(monkeypatch, num_steps=2)
        model = _kv_capable_fake_model()
        good_cache = _matching_kv_cache(model)
        tier2_cache = KVCache(
            cache=good_cache.cache,
            cumulative_length=good_cache.cumulative_length,
            geometry=good_cache.geometry,
            provenance=Provenance(
                minting_sequence=None,
                edit_script=(EditOp(op="ablate_full_attention", params={}),),
                model_repo_id=model.repo_id,
                tokenizer_fingerprint=tokenizer_fingerprint(model),
            ),
        )

        _, _, trace = run_diffusion(
            model, "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2, kv_cache=tier2_cache
        )

        assert trace.injected_cache_provenance.minting_sequence is None
        assert len(trace.injected_cache_provenance.edit_script) == 1
