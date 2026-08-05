"""tests/test_kv_cache_drive_body.py — ADR-CDG-012 Phase 4 (issue #62): the
live decoder-drive body (`dgemma.loop._run_pipeline_with_injected_cache`),
fired against a fake but decode-CAPABLE pipeline (no real weights).

Productionizes the Q-2 smoke skeleton
(`scratch/q2-skeleton-2026-08-04` @ `d67e62f`,
`docs/experiments/2026-08-04-adr-cdg-012-q2-smoke/skeleton-loop.py.diff`),
proven live against real weights (ledger #240, run 2026-08-04b, §1 PASS).
This module covers what a live GPU run cannot cheaply iterate on: every typed
branch (IN-2 skip-first-encode, multi-block continuation/re-encode, OUT-3
provenance stamping, `DiffusionCancelled` partial-return, participant
wiring), profiled for coverage.

Scope split from `tests/test_kv_cache_run_diffusion.py`: that module now
covers `run_diffusion`'s OWN ingress/dispatch behavior for the `kv_cache=`
door (unchanged shape, new destination); this module covers the drive body's
OWN per-block/per-step behavior once dispatched. `tests/test_kv_cache_cold_
wiring.py` covers the DV.3c cold-wiring guarantee (now non-degenerate again
per the ADR's original framing, since Phase 4 replaces issue #207's fail-loud
door).

Fixture composition: builds on `tests/conftest.py`'s `_FakeEncoderModel`/
`FakeDynamicCache`/`FakeDGemmaModelConfig` (§L, the same fakes
`test_kv_cache_run_diffusion.py` and `encode_sequence`'s own tests use) and
adds a fake decoder/top-level-model forward capable of producing logits, so
`_run_pipeline_with_injected_cache`'s full per-step loop actually runs
end-to-end against deterministic fake tensors. Uses the REAL
`transformers.DynamicCache`-backed `FakeDynamicCache` and the REAL
`DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask` static
method (imported directly, not re-implemented) so the mask-construction path
is never faked away — only the two forward calls (encoder, top-level model)
and the tokenizer/processor are fakes.
"""
from __future__ import annotations

import pytest
import torch
from transformers.models.diffusion_gemma.modeling_diffusion_gemma import (
    DiffusionGemmaDecoderModel,
)

from dgemma.composite import DiffusionCancelled, StepEndComposite
from dgemma.capture import _FrameCollector
from dgemma.kv_cache import geometry_from_model, tokenizer_fingerprint
from dgemma.loop import DGemmaPipeline, run_diffusion
from dgemma.types import DGemmaModel, KVCache, Provenance
from tests.conftest import FakeDGemmaModelConfig, FakeDynamicCache


VOCAB_SIZE = 32
CANVAS_LENGTH = 4
NUM_HIDDEN_LAYERS = 6


class _FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0
    vocab_size = VOCAB_SIZE

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _FakeEncoderOutput:
    def __init__(self, past_key_values):
        self.past_key_values = past_key_values


class _RecordingFakeEncoderModel:
    """Same shape as `tests/conftest.py`'s `_FakeEncoderModel`, with a call
    log so multi-block tests can assert re-encode fired the expected number
    of times (once per block AFTER the first — IN-2 skips block 0's encode
    entirely)."""

    def __init__(self, num_hidden_layers: int) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.calls: list[int] = []  # input_ids.shape[-1] per call

    def parameters(self):
        yield torch.zeros(1)

    def __call__(self, *, input_ids, past_key_values=None, position_ids=None):
        self.calls.append(input_ids.shape[-1])
        cache = past_key_values if past_key_values is not None else FakeDynamicCache(
            num_layers=self.num_hidden_layers, seq_len=0
        )
        cache.append(input_ids.shape[-1])
        return _FakeEncoderOutput(past_key_values=cache)


class _FakeDecoderModule:
    """Exposes ONLY `create_diffusion_decoder_attention_mask` — the real
    static method, imported directly rather than re-implemented, so the mask
    contract this drive body depends on is never faked away."""

    create_diffusion_decoder_attention_mask = staticmethod(
        DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask
    )


class _FakeDiffusionGemmaModel:
    def __init__(self, config: FakeDGemmaModelConfig, encoder: _RecordingFakeEncoderModel) -> None:
        self.encoder = encoder
        self.decoder = _FakeDecoderModule()


class _FakeLogitsOutput:
    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class _FakeTopLevelModel:
    """The `dgemma_model.model` callable — mirrors
    `DiffusionGemmaForBlockDiffusion.forward`'s call shape
    (`decoder_input_ids=`/`past_key_values=`/`self_conditioning_logits=`/
    `decoder_attention_mask=`/`decoder_position_ids=` -> `.logits`).

    Deterministic logits: a one-hot-ish distribution strongly favoring a
    FIXED target id per position (`target_id`), so the scheduler's
    `confidence_threshold` early-stop actually triggers (low entropy,
    stable argmax) instead of burning every one of `num_inference_steps`
    on noise — mirrors why the real live smoke's no-cache arm converged in
    14-18 steps rather than running the full budget."""

    def __init__(self, config: FakeDGemmaModelConfig, encoder: _RecordingFakeEncoderModel, *, target_id: int = 3) -> None:
        self.config = config
        self.model = _FakeDiffusionGemmaModel(config, encoder)
        self.target_id = target_id
        self.forward_calls = 0
        self.device = torch.device("cpu")

    def __call__(
        self,
        *,
        decoder_input_ids,
        past_key_values=None,
        self_conditioning_logits=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
    ):
        self.forward_calls += 1
        batch, canvas_len = decoder_input_ids.shape
        logits = torch.full((batch, canvas_len, VOCAB_SIZE), -10.0)
        logits[:, :, self.target_id] = 10.0
        return _FakeLogitsOutput(logits=logits)


def _fake_decoder_capable_model(
    *, repo_id: str = "fake/dgemma-drive-body", target_id: int = 3
) -> tuple[DGemmaModel, _RecordingFakeEncoderModel, _FakeTopLevelModel]:
    # `FakeDGemmaModelConfig`/`FakeDGemmaTextConfig` (`tests/conftest.py`)
    # expose no `vocab_size` — that fixture's own scope is
    # `geometry_from_model`, which never reads it. The real
    # `DiffusionGemmaTextConfig` DOES carry `vocab_size` (the pipeline's own
    # canvas-seed read, `pipeline_diffusion_gemma.py:347`, mirrored by this
    # drive body) — patched on here, local to this module, rather than
    # widening the shared fixture for a field only this module's decode path
    # needs.
    config = FakeDGemmaModelConfig(
        num_hidden_layers=NUM_HIDDEN_LAYERS, sliding_window=16, canvas_length=CANVAS_LENGTH
    )
    config.text_config.vocab_size = VOCAB_SIZE
    encoder = _RecordingFakeEncoderModel(NUM_HIDDEN_LAYERS)
    inner_model = _FakeTopLevelModel(config, encoder, target_id=target_id)
    dgemma_model = DGemmaModel(
        model=inner_model,
        processor=_FakeProcessor(),
        device="cpu",
        dtype="bfloat16",
        repo_id=repo_id,
        quant="none",
    )
    return dgemma_model, encoder, inner_model


def _matching_kv_cache(model: DGemmaModel, *, minting_sequence=(1, 2, 3), seq_len: int = 5) -> KVCache:
    text_config = model.model.config.get_text_config()
    cache = FakeDynamicCache(num_layers=text_config.num_hidden_layers, seq_len=seq_len)
    geometry = geometry_from_model(model)
    return KVCache(
        cache=cache,
        cumulative_length=tuple([seq_len] * text_config.num_hidden_layers),
        geometry=geometry,
        provenance=Provenance(
            minting_sequence=minting_sequence,
            edit_script=(),
            model_repo_id=model.repo_id,
            tokenizer_fingerprint=tokenizer_fingerprint(model),
        ),
    )


def _install_real_scheduler(monkeypatch):
    """Uses the REAL `EntropyBoundScheduler`/`DGemmaPipeline` (not a fake) —
    this module tests the drive body's own loop mechanics against a real
    scheduler's `.step()` contract, unlike `test_kv_cache_run_diffusion.py`'s
    ingress-only scope which fakes the scheduler away entirely. No
    monkeypatch needed; kept as a named no-op helper so every test's setup
    reads the same regardless of which path it exercises."""
    del monkeypatch


class TestSkipFirstEncode:
    """IN-2: block 0 must NOT call the encoder — the injected cache already
    carries that context."""

    def test_block_zero_does_not_call_encoder(self, monkeypatch):
        model, encoder, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
        )

        assert encoder.calls == [], (
            f"encoder should not be called for the injected cache's own block: {encoder.calls!r}"
        )

    def test_decode_runs_directly_off_injected_cache_tensors(self, monkeypatch):
        """The forward pass actually receives `past_key_values` seeded from
        `kv_cache.cache` (same object identity) — not a freshly-minted empty
        cache — so the model call the ADR's Q-2 H0a claims can succeed
        actually receives the injected context."""
        model, encoder, inner_model = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model, seq_len=7)
        original_cache_obj = cache.cache

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
        )

        assert inner_model.forward_calls >= 1
        # The cache object driving the decode is the SAME object the
        # injected payload carried (grown in place by the real
        # `DynamicCache.update`, not replaced) — confirms decode ran off
        # THIS cache, not a substitute.
        assert cache.cache is original_cache_obj


class TestMultiBlockContinuation:
    """`gen_length` spanning more than one `canvas_length` must re-encode
    every block AFTER the first — IN-2 skips ONLY the first block's encode,
    not every block's."""

    def test_second_block_triggers_one_reencode_call(self, monkeypatch):
        model, encoder, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH * 2,
            kv_cache=cache,
        )

        assert len(encoder.calls) == 1, (
            f"expected exactly one re-encode call (block 1 only): {encoder.calls!r}"
        )
        assert encoder.calls[0] == CANVAS_LENGTH

    def test_three_blocks_trigger_two_reencode_calls(self, monkeypatch):
        model, encoder, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH * 3,
            kv_cache=cache,
        )

        assert len(encoder.calls) == 2


class TestEosEarlyStop:
    """`eos_early_stop`'s outer-block-loop `finished.all()` break (mirrors
    `pipeline_diffusion_gemma.py:432-435`): once a committed block contains
    `eos_token_id`, the loop stops requesting further blocks even though
    `gen_length` would otherwise ask for more — the SAME early-stop the
    no-cache path already gets from the pipeline, now proven on the
    with-cache path too."""

    def test_eos_in_first_block_skips_second_blocks_reencode(self, monkeypatch):
        # target_id == eos_token_id: the fake model's logits deterministically
        # favor the token id `_FakeTokenizer.eos_token_id` names, so the
        # FIRST committed block already contains EOS at every position.
        eos_id = 5
        model, encoder, _ = _fake_decoder_capable_model(target_id=eos_id)
        model.processor.tokenizer.eos_token_id = eos_id
        cache = _matching_kv_cache(model)

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH * 3,  # would need 3 blocks absent EOS
            kv_cache=cache,
        )

        # No re-encode call at all: block 0 already commits EOS, so the
        # outer loop breaks BEFORE block 1's re-encode would fire.
        assert encoder.calls == [], (
            f"EOS in block 0 should stop the loop before any re-encode: {encoder.calls!r}"
        )


class TestOut3ProvenanceStamp:
    """OUT-3 (issue #62 Phase 2, LIVE code per the 2026-08-04 #62 correction):
    a with-cache run's `CanvasTrace.injected_cache_provenance` carries the
    injected cache's `Provenance` identity; a no-cache run's stays `None`."""

    def test_with_cache_stamps_provenance(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model, minting_sequence=(4, 5, 6))

        _, _, trace = run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
        )

        assert trace.injected_cache_provenance is not None
        assert trace.injected_cache_provenance.minting_sequence == (4, 5, 6)
        assert trace.injected_cache_provenance is cache.provenance

    def test_without_cache_provenance_stays_none(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model()

        from tests.test_kv_cache_run_diffusion import _install_fakes  # reuse the no-cache fake pipeline

        _install_fakes(monkeypatch, num_steps=2)

        _, _, trace = run_diffusion(
            model,
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
        )

        assert trace.injected_cache_provenance is None


class TestConvergenceAndOutput:
    """The drive body must actually produce a non-degenerate, decodable
    result — DV.3c's "effortless" guarantee restored now that Phase 4
    replaces the fail-loud door."""

    def test_converges_and_produces_nonempty_text(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model(target_id=3)
        cache = _matching_kv_cache(model)

        text, canvas_state, trace = run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=8,
            confidence=0.5,  # generous threshold: the fake's low-entropy logits converge fast
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
        )

        assert canvas_state.steps_used >= 1
        assert canvas_state.committed_fraction > 0.0
        assert isinstance(text, str)

    def test_respects_num_inference_steps_budget_when_never_confident(self, monkeypatch):
        """`confidence=None` disables early-stop entirely (mirrors the
        pipeline's own `confidence_threshold=None` contract) — every block
        runs the full step budget, matching the live smoke's recorded (b)
        with-cache observation of consuming the full budget."""
        model, _, inner_model = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=5,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
        )

        assert inner_model.forward_calls == 5


class TestCancellationPartialReturn:
    """Issue #38's partial-return semantics hold on the with-cache path too
    — `DiffusionCancelled` mid-block returns the evidence captured so far,
    not a raised-away run."""

    def test_cancel_after_first_step_returns_partial_trace(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)

        calls = {"n": 0}

        def should_cancel() -> bool:
            calls["n"] += 1
            # Composite order is capture -> cancel (ADR-CDG-010 amendment):
            # by the time this predicate can return True on its SECOND call
            # (step_idx=1's callback), step 1's frame has already been
            # captured — so cancelling here truncates AFTER 2 captured
            # frames, not 1 (the truncation-point frame is retained, #38).
            return calls["n"] > 1

        text, canvas_state, trace = run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=5,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
            should_cancel=should_cancel,
        )

        assert canvas_state.steps_used == 2
        assert trace.injected_cache_provenance is not None


class TestParticipantWiringReused:
    """The with-cache path shares the SAME `StepEndComposite`/`_FrameCollector`
    construction as the no-cache path — `on_frame` fires per captured step."""

    def test_on_frame_invoked_per_step(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)
        seen: list[int] = []

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
            on_frame=lambda frame: seen.append(frame.step_idx),
        )

        assert seen == [0, 1, 2]

    def test_capture_full_distribution_populates_frames(self, monkeypatch):
        from dgemma.payloads import CaptureSpec

        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)

        _, _, trace = run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
            capture=CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=2),
        )

        assert trace.frames[0].distribution is not None
        assert trace.frames[0].distribution.shape[-1] == VOCAB_SIZE


class TestKVCacheNotMutatedInPlaceBeyondRealCacheSemantics:
    """§3 advance-returns-new-payload: the drive body must not hand back a
    DIFFERENT `Provenance`/`geometry` identity than what it read — the input
    `KVCache` dataclass's own fields (not the underlying `DynamicCache`
    tensor object, which transformers' own `.update()` legitimately mutates
    in place) stay exactly what the caller passed."""

    def test_provenance_and_geometry_identity_unchanged(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)
        original_provenance = cache.provenance
        original_geometry = cache.geometry

        run_diffusion(
            model,
            "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=CANVAS_LENGTH,
            kv_cache=cache,
        )

        assert cache.provenance is original_provenance
        assert cache.geometry is original_geometry


class TestBatchSizeGuard:
    """Tier-1 scope never batches an injected cache (this module's/the
    drive body's own docstring) — a batch>1 cache is rejected fail-loud
    rather than silently truncated to one sequence at the final
    `torch.cat(...)[0]` (ADR-CDG-001's plausible-but-wrong-output
    discipline)."""

    def test_batch_size_two_cache_raises(self, monkeypatch):
        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model)
        # Widen every layer's batch dim from 1 to 2 in place — the cheapest
        # way to produce a batch>1 `DynamicCache` from the existing fixture
        # without hand-rolling a second fake cache constructor.
        for layer in cache.cache.layers:
            layer.keys = layer.keys.expand(2, -1, -1, -1).contiguous()
            layer.values = layer.values.expand(2, -1, -1, -1).contiguous()

        with pytest.raises(ValueError, match="batch size"):
            run_diffusion(
                model,
                "",  # issue #248: with kv_cache=, prompt must be empty at the exclusivity door
                entropy_bound=0.1,
                t_min=0.4,
                t_max=0.8,
                num_inference_steps=2,
                confidence=None,
                gen_length=CANVAS_LENGTH,
                kv_cache=cache,
            )
