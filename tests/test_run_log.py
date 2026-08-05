"""consumers/run_log.py — schema'd JSONL run-log builder tests (issue #72).

Mirrors `tests/test_analysis.py`/`tests/test_tally_audit.py`'s consumer-tier
test shape: pure functions over already-captured `CanvasTrace`/`CanvasState`
values, no ComfyUI, no live pipeline needed for the per-function unit tests.

The core acceptance test (T-1, driving the REAL `fake_pipeline_factory`
fixture through `run_diffusion`, not a hand-built trace) lives in
`TestRealPipelineRoundTrip` below, following `tests/test_constraints.py`'s
`_wire_fake_pipeline` pattern so a real `CanvasTrace`/`CanvasState` pair
comes out of an actual `run_diffusion` call.
"""
from __future__ import annotations

import pytest
import torch

from consumers.run_log import (
    SCHEMA_VERSION,
    RunConfig,
    build_final_record,
    build_run_log_header,
    frame_to_record,
)
from dgemma.loop import run_diffusion
from dgemma.types import CanvasState, CanvasTrace, DGemmaModel, DiffusionFrame, EditOp, Provenance


def _run_config(**overrides) -> RunConfig:
    defaults = dict(
        prompt="hello",
        model_repo_id="google/diffusiongemma-26B-A4B-it",
        seed=7,
        num_inference_steps_requested=48,
        gen_length=256,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=0.1,
        confidence=0.005,
        thinking=False,
        quant="none",
        device="cpu",
        dtype="bfloat16",
    )
    defaults.update(overrides)
    return RunConfig(**defaults)


def _trace(**overrides) -> CanvasTrace:
    defaults = dict(
        frames=[],
        scheduler_name="EntropyBoundScheduler",
        scheduler_config={
            "entropy_bound": 0.1,
            "t_min": 0.4,
            "t_max": 0.8,
            "num_inference_steps_requested": 48,
            "num_inference_steps_effective": 48,
        },
    )
    defaults.update(overrides)
    return CanvasTrace(**defaults)


def _frame(**overrides) -> DiffusionFrame:
    defaults = dict(
        canvas_idx=0,
        step_idx=0,
        t=0.9,
        temperature=0.75,
        committed_fraction_per_example=(0.5,),
        canvas=torch.tensor([1, 2, 3, 4]),
    )
    defaults.update(overrides)
    return DiffusionFrame(**defaults)


class TestSchemaVersion:
    def test_schema_version_is_the_namespaced_string(self):
        assert SCHEMA_VERSION == "dg-runlog/1"

    def test_header_schema_field_matches_the_module_constant(self):
        """The header this module emits and the constant a future reader
        checks against are the same value, not two independently-typed
        strings that happen to match today."""
        header = build_run_log_header(_run_config(), _trace())
        assert header["schema"] == SCHEMA_VERSION


def _assert_known_schema_or_reject(record: dict) -> None:
    """T-3 (parse-at-the-door / schema version, ratified plan §7): the
    reader-side contract this schema exists to make checkable — an unknown
    `schema` value is REJECTED, never guessed at, mirroring
    `consumers/tally_audit.py`'s `CompositeBlobExtractionError` honest-
    failure shape. The full `audit_from_jsonl` reader is out of scope for
    this issue (§9) — this stub proves the one enforcement point AC-4
    requires exists and is enforceable against real emitted headers,
    without building the reader itself."""
    if record.get("schema") != SCHEMA_VERSION:
        raise ValueError(f"unrecognized run-log schema: {record.get('schema')!r}")


class TestParseAtTheDoorSchemaVersion:
    def test_known_schema_passes(self):
        header = build_run_log_header(_run_config(), _trace())
        _assert_known_schema_or_reject(header)  # does not raise

    def test_unknown_schema_version_is_rejected_not_guessed(self):
        header = build_run_log_header(_run_config(), _trace())
        header["schema"] = "dg-runlog/999"
        with pytest.raises(ValueError, match="unrecognized run-log schema"):
            _assert_known_schema_or_reject(header)

    def test_missing_schema_field_is_rejected(self):
        with pytest.raises(ValueError, match="unrecognized run-log schema"):
            _assert_known_schema_or_reject({"record_type": "header"})


class TestBuildRunLogHeader:
    def test_header_carries_seed_and_full_knob_set(self):
        header = build_run_log_header(_run_config(seed=1234), _trace())

        assert header["schema"] == SCHEMA_VERSION
        assert header["record_type"] == "header"
        assert header["seed"] == 1234
        assert header["prompt"] == "hello"
        assert header["model_repo_id"] == "google/diffusiongemma-26B-A4B-it"
        assert header["num_inference_steps_requested"] == 48
        assert header["num_inference_steps_effective"] == 48
        assert header["gen_length"] == 256
        assert header["t_min"] == 0.4
        assert header["t_max"] == 0.8
        assert header["entropy_bound"] == 0.1
        assert header["confidence"] == 0.005
        assert header["thinking"] is False
        assert header["scheduler_name"] == "EntropyBoundScheduler"
        assert header["scheduler_config"] == _trace().scheduler_config
        assert header["quant"] == "none"
        assert header["device"] == "cpu"
        assert header["dtype"] == "bfloat16"
        # pack_version/run_id/timestamp_utc are present and non-empty, but
        # their exact values aren't asserted here (installed-distribution-
        # and-clock-dependent) — see TestPackVersionFallback/TestRunIdUniqueness.
        assert header["pack_version"]
        assert header["run_id"]
        assert header["timestamp_utc"]

    def test_seed_none_is_preserved_honestly(self):
        """`seed=None` (no seed given this run) must serialize as `null`,
        never coerced to `0` or omitted."""
        header = build_run_log_header(_run_config(seed=None), _trace())
        assert header["seed"] is None

    def test_each_header_call_mints_a_fresh_run_id(self):
        header_a = build_run_log_header(_run_config(), _trace())
        header_b = build_run_log_header(_run_config(), _trace())
        assert header_a["run_id"] != header_b["run_id"]


class TestInjectionProvenanceHeaderFields:
    """Issue #266: the run-log header must say whether a KV cache was
    injected (`injection_present`) and, when so, carry OUT-3's provenance
    identity compactly (`injected_cache_provenance`). Unit-level coverage
    against hand-built `CanvasTrace`/`Provenance` dataclass instances (real
    types, not mocks of the writer's input) — the real-`run_diffusion`
    round-trip variants live in `TestInjectionProvenanceRealRoundTrip`
    below."""

    def test_no_cache_trace_reports_injection_absent(self):
        """AIO-shaped run (no `kv_cache` ever threaded through
        `run_diffusion`): `CanvasTrace.injected_cache_provenance` stays at
        its dataclass default (`None`) — `injection_present` must read
        `False` and the stamp must be `null`, never fabricated."""
        header = build_run_log_header(_run_config(), _trace())
        assert header["injection_present"] is False
        assert header["injected_cache_provenance"] is None

    def test_injected_trace_reports_injection_present_with_tier1_stamp(self):
        """A composed (encode->denoise) run: the trace's OUT-3 field carries
        a tier-1 `Provenance` (`minting_sequence` present, `edit_script`
        empty per `dgemma.types.Provenance`'s own tiering) — the header
        stamp must be the compact length/identity form, never the raw
        minting-sequence tuple."""
        provenance = Provenance(
            minting_sequence=(4, 5, 6, 7),
            edit_script=(),
            model_repo_id="fake/repo",
            tokenizer_fingerprint="fp-abc123",
        )
        trace = _trace(injected_cache_provenance=provenance)

        header = build_run_log_header(_run_config(), trace)

        assert header["injection_present"] is True
        stamp = header["injected_cache_provenance"]
        assert stamp == {
            "minting_sequence_length": 4,
            "edit_script_length": 0,
            "model_repo_id": "fake/repo",
            "tokenizer_fingerprint": "fp-abc123",
        }

    def test_injected_trace_with_empty_prompt_still_stamps_and_prompt_is_derivable(self):
        """Pure-injection run (empty `prompt`, cache attached, ADR-CDG-024):
        `injection_present`/stamp behave identically to the non-empty-prompt
        composed case, and the header's existing `prompt` field alone
        already lets a reader derive emptiness (`header["prompt"] == ""`) —
        the basis for not adding a redundant third boolean (module
        docstring deviation note)."""
        provenance = Provenance(
            minting_sequence=(1, 2, 3),
            edit_script=(),
            model_repo_id="fake/repo",
            tokenizer_fingerprint="fp-xyz",
        )
        trace = _trace(injected_cache_provenance=provenance)

        header = build_run_log_header(_run_config(prompt=""), trace)

        assert header["prompt"] == ""
        assert header["injection_present"] is True
        assert header["injected_cache_provenance"]["minting_sequence_length"] == 3

    def test_tier2_edited_cache_stamps_null_minting_length_with_edit_script_count(self):
        """Tier-2 (`minting_sequence is None`, `edit_script` non-empty per
        `Provenance`'s own docstring): the stamp must serialize
        `minting_sequence_length: null`, never a fabricated `0` standing in
        for "no minting sequence at all" — mirrors this module's own
        honest-absence discipline for `entropy_summary`/`pinned_positions`."""
        provenance = Provenance(
            minting_sequence=None,
            edit_script=(EditOp(op="ablate", params={"layer": 3}),),
            model_repo_id="fake/repo",
            tokenizer_fingerprint="fp-tier2",
        )
        trace = _trace(injected_cache_provenance=provenance)

        header = build_run_log_header(_run_config(), trace)

        assert header["injection_present"] is True
        stamp = header["injected_cache_provenance"]
        assert stamp["minting_sequence_length"] is None
        assert stamp["edit_script_length"] == 1

    def test_new_keys_are_additive_schema_version_unchanged(self):
        """Additive-keys-same-schema decision (issue #266, named in the
        header builder's docstring): adding these fields does not bump
        `SCHEMA_VERSION` — no in-repo reader asserts a closed key set
        against `dg-runlog/1`, only the `schema` string itself."""
        header = build_run_log_header(_run_config(), _trace())
        assert header["schema"] == "dg-runlog/1"
        assert SCHEMA_VERSION == "dg-runlog/1"


class TestFrameToRecord:
    def test_frame_record_carries_decoded_step_text_and_core_fields(self):
        frame = _frame(canvas_idx=0, step_idx=2, t=0.6, temperature=0.5)
        record = frame_to_record(frame, "the sky is blue")

        assert record["record_type"] == "frame"
        assert record["canvas_idx"] == 0
        assert record["step_idx"] == 2
        assert record["t"] == 0.6
        assert record["temperature"] == 0.5
        assert record["committed_fraction_per_example"] == [0.5]
        assert record["decoded_step_text"] == "the sky is blue"
        assert record["canvas_ids"] == [1, 2, 3, 4]

    def test_entropy_none_serializes_null_never_a_fabricated_summary(self):
        """T-4/AC-5: `frame.entropy is None` -> `entropy_summary: null`,
        never a fabricated `0.0`-valued summary."""
        frame = _frame(entropy=None)
        record = frame_to_record(frame, "text")
        assert record["entropy_summary"] is None

    def test_entropy_present_reduces_to_tier0_summary(self):
        entropy = torch.tensor([0.1, 0.2, 0.3, 0.9])
        frame = _frame(entropy=entropy)
        record = frame_to_record(frame, "text")

        summary = record["entropy_summary"]
        assert summary["mean"] == pytest.approx(torch.mean(entropy).item())
        assert summary["min"] == pytest.approx(0.1)
        assert summary["max"] == pytest.approx(0.9)
        assert summary["canvas_len"] == 4

    def test_pinned_mask_none_serializes_null_never_an_empty_list(self):
        """T-4/AC-5: `frame.pinned_mask is None` -> `pinned_positions: null`,
        never `[]` standing in for "no pin participant ran this frame"."""
        frame = _frame(pinned_mask=None)
        record = frame_to_record(frame, "text")
        assert record["pinned_positions"] is None

    def test_pinned_mask_present_yields_nonzero_indices(self):
        frame = _frame(pinned_mask=torch.tensor([False, True, False, True]))
        record = frame_to_record(frame, "text")
        assert record["pinned_positions"] == [1, 3]

    def test_effective_knobs_propagate_when_present(self):
        frame = _frame(effective_entropy_bound=0.2, effective_t_min=0.3, effective_t_max=0.7)
        record = frame_to_record(frame, "text")
        assert record["effective_entropy_bound"] == 0.2
        assert record["effective_t_min"] == 0.3
        assert record["effective_t_max"] == 0.7

    def test_effective_knobs_none_propagate_honestly(self):
        frame = _frame(effective_entropy_bound=None, effective_t_min=None, effective_t_max=None)
        record = frame_to_record(frame, "text")
        assert record["effective_entropy_bound"] is None
        assert record["effective_t_min"] is None
        assert record["effective_t_max"] is None

    def test_2d_canvas_takes_example_0(self):
        """Matches `dgemma.loop.decode_frames`'s own 2-D-canvas convention:
        a `[batch, canvas_len]` frame canvas decodes example 0 only."""
        frame = _frame(canvas=torch.tensor([[9, 8, 7]]))
        record = frame_to_record(frame, "text")
        assert record["canvas_ids"] == [9, 8, 7]


class TestBuildFinalRecord:
    def test_final_record_carries_raw_canvas_ids_and_canvas_state_echo(self):
        trace = _trace(raw_canvas_ids=torch.tensor([1, 2, 3]))
        state = CanvasState(
            text="the answer",
            canvas_ids=torch.tensor([1, 2, 3]),
            converged=True,
            committed_fraction=1.0,
            steps_used=48,
            thought="reasoning",
            stray_thought_delimiter=False,
            turn_closed=True,
            answer_tokens=3,
        )

        record = build_final_record(trace, state)

        assert record["record_type"] == "final"
        assert record["raw_canvas_ids"] == [1, 2, 3]
        assert record["steps_used"] == 48
        assert record["text"] == "the answer"
        assert record["converged"] is True
        assert record["committed_fraction"] == 1.0
        assert record["turn_closed"] is True
        assert record["answer_tokens"] == 3
        assert record["thought"] == "reasoning"
        assert record["stray_thought_delimiter"] is False

    def test_raw_canvas_ids_none_serializes_null_never_an_empty_array(self):
        """Additive-optional discipline (ADR-CDG-014 Decision 1) extended to
        the final record: a legacy/no-capture trace's `raw_canvas_ids is
        None` must serialize `null`, never `[]`."""
        trace = _trace(raw_canvas_ids=None)
        state = CanvasState(
            text="x", canvas_ids=torch.tensor([1]), converged=False, committed_fraction=0.5, steps_used=1
        )
        record = build_final_record(trace, state)
        assert record["raw_canvas_ids"] is None


class TestPackVersionFallback:
    def test_pack_version_falls_back_to_unknown_when_distribution_not_found(self, monkeypatch):
        import importlib.metadata

        def _raise(name):
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr("consumers.run_log.importlib.metadata.version", _raise)
        header = build_run_log_header(_run_config(), _trace())
        assert header["pack_version"] == "unknown"


# ---------------------------------------------------------------------------
# T-1: the core acceptance test — real fake-pipeline fixture through
# run_diffusion, not a hand-built trace.
# ---------------------------------------------------------------------------


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0
    vocab_size = 8

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _fake_model_with(model) -> DGemmaModel:
    return DGemmaModel(
        model=model,
        processor=FakeProcessor(),
        device="cpu",
        dtype="bfloat16",
        repo_id="fake/repo",
        quant="none",
    )


def _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, **factory_kwargs):
    """Same pattern as `tests/test_constraints.py`'s `_wire_fake_pipeline` —
    reached only through the fixture (per the R4 self-test module's
    import-mode caveat), monkeypatching `dgemma.loop`'s two construction
    sites to the CLASSES of a `fake_pipeline_factory()`-built instance."""
    built = fake_pipeline_factory(**factory_kwargs)
    scheduler_cls = type(built.scheduler)
    pipeline_cls = type(built.pipeline)

    def _scheduler_factory(**kwargs):
        return scheduler_cls(**kwargs)

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)
    return built


class TestRealPipelineRoundTrip:
    """T-1: build a `CanvasTrace`/`CanvasState` from the REAL
    `fake_pipeline_factory` fixture driven through `run_diffusion` end to
    end (not a hand-built trace), then pass the result through this
    module's builders and assert the schema — the core acceptance test the
    ratified plan's §7/§10 AC-2 names."""

    def test_full_header_and_frame_round_trip(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=3, canvas_shape=(1, 4))

        text, canvas_state, canvas_trace = run_diffusion(
            _fake_model_with(built.model),
            "hello",
            seed=42,
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            gen_length=4,
        )

        run_config = RunConfig(
            prompt="hello",
            model_repo_id="fake/repo",
            seed=42,
            num_inference_steps_requested=3,
            gen_length=4,
            t_min=0.4,
            t_max=0.8,
            entropy_bound=0.1,
            confidence=0.005,
            thinking=False,
            quant="none",
            device="cpu",
            dtype="bfloat16",
        )

        header = build_run_log_header(run_config, canvas_trace)
        assert header["schema"] == SCHEMA_VERSION
        assert header["seed"] == 42
        assert header["scheduler_name"] == canvas_trace.scheduler_name
        # Issue #266 AIO specimen: an AIO (no-cache) real `run_diffusion`
        # call never threads `kv_cache=`, so `injected_cache_provenance`
        # stays at the `CanvasTrace` dataclass default (`None`) all the way
        # through to the header.
        assert header["injection_present"] is False
        assert header["injected_cache_provenance"] is None

        from dgemma.loop import decode_frames

        decoded = decode_frames(FakeProcessor(), canvas_trace.frames)
        assert len(decoded) == len(canvas_trace.frames) == 3

        frame_records = [
            frame_to_record(frame, text) for frame, text in zip(canvas_trace.frames, decoded)
        ]
        assert len(frame_records) == 3
        for record, frame in zip(frame_records, canvas_trace.frames):
            assert record["step_idx"] == frame.step_idx
            assert record["canvas_ids"] == (
                frame.canvas[0].tolist() if frame.canvas.dim() == 2 else frame.canvas.tolist()
            )

        final = build_final_record(canvas_trace, canvas_state)
        assert final["record_type"] == "final"
        assert final["steps_used"] == canvas_state.steps_used


class TestInjectionProvenanceRealRoundTrip:
    """Issue #266's two injected specimens, driven through the REAL
    with-cache drive body (`dgemma.loop._run_pipeline_with_injected_cache`,
    ADR-CDG-012 Phase 4) via `run_diffusion(kv_cache=...)` — not a
    hand-built trace. Reuses `tests/test_kv_cache_drive_body.py`'s
    decode-capable fixtures (`_fake_decoder_capable_model`/
    `_matching_kv_cache`) rather than re-deriving a second decode-capable
    fake model, per that module's own fixture-composition precedent
    (`TestOut3ProvenanceStamp` already proves `run_diffusion(kv_cache=...)`
    stamps `CanvasTrace.injected_cache_provenance` — this class carries that
    same real trace one step further, through `build_run_log_header`)."""

    def test_composed_run_non_empty_prompt_stamps_header(self, monkeypatch):
        """Composed run (ADR-CDG-024): non-empty `prompt` + a connected
        `kv_cache` — the with-cache path prefills the templated turn onto
        the cache before decoding. Header must report `injection_present`
        True and a stamp matching the trace's own OUT-3 identity exactly."""
        from tests.test_kv_cache_drive_body import _fake_decoder_capable_model, _matching_kv_cache

        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model, minting_sequence=(4, 5, 6))

        text, canvas_state, canvas_trace = run_diffusion(
            model,
            "hello there",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=4,
            kv_cache=cache,
        )
        assert canvas_trace.injected_cache_provenance is cache.provenance

        run_config = RunConfig(
            prompt="hello there",
            model_repo_id=model.repo_id,
            seed=None,
            num_inference_steps_requested=2,
            gen_length=4,
            t_min=0.4,
            t_max=0.8,
            entropy_bound=0.1,
            confidence=0.005,
            thinking=False,
            quant="none",
            device="cpu",
            dtype="bfloat16",
        )
        header = build_run_log_header(run_config, canvas_trace)

        assert header["injection_present"] is True
        assert header["injected_cache_provenance"] == {
            "minting_sequence_length": len(cache.provenance.minting_sequence),
            "edit_script_length": len(cache.provenance.edit_script),
            "model_repo_id": cache.provenance.model_repo_id,
            "tokenizer_fingerprint": cache.provenance.tokenizer_fingerprint,
        }

    def test_empty_prompt_with_cache_stamps_header_and_prompt_is_empty(self, monkeypatch):
        """Pure-injection run (ADR-CDG-024): empty `prompt` + a connected
        `kv_cache`, no prefill. Header must still report `injection_present`
        True with a stamp matching OUT-3, AND `prompt == ""` — the
        denoiser-prompt-empty discriminant the issue asks for, read directly
        off the existing `prompt` field rather than a second boolean (see
        `build_run_log_header`'s docstring for why no extra flag was
        added)."""
        from tests.test_kv_cache_drive_body import _fake_decoder_capable_model, _matching_kv_cache

        model, _, _ = _fake_decoder_capable_model()
        cache = _matching_kv_cache(model, minting_sequence=(1, 2, 3))

        text, canvas_state, canvas_trace = run_diffusion(
            model,
            "",  # empty prompt + kv_cache = pure injection, no prefill (ADR-CDG-024)
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            confidence=None,
            gen_length=4,
            kv_cache=cache,
        )
        assert canvas_trace.injected_cache_provenance is cache.provenance

        run_config = RunConfig(
            prompt="",
            model_repo_id=model.repo_id,
            seed=None,
            num_inference_steps_requested=2,
            gen_length=4,
            t_min=0.4,
            t_max=0.8,
            entropy_bound=0.1,
            confidence=0.005,
            thinking=False,
            quant="none",
            device="cpu",
            dtype="bfloat16",
        )
        header = build_run_log_header(run_config, canvas_trace)

        assert header["prompt"] == ""
        assert header["injection_present"] is True
        assert header["injected_cache_provenance"]["minting_sequence_length"] == 3
        assert header["injected_cache_provenance"]["model_repo_id"] == cache.provenance.model_repo_id
