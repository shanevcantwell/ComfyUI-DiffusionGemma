"""Phase 0 golden-trace oracle for issue #129's `dgemma/loop.py` decomposition
(ADR-CDG-018) — the no-behavior-change proof the ratified implementation plan
requires (issue #129 comment, "Ratification ask" §3, ruling 3).

Drives `dgemma.loop.run_diffusion` end-to-end over the deterministic
`fake_pipeline_factory` fixture (`tests/conftest.py`) — no real weights, no
RNG (the fake model returns all-zero logits every step, and
`FakeEntropyBoundScheduler.step` runs no random sampling), so the same call
produces byte-identical output on every invocation with no seed needed.

Serializes `(text, CanvasState, CanvasTrace)` to a pinned JSON golden file
(`tests/fixtures/loop_golden_trace.json`) captured once at Phase 0, on `main`
`1070792` — before any symbol left `loop.py`. Every subsequent phase's PR
re-runs this same test; a moved symbol that changed ANY emitted value fails
here loudly (plan §3.1, "the single strongest guard — the drive seam's full
output is the thing being conserved").

Regenerate the golden only if `run_diffusion`'s OWN behavior is deliberately
changed (never during this refactor) — run this file with
`DGEMMA_GOLDEN_TRACE_REGEN=1` set to rewrite the fixture, then inspect the
diff by hand before committing it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.types import DGemmaModel

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "loop_golden_trace.json"


class FakeTokenizer:
    """Deterministic order-preserving stand-in — no real vocab (mirrors
    `tests/test_run_diffusion_knobs.py::FakeTokenizer`, this suite's
    established fake-tokenizer idiom)."""

    eos_token_id = 999
    unk_token_id = 0

    def convert_tokens_to_ids(self, token: str) -> None:
        # No real vocab -> resolve_thought_channel_ids falls back to the
        # module-level THOUGHT_CHANNEL_START_ID/END_ID constants.
        return None

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _golden_model() -> DGemmaModel:
    return DGemmaModel(
        model=object(),
        processor=FakeProcessor(),
        device="cpu",
        dtype="bfloat16",
        repo_id="fake/golden-repo",
        quant="none",
    )


def _install_fake_pipeline(monkeypatch: pytest.MonkeyPatch, fake_pipeline_factory) -> Any:
    """Wires `fake_pipeline_factory`'s deterministic triple in at the module
    names `run_diffusion` constructs directly (`EntropyBoundScheduler`,
    `DGemmaPipeline`) — same monkeypatch seam every `dgemma.loop` engine-level
    test in this suite uses. Returns the built `FakePipelineFactory` handle so
    the caller can hand its `pipeline`/`scheduler` classes through."""
    built = fake_pipeline_factory(num_inference_steps=6, vocab_size=8, canvas_shape=(1, 4))

    def _scheduler_factory(**kwargs):
        assert kwargs == {
            "entropy_bound": pytest.approx(0.1),
            "t_max": pytest.approx(0.8),
            "t_min": pytest.approx(0.4),
            "num_inference_steps": 6,
        }
        return built.scheduler

    def _pipeline_factory(model, scheduler, processor):
        # `built.pipeline` already carries its own fixture-internal
        # `HookRecordingModel`/scheduler/processor from `fake_pipeline_factory`
        # — the driving `forward()` needs THAT model, not `run_diffusion`'s
        # `dgemma_model.model` (a bare `object()` placeholder here, never
        # called since no `logit_hook=`/`constraints=` is threaded this run).
        # Only `processor` (and its derived `eos_token_id`) needs to be the
        # one `run_diffusion` actually passes, so the golden's tokenizer
        # fake — not the fixture's `processor=None` default — drives decode.
        built.pipeline.processor = processor
        tokenizer = getattr(processor, "tokenizer", processor)
        built.pipeline.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        return built.pipeline

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", _pipeline_factory)
    return built


def _run_golden(monkeypatch: pytest.MonkeyPatch, fake_pipeline_factory) -> tuple[str, Any, Any]:
    _install_fake_pipeline(monkeypatch, fake_pipeline_factory)
    return run_diffusion(
        _golden_model(),
        "Why is the sky blue?",
        seed=0,
        gen_length=64,
        num_inference_steps=6,
        entropy_bound=0.1,
        t_min=0.4,
        t_max=0.8,
        confidence=0.005,
        thinking=False,
        keep_frames="all",
    )


def _tensor_to_jsonable(value: Any) -> Any:
    if torch.is_tensor(value):
        return {"__tensor__": True, "dtype": str(value.dtype), "data": value.tolist()}
    return value


def _frame_to_dict(frame: Any) -> dict:
    return {
        "canvas_idx": frame.canvas_idx,
        "step_idx": frame.step_idx,
        "t": frame.t,
        "temperature": frame.temperature,
        "committed_fraction_per_example": list(frame.committed_fraction_per_example),
        "canvas": _tensor_to_jsonable(frame.canvas),
        "entropy": _tensor_to_jsonable(frame.entropy),
        "top_k_ids": _tensor_to_jsonable(frame.top_k_ids),
        "top_k_weights": _tensor_to_jsonable(frame.top_k_weights),
        "distribution": _tensor_to_jsonable(frame.distribution),
        "pinned_mask": _tensor_to_jsonable(frame.pinned_mask),
        "effective_entropy_bound": frame.effective_entropy_bound,
        "effective_t_min": frame.effective_t_min,
        "effective_t_max": frame.effective_t_max,
    }


def _serialize(text: str, canvas_state: Any, canvas_trace: Any) -> dict:
    return {
        "text": text,
        "canvas_state": {
            "text": canvas_state.text,
            "canvas_ids": _tensor_to_jsonable(canvas_state.canvas_ids),
            "converged": canvas_state.converged,
            "committed_fraction": canvas_state.committed_fraction,
            "steps_used": canvas_state.steps_used,
            "thought": canvas_state.thought,
            "stray_thought_delimiter": canvas_state.stray_thought_delimiter,
            "turn_closed": canvas_state.turn_closed,
            "answer_tokens": canvas_state.answer_tokens,
        },
        "canvas_trace": {
            "scheduler_name": canvas_trace.scheduler_name,
            "scheduler_config": canvas_trace.scheduler_config,
            "raw_canvas_ids": _tensor_to_jsonable(canvas_trace.raw_canvas_ids),
            "injected_cache_provenance": canvas_trace.injected_cache_provenance,
            "frames": [_frame_to_dict(f) for f in canvas_trace.frames],
        },
    }


def test_golden_trace_matches_pinned_fixture(monkeypatch: pytest.MonkeyPatch, fake_pipeline_factory):
    """The Phase-0 oracle: `run_diffusion`'s full `(text, CanvasState,
    CanvasTrace)` output, serialized, must byte-match the pinned golden
    captured before any `loop.py` symbol moved. Any phase's extraction that
    changes what gets emitted — not just where the code lives — fails here.
    """
    text, canvas_state, canvas_trace = _run_golden(monkeypatch, fake_pipeline_factory)
    actual = _serialize(text, canvas_state, canvas_trace)

    if os.environ.get("DGEMMA_GOLDEN_TRACE_REGEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True) + "\n")
        pytest.skip("DGEMMA_GOLDEN_TRACE_REGEN=1: golden fixture (re)written, not compared this run.")

    assert GOLDEN_PATH.exists(), (
        f"Golden fixture missing at {GOLDEN_PATH}. Regenerate once with "
        "DGEMMA_GOLDEN_TRACE_REGEN=1 pytest tests/test_loop_golden_trace.py, "
        "then commit the fixture."
    )
    expected = json.loads(GOLDEN_PATH.read_text())
    assert actual == expected


def test_golden_trace_is_deterministic_across_two_runs(monkeypatch: pytest.MonkeyPatch, fake_pipeline_factory):
    """De-risks the oracle itself (plan §5 risk register, "golden-trace
    fixture is non-deterministic"): two independent calls against the same
    deterministic fake pipeline must serialize identically, confirming the
    fixture is a valid byte-identity oracle before it's trusted as one."""
    text1, state1, trace1 = _run_golden(monkeypatch, fake_pipeline_factory)
    text2, state2, trace2 = _run_golden(monkeypatch, fake_pipeline_factory)
    assert _serialize(text1, state1, trace1) == _serialize(text2, state2, trace2)
