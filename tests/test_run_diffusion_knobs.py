"""`dgemma.loop.run_diffusion` parameter-threading tests (P2: EB knobs +
thinking toggle land as real parameters, not hardcoded).

Mocks `EntropyBoundScheduler` and `DGemmaPipeline` at the `dgemma.loop`
module level — no real model/weights/diffusers-internals touched — so these
run with zero ComfyUI *and* zero real DiffusionGemma weights, per the task's
engine-level coverage requirement.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import THINK_TOKEN, THOUGHT_CHANNEL_END_ID, THOUGHT_CHANNEL_START_ID, run_diffusion
from dgemma.types import DGemmaModel


class FakeSchedulerOutput:
    def __init__(self, accepted: list[list[bool]]):
        self.accepted_index = torch.tensor(accepted, dtype=torch.bool)


class FakePipelineOutput:
    def __init__(self, sequences: list[torch.Tensor]):
        self.sequences = sequences
        # Deliberately wrong/leaky — asserts run_diffusion never trusts this
        # directly, since it re-derives `text` from the excised ids instead.
        self.texts = ["<<pipeline's own leaky pre-excision text, must be ignored>>"]


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0

    # Ground truth (verified against the real cached tokenizer, 2026-07-05):
    # the channel-name label's own ids decode to exactly "thought\n" — this
    # fake mirrors that one real mapping so the label-stripping behavior
    # (`dgemma.loop._extract_thought_text`) is exercised faithfully; every
    # other id sequence gets a generic, order-preserving stand-in decode.
    _KNOWN_DECODINGS = {(45518, 107): "thought\n"}

    def convert_tokens_to_ids(self, token):
        # No real vocab here -> resolve_thought_channel_ids must fall back
        # to the module constants.
        return None

    def decode(self, ids, skip_special_tokens=True):
        key = tuple(ids)
        if key in self._KNOWN_DECODINGS:
            return self._KNOWN_DECODINGS[key]
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _fake_model() -> DGemmaModel:
    return DGemmaModel(
        model=object(), processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


def _install_fakes(monkeypatch, captured: dict, sequence: list[int]):
    class FakeScheduler:
        def __init__(self, **kwargs):
            captured["scheduler_kwargs"] = kwargs

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            captured["pipeline_init"] = {"model": model, "scheduler": scheduler, "processor": processor}
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            captured["call_kwargs"] = kwargs
            callback = kwargs["callback_on_step_end"]
            callback_kwargs = {
                "scheduler_output": FakeSchedulerOutput([[True, True]]),
                "canvas": torch.tensor([[1, 2]]),
            }
            callback(self, 0, 0, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor(sequence, dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestKnobThreading:
    def test_eb_params_thread_into_scheduler_and_pipeline_call(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1, 2, 3])

        text, canvas_state = run_diffusion(
            _fake_model(),
            "hello",
            seed=3,
            gen_length=64,
            num_inference_steps=12,
            entropy_bound=0.25,
            t_min=0.1,
            t_max=0.5,
            confidence=0.9,
            thinking=False,
        )

        assert captured["scheduler_kwargs"] == {
            "entropy_bound": 0.25,
            "t_max": 0.5,
            "t_min": 0.1,
            "num_inference_steps": 12,
        }
        call_kwargs = captured["call_kwargs"]
        assert call_kwargs["prompt"] == "hello"
        assert "messages" not in call_kwargs
        assert call_kwargs["gen_length"] == 64
        assert call_kwargs["num_inference_steps"] == 12
        assert call_kwargs["confidence_threshold"] == pytest.approx(0.9)
        # Re-derived, not the pipeline's own (deliberately leaky) `texts[0]`.
        assert text == "TEXT:1,2,3"
        assert canvas_state.text == text

    def test_confidence_defaults_to_grounded_value_when_omitted(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1])

        run_diffusion(_fake_model(), "hi")

        assert captured["call_kwargs"]["confidence_threshold"] == pytest.approx(0.005)

    def test_thinking_false_passes_bare_prompt(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1])

        run_diffusion(_fake_model(), "hi", thinking=False)

        assert captured["call_kwargs"]["prompt"] == "hi"
        assert "messages" not in captured["call_kwargs"]

    def test_thinking_true_injects_think_token_system_message(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1])

        run_diffusion(_fake_model(), "hi", thinking=True)

        assert "prompt" not in captured["call_kwargs"]
        assert captured["call_kwargs"]["messages"] == [
            {"role": "system", "content": THINK_TOKEN},
            {"role": "user", "content": "hi"},
        ]

    def test_rejects_inverted_t_range_before_touching_pipeline(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1])

        with pytest.raises(ValueError, match="t_min must be"):
            run_diffusion(_fake_model(), "hi", t_min=0.8, t_max=0.4)

        assert "call_kwargs" not in captured  # never got far enough to call the pipeline

    def test_rejects_equal_t_range(self, monkeypatch):
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1])

        with pytest.raises(ValueError, match="t_min must be"):
            run_diffusion(_fake_model(), "hi", t_min=0.5, t_max=0.5)


class TestThoughtChannelIntegration:
    """End-to-end wiring between `run_diffusion` and
    `excise_thought_channel`/`resolve_thought_channel_ids` — the per-function
    excision unit tests live in `tests/test_thought_channel.py`; this checks
    they're actually plugged into the call site correctly."""

    def test_empty_thought_channel_stripped_from_final_text(self, monkeypatch):
        captured: dict = {}
        thought_word_id, newline_id = 45518, 107
        sequence = [THOUGHT_CHANNEL_START_ID, thought_word_id, newline_id, THOUGHT_CHANNEL_END_ID, 9, 10]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state = run_diffusion(_fake_model(), "hi", thinking=False)

        assert text == "TEXT:9,10"
        assert canvas_state.thought is None  # empty-frame content isn't surfaced as a "real" thought

    def test_non_empty_thought_channel_separated_into_canvas_state(self, monkeypatch):
        captured: dict = {}
        sequence = [THOUGHT_CHANNEL_START_ID, 55, 56, THOUGHT_CHANNEL_END_ID, 9, 10]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state = run_diffusion(_fake_model(), "hi", thinking=True)

        assert text == "TEXT:9,10"
        assert "55" not in text and "56" not in text  # never leaked onto the STRING payload
        assert canvas_state.thought == "TEXT:55,56"
        assert canvas_state.stray_thought_delimiter is False

    def test_second_thought_span_also_excised_from_text(self, monkeypatch):
        """Multi-span review finding (2026-07-05): a second frame's content
        must not leak onto the STRING payload either."""
        captured: dict = {}
        sequence = [
            THOUGHT_CHANNEL_START_ID, 55, THOUGHT_CHANNEL_END_ID,
            9, 10,
            THOUGHT_CHANNEL_START_ID, 66, THOUGHT_CHANNEL_END_ID,
            11,
        ]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state = run_diffusion(_fake_model(), "hi", thinking=True)

        assert text == "TEXT:9,10,11"
        assert canvas_state.thought == "TEXT:55\n\nTEXT:66"  # both spans surfaced, joined visibly

    def test_stray_mid_canvas_start_delimiter_keeps_answer_and_flags(self, monkeypatch):
        """Stray-delimiter review finding (2026-07-05): an unmatched
        start_id past the head must never truncate the answer; the anomaly
        surfaces on the validity side instead."""
        captured: dict = {}
        sequence = [9, 10, THOUGHT_CHANNEL_START_ID, 11, 12]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state = run_diffusion(_fake_model(), "hi")

        # Answer tokens after the stray delimiter survive on the payload
        # (the fake decode keeps the raw id visible; a real tokenizer's
        # skip_special_tokens drops just the delimiter itself).
        assert text == f"TEXT:9,10,{THOUGHT_CHANNEL_START_ID},11,12"
        assert canvas_state.thought is None
        assert canvas_state.stray_thought_delimiter is True

    def test_truncated_turn_start_frame_excised_to_end(self, monkeypatch):
        """Unmatched start_id AT the head (the documented turn-start frame
        position): excise-to-end stands — no answer text can precede
        index 0, so nothing real is lost."""
        captured: dict = {}
        sequence = [THOUGHT_CHANNEL_START_ID, 7, 8]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state = run_diffusion(_fake_model(), "hi")

        assert text == "TEXT:"  # empty answer, honestly — the whole canvas was a broken frame
        assert canvas_state.thought == "TEXT:7,8"
        assert canvas_state.stray_thought_delimiter is False
