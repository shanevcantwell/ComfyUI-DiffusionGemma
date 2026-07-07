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

from dgemma.loop import (
    THINK_TOKEN,
    THOUGHT_CHANNEL_END_ID,
    THOUGHT_CHANNEL_START_ID,
    _decode_ids,
    run_diffusion,
)
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


class TestDecodeIds:
    """`_decode_ids` (test-coverage-plan.md Phase 2, `dgemma/loop.py:361`):
    plain-input unit test, no monkeypatching needed — `_decode_ids` only
    needs a processor/tokenizer fake and a plain `ids` list."""

    def test_no_eos_token_id_leaves_ids_untrimmed(self):
        assert _decode_ids(FakeProcessor(), [1, 2, 3], eos_token_id=None) == "TEXT:1,2,3"

    def test_eos_token_id_absent_from_ids_leaves_ids_untrimmed(self):
        assert _decode_ids(FakeProcessor(), [1, 2, 3], eos_token_id=999) == "TEXT:1,2,3"

    def test_eos_token_id_present_trims_inclusive_of_eos(self):
        """The line this test exists for (`loop.py:361`): post-EOS
        canvas-fill/renoise-garbage tokens must not leak into the decode."""
        ids = [1, 2, 999, 3, 4]
        assert _decode_ids(FakeProcessor(), ids, eos_token_id=999) == "TEXT:1,2,999"

    def test_eos_token_id_as_first_id_trims_to_just_eos(self):
        assert _decode_ids(FakeProcessor(), [999, 1, 2], eos_token_id=999) == "TEXT:999"


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

        text, canvas_state, canvas_trace = run_diffusion(
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

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi", thinking=False)

        assert text == "TEXT:9,10"
        assert canvas_state.thought is None  # empty-frame content isn't surfaced as a "real" thought

    def test_non_empty_thought_channel_separated_into_canvas_state(self, monkeypatch):
        captured: dict = {}
        sequence = [THOUGHT_CHANNEL_START_ID, 55, 56, THOUGHT_CHANNEL_END_ID, 9, 10]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi", thinking=True)

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

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi", thinking=True)

        assert text == "TEXT:9,10,11"
        assert canvas_state.thought == "TEXT:55\n\nTEXT:66"  # both spans surfaced, joined visibly

    def test_stray_mid_canvas_start_delimiter_keeps_answer_and_flags(self, monkeypatch):
        """Stray-delimiter review finding (2026-07-05): an unmatched
        start_id past the head must never truncate the answer; the anomaly
        surfaces on the validity side instead."""
        captured: dict = {}
        sequence = [9, 10, THOUGHT_CHANNEL_START_ID, 11, 12]
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

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

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert text == "TEXT:"  # empty answer, honestly — the whole canvas was a broken frame
        assert canvas_state.thought == "TEXT:7,8"
        assert canvas_state.stray_thought_delimiter is False


class TestCanvasTrace:
    """P3 step 1/2: `CanvasTrace` must never ride without the scheduler
    identity that minted its commit readings (ADR-CDG-001 addendum) — the
    correctness must-have named in the plan's Risks section."""

    def test_carries_scheduler_identity_and_config(self, monkeypatch):
        captured: dict = {}

        class EntropyBoundScheduler:  # noqa: N801 — matching the real class's own __name__ on purpose
            def __init__(self, **kwargs):
                captured["scheduler_kwargs"] = kwargs

        class FakePipeline:
            def __init__(self, model, scheduler, processor):
                self.eos_token_id = 999

            def __call__(self, **kwargs):
                callback = kwargs["callback_on_step_end"]
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True, True]]),
                    "canvas": torch.tensor([[1, 2]]),
                }
                callback(self, 0, 0, callback_kwargs)
                return FakePipelineOutput(sequences=[torch.tensor([1, 2], dtype=torch.long)])

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", EntropyBoundScheduler)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)

        text, canvas_state, canvas_trace = run_diffusion(
            _fake_model(), "hi", num_inference_steps=7, entropy_bound=0.3, t_min=0.2, t_max=0.6
        )

        assert canvas_trace.scheduler_name == "EntropyBoundScheduler"
        assert canvas_trace.scheduler_config == {
            "entropy_bound": 0.3,
            "t_min": 0.2,
            "t_max": 0.6,
            "num_inference_steps": 7,
        }
        assert len(canvas_trace.frames) == 1

    def test_frames_match_collector_output(self, monkeypatch):
        """`CanvasTrace.frames` is `collector.frames` carried through, not a
        copy or a reshape — no new keying logic (plan.md step 1)."""
        captured: dict = {}
        _install_fakes(monkeypatch, captured, sequence=[1, 2, 3])

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert len(canvas_trace.frames) == 1
        assert canvas_trace.frames[0].step_idx == 0


class TestTurnClosedHonestyRider:
    """[SEVERABLE RIDER — issue #9] `turn_closed`/`answer_tokens` on
    `CanvasState`. Reproduces both named specimens from issue #9's comments
    behaviorally (the discrimination `turn_closed` exists for), not by
    replaying the issue's literal `steps_used` figures — these are synthetic
    single-callback fakes, so `steps_used` is always 1 here regardless."""

    def test_all_thought_empty_answer_is_not_turn_closed(self, monkeypatch):
        """Specimen (i): all-thought/empty-answer. The whole canvas is an
        unmatched turn-start thought frame — `excise_thought_channel`
        excises it to the end, leaving `remaining_ids == []`. No EOS to
        find in an empty answer."""
        captured: dict = {}
        sequence = [THOUGHT_CHANNEL_START_ID, 7, 8]  # no end_id anywhere -> excised to end
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert text == "TEXT:"
        assert canvas_state.answer_tokens == 0
        assert canvas_state.turn_closed is False

    def test_budget_truncated_mid_token_is_not_turn_closed_despite_converged(self, monkeypatch):
        """Specimen (ii): budget-truncated mid-token answer. The fake
        scheduler output always reports full acceptance
        (`committed_fraction == 1.0`, `converged == True`), but the canvas
        never contains the EOS id anywhere — the canvas ran out before EOS,
        the exact gap issue #9 names `converged` as unable to see."""
        captured: dict = {}
        sequence = [9, 10, 11, 12]  # no 999 (the fake eos_token_id) anywhere
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert canvas_state.converged is True
        assert canvas_state.committed_fraction == pytest.approx(1.0)
        assert canvas_state.turn_closed is False
        assert canvas_state.answer_tokens == len(sequence)

    def test_eos_inside_budget_is_turn_closed(self, monkeypatch):
        """Normal converged-with-EOS-inside-budget case: `turn_closed` must
        actually discriminate, not default one way regardless of input.
        `answer_tokens` counts pre-EOS content ONLY (review finding,
        2026-07-05): the EOS is the stop signal, not answer content, and
        anything after it is fill — for `[9, 10, 999, 3, 4]` the honest
        count is 2, not 5."""
        captured: dict = {}
        sequence = [9, 10, 999, 3, 4]  # 999 is the fake eos_token_id
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert canvas_state.turn_closed is True
        assert canvas_state.answer_tokens == 2

    def test_trailing_eos_fill_run_excluded_from_answer_tokens(self, monkeypatch):
        """The live-observed failure shape the eos-trim exists for (review
        finding, 2026-07-05): a converged run pads the canvas tail with a
        trailing EOS/renoise fill run (~30 tokens observed live). A bare
        `len(canvas_ids)` would report content + 1 + N; the honest
        `answer_tokens` is the pre-EOS content count alone."""
        captured: dict = {}
        content = [9, 10, 11]
        sequence = content + [999] + [999] * 5  # content + eos + eos-fill padding
        _install_fakes(monkeypatch, captured, sequence=sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert canvas_state.turn_closed is True
        assert canvas_state.answer_tokens == len(content)
