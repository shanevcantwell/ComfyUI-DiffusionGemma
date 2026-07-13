"""`CanvasTrace.raw_canvas_ids` conservation tests (ADR-CDG-014 Decision 6,
issue #61 Phase P-A, issue #11).

The raw, un-excised final canvas ids ride `CanvasTrace.raw_canvas_ids`,
populated in `dgemma.loop._build_result` from `sequences` BEFORE
`excise_thought_channel` runs. `CanvasState.canvas_ids` stays post-excision
(the #8 contract, unchanged) — these tests prove the two views diverge
exactly at the excision boundary, including the #9 EOS-in-thought-span
probe: an EOS committed INSIDE a thought span is visible in
`raw_canvas_ids` while absent from `CanvasState.canvas_ids`.

Drives `run_diffusion` end to end against a mocked scheduler/pipeline (same
pattern as `tests/test_run_diffusion_knobs.py`'s `_install_fakes`) so this
exercises the real `_build_result` call path, not a hand-rolled shortcut.
"""
from __future__ import annotations

import torch

from dgemma.loop import THOUGHT_CHANNEL_END_ID, THOUGHT_CHANNEL_START_ID, run_diffusion
from dgemma.types import DGemmaModel

START = THOUGHT_CHANNEL_START_ID  # 100
END = THOUGHT_CHANNEL_END_ID  # 101
EOS = 999


class FakeSchedulerOutput:
    def __init__(self, accepted: list[list[bool]]):
        self.accepted_index = torch.tensor(accepted, dtype=torch.bool)


class FakePipelineOutput:
    def __init__(self, sequences):
        self.sequences = sequences
        self.texts = ["<<pipeline's own leaky pre-excision text, must be ignored>>"]


class FakeTokenizer:
    eos_token_id = EOS
    unk_token_id = 0
    _KNOWN_DECODINGS = {(45518, 107): "thought\n"}

    def convert_tokens_to_ids(self, token):
        return None  # forces fallback to the module START/END constants

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


def _install_fakes(monkeypatch, sequence: list[int]):
    class FakeScheduler:
        def __init__(self, **kwargs):
            self.num_inference_steps = kwargs["num_inference_steps"]

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            callback_kwargs = {
                "scheduler_output": FakeSchedulerOutput([[True] * len(sequence)]),
                "canvas": torch.tensor([sequence]),
            }
            callback(self, 0, 0, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor(sequence, dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestRawCanvasIdsHoldsPreExcisionSequence:
    def test_raw_canvas_ids_is_the_full_un_excised_sequence(self, monkeypatch):
        """(a) `CanvasTrace.raw_canvas_ids` holds the pre-excision sequence —
        an empty thought-channel frame plus answer content: the raw view
        must retain the channel delimiters/label the excised `canvas_ids`
        removes."""
        thought_word_id, newline_id = 45518, 107
        sequence = [START, thought_word_id, newline_id, END, 1, 2, 3]
        _install_fakes(monkeypatch, sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hello")

        assert canvas_trace.raw_canvas_ids == sequence
        # Diverges from the post-excision view: the channel is gone there.
        assert canvas_state.canvas_ids.tolist() == [1, 2, 3]
        assert canvas_trace.raw_canvas_ids != canvas_state.canvas_ids.tolist()

    def test_raw_canvas_ids_is_a_plain_int_list_not_a_tensor(self, monkeypatch):
        """Mirrors `excise_thought_channel`'s own id-level normalization
        (plain ints, never torch scalars) so a consumer never has to branch
        on tensor-vs-list."""
        sequence = [1, 2, 3]
        _install_fakes(monkeypatch, sequence)

        _, _, canvas_trace = run_diffusion(_fake_model(), "hello")

        assert canvas_trace.raw_canvas_ids == [1, 2, 3]
        assert all(isinstance(x, int) for x in canvas_trace.raw_canvas_ids)


class TestEosInsideThoughtSpanProbe:
    """Issue #9's probe: an EOS committed INSIDE a thought span is visible in
    `raw_canvas_ids` while invisible in `CanvasState.canvas_ids` — the exact
    gap #9/#3 need the raw view to close. The thought span is excised
    wholesale (delimiters + content) before `CanvasState.canvas_ids`/`text`
    are derived, so an EOS living inside that span never reaches the
    post-excision view at all; the raw view retains it."""

    def test_eos_inside_thought_span_visible_in_raw_not_in_canvas_state(self, monkeypatch):
        # EOS (999) committed INSIDE the thought span, between START/END —
        # an anomalous but representable specimen (#9).
        sequence = [START, 5, EOS, 6, END, 1, 2, 3]
        _install_fakes(monkeypatch, sequence)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hello")

        # Raw view: the EOS is right there, inside the (still-intact) span.
        assert EOS in canvas_trace.raw_canvas_ids
        assert canvas_trace.raw_canvas_ids == sequence

        # Post-excision view: the whole thought span (with its embedded EOS)
        # was removed before canvas_ids was derived — the EOS the #9 probe
        # cares about is invisible here, exactly the gap the raw field
        # exists to close.
        assert EOS not in canvas_state.canvas_ids.tolist()
        assert canvas_state.canvas_ids.tolist() == [1, 2, 3]


class TestRawCanvasIdsAdditiveOptional:
    """(c) `raw_canvas_ids is None` on a legacy/no-capture path — the
    additive-optional discipline applied to `CanvasTrace` directly (not
    through `run_diffusion`, which always populates it): a `CanvasTrace`
    built the pre-R6 way (positional args only) must still construct and
    read `None`, never an empty-list stand-in."""

    def test_canvas_trace_without_raw_canvas_ids_defaults_to_none(self):
        from dgemma.types import CanvasTrace, DiffusionFrame

        frame = DiffusionFrame(
            canvas_idx=0, step_idx=0, t=1.0, temperature=0.8,
            committed_fraction_per_example=(1.0,), canvas=torch.tensor([1]),
        )
        trace = CanvasTrace(frames=[frame], scheduler_name="EntropyBoundScheduler", scheduler_config={})

        assert trace.raw_canvas_ids is None
