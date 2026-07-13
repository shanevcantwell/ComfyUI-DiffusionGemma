"""`dgemma.loop.run_diffusion`'s `should_cancel=` seam (issue #38, folded into
the R1 composer spec per the #35 handoff comment): "the cancel check belongs
at the step boundary... Candidate seam: `run_diffusion` accepts a
surface-neutral cancel signal (event/predicate checked once per step by the
engine)... Partial-run semantics on cancel: return what exists (`CanvasTrace`
so far...) rather than raising away the evidence."

This is the `run_diffusion`-level half of the cancellation enforcement
surface; `tests/test_step_end_composite.py`'s `TestCancellationSeam` covers
`StepEndComposite`'s own contract in isolation. Uses the same module-level
monkeypatch idiom as `tests/test_run_diffusion_knobs.py` (`_install_fakes`),
since this exercises `run_diffusion` itself, not the composite/fake-pipeline
fixture directly.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.types import DGemmaModel


class FakeSchedulerOutput:
    def __init__(self, accepted: list[list[bool]]):
        self.accepted_index = torch.tensor(accepted, dtype=torch.bool)


class FakePipelineOutput:
    def __init__(self, sequences: list[torch.Tensor]):
        self.sequences = sequences
        self.texts = ["<<unused>>"]


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _fake_model() -> DGemmaModel:
    return DGemmaModel(
        model=object(), processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


def _install_multistep_fakes(monkeypatch, num_steps: int, canvases: list[list[int]]):
    """Mirrors `test_run_diffusion_knobs.py`'s `_install_fakes`, but drives
    `num_steps` callback invocations (that file's fakes fire the callback
    exactly once — #35 F7's named gap — insufficient for a cancel-after-N-
    steps scenario)."""

    class FakeScheduler:
        def __init__(self, **kwargs):
            self.num_inference_steps = kwargs["num_inference_steps"]

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            for step_idx in range(num_steps):
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True] * len(canvases[step_idx])]),
                    "canvas": torch.tensor([canvases[step_idx]]),
                }
                callback(self, step_idx, step_idx, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor(canvases[-1], dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestCancellationPartialReturn:
    def test_no_should_cancel_completes_normally(self, monkeypatch):
        """Default (`should_cancel=None`, today's real call sites) must
        behave exactly as before R1 — the run always completes."""
        _install_multistep_fakes(monkeypatch, num_steps=3, canvases=[[1], [2], [3]])

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi")

        assert text == "TEXT:3"
        assert canvas_trace.frames[-1].step_idx == 2

    def test_cancel_mid_run_returns_partial_trace_not_an_exception(self, monkeypatch):
        """The load-bearing behavior #38 names: a cancelled run returns
        `(text, CanvasState, CanvasTrace)` built from the frames captured
        before cancellation — evidence, not a raised-away run."""
        _install_multistep_fakes(monkeypatch, num_steps=5, canvases=[[1], [2], [3], [4], [5]])
        state = {"count": 0}

        def should_cancel() -> bool:
            state["count"] += 1
            return state["count"] > 2  # cancel checking on step_idx 2 (3rd check)

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi", should_cancel=should_cancel)

        # Only steps 0 and 1 were captured before the cancellation raised on
        # step 2's cancellation check (which runs before that step's capture).
        assert [f.step_idx for f in canvas_trace.frames] == [0, 1]
        assert canvas_state.steps_used == 2
        # Partial text is derived from the LAST CAPTURED frame's canvas
        # ([2], the step_idx==1 frame), never the never-produced pipeline
        # output — proves the partial path doesn't silently reach for
        # output.sequences (which doesn't exist on a cancelled run).
        assert text == "TEXT:2"

    def test_cancel_before_any_step_captured_reraises(self, monkeypatch):
        """If cancellation fires before even the first frame is captured,
        there is no partial evidence to return — re-raising (rather than
        fabricating an empty CanvasState) is the honest behavior."""
        _install_multistep_fakes(monkeypatch, num_steps=3, canvases=[[1], [2], [3]])

        from dgemma.composite import DiffusionCancelled

        with pytest.raises(DiffusionCancelled):
            run_diffusion(_fake_model(), "hi", should_cancel=lambda: True)

    def test_cancel_never_triggers_pipeline_runs_to_completion(self, monkeypatch):
        _install_multistep_fakes(monkeypatch, num_steps=3, canvases=[[1], [2], [3]])

        text, canvas_state, canvas_trace = run_diffusion(_fake_model(), "hi", should_cancel=lambda: False)

        assert [f.step_idx for f in canvas_trace.frames] == [0, 1, 2]
        assert text == "TEXT:3"
