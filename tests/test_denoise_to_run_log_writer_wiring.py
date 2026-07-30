"""tests/test_denoise_to_run_log_writer_wiring.py — issue #166 wiring test:
`DGemmaDenoise`'s six outputs feed `DGemmaRunLogWriter` end to end and
produce a valid run log, against the existing fake-pipeline fixtures.

**Why this test (ratified scope, issue #166 decision comment):** the
scope decision's rationale for full connection parity is explicit — "the
KV-cache path must reach the log writer." A mechanical socket-signature
match (`tests/test_sampler_denoise_parity.py`) proves the TYPES line up;
this test proves the WIRE actually carries a real run end to end: call
`DGemmaDenoise.denoise()` against a fake `run_diffusion` (same
monkeypatch-the-one-call-this-node-makes pattern as
`tests/test_kv_cache_nodes.py`), take its real 6-tuple output, and feed
`canvas_trace`/`run_config`/`frames`/`canvas_state` straight into
`DGemmaRunLogWriter.write()` (real, unmocked — the writer's own
`tests/test_run_log_writer.py` already covers its byte-level contract in
isolation; this test only proves the denoise->writer JOIN).

No sampler->log-writer end-to-end test exists in this suite today (checked
before writing this one — `tests/test_run_log_writer.py` exercises the
writer alone, `tests/test_loader_contract.py` exercises the sampler alone);
this is the first end-to-end wiring test for either producer node, written
for denoise per this issue's explicit ask.
"""
from __future__ import annotations

import json

import torch

from dgemma.types import CanvasState, CanvasTrace, DiffusionFrame
from surfaces.comfyui.denoise import DGemmaDenoise
from surfaces.comfyui.run_log_writer import DGemmaRunLogWriter


class _StubModel:
    processor = object()
    repo_id = "fake/wiring-repo"
    quant = "none"
    device = "cpu"
    dtype = "bfloat16"


def _fake_frame(step_idx: int) -> DiffusionFrame:
    return DiffusionFrame(
        canvas_idx=0,
        step_idx=step_idx,
        t=0.9 - step_idx * 0.1,
        temperature=0.7,
        committed_fraction_per_example=(1.0,),
        canvas=torch.tensor([step_idx, step_idx + 1, step_idx + 2]),
        entropy=None,
        pinned_mask=None,
    )


def _fake_trace() -> CanvasTrace:
    return CanvasTrace(
        frames=[_fake_frame(0), _fake_frame(1)],
        scheduler_name="EntropyBoundScheduler",
        scheduler_config={
            "entropy_bound": 0.1,
            "t_min": 0.4,
            "t_max": 0.8,
            "num_inference_steps_requested": 2,
            "num_inference_steps_effective": 2,
        },
        raw_canvas_ids=torch.tensor([1, 2, 3]),
    )


def _fake_state() -> CanvasState:
    return CanvasState(
        text="the answer",
        canvas_ids=torch.tensor([1, 2, 3]),
        converged=True,
        committed_fraction=1.0,
        steps_used=2,
        turn_closed=True,
        answer_tokens=3,
    )


def test_denoise_outputs_feed_run_log_writer_and_a_valid_log_is_written(monkeypatch, tmp_path):
    trace_stub = _fake_trace()
    state_stub = _fake_state()

    def fake_run_diffusion(model, prompt, on_frame=None, **kwargs):
        return ("the answer", state_stub, trace_stub)

    def fake_decode_frames(processor, frames):
        return ["frame zero", "frame one"]

    monkeypatch.setattr("surfaces.comfyui.denoise.run_diffusion", fake_run_diffusion)
    monkeypatch.setattr("surfaces.comfyui.emission.decode_frames", fake_decode_frames)

    denoise_node = DGemmaDenoise()
    text, canvas_state, canvas_trace, frames, images, run_config = denoise_node.denoise(
        _StubModel(),
        prompt="wiring prompt",
        seed=11,
        num_inference_steps=2,
        t_min=0.4,
        t_max=0.8,
        entropy_bound=0.1,
        confidence=0.005,
        gen_length=8,
        thinking=False,
        kv_cache=None,
        unique_id="7",
    )

    # The denoise node's own outputs, unmocked past run_diffusion/decode_frames
    # — `run_config` is really assembled by the shared emission helper, and
    # `frames`/`images` are really produced from `canvas_trace.frames`.
    assert text == "the answer"
    assert canvas_state is state_stub
    assert canvas_trace is trace_stub
    assert frames == ["frame zero", "frame one"]
    assert run_config.prompt == "wiring prompt"
    assert run_config.model_repo_id == "fake/wiring-repo"
    assert run_config.seed == 11

    # The join: DGemmaDenoise's real outputs feed DGemmaRunLogWriter,
    # INPUT_IS_LIST-wrapped exactly the way ComfyUI wires a producer node's
    # single values into a INPUT_IS_LIST=True consumer.
    writer_node = DGemmaRunLogWriter()
    log_path = tmp_path / "denoise_wiring_run.jsonl"
    (returned_path,) = writer_node.write(
        canvas_trace=[canvas_trace],
        run_config=[run_config],
        frames=frames,
        canvas_state=[canvas_state],
        filename_prefix=["denoise_wiring"],
        debug_log_path=[str(log_path)],
    )

    assert returned_path == str(log_path)
    assert log_path.exists()

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1 + len(canvas_trace.frames) + 1  # header + frames + final

    header = json.loads(lines[0])
    assert header["record_type"] == "header"
    assert header["seed"] == 11
    assert header["model_repo_id"] == "fake/wiring-repo"

    frame_records = [json.loads(line) for line in lines[1:-1]]
    assert [r["record_type"] for r in frame_records] == ["frame", "frame"]
    assert [r["decoded_step_text"] for r in frame_records] == ["frame zero", "frame one"]

    final = json.loads(lines[-1])
    assert final["record_type"] == "final"
    assert final["steps_used"] == 2
