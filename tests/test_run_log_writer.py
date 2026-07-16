"""surfaces/comfyui/run_log_writer.py — DGemmaRunLogWriter tests (issue #72).

T-2 (the byte-level round-trip Requirement 3 turns on) lives here: after
writing, read the file back AS BYTES and assert (a) it splits into exactly
N+2 lines on a real `b"\\n"`, and (b) no line contains a literal `b"\\n"`
(the escaped-newline trap `consumers/tally_audit.py`'s `_FRAME_DELIMITER`
names as a real, observed failure this issue supersedes).

Also covers: the writer's path-set (`debug_log_path` override) vs
path-unset (`folder_paths` absent, and `folder_paths` present/mocked)
branches, and `DGemmaRunLogWriter.write`'s `INPUT_IS_LIST=True` unwrap.
"""
from __future__ import annotations

import json

import pytest
import torch

from consumers.run_log import RunConfig
from dgemma.types import CanvasState, CanvasTrace, DiffusionFrame
from surfaces.comfyui.run_log_writer import (
    DGemmaRunLogWriter,
    _resolve_output_path,
    write_run_log,
)


def _run_config() -> RunConfig:
    return RunConfig(
        prompt="hello",
        model_repo_id="fake/repo",
        seed=7,
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


def _frame(step_idx: int, *, entropy=None, pinned_mask=None) -> DiffusionFrame:
    return DiffusionFrame(
        canvas_idx=0,
        step_idx=step_idx,
        t=0.9 - step_idx * 0.1,
        temperature=0.7,
        committed_fraction_per_example=(0.5,),
        canvas=torch.tensor([step_idx, step_idx + 1, step_idx + 2]),
        entropy=entropy,
        pinned_mask=pinned_mask,
    )


def _trace(frames: list[DiffusionFrame]) -> CanvasTrace:
    return CanvasTrace(
        frames=frames,
        scheduler_name="EntropyBoundScheduler",
        scheduler_config={
            "entropy_bound": 0.1,
            "t_min": 0.4,
            "t_max": 0.8,
            "num_inference_steps_requested": 3,
            "num_inference_steps_effective": 3,
        },
        raw_canvas_ids=torch.tensor([1, 2, 3]),
    )


def _state() -> CanvasState:
    return CanvasState(
        text="the answer",
        canvas_ids=torch.tensor([1, 2, 3]),
        converged=True,
        committed_fraction=1.0,
        steps_used=3,
        turn_closed=True,
        answer_tokens=3,
    )


class TestWriteRunLogByteLevelRoundTrip:
    """T-2: the structural proof the escaping trap is closed."""

    def test_real_newlines_no_escaped_newlines(self, tmp_path):
        frames = [_frame(0), _frame(1, entropy=torch.tensor([0.1, 0.2, 0.3])), _frame(2)]
        trace = _trace(frames)
        decoded = ["noise", "partial", "the answer"]
        path = tmp_path / "run.jsonl"

        write_run_log(path, _run_config(), trace, _state(), decoded)

        raw = path.read_bytes()
        # (a) splits into exactly N+2 lines on a REAL b"\n". `bytes.split`
        # on a trailing-newline-terminated blob yields one trailing empty
        # element — strip it before counting (each written line ends with
        # "\n", there is no unterminated final line).
        lines = raw.split(b"\n")
        assert lines[-1] == b""
        lines = lines[:-1]
        assert len(lines) == len(frames) + 2  # header + 3 frames + final

        # (b) no literal backslash-n anywhere in the file.
        assert b"\\n" not in raw

        # Each line parses as its own JSON object (never a multi-line blob).
        records = [json.loads(line) for line in lines]
        assert records[0]["record_type"] == "header"
        assert [r["record_type"] for r in records[1:-1]] == ["frame", "frame", "frame"]
        assert records[-1]["record_type"] == "final"

    def test_header_carries_the_seed(self, tmp_path):
        trace = _trace([_frame(0)])
        path = tmp_path / "run.jsonl"
        write_run_log(path, _run_config(), trace, _state(), ["text"])

        lines = path.read_text().splitlines()
        header = json.loads(lines[0])
        assert header["schema"] == "dg-runlog/1"
        assert header["seed"] == 7

    def test_frame_lines_carry_decoded_step_text(self, tmp_path):
        trace = _trace([_frame(0), _frame(1)])
        path = tmp_path / "run.jsonl"
        write_run_log(path, _run_config(), trace, _state(), ["frame zero", "frame one"])

        lines = path.read_text().splitlines()
        frame_records = [json.loads(line) for line in lines[1:-1]]
        assert [r["decoded_step_text"] for r in frame_records] == ["frame zero", "frame one"]

    def test_final_line_carries_raw_canvas_ids(self, tmp_path):
        trace = _trace([_frame(0)])
        path = tmp_path / "run.jsonl"
        write_run_log(path, _run_config(), trace, _state(), ["text"])

        final = json.loads(path.read_text().splitlines()[-1])
        assert final["raw_canvas_ids"] == [1, 2, 3]
        assert final["steps_used"] == 3

    def test_mismatched_frames_length_raises(self, tmp_path):
        trace = _trace([_frame(0), _frame(1)])
        path = tmp_path / "run.jsonl"
        with pytest.raises(ValueError, match="parallel to canvas_trace.frames"):
            write_run_log(path, _run_config(), trace, _state(), ["only one"])

    def test_creates_parent_directories(self, tmp_path):
        trace = _trace([_frame(0)])
        path = tmp_path / "nested" / "dir" / "run.jsonl"
        write_run_log(path, _run_config(), trace, _state(), ["text"])
        assert path.exists()


class TestResolveOutputPath:
    """Path-set vs. path-unset branches (coverage requirement, ratified
    plan §Coverage: "the writer's write routine (path-set vs path-unset)")."""

    def test_debug_log_path_override_wins_outright(self, tmp_path):
        override = tmp_path / "explicit.jsonl"
        resolved = _resolve_output_path("ignored_prefix", str(override))
        assert resolved == override

    def test_no_override_and_no_folder_paths_raises(self, monkeypatch):
        monkeypatch.setattr("surfaces.comfyui.run_log_writer.folder_paths", None)
        with pytest.raises(RuntimeError, match="no ComfyUI folder_paths available"):
            _resolve_output_path("prefix", "")

    def test_no_override_resolves_via_folder_paths_when_present(self, monkeypatch, tmp_path):
        class _FakeFolderPaths:
            @staticmethod
            def get_output_directory():
                return str(tmp_path)

            @staticmethod
            def get_save_image_path(prefix, output_dir):
                return str(tmp_path), "ignored", 5, "", prefix

        monkeypatch.setattr("surfaces.comfyui.run_log_writer.folder_paths", _FakeFolderPaths)
        resolved = _resolve_output_path("dgemma_run_log", "")
        assert resolved.parent == tmp_path
        assert resolved.name.startswith("dgemma_run_log_00005_")
        assert resolved.suffix == ".jsonl"


class TestDGemmaRunLogWriterNode:
    """The node's `INPUT_IS_LIST=True` unwrap (ComfyUI convention, same
    shape as `DGemmaTallyAudit.audit`'s own unwrap)."""

    def test_declarations(self):
        spec = DGemmaRunLogWriter.INPUT_TYPES()
        assert set(spec["required"]) == {"canvas_trace", "run_config", "frames", "canvas_state", "filename_prefix"}
        assert spec["required"]["canvas_trace"] == ("DGEMMA_CANVAS_TRACE",)
        assert spec["required"]["run_config"] == ("DGEMMA_RUN_CONFIG",)
        assert set(spec["optional"]) == {"debug_log_path"}
        assert DGemmaRunLogWriter.INPUT_IS_LIST is True
        assert DGemmaRunLogWriter.RETURN_TYPES == ("STRING",)
        assert DGemmaRunLogWriter.RETURN_NAMES == ("log_path",)
        assert DGemmaRunLogWriter.FUNCTION == "write"
        assert DGemmaRunLogWriter.OUTPUT_NODE is True
        assert DGemmaRunLogWriter.CATEGORY == "DiffusionGemma"

    def test_write_unwraps_list_shaped_inputs_and_writes_the_file(self, tmp_path):
        trace = _trace([_frame(0)])
        node = DGemmaRunLogWriter()
        override = tmp_path / "node_written.jsonl"

        (returned_path,) = node.write(
            canvas_trace=[trace],
            run_config=[_run_config()],
            frames=["decoded text"],
            canvas_state=[_state()],
            filename_prefix=["prefix"],
            debug_log_path=[str(override)],
        )

        assert returned_path == str(override)
        assert override.exists()
        lines = override.read_text().splitlines()
        assert len(lines) == 1 + 1 + 1  # header + 1 frame + final

    def test_write_defaults_filename_prefix_and_debug_log_path_when_empty_lists(self, monkeypatch, tmp_path):
        """`INPUT_IS_LIST=True` optional inputs can arrive as an empty list
        (unconnected optional) — the node must not crash unwrapping them."""

        class _FakeFolderPaths:
            @staticmethod
            def get_output_directory():
                return str(tmp_path)

            @staticmethod
            def get_save_image_path(prefix, output_dir):
                return str(tmp_path), "ignored", 1, "", prefix

        monkeypatch.setattr("surfaces.comfyui.run_log_writer.folder_paths", _FakeFolderPaths)

        trace = _trace([_frame(0)])
        node = DGemmaRunLogWriter()
        (returned_path,) = node.write(
            canvas_trace=[trace],
            run_config=[_run_config()],
            frames=["decoded text"],
            canvas_state=[_state()],
            filename_prefix=[],
            debug_log_path=[],
        )
        assert returned_path.endswith(".jsonl")
