"""surfaces/comfyui/run_log_writer.py — DGemmaRunLogWriter tests (issue #72).

T-2 (the byte-level round-trip Requirement 3 turns on) lives here: after
writing, read the file back AS BYTES and assert (a) it splits into exactly
N+2 lines on a real `b"\\n"`, and (b) no line contains a literal `b"\\n"`
(the escaped-newline trap `consumers/tally_audit.py`'s `_FRAME_DELIMITER`
names as a real, observed failure this issue supersedes).

Also covers: the two-widget directory contract (issue #189) —
`debug_log_path` set (always a directory, `mkdir -p` + timestamped file
inside) vs unset (`folder_paths` absent, and `folder_paths`
present/mocked) branches — and `DGemmaRunLogWriter.write`'s
`INPUT_IS_LIST=True` unwrap.
"""
from __future__ import annotations

import json
from pathlib import Path

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
    """The two-widget directory contract (issue #189): `debug_log_path` is
    always a DIRECTORY, `filename_prefix` is always the FILE name. No
    `is_dir()` disambiguation survives — every non-empty `debug_log_path`
    is created (`mkdir -p`) and written into, deterministically."""

    def test_nonexistent_deep_directory_is_created_and_file_lands_inside(self, tmp_path):
        """A multi-level nonexistent directory is created (parents=True
        exercised) and the timestamped file resolves inside it."""
        target_dir = tmp_path / "does" / "not" / "exist" / "yet"
        assert not target_dir.exists()

        resolved = _resolve_output_path("my_run", str(target_dir))

        assert target_dir.is_dir()
        assert resolved.parent == target_dir
        assert resolved.name.startswith("my_run_")
        assert resolved.suffix == ".jsonl"

    def test_existing_directory_appends_timestamped_filename(self, tmp_path):
        """#124/#126 regression guard: pointing debug_log_path at an
        existing directory appends {filename_prefix}_{timestamp}.jsonl
        inside it — same observable behavior as before this fix."""
        resolved = _resolve_output_path("my_run", str(tmp_path))
        assert resolved.parent == tmp_path
        assert resolved.name.startswith("my_run_")
        assert resolved.suffix == ".jsonl"

    def test_existing_directory_repeated_runs_produce_unique_files(self, tmp_path):
        """Two calls with the same directory + prefix produce different
        paths (#124/#126 regression guard) — no overwrite across runs.
        Mock time.strftime to simulate distinct seconds."""
        from unittest.mock import patch
        timestamps = iter(["20260722T150000", "20260722T150001"])
        with patch("surfaces.comfyui.run_log_writer.time") as mock_time:
            mock_time.strftime = lambda fmt: next(timestamps)
            path_a = _resolve_output_path("my_run", str(tmp_path))
            path_b = _resolve_output_path("my_run", str(tmp_path))
        assert path_a != path_b
        assert path_a.name == "my_run_20260722T150000.jsonl"
        assert path_b.name == "my_run_20260722T150001.jsonl"

    def test_path_that_looks_like_a_file_is_still_treated_as_a_directory(self, tmp_path):
        """A debug_log_path ending in `.jsonl` is a directory by contract,
        deterministically — the exact #189 field failure. No is_dir()
        disambiguation: the "file-looking" path is created as a directory
        and the real file lands inside it, named from filename_prefix."""
        file_looking_path = tmp_path / "looks_like_a_file.jsonl"
        assert not file_looking_path.exists()

        resolved = _resolve_output_path("my_run", str(file_looking_path))

        assert file_looking_path.is_dir()
        assert resolved.parent == file_looking_path
        assert resolved.name.startswith("my_run_")
        assert resolved != file_looking_path

    def test_empty_debug_log_path_and_no_folder_paths_raises(self, monkeypatch):
        """Empty default keeps the ComfyUI output-directory convention
        branch unchanged: no folder_paths, no override -> loud failure."""
        monkeypatch.setattr("surfaces.comfyui.run_log_writer.folder_paths", None)
        with pytest.raises(RuntimeError, match="no ComfyUI folder_paths available"):
            _resolve_output_path("prefix", "")

    def test_empty_debug_log_path_resolves_via_folder_paths_when_present(self, monkeypatch, tmp_path):
        """Empty default -> convention branch unchanged: resolves through
        folder_paths.get_save_image_path, untouched by the directory-
        contract rewrite."""
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

    def test_uncreatable_directory_fails_loud(self, tmp_path):
        """An unwritable parent (no permission to create a subdirectory
        under it) fails loud and actionable — never silently degrades to
        writing elsewhere."""
        unwritable_parent = tmp_path / "unwritable_parent"
        unwritable_parent.mkdir()
        unwritable_parent.chmod(0o444)
        target_dir = unwritable_parent / "child_dir"

        try:
            with pytest.raises(RuntimeError, match="could not be created as a directory"):
                _resolve_output_path("my_run", str(target_dir))
        finally:
            # Restore permissions so pytest's tmp_path cleanup can remove it.
            unwritable_parent.chmod(0o755)

    def test_uncreatable_directory_error_chains_the_original_oserror(self, tmp_path):
        """The RuntimeError is actionable AND traceable — it chains the
        underlying OSError via `raise ... from exc`, not swallowed."""
        unwritable_parent = tmp_path / "unwritable_parent2"
        unwritable_parent.mkdir()
        unwritable_parent.chmod(0o444)
        target_dir = unwritable_parent / "child_dir"

        try:
            with pytest.raises(RuntimeError) as exc_info:
                _resolve_output_path("my_run", str(target_dir))
            assert isinstance(exc_info.value.__cause__, OSError)
        finally:
            unwritable_parent.chmod(0o755)


class TestDGemmaRunLogWriterNode:
    """The node's `INPUT_IS_LIST=True` unwrap (ComfyUI convention, same
    shape as `DGemmaTallyAudit.audit`'s own unwrap)."""

    def test_declarations(self):
        spec = DGemmaRunLogWriter.INPUT_TYPES()
        assert set(spec["required"]) == {"canvas_trace", "run_config", "frames", "canvas_state", "filename_prefix"}
        assert spec["required"]["canvas_trace"][0] == "DGEMMA_CANVAS_TRACE"
        assert spec["required"]["run_config"][0] == "DGEMMA_RUN_CONFIG"
        assert set(spec["optional"]) == {"debug_log_path"}
        # Issue #175 minimal phase: DESCRIPTION + per-input tooltips.
        assert DGemmaRunLogWriter.DESCRIPTION
        for name in ("canvas_trace", "run_config", "frames", "canvas_state", "filename_prefix"):
            assert "tooltip" in spec["required"][name][1]
        assert "tooltip" in spec["optional"]["debug_log_path"][1]
        assert DGemmaRunLogWriter.INPUT_IS_LIST is True
        assert DGemmaRunLogWriter.RETURN_TYPES == ("STRING",)
        assert DGemmaRunLogWriter.RETURN_NAMES == ("log_path",)
        assert DGemmaRunLogWriter.FUNCTION == "write"
        assert DGemmaRunLogWriter.OUTPUT_NODE is True
        assert DGemmaRunLogWriter.CATEGORY == "DiffusionGemma"

    def test_write_unwraps_list_shaped_inputs_and_writes_the_file(self, tmp_path):
        trace = _trace([_frame(0)])
        node = DGemmaRunLogWriter()
        override_dir = tmp_path / "node_written_dir"

        (returned_path,) = node.write(
            canvas_trace=[trace],
            run_config=[_run_config()],
            frames=["decoded text"],
            canvas_state=[_state()],
            filename_prefix=["prefix"],
            debug_log_path=[str(override_dir)],
        )

        returned = Path(returned_path)
        assert returned.parent == override_dir
        assert returned.name.startswith("prefix_")
        assert returned.exists()
        lines = returned.read_text().splitlines()
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
