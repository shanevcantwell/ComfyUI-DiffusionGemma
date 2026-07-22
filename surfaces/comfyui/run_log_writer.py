"""surfaces/comfyui/run_log_writer.py — DGemmaRunLogWriter: SaveImage-
convention writer node (issue #72).

**Why a self-writing node, not a sampler `STRING` output routed to a save
node (§4 of the ratified plan):** a multi-line `STRING` routed through a
generic text-save node is exactly the escaped-newline trap
`consumers/tally_audit.py`'s `_FRAME_DELIMITER` documents as a real,
observed `.txt`-artifact failure (real on-disk `\n` becoming a literal
`\n` through a generic save-text assembly). This node OWNS the file handle
and writes bytes itself — one `json.dumps(obj)` + `file.write(line + "\n")`
per record — so that trap is structurally unrepresentable: no JSONL text
ever round-trips through a `STRING` socket at all (Requirement 3, D-4).

Inputs: `canvas_trace` (`DGEMMA_CANVAS_TRACE`), `run_config`
(`DGEMMA_RUN_CONFIG`), `frames` (the sampler's already-decoded `STRING`
list — reused for `decoded_step_text`, one decode, three renderings
counting `frames_image`, never re-decoded here), `canvas_state`
(`DGEMMA_CANVAS_STATE`), and `filename_prefix`/output-dir handling per the
ComfyUI `SaveImage` idiom (`folder_paths.get_output_directory()` +
`folder_paths.get_save_image_path()`, the same functions
`surfaces/comfyui/loader.py` already treats as "real inside ComfyUI,
genuinely absent under pytest/standalone" — same guarded-import
discipline, not a new one). An optional `debug_log_path` `STRING` widget
overrides the destination entirely (an absolute/relative path written to
directly) — useful for a bare-script/pytest run with no ComfyUI process
alive, matching the writer's own test suite.

This node is the ONLY file-writing surface in `surfaces/` (G-6 — `grep
-rln "open(|.write(|Path(" surfaces/` returned only `web/live_view.js`
before this module) — a fresh pattern, not a violation of an existing one.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

# Dual-context import, explicit package-depth gate — same discipline as
# every other surfaces/comfyui/*.py module (see surfaces/comfyui/loader.py
# for the full rationale). This module lives two levels under the pack
# root, so the relative climb to consumers/ and dgemma/ is THREE dots.
if __package__ and __package__.count(".") >= 2:
    from ...consumers.run_log import build_final_record, build_run_log_header, frame_to_record
    from .socket_types import DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE, DGEMMA_RUN_CONFIG
else:
    from consumers.run_log import build_final_record, build_run_log_header, frame_to_record
    from surfaces.comfyui.socket_types import (
        DGEMMA_CANVAS_STATE,
        DGEMMA_CANVAS_TRACE,
        DGEMMA_RUN_CONFIG,
    )

# `folder_paths` is a ComfyUI-runtime module — real inside a live ComfyUI
# process, genuinely absent under pytest/standalone. Same narrow
# `try/except ImportError` discipline as `surfaces/comfyui/loader.py`
# (never a blanket catch — see that module's own comment for why).
try:
    import folder_paths
except ImportError:
    folder_paths = None

DEFAULT_FILENAME_PREFIX = "dgemma_run_log"


def _resolve_output_path(filename_prefix: str, debug_log_path: str) -> Path:
    """Where to write: `debug_log_path` (non-empty) wins outright — an
    explicit override for a headless/bare-script run with no ComfyUI
    process alive (this node's own test suite included). Otherwise fall
    back to the `SaveImage` convention: `folder_paths.get_save_image_path`
    against the configured output directory, timestamped so repeated runs
    never collide (unlike an image batch, there is no natural "batch
    counter" input here to disambiguate on)."""
    if debug_log_path:
        output_path = Path(debug_log_path)
        # User may provide an existing directory — append the default filename.
        # Unlike the ComfyUI fallback path (which uses timestamps/counters),
        # repeated runs to the same directory will overwrite the previous log.
        if output_path.is_dir():
            return output_path / f"{filename_prefix}.jsonl"
        return output_path

    if folder_paths is None:
        raise RuntimeError(
            "DGemmaRunLogWriter: no ComfyUI folder_paths available and no "
            "debug_log_path override was given — cannot resolve an output "
            "directory. Set debug_log_path explicitly outside a live ComfyUI "
            "process."
        )

    full_output_folder, filename, counter, subfolder, actual_prefix = folder_paths.get_save_image_path(
        filename_prefix, folder_paths.get_output_directory()
    )
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    name = f"{actual_prefix}_{counter:05}_{timestamp}.jsonl"
    return Path(full_output_folder) / name


def write_run_log(
    path: Path,
    run_config,
    canvas_trace,
    canvas_state,
    frames: list[str],
) -> None:
    """The byte-writing core (T-1/T-2's subject): header line, one frame
    line per `canvas_trace.frames` (parallel to `frames`, the sampler's
    already-decoded strings — length-checked, not zipped blindly), final
    line. Exactly one `json.dumps(obj)` + `file.write(line + "\\n")` per
    record — a real `\\n`, never an escaped one, and never a multi-line
    string written as a single blob (Requirement 3, D-4)."""
    if len(frames) != len(canvas_trace.frames):
        raise ValueError(
            "write_run_log: frames must be parallel to canvas_trace.frames "
            f"(got {len(frames)} decoded strings for {len(canvas_trace.frames)} frames)."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        header = build_run_log_header(run_config, canvas_trace)
        handle.write(json.dumps(header) + "\n")
        for frame, decoded_step_text in zip(canvas_trace.frames, frames):
            record = frame_to_record(frame, decoded_step_text)
            handle.write(json.dumps(record) + "\n")
        final = build_final_record(canvas_trace, canvas_state)
        handle.write(json.dumps(final) + "\n")


class DGemmaRunLogWriter:
    """Writes a schema'd JSONL run log (issue #72) — one header record, one
    record per captured frame (with `decoded_step_text`), one final record.
    SaveImage-convention: owns and writes its own file, never routes text
    through a `STRING` save node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_trace": (DGEMMA_CANVAS_TRACE,),
                "run_config": (DGEMMA_RUN_CONFIG,),
                "frames": ("STRING", {"forceInput": True}),
                "canvas_state": (DGEMMA_CANVAS_STATE,),
                "filename_prefix": ("STRING", {"default": DEFAULT_FILENAME_PREFIX}),
            },
            "optional": {
                # Explicit override — writes directly to this path instead
                # of resolving through folder_paths/SaveImage's counter
                # convention. Empty string (the default) means "use the
                # ComfyUI output-directory convention."
                "debug_log_path": ("STRING", {"default": ""}),
            },
        }

    # `frames` is DGemmaSampler's OUTPUT_IS_LIST=True STRING output — this
    # node needs the WHOLE ordered list at once (one JSONL file, not one
    # call per frame), the same INPUT_IS_LIST convention
    # `surfaces/comfyui/tally_audit.py`'s DGemmaTallyAudit already uses for
    # the identical reason.
    INPUT_IS_LIST = True

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log_path",)
    FUNCTION = "write"
    OUTPUT_NODE = True
    CATEGORY = "DiffusionGemma"

    def write(
        self,
        canvas_trace,
        run_config,
        frames,
        canvas_state,
        filename_prefix=(DEFAULT_FILENAME_PREFIX,),
        debug_log_path=("",),
    ):
        # INPUT_IS_LIST=True hands every input as a length-1 (or, for
        # `frames`, length-N) list — unwrap the scalar-shaped ones
        # ourselves (the same unwrap `DGemmaTallyAudit.audit` performs for
        # its own INPUT_IS_LIST=True `frames` argument, generalized to the
        # rest of this node's inputs since ALL of them arrive listed under
        # this convention, not just the list-shaped one).
        trace = canvas_trace[0]
        config = run_config[0]
        state = canvas_state[0]
        prefix = filename_prefix[0] if filename_prefix else DEFAULT_FILENAME_PREFIX
        override_path = debug_log_path[0] if debug_log_path else ""

        path = _resolve_output_path(prefix, override_path)
        write_run_log(path, config, trace, state, list(frames))
        return (str(path),)
