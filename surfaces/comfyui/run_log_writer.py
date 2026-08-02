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
discipline, not a new one).

**Two-widget directory contract (issue #189, supersedes the earlier
ad-hoc dual-mode):** `debug_log_path` is always a DIRECTORY and
`filename_prefix` is always the FILE name — there is no code path where
`debug_log_path` is treated as a file path, even when its value happens to
end in `.jsonl`. Non-empty `debug_log_path` means "write inside this
directory" unconditionally: the directory is created (`mkdir -p`) if it
doesn't exist, and the file `{filename_prefix}_{timestamp}.jsonl` is
written inside it. Empty `debug_log_path` (the widget default) falls back
to the ComfyUI output-directory convention, unchanged.

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
    """Where to write, per the two-widget directory contract (issue #189):
    `debug_log_path` names a DIRECTORY, `filename_prefix` names the FILE.
    Non-empty `debug_log_path` is unconditionally a directory — even a
    value that "looks like a file" (e.g. ends in `.jsonl`) is still a
    directory by contract, deterministically, no `is_dir()` disambiguation.
    The directory is created (`mkdir -p`) if it doesn't exist, and the
    returned path is `{filename_prefix}_{timestamp}.jsonl` inside it,
    timestamped so repeated runs never collide (unlike an image batch,
    there is no natural "batch counter" input here to disambiguate on).
    Directory creation failures are raised loud (`OSError` subclasses,
    e.g. `PermissionError` for an unwritable parent) — this function never
    falls back to writing elsewhere.

    Empty `debug_log_path` (the widget default) falls back to the
    `SaveImage` convention: `folder_paths.get_save_image_path` against the
    configured output directory."""
    if debug_log_path:
        directory = Path(debug_log_path)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"DGemmaRunLogWriter: debug_log_path {str(directory)!r} could "
                f"not be created as a directory ({exc}). debug_log_path is "
                "always a directory (filename_prefix names the file) — pass "
                "a writable directory path, or leave debug_log_path empty to "
                "use the ComfyUI output-directory convention."
            ) from exc
        timestamp = time.strftime("%Y%m%dT%H%M%S")
        return directory / f"{filename_prefix}_{timestamp}.jsonl"

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

    DESCRIPTION = (
        "Writes a schema'd JSONL run log to disk — one header record, one "
        "record per captured frame (with its decoded step text), one final "
        "record. Wire it from ONE sampler-class node (DGemmaSampler or "
        "DGemmaDenoise): run_config, canvas_trace, frames, and canvas_state "
        "all come from that same node's outputs. debug_log_path is always "
        "a directory and filename_prefix names the file — the writer "
        "creates the directory if missing and writes "
        "{filename_prefix}_{timestamp}.jsonl inside it; leave "
        "debug_log_path empty to use ComfyUI's output directory. Returns "
        "the written log's path."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "canvas_trace": (
                    DGEMMA_CANVAS_TRACE,
                    {
                        "tooltip": (
                            "Per-step trace from the sampler-class node "
                            "that produced this run."
                        )
                    },
                ),
                "run_config": (
                    DGEMMA_RUN_CONFIG,
                    {
                        "tooltip": (
                            "Run header bundle from the SAME sampler-class "
                            "node (seed, knobs, model identity)."
                        )
                    },
                ),
                "frames": (
                    "STRING",
                    {
                        "forceInput": True,
                        "tooltip": (
                            "The decoded per-step strings from the same "
                            "node's frames output."
                        ),
                    },
                ),
                "canvas_state": (
                    DGEMMA_CANVAS_STATE,
                    {
                        "tooltip": (
                            "The final save-state from the same node's "
                            "canvas_state output."
                        )
                    },
                ),
                "filename_prefix": (
                    "STRING",
                    {
                        "default": DEFAULT_FILENAME_PREFIX,
                        "tooltip": (
                            "The FILE name stem. The log is written as "
                            "{filename_prefix}_{timestamp}.jsonl."
                        ),
                    },
                ),
            },
            "optional": {
                # A DIRECTORY (issue #189) — filename_prefix names the
                # file, not this widget. Non-empty: written inside this
                # directory (created if missing), unconditionally, even if
                # the value looks like a file path (e.g. ends `.jsonl`).
                # Empty string (the default) means "use the ComfyUI
                # output-directory convention."
                "debug_log_path": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "A DIRECTORY to write into (created if "
                            "missing) — NOT a file path. The file name "
                            "comes from filename_prefix. Leave empty to "
                            "use ComfyUI's output directory. A trailing "
                            "slash or a .jsonl-looking value is still "
                            "treated as a directory."
                        ),
                    },
                ),
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
