"""consumers/run_log.py — schema'd JSONL run-log emission (issue #72).

Consumer tier per ADR-CDG-008 Open Question #1 (settled `consumers/`), same
discipline as `consumers/analysis.py`/`consumers/tally_audit.py`: this module
is pure — it builds plain `dict` records from already-captured `CanvasTrace`/
`CanvasState` values and a caller-supplied `RunConfig`, never wraps
`load_model`/`run_diffusion`, never drives the model, and imports only
`dgemma.types` from the core (nothing else). It lives outside `dgemma/`'s
import graph (`tests/test_seam.py`'s subprocess leak-check covers
`consumers/*`, this module included by name).

**Why `RunConfig` lives here, not in `dgemma/types.py` (D-1/D-3, issue #72
design-gate ratification):** `run_diffusion` does not receive everything a
correct run-log header needs (seed flows in but is never recorded on the
returned `CanvasTrace`; confidence/gen_length/thinking/prompt/model-id are
similarly absent from `_build_result`'s inputs) — see the ratified plan's
G-1/G-2. Rather than widen the core's signature to echo back its own call
args (rejected as Option B — a request-echo value pushed core-side, inverting
the "core emits measurement" flow, ADR-CDG-008 rule 6 optics), `RunConfig` is
assembled surface-side (`surfaces/comfyui/sampler.py`, the sampler already
holds every value simultaneously) and threaded to this module's builders. The
`DGEMMA_RUN_CONFIG` socket STRING is minted once in
`surfaces/comfyui/socket_types.py` (envelope); this dataclass is the payload
it carries (identity) — `IDENTITY⊥ENVELOPE` (ARCHITECTURE.md rule 4).

Recorded promotion trigger (D-1, banked on the ratified plan, not
re-derived): when a second surface (MCP) needs this same header without
re-assembling it, promote `RunConfig` to a core-emitted `CanvasTrace` field
under a new ADR-CDG record. Today, with emission shipping on the ComfyUI
surface only, `RunConfig` stays here.

**Schema versioning (D-5 / `EMIT-CANONICAL / PARSE-AT-THE-DOOR`):**
`SCHEMA_VERSION` is a namespaced string (`"dg-runlog/1"`), not a bare int — a
future reader asserts against it and REJECTS an unrecognized value rather
than guessing, mirroring `consumers/tally_audit.py`'s
`CompositeBlobExtractionError` honest-failure shape. Tier-1 (`top_k`)/Tier-2
(`distribution`) fields are deliberately excluded from this schema version;
a `dg-runlog/2` bump adds them once the core populates them (ADR-CDG-014
Phases P-B/P-C).

**Real newlines only, structurally (Requirement 3 / D-4):** every record
this module builds is a plain `dict` intended for exactly one
`json.dumps(...)` call by the writer — no field here is a pre-joined
multi-line string, and this module never itself writes to a file or a
`STRING` socket. The escaped-newline trap (`consumers/tally_audit.py`'s
`_FRAME_DELIMITER`, a real `.txt` artifact this issue supersedes) is
structurally unrepresentable because no text ever round-trips through a
generic save-text node — see `surfaces/comfyui/run_log_writer.py` for the
byte-writing half of that guarantee.
"""
from __future__ import annotations

import importlib.metadata
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

# Dual-context import, same discipline and same depth as
# `consumers/analysis.py`/`consumers/tally_audit.py` (see those modules'
# docstrings for the full rationale) — `consumers/` sits one level under the
# pack root, same depth as `dgemma/` itself, so the relative climb is exactly
# two dots.
if __package__ and "." in __package__:
    from ..dgemma.types import CanvasState, CanvasTrace, DiffusionFrame
else:
    from dgemma.types import CanvasState, CanvasTrace, DiffusionFrame

SCHEMA_VERSION = "dg-runlog/1"
"""Namespaced version string (D-5) — a reader asserts equality and rejects
an unrecognized value rather than guessing (parse-at-the-door)."""


def _pack_version() -> str:
    """Installed distribution version (G-7): `importlib.metadata.version`
    over the pack's own distribution name, `"unknown"` on
    `PackageNotFoundError` (the headless/uninstalled-source condition — e.g.
    running straight out of a git checkout with no `pip install -e .`)."""
    try:
        return importlib.metadata.version("comfyui-diffusiongemma")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


@dataclass
class RunConfig:
    """Caller-supplied (surface-assembled) run configuration — the header
    fields a `CanvasTrace` alone cannot supply (G-1). Rides the
    `DGEMMA_RUN_CONFIG` socket (minted in `surfaces/comfyui/socket_types.py`);
    this dataclass is the payload identity that string carries.

    Every field here is a value the sampler already holds as a widget arg or
    a `DGemmaModel` attribute at the moment it calls `run_diffusion` (G-2) —
    nothing here is derived or measured, it is request-echo by design,
    which is exactly why it stays out of `dgemma/types.py` (module
    docstring)."""

    prompt: str
    model_repo_id: str
    seed: int | None
    num_inference_steps_requested: int
    gen_length: int
    t_min: float
    t_max: float
    entropy_bound: float
    confidence: float
    thinking: bool
    quant: str
    device: str
    dtype: str


def build_run_log_header(run_config: RunConfig, canvas_trace: CanvasTrace) -> dict:
    """Line-1 header record (§5 of the ratified plan). Pulls
    `num_inference_steps_effective`/`scheduler_name`/`scheduler_config`
    verbatim from `canvas_trace` (the one thing the core DOES measure and
    return) and everything else from `run_config` (the one thing the core
    does NOT carry, G-1). A fresh `run_id` (uuid4 hex) is minted per header
    call — the cross-reference key a future image-filename correlation
    (out of scope, §9) would join against
    (`CONSERVE-ACROSS-THE-DATA-BOUNDARY`)."""
    scheduler_config = canvas_trace.scheduler_config
    return {
        "schema": SCHEMA_VERSION,
        "record_type": "header",
        "pack_version": _pack_version(),
        "run_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "prompt": run_config.prompt,
        "model_repo_id": run_config.model_repo_id,
        "seed": run_config.seed,
        "num_inference_steps_requested": run_config.num_inference_steps_requested,
        "num_inference_steps_effective": scheduler_config.get("num_inference_steps_effective"),
        "gen_length": run_config.gen_length,
        "t_min": run_config.t_min,
        "t_max": run_config.t_max,
        "entropy_bound": run_config.entropy_bound,
        "confidence": run_config.confidence,
        "thinking": run_config.thinking,
        "scheduler_name": canvas_trace.scheduler_name,
        "scheduler_config": scheduler_config,
        "quant": run_config.quant,
        "device": run_config.device,
        "dtype": run_config.dtype,
    }


def _entropy_summary(entropy: Any | None) -> dict | None:
    """Tier-0 scalar reduction (mean/min/max/canvas_len) of
    `DiffusionFrame.entropy` — never the raw per-position vector (§5
    "Deliberately EXCLUDED": a `float32[canvas_len]` tensor on every line
    would bloat the file for a per-step scalar summary). `None` when
    `entropy` was not captured this run (ADR-CDG-014 Decision 2's honest-
    absence discipline — never a fabricated `0.0`)."""
    if entropy is None:
        return None
    return {
        "mean": float(entropy.mean().item()),
        "min": float(entropy.min().item()),
        "max": float(entropy.max().item()),
        "canvas_len": int(entropy.shape[-1]),
    }


def _pinned_positions(pinned_mask: Any | None) -> list[int] | None:
    """Nonzero indices of `DiffusionFrame.pinned_mask`, or `None` when the
    mask is absent (T-4 / AC-5: `pinned_mask is None` must serialize
    `pinned_positions: null`, never an empty list standing in for "no pin
    participant ran this frame")."""
    if pinned_mask is None:
        return None
    return [idx for idx, value in enumerate(pinned_mask) if bool(value)]


def _canvas_ids(canvas: Any) -> list[int]:
    """Plain JSON int array from `DiffusionFrame.canvas` (a torch tensor or
    already-plain sequence, §5 "the torch object is never serialized
    directly") — example 0 only, matching `dgemma.loop.decode_frames`'s own
    2-D-canvas convention (`run_diffusion` is single-example/batch-1 today)."""
    if hasattr(canvas, "dim") and canvas.dim() == 2:
        canvas = canvas[0]
    return canvas.tolist() if hasattr(canvas, "tolist") else list(canvas)


def frame_to_record(frame: DiffusionFrame, decoded_step_text: str) -> dict:
    """One `record_type: "frame"` line (§5) — every field either read
    directly off `frame` or reduced from it. `decoded_step_text` is passed
    in rather than derived here: decoding needs a tokenizer, which this
    pure module deliberately does not hold (mirrors
    `consumers/tally_audit.py`'s "already-decoded strings, no tokenizer"
    contract) — the caller (the writer node) already has `frames` from
    `dgemma.loop.decode_frames`, one decode reused, not a second one."""
    return {
        "record_type": "frame",
        "canvas_idx": frame.canvas_idx,
        "step_idx": frame.step_idx,
        "t": frame.t,
        "temperature": frame.temperature,
        "committed_fraction_per_example": list(frame.committed_fraction_per_example),
        "entropy_summary": _entropy_summary(frame.entropy),
        "pinned_positions": _pinned_positions(frame.pinned_mask),
        "effective_entropy_bound": frame.effective_entropy_bound,
        "effective_t_min": frame.effective_t_min,
        "effective_t_max": frame.effective_t_max,
        "decoded_step_text": decoded_step_text,
        "canvas_ids": _canvas_ids(frame.canvas),
    }


def build_final_record(canvas_trace: CanvasTrace, canvas_state: CanvasState) -> dict:
    """The `record_type: "final"` line (§5): `raw_canvas_ids` (from
    `CanvasTrace`, pre-excision) plus the frame count and a `CanvasState`
    echo, so a reader has the validity readout without a second artifact."""
    raw_canvas_ids = canvas_trace.raw_canvas_ids
    return {
        "record_type": "final",
        "raw_canvas_ids": _canvas_ids(raw_canvas_ids) if raw_canvas_ids is not None else None,
        "steps_used": canvas_state.steps_used,
        "text": canvas_state.text,
        "converged": canvas_state.converged,
        "committed_fraction": canvas_state.committed_fraction,
        "turn_closed": canvas_state.turn_closed,
        "answer_tokens": canvas_state.answer_tokens,
        "thought": canvas_state.thought,
        "stray_thought_delimiter": canvas_state.stray_thought_delimiter,
    }
