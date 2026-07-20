"""surfaces/mcp/commands/generate.py — the `generate` tool: this surface's
wrap of `dgemma.loop.run_diffusion` (`dgemma/loop.py:465`).

Thin adapter (ARCHITECTURE.md surface-tier rules): unpack `args`, call
`run_diffusion` exactly once, wrap `(text, CanvasState, CanvasTrace)` into a
JSON-safe dict. No denoising-step loop, no scheduler construction here — all
of that is `dgemma`'s (ADR-CDG-008 Correction 1: this module builds nothing
that could accumulate cross-call state; `run_diffusion` itself constructs a
fresh scheduler/collector/composite every call, see that function's own
docstring).

**Cancellation (issue #38's MCP remainder).** `run_diffusion(should_cancel=)`
wants a zero-argument, surface-neutral predicate polled once per step
(`dgemma/composite.py:_CancellationParticipant`). MCP's stdio/JSON-RPC
transport is request/response per tool call — there is no in-band channel
for "abort the call that's still running" other than a SEPARATE tool call
racing the first, so the cheap transcribable mechanism is: run
`run_diffusion` in a worker thread (`asyncio.to_thread`, keeps the event loop
free to accept another call while generation runs), register a
`threading.Event` for the run under a caller-supplied `run_id`, and let a
sibling `cancel_run` tool call set that event from a concurrent request.
`should_cancel` is then just `event.is_set` — no polling of transport-level
primitives, no dependency on a client sending MCP's own (rarely-implemented)
`CancelledNotification`. This is deliberately NOT wired to the low-level
`mcp.server.Server`'s task-group cancellation (`anyio` scope cancel on
transport-level notification) — that would only fire if the connected client
actually sends one, which is not guaranteed across MCP clients, whereas a
plain `cancel_run` tool call works with any client able to make a second
tool call. `run_id` is caller-chosen (any hashable string); omitting it means
no cancellation wiring for that call (`should_cancel=None`, `run_diffusion`'s
own default — identical to today's uncancellable behavior).

**Widened doors (issue #103 Scope A).** `constraints=`/`control_signals=`/
`capture=` are now reachable through this tool's JSON schema —
`_unpack_constraints`/`_unpack_control_signals`/`_unpack_capture` below
unpack the JSON shape into the exact `dgemma.payloads` dataclasses
(`Constraints`/`Pin`, `ControlSignals`/`Binding`, `CaptureSpec`) and hand
them straight to `run_diffusion`, which validates them at ingress
(`dgemma.ingress.validate_ingress` — never re-implemented here, per
ARCHITECTURE.md rule 5, `EMIT-CANONICAL / PARSE-AT-THE-DOOR`). Omitting any
of the three is `None`, byte-identical to before this widening.

**`kv_cache=` deliberately NOT exposed here (issue #103 scope note).**
`KVCache.cache` (`dgemma/types.py`) is a live `transformers.DynamicCache`
tensor object — there is no JSON/disk encoding for it today
(`dgemma/kv_cache.py`'s module docstring names `save_kv_cache`/
`load_kv_cache`, the disk-crossing serialization a JSON payload would need,
as explicit Phase-3 `NOT-YET-IMPLEMENTED`; the live decoder-drive body is
also unbuilt, Phase 4). Exposing `kv_cache=` as a declarative JSON payload
here would require inventing that serialization scheme — a real design
decision `run_diffusion`'s ratified ADR-CDG-012 has not made yet, not a
transcription of already-landed design. Per the autonomy contract, this is
bounced alongside Scope B rather than absorbed; see the issue #103 tracking
comment.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any

if __package__ and __package__.count(".") >= 3:
    from .._mcp_sdk_guard import require_mcp_sdk
    from ..state_manager import StateManager
    # 4 dots, not 3: this module's package is "<pack>.surfaces.mcp.commands"
    # under the ComfyUI-loader context — one level deeper than
    # surfaces/comfyui/loader.py's "<pack>.surfaces.comfyui" (2 dots to reach
    # "<pack>", 3 to reach "<pack>.dgemma"), because `commands/` is an extra
    # directory `surfaces/comfyui/*.py` doesn't have. 3 dots here would land
    # one level short (`<pack>.surfaces.dgemma`, which doesn't exist) — see
    # `tests/test_mcp_dual_context_import.py`, which is the tripwire that
    # caught this depth being off-by-one during authoring.
    from ....dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        KNOB_DOCS,
        run_diffusion,
    )
    from ....dgemma.payloads import Binding, CaptureSpec, Constraints, ControlSignals, Pin
else:
    from surfaces.mcp._mcp_sdk_guard import require_mcp_sdk
    from surfaces.mcp.state_manager import StateManager
    from dgemma.loop import (
        DEFAULT_CONFIDENCE,
        DEFAULT_ENTROPY_BOUND,
        DEFAULT_GEN_LENGTH,
        DEFAULT_NUM_INFERENCE_STEPS,
        DEFAULT_T_MAX,
        DEFAULT_T_MIN,
        KNOB_DOCS,
        run_diffusion,
    )
    from dgemma.payloads import Binding, CaptureSpec, Constraints, ControlSignals, Pin

require_mcp_sdk()
from mcp.types import Tool  # noqa: E402

# issue #38 MCP remainder: a process-global registry of in-flight runs' cancel
# events, keyed by caller-supplied `run_id`. Deliberately NOT part of
# `StateManager` (ADR-CDG-008 Correction 1) — a cancel event is per-CALL
# transient plumbing, not model-load state, and is removed the instant its
# call returns (the `finally` below), so it never accumulates across calls
# the way a cached scheduler would. A `threading.Lock` guards the dict
# itself (registration/lookup/removal), not the run — two concurrent
# `generate` calls with different `run_id`s never contend on each other.
_active_runs: dict[str, threading.Event] = {}
_active_runs_lock = threading.Lock()


def _register_run(run_id: str) -> threading.Event:
    event = threading.Event()
    with _active_runs_lock:
        _active_runs[run_id] = event
    return event


def _unregister_run(run_id: str) -> None:
    with _active_runs_lock:
        _active_runs.pop(run_id, None)


def get_tools() -> list[Tool]:
    """`generate` + `cancel_run` tool definitions."""
    return [
        Tool(
            name="generate",
            description=(
                "Run one prompt through the DiffusionGemma denoising loop "
                "(dgemma.loop.run_diffusion) on the currently loaded model. "
                "Returns the decoded text, a validity readout (converged, "
                "committed_fraction, turn_closed, ...), and a summary of the "
                "captured per-step trace. Requires a model already loaded via "
                "the load_model tool. Optional constraints/control_signals/capture "
                "payloads expose the same declarative step-end intervention doors "
                "run_diffusion validates at ingress (ADR-CDG-010/011/014) — see "
                "each property's own description."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt to generate from"},
                    # Every "description" below is sourced from the ONE-MINT
                    # KNOB_DOCS vocabulary (`dgemma/loop.py`) — the same text
                    # `surfaces/comfyui/sampler.py`'s widget tooltips read, so
                    # the ComfyUI and MCP doors describe each knob identically
                    # by construction (rule-8 parity), never a re-typed copy
                    # that can drift.
                    "seed": {"type": "integer", "description": KNOB_DOCS["seed"]},
                    "gen_length": {
                        "type": "integer",
                        "description": KNOB_DOCS["gen_length"],
                        "default": DEFAULT_GEN_LENGTH,
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "description": KNOB_DOCS["num_inference_steps"],
                        "default": DEFAULT_NUM_INFERENCE_STEPS,
                    },
                    "entropy_bound": {
                        "type": "number",
                        "description": KNOB_DOCS["entropy_bound"],
                        "default": DEFAULT_ENTROPY_BOUND,
                    },
                    "t_min": {"type": "number", "description": KNOB_DOCS["t_min"], "default": DEFAULT_T_MIN},
                    "t_max": {"type": "number", "description": KNOB_DOCS["t_max"], "default": DEFAULT_T_MAX},
                    "confidence": {
                        "type": "number",
                        "description": KNOB_DOCS["confidence"],
                        "default": DEFAULT_CONFIDENCE,
                    },
                    "thinking": {
                        "type": "boolean",
                        "description": KNOB_DOCS["thinking"],
                        "default": False,
                    },
                    "run_id": {
                        "type": "string",
                        "description": (
                            "Optional caller-chosen id for this run. If given, a concurrent "
                            "cancel_run(run_id=...) call can abort it mid-generation (issue "
                            "#38); the partial (text, canvas_state, trace_summary) captured "
                            "so far is still returned. Omit for no cancellation wiring."
                        ),
                    },
                    "include_frames": {
                        "type": "boolean",
                        "description": "Include per-step frame telemetry in trace_summary (default: false — summary-only)",
                        "default": False,
                    },
                    "constraints": {
                        "type": "object",
                        "description": (
                            "ADR-CDG-010 declarative id-level givens. Thin-adapter mapped onto "
                            "dgemma.payloads.Constraints/Pin; validated core-side "
                            "(dgemma.ingress.validate_constraints) — this schema only shapes the "
                            "JSON, it re-implements no check. {\"pins\": [{\"position\": int, "
                            "\"token_id\": int}, ...]}. Omit for no constraints (today's behavior)."
                        ),
                        "properties": {
                            "pins": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "position": {"type": "integer"},
                                        "token_id": {"type": "integer"},
                                    },
                                    "required": ["position", "token_id"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["pins"],
                        "additionalProperties": False,
                    },
                    "control_signals": {
                        "type": "object",
                        "description": (
                            "ADR-CDG-011 CV/LFO bindings. Thin-adapter mapped onto "
                            "dgemma.payloads.ControlSignals/Binding; validated core-side "
                            "(dgemma.ingress.validate_control_signals). {\"bindings\": "
                            "[{\"target\": str, \"signal\": [float, ...] (length == "
                            "num_inference_steps), \"low\": float, \"high\": float}, ...]}. "
                            "Omit for no control signals (today's behavior)."
                        ),
                        "properties": {
                            "bindings": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "target": {"type": "string"},
                                        "signal": {"type": "array", "items": {"type": "number"}},
                                        "low": {"type": "number"},
                                        "high": {"type": "number"},
                                    },
                                    "required": ["target", "signal", "low", "high"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "required": ["bindings"],
                        "additionalProperties": False,
                    },
                    "capture": {
                        "type": "object",
                        "description": (
                            "ADR-CDG-014 capture= payload. Thin-adapter mapped onto "
                            "dgemma.payloads.CaptureSpec; validated core-side "
                            "(dgemma.ingress.validate_capture). \"top_k\" (Tier 1, default 0 = "
                            "off): non-negative int, in-vocab ceiling enforced at ingress. "
                            "\"keep_frames\": \"last\" or \"all\" (validated; not yet wired to "
                            "override retention — see dgemma/payloads.py:CaptureSpec). Omit for "
                            "no capture widening (today's behavior)."
                        ),
                        "properties": {
                            "top_k": {"type": "integer", "default": 0},
                            "keep_frames": {"type": "string", "enum": ["last", "all"], "default": "all"},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="cancel_run",
            description=(
                "Request cancellation of an in-flight generate call started with the "
                "given run_id (issue #38). No-op (reports found=false) if no such run "
                "is currently active — e.g. it already finished or was never given a "
                "run_id."
            ),
            inputSchema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}},
                "required": ["run_id"],
            },
        ),
    ]


def _unpack_constraints(raw: Any) -> "Constraints | None":
    """JSON `{"pins": [{"position": ..., "token_id": ...}, ...]}` ->
    `dgemma.payloads.Constraints`. Thin unpack only — never re-implements
    ADR-CDG-010's validation (`dgemma.ingress.validate_constraints`, fired
    inside `run_diffusion` itself). Fail-on-unknown is structural: `Pin`/
    `Constraints` are frozen dataclasses, so an unrecognized field in a pin
    dict raises `TypeError` from the constructor call below, not a silent
    drop — the same enforcement surface `dgemma/ingress.py`'s module
    docstring names for the core's own ingress. A malformed shape (missing
    key, wrong type) surfaces as the same `TypeError`/`KeyError` the
    dataclass constructor raises, caught by `server.py`'s outer handler and
    returned as a structured `{"error": ...}` — never crashes the process.
    """
    if raw is None:
        return None
    pins = tuple(Pin(**pin) for pin in raw.get("pins", ()))
    return Constraints(pins=pins)


def _unpack_control_signals(raw: Any) -> "ControlSignals | None":
    """JSON `{"bindings": [{"target": ..., "signal": [...], "low": ...,
    "high": ...}, ...]}` -> `dgemma.payloads.ControlSignals`. Thin unpack
    only — `signal` arrives as a JSON array and is tupled here (`Binding.
    signal` is typed `tuple[float, ...]`); every value/range/length check
    stays core-side (`dgemma.ingress.validate_control_signals`)."""
    if raw is None:
        return None
    bindings = tuple(
        Binding(
            target=b["target"],
            signal=tuple(b["signal"]),
            low=b["low"],
            high=b["high"],
        )
        for b in raw.get("bindings", ())
    )
    return ControlSignals(bindings=bindings)


def _unpack_capture(raw: Any) -> "CaptureSpec | None":
    """JSON `{"top_k": ..., "keep_frames": ...}` ->
    `dgemma.payloads.CaptureSpec`. Thin unpack only — `top_k`/`keep_frames`
    validation (non-negative int, in-vocab ceiling, retention-policy
    membership) stays core-side (`dgemma.ingress.validate_capture`)."""
    if raw is None:
        return None
    kwargs: dict[str, Any] = {}
    if "top_k" in raw:
        kwargs["top_k"] = raw["top_k"]
    if "keep_frames" in raw:
        kwargs["keep_frames"] = raw["keep_frames"]
    return CaptureSpec(**kwargs)


def _summarize_trace(canvas_trace, include_frames: bool) -> dict[str, Any]:
    """`CanvasTrace` -> JSON-safe summary. Never re-derives analysis
    (ARCHITECTURE.md rule 3 — analysis is a downstream consumer, not this
    surface's job): only the scheduler identity/config plus per-frame
    scalars `DiffusionFrame` already carries, no heatmap/avalanche-curve
    computation. `include_frames=False` (default) keeps the response small —
    a full per-step frame list can be `num_inference_steps` entries long."""
    summary: dict[str, Any] = {
        "scheduler_name": canvas_trace.scheduler_name,
        "scheduler_config": canvas_trace.scheduler_config,
        "num_frames": len(canvas_trace.frames),
    }
    if include_frames:
        summary["frames"] = [
            {
                "canvas_idx": f.canvas_idx,
                "step_idx": f.step_idx,
                "t": f.t,
                "temperature": f.temperature,
                "committed_fraction_per_example": list(f.committed_fraction_per_example),
            }
            for f in canvas_trace.frames
        ]
    return summary


async def generate(manager: StateManager, args: dict[str, Any]) -> dict[str, Any]:
    """Thin adapter over `run_diffusion`: unpack, call once (in a worker
    thread so the event loop stays free for a concurrent `cancel_run`), wrap.

    Stateless per ADR-CDG-008 Correction 1: every call resolves `manager.
    require_model()` (the one persisted object) and otherwise passes plain
    kwargs straight through to `run_diffusion`, which builds its own fresh
    scheduler/collector/composite internally. This function retains nothing
    across calls except the transient cancel-event registration, which is
    always removed in `finally` before returning.
    """
    prompt = args.get("prompt")
    if not prompt:
        return {"error": "prompt is required"}

    model = manager.require_model()
    run_id = args.get("run_id")
    include_frames = bool(args.get("include_frames", False))

    should_cancel = None
    if run_id:
        event = _register_run(run_id)
        should_cancel = event.is_set

    # Thin-adapter unpack: JSON -> dgemma.payloads dataclasses. No
    # validation happens here — `run_diffusion` calls `dgemma.ingress.
    # validate_ingress` on these before constructing a scheduler/pipeline
    # (ARCHITECTURE.md rule 5, core-side validation only). A malformed
    # shape (unknown key, wrong type) raises here or inside `run_diffusion`
    # itself; either way `server.py:call_tool`'s outer try/except turns it
    # into a structured `{"error": ...}` response, never a transport-level
    # crash.
    kwargs: dict[str, Any] = {
        "seed": args.get("seed"),
        "gen_length": args.get("gen_length", DEFAULT_GEN_LENGTH),
        "num_inference_steps": args.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS),
        "entropy_bound": args.get("entropy_bound", DEFAULT_ENTROPY_BOUND),
        "t_min": args.get("t_min", DEFAULT_T_MIN),
        "t_max": args.get("t_max", DEFAULT_T_MAX),
        "confidence": args.get("confidence", DEFAULT_CONFIDENCE),
        "thinking": bool(args.get("thinking", False)),
        "should_cancel": should_cancel,
        "constraints": _unpack_constraints(args.get("constraints")),
        "control_signals": _unpack_control_signals(args.get("control_signals")),
        "capture": _unpack_capture(args.get("capture")),
    }

    try:
        text, canvas_state, canvas_trace = await asyncio.to_thread(
            run_diffusion, model, prompt, **kwargs
        )
    finally:
        if run_id:
            _unregister_run(run_id)

    return {
        "text": text,
        "canvas_state": {
            "converged": canvas_state.converged,
            "committed_fraction": canvas_state.committed_fraction,
            "steps_used": canvas_state.steps_used,
            "thought": canvas_state.thought,
            "stray_thought_delimiter": canvas_state.stray_thought_delimiter,
            "turn_closed": canvas_state.turn_closed,
            "answer_tokens": canvas_state.answer_tokens,
            "finished_honestly": canvas_state.finished_honestly,
        },
        "trace_summary": _summarize_trace(canvas_trace, include_frames),
    }


async def cancel_run(manager: StateManager, args: dict[str, Any]) -> dict[str, Any]:
    """Set the cancel event for `run_id`, if a run is currently registered
    under it. Pure signal — never touches `manager` or the core directly;
    the running `generate` call's own `should_cancel` poll (checked once per
    step inside `dgemma.composite.StepEndComposite`) is what actually stops
    the loop, on ITS OWN thread, the next time it's checked."""
    run_id = args.get("run_id")
    if not run_id:
        return {"error": "run_id is required"}
    with _active_runs_lock:
        event = _active_runs.get(run_id)
    if event is None:
        return {"found": False, "run_id": run_id}
    event.set()
    return {"found": True, "run_id": run_id, "status": "cancel requested"}
