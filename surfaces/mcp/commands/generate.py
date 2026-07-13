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
        run_diffusion,
    )
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
        run_diffusion,
    )

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
                "the load_model tool."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "The prompt to generate from"},
                    "seed": {"type": "integer", "description": "RNG seed (omit for nondeterministic)"},
                    "gen_length": {
                        "type": "integer",
                        "description": "Canvas length in tokens",
                        "default": DEFAULT_GEN_LENGTH,
                    },
                    "num_inference_steps": {
                        "type": "integer",
                        "description": "Number of denoising steps",
                        "default": DEFAULT_NUM_INFERENCE_STEPS,
                    },
                    "entropy_bound": {
                        "type": "number",
                        "description": "Per-step commit entropy threshold",
                        "default": DEFAULT_ENTROPY_BOUND,
                    },
                    "t_min": {"type": "number", "default": DEFAULT_T_MIN},
                    "t_max": {"type": "number", "default": DEFAULT_T_MAX},
                    "confidence": {
                        "type": "number",
                        "description": "Pipeline confidence_threshold for adaptive early-stop",
                        "default": DEFAULT_CONFIDENCE,
                    },
                    "thinking": {
                        "type": "boolean",
                        "description": (
                            "EXPERIMENTAL: inject the <|think|> control token via a system "
                            "turn (see dgemma.loop.run_diffusion's docstring for the honest "
                            "one-token gap versus native enable_thinking=True)."
                        ),
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
