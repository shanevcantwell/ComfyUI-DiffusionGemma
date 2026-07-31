"""surfaces/comfyui/live_view.py — live per-step view capability, minted once
(ONE-MINT, issue #188).

**Why this exists.** Before this module, "which nodes push a live per-step
view" was declared TWICE with no shared source: Python-side, each node
(`sampler.py`, `denoise.py`) built its own `on_frame` closure and picked its
own `send_sync` event-name string; JS-side, `web/live_view.js` hardcoded a
single event name (`"dgemma.sampler.step"`) and a single decorated node type
(`"DGemmaSampler"`). The two copies drifted the moment `DGemmaDenoise` landed
(issue #62 Phase 3): denoise.py grew its own working `on_frame` push under
its OWN event name (`"dgemma.denoise.step"`, deliberately namespaced apart —
see the pre-fix docstring history in git blame), but the JS extension never
learned that name, and never added `DGemmaDenoise` to its decoration gate.
Python-side emission was correct and complete for both nodes; the gap was
JS hardcoding a closed list nothing kept in sync (issue #188's own
mechanism section).

**The fix — single mint, two halves:**

1. `DGEMMA_STEP_EVENT` — ONE event name for every live-view-capable node
   (previously two: `dgemma.sampler.step` / `dgemma.denoise.step`). The
   payload's own `"node"` key already disambiguates by originating node id
   (`_build_on_frame`'s `unique_id` below) — a per-node-type event
   namespace bought nothing but a second string to keep in sync, which is
   exactly the drift this issue reports. Still NOT a ComfyUI socket type
   (`tests/test_socket_mint.py` deliberately excludes it), so it does not
   belong in `socket_types.py` — same rationale as before, now with one
   name instead of two.
2. `LIVE_VIEW_HIDDEN_INPUT` — a hidden-input DECLARATION any live-view-
   capable node merges into its own `INPUT_TYPES()["hidden"]` dict. ComfyUI
   serializes a node's ENTIRE `INPUT_TYPES()` return value verbatim into the
   `/object_info` payload (`server.py`'s `node_info`: `info['input'] =
   obj_class.INPUT_TYPES()`), which the frontend attaches to
   `node.constructor.nodeData.input` (grounded against the installed
   `comfyui_frontend_package` bundle — `nodeData?.inputs?.[name]?.tooltip`,
   `nodeData?.output_node`, etc. are all real reads against this same
   structure). So `web/live_view.js` can ask "does THIS node's own
   `nodeData.input.hidden` carry the live-view sentinel key?" generically,
   for any node type, with zero hand-maintained type list — the fix this
   issue asks for ("decorate/listen for any node type declaring the
   live-view hidden input").

Both `sampler.py` and `denoise.py` now merge `LIVE_VIEW_HIDDEN_INPUT` into
their `hidden` dict and call `build_on_frame` here (deduplicating what were
two near-identical closures) rather than each rolling its own event name and
closure body.
"""
from __future__ import annotations

import logging

# Event name for the live per-step push (plan.md Phase 3 (a); unified under
# issue #188 — previously two separately namespaced names, one per node,
# which is what let DGemmaDenoise's pushes go unheard). `send_sync`'s
# receiving side has no event-name whitelist (`loose-ends.md`), so any
# string works, but a collision with another pack's event name would
# silently cross-wire two unrelated `web/` extensions' `addEventListener`
# handlers — hence the pack-own prefix.
DGEMMA_STEP_EVENT = "dgemma.step"

# Hidden-input sentinel key any live-view-capable node's `INPUT_TYPES()`
# merges into its own `"hidden"` dict. The VALUE ("DGEMMA_LIVE_VIEW") is
# never used as a real ComfyUI socket type (no node ever inputs or outputs
# it) — it exists purely so `nodeData.input.hidden` carries a key the JS
# extension can `in`-check generically, the single mint this issue asks for.
# Deliberately not added to `socket_types.ALL_SOCKET_TYPES`: it never rides
# a wire between two nodes the way a real `DGEMMA_*` socket does.
LIVE_VIEW_HIDDEN_INPUT = {"dgemma_live_view": "DGEMMA_LIVE_VIEW"}


def build_on_frame(unique_id):
    """Build the live-push closure handed to `run_diffusion` as `on_frame`.

    Single shared implementation for every live-view-capable node
    (previously duplicated near-identically between `sampler.py` and
    `denoise.py`, one copy per node, each hardcoding its own event-name
    string — issue #188's python-side half of the drift). Lives here, not in
    `dgemma/loop.py` (ADR-CDG-003): this is the shared spot allowed to import
    ComfyUI server infrastructure on behalf of any surface node.
    `PromptServer` is imported lazily inside the closure (not at module top)
    so this module stays importable — and callers keep running — with no
    ComfyUI process alive (the normal pytest condition); a real live session
    is the only context where the import succeeds and the push actually
    fires.

    Display must never kill generation (review finding, 2026-07-05): the
    whole push — import, instance lookup, `send_sync` — is guarded, and any
    failure (no server, serialization error, dropped websocket) is logged
    and swallowed rather than propagated. The guard lives HERE, not in
    `dgemma/loop.py`'s hook site, deliberately: the engine's `on_frame`
    contract propagates callback exceptions (see `_FrameCollector`'s
    docstring — an engine that silently ate a user's analysis-callback error
    would be its own dishonesty), so the display-only closure guards itself
    at the layer that owns the display concern. A `send_sync` hiccup must
    not abort a multi-step 26B generation run.
    """

    def on_frame(frame) -> None:
        try:
            from server import PromptServer

            instance = PromptServer.instance
            if instance is None:
                return
            instance.send_sync(
                DGEMMA_STEP_EVENT,
                {
                    "node": unique_id,
                    "canvas_idx": frame.canvas_idx,
                    "step_idx": frame.step_idx,
                    "t": frame.t,
                    "temperature": frame.temperature,
                    "committed_fraction": frame.committed_fraction,
                },
            )
        except ImportError:
            return  # No live ComfyUI process (e.g. pytest) — skip the push, not an error.
        except Exception as exc:  # noqa: BLE001 — deliberate breadth: display-only, see docstring.
            logging.warning(
                "DiffusionGemma live push failed (display only, generation continues): %s", exc
            )

    return on_frame
