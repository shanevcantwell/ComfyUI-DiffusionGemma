"""tests/test_live_view_mint.py — ONE-MINT enforcement for the live per-step
view (issue #188).

**Why this exists.** Before issue #188's fix, "which nodes are live-view-
capable" was declared twice with no shared source: python-side, each node
picked its own `send_sync` event name (`DGemmaSampler`:
`"dgemma.sampler.step"`, `DGemmaDenoise`: `"dgemma.denoise.step"`); JS-side,
`web/live_view.js` hardcoded a single event name and a single decorated node
type (`"DGemmaSampler"`), so `DGemmaDenoise`'s otherwise-correct push was
silently dropped. The fix mints both halves once, in
`surfaces/comfyui/live_view.py`:
`DGEMMA_STEP_EVENT` (the one event name) and `LIVE_VIEW_HIDDEN_INPUT` (the
hidden-input sentinel any live-view-capable node merges into its own
`INPUT_TYPES()["hidden"]`, which the JS extension reads generically off
`node.constructor.nodeData.input.hidden` rather than a hardcoded node-type
list — see `web/live_view.js`'s header comment for the full mechanism).

Four checks, mirroring `tests/test_socket_mint.py`'s house pattern:
1. The mint module exposes `DGEMMA_STEP_EVENT` and `LIVE_VIEW_HIDDEN_INPUT`
   with the expected shape.
2. No `surfaces/comfyui/*.py` file OTHER than `live_view.py` contains an
   inline `DGEMMA_STEP_EVENT`-shaped WebSocket event-name literal (a literal
   reintroduced at a node site, under a NEW name, is exactly the drift that
   caused this issue).
3. Every live-view-capable node (`DGemmaSampler`, `DGemmaDenoise`) declares
   the mint's hidden-input sentinel — round-tripping the mint against the
   nodes that actually use it.
4. `build_on_frame` is the SAME function object both nodes call (not two
   independently-authored lookalikes) — the actual mechanism that makes a
   future third node's live-view wiring "merge the sentinel, call the shared
   builder" rather than "write a new closure and hope it matches."
"""
from __future__ import annotations

import re
from pathlib import Path

from surfaces.comfyui import live_view
from surfaces.comfyui.denoise import DGemmaDenoise
import surfaces.comfyui.denoise as denoise_module
from surfaces.comfyui.sampler import DGemmaSampler
import surfaces.comfyui.sampler as sampler_module

_SURFACE_DIR = Path(live_view.__file__).parent
_MINT_MODULE_NAME = Path(live_view.__file__).name

# Scoped to string literals that look like a dgemma-namespaced WebSocket
# event name (lowercase-dotted, "dgemma." prefixed) — mirrors
# `test_socket_mint.py`'s `_INLINE_LITERAL_RE` shape for the socket-type
# mint, applied to this mint's event-name half instead.
_INLINE_EVENT_LITERAL_RE = re.compile(r'["\'](dgemma\.[a-z_.]+)["\']')


def test_mint_exposes_the_step_event_name():
    assert live_view.DGEMMA_STEP_EVENT == "dgemma.step"
    assert live_view.DGEMMA_STEP_EVENT.islower()


def test_mint_exposes_the_live_view_hidden_input_sentinel():
    assert live_view.LIVE_VIEW_HIDDEN_INPUT == {"dgemma_live_view": "DGEMMA_LIVE_VIEW"}


def test_no_inline_event_literal_outside_the_mint_module():
    """The grep-gate itself: walk every surfaces/comfyui/*.py file except
    live_view.py, and assert zero inline dgemma-namespaced event-name
    literals whose value is the minted event name (a per-node re-inlining,
    under whatever name, is exactly the #188 drift)."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(_SURFACE_DIR.glob("*.py")):
        if path.name == _MINT_MODULE_NAME:
            continue
        text = path.read_text()
        found = [m for m in _INLINE_EVENT_LITERAL_RE.findall(text) if m == live_view.DGEMMA_STEP_EVENT]
        if found:
            offenders[str(path.relative_to(_SURFACE_DIR))] = found

    assert not offenders, (
        f"inline live-view event-name literal(s) found outside {_MINT_MODULE_NAME}: "
        f"{offenders} — import DGEMMA_STEP_EVENT from live_view instead"
    )


def test_live_view_capable_nodes_declare_the_hidden_input_sentinel():
    """Round-trip: every node this pack ships that pushes live per-step
    frames must declare the mint's hidden-input sentinel — the JS
    extension's decoration/listen gate reads exactly this key
    (`node.constructor.nodeData.input.hidden`), so a node missing it is
    invisible to the live view regardless of whether its python side pushes
    correctly (the exact #188 failure mode for `DGemmaDenoise`)."""
    sentinel_key = next(iter(live_view.LIVE_VIEW_HIDDEN_INPUT))

    sampler_hidden = DGemmaSampler.INPUT_TYPES()["hidden"]
    denoise_hidden = DGemmaDenoise.INPUT_TYPES()["hidden"]

    assert sentinel_key in sampler_hidden, (
        f"DGemmaSampler is missing the live-view hidden-input sentinel {sentinel_key!r} "
        "— the JS extension will not decorate or listen for it"
    )
    assert sentinel_key in denoise_hidden, (
        f"DGemmaDenoise is missing the live-view hidden-input sentinel {sentinel_key!r} "
        "— the JS extension will not decorate or listen for it"
    )
    assert sampler_hidden[sentinel_key] == live_view.LIVE_VIEW_HIDDEN_INPUT[sentinel_key]
    assert denoise_hidden[sentinel_key] == live_view.LIVE_VIEW_HIDDEN_INPUT[sentinel_key]


def test_both_nodes_call_the_same_shared_on_frame_builder():
    """Not just two closures that happen to behave alike — the SAME function
    object, imported from the one shared module, so a future change to the
    push mechanism (e.g. a new payload field) lands for every live-view-
    capable node in one edit rather than needing to be copied N times (the
    exact shape of the pre-#188 drift: two independently-authored copies,
    one of which silently used a different event name)."""
    assert sampler_module.build_on_frame is live_view.build_on_frame
    assert denoise_module.build_on_frame is live_view.build_on_frame
