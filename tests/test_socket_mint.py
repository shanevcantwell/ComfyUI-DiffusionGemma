"""Grep-gate for the #35 R2 socket-type mint (ADR-CDG-008 Phase 1, issue #52 §2).

The `DGEMMA_*` socket-type strings are minted exactly once, in
`surfaces/comfyui/socket_types.py`. This test asserts against the module
OBJECT (not a hardcoded set of expected strings), so a future rename of the
mint module's path only churns this file's one import line, never its
expectations — per the plan's "enforcement surface asserts against the
module object" design.

Three checks:
1. The module exposes the minted `DGEMMA_*` constants.
2. No `surfaces/comfyui/*.py` file OTHER than `socket_types.py` contains an
   inline `"DGEMMA_..."` string literal whose value is one of the minted
   constants (a literal reintroduced at a node site is exactly the drift
   this gate exists to catch).
3. Every live node's `INPUT_TYPES`/`RETURN_TYPES` socket entry that looks
   like a `DGEMMA_*` value is drawn from the minted set — round-tripping the
   mint against the nodes that actually use it.
"""
from __future__ import annotations

import re
from pathlib import Path

from surfaces.comfyui import socket_types
from surfaces.comfyui.denoise import DGemmaDenoise
from surfaces.comfyui.encode import DGemmaEncode
from surfaces.comfyui.loader import DGemmaLoader
from surfaces.comfyui.sampler import DGemmaSampler
from surfaces.comfyui.trace import DGemmaTrace

_MINTED = {
    value
    for key, value in vars(socket_types).items()
    if key.startswith("DGEMMA_") and isinstance(value, str)
}

# Scoped to string literals whose VALUE looks like a minted socket type
# (upper-snake, DGEMMA_ prefixed) — this deliberately does not match
# DGEMMA_STEP_EVENT's value ("dgemma.sampler.step", lowercase-dotted), which
# is a WebSocket event name, not a socket type, and stays inline by design.
_INLINE_LITERAL_RE = re.compile(r'["\'](DGEMMA_[A-Z_]+)["\']')

_SURFACE_DIR = Path(socket_types.__file__).parent
_MINT_MODULE_NAME = Path(socket_types.__file__).name


def test_mint_exposes_the_three_named_socket_types():
    assert socket_types.DGEMMA_MODEL == "DGEMMA_MODEL"
    assert socket_types.DGEMMA_CANVAS_STATE == "DGEMMA_CANVAS_STATE"
    assert socket_types.DGEMMA_CANVAS_TRACE == "DGEMMA_CANVAS_TRACE"
    assert _MINTED == {
        "DGEMMA_MODEL",
        "DGEMMA_CANVAS_STATE",
        "DGEMMA_CANVAS_TRACE",
        "DGEMMA_CONSTRAINTS",
        "DGEMMA_CONTROL_SIGNALS",
        "DGEMMA_RUN_CONFIG",
        "DGEMMA_KV_CACHE",
    }


def test_mint_exposes_the_run_config_socket_type():
    """Issue #72, D-3: `DGEMMA_RUN_CONFIG` minted once in this module — the
    sampler's assembled seed+knob bundle threaded to `DGemmaRunLogWriter`."""
    assert socket_types.DGEMMA_RUN_CONFIG == "DGEMMA_RUN_CONFIG"


def test_mint_exposes_the_constraint_and_control_signal_socket_types():
    """ADR-CDG-010 D6 / ADR-CDG-011, issue #64 §3.3: minted ahead of any node
    wiring them — Phase 1 lands the engine-side payloads
    (`dgemma/payloads.py`: `Constraints`, `ControlSignals`) and this mint
    entry only; no node reaches these sockets yet (the surface/widget phase
    is out of scope, gated behind this one)."""
    assert socket_types.DGEMMA_CONSTRAINTS == "DGEMMA_CONSTRAINTS"
    assert socket_types.DGEMMA_CONTROL_SIGNALS == "DGEMMA_CONTROL_SIGNALS"


def test_mint_exposes_the_kv_cache_socket_type():
    """ADR-CDG-012 (issue #62 Phase 3, DV.3a / ratification Q-4): the
    `KV_CACHE` seam's socket string, minted here per rule 4
    (`IDENTITY⊥ENVELOPE`) — the payload dataclasses live in `dgemma/types.py`."""
    assert socket_types.DGEMMA_KV_CACHE == "DGEMMA_KV_CACHE"


def test_step_event_name_is_not_in_the_mint():
    """DGEMMA_STEP_EVENT is a runtime WebSocket event name, not a ComfyUI
    socket type (#52 §2) — it must never be picked up by the mint."""
    assert "dgemma.sampler.step" not in _MINTED
    assert not any(v.islower() for v in _MINTED)  # minted values are all upper-snake


def test_no_inline_dgemma_socket_literal_outside_the_mint_module():
    """The grep-gate itself: walk every surfaces/comfyui/*.py file except
    socket_types.py, and assert zero inline `"DGEMMA_..."` literals whose
    value is one of the minted socket types."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(_SURFACE_DIR.glob("*.py")):
        if path.name == _MINT_MODULE_NAME:
            continue
        text = path.read_text()
        found = [m for m in _INLINE_LITERAL_RE.findall(text) if m in _MINTED]
        if found:
            offenders[str(path.relative_to(_SURFACE_DIR))] = found

    assert not offenders, (
        f"inline DGEMMA_* socket literal(s) found outside {_MINT_MODULE_NAME}: "
        f"{offenders} — import from socket_types instead"
    )


def test_live_node_sockets_are_drawn_from_the_mint():
    """Round-trip: every DGEMMA_*-shaped socket value actually used by the
    live nodes must be a member of the minted set — asserted against the
    module object, so this stays valid even if the mint's own constant names
    change (only their values matter here)."""

    def _dgemma_values(*sockets):
        return {s for s in sockets if isinstance(s, str) and s.startswith("DGEMMA_")}

    loader_values = _dgemma_values(*DGemmaLoader.RETURN_TYPES)

    sampler_input = DGemmaSampler.INPUT_TYPES()
    sampler_model_socket = sampler_input["required"]["model"][0]
    sampler_values = _dgemma_values(sampler_model_socket, *DGemmaSampler.RETURN_TYPES)

    trace_input = DGemmaTrace.INPUT_TYPES()
    trace_canvas_socket = trace_input["required"]["canvas_trace"][0]
    trace_values = _dgemma_values(trace_canvas_socket)

    encode_input = DGemmaEncode.INPUT_TYPES()
    encode_model_socket = encode_input["required"]["model"][0]
    encode_kv_cache_socket = encode_input["optional"]["kv_cache"][0]
    encode_values = _dgemma_values(encode_model_socket, encode_kv_cache_socket, *DGemmaEncode.RETURN_TYPES)

    denoise_input = DGemmaDenoise.INPUT_TYPES()
    denoise_model_socket = denoise_input["required"]["model"][0]
    denoise_kv_cache_socket = denoise_input["optional"]["kv_cache"][0]
    denoise_values = _dgemma_values(
        denoise_model_socket, denoise_kv_cache_socket, *DGemmaDenoise.RETURN_TYPES
    )

    all_live_values = (
        loader_values | sampler_values | trace_values | encode_values | denoise_values
    )
    assert all_live_values, "expected at least one DGEMMA_* socket among the live nodes"
    assert all_live_values <= _MINTED, (
        f"live node socket value(s) not present in the mint: {all_live_values - _MINTED}"
    )


def test_kv_cache_socket_present_on_encode_and_denoise():
    """DV.3a's specific round-trip for the new seam: `DGEMMA_KV_CACHE`
    actually appears on both `DGemmaEncode`'s output and `DGemmaDenoise`'s
    optional input/output — a wire between the two is legal at the graph
    level, and a mis-wire to any other socket type is not."""
    encode_input = DGemmaEncode.INPUT_TYPES()
    denoise_input = DGemmaDenoise.INPUT_TYPES()

    assert socket_types.DGEMMA_KV_CACHE in DGemmaEncode.RETURN_TYPES
    assert encode_input["optional"]["kv_cache"][0] == socket_types.DGEMMA_KV_CACHE
    assert denoise_input["optional"]["kv_cache"][0] == socket_types.DGEMMA_KV_CACHE
