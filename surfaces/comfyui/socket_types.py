"""ComfyUI socket-type vocabulary — minted once (ONE-MINT, #35 R2).

The DGEMMA_* strings are this surface's envelope; payloads are
`dgemma/types.py` dataclasses (the identity). No inline DGEMMA_* literal
may appear at any other site in `surfaces/comfyui/` — enforced by
`tests/test_socket_mint.py`'s grep-gate.

`DGEMMA_STEP_EVENT` (defined in `sampler.py`) is NOT a socket type — it is
a runtime WebSocket event name (`"dgemma.sampler.step"`), lowercase-dotted,
and deliberately excluded from this mint.
"""

DGEMMA_MODEL = "DGEMMA_MODEL"
DGEMMA_CANVAS_STATE = "DGEMMA_CANVAS_STATE"
DGEMMA_CANVAS_TRACE = "DGEMMA_CANVAS_TRACE"

ALL_SOCKET_TYPES = (DGEMMA_MODEL, DGEMMA_CANVAS_STATE, DGEMMA_CANVAS_TRACE)
