"""ComfyUI socket-type vocabulary — minted once (ONE-MINT, #35 R2).

The DGEMMA_* strings are this surface's envelope; payloads are
`dgemma/types.py` dataclasses (the identity). No inline DGEMMA_* literal
may appear at any other site in `surfaces/comfyui/` — enforced by
`tests/test_socket_mint.py`'s grep-gate.

`DGEMMA_STEP_EVENT` (defined in `sampler.py`) is NOT a socket type — it is
a runtime WebSocket event name (`"dgemma.sampler.step"`), lowercase-dotted,
and deliberately excluded from this mint.

`DGEMMA_CONSTRAINTS`/`DGEMMA_CONTROL_SIGNALS` (ADR-CDG-010 D6 / ADR-CDG-011,
issue #64 §3.3): minted here ahead of any node wiring them — Phase 1 lands
the engine-side payload dataclasses (`dgemma/payloads.py`: `Constraints`,
`ControlSignals`) and this mint entry; no ComfyUI node wires these sockets
yet (the surface/widget phase is explicitly out of scope, gated behind this
one per ADR-CDG-010's Open Question 1 resolution trigger). `MUTABLE_TARGETS`
(the walker's bindable-knob registry) deliberately does NOT live here — it
names scheduler-config knobs the engine owns, not socket envelope strings
this surface owns (issue #64 §7 O4); it lives in `dgemma/payloads.py`.

`DGEMMA_RUN_CONFIG` (issue #72, D-3 of the design-gate ratification): the
sampler's assembled seed+knob+model-id bundle, threaded to
`DGemmaRunLogWriter` (`surfaces/comfyui/run_log_writer.py`). Its payload
dataclass, `RunConfig`, deliberately lives in `consumers/run_log.py`, NOT
`dgemma/types.py` — it is caller-supplied request-echo the surface already
holds, not core-measured state (see that module's docstring for the full
Option A vs. B rejection).

`DGEMMA_KV_CACHE` (ADR-CDG-012, issue #62 Phase 3, DV.3a / ratification Q-4):
the `KV_CACHE` seam's socket string — envelope, minted here per rule 4
(`IDENTITY⊥ENVELOPE`); the payload identity (`KVCache`/`Provenance`/`EditOp`
dataclasses) lives in `dgemma/types.py`. Rides `DGemmaEncode`'s output and
`DGemmaDenoise`'s optional cache input/output (`surfaces/comfyui/encode.py`,
`denoise.py`).
"""

DGEMMA_MODEL = "DGEMMA_MODEL"
DGEMMA_CANVAS_STATE = "DGEMMA_CANVAS_STATE"
DGEMMA_CANVAS_TRACE = "DGEMMA_CANVAS_TRACE"
DGEMMA_CONSTRAINTS = "DGEMMA_CONSTRAINTS"
DGEMMA_CONTROL_SIGNALS = "DGEMMA_CONTROL_SIGNALS"
DGEMMA_RUN_CONFIG = "DGEMMA_RUN_CONFIG"
DGEMMA_KV_CACHE = "DGEMMA_KV_CACHE"

ALL_SOCKET_TYPES = (
    DGEMMA_MODEL,
    DGEMMA_CANVAS_STATE,
    DGEMMA_CANVAS_TRACE,
    DGEMMA_CONSTRAINTS,
    DGEMMA_CONTROL_SIGNALS,
    DGEMMA_RUN_CONFIG,
    DGEMMA_KV_CACHE,
)
