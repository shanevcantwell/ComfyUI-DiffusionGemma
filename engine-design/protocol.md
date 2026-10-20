# Native worker and facade contract details

Proposed protocol version 1; no worker implementation is supplied. Review this contract in I1 and implement it in I3. This is a CDG-side transport, not a proposed new llama-server protocol.

## Local framing

Every message starts with a 20-byte little-endian header: four ASCII bytes `DGEM`, uint16 protocol version, uint16 message kind, uint32 JSON-byte length, uint64 binary-payload length. Kinds are request, response, event, error and control. JSON is UTF-8, a single object, at most 1 MiB. Payload length is checked against the negotiated budget before allocation. No native structs are written directly to the wire; padding, pointer width and endianness must not affect serialization.

The JSON envelope has `id` (monotonically increasing request integer), `op`, `model_id`, `session_id`, and `payload`. Responses carry the same ID; events carry `run_id`, `canvas_idx`, `step_idx`, `generation` and `event_seq`. Opaque IDs are process-local registry keys, never pointers. A frame event names its prediction stage and includes control values and stop reason. Status values distinguish invalid input, unsupported capability, resource exhaustion, stale state, cancellation and internal failure.

Binary tensors are described by `name`, `dtype`, `shape`, `offset`, `nbytes`, and SHA-256. For protocol v1 only dense contiguous little-endian int32, float32 and uint8 are required. Offsets are disjoint, bounded and account for the complete payload; validate products before addition/alignment. Shape/type must match the operation and the loaded model's validated limits. A checksum is transport integrity, not proof of model provenance. Native tensor views may have strides, but the worker packs a canonical representation at the transport boundary.

The handshake returns protocol/facade versions, engine build/source/patch hashes, actual backend/device, checkpoint/tokenizer identities, capability bits, maximum control/tensor/retention sizes, and numerical profile. Fail before model use on an incompatible major version. Unknown required fields/flags are rejected; optional extensions must be declared by capability. No automatic fallback to another model, numerical profile or unpinned executable.

## Operation mapping

| Worker operation | Native operation / ownership |
|---|---|
| `load`, `unload`, `status` | Model lifetime and capability query; unloading closes sessions and invalidates IDs |
| `encode` | Mint or advance a native cache from token IDs; return new cache ID and provenance |
| `begin` | Fresh run state against borrowed prefix, validated sampler/capture configuration |
| `forward` | Denoiser evaluation only; no context commitment |
| `prediction` | Explicit bounded copy of a named prediction stage; unavailable stages return state error |
| `sample`, `step` | Sampling/intervention primitive or its forward/sample composition |
| `commit` | Causal re-encode of finalized canvas, producing a new prefix handle |
| `cancel` | Separate control-channel request; safe while generation or frame transfer is pending |
| `release` | Release an owned cache/session ID and outstanding leases |

The normal Python core drives complete generation using these primitives or a worker-side composition of the identical controller. ComfyUI nodes and MCP wrappers do not acquire independent denoise loops. Tokenization and templating use the pinned model tokenizer/template; a worker implementation may expose bounded tokenize/detokenize helpers backed by existing llama.cpp vocabulary functions. Do not embed an unrelated template into the model graph. Compare actual rendered IDs against the reference before replacing its processor path.

## Facade input and output rules

Every extensible struct starts with `struct_size`; the model initializer also declares ABI version. Validate known-prefix size before access. Unknown nonzero reserved fields or unsupported flags fail. Mandatory output pointers must be non-null; on failure outputs remain null/empty. Tensor rank is 1-4; unused dimensions/strides are zero. Size arithmetic is checked before allocation. Token input uses int32 and embeddings float32 in the initial contract.

Exactly one of tokens/embeddings is supplied to `session_set_input`. The embedding-stage selector distinguishes unscaled lookup rows, scaled embeddings and post-self-conditioning input. Those are different intervention sites, never interchangeable. Embedding injection remains unsupported until its research gate and tests pass; declaring the selector is not an implementation claim. `get_prediction` is read-only and only valid when the named stage exists. Requesting a host tensor may synchronize/copy; normal forward execution must not force a full-logit download.

`session_sample` consumes the current prediction at most once. Per-step parameters declare effective controls and pin arrays; they cannot change the original schedule length. Invalid state transitions are errors. `session_step` composes the same forward/sample logic. The stop-reason enumeration is independent of per-position acceptance. `session_error` supports a size query with null output/capacity zero; with a buffer it copies at most capacity-1 bytes plus a terminator and returns the full required size including terminator.

Cache mutation is not hidden inside rendering, capture or handle lookup. A failed `encode`/`commit` leaves the input prefix unchanged and does not register a partial output. A model handle's release does not permit use-after-free of sessions; implementation must close or reject live children through documented ownership and retain backing allocations until outstanding operations finish.

## Backpressure and fault behavior

Default scientific mode allows a bounded number of outstanding complete frames, normally two, plus an explicitly budgeted large-tensor transfer. Producer blocks at this boundary. A dedicated control reader/atomic cancellation flag remains available while blocked. Consumer disappearance cancels generation, releases buffers and closes the session. Never deadlock cancellation behind the same full output queue.

Separate display events from retained scientific frames. Coalesced display updates report skipped display event numbers; scientific events are lossless or the run fails with a partial trace. An interrupted binary payload is not a valid event. Preserve earlier complete events and report the last complete sequence number. Write diagnostics to stderr, never into the protocol stream. Do not log prompt/canvas contents by default; explicitly requested experiment logs follow the declared evidence sink.

No network listener, arbitrary shell command, pickle, client-controlled output path or automatic download is part of the worker protocol. Model paths come from the explicit local loader configuration. Keep release-asset fetching and checksum verification in the existing installer/adapter boundary.
