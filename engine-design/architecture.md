# Architecture: instrumentable DiffusionGemma over llama.cpp

Status: proposed design; independent implementation review and real-weight validation pending. Source IDs refer to `sources.md`. The accompanying header is a proposed contract, not an existing llama.cpp API or working engine.

## 1. System boundary

Use llama.cpp for GGUF loading, device allocation, quantized operators, tokenizer support and graph execution. Do not build another tensor runtime. Maintain an exact upstream revision plus a small, ordered, attributed patch series. Derive model mathematics from the pinned Hugging Face reference, not by voting between the two draft PRs. DG23 supplies region-aware masks, phase-dependent scalars and Gemma4 reuse patterns. DG27 supplies context-oriented execution and device-resident sampling patterns. Every imported hunk gets source commit, original license, purpose and a validating test in the patch ledger.

The native product is a library with a versioned C facade. A persistent local worker links it; CLI, tests and the CDG adapter are clients. The default CDG transport is that worker, not repeated CLI invocations: model weights stay loaded, while DLL/symbol/CUDA runtime collisions with Python packages are isolated. A direct C facade remains usable by non-Python clients. Do not claim ABI compatibility with arbitrary installed llama.cpp binaries. Ship the worker and its matched native libraries together.

The C++ library imports neither Python nor CDG nor ComfyUI. The Python adapter owns conversion into `DGemmaModel`, `KVCache`, `DiffusionFrame`, `CanvasState` and `CanvasTrace`. Existing analysis, tally audit, image rendering and log writers remain consumers of those payloads. Native inference does not require loading a Transformers model. Torch CPU tensors may still be required by current CDG consumers; eliminating all Torch imports is not a release requirement.

    ComfyUI nodes / MCP tools / headless clients
                      |
              dgemma contract + adapter
                      |
         bounded local worker protocol (default)
                      |
                native C facade
                      |
          step controller + state ownership
                      |
      llama.cpp model graphs / memory / ggml backends

### Proposed implementation layout

`engine/upstream.lock.json` records base revision and patch hashes; `engine/patches/series` orders changes. `engine/include/dgemma-engine.h` is the facade. `engine/src/` contains the facade and worker, not a duplicate transformer. `dgemma/backends/native.py` owns adapter dispatch. Model/graph changes belong in the pinned llama.cpp tree; native tests belong beside its existing tests. The harness may choose final paths after a full current-tree inventory, recording the change without altering the boundary.

## 2. Ownership and lifetimes

A loaded model owns immutable weights, metadata and immutable derived embedding caches. A session owns the encoder prefix, decoder scratch, current canvas, self-conditioning, RNG state, stop history, constraints, capture leases and run status. No per-request phase, mutable KV tensor, prior logits, token buffer or stream belongs on a shared `llama_model`. This corrects the risky ownership shape of DG23 without presuming its current single-request CLI is already failing.

A native cache is a separately reference-counted logical snapshot. Encoding onto an input cache returns a new cache. A borrowed input must remain unchanged; use copy-on-write on the first mutation or a proved persistent-memory mechanism, not a promise that callers will avoid fan-out. Preserve absolute logical positions separately from stored sliding-window length. Cache handles invalidate on model unload and cannot resolve under another model revision. Session calls are single-writer; separate sessions may share weights. Synchronization belongs to the engine, never to undocumented caller folklore.

The C facade returns bounded host views only when requested. Borrowed views are valid until the next mutating call on that session or its destruction; retained evidence must be copied or explicitly leased. A release-build error crosses the C boundary as a status and diagnostic, never an uncaught C++ exception. File- or protocol-derived errors must not abort the process with an assertion. RAII owns buffers, handles and temporary graph state.

## 3. Two attention modes, one deliberately shared backbone

The normal encoder is causal over the prompt and completed blocks. It writes prefix K/V. A denoiser call reads that prefix and combines it with freshly computed canvas K/V, bidirectionally over the entire canvas. Its transient canvas K/V is not committed to prefix memory. Once tokens are finalized, the causal encoder re-encodes those token IDs before the next canvas. Denoiser hidden states are not substituted for that re-encoding.

Use phase-aware graph construction with explicit inputs. Share the Gemma4 transformer body where the mathematics matches, but do not erase encoder/decoder distinctions in input normalization, self-conditioning, layer scalars or masks. HF's tied parameter declarations do not establish equality of all buffers. An intentionally unequal encoder/decoder layer-scalar fixture must distinguish the two paths. The first forward pass must disable self-conditioning explicitly: zero logits followed by softmax would create a uniform distribution, not an absent signal.

For a prefix of length P, canvas length C and sliding window W, the target mask behavior is:

| Query / layer | Visible keys |
|---|---|
| Encoder / full | Causal prefix keys through the query |
| Encoder / sliding | Causal keys in the configured retained window |
| Decoder / full | All valid prefix keys and all C canvas keys |
| Decoder / sliding | Retained prefix tail, then all C canvas keys |

Do not apply a moving query-relative window across the canvas merely because the layer is marked sliding. Verify actual HF cache retention and masks, including W-1 boundary conventions and attention backend behavior. Tests must cross the first eviction and multiple block boundaries. The checkpoint's layer configuration, head widths and RoPE variants are loaded from validated metadata; the observed 25 sliding / 5 full pattern is a fixture, not a hard-coded universal architecture rule.

A separate prefix memory plus scratch canvas is the preferred semantics. Reusing an upstream cache with append-and-rollback is allowed only after tests prove that no eviction, position accounting, quantized-cache packing or failed operation can mutate the prefix. A scalar cache-length rollback is not that proof. First establish reference correctness with an unfused graph; optimize without changing the state machine.

## 4. Step contract and state machine

    READY -> PREFILLED -> CANVAS_READY -> PREDICTED -> STEPPED
                               ^                        |
                               +------ next step -------+
                                                        |
                                block finalization -> COMMITTED
                                                        |
                                              next canvas / DONE

`cache_encode` creates or advances causal context. `session_begin` starts a run against a borrowed immutable cache. `session_set_input` selects hard IDs or, only with a declared research capability, an embedding payload. `session_forward` produces a prediction without sampling or committing context. `session_sample` performs the configured sampler and interventions. `session_step` is a convenience composition of the same operations, not another algorithm. `session_commit` causally encodes the finalized token sequence, returning a new prefix snapshot. The library also exposes cancellation and state inspection.

A predicted result is tied to the session generation, canvas and step. It cannot be sampled after another mutation. Finalization reasons are separate: confidence/stability, EOS, requested budget, cancellation and error. Per-step acceptance fraction is not convergence, and convergence is not proof that the model ended its turn. The adapter preserves CDG's distinction between raw IDs, post-excision answer IDs, `turn_closed`, answer-token counts and trace data.

At I0, capture current behavior for requested lengths 1, 255, 256, 257 and 512. Fix the difference between requested output length, executed canvas length, returned padding and EOS trimming in the contract before implementing it. Do not guess that `gen_length` changes the trained canvas width. Padding IDs and logical positions must be explicit in all frame and cache interfaces.

### Reference entropy-bound sampling

For step s in a schedule of N, use `t=(N-s)/N` and `T=t_min+(t_max-t_min)*t` for the pinned reference profile. Thus the last step is at 1/N, not zero. Shape logits with constraints at the declared stage, divide by T once, and use the resulting distribution for sampling and acceptance entropy.

Sort position entropies ascending: h[0] ... h[C-1]. Accept each sorted position j for which `sum(h[0:j+1]) - h[j] <= entropy_bound`; map that mask back to canvas positions. Recompute it on every step. Accepted positions take their sampled IDs; the rest take uniform vocabulary draws. Do not replace this with `h[i] <= bound` or persistent unmasking. Record tie-breaking and numerical threshold margins.

Carry the previous step's actual self-conditioning distribution or its exactly equivalent representation. Do not reconstruct it using the next step's temperature, truncate to top-k silently, or apply temperature twice. Full-softmax self-conditioning is the conformance profile. Approximations require separate capability/configuration labels and quality evidence; they are not an unannounced fast path.

Keep stop-history, prior prediction, canvas, schedule coordinate, RNG state and pending interventions in a resumable internal state. A token-only save is not exact mid-run resumption. Public checkpoint/fork capability is advertised only when implemented and tested; accepted-but-unimplemented research designs are not silently promoted into release requirements.

## 5. Observation is not intervention

Expose three distinctly named prediction stages: `model_output` (after model softcap), `constraint_output`, and `sampler_input` (after temperature). Their entropies are different measurements despite sharing the unit nats. Forward hooks can change logits before CDG's callback; the existing capture is not necessarily unconstrained evidence.

The native frame carries stage identity, temperature, effective sampler controls, accepted mask, pinned mask, sampled canvas before post-step pinning, and next-input canvas after interventions. Compatibility adapter v1 maps legacy CDG `entropy`, top-k and full distribution to the actual callback-logit stage, preserving values rather than silently renaming them. New fields identify that stage. A strict research view can additionally request unconstrained logits. Record the semantic correction separately from an inference change.

CDG's existing top-k weights are normalized over the retained k candidates. Preserve that contract. A separate retained-mass field is needed to reconstruct full-distribution probabilities; never substitute conditional weights for them. The native ABI uses int32 token IDs; conversion to CDG int64 tensors is checked and explicit.

Preserve the source callback ordering: capture, cancellation check, optional beta rebuild, pin, control walker. The forward constraint mask happens earlier. The walker prepares the next step's controls; changing signal[0] behavior would be a semantic migration. Cancellation returns the last fully captured partial result and its exact boundary, not a fabricated completed answer. A failure of an analysis callback propagates unless the consumer explicitly marks it display-only and catches it.

Tier 0 reductions run on-device when possible and copy only O(C) data. Tier 1 copies O(C*k). Tier 2 copies O(C*V) only for budgeted steps. Turning capture on must not disable early stopping or switch the sampler algorithm. Scientific capture applies backpressure instead of dropping data; display-only frames may be coalesced only with explicit sequence/drop accounting, without changing retained scientific evidence.

## 6. Cache provenance and experimental seams

The cache envelope binds engine ABI and layout version, checkpoint/quantization and tokenizer hashes, layer pattern, per-layer head widths, dtype, device layout, RoPE parameters and applied-rotation convention, absolute positions, stored ranges, cumulative lengths and provenance. Provenance is a minting token sequence or source sequence identities plus an ordered edit script. Edited tensors also need a content digest; a recipe alone is not a saved experimental artifact.

Native snapshots can be passed without host copies. A portable raw-KV export is a distinct capability. HF `DynamicCache` memory is never reinterpreted as a native cache: an explicit per-layer conversion must validate shape, normalization, positions and RoPE conventions and measure numerical equivalence. Quantized caches require their own converter or a declared rejection. Re-prefilling a sequence is valid only for a cache that still has that sequence as a complete provenance record.

Research extensions are data operations, not arbitrary executable callbacks: replace/ablate a validated cache slice, inject canvas embeddings, or select an explicitly experimental mask profile. Default text release behavior remains reference-compatible. The facade reserves capability negotiation and tensor descriptors, not an unrestricted plugin framework. ADR-CDG-015's continuous embedding experiment remains a future capability, not something output-logit reweighting can impersonate.

## 7. Transport, safety and CDG integration

The default worker uses local pipes, no listening network service. Control envelopes are length-prefixed UTF-8 JSON with a 1 MiB cap; large tensors use separately bounded binary frames with dtype, shape, byte count and checksum. Token arrays and scalar telemetry may stay in small JSON frames. Check products, offsets and alignment for overflow before allocation. Reject unsupported versions, negative dimensions, wrong element types, invalid token IDs, malformed sequence ordering and inconsistent byte counts. No shell expansion, arbitrary path writes or pickle deserialization.

Use a separate control channel or reader thread so cancellation is not trapped behind a large tensor stream. Bound queues and outstanding leases. Timeouts and worker death invalidate handles and return complete previously received frames plus a failure state. On unload, free model sessions, device memory and handles; a new load cannot resolve an old handle. Never mix partial bytes into a valid frame.

Dispatch in `dgemma` before reaching Transformers-specific helpers. Preserve the single core entry contracts across ComfyUI and MCP; do not trust stale prose about which surface wraps which. Existing code is authoritative for current call topology. Cache encode/advance and prompt-plus-cache composition must use the same native model lifetime and registry as generation. Reuse existing chat-template and excision behavior; compare rendered token IDs, including thinking-mode newline differences, rather than just similar text.

The node graph remains unchanged except for explicit native model selection and any needed capability indicator. Unsupported combinations fail at ingress, before allocation or generation. Existing known fail-loud rejections remain documented until a tested repair removes them; silent cache contamination is never a compatibility requirement.

## 8. Packaging and upstream boundary

Build from a verified source lock and ordered patch hashes; record compiler, build flags, toolkit, target architecture, runtime dependencies, C ABI version and test-evidence digest. Release assets contain the worker, matched libraries, license notices and a checksum manifest, not model weights. Integrate with CDG's existing fetch/verify packaging rather than introducing an unpinned source-build fallback. Test offline startup after provisioning and absence/corruption/version mismatch failure paths.

Build CPU plus CUDA sm75/sm86 configurations for Windows and Linux. Initially certify actual tested host pairs, normally RTX-8000/Linux and RTX-3090/Windows; compiling a fourth combination is not evidence of its runtime support. The release manifest lists only measured supported combinations. No universal VRAM or throughput promise is made. Budget weights, embedding copies, prefix K/V, scratch, logits, self-conditioning and capture separately; a C=256,V=262144 float32 tensor alone is 256 MiB.

For upstream, separate: model/conversion correctness; a minimal experimental step/phase seam; a generic discrete-diffusion controller/example; independent backend optimization patches. Keep CDG protocol, registry and research policy outside libllama. Use `src/llama-ext.h` for an experimental upstream seam until maintainers choose a permanent API. Prefer existing evaluation callbacks for observation; their existence does not license cache mutation through them. CPU-first review slices need not delay the separately packaged CUDA product. Do not create another model-specific public CLI flag family or duplicate server.

Upstream eligibility additionally needs a qualified human maintainer who understands and will maintain the changes. That person need not be the operator. Automated tests and separate agent reviews cannot substitute for upstream's human-ownership requirement. This branch makes no upstream submission.
