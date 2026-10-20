# Design decisions and review record

Status of all entries: proposed for implementation, not ratified by an independent reviewer in this chat. Authority is the user's requested product scope plus the inspected sources. Existing CDG ADRs are not edited. The eventual implementation must record amendments/supersession of ADR-CDG-020's no-owned-patch and inference-only limitations; a design branch does not silently change runtime policy.

## DE-01: Adapt llama.cpp, do not replace ggml

Use a pinned upstream base, attributed donor patches and shared graph helpers. Reject a new GGUF parser/tensor runtime, a wholesale merge of both draft PRs, and dependence on future upstream acceptance. This contains the maintenance surface while preserving upstream review options. Gate: inspect full donor diffs, test provenance and graph equivalence, and list every retained local patch.

## DE-02: Library plus persistent worker

The stable application boundary is a small versioned C facade; the default Python deployment isolates it in a persistent worker. Reject process-per-prompt CLI invocation, a stock llama-server-only contract, and direct exposure of C++/ggml internal structs to ComfyUI. Cost: a local protocol and bounded data copies. Gate: lifecycle, interruption, corruption, backpressure, fresh-install and DLL/runtime-isolation tests. Revisit direct in-process deployment only with measured benefit and isolation tests.

## DE-03: Session-owned state and immutable borrowed prefix

Request phase, canvas, RNG, K/V and self-conditioning are per-session. Shared weights remain immutable. Cache advance returns a new payload using tested copy-on-write or equivalent ownership. Reject model-global request setters and convention-only single-consumer safety. Gate: A/B/A replay, interleaved contexts, failed-advance rollback, unload invalidation and prefix digest checks.

## DE-04: Explicit forward/sample/commit seam

Native stepping separates prediction, sampling/interventions, and causal re-encoding. A convenience run path composes the same primitives. Reject output-only telemetry as a substitute for transformer access and reject latent-input experiments implemented merely as output-logit biasing. Gate: single-step reference comparison and manipulated-input propagation fixtures.

## DE-05: Preserve compatibility, expose measurement stages honestly

Map legacy fields to their actual source calculation while adding explicit stage labels and separating raw, constrained and temperature-shaped distributions. Reject silently changing existing field meaning or interpreting conditional top-k weights as full softmax mass. Gate: pinned legacy fixtures, stage-specific probability/entropy checks, and user-facing migration notes for any actual semantic change.

## DE-06: GPU observation must not change computation

Keep sampling and stop decisions equivalent with capture enabled or disabled. Copy reductions/slices rather than full tensors when possible. Reject an instrumentation switch that implicitly selects a different host sampler or stopping rule. Gate: fixed-input differential checks, same-backend trajectory comparisons, copy profiling and bounded-memory stress.

## DE-07: Text release waits for all eight existing nodes

Earlier I2 and I3 checkpoints are experimental. I4 completes current node and MCP parity; I5 certifies distribution. Accepted but unimplemented research capabilities do not become requirements by prose alone. Image workflows remain disabled. Gate: frozen node/mode inventory mapped to non-skipped real tests, black-box ComfyUI evidence, and clean-install matrix.

## DE-08: Technical gates are evidence-based, not operator exams

Each implementation phase has an implementer, a verifier and two separate review sessions (model semantics; systems/security). Findings reference exact candidate commits and artifact hashes. A failed technical gate returns to the responsible role. HITL checks assess usefulness and surprising behavior, not C++ correctness. Self-review is labeled self-review. The workflow checker checks records; it cannot prove that independently named sessions or human expertise are genuine.

## DE-09: Preserve licensing and contribution boundaries

CDG's existing license is unchanged. Do not copy GPL CDG implementation into an intended upstream MIT component and merely change its header. Keep provenance for reused llama.cpp code; inspect licenses for any HF code reuse and obtain a qualified compatibility review before distribution. A fresh implementation from documented behavior is not an excuse to erase actual code provenance. No relicensing decision, upstream PR description or automated upstream submission is included here.

## Self-review findings at D1

- Corrected the earlier oversimplification that tied weights imply identical encoder/decoder buffers.
- Corrected the independent per-position entropy-threshold description to the cumulative EB rule.
- Required source-based reconciliation of all node modes instead of treating README claims as implemented behavior.
- Split desired build matrix from certified runtime matrix; four compilations cannot establish four hardware/OS support claims.
- Kept raw-KV interoperability, checkpoint resumption and embedding intervention distinct capabilities.
- Marked no-GPU and missing full-checkout constraints explicitly. No engine test or independent review has passed here.
