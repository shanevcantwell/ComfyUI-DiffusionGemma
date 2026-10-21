# Verification specification and release gates

These are required tests for a future engine implementation, not a report that they have passed. Workflow policy, C/C++ header syntax, the I0 authenticated worker boundary, and the BASE-01/BASE-02 oracle lanes are executable; later native-engine groups remain future gates. Full requirement coverage is a target measured by the matrix below, not a claim of 100% engine code coverage.

## Oracles and evidence

I0 locks HF source/package versions, tokenizer/chat template, original checkpoint shards, GGUF conversion script and outputs, sampler configuration, seed/draw fixtures, and the exact CDG node/mode inventory. Record SHA-256 hashes, not moving model names. Compare a native unquantized conversion with the original checkpoint before using a quantized model as a target. Treat the two draft PRs as independent comparison arms, not the oracle.

The BASE worker lanes execute through `tests/run_i0_group.py` and `tests/i0_oracle_worker.py`; they are not plan-only placeholders. Production evidence is accepted only from a clean committed candidate. The parent reads `i0-harness.lock.json` from the candidate Git blob, verifies HEAD commit/tree, the exact relevant tree inventory, and every runner/worker/support/test/schema/reference-lock worktree descriptor against both its committed blob and lock identity. It retains those descriptors and passes them to the child; the child imports support and test modules through a retained-byte loader and reads schemas/reference lock from the retained snapshot, never by authenticate-then-reopen. Development validation therefore creates an isolated temporary repository, overlays candidate bytes, makes an actual temporary commit, and runs there; it never labels dirty execution with the old `acdbdaa` identity.

Parent and child enter through the exact launcher with `-I -S -B -u`. A Git-pinned external Merkle manifest closes the exact stdlib and dedicated site-packages roots, rejects extra files and startup injection (`.pth`, `sitecustomize`, `usercustomize`), and verifies every top-level distribution `RECORD`; no arbitrary user-site/support root is derived. The interpreter image is opened once and executed through `/proc/self/fd`, while the child binds the retained image descriptor to `/proc/self/exe`. The result contains the parent nonce plus exact candidate/tree, environment, interpreter, worker, source/test, lock, evidence, blocker/category, disposition and exit identities.

The parent imports no Transformers, Diffusers, or CDG product module and calls `subprocess.Popen` directly; the separate enumerated boundary-fault entry cannot return production success. A selector drains bounded result/stdout/stderr pipes concurrently. Any Python-level write is stopped by fail-on-write wrappers, any fd-level byte is parent-visible BLOCKED, and `-u` prevents buffered output from escaping detection. The result fd must be a unique write-only pipe and is made non-inheritable. Worker at-fork handlers close result/stdout/stderr in fork children; the Linux subreaper plus pidfd-tracked descendant set contains even fork+`setsid` faults without signaling a reused PID/PGID. Spawn failures close every parent-owned fd. The final oracle closure assertion is followed only by one bounded `os.write` and one `os._exit`; process exit remains the sole cleanup boundary.

For precommit work, create a disposable clone/worktree, copy the candidate patch, stage all candidate paths, make a temporary local commit, provision the locked launcher/evidence pointers in its ignored `build/` tree, and run CMake/CTest and BASE commands there. Do not commit that validation commit to the implementation branch and do not run production BASE against a dirty checkout.

Use teacher-forced differential runs: same prefix tokens, canvas IDs, previous self-conditioning, masks, positions and explicit random draws. Compare encoder K/V, selected residuals, raw logits, constrained logits, sampler probabilities, entropy and accepted masks. Compare max absolute/relative error, normalized RMSE and full-distribution KL where appropriate. Calibrate and lock tolerances on baseline numerical/reduction noise before testing a candidate change; record absolute tolerance for near-zero values. No post-hoc threshold inflation. Near-tied argmaxes and EB boundary margins are reported explicitly, not hidden by a global cosine score.

For stochastic end-to-end quality, use a frozen prompt set spanning ordinary text, code, constraints, thinking, empty prompts and multi-canvas output. Bank all seeds, including degeneracy. Cross-framework equal seeds do not guarantee equal RNG streams. Exact trajectory/replay claims are restricted to a declared engine/backend/RNG configuration; distribution-level comparisons use fixed inputs and common draws. Ordinary AR perplexity is a regression check for unaffected AR models, not automatically a valid diffusion-likelihood metric.

Every mandatory test group needs a nonempty log/artifact, executed count >0, zero failed/error/skipped cases, candidate commit and relevant binary/source hashes. A successful pytest process whose required live tests all skipped is a failed gate. Known broken reference behaviors get reproductions and explicit classifications: supported behavior, intentional rejection, defect to repair, or unsupported research capability. Do not preserve silent data corruption merely to get golden equality.

## Test catalog

| ID | Required cases and discriminators |
|---|---|
| BASE-01 | Extract all eight registered classes, every INPUT_TYPES choice/optional socket, output type and `/object_info` contract. Compare pinned source with current checkout. Changed inventory blocks promotion until a recorded delta. |
| BASE-02 | Lock checkpoint, tokenizer, converter and HF implementation hashes. Resolve requested length versus canvas width and all source conflicts in `sources.md`. Verify ten model reads and three current merged examples before graph implementation. |
| GRAPH-01 | Tiny deterministic GGUF with distinguishable encoder/decoder scalars; tensor-name mapping, tied weights versus untied buffers, norms, softcap, MoE routing, per-layer head widths and RoPE. Reject missing mandatory tensors. |
| GRAPH-02 | Causal encoder masks and decoder prefix-plus-whole-canvas masks. Lengths P=0,1,W-1,W,W+1 and >2W; C=1 and 256 where legal. Explicitly check far-left/far-right canvas visibility and retained-prefix boundary. |
| GRAPH-03 | Self-conditioning absent on step 0, full-softmax on later steps, changing temperature, saturated logits, quantized embeddings, unequal phase scalars, full versus declared top-k approximation. |
| GRAPH-04 | Original versus F16/BF16 GGUF then Q8_0 and Q4_K_M fixed-input numerical comparisons on multiple steps/blocks. Verify baseline Gemma4 regression before attributing donor differences to DG changes. |
| SCHED-01 | Hand-calculated EB entropy sets including bound=0, ties, all-equal, highly skewed, NaN/Inf rejection and a set that distinguishes cumulative-minus-max from independent thresholding. Fresh decommit every step. |
| SCHED-02 | Temperature endpoints N=1,2,48; no double scaling; effective controls; walker applies next-step values with current signal[0] semantics; illegal controls rejected before work. |
| SCHED-03 | Common-draw sampler oracle, vocabulary bounds, uniform renoise and replay stream accounting; exact same-backend restart only under stated determinism conditions. |
| SCHED-04 | Stability plus mean-entropy early stop; budget exhaustion; EOS at first/middle/last position; repeated and absent thought delimiters; acceptance fraction distinct from turn completion. |
| STATE-01 | Two sessions sharing weights: A/B/A and interleaving. Repeated load/unload, failed allocation and failed forward leave no stale state or prefix mutation. Sanitizer coverage of ownership branches. |
| STATE-02 | Cancel before work, after forward, after capture, during binary transfer and on final step. Preserve exactly the last complete partial evidence; test callback error, worker death and recovery. |
| CACHE-01 | Encode then advance equals one causal encode within tolerance. Original cache unchanged after advance, failed advance and consumer fan-out. Model reload invalidates registry handles. |
| CACHE-02 | Inject context A, compose templated turn B once, verify token IDs, offsets and cache provenance. Exercise multi-block composition including existing rejection/defect reproductions. |
| CACHE-03 | Multiple blocks across window eviction; prefix hash invariant across denoising; separate global/sliding cumulative positions. Native and quantized cache layouts tested or explicitly rejected. |
| CACHE-04 | Wrong model/tokenizer/RoPE/head/layout/dtype/position fingerprint; orphan edited cache; stale handle and malformed raw-KV imports. No reinterpretation of HF objects. |
| TRACE-01 | Distinguish model, constrained and sampler stages at T!=1. Forward-mask versus post-pin ordering. Legacy fields match actual CDG calculations; accepted and pinned masks remain distinct. |
| TRACE-02 | Top-k conditional weights sum to one; full-distribution mass and entropy differ appropriately. k=0,1,V,>V. Optional fields absent versus zero. Batch>1 explicitly supported or rejected. |
| TRACE-03 | Tier-2 byte multiplication, dtype, retained-step/byte budget, OOM, last/all retention, overflow and malformed tensor lengths. Verify 256 MiB float32 example. |
| TRACE-04 | Capture off/Tier0/Tier1/Tier2 does not alter stop rules or the declared exact sampler. Record D2H bytes, latency and peak memory; retain scientific frames under backpressure. |
| NODE-01 | DGemmaLoader native selection, invalid/missing/corrupt model, unload/reload, capabilities and correct handle type. Existing Transformers none/autoround options remain backend-specific, not silently mapped to GGUF. |
| NODE-02 | DGemmaSampler text, state, trace, frames and IMAGE batch; all current knobs, seed, thinking and constraints; short/multi-block lengths and malformed input. |
| NODE-03 | DGemmaTrace commit and entropy modes, empty/partial traces, block coordinates and valid image dimensions. |
| NODE-04 | DGemmaTallyAudit receives real generated frames; deterministic fixture outputs match existing pure consumer. |
| NODE-05 | DGemmaRunLogWriter round-trip JSONL, escaped text, partial runs, file permission/error handling and atomic record boundaries. |
| NODE-06 | DGemmaEncode mint and advance, template-free path, fan-out and model mismatch; actual graph-cached input reuse. |
| NODE-07 | DGemmaDenoise with no cache, injected cache, composed prompt/cache, pins, controls and cancellation; supported multi-block behavior or existing explicit rejection. |
| NODE-08 | DGemmaTokenTrace raw pre-excision IDs versus cleaned text; top-k and full distribution retain token identity. |
| SURFACE-01 | All MCP load/status/encode/generate/cancel operations, model-scoped LRU registry, replacement eviction, partial results, concurrency and stale IDs. Both surfaces use the same core semantics. |
| SURFACE-02 | Real headless ComfyUI `/prompt`, `/history`, `/ws`, `/object_info`, `/interrupt`: execute graphs covering every registered node and every supported mode; import no implementation into the black-box assertions. |
| SECURITY-01 | Fuzz malformed GGUF metadata, overflow-before-alignment, wrong tensor/array element types, negative sizes, token IDs, truncated binary frames, unknown ABI/version, untrusted filenames and forbidden serialization. ASan/UBSan; TSan on supported CPU session tests. |
| PORT-01 | CPU builds and targeted tests on Linux/Windows; CUDA builds for sm75/sm86; actual runtime evidence on the declared host pairs. Changed ggml operations cross-checked on at least CPU and CUDA with test-backend-ops. |
| DIST-01 | Fresh install of packaged worker/libraries and CDG; no compiler required for users; checksum/source manifest, wrong ABI, missing runtime, download interruption, offline-after-provisioning and clean uninstall. |
| REGRESS-01 | Existing CDG fast/live/e2e suites with explicit markers and audited JUnit; relevant llama.cpp CI, quantize/imatrix behavior, unaffected Gemma4 logits, AR perplexity and benchmark regressions. |
| REVIEW-01 | Two separate role reviews of the exact tested candidate: model semantics and systems/security. Findings resolved with rerun evidence; contributor/maintainer ownership and licensing review before upstream submission or public distribution. |

## Phase gates

I0 resolves the baseline and oracles. I1 validates portable graph/math and the proposed API. I2 provides basic prompt-to-text plus Tier-0/trace functionality on real weights, followed by an optional hands-on checkpoint. I3 adds constraints, controls, observation, worker and core-adapter behavior for limited current workflows, followed by another optional checkpoint. I4 must execute every current node and MCP operation. I5 certifies fresh-install and platform distribution with all earlier tests rerun at the candidate revision. Public release eligibility is false before I5.

Each phase's test IDs are declared in `workflows/text.json`; gates require the cumulative set through that phase. Later changes cannot borrow old passing logs instead of retesting the new candidate. A prerequisite gate receipt must be banked and refer to an ancestor candidate; the harness checks ancestry separately from the receipt validator. A record is evidence only for the bytes it names.

## Commands for the code harness

After implementing/registering the native tests, run `cmake -S <llama-tree> -B <build> -DLLAMA_BUILD_TESTS=ON`, build, then `ctest --test-dir <build> --output-on-failure --no-tests=error`. Select focused test groups only when their collection counts are verified nonzero. Use the pinned build documentation for CUDA/toolchain flags, not a guessed current toolkit. Run `bash ci/run.sh <results> <mount>` for the upstream CPU lane and `GG_BUILD_CUDA=1 bash ci/run.sh <results> <mount>` for a scheduled CUDA lane.

Run CDG in its own reference environment: `python -m pytest -o addopts= --import-mode=importlib -m "not live and not e2e" --junitxml=<fast.xml> tests`, then corresponding explicit `-m live` and `-m e2e` jobs. Parse JUnit and fail required groups on skips, xfails, empty collection or missing fixtures; exit code zero alone is insufficient. Do not seize a busy GPU or kill unrelated processes; use the harness's resource lease and record the actual device, driver and free memory.

Before I5, bank coverage reports for changed C++/Python code and prove every matrix requirement maps to tests. Review uncovered branches in cache ownership, cancellation, bounds and allocation errors. A high mocked coverage percentage is neither numerical correctness nor node-wiring coverage. Zero unresolved critical defects and zero skipped mandatory release cases are required.
