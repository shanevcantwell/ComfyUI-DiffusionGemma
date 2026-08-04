# 2026-08-04 — ADR-CDG-012 Q-2 real-weights smoke + §2 liveness sweep: BLOCKED

**Protocol:** `q2-smoke-protocol.md` (design-gate-ratified, PASS_WITH_FINDINGS
2026-08-04; banked in full — see `q2-smoke-protocol.md` copy below), pre-
registered on issue #62
(https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62).

**Outcome: both §1 (Q-2 smoke) and §2 (liveness sweep) are BLOCKED** — not
FAIL, not PASS. The blocking condition is upstream of either protocol's own
observables: `dgemma.kv_cache.encode_sequence` (the shared mint step both
sections depend on to produce a donor `KVCache`) raises `torch.
OutOfMemoryError` **deterministically**, on every attempt, before any
`run_diffusion`/door code executes. Neither the Q-2 skeleton's with-cache
decode path nor the §2 door's exception-identity claim was ever actually
exercised with a real cache. Per protocol §1's own BLOCKED clause: "OOM under
CPU-spill before any decoder code runs... does not count as a FAIL data
point."

## Host identity

`inference-host`, Quadro RTX-8000, 48GB (49152 MiB reported), driver
580.173.02, CUDA 13.0, sm_75 (Turing). `torch==2.12.1+cu130`,
`transformers==5.13.0`, `diffusers==0.39.0` (main repo `.venv`, shared by
both the main-tree and skeleton-tree runs).

## What ran, in order

1. **Setup preconditions confirmed** (all via direct command output, not the
   `llauncher` MCP tools — not available in this session; verified by direct
   inspection instead):
   - `localhost:8081` (chat model): no `llama-server` process bound to that
     port — confirmed already swapped out.
   - `localhost:8082` (embeddinggemma, kept-resident): confirmed serving
     (`curl /v1/models` responded), 1402 MiB per `nvidia-smi` process list.
   - `nvidia-smi` baseline: 1631 MiB used total (Xorg 110 + gnome-shell 9 +
     embedding server 1402) — matches the #59 2026-08-03 precedent exactly.
     Banked verbatim: `nvidia_smi_baseline_before.txt`.
   - Weights cached: `google/diffusiongemma-26B-A4B-it`, 51.7G, confirmed via
     `scan_cache_dir()` (matches #59's ~51.7G evidence).
   - Pack/tree sync: main repo at `33551d50d46faaa13a65cdcd7314a49b5157df6f`
     (HEAD at window start), carrying the fail-loud door at
     `dgemma/loop.py:422-432` unmodified — confirmed by reading the file
     before any run.

2. **Isolated worktree/scratch branch built** (§1's runnability-floor
   requirement — the skeleton drive body the with-cache observables need):
   - Path: `/srv/dev/shanevcantwell/.worktrees/cdg-q2-skeleton`
   - Branch: `scratch/q2-skeleton-2026-08-04` (created off main
     `33551d5`, **not pushed**, shared checkout's branch never switched)
   - Change: `dgemma/loop.py` — the `kv_cache is not None` door's
     unconditional `raise NotImplementedError` replaced with a call to a new
     `_run_diffusion_with_injected_cache` helper. The helper skips the first
     encode (mirrors ADR-CDG-012 §4's "`DGemmaDenoise`... skips the first
     encode when a `KV_CACHE` is supplied") and denoises exactly one canvas
     block directly off the injected cache's `past_key_values`, reusing the
     REAL `_FrameCollector` as the per-step capture (not a hand-rolled
     frame-construction — lower risk of drifting from the tested collector's
     `t`/`temperature`/`committed_fraction_per_example` derivation). Full
     diff: `skeleton-loop.py.diff`. Explicitly NOT the Phase-4
     implementation (single block only, no multi-canvas loop, no OUT-1
     cache re-emission, no Opus design-gate review) — a smoke-only skeleton,
     named as such in its own docstring.
   - The no-cache path is byte-identical to main (confirmed by diff — the
     only changed lines are inside the `if kv_cache is not None:` branch).

3. **§2 liveness sweep — attempt 1** (`run_sec2_liveness_sweep.py`, main
   tree, unmodified door): loaded the model (bf16, `device_map="auto"`,
   CPU-spill; load wall-clock 31.4s per the script's own timer — see
   `sec2_results_attempt1.json`'s `load_wall_clock_s`), then attempted 12
   `encode_sequence(text)` mint calls (3 seeds × 4 texts). **All 12 mint
   calls failed before any `run_diffusion` call**: 9× `OutOfMemoryError`,
   3× `IndexError` (the empty-string text, `text_idx=2` — a genuine
   `encode_sequence` edge-case bug: zero-length `position_ids` breaks
   `transformers.masking_utils.find_packed_sequence_indices`'s
   `packed_sequence_mask[:, -1]` indexing; NOT gated by Q-2 per §0's own
   citation of `encode_sequence`'s docstring, scoped as a separate,
   pre-existing finding). 2 distinct `(status, exception_type)` hashes,
   not the 1 required for PASS. Raw: `sec2_results_attempt1.json`,
   `sec2_stdout_attempt1.log`.

4. **§2 liveness sweep — attempt 2** (same script, same code, fresh model
   load — re-run specifically to rule out a one-off fragmentation fluke
   before typing BLOCKED): **identical outcome**, same 2 hashes, same
   per-call exception types, at every one of the 12 calls. Raw:
   `sec2_results_attempt2.json`, `sec2_stdout_attempt2.log`.

5. **Root-cause isolation probes** (ungrounded exploration would have been
   scope creep on `encode_sequence`'s own implementation — these probes are
   grounding checks only, no code was changed to route around the block):
   - `torch.cuda.mem_get_info()` immediately after a bare model load: 6.52
     GB free of 50.7 GB visible. But the `OutOfMemoryError`'s own message
     (raised inside the encoder's MoE expert dispatch, via
     `accelerate.hooks.set_module_tensor_to_device`) reports only 347.56 MiB
     free and 45.33 GiB already resident in the same process at raise time —
     a real, large discrepancy between "free right after load" and "free at
     the moment `encode_sequence` calls the bare encoder."
   - A plain **no-cache** `run_diffusion` call (same model, same load,
     `gen_length=64, num_inference_steps=8`) **succeeded** in this exact VRAM
     regime — proving the OOM is not a hard capacity wall for this load in
     general.
   - Warming up via a `run_diffusion` call first, then calling
     `encode_sequence` on the same donor prompt: **still OOM'd**, same
     484.00 MiB allocation failure, same 347.56 MiB reported free — ruling
     out "cold first-call" as the sole cause.
   - Passing an explicit `attention_mask` to the bare `encoder(...)` call
     (the one argument `encode_sequence`'s call site omits relative to the
     pipeline's own internal encoder call, `pipeline_diffusion_gemma.py:
     326-332`): **still OOM'd**, identical error — ruling out the missing
     `attention_mask` as the proximate cause.
   - System RAM/swap: 58 GB available, 56 GB in reclaimable buff/cache — no
     host-side memory pressure; the effect is GPU-VRAM-side only.
   - Working hypothesis (NOT confirmed further — out of this smoke's scope
     to chase deeper): `dgemma_model.model.model.encoder` invoked as a bare
     standalone module call under `accelerate`'s CPU-offload hooks behaves
     differently from the SAME module invoked through
     `DiffusionGemmaPipeline.__call__`'s own internal per-block encoder call
     — something about the offload-hook materialization path taken only on
     a direct `encoder(...)` call leaves too little contiguous VRAM for the
     ~484 MiB MoE-expert weight move that both call shapes eventually need.
     This is a finding about `encode_sequence`'s existing (pre-smoke)
     implementation under this load regime, not about the KV-cache door
     Q-2 tests, and not something this smoke is licensed to fix (§0: the
     encoder's own call path is explicitly out of Q-2's scope).
   - Every probe left VRAM at the clean 1631 MiB baseline on process exit —
     no leak, no lingering resident state.

6. **§1 Q-2 smoke** (`run_sec1_q2_smoke.py`, **skeleton** tree,
   `scratch/q2-skeleton-2026-08-04`): loaded the model once (31.9s wall —
   see `sec1_results.json`'s `load_wall_clock_s`), attempted the 3 mint
   calls: **all 3 raised `OutOfMemoryError`**, identical shape to §2's.
   Because minting failed for every seed, the with-cache arm's `kv_cache`
   argument to `run_diffusion` was `None` for all 3 with-cache calls (the
   script's `caches.get(seed)` returned `None`) — confirmed directly in the
   raw JSON: every with-cache run's `string_sha256` is **byte-identical** to
   its same-seed no-cache run (e.g. seed 7: both arms hash to
   `87c9e11b32b3611e3062fe72b6873d9d4524f75bac1fbfaf43e1f85c97521ddf`), and
   every run's `injected_cache_provenance` is `None`. This is the honest
   signature of "no cache was ever actually injected" — not evidence about
   the skeleton's decode-path correctness one way or the other. All 6
   `run_diffusion` calls themselves completed successfully (`status=
   "success"`) at ~42-46s each, but as no-cache runs in both nominal arms;
   the skeleton's `_run_diffusion_with_injected_cache` code path was never
   entered. Raw: `sec1_results.json`, `sec1_stdout.log`.

## Typed outcomes (per protocol's fixed thresholds)

- **§2 (liveness sweep): BLOCKED.** Cause: `encode_sequence` OOM under
  bf16 CPU-spill, reproduced twice, before any `run_diffusion` call — the
  protocol's own named BLOCKED cause ("OOM under CPU-spill before any
  decoder code runs"). Not FAIL: the door itself (`dgemma/loop.py:422-432`)
  was never reached with a real cache to test its invariance against; the
  12 recorded outcomes are `encode_sequence` failures, not door-behavior
  observations. Re-open the window once `encode_sequence`'s OOM (or the
  CPU-spill load's VRAM margin) is independently addressed — NOT a repair
  this smoke is scoped to make.
- **§1 (Q-2 smoke): BLOCKED**, same root cause. H0a/H0b remain untested —
  no with-cache `run_diffusion` call in this window ever actually received
  a non-`None` `kv_cache`. The skeleton drive body itself (position/mask
  arithmetic, `injected_cache_provenance` stamping) was built and is banked
  (`skeleton-loop.py.diff`) but was **never exercised** — its correctness is
  simply unknown from this run, not confirmed or falsified.
- **Q-2 (ADR Open Question #1): remains OPEN.** No evidence either way was
  produced this window. Do not read the §1/§2 BLOCKED outcomes as license to
  proceed past the skeleton (§4 Non-goals already forecloses that reading:
  "A PASS licenses the drive body to proceed past its skeleton" — and this
  window produced no PASS).

## Findings to bank on #62 / as new issues (not resolved here)

1. `dgemma.kv_cache.encode_sequence` OOMs deterministically under bf16
   CPU-spill for realistic (non-empty) content, in a VRAM regime where
   `run_diffusion`'s own no-cache path succeeds — a real gap in
   `encode_sequence`'s production-readiness under the one load regime this
   repo has live E2E precedent for. Root cause not fully isolated (ruled out:
   missing `attention_mask`, cold-call ordering; not yet ruled out: the
   accelerate CPU-offload hook's per-call materialization behavior).
2. `encode_sequence("")` (empty token id list) raises `IndexError` inside
   `transformers.masking_utils.find_packed_sequence_indices` — a genuine
   edge-case gap, independent of finding 1, surfaced only because this
   protocol's §2 named `""` as a deliberate degenerate input to test.

## Artifact inventory (this directory)

- `README.md` — this file.
- `q2-smoke-protocol.md` — verbatim copy of the pre-registered protocol this
  run executed against.
- `run_sec1_q2_smoke.py` / `run_sec2_liveness_sweep.py` — the exact runner
  scripts executed (banked verbatim, not summarized).
- `sec1_results.json` — §1's full raw per-call record (mint_calls,
  run_calls, gating_notes).
- `sec2_results_attempt1.json` / `sec2_results_attempt2.json` — §2's full
  raw per-call record, both attempts.
- `sec1_stdout.log`, `sec2_stdout_attempt1.log`, `sec2_stdout_attempt2.log`
  — full captured stdout/stderr (includes the OOM/IndexError tracebacks
  in-line via the JSON, plus the load-progress bars).
- `nvidia_smi_baseline_before.txt` — pre-window GPU baseline.
- `skeleton-loop.py.diff` — the full diff of the Q-2 skeleton drive body
  against main, as built in the isolated worktree.
