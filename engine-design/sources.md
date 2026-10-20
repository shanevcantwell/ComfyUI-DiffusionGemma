# Source register and reconnaissance

Observed 2026-10-19. Source inspection is not a runtime validation. References to files below are repository-relative to the named immutable revision unless stated otherwise. No third-party source is vendored in this package.

## Baselines

| ID | Repository | Revision | Role |
|---|---|---|---|
| CDG | shanevcantwell/ComfyUI-DiffusionGemma | fed377afc7b54f03cb7faa4dd798c80c15279d8a | Current public node contract and reference behavior |
| LLAMA | ggml-org/llama.cpp | 3bb63cbcbfb62056c84fe9e521b0f3d756d29c32 | Upstream architecture and idioms baseline |
| DG23 | danielhanchen/llama.cpp | dd7f73b19deb9e79dd441904cf0158af459cd034 | PR #24423 donor implementation |
| DG27 | lnigam/llama.cpp | dd0cf04459b0c4f43aa6667dbc0879ac0cd50323 | PR #24427 and historical CDG pin |

https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/tree/fed377afc7b54f03cb7faa4dd798c80c15279d8a
https://github.com/ggml-org/llama.cpp/tree/3bb63cbcbfb62056c84fe9e521b0f3d756d29c32
https://github.com/ggml-org/llama.cpp/pull/24423
https://github.com/ggml-org/llama.cpp/pull/24427

## Inspected implementation and local requirements

CDG: `__init__.py` (eight registered nodes); `dgemma/loop.py` (cache-injected drive and participant order); `dgemma/types.py` (frames and cache envelopes); `dgemma/capture.py` (actual entropy/top-k/full-distribution calculation); `pyproject.toml`; `tests/README.md`; `AGENTS.md`; `ARCHITECTURE.md`; ADR-CDG-012, -014, -015, -020; issues #131, #245, #265 and #284.

DG23: changed-file inventory, `src/models/diffusion-gemma.cpp`, `src/models/gemma4-common.h` integration, and `include/llama.h` diff. DG27: changed-file inventory, `src/models/diffusion-gemma.cpp`, KV/cache/context patch excerpts, and graph/CLI split. Both remain unmerged draft implementations at the observations in this conversation. Neither donor is a correctness oracle simply because it emits readable text.

Transformers: `src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`, inspected blob `7f6edfcae266fe8e30dadd5dd4a4a4930872944d`. Diffusers: `src/diffusers/schedulers/scheduling_entropy_bound.py`, inspected blob `5382190ec6bf2374bab0932e1f08a0423160a543`, and `src/diffusers/pipelines/diffusion_gemma/pipeline_diffusion_gemma.py`. Those live-main inspections explain mechanisms but do not replace a reproducible oracle lock. CDG pins Transformers 5.13.0 and only floors Diffusers at 0.39.0. I0 must resolve exact source, package, checkpoint and tokenizer hashes before measurements.

https://ai.google.dev/gemma/docs/diffusiongemma/explained
https://ai.google.dev/gemma/docs/diffusiongemma/model_card
https://github.com/huggingface/transformers/blob/main/src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py
https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_entropy_bound.py

## Upstream developer documentation inspected

At LLAMA: `AGENTS.md`, `CONTRIBUTING.md`, `docs/development/HOWTO-add-model.md`, `skills/code-review/SKILL.md`, `skills/add-new-model/SKILL.md`, `examples/model-conversion/README.md`, `ci/README.md`, and `src/llama-ext.h`. The directory inventories `docs/` and `docs/development/` were also inspected. These establish CPU-first review slices, C-first API conventions, model conversion checks, narrow changes, experimental API staging, and mandatory backend tests.

`llama-ext.h` is in `src/`, not `include/`, at this revision. It explicitly allows breaking staging changes; it is not a stable plugin ABI. A drop-in binary against arbitrary stock llama.cpp is not promised.

## Required continuation reads (not claimed completed here)

Before I1 model code: read the full donor files and diffs, at least ten current `src/models/` examples, three recent merged model additions, `include/llama.h`, `src/llama-memory*`, `src/llama-graph*`, `ggml/include/ggml-backend.h`, `examples/eval-callback/`, `docs/ops.md`, the backend operation tables, `docs/build.md`, `docs/build-profiling.md`, `docs/development/debugging-tests.md`, `docs/development/token_generation_performance_tips.md`, and `skills/slice-model/SKILL.md`. For integration read `tools/server/README-dev.md`, `tools/server/README.md`, `common/jinja/README.md` and parsing docs. For image planning read `docs/multimodal.md`, `tools/mtmd/README-dev.md`, Gemma4 multimodal conversion and image preprocessors. Follow the GGUF specification in ggml's `docs/gguf.md`. Record exact revisions and exclusions; do not claim exhaustive documentation coverage.

## Findings requiring discrimination

1. DG23 exposes request self-conditioning and phase state through `llama_model *`. Shareable weights must not own mutable per-run state. Test interleaved contexts.
2. DG27 loads encoder layer-output-scale tensors but its inspected transformer body applies `out_scale` in both phases. HF ties parameters but not every buffer. Use intentionally unequal synthetic encoder/decoder scalars; do not assume equality from tied weights.
3. The decoder's sliding-layer context is the retained prefix tail plus the entire canvas, not necessarily an ordinary query-relative sliding window. Test prefix eviction, full-canvas visibility and rollback.
4. CDG capture uses callback logits before scheduler temperature scaling, and the forward hook can already have constrained those logits. Its top-k weights are renormalized within the slice. Do not label this unconstrained raw evidence or sampler entropy.
5. EB acceptance uses a sorted cumulative entropy budget minus the largest accepted entropy, not an independent per-position threshold. Acceptance is recomputed every step.
6. A float32 distribution for 256 positions and 262144 tokens is 268435456 bytes (256 MiB), not the historical 134 MB note. Validate actual shape and dtype before budgeting.
7. Existing cache composition, mutable-cache ownership and block-offset problems are tracked separately. Reproduce and classify them; do not require the new engine to silently reproduce known data corruption.
