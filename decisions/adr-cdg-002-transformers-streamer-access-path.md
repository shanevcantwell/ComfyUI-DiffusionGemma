# ADR-CDG-002 — Use transformers DiffusionGemmaForBlockDiffusion + TextDiffusionStreamer as the inference access path

**Status**: superseded by ADR-CDG-004 (partial — drive seam only; see Supersession Relationships)
**Date**: 2026-06-30
**Related**: ADR-CDG-001 (socket types), ADR-CDG-004 (amends the drive seam and resolves the open question below, 2026-07-05)

---

## Context

The pack must run DiffusionGemma's denoising loop, and — for the trace/viz
nodes that are the project's whole point — needs **per-step** access to the
canvas, the per-slot entropy, and the commit set. Three runtimes are viable:

- **HF transformers** (`DiffusionGemmaForBlockDiffusion`) — reference impl.
- **llama-cpp-python over GGUF** — fast, low-VRAM; already running locally via
  the `llama-diffusion-cli` build from PR #24423.
- **vLLM** — production serving; integrates via model-runner v2 `ModelState`
  hooks with custom per-step sampling.

## Decision

Use **HF transformers `DiffusionGemmaForBlockDiffusion` with
`TextDiffusionStreamer`** as the primary access path, dropping to the documented
per-step denoising loop where trace capture needs it.

- **Phase 1** wraps `.generate()` for the thin text-out slice.
- **Phase 3** uses the streamer / explicit loop for per-step capture.

## Rationale

### Positive Consequences
- transformers publishes the full per-step loop and ships `TextDiffusionStreamer`.
- A working reference exists: the HF `diffusiongemma-3d-gen` Space streams one
  JSON frame per denoising step — exactly the `CANVAS_TRACE` shape we need.
- Step-level hooks are precisely what the instrumentation phase requires.

### Negative Consequences
- Slower than vLLM and heavier than the GGUF/llama.cpp path already running on
  the dev box. We are deliberately choosing the slower route for access.
- `top_k` and some standard `generate` flags are not available at release; the
  model always uses a KV cache and rejects `use_cache`.

## Alternatives Considered

### Option A: llama-cpp-python over the GGUF

Already running locally, fast, fits in ~18 GB VRAM at Q4_K_M.

**Why rejected (as primary):** per-step loop exposure in the Python bindings is
unconfirmed; `llama-diffusion-cli` is a separate target and the bindings may not
surface the step loop the trace nodes need. **Graduation trigger:** revisit as a
fast *inference-only* backend once the trace path is proven on transformers.

### Option B: vLLM ModelState hooks

Has genuine per-step custom-sampling hooks and matches HF accuracy.

**Why rejected:** overkill for a single-user local playground — heavier setup,
oriented to batched serving. **Graduation trigger:** if the pack ever needs
production throughput, revisit.

## Open Questions

- [x] **`mask_token=4` vs. pure uniform-state.** ~~The `llama-diffusion-cli` EB
      run prints `algorithm=4 ... mask_token=4`, which sits oddly against the
      "uniform-state, renoise to *random vocabulary*, no `[MASK]`"
      characterization we've been working from. Is `mask_token` a generic field
      shared with the masked-diffusion models llama.cpp also runs (LLaDA, Dream)
      and used only vestigially / as a renoise-or-pad target, or does
      `algorithm=4` lean on an absorbing mask in earnest?~~
      **Resolved 2026-07-05 (documentary; banked in `#2`):** pure uniform-state
      renoise, **no absorbing mask**. Grounding:
      - Diffusers' `DiffusionGemmaPipeline` doc runs `BlockRefinementScheduler`
        in "uniform corruption mode, `mask_token_id=None`"
        (https://huggingface.co/docs/diffusers/api/pipelines/diffusion_gemma).
      - The scheduler's `step()` docstring states it renoises uncommitted
        positions "with uniformly random tokens, matching DiffusionGemma's
        block refinement sampler"
        (https://huggingface.co/docs/diffusers/v0.39.0/en/api/schedulers/entropy_bound).
      - The default `EntropyBoundScheduler` config has no `mask_token_id`
        concept at all.
      - A source search of `generation_diffusion_gemma.py` (v5.13.0) for
        `mask_token` / `algorithm` / `absorb` returns zero hits
        (https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py).
      - `mask_token_id` exists on `BlockRefinementScheduler` only because that
        scheduler class is shared with `LLaDA2Pipeline`, which *does* use
        absorbing masking — the field is vestigial for DiffusionGemma, not
        evidence of one.
      **Consequence:** confirms ADR-CDG-001's "no MASK" claim as-is — no
      footnote, no `CANVAS_STATE` mask sentinel needed. This is a documentary
      resolution; empirical corroboration still lands in Phase 3 per the
      original trigger, but the type-design question is settled — build the
      type layer accordingly.

## Supersession Relationships

**Supersedes:** none
**Superseded by:** ADR-CDG-004 (partial — the *drive* seam: `diffusers`
pipeline + scheduler replaces raw `.generate()` + `TextDiffusionStreamer`. The
*load* seam, `DiffusionGemmaForBlockDiffusion.from_pretrained()`, and the
rejection of vLLM/GGUF-as-primary both stand unchanged.)
