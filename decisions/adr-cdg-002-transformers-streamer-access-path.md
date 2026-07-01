# ADR-CDG-002 — Use transformers DiffusionGemmaForBlockDiffusion + TextDiffusionStreamer as the inference access path

**Status**: accepted (implementation pending)
**Date**: 2026-06-30
**Related**: ADR-CDG-001 (socket types)

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

- [ ] **`mask_token=4` vs. pure uniform-state.** The `llama-diffusion-cli` EB run
      prints `algorithm=4 ... mask_token=4`, which sits oddly against the
      "uniform-state, renoise to *random vocabulary*, no `[MASK]`"
      characterization we've been working from. Is `mask_token` a generic field
      shared with the masked-diffusion models llama.cpp also runs (LLaDA, Dream)
      and used only vestigially / as a renoise-or-pad target, or does
      `algorithm=4` lean on an absorbing mask in earnest?
      **Resolution trigger:** investigate when instrumenting the loop in Phase 3
      (and/or read the PR #24423 sampler source). If it genuinely uses an
      absorbing mask, the "pure uniform-state" claim needs a footnote and
      `CANVAS_STATE` (ADR-CDG-001) may need a mask sentinel value.

## Supersession Relationships

**Supersedes:** none
**Superseded by:** TBD
