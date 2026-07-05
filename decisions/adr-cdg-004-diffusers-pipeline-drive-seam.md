# ADR-CDG-004 — Drive DiffusionGemma via the Diffusers pipeline + scheduler, not raw `.generate()` + `TextDiffusionStreamer`

**Status**: accepted
**Date**: 2026-07-05
**Related**: ADR-CDG-002 (access path — this ADR amends the *drive* seam only
and partially supersedes it), ADR-CDG-001 (socket types — the mask_token
resolution below confirms ADR-CDG-001's "no MASK" claim)

---

## Context

ADR-CDG-002 chose HF transformers `DiffusionGemmaForBlockDiffusion` +
`TextDiffusionStreamer`, planning to fall back to the documented per-step loop
for trace capture in Phase 3 if the streamer didn't expose enough. Grounding
this against the actual HF docs (banked in
`shanevcantwell/ComfyUI-DiffusionGemma#2`, 2026-07-05) surfaced two facts that
move the decision:

1. transformers' own docs state DiffusionGemma's primary home is Diffusers:
   "The Transformers implementation only receives bug fixes but no new
   features"
   (https://huggingface.co/docs/transformers/model_doc/diffusion_gemma).
2. Diffusers' `DiffusionGemmaPipeline` does **not** reimplement the model — it
   instantiates the same `DiffusionGemmaForBlockDiffusion` (transformers)
   class and drives it through a swappable scheduler
   (https://huggingface.co/docs/diffusers/api/pipelines/diffusion_gemma).

So the access path splits cleanly into two seams: **load** (which class holds
the weights) and **drive** (what runs the denoising loop and exposes per-step
state). ADR-CDG-002's load seam was already correct and does not change here.
Its drive seam — raw `.generate()` + `TextDiffusionStreamer` — is what this
ADR replaces, because the streamer path cannot deliver two requirements this
project cannot ship without:

- **Per-step commit mask.** `EntropyBoundScheduler.step()` /
  `BlockRefinementScheduler.step()` natively return `EntropyBoundSchedulerOutput`
  / `BlockRefinementSchedulerOutput` carrying `accepted_index`/`transfer_index`
  (the commit mask), `sampled_probs`, and `pred_logits`
  (https://huggingface.co/docs/diffusers/v0.39.0/en/api/schedulers/entropy_bound).
  The transformers streamer path exposes no commit mask anywhere — confirmed
  by reading `EntropyBoundSampler.accept_canvas()` in
  `generation_diffusion_gemma.py`
  (https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py,
  v5.13.0): it is internal-only, never surfaced to the caller.
- **Mid-generation constraint injection.** The Diffusers callback contract
  (`callback_on_step_end(pipe, step, timestep, callback_kwargs)`) lets the
  callback return `{"canvas": ...}` to overwrite the canvas mid-loop —
  first-class slot pinning, which `DGemmaConstraints` (plan.md Phase 5) needs.
  transformers' `decoder_input_ids` sets only the *starting* canvas, consumed
  once before the loop begins (`generation_diffusion_gemma.py` ~line 1153) —
  confirmed no mid-loop mechanism exists.

Two requirements do **not** move the decision, named so a future reader
doesn't reopen them: custom entropy/temperature curves (Phase 4) cost about
the same either way — a scheduler subclass vs. a `LogitsProcessor` — and
quantized loading is identical on both paths (same underlying model class;
bitsandbytes supported either way).

## Decision

**Amend ADR-CDG-002 to a hybrid access path.** Load stays as decided; drive
changes:

- **Load** (unchanged): `DiffusionGemmaForBlockDiffusion.from_pretrained()`
  (transformers).
- **Drive** (changed): `diffusers.DiffusionGemmaPipeline`, wrapping the loaded
  model, with a swappable scheduler — default `EntropyBoundScheduler`,
  matching the already-grounded defaults (`entropy_bound=0.1`,
  `t=[0.4, 0.8]`, `max_steps=48`, `confidence=0.005`, `canvas_length=256`) —
  instead of `model.generate()` + `TextDiffusionStreamer`.
- Per-step capture (Phase 3) reads the commit mask/logits/probs directly off
  scheduler `.step()` outputs and/or `callback_on_step_end_tensor_inputs`
  (`canvas`, `logits`), rather than reimplementing the entropy-bound
  commit/renoise/stop rule in `dgemma/sampling.py`.
- Constraint injection (Phase 5) uses the callback's `{"canvas": ...}` return
  to pin slots mid-loop.
- Custom curves (Phase 4) are implemented as a scheduler subclass, not a
  `LogitsProcessor` — the idiom the Diffusers scheduler API is built around.
- GGUF/llama.cpp remains a graduation-triggered *inference-only* backend
  (ADR-CDG-002, unchanged).

This is a **partial supersession**: only ADR-CDG-002's drive-seam decision and
its `mask_token=4` open question (resolved on ADR-CDG-002 itself) are
superseded. Its load-seam decision and its rejection of vLLM/GGUF-as-primary
stand unchanged.

## Rationale

### Positive Consequences
- Native commit mask (`accepted_index`/`transfer_index`) removes the P3 fork
  entirely: `dgemma/sampling.py` was only going to exist *if* the streamer
  didn't expose enough (ADR-CDG-002's own hedge). It doesn't, so Phase 3
  becomes a pure capture task — no reimplementation of entropy-bound sampling,
  no risk of drifting from the reference numerics.
- First-class mid-loop constraint injection is a documented pipeline feature,
  not something we'd have to hack into the transformers loop.
- Staying on the actively-developed side of the codebase (Diffusers, per the
  transformers docs' own bug-fixes-only framing) reduces the odds of building
  against a path HF later stops extending.
- The load seam is untouched, so any Phase 1/2 work already landed is not
  invalidated — this is a drive-seam swap, not a rebuild.

### Negative Consequences
- One more dependency surface: `diffusers` (≥0.39.0) alongside `transformers`
  (5.13.0) must track compatible versions for this model.
- The pipeline callback's documented `callback_on_step_end_tensor_inputs` only
  lists `canvas`/`logits`, not `accepted_index` directly (open question (a)
  below) — may need a thin pipeline subclass, an extra layer ADR-CDG-002's
  plan didn't anticipate.
- The scheduler-subclass idiom for custom curves is inferred from the API
  shape, not walked through in a worked example in the docs (open question
  (b) below).

## Alternatives Considered

### Option A: Keep ADR-CDG-002 as-is (raw `.generate()` + `TextDiffusionStreamer`, reimplement commit/renoise in `dgemma/sampling.py` for Phase 3)

**Why rejected:** Would require reimplementing
`EntropyBoundSampler.accept_canvas()`'s logic from scratch to get a commit
mask, duplicating the reference implementation with real risk of numeric
drift, purely to avoid taking on the Diffusers dependency. The commit mask and
constraint-injection requirements are hard requirements for this project
(trace/viz nodes, `DGemmaConstraints`), not nice-to-haves — the streamer path
cannot deliver them at all, not just less conveniently.

### Option B: Drop transformers, load directly through Diffusers

**Why rejected:** Diffusers' `DiffusionGemmaPipeline` does not reimplement the
model — it holds the same `DiffusionGemmaForBlockDiffusion` instance from
transformers. Loading "directly through Diffusers" is the same load seam
under a different call site; there is no independent alternative here, so
there is nothing to decide beyond keeping `from_pretrained()` as the load
path and wrapping it in the pipeline for driving.

## Open Questions

- [ ] **(a) Can `accepted_index` be threaded through
      `callback_on_step_end_tensor_inputs`?** The documented tensor-input keys
      are `canvas`/`logits` only; `accepted_index` may need a thin
      `DiffusionGemmaPipeline` subclass to surface.
      **Resolution trigger:** read `pipeline_diffusion_gemma.py` in Phase 1
      before committing `nodes/sampler.py`'s call shape.
- [ ] **(b) Scheduler-subclass idiom for custom curves is inferred, not
      documented with a worked example.**
      **Resolution trigger:** Phase 4, when `DGemmaEntropySchedule` is built —
      confirm the subclass contract against the actual `EntropyBoundScheduler`
      base class at that diffusers version.

**Diffusers version grounding:** docs referenced at v0.39.0; verified
installed in the repo venv alongside transformers 5.13.0.

## Supersession Relationships

**Supersedes:** ADR-CDG-002 (partial — drive seam and the `mask_token=4` open
question only; ADR-CDG-002's load seam and its rejection of
vLLM/GGUF-as-primary stand)
**Superseded by:** TBD

## References

- https://huggingface.co/docs/transformers/model_doc/diffusion_gemma
- https://huggingface.co/docs/diffusers/api/pipelines/diffusion_gemma
- https://huggingface.co/docs/diffusers/v0.39.0/en/api/schedulers/entropy_bound
- https://github.com/huggingface/transformers/blob/v5.13.0/src/transformers/models/diffusion_gemma/generation_diffusion_gemma.py
- `shanevcantwell/ComfyUI-DiffusionGemma#2` (grounding issue, 2026-07-05)
