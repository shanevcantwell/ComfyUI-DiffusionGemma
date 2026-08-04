# Handoff — step flip-book built; entropy-bound sampler understood; clear-alpha shape

**Date:** 2026-07-05 (late) · **HEAD context:** work on `p3-instrumentation`.
Cold-start: `/orient`. This file carries the day's reasoning; the durable
artifacts are `tools/flipbook/` and this note. The design arc below was worked
out in-session and is banked here so it is not lost.

## What was built (real, on disk)

`tools/flipbook/flipbook.py` (+ `README.md`) — bridges the working
`llama-diffusion-cli` into a **navigable per-step flip-book** so you can flip
through individual DiffusionGemma diffusion steps at your own pace.
- Run: `python3 tools/flipbook/flipbook.py --prompt "<p>" --steps 48 --canvas-len 256`
- Output: `tools/flipbook/out/<slug>/index.html` (slider + ←/→ scrubber), plus
  `step_NNN.png/.txt`. `out/` is gitignored (regenerable).
- v2 features: **per-step temperature overlay** (an anneal gauge — read the temp
  it stopped at), **full-width PTY capture** (no more 80-col truncation), no
  phantom final-dump frame. Temp computed on the run's `max_steps` denominator so
  an early-stop visibly reads as "quit hot" (e.g. last frame ≈ 0.62, not 0.40).

## Key findings (grounded)

1. **DiffusionGemma is block diffusion — a diffusion decode over an AR Gemma
   MoE**, not diffusion-from-scratch (`DiffusionGemmaForBlockDiffusion`). The
   flip-book shows AR-macro structure + local remask. Output token is always the
   **argmax** (temperature-invariant); temperature gates *acceptance*, not token
   identity.
2. **Entropy-bound decoder mechanics** (source: `/srv/dev/llama.cpp-diffusiongemma/
   examples/diffusion/diffusion.cpp`): linear temp anneal
   `t = t_min + (t_max−t_min)·(cur_step/S)`, hot→cool, denominated on
   `S = max_steps`. Rejected positions get a **uniform-random renoise** (the
   ancestral-noise-blast analog). Early-stop = argmax stable
   (`stability_threshold=1`, trivially weak) **AND** mean entropy
   `< confidence_threshold` (0.005). Early-stop shares the anneal counter, so it
   **truncates the anneal** — observed runs stopped at t≈0.60–0.63 (~mid-anneal,
   still hot) → under-annealed / "unfinished" output.
3. **Lever to run the full anneal:** `--diffusion-eb-confidence 0.0` (mean Shannon
   entropy can't be <0 → stop never fires → full `max_steps`). Raising
   `max_steps` is *unreliable* (makes per-absolute-step temp hotter). `confidence`
   is discontinuous (repo issue #10) because it thresholds a per-step-sampled scalar.
4. **Failure-mode theory:** the rule minimizes *local* entropy (self-consistency),
   not correctness → converges into coherent-but-wrong basins; the correct token is
   high-entropy vs the committed context, so the rule renoises it away; escape is
   *undirected* (random token). "Keep it hot to keep twisting" = keep the system
   open longer before it closes around a (possibly wrong) fixed point.
5. **Sigma toolkit transfers** (don't hand-wave "not gaussian sigmas"): temp anneal
   is the sigma-analog; entropy is a *richer* handle (per-position,
   content-adaptive, the actual objective). Ports: schedule-shape (Karras/exotic
   curves), churn (= basin escape), img2img-strength (= seeded renoise, the
   DGemmaRenoise loose-end), SplitSigmas (= step-window resume, ADR-006). The
   frontier is where discreteness breaks the analogy (full-token renoise; per-position schedules).

## Design arc (decisions, for the alpha)

- **Standard vs (Advanced) tiers**, not right/wrong — keep both. The basic
  `DGemmaSampler` stays; the decomposed tier is additive.
- **ComfyUI has no runtime type enforcement** (`AlwaysEqualProxy`/`AnyType`); the
  only "typing" is frontend link-illumination. **Quiet bespoke types earn value by
  *steering* wiring** (only compatible sockets light up), not safety. Rule: bespoke
  when a value should reach a *narrow* set of CDG sockets (steering); stock when it
  should compose broadly (illumination). → bespoke `DGEMMA_MODEL` / heat /
  resume-state; **stock `IMAGE`/`STRING` for flip-book frames**.
- **Loader = clone "Load Diffusion Model"** (UNETLoader is the *DiT* loader; "UNET"
  is a legacy envelope name), **not** Load CLIP (arch-gated + wrong interface,
  source-confirmed). Style-after, not fork-from: list directories +
  `from_pretrained(local_files_only=True)`; `models/diffusion_models/`. Case for a
  new node is *mechanical* (stock detection returns None; the monkeypatch path
  forces a sigma/latent `BaseModel` interface DG can't honor).
- **Flip-book idiom** (from operator's ComfyUI graphs): EasyUse For-loop drives
  `index → scheduler_start_step`, `index+1 → end_step`; `canvas_state` is the loop
  accumulator (foldl); a Batch node is the scanl that builds the flip-book. EasyUse
  `Batch Any` (`easy batchAnything`) is binary, isinstance-dispatch (custom objects
  need `__add__`) → use stock `IMAGE`/`STRING` frames (batch natively). For-loop
  accumulators are wildcard (any object threads).
- **Sigma→heat translation node** — bridge RES4LYF/stock `SIGMAS` schedulers → DG
  heat with its own tuning. It *is* the honest parse-at-the-door (SIGMAS stays
  SIGMAS until explicitly translated). Needs the fork to accept a **per-step heat
  array** (bounded change: e.g. `--eb-temp-schedule <file>`) or it can only set
  endpoints (loses the curve — the point).

## Clear-alpha MVP shape

- **Backend: GGUF/llama.cpp fork** (Q4_K_M ~15.6 GB, consumer-fittable, *proven
  tonight*). NOT diffusers — that path is quant-blocked on the 48 GB box (AWQ dead,
  bnb walled, bf16 won't fit).
- **Nodes:** loader · `SIGMAS→heat` translation · run+flip-book.
- **One fork extension:** per-step heat array (unlocks the curves).
- **Defaults:** `confidence 0.0` (full anneal); bespoke `DGEMMA_MODEL`/heat; stock
  `IMAGE` frames.
- **Milestone 2 (the prize):** MXFP4-on-diffusers → native per-step stepping + the
  rich For-loop/`canvas_state` nodes — *if it loads*.

## Model loading — the biggest problem, reframed

Not VRAM (Q4_K_M fits consumer Blackwell; A4B MoE + `-ngl`/`-cmoe` offload covers
16 GB). The residual is **distribution**: the consumer-fittable inference is the
**DiffusionGemma llama.cpp fork** (`/srv/dev/llama.cpp-diffusiongemma`), not stock.
**Open question to pin:** is that fork public/buildable by a user today, or does it
need publishing? That decides whether the alpha ships now.

## Quant dead-ends (re-confirmed — don't re-burn)

- `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` is **dead** on `transformers==5.13.0`
  (a `param_element_size` gap **and** an arch-key **revision mismatch**:
  `model.decoder.layers.*` vs `model.encoder.language_model.layers.*`).
- The revision mismatch is **not an AWQ problem** — it **transfers** to any same-era
  third-party quant (MXFP4 included). Verify a candidate loads against the *current*
  modeling revision before betting.
- NVFP4 = Blackwell-only (too narrow). MXFP4 = portable-to-NVIDIA (suboptimal
  pre-Blackwell), several checkpoints exist.

## Also grounded / minor

- `mask_token=4` in the CLI startup log is **vestigial** — it belongs to the unused
  LLaDA absorbing-mask code path (`diffusion_generate`), not the entropy-bound path
  this pack uses. Not a contradiction of ADR-CDG-001's "no MASK". (loose-ends note.)
- **ADR-CDG-006 audit:** its bit-exact resume claim is NOT achievable as written —
  `ResumeState` omits `self_conditioning_logits` and `argmax_history` (real
  cross-step model state). Plus padding (Risk/Observability duplicates Negative
  Consequences). Fix on any revision.
- Env: `llama-diffusion-cli` at `/srv/dev/llama.cpp-diffusiongemma/build/bin/`;
  GGUFs — Q8_0 ~25 GB local (`/mnt/storage/LLMs/unsloth/...`), Q4_K_M ~15.6 GB on
  CIFS `/mnt/i` (the 3090 box). GPU free tonight.

## Next session (carried forward)

1. Write the **alpha-spec ADR** (superseding) that banks this arc as the build
   target — the "package it right" job, fresh heat.
2. Ground whether **MXFP4 checkpoints load against the current modeling revision**
   (decides if the diffusers/native-stepping path is revivable).
3. Pin the **fork's public/buildable status** (decides alpha shippability).
4. If reviving ADR-006: add the two missing `ResumeState` fields; cut the padding.

## Deliberate holds (operator's call, no clock)

- **Merge `p3-instrumentation` → `main`** = the 0.1.0 publish event. No forcing
  trigger; the operator's decision, not waiting on anything.
