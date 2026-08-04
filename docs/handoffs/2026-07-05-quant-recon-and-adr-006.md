# Handoff — P3 merge deliberately deferred; quant dead-end grounded; ADR-CDG-006 designed

**Date:** 2026-07-05 (fourth session of the day) · **From:** orchestrator session
(quant recon + ADR audit + ADR-CDG-006 design) · **HEAD at handoff:** `c03fd8e`
on `p3-instrumentation`

Cold-start path: `/orient`. Supersedes `2026-07-05-p3-publish-armed.md` — its one
open loop (#12 eyeball) resolved **PASS** this session, but the merge it gated
was **not taken** (see below — that file's "merge blocked" framing is now stale).

## State in one line

#12/#13 verified PASS and closed. Merge to `main` is available anytime (auto-
publishes 0.1.0 to registry.comfy.org) but was **deliberately deferred** this
session — the operator chose to keep building on `p3-instrumentation` (quant
recon, ADR audit, new ADR-CDG-006 design) rather than merge immediately.
`p3-instrumentation` is 8 commits ahead of `main`, unmerged, pushed to origin.

## What happened this session, in order

1. **#12/#13 closed, `verified` label applied** — operator eyeball confirmed the
   live per-timestep readout renders as a bottom layout-participating widget,
   values populate/count per step. Comments posted on both issues.
2. **Quant path (issue #4): AWQ-INT4/compressed-tensors checkpoint is a DEAD
   END.** Smoke-tested `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` against
   this pack's pinned `transformers==5.13.0`. Two stacked, unworkaround-able
   failures: (a) `CompressedTensorsHfQuantizer` doesn't override
   `param_element_size()` for the pre-load VRAM warmup, so it OOMs assuming
   full-precision size before touching the packed weights; (b) bypassing that,
   the checkpoint's backbone weights live under `model.decoder.layers.*` but
   the pinned transformers expects `model.encoder.language_model.layers.*` —
   662/1059 param names don't line up, checkpoint was quantized against a
   different modeling-code revision. Full grounding on #4's comment thread.
3. **Accessibility reframe (operator).** The quant target is NOT "fits the
   48GB dev box" — it's consumer-hardware accessibility (8-24GB cards is the
   realistic install base). Under that lens, RedHatAI FP8-dynamic/NVFP4 and
   nvidia NVFP4 are ruled out on accessibility grounds (Hopper/Blackwell-only),
   not just "doesn't fit here." Recorded on #4.
4. **GGUF backend parked as issue #15** (`enhancement`/`auto:draft`/`pri:next`)
   — scoped as a design/ADR question, not a code task: llama.cpp has no
   equivalent to the diffusers scheduler's per-step commit-mask output that
   P3's live-view/trace machinery depends on (ADR-CDG-004's drive seam), so
   GGUF would be a second drive path with an instrumentation gap, not a
   drop-in loader option.
5. **3090 (Ampere, 24GB, `192.168.137.1`) cross-machine testing parked as
   issue #16** (`enhancement`/`user:gate`/`pri:later`) — not currently
   reachable from this seat (`ssh 192.168.137.1` timed out, no SSH config
   found). Would validate native bf16 tensor-core throughput (this dev box is
   Turing/sm_75, no native bf16) and any Ampere-gated quant candidates from
   #4's survey (NOT the Hopper-only ones, those stay out of reach regardless).
6. **ADR audit, two passes.** First pass (Haiku, cross-referenced against
   `plan.md`): flagged CDG-001 and CDG-003 status fields as stale
   (`accepted (implementation pending)` when implementation was actually
   complete since P1-P3) — both corrected to plain `accepted`. CDG-003 also
   gained a "Negative Consequences" note cross-referencing its
   `loose-ends.md` observed-violation entry (the ComfyUI-loader `dgemma`
   import bug). Second pass (Haiku, code-grounded — ignored `plan.md` as
   evidence, inspected `nodes/`/`dgemma/`/`tests/` directly): confirmed the
   corrected statuses hold; surfaced that ADR-CDG-001's "5 socket types" are
   3/5 wired in code (`ENTROPY_SCHEDULE`/`CONSTRAINTS` are P4/P5 deliverables
   by design, not a doc gap) — no further edits warranted.
7. **New design: ADR-CDG-006** (`decisions/adr-cdg-006-advanced-sampler-step-window-resume.md`,
   status `proposed`, indexed in `decisions/README.md`). `DGemmaSamplerAdvanced`
   — additive alongside `DGemmaSampler` (the `KSampler`/`KSamplerAdvanced`
   idiom). Key decisions: discrete `start_at_step`/`end_at_step` (not
   continuous `t`, avoids a 3-way name collision); `canvas_state` is **both an
   optional input and an output of the same `DGEMMA_CANVAS_STATE` type** —
   instance-to-instance chaining, per explicit operator requirement; state
   fidelity is snapshot-based in-memory handoff first (disk persistence
   deferred to Phase C, but serialization-ready by construction); realizes
   ADR-CDG-005's deferred resumable contract for `EntropyBoundScheduler`,
   single-block only. Found a real landmine: the anneal temperature is a
   function of step-index *and* total step count, so naively shrinking
   `num_inference_steps` to "stop early" silently reruns a different, hotter
   trajectory — closed structurally (carry `num_inference_steps` in the
   resume state, raise on mismatch), not by convention. Phased A (engine,
   headless, the correctness keystone) → B (node) → C (persistence,
   deferred) → D (multi-block + BlockRefinement, deferred). **Design only —
   nothing built.** The ADR's own closing note: Phase A+B is more than one PR;
   a `decompose-problem` pass is warranted before dispatching build work.

## Environment standing state (not derivable from the repo)

- **`:8189` ComfyUI is STOPPED** (deliberately, to free GPU for the quant smoke
  test) — relaunch recipe: `cd /srv/dev/ComfyUI && .venv/bin/python main.py
  --port 8189 --listen 0.0.0.0 --verbose DEBUG --database-url
  sqlite:////tmp/comfyui-8189.db` (background it). `:8188` remains the
  operator's, untouched throughout.
- **GPU is free** (~46.8GB) as of this handoff — nothing resident.
- **`compressed_tensors==0.17.1`** now installed in `/srv/dev/ComfyUI/.venv`
  (harmless, unused after the AWQ dead-end — not worth uninstalling).
- **RTX-3090 box (`192.168.137.1`) unreachable** from this seat this session
  — no SSH config, connection timed out. See #16.

## Pending decisions (operator)

1. **Merge `p3-instrumentation` → `main`?** Available anytime (#12/#13 both
   verified), auto-publishes 0.1.0. Deliberately not taken this session —
   next trigger is purely the operator's own call, not a re-check of #12.
2. **ADR-CDG-006 Phase A/B build** — ripe for a `decompose-problem` pass per
   the ADR's own closing note before dispatching implementation.
3. **Quant path (#4)** — parked `pri:later`; no candidate currently viable on
   this hardware/software stack. Self-quantize vs. pivot-to-GGUF still
   undecided, intentionally.
4. **GGUF (#15) and 3090 testing (#16)** — both parked, no action pending.
