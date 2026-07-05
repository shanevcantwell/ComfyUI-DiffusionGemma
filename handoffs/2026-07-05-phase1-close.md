# Handoff — Phase 1 closed, load-path bracket open

**Date:** 2026-07-05 (UTC) · **From:** orchestrator session (P1 build) · **HEAD at handoff:** `4dde268`

Cold-start path: run `/orient` against this repo. This file carries only what the
standing record does not; the record itself (README → plan.md → decisions/ →
loose-ends.md → issues) is authoritative.

## State in one line

Phase 1 is **closed with a clean completion ring** (`4dde268` banks the evidence in
plan.md): real-weights integration PASS + headless-ComfyUI graph PASS
(`DGemmaLoader → DGemmaSampler → PreviewAny`, 51.65s, early-stop 13/48 steps,
validity readout honest). Phase 2 (knobs) is next per plan.md, but an
**INT4 load test precedes it** if the operator confirms (see "Pending decision").

## Environment standing state (not derivable from the repo)

- **ComfyUI:** `/srv/dev/ComfyUI` (commit `985fb9d6`, own `.venv`, torch 2.12.1+cu130).
  The pack is **symlinked** into `custom_nodes/` and its deps are installed in that venv
  (transformers 5.13.0, diffusers 0.39.0, accelerate 1.14.0, bitsandbytes 0.49.2).
- **Repo engine venv:** `.venv/` (gitignored, 5.2G), same torch build. Deliberately
  separate from ComfyUI's venv AND from user shane's `~/.local` global stack
  (torch 2.11 + training tools) — three environments, isolation is the point.
  This seat runs as user `claude`; shane's `pip --user` packages are invisible to it.
- **Weights cached (HF cache, user claude):** `google/diffusiongemma-26B-A4B-it`
  (bf16, 49GiB) and `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` (17.3GB —
  **downloaded, never loaded**; see issue #4). Loads hit HF for revalidation even
  warm — set `HF_HUB_OFFLINE=1` for airgapped runs.
- **GPU:** Quadro RTX 8000 48GB, sm_75 (no bf16 tensor cores — bf16 works via upcast;
  Gemma-family fp16 overflow risk is why we run bf16). Desktop stack holds ~1.3GB
  baseline; that is the floor, nothing to evict.
- **Known-good load recipe (P1):** `quant="none"`, bf16, `device_map="auto"` — CPU
  spill is normal and logged by accelerate; ~21-25s warm load, ~2.0-2.3s/step.
  bnb NF4 **cannot work** for this model (fused 3D MoE experts; loose-ends + issue #4).

## Pending decision (operator, from the P1 completion ring)

1. **Issue #1** (`user:gate`, mask_token) — evidence + close recommendation posted;
   operator has not yet closed.
2. **Next bracket** — recommended: INT4 checkpoint load test (the `run_compressed`
   -vs-decompress-at-load question on issue #4) before P2 code. Operator had not
   answered at handoff time.

## Open threads with no file home (conversation → record pointers)

- **Issue #3** — "mot juste" gap-fill workflow (three necessity signals). Banked.
- **Issue #4** — load-path survey + the config-readback correction (the "AWQ" repo is
  actually compressed-tensors W4A16; experts ARE quantized; no custom code). Banked.
- **Adversarial-renoise experiment** (structured-noise proposal distribution vs the
  entropy-bound commit rule — "does plausible noise fool a confidence-based commit
  test") — discussed and operator-endorsed in conversation, **filed as an issue at
  handoff time** (see issue list; it is the newest one if present, else file it —
  content sketch: swap renoise source from U(V) to a plausible-token proposal,
  measure false-commit rate vs proposal plausibility on the P1 frame instrumentation).
- **P2 first target with runtime evidence:** the leaked `thought\n` reasoning preamble
  on the STRING payload (both real runs) — plan.md P2's thinking toggle.

## Session-learned cautions (cheap to re-derive wrongly)

- ComfyUI's loader puts only `custom_nodes/` on sys.path — the pack's dual-context
  import gates exist for this; `tests/test_comfyui_loader_context.py` is the
  enforcement surface. Don't "simplify" the gates.
- `committed_fraction` semantics are scheduler-relative (ADR-CDG-001 addendum);
  `converged` is a per-step reading under EntropyBoundScheduler, deliberately narrow.
- The operator prefers waterfall closes, readback over reassurance, and is
  NLP-trained — validation-flavored prose registers as noise. Lead with evidence.
