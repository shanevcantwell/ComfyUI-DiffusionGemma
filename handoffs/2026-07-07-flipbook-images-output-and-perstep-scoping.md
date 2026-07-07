# Session handoff — 2026-07-07: flipbook `images` output + per-step input-scheduling scoping

## What shipped (branch `feat/21-flipbook-image-collection`, unpushed / unmerged)
- **Flipbook as a sampler output.** `DGemmaSampler` gained a 5th output **`images`** (IMAGE, a single `(N,H,W,3)` batch tensor — NOT `OUTPUT_IS_LIST`), rendered per captured step from the same decoded strings as the `frames` STRING list. Feeds PreviewImage's batch scrubber, `SaveAnimatedWEBP`, VHS. Reworked from an initial standalone `DGemmaFlipbook` node → sampler output (operator's simpler intent); the `CanvasTrace.processor` change from the first cut was reverted clean (`dgemma/types.py` + `loop.py` byte-identical to `p3-instrumentation`).
- Commits: `7563463` (initial standalone node) → `4b1ce33` (rework to sampler output) → `96802be` (rename label `frames_image`→`images`). 129 tests green, ruff clean.
- **Live + verified.** ComfyUI on `192.168.137.2:8189`, `feat/21` live via a review worktree at `/srv/dev/cdg-feat21-review` symlinked into `custom_nodes`. Operator confirmed the PreviewImage scrubber works on the batch output.

## Issues filed this session
- **#20** (bug) — `anneal_temperature` reconstructs per-step temperature; desyncs under corrector schedulers. **Now subsumed by #23.**
- **#21** (enh) — the flipbook feature (implemented on `feat/21`).
- **#22** (enh) — deep-review findings: `thinking` widget experimental, VRAM status badge, `converged`-reads-False-on-adaptive-stop, `corroborate_no_mask_token` tri-state; + docs posture (no knob-tuning guides).
- **#23** (enh, post-0.1.0) — per-step `ENTROPY_SCHEDULE` input seam + honest per-step telemetry. **Ends with a DISCONNECT FLAG — talk before building.**

## Key decision
Per-step input scheduling is **POST-0.1.0**. Per-**run** input manipulation (fixed seed + swept scalars via widget→input + any value node, e.g. Comfyroll's value scheduler) is **already live in 0.1.0** — that IS the operator's "manipulate inputs, fixed seed, see what happens" value, delivered, no code. Adding per-**step** later is additive-safe (ComfyUI validates inputs by name — `execution.py:884-901`; `KSampler`/`SamplerCustomAdvanced` coexistence precedent). No pre-0.1.0 code needed. Full analysis: `~/.claude/plans/zany-whistling-flute.md`.

## ⚠️ The disconnect (banked, unresolved — do not resolve by assumption)
Across the session the assistant repeatedly under-modeled ComfyUI's modularity / input semantics (per-run vs per-step scheduling; widget→input conversion; the Comfyroll value scheduler; per-step intervention seams). The operator flagged a conceptual disconnect at close and chose "not worth it rn." **Do not begin #23 by assuming the current framing is complete — have the conversation first.** (Recorded verbatim in #23's closing note.)

## Pending / banked but NOT filed (deferred to keep the close lean)
- **Experiment ideas:** semantic-renoise swap (von Rütte); cross-lingual diffusion probe; reheating / modulated-entropy schedule; per-step-hook seam (expose the diffusers `callback_on_step_end` for external read + write-back); `SIGMAS`→`ENTROPY_SCHEDULE` translation node.
- **Manifesto idea:** crystallization / 18-bit melt (vocab_size **verified 262144 = 2^18**) + "sampler-as-semantic-probe" thesis. Draft offered, not written.
- **Docs:** "Scheduling (0.1.0)" note (per-run available via widget→input; per-step is P4) + a fixed-seed sweep example workflow.

## Open operator decisions
- **Merge `feat/21`?** Unpushed / unmerged — review-first call. **On merge:** repoint `custom_nodes/ComfyUI-DiffusionGemma` symlink back to the main repo (currently → `/srv/dev/cdg-feat21-review`) and `git worktree remove` that review worktree.
- **p3→main 0.1.0 publish** — still operator-gated (per the prior handoff).

## Working-ground notes
- Untracked on `p3-instrumentation`: three operator screenshots in `handoffs/` (`2026-07-07-00-06/00-37/01-31.png`) — operator-dropped, provenance clear.
- `feat/21` not on origin. Infra: host ComfyUI freshly rebuilt this session (Manager/Crystools/Easy-Use + DiffusionGemma symlink); user-workflow backup at `/srv/dev/comfyui-user-backup-20260706-224848/`.
