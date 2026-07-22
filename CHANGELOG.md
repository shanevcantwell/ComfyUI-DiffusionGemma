# Changelog

This changelog starts at 0.3.0 — no retroactive entries for earlier releases.

All notable user-facing changes to this project are documented here.

## [Unreleased]

## [0.4.0] — 2026-07-21
### Added
- Declarative doors end-to-end: constraints/pins, control-signal walker, capture tiers 0–2 + full DISTRIBUTION.
- **Experimental:** `KV_CACHE` surfaces (`DGemmaEncode`/`DGemmaDenoise` + `DGEMMA_KV_CACHE` socket). Encode/mint/advance are functional; the decoder does not yet drive generation off an injected cache (ADR-CDG-012 Phase 4 pending) — `DGemmaDenoise` is limited accordingly.
- Control-signal walker: per-step ramps of scheduler knobs (e.g. `entropy_bound`) via declarative `control_signals`, with effective-knob telemetry riding each frame.
- MCP `generate` widened: constraints/control_signals/capture; `run_id` cancel.
- `DGemmaTokenTrace` (closes #11).
- `DGemmaTrace` entropy mode.
- Sampler terms-and-units mint (widget tooltips + MCP schema descriptions + contract docstrings).
- β-rebuild composite slot (internal groundwork for 0.5.0 remelt kernels).
- ARCHITECTURE data-boundary crossing discipline.

### Changed
- ROADMAP Track B status column + runnable-today notes.
- concept.md seam inventory reconciled.
- test-coverage-plan.md rewritten (100%/100% on run-landed files).

### Fixed
- #124 — `debug_log_path` as a directory now appends `{filename_prefix}.jsonl` instead of writing to the directory path itself (which created a file destroying the directory).
- Stale live-seam sampler arity test.
- Stale walker docstring.

### Known issues
- #36 — ComfyUI's node cache is not invalidated by an `entropy_bound`-only change inside a For-loop (e.g. ComfyUI-Easy-Use): knob sweeps can silently serve stale results. Workaround: wire the swept value as a **linked input** (e.g. loop `index` → math/map node → the converted input), not a widget you edit — widget literals never vary in the executor's cache-signature view. Or drive sweeps through the MCP surface, which bypasses the node cache entirely. Fix scheduled next release.
- #110 (`t_min==t_max` ingress vs ADR-CDG-011 clause; strict-xfail pre-registered).
- #38 — mid-run cancellation may not reach the sampling loop (most visible with thinking=true).
- #9 — thinking=true can consume the whole canvas (empty STRING with converged=True).

## [0.3.1] — 2026-07-14
### Documentation
- Measurement-validity callout: per-step telemetry (`committed_fraction`, the commit
  heatmap, `DGemmaTrace`) measures *commit dynamics* (when a position freezes), not
  *provenance* (whether the frozen token was diffusion-computed from in-canvas evidence
  or emitted one-shot from the model's memorized autoregressive prior). The two can
  diverge under default usage, evidenced by the
  [2026-07-14 gatsby-counts experiment](https://github.com/shanevcantwell/design-docs/blob/main/experiments/2026-07-14-dg-gatsby-counts-ar-prior-latch/README.md)
  (0/14 numeral revisions; counts frozen against evidence the canvas never contained).
  Added to the README as "What the telemetry does and doesn't show." No code changes
  this release. See [issue #78](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/78).

## [0.3.0] — 2026-07-13
### Fixed
- Identical seeds/knobs now reliably reproduce identical output across runs on one loaded model: forward hooks are torn down after every run, and the scheduler/run-state is rebuilt fresh per call (was: a leaked hook or cached scheduler config could contaminate the next run).
- Loud, actionable error at load time on an incompatible `diffusers` install (version floor + structural probe), instead of silently wrong temperature reporting mid-run.
- `DGemmaTrace` summary now labels `committed_fraction` as block-local (it resets near zero at canvas/block boundaries — previously read like a whole-canvas re-melt).
- Mask-token corroboration verdict in `DGemmaTrace` is now tri-state: "no evidence" is no longer reported with the same wording as genuine evidence against a mask sentinel.
- Multi-canvas runs: per-frame captions on the frames IMAGE output are keyed per canvas ("canvas k/N · step i/M"), fixing a fragile flat-index zip.

### Internal groundwork (not yet user-visible — no new node inputs/outputs this release)
- Per-step frame telemetry now captured engine-side: per-position predictive entropy (Tier 0), pinned-position mask, effective per-step knob values, raw pre-excision canvas ids. Captured but not yet rendered by any node; exposure is scheduled work.
- Constraint/control-signal/capture ingress validation landed engine-side (validated, not yet driving generation; unreachable from any node input).
- Internal topology reorganized (surfaces/comfyui, surfaces/mcp, consumers) — node names, sockets, and behavior unchanged; an MCP surface over the same core is available as an optional extra for non-ComfyUI use.
