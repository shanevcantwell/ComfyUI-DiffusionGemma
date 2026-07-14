# Changelog

This changelog starts at 0.3.0 — no retroactive entries for earlier releases.

All notable user-facing changes to this project are documented here.

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
