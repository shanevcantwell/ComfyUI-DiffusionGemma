# Changelog

This changelog starts at 0.3.0 — no retroactive entries for earlier releases.

All notable user-facing changes to this project are documented here.

## [0.5.2] - 2026-08-05

The KV channel is the headline: the decoder now genuinely drives generation
off an injected cache, composed with a templated prompt turn, with the
composition's remaining edges interim-guarded rather than silently wrong.
Operator's version rationale: "0.5.0 was refactor and this is activating
latent functionality."

### Added
- #62 / ADR-CDG-012 Phase 4 (PR #242, merge `ac3c832`) — **KV-cache
  decoder-drive body**: `run_diffusion(kv_cache=...)` now drives the decoder
  off an injected cache (IN-2 skip-first-encode, full multi-block loop to
  completion/EOS). The encode→denoise KV path is **active** — 0.4.0's "the
  decoder does not yet drive generation off an injected cache" limitation is
  lifted. Gated on the Q-2 real-weights smoke: PASS, 3/3 seeds (run
  2026-08-04b, ledger #240). `test_kv_door_contract` flipped from
  strict-xfail to a success contract.
- ADR-CDG-024 (accepted 2026-08-05) / #257 (PR #262, merge `0b6bd1a`) —
  **prompt-under-injection composition**: the templated denoiser turn is
  prefilled onto the injected KV cache instead of the prompt being ignored
  under injection (ADR-CDG-012 §D.1 IN-2's open question, now resolved).
  Live-verified at AIO parity and operator-field-confirmed during the PR's
  acceptance session. Supersedes #248's interim exclusivity invariant
  (merged PR #253 at `50ea909`) by name.
- ADR-CDG-025 (accepted 2026-08-05, #103, PR #258, merge `d6b9991`) — MCP
  KV-path parity via a server-side cache-handle registry; composition-
  compatible with ADR-CDG-024's accepted contract.
- #266 (PR #268, merge `ad8a8e7`) — run-log injection-provenance: the
  `dg-runlog/1` header now carries injection-provenance fields, closing the
  filename-only attribution gap on the KV path.
- #228 (PR #234) — standing `DGemmaEncode` E2E scenario in the live battery +
  function-scoped live-proof provenance banking.
- `tools/q2_preflight.py` (PR #233) — push-button Q-2 window precondition check
  with environment-provenance banking (#62, #228).
- `tools/leak_guard.py` + CI leak-guard workflow (PR #239, ADR-CDG-022
  Decision 4b) — generic-pattern topology-leak guard (private-IP ranges,
  host-path shapes; never a literal denylist).

### Fixed
- #263, #265 (PR #270, merge `921f944`) — **interim ingress guards** for
  composed prompt+KV runs: #263 (block>0 splice offset ignores the
  prefilled turn length) and #265 (`prefill_templated_turn` mutates the
  injected `KVCache` in place, leaking composed-turn state across
  invocations sharing the object) are fenced at the door
  (`reject_multi_block_composed_prefill`, ingress check V7) rather than
  fixed — both remain **open**, tracked for retirement: #263 needs a small
  design confirmation, #265 needs an ADR-CDG-024 amendment (stateless-core
  tension). The guard retires when its issue closes.
- #210 (PR #249, merge `1e6b116`) — backend-aware quant checkpoint guard: a
  GPTQ-format checkpoint under `quant="autoround"` no longer passes the
  guard and crashes unhandled in `from_pretrained`.
- #248 (PR #253, merge `50ea909`) — reject a `prompt`+`KV_CACHE` pair at
  ingress (exclusivity invariant); **superseded by ADR-CDG-024** once
  composition landed (see Added).
- #227 (PR #235) — empty-text encode is rejected with a typed `ValueError` at
  the `encode_sequence` door instead of failing opaquely downstream; includes a
  #226 typed-OOM hardening slice (typed re-raise at the same door).
- #222 — publish workflow token-status checks read the secret store directly
  (`679df6a`); publish docs state the manual-dispatch reality while the token
  is absent pending reconstruction (`929770f`).

### Changed
- ADR-CDG-022 publish policy, accepted 2026-08-04 (PRs #238, #239, #244) —
  the public tree carries product + decision record; session residue (handoffs,
  evidence, raw run artifacts, the deferred q2-smoke folder) evacuated to the
  private design-docs annex, with a `docs/session-record.md` stub naming the
  annex location.

### Documentation
- ADR-CDG-023 (accepted, ratified via Opus design-gate PASS on PR #250,
  merge `68458ee`) — mcp SDK 2.x port strategy: hold the `<2.0.0` cap as
  standing posture, with a named revisit trigger (#151 stays open, tracking
  the deferred port).
- ADR-CDG-020 (proposed, PR #224) — GGUF engine sourcing via pinned upstream PR
  branch; ADR-CDG-007 supersession flip. Ratification gated on the #131
  rung-1 probe (GPU-window-gated, un-run).
- ADR-CDG-021 (proposed, PR #230; provenance correction PR #231) — per-surface
  VRAM tenancy ownership; operator disposition open.
- Doc-staleness sweep (PR #246) — ROADMAP.md, ADR-CDG-012 (ratified
  prompt-under-injection write-in + Phase-4 status + resolved smoke OQ),
  ADR-CDG-010 OQ2 cross-ref, README/CHANGELOG/ADR-CDG-009 accuracy annotations
  re-grounded to the post-Phase-4 reality.
- Quant/GGUF ground-truth pass (#269, merge `0d348ea`) — README/AGENTS/CLAUDE
  corrected: bf16 is the only working load path; AutoRound INT4 is
  non-functional end-to-end (#264, tracked inside #211); GGUF is the audience
  path (operator ruling, #131), not a dev backend.
- Enforcement-ledger truth pass (#261, PR #272, merge `ce3c00b`) —
  ARCHITECTURE.md and ADR enforcement tables corrected off stale
  `NOT-YET-IMPLEMENTED` rows for landed ADR-CDG-010/011 Phase-3 work; adds a
  rule-6 GAP row.
- Roadmap revisit (PR #271, merge `f63bad7`) — ROADMAP.md's operator-set
  section rewritten against ground truth at `0d348ea`: composition/KV
  bracket marked closed, remainders recorded with their true shapes, next
  bracket (#259, MCP KV-path parity) named, quant re-sequenced (#264 gates
  #211), release line recorded.
- 2026-08-04 session records: live-window banking (`8649166`), roadmap refresh +
  handoffs (PRs #232, #236), 2026-07 evidence compendium (`f92e558`), legacy
  session records restored (`ae78968`) — subsequently evacuated to the annex
  per ADR-CDG-022.

### Known limitations
- #263, #265 — composed prompt+KV runs beyond the guarded shape are
  rejected at ingress rather than supported; see Fixed above.
- Quant: bf16 (`quant="none"`) is the only working load path. AutoRound
  INT4 (`quant="autoround"`) is non-functional end-to-end — every load
  crashes post-load in `_assert_tie_integrity` on a `QuantLinear` object
  (#264), tracked inside #211's quantized-engine bracket.
- GGUF loading remains parked on #131 (rung-1 probe GPU-window-gated,
  un-run); ADR-CDG-020's ratification and pin choice wait on it.
- **Deployed-venv note:** the run-log's `pack_version` field
  (`consumers/run_log.py`) reads installed distribution metadata via
  `importlib.metadata.version()`, not `pyproject.toml` directly. A deployed
  venv that hasn't re-run `pip install -e .` since this release will keep
  stamping the prior `pack_version` into run logs until the editable
  install is refreshed.

## [0.5.1] - 2026-08-03
0.5.1 — documentation release + field fix. ARCHITECTURE.md re-grounded against the decomposed tree (#220 audit); README/CLAUDE/AGENTS refreshed; fix(#119): offload-aware tied-weights integrity guard (first tagged release carrying it).

## [0.5.0] - 2026-08-03
### Changed
- #129 — `dgemma/loop.py` (1665 lines) decomposed into responsibility modules per ADR-CDG-018: `dgemma/config.py`, `dgemma/compat.py`, `dgemma/capture.py`, `dgemma/excision.py`. `loop.py` is now a 691-line re-export facade holding `DGemmaPipeline`/`run_diffusion`/`_build_result`; every prior `from dgemma.loop import X` path stays valid. Pure structural refactor, no behavior change — verified against a golden-trace oracle (`tests/test_loop_golden_trace.py`) and live tier 4/4 against real bf16 weights.

## [0.4.2] - 2026-08-01
### Fixed
- #191 — GPU memory diagnostics now report measured memory holders instead of hypothetical causes.
- #187 — `encode_sequence`'s minted tensors are pinned to the encoder device.
- #188 — live-view capability is single-minted (PR #206).
- #207 — the `kv_cache` ingress now fails loud on an inert/unwired input instead of silently no-opping (#209). **Behavioral change:** callers relying on the prior silent no-op will now see an explicit error.
- #212 — `DGemmaDenoise` accepts the minted live-view input; added an executor-faithful declared-input/signature parity test, in force for all nodes going forward (#213).
- #151 — pin the `mcp` extra to `<2.0.0` (2.x moved its API under `surfaces/mcp/server.py`).

### Added
- #175 — per-node descriptions and widget tooltips (minimal phase, #208).

## [0.4.1] - 2026-07-29
### Added
- `install.py` — belt-and-braces post-install script (issue #147). ComfyUI's Extensions flow (Manager merged into core) honors it post-requirements: logs interpreter + installed dependency versions via `importlib.metadata`, re-checks each `requirements.txt` pin, and installs anything still missing into the correct interpreter. Diagnostic insurance for a broken fresh install, not a claimed root-cause fix — root cause on the affected box remains unpinned pending operator data (#147).
- README install section split into Extensions/registry (automatic) vs. manual clone (explicit `pip install -r requirements.txt` fallback, Windows-portable and venv forms) paths.

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
