# Handoff — quant dead-end grounded, ADR-CDG-006 proposed, merge held open on operator signal

**Date:** 2026-07-05 (fifth session of the day) · **From:** orchestrator session
(quant recon + ADR audit + ADR-CDG-006 design + doc cleanup) · **HEAD at
handoff:** `b73e15e` on `p3-instrumentation`

Cold-start path: `/orient`. The record is authoritative (`README.md` →
`plan.md` → `decisions/` — no index file, `ls` the directory or check each
ADR's own header — → issues #4/#12/#13/#15/#16 → `examples/`); this file
carries only what it does not. Supersedes
`2026-07-05-quant-recon-and-adr-006.md` (itself superseding
`2026-07-05-p3-publish-armed.md`) — nothing in that file was wrong, it's
extended by everything below.

## State in one line

`p3-instrumentation` is 9 commits ahead of `main`, unmerged, on the operator's
own explicit hold (no remaining gate — #12/#13 both passed and closed last
session). This session's quant-accessibility push hit a grounded dead end and
pivoted into a new proposed design (`ADR-CDG-006`, not built); documentation
hygiene closed out (`decisions/README.md` removed as a drift-prone duplicate
index, `CLAUDE.md`/`README.md` updated to match).

## The one open loop (start here)

Unlike last handoff, there is no single blocking gate — name that honestly
rather than force one. The highest-leverage next step if resuming cold:
**`ADR-CDG-006`** (`decisions/adr-cdg-006-advanced-sampler-step-window-resume.md`,
status `proposed`) designs `DGemmaSamplerAdvanced` — a step-windowed sampler
whose `canvas_state` socket is both input and output of the same type
(instance-to-instance chaining), realizing `ADR-CDG-005`'s deferred resumable
contract for `EntropyBoundScheduler`, single-block. **Nothing is built.** The
ADR's own closing note: Phase A (engine, headless correctness keystone) +
Phase B (node adapter) together are more than one PR — a `decompose-problem`
pass is warranted before dispatching implementation. That pass has not been
run. Merging `p3-instrumentation` → `main` remains available anytime
(auto-publishes 0.1.0) but has no forcing trigger; that's the operator's own
call to make, not a to-do waiting on anything.

## Environment standing state (not derivable from the repo)

- **`:8189` ComfyUI is STOPPED** (deliberately, to free GPU for a quant smoke
  test this session) — relaunch: `cd /srv/dev/ComfyUI && .venv/bin/python
  main.py --port 8189 --listen 0.0.0.0 --verbose DEBUG --database-url
  sqlite:////tmp/comfyui-8189.db` (background it). `:8188` remains the
  operator's, untouched all session.
- **GPU is free** (~46.8GB, nothing resident) as of this handoff.
- **`compressed_tensors==0.17.1`** is installed in `/srv/dev/ComfyUI/.venv` —
  harmless residue from the AWQ-INT4 smoke test, unused after the dead-end
  finding, not worth uninstalling.
- **RTX-3090 box (`192.168.137.1`) is unreachable from this seat** — no SSH
  config, connection timed out when tried this session. Tracked as issue #16.
- **Commit-per-artifact cadence adopted this session** (operator feedback,
  contrasting with a "Fable" workflow that commits/pushes after every action).
  Going forward: land + push each durable artifact (an ADR edit, a new
  decision record, a handoff) immediately, not batched to a session-end
  sweep. This session's own tail end followed it (three separate commits:
  ADR fixes + CDG-006, this handoff's predecessor, then the README/decisions-
  index cleanup) after starting the session batched.

## Pending decisions (operator)

1. **Merge `p3-instrumentation` → `main`?** Available anytime, auto-publishes
   0.1.0 to registry.comfy.org. Deliberately not taken across two sessions
   now — worth naming that the trigger has *shifted* (was "#12 eyeball,"
   which fired and passed; now it's purely "whenever you decide"), not that
   it's stuck.
2. **`ADR-CDG-006` Phase A/B build** — needs a `decompose-problem` pass
   before implementation dispatch, per the ADR's own scope note. Not started.
3. **Quant path (issue #4)** — dead end grounded this session (AWQ-INT4/
   compressed-tensors checkpoint incompatible with pinned `transformers`,
   architecture-revision mismatch, not fixable by config). Parked
   `pri:later`. Self-quantize vs. pivot-to-GGUF still genuinely undecided.
4. **GGUF (#15) and 3090 testing (#16)** — both parked (`enhancement`,
   appropriately tiered), no action pending on either.
5. **3090 access** — manual operator-driven testing vs. setting up SSH from
   this seat: undecided, blocks nothing yet since #16 is `pri:later`.

## Session-learned cautions (cheap to re-derive wrongly)

- **ADR handles and GitHub issue numbers are two independent sequences that
  collide numerically.** `ADR-CDG-005` exists (`decisions/`); GitHub issue
  `#5` does not (never created). Don't conflate them when cross-referencing —
  this exact confusion happened live this session.
- **`decisions/README.md` is gone by design, not oversight.** A future
  session (or agent) should not reflexively recreate it — it was a
  hand-maintained duplicate of each ADR's own Status/Title/Date header and
  had already drifted out of sync twice before removal. Check an ADR's own
  header, or `ls decisions/`.
- **The AWQ-INT4 checkpoint
  (`cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4`) is a confirmed dead end**
  on this pack's pinned `transformers==5.13.0` — two stacked failures (a
  transformers-side `param_element_size` gap, and a real architecture-
  revision key mismatch, `model.decoder.layers.*` vs.
  `model.encoder.language_model.layers.*`). Don't re-attempt without either a
  `transformers` version bump or a checkpoint requantized against the
  current modeling-code revision. Full grounding: issue #4's comment thread.
- **The quant-accessibility target is consumer 8–24GB cards** — explicitly
  *not* this 48GB dev box, and explicitly *not* H100/Hopper-class hardware
  (which rules out the RedHatAI FP8/NVFP4 candidates on principle, not just
  on fit). Keep that framing when evaluating any future quant candidate;
  it's an operator-stated reframe, not derivable from the code.
