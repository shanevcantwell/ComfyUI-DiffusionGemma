# 2026-08-04 — live window BLOCKED honestly; re-run route ratified; two lifecycle ADRs banked

State at close: `main` clean and in sync (tip `e4bdcae`). Card holds only the
resident embedding server (~1.6 GiB); chat model on `:8081` still swapped out —
restore is the operator's, when wanted.

## What happened (pointers, not restatement)

- **Run #225** (autonomous, operator-cleared): Q-2 smoke + liveness sweep both
  typed **BLOCKED** — `encode_sequence` OOM on the bare transformers lane,
  upstream of everything under test. Not a FAIL data point, per the protocol's
  own clause. Raw evidence committed under `docs/experiments/2026-08-04-*/`.
- **#226** — the OOM, reframed by provenance audit as a **regression**
  (S-B encode PASS at `a68e29d` 2026-07-30 vs deterministic OOM at `33551d5`;
  bisect window named; torch/CUDA unpinned+unbanked caveat). **#227** —
  `encode_sequence("")` IndexError ingress gap. **#228** — the process gap
  that hid the precedent (gate results never banked function-scoped; no
  standing Encode E2E scenario).
- **Amendment 1 (ratified, banked on #62)** — Q-2 re-runs unmodified through
  the ComfyUI API lane. First submission FAILED its re-gate on a falsified
  precedent claim; re-anchored (encode leg: S-B stands; decode leg: no
  precedent anywhere — that is the thing under test) and re-gated PASS.
  Skeleton: `scratch/q2-skeleton-2026-08-04` @ `d67e62f`.
- **rung-1 (#131)** — #24423 leg measured live (55.5–74.6 tok/s @ Q8_0,
  byte-identical seed rerun; zero distribution visibility in
  `--diffusion-visual`). #24427 never built on this host → **ADR-CDG-020
  ratification: HOLD** (adjudication on #131).
- **ADR-CDG-021** (proposed) — per-surface VRAM tenancy. Grown from an
  operator **observation** (not a directive — provenance corrected in-file
  and on #229 after an apparatus misread; correction PR #231). Explains the
  S-B-pass-vs-bare-OOM split. Ordinary proposal; nothing binding propagated.
- Morning, pre-bracket: legacy `handoffs/` dir healed (session records →
  `docs/handoffs/`, operator evidence → `docs/evidence/2026-07-early-runs-compendium/`),
  `comfyui_detail.log` retired.

## Open, with owners/triggers

| item | owner | trigger |
|---|---|---|
| Reopen GPU window, run Amendment-1 runsheet (#62) | operator schedules; apparatus drives | operator says go |
| #226 bare-lane fix (fail-loud hardening + bisect) | apparatus (auto:fix-proposed, untriaged) | next backlog pass |
| CDG-020 HOLD → build #24427 legs | next GPU window | rides the same window |
| CDG-021 disposition | operator | whenever |
| CDG-019 reciprocal note + #138 cross-link | apparatus | only on CDG-021 acceptance |
| Branch-protection-vs-direct-docs-push question | operator | flagged on run #225 report |
| Chat model restore `:8081` | operator | wanted again |

Continuity spine: ledger **#225**, bracket thread **#62**, engine thread
**#131**. A cold session orients from those three plus this file.
