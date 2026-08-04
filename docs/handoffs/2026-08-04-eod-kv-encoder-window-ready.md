# 2026-08-04 EOD — KV-encoder prep batch merged; window is push-button; this context is gone

Written assuming the authoring context will not be returned to. A cold session
resumes from this file + the spine: run ledger **#225**, bracket thread **#62**,
engine thread **#131**. Nothing in any chat window is load-bearing.

## State at close

`main` @ `aa0c963`, clean, in sync with origin. Only the primary checkout
remains — no stray worktrees, no stashes carrying unbanked work. Card holds
only the resident embedding server; chat model on `:8081` still swapped out
(operator restores when wanted).

## What merged today (evening prep batch — all Opus-reviewed)

| PR | merge | what it gives you |
|---|---|---|
| #235 | `47f2a2e` | `encode_sequence` door hardened: typed ValueError on empty `token_ids` (closes **#227**); `torch.OutOfMemoryError` re-raised typed with lane naming + `mem_get_info` readback (#226 hardening slice — root cause still open) |
| #234 | `aa0c963` | **Standing Encode E2E scenario**: `test_encode_live` (stable now and after Phase 4) + `test_kv_door_contract` (strict-xfail asserting today's fail-loud door; **the Phase-4 drive-body PR must flip this marker** — same convention as S3/#9). Plus `encode_sequence` docstring provenance line (#228 part 1, one instance) |
| #233 | `32bbea0` | `tools/q2_preflight.py --label <run-label>` — push-button window preconditions + **env-provenance banking** (torch/CUDA/driver versions — the thing gate run 4 never recorded) |

Morning/afternoon context (already in prior handoff
`2026-08-04-live-window-blocked-amended-route-cdg021.md`): Q-2 typed BLOCKED
on the bare lane (#226 regression, bisect `a68e29d..33551d5`), re-run route
ratified as **Amendment 1 on #62** (ComfyUI API lane), ADR-CDG-020 HOLD,
ADR-CDG-021 proposed (grown from an operator *observation* — provenance
corrected, see #229).

## The next session's likely play (KV-encoder to ACTIVE)

1. Operator frees the card (~1 hr). Run `python tools/q2_preflight.py --label q2-rerun-<date>` — all green or it tells you what's missing.
2. Execute Amendment 1's §3 runsheet (banked in full on #62, 2026-08-04
   comments): §2 sweep against a main-HEAD server on port 8199 → relaunch the
   server from a worktree of `scratch/q2-skeleton-2026-08-04` (@ `d67e62f`,
   on origin) → §1 Q-2 smoke, seeds {7,13,21}. Typed outcomes are
   pre-registered; do not improvise predicates.
3. **On Q-2 PASS:** drive body implements against ADR-CDG-012 §D.1 (IN-2
   skip-first-encode; OUT-1 stop-at-block per `denoise.py` docstring deferral;
   preserve the OUT-3 stamp at `loop.py` — it is LIVE code, see #62's
   2026-08-04 corrections). The implementing PR flips `test_kv_door_contract`'s
   strict-xfail. That merge = **encoder functionality active**; #47's
   cache-perturbation experiments unlock.
4. **On Q-2 FAIL:** the finding banks per protocol; drive body stays gated —
   that is the gate working, not a setback.
5. Optional same-window rider: build llama.cpp PR #24427 legs → feeds
   ADR-CDG-020's HOLD (adjudication on #131).

## Open ledger (owner / trigger)

| item | owner | trigger |
|---|---|---|
| GPU window + Q-2 re-run | operator schedules; any session drives | operator go |
| Drive body (Phase 4) | next session, auto lane per repo conventions | Q-2 PASS |
| #226 root cause (offload-hook differential; bisect window named) | untriaged auto:fix candidate | backlog pass |
| #228 generalized convention (function-scoped live-proof banking) | open | next doctrine pass |
| ADR-CDG-020 `proposed`, ratification HOLD | gate re-adjudicates | #24427 legs run |
| ADR-CDG-021 `proposed` (per-surface VRAM tenancy) | operator disposition | whenever |
| CDG-019 reciprocal note + #138 cross-link | apparatus | only on CDG-021 acceptance |
| Branch-protection vs direct-docs-push question | operator | flagged on #225 run report |
| Chat model restore `:8081` | operator | wanted again |
| Subagent mid-thought truncations (4× today, all recovered by nudge) | harness-tools#225 | HT backlog |

## Residue accounting (deliberate, none at risk)

- `scratch/q2-skeleton-2026-08-04` (local + origin, @ `d67e62f`): KEEP — the
  §1 arm consumes it; unvalidated by design.
- Remote branches `fix/227-226-kv-cache-door` + `tools/q2-window-preflight`
  and two stashes labeled "stale-ref artifact… banked in merged PR": deleted/
  dropped at close (content verified byte-identical to merged history first).
  If you are reading this and they still exist, the close's final sweep was
  interrupted — safe to delete, verification is recorded on #225.
