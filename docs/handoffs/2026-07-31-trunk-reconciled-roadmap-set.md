# Handoff — 2026-07-31 (late-night session): trunk reconciled, roadmap recorded

**State**: `main` = single clean trunk (reconciliation PR #192, Opus gate PASS; then #197 post-mortem, #198 ADR renumber, #200 roadmap), version-true at **0.4.1**, suite 990 passed / 0 failed. The 0.5.0-autoround release attempt is dead and post-mortemed (`docs/postmortems/2026-07-31-0.5.0.md`); the name 0.5.0 is reused for the refactor (nothing was ever minted under it).

## Next session opens on two operator decisions (both fully briefed on their issues)
1. **Ratify #129** — loop.py decomposition plan + ARCHITECTURE-fate amendment + three sub-decisions (facade lifetime, excision.py cluster membership, Phase-0 oracle scope). Ratification opens the 0.5.0 refactor milestone.
2. **Encode/Denoise visibility call** — hide the pair vs fail-loud on the inert `kv_cache` door (context in the session record around #187/#62 Phase 4). Closes 0.4.2's scope.

## Then: run the 0.4.2 batch
Milestone "0.4.2 — stabilization": #191 (guard reports measured condition — operator principle: never a menu of causes), #187, #188, #169, #151, #161, #175 (draft tier). All tier-labeled; `shanes-autonomous-run` preconditions met once the visibility call lands.

## Parked decisions (no action till operator rules)
- #193 README SEO rewrite (retained diff, user:gate)
- #196 version/tag timing (0.4.2 is its first application)
- harness-tools#229 + #201 here: CLAUDE.md tracking policy fork (track-public vs deploy-from-private)
- harness-tools#227: handoff-landing path (this handoff landed via PR — the no-bypass shape, working)
- Branch cull: `pre-0.5.0-release`, `fix/183-*`, `fix/189-*`, `fix/135-*`, `docs/postmortem-0.5.0` remotes + ~37 local branches (#114) + stash@{0} — all redundant ancestry now; awaiting operator's cut
- `comfyui_detail.log` untracked at repo root — operator shakeout artifact, never dispositioned

## Standing constraints
- RTX-8000 tenancy: INT4 whole-fit (30 GiB floor) and the llauncher chat model (~35 GiB) cannot coexist — every live bracket schedules its llauncher swap explicitly.
- plan.md is TOMBSTONED (trained-name attractor, operating-doctrine#44); direction lives in ROADMAP.md, evidence in ARCHITECTURE.md, tracking on the dashboard. Do not regrow it.
- Doctrine minted this session: operating-doctrine#43 (warm-subagent routing), #44 (trained names); harness-tools#227, #229.

**Resumability**: this file + the issue dashboard + ROADMAP.md are the full state; no session window is load-bearing.
