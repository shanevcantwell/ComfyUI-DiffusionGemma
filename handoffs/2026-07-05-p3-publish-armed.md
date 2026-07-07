# Handoff — P3 verified-minus-one-eyeball, publish armed on merge

**Date:** 2026-07-05 (third session of the day, quota cutoff) · **From:** orchestrator
session (P2+P3 waves) · **HEAD at handoff:** `c895b52` on `p3-instrumentation`

Cold-start path: `/orient`. The record is authoritative (README → plan.md →
decisions/ → issues #8–#14 → examples/); this file carries only what it does not.
Supersedes `2026-07-05-gui-bracket-close.md` (its pending items all resolved:
#4 deferred `pri:later`, #1 still open, backlog fully tiered).

## State in one line

P2 merged+published to main; P3 complete on `p3-instrumentation` (6 commits,
verifier PASS, docs flipped, publish wiring included) — **merge is blocked on
exactly one check** (operator eyeball of the #12 round-2 live-view fix) and
**merge = publish**: pyproject 0.1.0 + the Action land on main together, so the
P3 merge IS the registry publication event. Do not merge casually.

## The one open loop (start here)

1. Operator hard-reloads browser → `DGemmaSampler` shows green `live: (idle)`
   as its **bottom widget** (below `thinking`, resize-proof) → queue a run →
   line updates per step. PASS → close #12 + #13 (`verified`), bank the
   operator's saved workflow as the P3 UI twin (P1 precedent), merge with
   `--no-ff`, ring the bell. The merge auto-publishes 0.1.0 to
   registry.comfy.org under publisher `reflectiveattention`.
   FAIL → round 3 rides issue #12 (round-1 post-mortem + bundle grounding are
   in its comments; the widget-interface excerpts are in the `4a264c5` commit
   message trail).

## Environment standing state (not derivable from the repo)

- **ComfyUI on 8189 is UP** with round-2 JS + P3 nodes loaded (relaunch recipe
  in `/tmp/comfyui-8189.log` first line; `/tmp` DB+log gone on reboot).
  **8188 is the operator's — leave it free.** Model resident (~40GB GPU).
- **Registry:** publisher `reflectiveattention` minted (immutable); repo secret
  `REGISTRY_ACCESS_TOKEN` set operator-side; `.github/workflows/publish_action.yml`
  armed (fires on pyproject.toml change on main + manual dispatch).
- **`websocket-client` was pip-installed into `/srv/dev/ComfyUI/.venv`** (ws
  probe for the P3 verify; benign, operator may remove).
- **`~/.claude/keybindings.json` is NEW this session:** Esc no longer cancels
  a Claude Code run; cancel = `ctrl+k ctrl+k ctrl+k`. Takes effect on harness
  restart — next session, run `/doctor` to validate. Ctrl-C remains hardcoded;
  the wrapper/tmux capture designs are banked as keepalive_claude#1.
- **Subagent narration-stop bug** (harness-tools#132, 2 specimens): long
  multi-step dispatches may "complete" on a forward-looking narration line.
  Mitigation that worked twice: SendMessage resume + "your final message must
  be the full report, not a progress note." Check returned artifacts exist
  before dispatching dependents.

## Pending decisions (operator)

1. **#12 eyeball** — the merge/publish gate (above).
2. **#1** (`user:gate`, mask_token) — close recommendation posted since
   morning; P3's corroboration line ("no fixed sentinel") is further evidence.
   One click.
3. **UI twin** — operator saves their laid-out workflow; bank to `examples/`.
4. Optional: WGE notes (Gemini 2.5 era) → `design-docs/incoming-ideas/` if
   they exist and the operator wants them fished.

## Session-learned cautions (cheap to re-derive wrongly)

- **Hard-refresh the browser after every 8189 restart** — stale `/object_info`
  in the tab orphans links on API-graph import (observed; cost a false bug report).
- **ComfyUI layout hands surplus node height to `computeLayoutSize` widgets
  only** (the multiline prompt textarea eats it). Space for custom rendering
  must be a layout-participating widget (`computeSize`), never foreground
  paint — #12 round 1 vs round 2, bundle-grounded in the issue.
- The `[tool.comfy]` publish stub existed in pyproject since P1 — check for
  banked sockets before scaffolding new ones.
- Front-loaded avalanche (~40% commit by step 2) observed on BOTH live runs
  today (seeds 7, 23; different prompts) — pattern, not fluke; #7 material.
- The experiment stack ordering: #14 (entropy view) unblocks #3-signal-1 and
  the renoise injection experiment; #11 (token identity) unblocks #9's EOS
  question and #3-signal-2. Instrumentation before experiments.
