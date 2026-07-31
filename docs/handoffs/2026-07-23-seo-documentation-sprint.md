# Handoff: 2026-07-23 SEO + Documentation Sprint

**Date:** 2026-07-23 (UTC)
**Session goal:** Counter DiffusionGemmaPromptBuilder's SEO capture; restructure README for layered discovery; document MCP surface for agent consumption.

---

## What landed

### `pre-0.5.0-release` branch (pushed, ready to merge)

| File | Change | Commit |
|------|--------|--------|
| `README.md` | Restructured into three audience layers: Quick start (beginner), How it works (power user), Under the hood (researcher). Front-loaded INT4 VRAM breakthrough (~30.7 GB) and MCP toolkit positioning. Added YouTube channel link. Normalized bullet syntax. | `c896b7e` → `40f314d` → `6b999b8` |
| `AGENTS.md` | New file — agent-facing contract for Copilot, Perplexity, etc. Documents MCP topology (core → MCP → consumers), available tools with params, how DiffusionGemma works, protected load-time mechanics for INT4 path. | `a2e14ce` → `102e6be` |

### `main` branch (pushed)

| File | Change | Commit |
|------|--------|--------|
| `decisions/adr-cdg-018-decompose-loop-py.md` | ADR: decompose dgemma/loop.py from 1,631 lines into five responsibility modules. Staged implementation plan (5 stages). | `0de86fd` |

### GitHub metadata (updated via `gh`)

| What | Before | After |
|------|--------|-------|
| Repo description | "Watch text crystallize out of noise..." (poetic, no keywords) | "DiffusionGemma node pack for ComfyUI — discrete diffusion text generation with per-step canvas snapshots, commit heatmaps, and structured trace data. Watch meaning crystallize out of noise." |
| Topics | 11 topics | +3: `gemma4`, `text-generation-ai`, `comfyui-workflow` (now 14) |

### GitHub issues opened

| Issue | Title | Labels |
|-------|-------|--------|
| #129 | Decompose dgemma/loop.py into responsibility modules (ADR-CDG-018) | auto:draft |

---

## What is NOT done — next context targets

### 1. Triaging the issue backlog (30 open issues)

Current state: 3 `auto:fix` (#119, #124, #126), 7 `auto:draft`, 1 `user:gate`. The rest are unlabeled or enhancement-tier.

**What to do:** Run the triage-issues skill against the dashboard. Label by autonomy tier (`auto:fix` / `auto:draft` / `user:gate`) and priority (`pri:now` / `pri:next` / `pri:later`). This creates the precondition for an autonomous run (shanes-autonomous-run skill).

**Why now:** The SEO work is done — the dashboard is calm. A triaged backlog enables clearing bugs end-to-end in the next session without re-reading every issue.

### 2. Update ROADMAP.md against current state

Current `ROADMAP.md` was written before:
- INT4 AutoRound path landed (issue #128 merged)
- MCP surface Phase 2 completed (`surfaces/mcp/`)
- ADR-CDG-018 decomposition plan accepted
- The crystalline proxy framework (ADR-CDG-016, -017)

**What to do:** Read current ROADMAP.md, cross-reference against merged issues and accepted ADRs, update the "what's done" vs "what's next" sections. Add the loop.py decomposition as a near-term engineering seam item.

### 3. Merge `pre-0.5.0-release` to `main`

The branch is pushed and ready. Decision: merge directly or open PR for review first.

**What to do:** Operator decides — merge or PR. After merge, switch working context to `main`.

---

## Open questions from this session

1. **pyproject.toml restructure?** Gemini recommended pulling inline comments above arrays and grouping tool configs logically. Current file is already well-structured (rationale above arrays). Decision deferred — operator said "don't drive."
2. **Video embed in README?** 0.3.0 video exists on YouTube (@reflectiveattention), no 0.4.0 video yet. Embedding the 0.3.0 video under "Quick start" was discussed but not executed. Open for next session.
3. **Landing page (GitHub Pages)?** Gemini's HTML structure + actual video script content could become a standalone landing page for HN/Reddit backlinks. Not started — requires operator direction on URL/domain.

---

## Branch state

| Branch | Status | Disposition |
|--------|--------|-------------|
| `pre-0.5.0-release` | Pushed, 4 commits ahead of main (includes README restructure + AGENTS.md) | **Ready to merge** — operator decision |
| `main` | Current default, has ADR-CDG-018 | Clean |
| `remotes/origin/docs/adr-cdg-017-neighborhood-remelt` | Unmerged | Banked on #114 (hygiene) |
| `remotes/origin/fix/119-tied-weights-device-map-guard` | Unmerged | Related to issue #119 — verify if fix landed elsewhere |
| `remotes/origin/fix/e2e-pack-identity-gate` | Unmerged | Banked on #114 (hygiene) |
| `remotes/origin/pi-changes` | Unmerged | Banked on #114 (hygiene) |
| `remotes/origin/release/0.4.0-patch` | Unmerged | Banked on #114 (hygiene) |
| `remotes/origin/salvage/s5-beta-rebuild-crash-wip` | Unmerged | Salvage branch — likely dead, prune candidate |

---

## Evidence from this session

- Run log examined: `/mnt/storage/DG-runs/dgemma_run_log.jsonl` (20 lines, step 8 mid-run showing molten count cells in table structure, final state clean)
- No new code written — all changes are documentation/metadata
- All commits pushed to origin; no local-only residue
