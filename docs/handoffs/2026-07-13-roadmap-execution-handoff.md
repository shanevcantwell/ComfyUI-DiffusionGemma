# Handoff — 2026-07-13 — roadmap-execution directive

**From:** orchestrator seat, session of 2026-07-13 (design marathon: architecture review → roadmap → sweep findings → AR/KV grounding).
**To:** the next cold session. Reconstruct from this record + `/orient` readback — never from remembered prose.

## Standing directive (operator, this date)

> Follow the roadmap autonomously with high agency, each phase dropped into its own branch. Build in
> the invariants and a test/test-coverage plan to at least soft-gate confirming each branch complete.

Operationalized in §Execution model below.

## Ground state at handoff — CLOSEOUT LANDED (verified at seat, session close; re-verify at orient)

- **`main` @ `aa455de`, clean, in sync with origin; PR queue EMPTY; final suite 193 passed / 4
  deselected.** The closeout batch cleared everything under the no-HITL policy:
  - #33 merged `b8e84ab` (loader 100% cov, disabled-flag + traversal gates green).
  - #39 merged `6575493` — **`ROADMAP.md` is on main and governing**, with the liquid-record
    addendum + H0-renoise pre-registered.
  - #31 merged `e1ed859` (main merged into branch at `d03ef1e`; sole conflict resolved preserving
    both #26 and #34 intents, Opus-reviewed, note banked on the PR). Mints ADR-CDG-009.
  - Follow-up PR #41 merged `aa455de`: all four #39 delta comments folded into the record
    (tied weights, AR-at-block-scale, H0-cache mechanics, #40 cross-refs, feedstock geometry) +
    README public links repointed (`plan.md`→`ROADMAP.md`; `loose-ends.md` link removed).
  - Ledgers #22, #26, #30 comment-closed with merge SHAs.
- **ARCHITECTURE.md is merged and governing** (PR #37, `fd1c295`): its `NOT-YET-IMPLEMENTED`
  conformance/enforcement rows are the phase checklist this directive executes.
- ADR-CDG-**010** (constraints/pinning) and **011** (control-signal/mod-matrix) are **NOT-YET-WRITTEN**;
  their complete drafting spec (required clauses) is in **issue #35** + its delta comment.
- **Known residue, deliberately left:** stale local branches (`docs/adr-cdg-008-mcp-center-topology`,
  `docs/liquid-phase-decoding` — content on main, `-d` refuses, `-D` needs operator word;
  `fix/20-anneal-effective-denominator`; orphaned `worktree-agent-*` bookkeeping branches);
  `plan.md`/`loose-ends.md` untracked by policy — closed-phase P0–P3 evidence has no published home
  (standing doc-debt); ADR-CDG-009's held design items (heatmap sentinel encoding, divider-frame
  design) remain open in the ADR.

## Execution model (the directive, operationalized)

**POLICY AMENDMENT (operator, later this same session): the PR-stamp/HITL ratification process is
retired.** High agency, phase-per-branch, merges proceed autonomously on the gate conditions below —
"sensible conditions to move to the next," not operator stamps. Operator visibility rides the record
(issues, PRs, ledger), not approval gates. The `user:gate` label survives ONLY for physical
hardware/secrets tasks (#4, #16) — hands, not judgment. A closeout batch clearing #31/#33/#39 under
this policy was dispatched at handoff time — verify their merge state via `gh` at orient, do not
assume.

**Note on this file's home:** `handoffs/` is gitignored by operator policy (`aa6bcb8`, agent-ops
residue stays out of the public tree) — this handoff is deliberately local-only; the public durable
record is the issues/PRs named below.

**Then, phase-per-branch autonomous execution of ROADMAP.md:**

1. **Branching:** one phase = one branch, `feat/roadmap-<phase-id>-<slug>`
   (e.g. `feat/roadmap-R4-conftest-fixture`). All work in isolated worktrees; the shared checkout's
   HEAD never moves; no force flags, no `--no-verify`.
2. **Order (Track A first):** R4 → R1 → R5 (cluster, in that order — R4's shared fake-pipeline
   fixture is what R1's composition-ordering tests are written against), R3 in parallel anytime,
   R2 with/before CDG-008 Phase 1; then CDG-008 Phases 1–4. ADR-CDG-010/011 drafted from #35's
   clause spec before/alongside the R-cluster (they are `auto:draft` — PR for ratification, never
   auto-merged). Track B research rungs gate on the R0 bench (Track A cluster + the two ADRs +
   #14/#11 capture).
3. **Invariants are built in, not bolted on:** each phase branch (a) names the invariant(s) it
   creates or flips, sourced from #35's enforcement-surface table / ARCHITECTURE.md's NYI rows;
   (b) adds the enforcement test(s) for them (composition-ordering test, zero-hooks-after-run,
   same-in/same-out walker statelessness, socket grep-gate, diffusers structural probe — per phase);
   (c) **updates ARCHITECTURE.md's conformance row from NOT-YET-IMPLEMENTED to in-force in the same
   branch** — the doc and the code move together or not at all.
4. **Soft gate per branch (completion = confirm-complete ring, readback not prose):** full suite
   green against baseline F0=∅; changed-path coverage evidenced; the phase's new enforcement tests
   present and passing; ARCHITECTURE.md row flipped. A phase that cannot ring exits honest
   (BLOCKED/HALTED with the gap named) — it is never merged "mostly done."
5. **Autonomy tiers (amended — no HITL):** all work runs fix→gate→review→merge. Sonnet implements,
   Opus reviews and gates — Sonnet output is a draft by definition, and the Opus review is the
   quality gate that replaces the operator stamp. Design-adjacent artifacts (ADR text, socket
   semantics, payload schemas) still get the strictest review tier and their reasoning banked in
   the PR, but they MERGE on passing the gate — nothing waits on an operator turn except
   `user:gate`-labeled hardware/secrets work.

## Key artifacts of this session (pointers, not summaries)

- **#35** — architecture review: verdict, F1–F9 findings, R1–R6 refactor spec (delta comment
  corrects: R4 before R1; live-view stays on `on_frame`, not a composite participant;
  declarative-payloads-only is a hard clause), ADR-010/011 required clauses.
- **#40** — entropy_bound sweep results + corrections: liquid phase sighted at
  committed_fraction=1.0 (`dimension↔diversity`); committed_fraction is a **lying convergence
  signal**; constraint-blind freezing / "waves" fossil mechanism, grounded (tied encoder/decoder
  weights; step-1≈AR-prior holds-with-caveats; committed blocks re-enter the cache by causal
  re-encode — DG is AR at block scale, diffusion within the block); seed-not-logged telemetry gap.
- **#36** — ComfyUI loop-sweep semantics (prompt snapshot; sweeps need linked inputs; ensembles
  belong in the engine, not graph loops). **#38** — cancellation seam (fold into R1 composer spec).
- **PR #39 thread** — record deltas incl. feedstock geometry (262,144-position ceiling; 5-of-30
  full-attention layers carry long range; feedstock tail dominates via sliding windows).
- Ledger of the morning's autonomous run: issue #30 (left open until drafts ratified).

## Open operator decisions carried forward

- #31 conflict-resolution authorization (merge main into reviewed branch).
- #39 README-link repoint: fold-in vs follow-up.
- Fossil-provenance seed sweep: pre-registration offered (seat prediction: high recurrence of the
  "Distant waves dance" fossil across seeds → prior-driven, not nucleation) — not yet registered.
- #4 note: the AWQ-INT4 lead-candidate checkpoint (`cyankiwi/...-AWQ-INT4`) is already in the HF
  hub cache — the user:gate hardware step is closer than the issue text implies.
