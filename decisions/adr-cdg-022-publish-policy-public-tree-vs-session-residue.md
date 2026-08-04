# ADR-CDG-022 — Publish policy: public tree carries product + decision record; session residue evacuates to the private annex

**Status**: `proposed` — **ratification authority: the OPERATOR** (operator-ruled 2026-08-04; this ADR records the ruling precisely, it does not re-litigate it). Decisions 1–4 below are the operator's call as stated; this document's job is internal consistency, doctrine conformance, and naming the enforcement surface honestly — not re-opening the policy question.
**Date**: 2026-08-04
**Related**:
- [operating-doctrine#17](https://github.com/shanevcantwell/operating-doctrine/issues/17) — `design-docs/` write-scope-to-the-pools-seat doctrine; the operator's 2026-08-04 carve-out ruling (an agent-writable, per-repo annex under `design-docs/experiments/<repo>/`, outside the frozen embedding corpus) is what makes Decision 2 below possible without contradicting the corpus-freeze default that issue states
- [harness-tools#199](https://github.com/shanevcantwell/harness-tools/issues/199) — the future mechanical write-guard for `design-docs/`; **must allow the annex path** carved out by operating-doctrine#17, or this ADR's Decision 2 becomes unenforceable by the guard it will eventually sit under
- [CDG #163](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/163) — release gate (seat-run fresh-install + live smoke); Decision 4's leak-guard ties into this gate, not a separate CI job
- [CDG #201](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/201) — `CLAUDE.md` untracked since `aa6bcb8`, blocked on this policy fork; **consistent with, not caused by, this ADR** — #201's (a)-track-publicly vs (b)-deploy-from-private question is the same public/private-tree boundary this ADR draws for docs, and #201 unblocks once this ADR's Decision 1 confirms instruction files are product-adjacent (public) rather than session residue (annex) — see §Relation to #201
- ADR-CDG-020 (`docs/handoffs/2026-08-03-v0.5.0-released-next-kv-or-gguf.md:21` citation — repointed by this ADR's Decision 2; see §Implementation Notes)
- The remediation issue — [#237](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/237), filed for the 2026-08-04 leak sweep this ADR's Context reports

Grounding reads (verify the current-system claims by following these):
- GitHub traffic API, 14-day window ending 2026-08-04: 748 clones, 261 unique cloners, 173 views; top-referenced content issue #131.
- The 78-file docs push (7,703 insertions) — `docs/handoffs/` (27 files), `docs/evidence/` (17 files, 2.4 MB PNGs), raw experiment runs (logs/JSON/`nvidia-smi` dumps) under `docs/experiments/`.
- The 2026-08-04 leak sweep (banked on the remediation issue, number pending): no credentials found; IP↔hostname↔hardware↔OS topology mapping in handoffs (e.g. `192.168.137.x` lines), hostnames in experiment docs, operator system username in absolute paths across 12 files including live script constants.
- Inbound-reference audit: `docs/evidence/` 0 refs; `docs/handoffs/` 1 ADR citation (ADR-CDG-020, 3 lines, cited above); `docs/experiments/` unreferenced except `liquid-phase-decoding/` (4 ADRs + ROADMAP + ARCHITECTURE cite it).
- `.gitignore:45-49` — the existing "Agent-ops residue (kept locally, never shipped in the public tree)" stanza this ADR's Decision 4a extends.

---

## Context

The public repo has a real audience: 748 clones / 261 unique cloners / 173 views in the trailing 14 days, with issue #131 (the GGUF engine thread) as the top-referenced content. A 78-file docs push (7,703 insertions) shipped internal session residue into that public tree: `docs/handoffs/` (27 files — per-session handoff notes), `docs/evidence/` (17 files, 2.4 MB of PNGs), and raw experiment-run artifacts (logs, JSON, `nvidia-smi` dumps) under `docs/experiments/`.

A leak sweep run 2026-08-04 and banked on the remediation issue found no credentials — but did find IP↔hostname↔hardware↔OS topology mapping in handoff docs (e.g. `192.168.137.x` lines — the RTX-3090 workstation address named in the operating constitution's Environment section), hostnames in experiment docs, and the operator's system username embedded in absolute paths across 12 files, including constants in live scripts. None of this is a credential leak; all of it is infrastructure-topology disclosure the public tree has no product reason to carry.

An inbound-reference audit sharpens the picture: this residue is not load-bearing for the product record. `docs/evidence/` has zero inbound references from anywhere in the tree. `docs/handoffs/` has exactly one — a 3-line citation from ADR-CDG-020 (`docs/handoffs/2026-08-03-v0.5.0-released-next-kv-or-gguf.md:21`, the three packaging questions). `docs/experiments/` is unreferenced except `liquid-phase-decoding/`, which four ADRs plus ROADMAP and ARCHITECTURE cite — that subtree is doing real decision-record work, not sitting as session log.

The forcing question: **what belongs in a public tree with a real, measured audience, and where does the material that doesn't belong there go, without silently thinning the durable record the ecosystem's lab-notebook-floor and born-replicated disciplines both require?**

## Decision

### 1. What the public tree carries

Product code, tests, examples, `decisions/` (ADRs), `ROADMAP.md`/`ARCHITECTURE.md`/`README.md`/`CHANGELOG.md`, `docs/postmortems/`, and *concept/protocol* experiment docs — `docs/experiments/liquid-phase-decoding/` stays, by citation fan-in (four ADRs + ROADMAP + ARCHITECTURE). The public tree is the product and its decision record — not the lab's session log. The line is inbound-reference-shaped, not location-shaped: a `docs/experiments/` subtree that accrues real ADR/ROADMAP citations is concept-doc and stays; one that doesn't is raw-run residue and evacuates (Decision 2).

### 2. Where session residue lives

`docs/handoffs/`, `docs/evidence/`, and raw-run experiment artifacts (logs, JSON, `nvidia-smi` dumps, anything under `docs/experiments/` that isn't `liquid-phase-decoding/`-class concept documentation) evacuate to the fenced annex **`design-docs/experiments/ComfyUI-DiffusionGemma/`** — a private sibling repo, replicated — per the operator's carve-out ruling on [operating-doctrine#17](https://github.com/shanevcantwell/operating-doctrine/issues/17) (2026-08-04). That issue's default rule write-scopes all of `design-docs/` to the pools seat; the carve-out is a **named exception**: the annex sits outside `design-docs/`'s frozen embedding corpus, is agent-writable, and is organized as per-repo subfolders (`design-docs/experiments/<repo-name>/`) rather than the pools' hand-maintained top-level structure. This resolves the doctrine tension honestly rather than picking one side and dropping the other: lab-notebook-floor banking (every session's record is kept somewhere durable) and born-replicated (the record lives on more than one disk) both still hold — **the record moves, it does not thin.** A stub pointer file remains in the public `docs/` tree naming the annex location, URL-resolvable for the operator and agents with pools-seat access, 404 for the public — a deliberate asymmetry, not an oversight: the pointer preserves "a future agent finds where this went" without republishing the topology data the move exists to remove.

### 3. History is disclosed, not rewritten

261 cloners already hold the git history containing the leaked material. Running `git filter-repo` (or equivalent history rewrite) would break every one of those 261 clones on next pull and would **not** un-disclose anything already fetched — the topology data is already in hands outside this repo's control. HEAD is scrubbed (the residue is removed going forward, per Decision 2). The disclosure window is named on the remediation issue (number pending). Network-side mitigation — e.g. rotating anything topology-adjacent that a leaked IP/hostname mapping could inform — is the operator's call and is out of this repo's scope; this ADR does not decide it.

### 4. Enforcement surface

Per GROUND_PHYSICS discipline 6 (name the surface that enforces each invariant, and what takes over when it changes):

- **(a) `.gitignore` stanza extension.** The existing "Agent-ops residue (kept locally, never shipped in the public tree)" stanza (`.gitignore:45-49`, currently `CLAUDE.md`, `plan.md`, `loose-ends.md`, `test-coverage-plan.md`) extends to cover `docs/handoffs/`, `docs/evidence/`, and raw-run patterns under `docs/experiments/` (concept-doc subtrees like `liquid-phase-decoding/` are **not** matched — the ignore pattern must exempt cited concept-doc paths explicitly, not blanket-exclude `docs/experiments/`). This is the mechanical stop for *future* accidental re-adds; it does nothing for content already tracked, which Decision 2's evacuation handles separately.
- **(b) CI/pre-push leak-guard, generic-pattern-only.** A guard tied into the release gate (#163, not a standalone CI job) greps for **generic patterns only** — private-IP ranges (e.g. RFC 1918 CIDR blocks), `/home/\w+`, `/srv/dev/\w+` — and **never** a literal hostname/IP denylist. **Anticipated failure named per the repo's greenfield-adaptation convention (CLAUDE.md, harness-tools#18):** a tracked public workflow file containing a literal denylist of the actual hostnames/IPs it's trying to catch *re-leaks the exact data it exists to guard*, the moment that workflow file itself is public — the guard becomes the leak. Generic structural patterns (address-shape, path-shape) catch the failure class without ever encoding the specific values.
- **What stays prose-only, named honestly.** The judgment call of concept-doc vs. raw-run classification (Decision 1's inbound-reference test) is not mechanized by (a) or (b) — a human or agent applying the citation-fan-in heuristic decides which `docs/experiments/` subtree is which. This is a known-fragile enforcement gap, not a pretended-solved one: the `.gitignore` pattern in (a) can only encode the classification once made, not make it.

## Rationale

### Positive Consequences
- **The public tree's audience gets a signal-dense repo.** 261 cloners pulling handoffs and 2.4 MB of screenshot evidence with zero inbound references were paying a bandwidth and noise cost for material that was never written for them.
- **Topology disclosure stops going forward.** IP/hostname/username exposure in future commits is closed by (a); the historical exposure is named and bounded (Decision 3), not hidden.
- **The record doesn't thin.** The annex is replicated (private sibling repo), not deleted — lab-notebook-floor and born-replicated both hold, satisfying the same disciplines the public tree's own founding leaned on.
- **The concept-doc carve-out is principled, not arbitrary.** `liquid-phase-decoding/` stays because citation fan-in proves it is decision-record material, not because someone judged it "interesting enough" — the same test that moves the rest out.
- **The leak-guard cannot become a second leak.** Decision 4b's generic-pattern-only rule is chosen specifically to avoid the self-defeating case a literal denylist would create in a public repo.

### Negative Consequences
- **The public repo loses lab-notebook transparency.** Some fraction of the 748-clone/173-view traffic may have been *for* the research record itself (handoffs and evidence as a transparency artifact), not only for the product. This ADR trades that transparency for topology-safety and signal density; it is a real loss, not a free move.
- **A private annex splits the durable record across two repos.** A future reader following the public ADR trail into `docs/handoffs/` hits a 404-to-the-public stub instead of the record. Pointer discipline (the stub names the annex path) mitigates but does not eliminate the split — cross-repo pointer-following requires pools-seat access this ADR does not grant to the public.
- **The concept-vs-residue line requires ongoing judgment.** Decision 4's honest naming of the prose-only classification gap means every future `docs/experiments/` addition needs the same citation-fan-in call made again; there is no mechanical gate preventing a raw-run dump from landing under a concept-doc-shaped path.
- **Stub pointers 404 for the public by design.** This is deliberate (Decision 2) but is still a dead-end UX for any public reader who follows a citation into the annex without pools-seat access.

## Alternatives Considered

### Option A: Leave everything in the public tree, just scrub the leaked topology data in place
Keep `docs/handoffs/` and `docs/evidence/` public, edit out IPs/hostnames/usernames per-file.

**Why rejected:** treats the symptom (specific leaked strings) without addressing the cause (zero-inbound-reference session residue does not belong in a product's public tree regardless of whether it currently contains a leak). The next handoff would recreate the same exposure the next session it's written, since the underlying practice — writing session notes directly into the public tree — is what generates topology-adjacent content in the first place.

### Option B: Delete the residue outright, no annex
Remove `docs/handoffs/`, `docs/evidence/`, and raw experiment runs with no replicated destination.

**Why rejected:** violates lab-notebook-floor banking and born-replicated in one move — the record thins rather than moves. The operator's ruling on operating-doctrine#17 exists specifically to avoid this: an agent-writable annex outside the frozen corpus was carved out *because* deletion-with-no-replication was the unacceptable alternative.

### Option C: Rewrite git history to remove the residue from all past commits
Run `git filter-repo` or equivalent to strip the leaked material retroactively.

**Why rejected (deciding factor):** 261 unique cloners already hold the current history. A history rewrite breaks every existing clone on next pull and does not un-disclose data already fetched by those clones — it imposes a real cost (every downstream user must re-clone or force-reconcile) for zero disclosure benefit. Decision 3 chooses disclosure-named-and-bounded over rewrite-that-doesn't-undisclose.

### Option D: Literal hostname/IP denylist in the CI leak-guard
Encode the specific leaked values (the actual `192.168.137.x` range, the actual hostnames, the actual username) as the guard's match patterns.

**Why rejected (deciding factor):** the guard workflow file is itself tracked in the public repo. A literal denylist of the sensitive values re-publishes those exact values in the thing meant to catch them — the guard becomes a leak vector the moment it's committed. Generic structural patterns (Decision 4b) catch the failure class without ever writing the specific values into a public file.

## Open Questions

- [ ] Whether the annex later gets selective re-publication (a curated compendium) once findings graduate to ADRs. **Resolution:** revisit when a `liquid-phase-decoding/`-style citation-fan-in pattern emerges from annex content — the same Decision-1 test that keeps concept docs public would apply retroactively to annex material that earns citations.
- [ ] Whether other public repos in the ecosystem adopt this policy wholesale. **Resolution:** this ADR is CDG-scoped; the general rule (if any) is an `operating-doctrine` question, not decided here. Points at operating-doctrine as the place a cross-repo version would be recorded, per ADR-CON-0001 (one home per concept).

**Resolution plan:** both are post-ratification, non-blocking observations, not acceptance gates — this ADR ratifies on the operator's confirmation of Decisions 1–4 as recorded, independent of either question resolving.

## Relation to #201

[CDG #201](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/201) (`CLAUDE.md` untracked since `aa6bcb8`) is blocked on "the (a)-track-publicly vs (b)-deploy-from-private policy fork" — the same public/private-tree boundary question this ADR resolves for `docs/`. This ADR does not itself unblock #201 (instruction files are a distinct class from session-residue docs, and #201 cites its own harness-tools#229 dependency), but the two are **consistent, not coincidental**: both name the same underlying boundary (public product tree vs. private/agent-ops material) and resolve it the same direction (agent-ops residue stays out of the public tree — see `.gitignore:45-49`, which already lists `CLAUDE.md` in the same stanza Decision 4a extends). #201's own resolution still requires harness-tools#229's policy-fork decision; this ADR is cited there as precedent, not as the unblocking event.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD.

## Implementation Notes

Downstream work, tracked on the remediation issue (number pending — implementing PR fills this citation); **not performed by this ADR.**

| File | Change Type | Description |
|------|-------------|-------------|
| `decisions/adr-cdg-022-publish-policy-public-tree-vs-session-residue.md` | Created | This decision record |
| `docs/handoffs/*` (27 files) | Moved | Evacuate to `design-docs/experiments/ComfyUI-DiffusionGemma/handoffs/`; leave a stub pointer in `docs/` |
| `docs/evidence/*` (17 files, 2.4 MB) | Moved | Evacuate to `design-docs/experiments/ComfyUI-DiffusionGemma/evidence/`; leave a stub pointer in `docs/` |
| `docs/experiments/*` (raw-run subtrees, excl. `liquid-phase-decoding/`) | Moved | Evacuate per Decision 1's citation-fan-in test; `liquid-phase-decoding/` stays public |
| `decisions/adr-cdg-020-gguf-engine-sourcing-pinned-pr-branch.md` | Modified (citation repoint) | Its three citations of `docs/handoffs/2026-08-03-v0.5.0-released-next-kv-or-gguf.md:21` repoint to the annex path post-evacuation |
| `.gitignore` | Modified | Extend the "Agent-ops residue" stanza (currently `.gitignore:45-49`) per Decision 4a |
| `.github/workflows/*` (or equivalent, tied to #163) | Created | Generic-pattern leak-guard per Decision 4b, tied into the release gate |
| The remediation issue | Durable-emission | Disclosure window, evacuation completion, and guard-live confirmation recorded there |

## References

- GitHub repository traffic API (14-day window ending 2026-08-04): 748 clones, 261 unique cloners, 173 views.
- [operating-doctrine#17](https://github.com/shanevcantwell/operating-doctrine/issues/17) — `design-docs/` write-scope doctrine and the 2026-08-04 annex carve-out.
- [harness-tools#199](https://github.com/shanevcantwell/harness-tools/issues/199) — future mechanical write-guard for `design-docs/`.
- [CDG #163](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/163) — release gate the leak-guard ties into.
- [CDG #201](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/201) — `CLAUDE.md` untracked, consistent policy fork.
- ADR-CDG-020 (`decisions/adr-cdg-020-gguf-engine-sourcing-pinned-pr-branch.md`) — citation repointed by this ADR.
- `.gitignore:45-49` — the existing "Agent-ops residue" stanza extended by Decision 4a.
- `CLAUDE.md` (this repo) — greenfield-adaptation convention (harness-tools#18) invoked by Decision 4b's anticipated-failure naming.
- The remediation issue — [#237](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/237).

---
