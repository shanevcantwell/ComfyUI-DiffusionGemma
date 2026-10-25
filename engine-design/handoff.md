# DiffusionGemma native-engine handoff

## Status

- **Overall:** BLOCKED before I1.
- **Native engine:** not implemented.
- **I1:** not started.
- **Validated capability:** no real-weight parity, CUDA, ComfyUI integration, release, or phase PASS.
- **Image lane:** disabled.

## Authority and seat

- **Repository:** `https://github.com/shanevcantwell/ComfyUI-DiffusionGemma`
- **Branch:** `design/dg-engine-20261019`
- **Active worktree:** `~/worktrees/ComfyUI-DiffusionGemma-dg-engine`
- **Workflow ledger:** [issue #312](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/312)
- **Canonical Ground Physics:** `~/github/shanevcantwell/operating-doctrine/ground-physics/GROUND_PHYSICS.md`
- Do not work from the detached repository root or the dirty main worktree.

## Durable branch state

Facts:

- `acdbdaa` is the blocked I0 evidence-contract checkpoint.
- `3299ceb` is a non-authoritative worker-harness bootstrap. It is not an I0 PASS or an implementation base to promote by default.
- Local and upstream branch tips are both `3299ceb27b835a85f7da6a5f49d730b57b2f6429`.
- The branch was clean at task entry; this handoff replacement is the only intended uncommitted path.
- No implementation phase tag, receipt, release artifact, or implementation claim exists.

## Scope correction

Use the threat model of reviewed maintainer code running in ordinary trusted CI.

In scope:

- detect source or artifact drift;
- reject missing, stale, or malformed evidence;
- surface test failures;
- prevent accidental process leaks and bound worker resources.

Out of scope:

- malicious candidate code;
- hostile same-UID processes;
- root or kernel compromise;
- malicious CI or reviewer infrastructure;
- candidate-independent trust roots, CI root rotation, and recursive attestation.

The worker subprocess is for fault and resource isolation. It is not a security sandbox. Do not continue hostile-candidate sandboxing, external-root/fix23 work, or bespoke attestation in CDG.

## Retain from I0

These artifacts remain useful:

- `engine-design/reference.lock.json`;
- `engine-design/node-modes.json`;
- `engine-design/source-conflicts.md`;
- semantic fixtures and baseline-defect fixtures;
- schemas required by those records;
- ordinary fail-closed BASE tests;
- exact artifact and lineage blockers;
- AutoRound exclusion from native-engine distribution and release evidence.

## Do not promote

- Do not treat the `3299ceb` harness as an authority.
- Do not merge or promote the external-root/fix23 plan.
- Do not turn reusable import-attestation or security-platform machinery into CDG product scope.
- Preserve this history for diagnosis. Reuse requires a separately approved scope.

## Current blockers

- `/mnt/storage` is mounted read-only.
- `build/evidence` is a dangling link to the intended evidence store.
- Eleven official checkpoint shard objects are not provisioned in the evidence store; cached model bytes do not satisfy that gate.
- The selected-lineage BF16, Q8_0, and Q4_K_M GGUF artifacts are not provisioned.
- Three GGUF derivation records and their command/tool evidence are absent.
- ADR-CDG-020 requires an explicit sourcing and ownership decision before owned-patch or native-graph work.
- The observed local Q8 differs from the selected remote identity by 128 bytes and SHA-256; do not substitute it.
- AutoRound is excluded from distribution and is not a blocker.

External prerequisites keep the bracket BLOCKED. More gate machinery does not resolve them.

## Worktree safety

- **Active design worktree:** `~/worktrees/ComfyUI-DiffusionGemma-dg-engine` — use this seat only.
- **Detached rescue/root checkout:** `~/github/shanevcantwell/ComfyUI-DiffusionGemma` — no touch.
- **Dirty main experiment worktree:** `~/worktrees/ComfyUI-DiffusionGemma-main` — no touch.
- **Separate docs worktree:** `~/worktrees/ComfyUI-DiffusionGemma-cdg-026-027` — do not mix with engine work.

Before any write, verify the active branch, HEAD, upstream, worktree status, and path allowlist. Never reset, clean, stash, or reconcile another worktree.

## Next bounded bracket

Recommendations, in order:

1. Make the evidence store writable and provenance-bearing.
2. Decide whether ADR-CDG-020 is amended or superseded; update both decision directions.
3. Choose a clean implementation branch strategy. Recommended: create a new branch from D3 `7febb63` and curate only the useful I0 artifacts. Do not merge the non-authoritative harness wholesale.
4. Generate or acquire the exact checkpoint/GGUF artifacts and three derivation records. Close the minimal I0 contract using standard trusted CI and maintainer review.
5. Begin I1 with the smallest product proof: model metadata/load, one forward graph, and one authoritative reference vector.

Do not add generic enforcement mechanisms unless separately requested and scoped.

## Closure rules

- Allow at most two review-repair rounds before a frame review.
- Every review finding must receive one disposition:
  - `FIX_EXISTING_REQUIREMENT`
  - `WITHDRAW_CLAIM`
  - `TRACK_RISK`
  - `REQUEST_SCOPE_CHANGE`
- An external BLOCKED prerequisite stops the bracket.
- Commit blocked checkpoints early and label them accurately.
- Do not apply a phase label when its semantic scope has changed. Rename or re-review the phase instead.
- A gate certifies only its declared axis; source/synthetic checks are not real-weight or runtime validation.

## Pointers

- Workflow ledger: https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/312
- Current branch worktree: `~/worktrees/ComfyUI-DiffusionGemma-dg-engine`
- Canonical Ground Physics: `~/github/shanevcantwell/operating-doctrine/ground-physics/GROUND_PHYSICS.md`
- Durable forensic analysis: `~/forensic-runs/implement-llama-cpp-dg-engine--analysis--20261024/`

Do not use `/tmp` material as authority. Temporary copies may be verification evidence only.
