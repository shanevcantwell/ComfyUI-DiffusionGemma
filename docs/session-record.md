# Session record — where it lives

This repo's public tree carries the product and its decision record, not the lab's
session log. Per [ADR-CDG-022](../decisions/adr-cdg-022-publish-policy-public-tree-vs-session-residue.md)
(Decision 2), session residue — per-session handoffs (`docs/handoffs/`), screenshot/log
evidence (`docs/evidence/`), and raw-run experiment artifacts (logs, JSON, `nvidia-smi`
dumps) previously under `docs/experiments/` — has been evacuated to the private annex:

`https://github.com/shanevcantwell/design-docs/tree/main/experiments/ComfyUI-DiffusionGemma`

That URL 404s for the public. This is deliberate, not a bug — see ADR-CDG-022 Decision 2:
the annex is a replicated private sibling repo (pools-seat access only), and the stub
here exists so a future agent or operator with that access can find where the record
went, without republishing the topology data the move exists to remove.

Tracking issue: [CDG #237](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/237).

## What's still public

`docs/experiments/liquid-phase-decoding/` stays — real citation fan-in (four ADRs plus
ROADMAP and ARCHITECTURE cite it), so it's concept-doc, not session residue (ADR-CDG-022
Decision 1's inbound-reference test).

`docs/experiments/2026-08-04-adr-cdg-012-q2-smoke/` is a **deferred** case: it stays
public for now because it's the live arm of a pending run, not because it has earned
citation fan-in yet. It evacuates (or graduates to concept-doc status) once the Q-2
bracket closes — re-apply the Decision 1 test at that point.

`docs/postmortems/` is unaffected by this evacuation — postmortems are decision-record
material, not session residue.
