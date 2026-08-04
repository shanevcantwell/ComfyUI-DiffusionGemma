# docs/experiments/ — the repo's lab notebook

This directory is this repo's lab notebook: each record states its hypothesis (H0) before
observation, and null results are banked with the same care as confirmations — a falsified
hypothesis is a finding, not a loss.

Per [ADR-CDG-022](../../decisions/adr-cdg-022-publish-policy-public-tree-vs-session-residue.md),
the public tree carries only *concept/protocol* experiment docs — subtrees that accrue real
ADR/ROADMAP citations (the inbound-reference test, Decision 1). Raw-run subtrees without that
citation fan-in evacuated to the private annex,
[`design-docs/experiments/ComfyUI-DiffusionGemma/`](https://github.com/shanevcantwell/design-docs/tree/main/experiments/ComfyUI-DiffusionGemma)
(404s for the public — deliberate, see `../session-record.md`).

| directory | one-line finding | status/date |
|---|---|---|
| `liquid-phase-decoding/` | reframes DiffusionGemma's sharp commit threshold as a missing liquid phase between steam (renoised) and frozen (committed) states; 5 H0s (control/observe/project/substrate/cache) | untested, minted 2026-07-12 |

Evacuated (raw-run residue, no inbound citations — now in the annex): the 2026-08-04 issue-131
rung-1 GGUF probe, the 2026-07-30 AutoRound unified-path split check, the bf16-fit-mechanism
run, three DiffusionGemma numeral/KV-authorship H0 sweeps (2026-07-14, 2026-07-15,
2026-07-16), and the Q-2 window preflight / live smoke run (`2026-08-04-adr-cdg-012-q2-smoke/`,
deferred until the Q-2 bracket closed at Phase-4 merge `ac3c832`, then evacuated per CDG #237).
