# examples/

Known-good graphs, banked with provenance. These are **API-format** payloads
(the `/prompt` body a running ComfyUI accepts via
`POST /prompt {"prompt": <file contents>}`) — not the UI drag-drop format.

| file | provenance |
|---|---|
| `ping-smoke.api.json` | The first operator-driven live-GUI PASS (2026-07-05, plan.md P1 evidence addendum): `ping` → `thought\nPong! How can I help you today?`, `converged=True committed_fraction=1.0 steps_used=3/48`. `quant="none"` is deliberate — the known-good load path on the 48GB dev box (issue #4). Sampler seed pinned as run; note the pipeline's canvas init is not seed-pinned across library versions (see loose-ends on seeding), so treat text as characteristic, not byte-exact. |
