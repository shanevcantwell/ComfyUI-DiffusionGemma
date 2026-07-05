# examples/

Known-good graphs, banked with provenance. Two formats, named by suffix:
`.api.json` is the `/prompt` body a running ComfyUI accepts via
`POST /prompt {"prompt": <file contents>}`; `.ui.json` is the canvas format
(drag onto the ComfyUI canvas, or Workflow → Open).

| file | provenance |
|---|---|
| `ping-smoke.api.json` | The first operator-driven live-GUI PASS (2026-07-05, plan.md P1 evidence addendum): `ping` → `thought\nPong! How can I help you today?`, `converged=True committed_fraction=1.0 steps_used=3/48`. `quant="none"` is deliberate — the known-good load path on the 48GB dev box (issue #4). Sampler seed pinned as run; note the pipeline's canvas init is not seed-pinned across library versions (see loose-ends on seeding), so treat text as characteristic, not byte-exact. |
| `ping-smoke.ui.json` | The same graph in UI format, saved from the same PASS session (operator's `diffusiongemma_base_template`): node layout, link table, native socket types visible on the wire (`DGEMMA_MODEL`, `DGEMMA_CANVAS_STATE` — ADR-CDG-001). Seed widget carries control mode `"randomize"` — flip to `"fixed"` for repeatable runs. |
