# examples/

Known-good graphs, banked with provenance. Two formats, named by suffix:
`.api.json` is the `/prompt` body a running ComfyUI accepts via
`POST /prompt {"prompt": <file contents>}`; `.ui.json` is the canvas format
(drag onto the ComfyUI canvas, or Workflow → Open).

| file | provenance |
|---|---|
| `ping-smoke.api.json` | The first operator-driven live-GUI PASS (2026-07-05, plan.md P1 evidence addendum): `ping` → `thought\nPong! How can I help you today?`, `converged=True committed_fraction=1.0 steps_used=3/48`. `quant="none"` is deliberate — the known-good load path on the 48GB dev box (issue #4). Sampler seed pinned as run; note the pipeline's canvas init is not seed-pinned across library versions (see loose-ends on seeding), so treat text as characteristic, not byte-exact. |
| `ping-smoke.ui.json` | The same graph in UI format, saved from the same PASS session (operator's `diffusiongemma_base_template`): node layout, link table, native socket types visible on the wire (`DGEMMA_MODEL`, `DGEMMA_CANVAS_STATE` — ADR-CDG-001). Seed widget carries control mode `"randomize"` — flip to `"fixed"` for repeatable runs. |

## End-to-end probe

The `.api.json` + a running ComfyUI is a complete E2E test of the
ComfyUI-loaded nodes — no GUI, no hand-wiring:

```sh
PID=$(curl -s -X POST http://127.0.0.1:8188/prompt -H 'Content-Type: application/json' \
  -d "{\"prompt\": $(cat examples/ping-smoke.api.json)}" | jq -r .prompt_id)
# poll until non-empty, then assert:
curl -s http://127.0.0.1:8188/history/$PID
```

`history[<PID>].status.status_str == "success"` is the pass condition;
`outputs` carries both PreviewAny payloads — node 74 the `STRING` text,
node 75 the `CanvasState` repr (validity readout: `converged`,
`committed_fraction`, `steps_used`) — so an assertion can check *honesty*,
not just non-emptiness. Verified 2026-07-05 against a live instance:
`success`, warm-load wall time well under the 40×5s poll budget. This is
the regression probe for P2, where the sampler's input schema is exactly
what changes.

## p2-knobs-smoke.api.json (2026-07-05, P2)

Derived mechanically from the running instance's `/object_info` after the P2 commit
(`c10ced0`) — all eight sampler knobs present with served defaults, prompt is the
thought-leak repro ("Why do birds suddenly appear?"), seed=7. Verifier PASS same day:
STRING clean of the thought frame, validity readout intact. The entropy_bound sweep is
this graph with only `entropy_bound` varied.
