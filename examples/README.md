# examples/

Known-good graphs, banked with provenance. Two formats, named by suffix:
`.api.json` is the `/prompt` body a running ComfyUI accepts via
`POST /prompt {"prompt": <file contents>}`; `.ui.json` is the canvas format
(drag onto the ComfyUI canvas, or Workflow → Open).

| file | provenance |
|---|---|
| `ping-smoke.api.json` | The first operator-driven live-GUI PASS (2026-07-05, plan.md P1 evidence addendum): `ping` → `thought\nPong! How can I help you today?`, `converged=True committed_fraction=1.0 steps_used=3/48`. `quant="none"` is deliberate — the known-good load path on the 48GB dev box (issue #4). Sampler seed pinned as run; note the pipeline's canvas init is not seed-pinned across library versions (see loose-ends on seeding), so treat text as characteristic, not byte-exact. |
| `ping-smoke.ui.json` | The same graph in UI format, saved from the same PASS session (operator's `diffusiongemma_base_template`): node layout, link table, native socket types visible on the wire (`DGEMMA_MODEL`, `DGEMMA_CANVAS_STATE` — ADR-CDG-001). Seed widget carries control mode `"randomize"` — flip to `"fixed"` for repeatable runs. |
| `p3-trace-annotated.ui.json` | **Start here (text/instrumentation).** Derived from the live `p3-trace-smoke_2` graph (2026-07-06), reflecting the post-#18 loader schema (`quant=["none"]` only, new `local_files_only` toggle) — annotated with descriptive node titles and four `Note` nodes (overview, loader, sampler knobs, outputs) grounding every widget in the repo's own docs/issues. No separate `.api.json`: this file is for opening in the ComfyUI canvas to read, not for POSTing. |
| `p3-flipbook-annotated.ui.json` | **The picture-flipbook (0.2.0+).** The full P3 instrumentation graph plus the #21 `images` sampler output wired to `Preview Image` — the per-step canvas rendered as a shareable animation alongside the text `frames`. Four onboarding `Note` nodes (what it is / model & hardware / knobs / outputs) written for a first-time player. Requires the `images` output added in 0.2.0; won't load against a 0.1.0 sampler (4 outputs, no `images` slot). |

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

## p3-trace-smoke.api.json (+ -thinking variant) (2026-07-05, P3)

Derived from the live `/object_info` after the P3 commit (`eabc13f`). Full instrumentation
chain: `DGemmaLoader → DGemmaSampler → DGemmaTrace` with previews on text, canvas_state,
heatmap, and summary. Verifier PASS same day: 15/15 ws `dgemma.sampler.step` events vs
steps_used, heatmap IMAGE in history, `turn_closed=True answer_tokens=64` on a 43-word
answer. The `-thinking` variant reproduces the issue #9 empty-answer signature legibly
(`turn_closed=False answer_tokens=0`, unclosed thought span).

## p3-trace-annotated.ui.json (2026-07-06, post-#18)

Copied from `/srv/dev/ComfyUI/user/default/workflows/p3-trace-smoke_2.api.json` (the
same P3 chain, valid widget arity/links/layout) and transformed in place: node titles
renumbered (`1 - Load...`, `2 - Entropy-Bound Sampler`, `3 - Trace / Instrumentation`),
the `DGemmaLoader` node updated to the current `INPUT_TYPES` (`repo_id`, `quant` now
`["none"]`-only per issue #18, and a new `local_files_only` BOOLEAN — widgets_values
`["google/diffusiongemma-26B-A4B-it", "none", false]`), and four `Note` nodes placed in
empty canvas space (no overlap with existing nodes) explaining the graph's flow,
positioning (instrumentability over throughput — see loose-ends.md "DiffusionGemma ≤
its AR base"), the loader's knobs, the sampler's entropy-bound knobs (citing issues #9
and #10 for the thinking/confidence caveats), and what each Preview output means.
Structural validation only (JSON parse, node-type registration, link-table integrity,
widget-arity check) — not yet opened in a live ComfyUI instance; that confirmation is
the operator's to run.

**Recommended starting point** for a first-time user reading the pack: open this file
in the ComfyUI canvas (Workflow → Open) before `p2-knobs-smoke`/`p3-trace-smoke` — it's
annotated to teach the graph, not just to smoke-test it.

## kv-cache-tier1.api.json (ADR-CDG-012, issue #62 Phase 3)

The `KV_CACHE` seam's DV.2 minimum: the tier-1 honest-cache path —
`DGemmaLoader → DGemmaEncode` (mints a `DGEMMA_KV_CACHE` from a prompt) `→
DGemmaDenoise` (consumes the injected cache) `→ DGemmaTrace` (reads back
`injected_cache_provenance`, OUT-3), with `PreviewAny` on the decoded text
and `CanvasState`. Statically validated by `tests/test_kv_cache_workflows.py`
(class_type resolution, required-input completeness, wired-socket-type
round-trip against the live node definitions) — not yet run against a live
ComfyUI instance with real weights; that confirmation is the operator's
real-weights de-risk pass (issue #62 Phase 4, gated per the ADR's Open
Question #1). Tier-2 (`kv-cache-tier2.api.json`, per-layer cache surgery) is
deferred (issue #62 Q-1: out of first-implementation scope), not shipped
here.
