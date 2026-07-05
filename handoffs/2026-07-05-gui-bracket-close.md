# Handoff — live-GUI bracket closed, INT4-vs-P2 decision still open

**Date:** 2026-07-05 (second session of the day) · **From:** orchestrator session
(GUI verification) · **HEAD at handoff:** `169e6ad`

Cold-start path: run `/orient` against this repo. This file carries only what
the standing record does not; the record (README → plan.md → decisions/ →
loose-ends.md → examples/ → issues) is authoritative. Supersedes nothing in
`2026-07-05-phase1-close.md` except its "pending decision" framing — the
INT4-before-P2 question is *still* unanswered and now better evidenced.

## State in one line

Live-GUI verification **closed, PASS, all banked**: plan.md P1 addendum
(`347381d`), verified graph archived in both formats (`d865b14`, `1d706a3`),
E2E-probe recipe recorded and verified live (`169e6ad`). P2 starts with its
regression probe already in hand (`examples/README.md`).

## Environment standing state (not derivable from the repo)

- **Host rebooted mid-session** (~11:54 US/Mountain). Everything below postdates it.
- **ComfyUI instance up on port 8189** — bound `0.0.0.0`, `--verbose DEBUG`,
  log `/tmp/comfyui-8189.log`, own DB `/tmp/comfyui-8189.db`. **Both /tmp →
  gone on next reboot**; relaunch recipe is the command in the log's first
  line or just `main.py --port 8189`. **Port 8188 is the operator's — leave it free.**
- **`user/default/workflows` + `subgraphs` dirs created** in `/srv/dev/ComfyUI`
  (kills the frontend's userdata 404s; keep them).
- **Permissions opened for the operator's own env** (deliberate, operator-driven):
  `shane` is in groups `claude` + `assistant`; `/home/claude` is `g+x`
  (traverse-only); the diffusiongemma HF-cache tree is `g+rX` throughout.
  Cross-env recipe: `HF_HUB_CACHE=/home/claude/.cache/huggingface/hub` +
  `HF_HUB_OFFLINE=1` (avoids re-download *and* write attempts against a
  read-only-to-them cache).
- **Root `python main.py` (old pid 3740, since Jun 26) died with the reboot.**
  If it hasn't respawned, it was a one-off; stop holding the question.

## Pending decisions (operator, unchanged from last handoff)

1. **INT4 load test (issue #4) vs P2 knobs** — still the fork. #4 gained two
   runtime-evidence comments this session (nf4 widget-default OOM is
   structural: `{"": 0}` device map + bnb-unquantizable MoE experts; stale
   docstring `model.py:90-92` claims `none` "not viable" — contradicted by
   two PASSes). The widget default flip rides with #4's resolution.
2. **Issue #1** (`user:gate`, mask_token) — still open, close recommendation
   still posted.
3. Backlog note: #3, #4, #6, #7 remain untiered (orient flagged; deferred to
   next triage sweep).

## Session-learned cautions (cheap to re-derive wrongly)

- **The E2E probe is the cheap verification now** — POST
  `examples/ping-smoke.api.json` to `/prompt`, poll `/history/<id>`, assert
  `success` + the `CanvasState` validity readout. Verified live this session.
  Use it before believing any ComfyUI-adapter change.
- **Dead frontend with a healthy backend and no UI error → reboot first.**
  Full exoneration ladder is in loose-ends.md (2026-07-05 entry); don't
  re-derive it.
- First model load after reboot is **cold**: minutes of NVMe reads, looks
  like pagefile thrash. Not a hang. Warm loads ~25s.
- Operator's Ctrl-C copy reflex can kill background agents/harness — a
  late "agent failed" notification may just be that plus the reboot.
