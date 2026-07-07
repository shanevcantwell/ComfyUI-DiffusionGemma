# Architecture

How ComfyUI-DiffusionGemma's pieces fit together, and why. This document is
the map; the ADRs in [`decisions/`](decisions/) are the territory — where a
claim here rests on a decision record, it's cited rather than re-argued.

## 1. The thesis: instrumented exploration, not speed

This pack exists to let you **watch text crystallize step by step**, not to
generate text fast. DiffusionGemma has never been observed to run faster than
its autoregressive Gemma base in any harness — the "K steps ≪ T tokens" pitch
loses to a structural KV-cache asymmetry (AR pays for canvas width once,
cached; diffusion pays for it every step, uncached — full account in
[`loose-ends.md`](loose-ends.md)'s 2026-07-06 "DiffusionGemma ≤ its AR base"
entry). Don't sell it, or judge it, on throughput.

The value is observability: the commit-front sweeping across a canvas of
noise, the entropy field collapsing, a wrong answer visibly locking in and
(sometimes) escaping. ComfyUI is the right host for this because **ComfyUI is
about exploration** — a graph invites poking at the seams, rewiring one node
at a time. That's also why depth here lives in docs and ADRs rather than a
wall of canvas text: the graph should stay legible at a glance, and the
knob-level reasoning is one click away in `decisions/`.

## 2. The mental model

DiffusionGemma does not autoregress within a block. It starts from a **fixed
256-token canvas of random vocabulary tokens** and, over a schedule of steps,
commits the lowest-entropy positions under a per-step entropy budget while
re-noising the rest — full uniform-state renoise (a fresh random vocabulary
token), never an absorbing `[MASK]` (confirmed documentarily in
[ADR-CDG-002](decisions/adr-cdg-002-transformers-streamer-access-path.md)'s
resolved open question and reasserted in
[ADR-CDG-001](decisions/adr-cdg-001-native-socket-types.md)). Finished
canvases append to the KV cache and the next canvas begins
(block-autoregressive).

What the instruments (§6) show is the **commit front**: which positions have
locked in as of a given step, and which are still being renoised. The
temperature anneal runs hot→cool (`t_min`/`t_max`), and the acceptance rule
minimizes local self-consistency against the already-committed context, not
correctness — so a run can crystallize into a **coherent-but-wrong basin**
(the model commits to a plausible-sounding wrong answer because the correct
token reads as high-entropy against the closed context around it, and escape
from a closed basin is undirected). This mechanics-level account —
temperature schedule, renoise rule, the early-stop discontinuity, and the
basin failure mode — is grounded against the reference sampler in
`loose-ends.md`'s 2026-07-06 entries; this document only carries enough of it
to read what the trace instruments are showing.

## 3. The graph and its seams

**`DGemmaLoader` → `DGemmaSampler` → `DGemmaTrace`** (`nodes/loader.py`,
`nodes/sampler.py`, `nodes/trace.py`; registered in `__init__.py:39-43`).

**Engine/adapter split.** Every `nodes/*.py` module is a thin ComfyUI adapter:
unpack kwargs → call one `dgemma.*` function → wrap the result in a tuple, no
loop over denoising steps ever appears there
([ADR-CDG-003](decisions/adr-cdg-003-node-engine-seam.md)). All real logic —
the model, the types, the denoising loop, the trace-analysis math — lives in
`dgemma/`, which imports and runs with zero ComfyUI present (`dgemma/__init__.py:1-6`).
This split exists because the pack's whole point is per-step
instrumentation, which has to be developed and tested from a bare script, not
from inside a live node call.

**Native socket types.** `DGEMMA_MODEL`, `DGEMMA_CANVAS_STATE`,
`DGEMMA_CANVAS_TRACE` are bespoke payload types, not ComfyUI's `SIGMAS`/
`LATENT` — an entropy budget disguised as a sigma tensor is exactly the
trust-and-degrade failure ADR-CDG-001 forbids ("lying sigmas"). Each bespoke
socket type is just a string ComfyUI matches by equality; the object riding
it is the corresponding `dgemma/types.py` dataclass, passed through untouched
(ADR-CDG-003). See ADR-CDG-001 for the full rationale, including its
addendum on why a bare `STRING` output is its own instance of the same
lying-payload risk (§4 below).

**Hybrid runtime access path.** *Load* and *drive* are different seams
(`dgemma/model.py:1-8`,
[ADR-CDG-004](decisions/adr-cdg-004-diffusers-pipeline-drive-seam.md)):

- **Load** via transformers, unchanged since
  [ADR-CDG-002](decisions/adr-cdg-002-transformers-streamer-access-path.md):
  `DiffusionGemmaForBlockDiffusion.from_pretrained()` (`dgemma/model.py:93`).
- **Drive** via `diffusers.DiffusionGemmaPipeline` + a swappable scheduler
  (default `EntropyBoundScheduler`), not raw `.generate()` +
  `TextDiffusionStreamer` — ADR-CDG-004 amends ADR-CDG-002 here because the
  Diffusers scheduler's `.step()` natively returns the commit mask
  (`accepted_index`) that the trace/viz nodes need, and its callback contract
  supports mid-loop canvas overwrite (constraint injection, deferred to
  Phase 5). `dgemma/loop.py`'s `DGemmaPipeline` (`dgemma/loop.py:62-81`) is a
  one-line subclass widening the callback's tensor-input allowlist to include
  the full `scheduler_output`, per ADR-CDG-004's resolved open question (a).

## 4. The data model

Three dataclasses (`dgemma/types.py`) carry the engine's state across socket
boundaries, keyed on `(canvas_idx, step_idx, t, temperature)` — never on loop
index alone, so a run that starts mid-schedule stays comparable
(`dgemma/types.py:35-49`):

- **`DiffusionFrame`** — one denoising step's canvas snapshot plus its
  per-example commit fraction (`dgemma/types.py:31-89`).
- **`CanvasTrace`** — the complete per-step record of one `run_diffusion`
  call, carrying `scheduler_name`/`scheduler_config` alongside the frames.
  This is deliberate, not incidental metadata: "committed" means a persistent
  ratchet under `BlockRefinementScheduler` and a stateless per-step reading
  under `EntropyBoundScheduler` — a commit mask without the scheduler
  identity that minted it is a lying payload (ADR-CDG-001's addendum,
  `dgemma/types.py:92-113`).
- **`CanvasState`** — the validity readout riding alongside the decoded
  `STRING`. Its honesty fields exist because a bare string cannot say whether
  the canvas it came from actually finished denoising:
  - `converged` / `committed_fraction` / `steps_used` — did the schedule
    bottom out by the last captured step (ADR-CDG-001's "time-axis lying
    payload" addendum).
  - `turn_closed` / `answer_tokens` — issue #9's honesty rider: did the model
    actually hit EOS inside the generated region, versus running out of
    canvas with a plausible-looking but unfinished answer. `turn_closed=False`
    covers both the all-thought/empty-answer case and the budget-truncated
    case; `answer_tokens` counts pre-EOS content only, excluding the
    trailing renoise-fill pad a converged run leaves (`dgemma/types.py:184-212`,
    `dgemma/loop.py:244-264`).
  - `thought` — decoded content of the excised `<|think|>` channel, when
    non-empty (`dgemma/types.py:163-172`).

**Two per-step outputs, not one.** `DGemmaSampler` emits both `frames` (a
`STRING` list, one raw decode per captured frame — the in-graph text
flipbook, `dgemma/loop.py:432-462`) and `images` (a single stacked `(N, H, W,
3)` `IMAGE` batch rendering that same series, issue #21,
`nodes/frames_image.py`). They share one decode pass (`nodes/sampler.py:238-244`)
rather than decoding twice. `frames` is a list output (`OUTPUT_IS_LIST`);
`images` is one batch tensor, not a list — a list would fan out per-frame and
break `PreviewImage`'s scrubber, `SaveAnimatedWEBP`, and VHS-style consumers
that expect a single batched `IMAGE` (`nodes/sampler.py:202-207`, docstring
44-48).

## 5. The two display paths

This is the thing that trips people up: **the live per-step view and the
node's return outputs are two different mechanisms, not one.**

- **Per-step LIVE view** — a feature of `DGemmaSampler`'s own node body, not
  a downstream node. ComfyUI hands a node's outputs to downstream sockets
  only once its `FUNCTION` returns, so there is no way to stream per-step
  state through a socket while the sampler's own loop is still running. The
  sampler's sync `sample()` method instead calls
  `PromptServer.instance.send_sync("dgemma.sampler.step", ...)` once per
  captured step (`nodes/sampler.py:114-161`, `_build_on_frame`), and a
  `WEB_DIRECTORY`-registered JS extension (`web/live_view.js`,
  `__init__.py:53`) listens for that event and paints the canvas as it
  denoises. This display push is guarded so a websocket hiccup can never
  abort a multi-step generation (`nodes/sampler.py:124-133`).
- **Node outputs** — `text`, `canvas_state`, `canvas_trace`, `frames`,
  `images` all exist only once `sample()` returns, and are the *complete*
  post-hoc record, not a display feed.

**`CANVAS_STATE` is a resumable save-state, not a display snapshot**
([ADR-CDG-005](decisions/adr-cdg-005-canvas-state-resumable-savestate.md)):
its full contract (canvas token ids, schedule position, scheduler identity +
config + commit state, RNG generator state) is a sufficient statistic for
rewinding or branching a trajectory at any step — deliberately excluding the
KV cache, which is cheaply recomputable via one prefill pass. Today's
`CanvasState` ships a subset of that contract (the validity-readout fields
above); the full resumable contract is what
[ADR-CDG-006](decisions/adr-cdg-006-advanced-sampler-step-window-resume.md)
(status `proposed`, not yet built) would realize.

## 6. The instruments

`DGemmaTrace` (`nodes/trace.py`) is post-hoc analysis over a complete
`CANVAS_TRACE` — built from `dgemma/sampling.py`'s pure functions, which read
the trace's frames and derive per-position signal by **diffing consecutive
canvas snapshots** (no per-position entropy or commit mask is retained
per-frame; only the aggregate `committed_fraction`, so "did this position's
token change since last frame" is the only signal actually available —
`dgemma/sampling.py:12-23`).

- **Commit heatmap** (`build_commit_heatmap`, `dgemma/sampling.py:40-83`) —
  a steps × positions grid, `1` where a position's token changed since the
  previous frame (still being renoised), `0` where it held steady (locked
  in). Rendered as an `IMAGE` in `nodes/trace.py:38-46`. What it tells you:
  where and when the commit front swept across the canvas — white for
  still-moving, black for locked. First frame of each block reads all-`1`
  honestly (nothing has committed yet).
- **Avalanche / commit-fraction curve** (`build_avalanche_curve`,
  `dgemma/sampling.py:86-90`) — `committed_fraction` per step, in order. What
  it tells you: the "Neither Parallel Nor Sequential" shape of the
  convergence — plateaus and cliffs rather than a smooth ramp (corroborated
  in `loose-ends.md`'s discontinuity findings, issue #10).
- **Mask-token corroboration** (`corroborate_no_mask_token`,
  `dgemma/sampling.py:93-158`) — checks whether positions caught mid-renoise
  held a *varying* set of prior token ids (uniform-state renoise) or
  repeatedly the *same* one (an absorbing-MASK signature). What it tells
  you: the empirical confirmation of ADR-CDG-002/ADR-CDG-001's documentary
  "no MASK" claim — this is Phase 3's runtime evidence for a fact the type
  design already assumed.

## 7. Anticipated evolution — ecosystem-alignment refactor (post-0.2.0)

This pack was built **deliberately standalone** from the start. The operator
wasn't sure how big or useful it would get, so the broader ecosystem's
conventions (the harness-tools ground-physics / opinion-locality doctrine —
see this repo's `CLAUDE.md` doctrine pointers) were not imposed early. It has
since proven out.

An alignment refactor to bring this pack in line with the rest of the
ecosystem's conventions is **anticipated as likely the first work after the
0.2.0 release ships** — it is not scoped, not designed, and not started. This
document describes the pack's current, pre-refactor structure. When that
refactor is actually scoped, it gets its own ADR and roadmap entry; nothing
here should be read as a decision about its shape.
