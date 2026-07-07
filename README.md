# ComfyUI-DiffusionGemma

A ComfyUI node pack for **DiffusionGemma** — text generation by *uniform-state
discrete diffusion*, exposed as a ComfyUI graph you can watch, instrument, and
take apart.

> ### ✅ Status: working — loader, sampler, knobs, and live instrumentation
>
> As of **2026-07-05** the pack runs end-to-end in ComfyUI: prompt in → text out,
> with every entropy-bound knob exposed as a widget, a **live per-step canvas
> view** while the sampler runs, and a post-hoc **trace node** (commit heatmap +
> summary). Phases 0–3 of the [build plan](plan.md) are closed, each with live
> verification on real weights banked in the record. P4 (schedule node + curve
> zoo) and P5 (constraints + options chain) are next.

| Phase | What landed | Evidence |
|-------|-------------|----------|
| P0 — recon & spec | ADRs 001–003, build plan | [decisions/](decisions/) |
| P1 — vertical slice | `DGemmaLoader` + `DGemmaSampler`, prompt→text + validity readout | plan.md P1 evidence (3 live PASSes) |
| P2 — knobs | EB params/seed/thinking as widgets; thought-channel leak fixed (#8); quant default grounded | plan.md P2 evidence; entropy_bound sweep |
| P3 — instrumentation | `CANVAS_TRACE` + `DGemmaTrace`, live per-step push (`web/`), honesty readout (`turn_closed`/`answer_tokens`) | verifier PASS: ws events 1:1 with steps; [examples/](examples/) |

## What it is (the idea)

DiffusionGemma doesn't autoregress within a block. It starts from a fixed
**256-token canvas of random vocabulary tokens** and iteratively refines it: an
entropy-bound sampler commits the lowest-entropy positions under a budget and
re-noises the rest, step after step, until it stabilizes. Finished canvases are
appended to the KV cache and the next canvas begins (block-autoregressive).
There is **no sigma schedule and no latent space** — the "schedule" is a
per-step *temperature + entropy-budget* trajectory, and the working state is a
discrete token canvas.

ComfyUI's sampling ecosystem (`KSampler`, `BasicScheduler`, RES4LYF, the solver
zoo) is built on `SIGMAS` and `LATENT` — continuous Gaussian diffusion.
DiffusionGemma's loop has no input of that shape. So this pack keeps RES4LYF's
**node topology** (schedule → constraints → sampler → options chain) but swaps
every socket **payload** for entropy-native types. See
**[ADR-CDG-001](decisions/adr-cdg-001-native-socket-types.md)**.

## What works today

- **`DGemmaLoader`** — loads `google/diffusiongemma-26B-A4B-it` via transformers,
  drives via the Diffusers pipeline (ADR-CDG-004). `quant` offers `none` only
  (bf16 with CPU spill — fits a 48 GB card). bitsandbytes `nf4`/`int8` were
  removed (issue #18): they can't touch this model's fused 3D MoE experts —
  bnb only swaps `nn.Linear`, silently skipping ~22.84 B of 26 B params, so the
  "quantized" load is still ~46 GB and mislabeled as 4-bit on *any* card. A real
  quantized path for smaller cards is tracked in issue #4.
- **`DGemmaSampler`** — all knobs as widgets, defaults from grounded live runs:
  `num_inference_steps=48`, `t=[0.4, 0.8]`, `entropy_bound=0.1`,
  `confidence=0.005`, `gen_length=256`, `seed`, and a **`thinking` toggle**
  (injects the model's `<|think|>` control token). Outputs: `STRING` (clean —
  the model's thought-channel frame is excised at the id level, never leaked),
  `CANVAS_STATE`, `CANVAS_TRACE`, and `frames` — a per-step `STRING` list (raw,
  unexcised decode of every captured canvas snapshot: the in-graph "flipbook"
  from noise to coherent text).
- **Honesty readout** on `CANVAS_STATE`: `converged`, `committed_fraction`,
  `steps_used`, `turn_closed` (did the model actually end its turn, vs. run out
  of canvas), `answer_tokens` (pre-EOS count — trailing canvas-fill excluded),
  `thought` (channel content when thinking is on). A wrong-knob run *tells you*
  it's wrong instead of handing you plausible garbage.
- **Live view** — while the sampler runs, its node paints the canvas denoising
  step by step (`web/live_view.js`, fed by per-step server events; one event per
  step, verified 1:1 against `steps_used`).
- **`DGemmaTrace`** — post-hoc analysis over the complete trace: commit heatmap
  (`IMAGE`, positions × steps) + text summary. Frames are keyed by absolute
  noise level `(t, temperature, step_idx)`, so traces from different runs stay
  comparable.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/shanevcantwell/ComfyUI-DiffusionGemma
# restart ComfyUI
```

Requires `transformers==5.13.0` (DiffusionGemma support) and
`diffusers>=0.39.0` (the pipeline + schedulers — see ADR-CDG-004). Weights
(~54 GB bf16, ungated) download from
[google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it)
on first load. A 48 GB-class GPU runs the default `quant=none` path with CPU
spill (~2.3 s/step observed); smaller cards need a working quantized path,
which remains unresolved (#4).

**Example graphs** ([examples/](examples/)): start with
**`p3-trace-annotated.ui.json`** — the annotated canvas graph that *teaches* the
Loader → Sampler → Trace flow; open it in the ComfyUI canvas and read the embedded
Note nodes. The runnable smoke graphs (API format, operator-verified live) build
up the same shape in steps: `ping-smoke` (P1 minimal), `p2-knobs-smoke` (all
widgets), `p3-trace-smoke` (full instrumentation chain, + a `-thinking` variant).

## Known limitations (tracked, not hidden)

- **`thinking=true` can spend the whole canvas thinking** and return an empty
  answer — the readout flags it (`turn_closed=False, answer_tokens=0`) and
  issue #9 tracks the budget-policy design question.
- Knob response is **not a smooth dial**: block-autoregression makes output
  respond discontinuously to threshold knobs (plateaus and cliffs — issue #10
  has measured sweeps).
- Raw pre-excision canvas ids aren't yet exposed on any socket (issue #11) —
  wanted for token-level trace analysis.
- Quantized loading for accessible (8–24 GB) consumer cards is unresolved
  (issue #4) — the AWQ-INT4/compressed-tensors candidate surveyed there was
  smoke-tested and found incompatible with this pack's pinned `transformers`
  version (a real architecture-revision mismatch, not a config error); no
  viable candidate is currently identified. GGUF/llama.cpp (issue #15) is the
  most promising remaining direction on accessibility grounds, but is parked
  pending a design bridge for the live-view/trace instrumentation gap.

## Where the design lives

| Doc | What it holds |
|-----|---------------|
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Contributor-facing map — how the pieces fit and why. |
| **[plan.md](plan.md)** | The 6-phase build roadmap with per-phase evidence. |
| **[decisions/](decisions/)** | ADRs — *why* the load-bearing choices were made. |
| **[ADR-CDG-001](decisions/adr-cdg-001-native-socket-types.md)** | Native socket types instead of reusing `SIGMAS`/`LATENT`. |
| **ADR-CDG-002 → 004** | Access path: load via transformers, **drive via the Diffusers pipeline** (004 amends 002). |
| **ADR-CDG-005** | `CANVAS_STATE` is a resumable save-state, not a display snapshot. |
| **[ADR-CDG-006](decisions/adr-cdg-006-advanced-sampler-step-window-resume.md)** | `DGemmaSamplerAdvanced` — step-windowed, chainable/resumable sampler (**proposed**, not yet built). |
| **[loose-ends.md](loose-ends.md)** | Tactical decisions below the ADR bar. |

## Relationship to RES4LYF

This pack **steals RES4LYF's shape and rejects its substrate.** RES4LYF honestly
reuses `SIGMAS`/`LATENT` because it *is* genuinely sigma/latent-based. This pack
is not — so reusing those types would be a literal instance of the "lying
sigmas" trap RES4LYF jokingly named, but unintentional and load-bearing. The
node graph here reflects the real substrate, which is what makes it teachable
rather than a disguise. (See ADR-CDG-001.)

## License

GPL-3.0 (matching ComfyUI core). LICENSE file lands with registry publication.
