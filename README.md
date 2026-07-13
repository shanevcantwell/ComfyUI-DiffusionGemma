# ComfyUI-DiffusionGemma
<img width="1774" height="1674" alt="image" src="https://github.com/user-attachments/assets/38871944-af3f-42ba-9422-cc222ec3e4eb" />

A ComfyUI node pack for **DiffusionGemma** — text generation by *uniform-state
discrete diffusion*, exposed as a ComfyUI graph you can watch, instrument, and
take apart.

> ### ✅ Status: working end-to-end
>
> Prompt in → text out, live in ComfyUI: every entropy knob on a widget, a
> **live per-step view** of the canvas denoising as it runs, a **picture
> flipbook** of the whole process, and a **trace node** (commit heatmap +
> summary) to read what happened. Verified on real weights across two GPUs.
>
> **VRAM footprint today: ~50GB bf16 + CPU spill, needs a ≥48GB card.**
> `quant="none"` is the only load path (bitsandbytes can't touch this
> model's fused MoE experts — see "What works today" below); the model
> card's ~18GB quantized / consumer-GPU footprint is **not yet reachable
> through this pack**. A real quantized load path is tracked in
> [issue #4](../../issues/4).
>
> Where it's headed lives in the [roadmap](ROADMAP.md).

| Phase | What landed | Evidence |
|-------|-------------|----------|
| P0 — recon & spec | ADRs 001–003, build plan | [decisions/](decisions/) |
| P1 — vertical slice | `DGemmaLoader` + `DGemmaSampler`, prompt→text + validity readout | 3 live PASSes (recorded in-repo) |
| P2 — knobs | EB params/seed/thinking as widgets; thought-channel leak fixed (#8); quant default grounded | live PASS + entropy_bound sweep |
| P3 — instrumentation | `CANVAS_TRACE` + `DGemmaTrace`, live per-step push (`web/`), honesty readout (`turn_closed`/`answer_tokens`) | verifier PASS: ws events 1:1 with steps; [examples/](examples/) |

## What it is — meaning annealed out of noise

Every answer starts as a **canvas of pure noise**: 256 positions, each a random
token drawn from the whole multilingual vocabulary — maximum entropy, no meaning
anywhere. Generation is an **annealing**. A temperature schedule starts hot and
cools; at each step the positions the model is most *certain* about — the
lowest-entropy ones — freeze into place, while the rest are re-noised and tried
again. The text isn't written left-to-right. It **precipitates out of the
entropy field**: the confident tokens crystallizing first, the uncertain ones
settling last, until a coherent answer has cooled out of the noise.

That process is the thing this pack lets you **watch**. The "schedule" here is
a temperature-and-entropy trajectory over a canvas of discrete tokens — no sigma
curve, no latent space. The whole point is to see the **commit-front** sweep
across the canvas: where meaning locks in early, where it stays molten, and —
sometimes — where the model anneals confidently into a *wrong* answer and can't
climb back out, exactly the way real annealing gets trapped in a local minimum.
You can't catch that by reading the final text. You can watch it happen here.

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
  `CANVAS_STATE`, `CANVAS_TRACE`, `frames` — a per-step `STRING` list (raw,
  unexcised decode of every captured canvas snapshot: the in-graph text
  "flipbook" from noise to coherent text) — and **`images`** (#21): that same
  per-step series rendered as a single batched `IMAGE`. Being a standard IMAGE
  batch (not a per-frame list), it plugs straight into **VideoHelperSuite's
  `Video Combine`** or `SaveAnimatedWEBP` for a shareable **GIF / MP4 / WEBP** —
  no adapter node needed.
- **Honesty readout** on `CANVAS_STATE`: `converged`, `committed_fraction`,
  `steps_used`, `turn_closed` (did the model actually end its turn, vs. run out
  of canvas), `answer_tokens` (pre-EOS count — trailing canvas-fill excluded),
  `thought` (channel content when thinking is on). A wrong-knob run *tells you*
  it's wrong instead of handing you plausible garbage. If you're asking "did
  this run finish?", read `turn_closed` (or the `finished_honestly` property),
  not `converged` — adaptive stopping can legitimately halt with
  `converged=False` on a clean, correct run.
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
on first load.

### Hardware & memory — the honest requirements

This is a **large model with no quantized path yet** (issue #4 — bitsandbytes
can't quantize its fused MoE experts, so you load full bf16, ~54 GB). The model
card asks for a ≥ 60 GB GPU for a naïve full-VRAM load — but **you do not need
one**, because **ComfyUI's memory management carries it**: it offloads weights to
system RAM and streams them to the GPU as needed.

- **Disk — ~54 GB free.** The weights download once to your HuggingFace cache
  (`~/.cache/huggingface`, or wherever `HF_HOME` points); budget the space before
  you start.
- **First run is slow — that's the download, not a hang.** The very first load
  pulls the full ~54 GB from HuggingFace before generation begins; on a normal
  connection that's a long, silent wait. It's **cached after**, so every load
  afterward is far faster. Once it's cached, flip the loader's `local_files_only`
  on to skip the network check entirely.
- **VRAM — confirmed running on 48 GB (RTX-8000) and, squeezed, 24 GB
  (RTX-3090).** 24 GB is the tested practical floor: it *just* fits, riding
  ComfyUI's automatic offload.
- **System RAM — the requirement people miss.** Whatever isn't resident in VRAM
  lives in system RAM, so you need room to hold most of a ~54 GB model off-GPU.
  On a 24 GB card the bulk of it rides in RAM — thin system memory, not VRAM, is
  what actually stops a run.
- **Speed — offload costs time:** ~2.3 s/step on the 48 GB card with CPU spill,
  slower as VRAM shrinks. More VRAM → less offload → faster. (Instrumentability,
  not speed — as ever.)
- **Below ~24 GB VRAM:** not yet — a quantized / GGUF path for 8–16 GB cards is
  still open (issues #4, #15).

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
- Quantized loading for consumer cards **below the ~24 GB offload floor**
  (8–16 GB) is unresolved
  (issue #4) — the AWQ-INT4/compressed-tensors candidate surveyed there was
  smoke-tested and found incompatible with this pack's pinned `transformers`
  version (a real architecture-revision mismatch, not a config error); no
  viable candidate is currently identified. GGUF/llama.cpp (issue #15) is the
  most promising remaining direction on accessibility grounds, but is parked
  pending a design bridge for the live-view/trace instrumentation gap.

## Where the design lives

| Doc | What it holds |
|-----|---------------|
| **[VISION.md](VISION.md)** | *Why it might matter* — the questions the instrument was built to ask, each tagged `[established]` / `[hypothesis]` / `[open]`. Speculative by design, cited throughout. |
| **[ROADMAP.md](ROADMAP.md)** | *Where it's headed* — the forward view in two tracks: engineering seam work (issue #35) and the liquid-phase research program. Pointer-heavy; VISION holds the *why*, `decisions/` the *decided*. |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Contributor-facing map — how the pieces fit and why. |
| **[decisions/](decisions/)** | ADRs — *why* the load-bearing choices were made. |
| **[ADR-CDG-001](decisions/adr-cdg-001-native-socket-types.md)** | Native socket types instead of reusing `SIGMAS`/`LATENT`. |
| **ADR-CDG-002 → 004** | Access path: load via transformers, **drive via the Diffusers pipeline** (004 amends 002). |
| **ADR-CDG-005** | `CANVAS_STATE` is a resumable save-state, not a display snapshot. |
| **[ADR-CDG-006](decisions/adr-cdg-006-advanced-sampler-step-window-resume.md)** | `DGemmaSamplerAdvanced` — step-windowed, chainable/resumable sampler (**proposed**, not yet built). |

## Come explore

This is an instrument for poking at how this diffusion LLM thinks. Questions,
findings, and half-formed ideas are exactly the point. The
**[Discussions](../../discussions)** tab is open for show-and-tell (post a
trace, a heatmap, a run that annealed somewhere strange) and for ideas. See
**[CONTRIBUTING.md](CONTRIBUTING.md)** for how to jump in.

## License

GPL-3.0 (matching ComfyUI core). LICENSE file lands with registry publication.
