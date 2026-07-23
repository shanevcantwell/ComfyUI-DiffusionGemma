# ComfyUI-DiffusionGemma
<img width="1774" height="1674" alt="image" src="https://github.com/user-attachments/assets/38871944-af3f-42ba-9422-cc222ec3e4eb" />

A Model Context Protocol (MCP) toolkit and ComfyUI node pack for **DiffusionGemma** —
discrete diffusion text generation. Exposes the full 18-bit uniform-state melt with
per-step canvas snapshots, commit heatmaps, and structured trace data. Watch meaning
crystallize out of noise.

> ### ✅ Status: Working End-to-End (Now on 32GB Hardware)
>
> Prompt in → text out, every entropy knob on a widget, per-step canvas tracking,
> flipbooks, and a **trace node** (commit heatmap + summary) to read what happened.
> Verified on real weights across two GPUs.
>
> **VRAM barrier broken:** The custom loader now supports `quant='autoround'` (INT4),
> dropping the footprint to **~30.7 GB** with a 27-second load time. The 50 GB bf16
> (`quant='none'`) path remains available. Details in [issue #4](../../issues/4).

| Phase | What landed | Evidence |
|-------|-------------|----------|
| P0 — recon & spec | ADRs 001–003, build plan | [decisions/](decisions/) |
| P1 — vertical slice | `DGemmaLoader` + `DGemmaSampler`, prompt→text + validity readout | 3 live PASSes (recorded in-repo) |
| P2 — knobs | EB params/seed/thinking as widgets; thought-channel leak fixed (#8); INT4 quant grounded | live PASS + entropy_bound sweep |
| P3 — instrumentation | `CANVAS_TRACE` + `DGemmaTrace`, per-step push (`web/`), honesty readout (`turn_closed`/`answer_tokens`) | verifier PASS: ws events 1:1 with steps; [examples/](examples/) |

---

## Quick start

Unlike autoregressive LLMs, DiffusionGemma generates text by *annealing a canvas of noise* — every position starts random and crystallizes into coherent output step-by-step.

This repository provides an instrumentable graph to capture the full per-step trajectory (schedule position `t`, temperature `T`, entropy, commit fraction) and exposes it as heatmaps, flipbooks, and structured trace data. **Crucially, the underlying code acts as an MCP toolkit.** The exact same execution logic that drives the ComfyUI nodes can be utilized directly by autonomous agents for multi-agent reasoning and constraint propagation.

**Nodes:** `DGemmaLoader` → `DGemmaSampler` → `DGemmaTrace`

### Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/shanevcantwell/ComfyUI-DiffusionGemma
# restart ComfyUI
```

Requires `transformers==5.13.0` (DiffusionGemma support) and `diffusers>=0.39.0`. Weights download from [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google/diffusiongemma-26B-A4B-it) on first load. The loader automatically bypasses the 46GB KV-cache warmup and patches tied-weight finalization when the INT4 path is selected.

### Hardware & Memory

| Mode | VRAM | Load time | Notes |
|------|------|-----------|-------|
| `quant='autoround'` (INT4) | ~30.7 GB | ~27 s | Pre-quantized checkpoint. Accessible on standard 32GB/48GB setups. |
| `quant='none'` (bf16) | ~50 GB | slower | Full precision, CPU spill via ComfyUI offload. |

* **Disk:** ~54 GB free for bf16 cache; INT4 pulls a smaller pre-quantized checkpoint.
* **First run is slow:** That's the download, not a hang. Cached after first load; flip `local_files_only` on to skip network checks.
* **System RAM matters:** Whatever isn't in VRAM lives in system RAM. Thin system memory, not VRAM, is what actually stops a run under heavy offload.
- **Speed — offload costs time:** ~2.3 s/step on 48 GB with CPU spill, faster as VRAM grows.

**Example graphs** ([examples/](examples/)): start with
**`p3-trace-annotated.ui.json`** — the annotated canvas graph that *teaches* the
Loader → Sampler → Trace flow; open it in ComfyUI and read the embedded Note nodes.

---

## How it works

### Uniform-state diffusion vs. autoregressive & masked models

Traditional text generation is **left-to-right**: each token locks in permanently, no
revision. Masked diffusion improves on this with "fill-in-the-blank" slots, but once a
position is unmasked, the token is fixed — still one-way commitment.

**DiffusionGemma departs from both.** It starts every position at **maximum entropy** —
a uniform draw from the full 262,144-token vocabulary (≈18 bits per position). The
generative process is a **cooling schedule**: temperature shrinks, distributions sharpen,
and positions freeze when their entropy drops below a configurable bound. Unfrozen
positions are re-noised each step, giving the model **native self-correction** — a token
can commit, then remelt if context shifts push its entropy back above budget.

The polyglot cascade you see mid-run (Katakana, Bengali, CJK flickering) isn't a glitch;
it's the visual signature of maximum-entropy categorical noise before the model narrows
its focus.

### The MCP-Powered Nodes

Because these nodes are built on a Model Context Protocol backend, their parameters are strictly typed and verifiable.

- **`DGemmaLoader`** — Loads the weights. The bespoke loader accepts `dtype='auto'`, seamlessly handling the INT4 `autoround` path without forcing massive BF16 allocations.
- **`DGemmaSampler`** — All knobs as widgets. Defaults: `num_inference_steps=48`, `t=[0.4, 0.8]`, `entropy_bound=0.1`. Features a **`thinking` toggle** (injects `<|think|>` control token). Outputs: `STRING`, `CANVAS_STATE`, `CANVAS_TRACE`, `frames`, and **`images`** (per-step series as a batched `IMAGE` for VideoHelperSuite export).
- **`DGemmaTrace`** — Post-hoc analysis: commit heatmap (`IMAGE`, positions × steps) + text summary. Frames keyed by `(t, temperature, step_idx)`.

### Knobs & units

The sampler's knobs mix schedule positions, temperatures, and an entropy budget — same-looking names, different units:

| Symbol | What it is | Units | Default |
|--------|-----------|-------|---------|
| `T` (temperature) | Divisor in `softmax(z/T)`; sharpens distribution as it shrinks. `T=1` = trained calibration. | dimensionless | — |
| `t` (schedule position) | `(N − step_idx)/N`; decreasing 1 → `1/N`. Not a temperature. | dimensionless | — |
| `t_min` / `t_max` | Temperature endpoints of `T = t_min + (t_max − t_min)·t` | dimensionless (temperatures) | `0.4`, `0.8` |
| `entropy_bound` | Per-step joint acceptance budget — positions below this freeze | **nats** | `0.1` |
| `confidence` | Early-stop threshold | probability | `0.005` |

Full provenance: [`KNOB_DOCS`](dgemma/loop.py) in `dgemma/loop.py`.

### Honesty readout

`CANVAS_STATE` reports: `converged`, `committed_fraction`, `steps_used`,
`turn_closed` (did the model end its turn, or run out of canvas?), `answer_tokens`
(pre-EOS count), `thought` (channel content when thinking is on). A wrong-knob run
*tells you* it's wrong instead of handing you plausible garbage.

**What telemetry does and doesn't show:** commit dynamics measure *when* positions
freeze, not *whether* they were diffusion-computed vs. emitted from the model's
autoregressive prior. Read committed_fraction as "when did this settle," never as
"this was computed in-canvas." See [issue #78](../../issues/78) for the full finding.

---

## Under the hood

### The sublimation problem

Current uniform-state models **skip the liquid phase** — snapping from maximum-entropy
noise directly to solid tokens without holding superposition across steps. Lowering
confidence thresholds doesn't help; outputs just boil off into uncorrelated noise.

The bottleneck is architectural: all intervention points cluster at the pipeline's output,
forcing binary commit-or-reset decisions. The model has no continuous embedding space to
hold multi-step distributional beliefs — it jumps from noise to solid because that's the
only seam available.

### The crystalline proxy

Without access to a trained mask anchor (DiffusionGemma lacks one), this pack enforces
constraint propagation through a **cellular automaton over solid tokens**. When a position
commits, it triggers a **local remelt** — briefly re-annealing neighbors so they adapt
structurally. This sidesteps sublimation by treating text as linked crystals that adjust
to each other, not a single liquid pool.

The division of labor: the cellular automaton enforces strict topological rules locally;
the model's bidirectional attention ensures semantic coherence globally. It works with
existing output seams — no weight modifications required.

### The embedding seam barrier

True **liquid-state decoding** requires intervention at the input embedding seam, not the
output. Models using latent refinement decoding and soft-masked diffusion solve
superposition by operating in continuous embedding space (Equation 3: split embeddings
into mask weight + predicted token weight, balanced by `α`). DiffusionGemma has no trained
mask anchor — substituting `Ē` (expected uniform mixture of all tokens) lands off-manifold
in a semantic dead zone. Without continued pre-training, the model interprets it as garbage.

**Next frontier:** shifting intervention to input embeddings for true distributional
smoothing before the first token forms. See [VISION.md](VISION.md) and
[ROADMAP.md](ROADMAP.md).

---

## Known limitations (tracked, not hidden)

- **`thinking=true` can spend the whole canvas thinking** and return an empty answer —
  the readout flags it (`turn_closed=False, answer_tokens=0`). Issue #9 tracks budget-policy design.
- Knob response is **not a smooth dial**: block-autoregression makes output respond
  discontinuously to threshold knobs (plateaus and cliffs — issue #10 has measured sweeps).
- Raw pre-excision canvas ids are captured engine-side as of 0.3.0, not yet exposed on any
  socket (issue #11) — wanted for token-level trace analysis.
- Quantized loading for consumer cards **below ~24 GB VRAM** is unresolved (issues #4, #15).
  GGUF/llama.cpp is the most promising remaining direction, parked pending a design bridge
  for live-view/trace instrumentation.

---

## Where the design lives

| Doc | What it holds |
|-----|---------------|
| **[VISION.md](VISION.md)** | *Why it might matter* — questions tagged `[established]` / `[hypothesis]` / `[open]`. Speculative by design, cited throughout. |
| **[ROADMAP.md](ROADMAP.md)** | *Where it's headed* — engineering seam work and the liquid-phase research program. Pointer-heavy; VISION holds the *why*, `decisions/` the *decided*. |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | Contributor-facing map — how the pieces fit and why. |
| **[decisions/](decisions/)** | ADRs — *why* load-bearing choices were made. |

---

## Come explore

This is an instrument for poking at how this diffusion LLM thinks. Questions, findings,
and half-formed ideas are exactly the point. The **[Discussions](../../discussions)** tab
is open for show-and-tell (post a trace, a heatmap, a run that annealed somewhere strange)
and for ideas. See **[CONTRIBUTING.md](CONTRIBUTING.md)** for how to jump in.

Watch the process in action on YouTube: [**@reflectiveattention**](https://youtube.com/@reflectiveattention).

---

## License

GPL-3.0 (matching ComfyUI core). LICENSE file lands with registry publication.
