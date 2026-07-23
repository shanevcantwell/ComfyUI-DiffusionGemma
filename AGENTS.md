# AGENTS.md — ComfyUI-DiffusionGemma

This file is the **agent-facing contract** for `ComfyUI-DiffusionGemma`. It tells AI agents (Copilot, Perplexity, Claude, Cursor, etc.) how to reason about this project: what it does, its architecture, available interfaces, and where decisions live.

---

## What this project is

An implementation of Google's DiffusionGemma model — discrete diffusion text generation with **MCP as the canonical surface**. The core (`dgemma/`) exposes two functions: `load_model()` and `run_diffusion()`. MCP wraps them into a complete tool interface. **ComfyUI consumes MCP** — it is one consumer among many, not a privileged path.

```
┌─────────────── Core ───────────────┐
│  dgemma/                          │
│    load_model() + run_diffusion() │
└──────────────┬────────────────────┘
               │ wrapped by
     ┌─────────▼──────────┐
     │   MCP (base)       │
     │  surfaces/mcp/     │
     │                    │
     │  load_model        │
     │  model_status      │
     │  generate          │
     │  cancel_run        │
     └─────────┬──────────┘
               │ consumed by
    ┌──────────┼───────────────┐
    ▼          ▼               ▼
 ComfyUI   any agent      CLI / web
```

**Any AI agent can do everything the ComfyUI surface does** — just call the MCP tools. The ComfyUI node pack is one consumer of MCP, not a separate path to the core.

---

## Architecture (MCP as the canonical surface)

The topology is hierarchical: core → MCP → consumers. New consumers wrap MCP tools, not core functions directly.

**Rules agents must respect:**

1. **Core is the contract.** All denoising logic lives in `dgemma/`. No surface holds a for-loop over denoising steps.
2. **MCP is the base surface.** ComfyUI consumes MCP — it is one consumer among many, not a privileged path. Any agent calling MCP tools has equal access to all capability.
3. **Stateless core.** Only the model load persists (~50 GB bf16 or ~30 GB INT4). Every run constructs a fresh scheduler, canvas, and run state. Two identical calls yield identical results.
4. **Analysis is downstream.** `consumers/analysis.py` parses already-emitted `CanvasTrace`. It does not live in `dgemma/`'s import graph.
5. **Protected load-time mechanics (Do not refactor).** The `load_model` INT4 (`autoround`) path relies on three structural patches to prevent consumer hardware OOMs and CPU bottlenecks. **Do not remove or bypass these:**
   - `auto-round` regex pre-compilation (prevents O(N×M) CPU pinning during load).
   - KV-cache warmup bypass (prevents 46GB bf16 buffer pre-allocation on INT4 loads).
   - Tied-weight finalization guard (catches `AttributeError` for quantized embeddings).

---

## MCP surface — available tools

Run the server: `python -m surfaces.mcp.server` (requires `pip install -e '.[mcp]'`)

### `load_model`
Load DiffusionGemma into memory. Must specify model and quant explicitly — no silent defaults.

**Parameters:**
- `model` (string, required) — HuggingFace repo ID or local path; default: `google/diffusiongemma-26B-A4B-it`
- `quant` (string, required) — `'none'` for bf16 (~50 GB VRAM), `'autoround'` for INT4 pre-quantized (~30 GB VRAM)

**Returns:** model loaded status, device placement, dtype.

### `model_status`
Query the current state of a loaded model without triggering generation.

**Returns:** whether a model is loaded, its quantization mode, device, memory footprint.

### `generate`
Run discrete diffusion text generation with full per-step telemetry.

**Parameters:**
- `prompt` (string, required) — input text prompt
- `num_inference_steps` (int, default 48) — denoising steps
- `t_min`, `t_max` (float, default 0.4 / 0.8) — temperature schedule endpoints
- `entropy_bound` (float, default 0.1) — per-step acceptance budget in nats
- `confidence` (float, default 0.005) — early-stop threshold
- `gen_length` (int, default 256) — canvas size in tokens
- `seed` (int, optional) — for deterministic reproducibility
- `thinking` (bool, default false) — inject `<|think|>` control token
- `run_id` (string, optional) — enables mid-run cancellation via `cancel_run`

**Advanced parameters (widened doors):**
- `constraints` — pinned positions and constraint rules
- `control_signals` — CV/LFO/mod matrix bindings for dynamic knob modulation
- `capture` — frame capture specification (which steps to record)

**Returns:** generated text, CanvasState (convergence metrics), CanvasTrace (per-step telemetry).

### `cancel_run`
Cancel an in-progress generation by its `run_id`.

---

## How DiffusionGemma works (for agent reasoning)

DiffusionGemma generates text through **uniform-state discrete diffusion**:

1. **Melt state:** Every canvas position starts as maximum entropy noise — a uniform draw from 262,144 tokens (≈18 bits per position). This renders as polyglot flickering (Katakana, Bengali, CJK) before the model narrows focus.
2. **Annealing:** A temperature schedule cools over `num_inference_steps`. Temperature divides logits in softmax, sharpening distributions and lowering per-position entropy.
3. **Freezing:** The EntropyBoundScheduler accepts only positions whose entropy falls below `entropy_bound` (default 0.1 nats). Unaccepted positions are re-noised each step — this is the model's native self-correction mechanism.
4. **Remelt:** A committed token can remelt if context shifts push its entropy back above budget. This gives uniform-state diffusion dynamic revision capability that masked diffusion lacks.

**Key difference from autoregressive models:** AR generates left-to-right, one token at a time, no revision. DiffusionGemma negotiates all positions simultaneously and allows mid-generation correction.

---

## Where decisions live

| Document | Purpose |
|----------|---------|
| `README.md` | User-facing: install, hardware, nodes, knobs, limitations |
| `ARCHITECTURE.md` | Agent-facing: layering rules, enforcement surfaces, conformance table |
| `VISION.md` | Research questions tagged `[established]` / `[hypothesis]` / `[open]` |
| `ROADMAP.md` | Forward view: engineering seams + liquid-phase research program |
| `decisions/adr-cdg-*.md` | ADRs — why load-bearing choices were made (18 documents) |

**Key ADRs for agents:**
- **ADR-CDG-008** — MCP-center topology, multi-surface architecture
- **ADR-CDG-004** — Drive seam: `run_diffusion` single-entry contract
- **ADR-CDG-015** — Input embedding seam (next frontier for liquid-state decoding)

---

## Testing the seam

The core/surface boundary is enforced by subprocess tests:

```bash
# Core imports no surface (ComfyUI or MCP)
python -m pytest tests/test_seam.py

# MCP surface doesn't leak into core
python -m pytest tests/test_mcp_surface_seam.py

# ComfyUI dual-context import gate survives across surfaces
python -m pytest tests/test_comfyui_loader_context.py
```

---

## Quick reference — knobs and units

| Knob | What it controls | Units | Default |
|------|-----------------|-------|---------|
| `t_min` / `t_max` | Temperature schedule endpoints | dimensionless (temperatures) | 0.4 / 0.8 |
| `entropy_bound` | Per-step acceptance budget — positions below this freeze | nats | 0.1 |
| `confidence` | Early-stop threshold | probability | 0.005 |
| `num_inference_steps` | Denoising steps (48 = ~2.3s/step on RTX-8000) | count | 48 |
| `gen_length` | Canvas size in tokens | tokens | 256 |

Full provenance: [`KNOB_DOCS`](dgemma/loop.py) in `dgemma/loop.py`.

---

## Hardware requirements

| Mode | VRAM | Load time | Notes |
|------|------|-----------|-------|
| `quant='autoround'` (INT4) | ~30 GB | ~27 s | Pre-quantized checkpoint, working forward pass |
| `quant='none'` (bf16) | ~50 GB | slower | Full precision, CPU spill via ComfyUI offload |

System RAM matters more than VRAM under heavy offload. Below ~24 GB VRAM: not yet supported.

---

## External resources

- **YouTube:** [@reflectiveattention](https://youtube.com/@reflectiveattention) — process videos showing crystallization
- **Discussions:** [GitHub Discussions](../../discussions) — show-and-tell, traces, heatmaps
- **Experiments:** [`docs/experiments/`](docs/experiments/) — dated experiment records with raw data
