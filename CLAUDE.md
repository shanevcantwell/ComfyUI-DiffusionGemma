# ComfyUI-DiffusionGemma — agent instructions

A ComfyUI node pack exposing **DiffusionGemma** (uniform-state discrete diffusion
text generation) as an instrumentable node graph. **Working as of 2026-07-05**:
phases P0–P3 closed (loader, sampler with full knob surface + thinking toggle,
live per-step view, `DGemmaTrace` analysis) — each phase's evidence is recorded
in `plan.md`. Start at [`README.md`](README.md) (what + status),
[`plan.md`](plan.md) (what's next), and [`decisions/`](decisions/) (why).

## Canonical path

`/srv/dev/shanevcantwell/ComfyUI-DiffusionGemma` (a sibling repo under the
shanevcantwell parent). Remote: `github.com/shanevcantwell/ComfyUI-DiffusionGemma`
(public). **Never `find`/`grep` from `~`** — roots in play are this repo,
`/srv/dev/shanevcantwell/`, and the sibling `harness-tools`.

## Doctrine — included by reference, not duplicated

This repo inherits the ecosystem's ground-physics doctrine. Do **not** copy it
here (opinion locality — a substrate detail promoted into a principle is how tools
degrade). Resolve the pointers:

- **Ground physics (the invariants + why):**
  `../harness-tools/docs/ground-physics/GROUND_PHYSICS.md`
- **Code Constitution (the enforceable rule set + PR checklist):**
  `../harness-tools/docs/ground-physics/CODE_CONSTITUTION.md`
- **Operating constitution (every agent):** `~/.claude/CLAUDE.md`.

### How the doctrine binds *this* repo specifically

ADR-CDG-001 (native socket types, reject "lying sigmas") is a **native instance of
`EMIT-CANONICAL / PARSE-AT-THE-DOOR`**: an entropy budget disguised as a `SIGMAS`
tensor is exactly the trust-and-degrade failure the doctrine forbids. Payloads
mean what they say. When you add a socket type or a node boundary, run the
Code Constitution's PR checklist against it — the data-plane questions apply to
`ENTROPY_SCHEDULE` / `CANVAS_STATE` / `CONSTRAINTS` the same way they apply to a
model identity.

### Greenfield adaptation — read this before writing an ADR

The Code Constitution's hard-never-do *"never invent an invariant without an
observed violation to anchor it"* carries an explicit **greenfield exception**
(`harness-tools#18`): a new project with no running history anchors its invariants
to **anticipated failure modes**, not observed violations. That is what the ADRs
here already do — their "Negative Consequences" and "Open Questions" sections are
anticipated-failure reasoning. Keep that discipline: every invariant this repo
introduces names the failure it prevents, even before that failure has occurred.

## ADR convention

Decision records live in [`decisions/`](decisions/), numbered `NNNN-slug.md`, with
[`decisions/README.md`](decisions/README.md) as the index table. Format SSoT is
the `writing-adrs` skill (`../harness-tools/internal-skills/writing-adrs`). Record
a decision when it's hard to reverse, surprising without context, and the result
of a real trade-off — otherwise it's a [`loose-ends.md`](loose-ends.md) entry.

## Grounded facts (don't re-derive)

- **Runtime access path (hybrid — ADR-CDG-004 amends ADR-CDG-002):** **load**
  via transformers `DiffusionGemmaForBlockDiffusion.from_pretrained()`
  (unchanged); **drive** via `diffusers.DiffusionGemmaPipeline` + swappable
  scheduler (default `EntropyBoundScheduler`) instead of raw `.generate()` +
  `TextDiffusionStreamer` — needs `diffusers>=0.39.0` alongside
  `transformers==5.13.0`. GGUF/llama.cpp is still a graduation-triggered
  *inference-only* backend, not the primary path.
- **Type-design question resolved:** pure uniform-state renoise, **no
  absorbing mask** (`mask_token_id=None` for this model). ADR-CDG-001's
  "no MASK" claim is confirmed — no `CANVAS_STATE` mask sentinel needed. The
  type layer may be built as settled on this point; see ADR-CDG-002's
  resolved open question and ADR-CDG-004 for sourcing.
- **Weights:** `google/diffusiongemma-26B-A4B-it`
  (https://huggingface.co/google/diffusiongemma-26B-A4B-it), ~53.6 GB
  safetensors (bf16), ungated (Apache-2.0). Model card notes ≥60 GB GPU memory
  for a bf16 load — the 48 GB RTX-8000 dev box needs quantized and/or
  offloaded loading, not a full bf16 load.
- **Local run defaults (Q4_K_M, first run):** `max_steps=48 t=[0.4,0.8]
  entropy_bound=0.1 confidence=0.005 canvas_length=256`. Pass `-ngl 99` (+
  `-cmoe`/`--n-cpu-moe` for overflow) or MoE experts spill to CPU (24 tok/s vs.
  456 tok/s in-step parallel).
- **Seeding/renoise:** the diffusers pipeline has no init-canvas param (canvas
  is a hardcoded `torch.randint`); injected text is at best a soft prior, since
  scheduler commit state resets or is absent at step 0. Hard pinning is a
  per-step callback re-assertion, not a scheduler feature. transformers'
  `.generate(decoder_input_ids=...)` seeds only the *first* canvas
  (`generation_diffusion_gemma.py:987`). See `loose-ends.md` (DGemmaRenoise).
- **Per-step LIVE display and node output are different mechanisms.**
  ComfyUI node outputs exist only on return, so live per-step frames ride
  `PromptServer.instance.send_sync` custom events (thread-safe by
  construction, `server.py:1374-1376`) plus a `WEB_DIRECTORY`-registered JS
  extension (`nodes.py:2269-2272`) — never `ProgressBar`'s `preview=` slot,
  which is image-typed downstream (`server.py:1293-1301`) and throws on
  text. `CANVAS_STATE` is a resumable save-state (ADR-CDG-005), not a
  display snapshot — canvas + schedule position + scheduler identity/config/
  commit-state + RNG generator state, KV cache deliberately excluded
  (recomputable via one prefill pass).
