# ComfyUI-DiffusionGemma — agent instructions

A ComfyUI node pack exposing **DiffusionGemma** (uniform-state discrete diffusion
text generation) as an instrumentable node graph. This repo is currently
**design-only** — decision records + build plan, no working nodes yet. Start at
[`README.md`](README.md) (what + status), [`plan.md`](plan.md) (what's next), and
[`decisions/`](decisions/) (why).

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

- **Runtime access path:** HF transformers `DiffusionGemmaForBlockDiffusion` +
  `TextDiffusionStreamer` (ADR-CDG-002). GGUF/llama.cpp is a graduation-triggered
  *inference-only* backend, not the primary path.
- **Open question that can move the type design:** `mask_token=4` / `algorithm=4`
  (ADR-CDG-002) — whether DiffusionGemma leans on an absorbing `[MASK]` vs. pure
  uniform-state renoise. Resolves in Phase 3. If it uses an absorbing mask,
  ADR-CDG-001's "no MASK" claim needs a footnote and `CANVAS_STATE` may need a mask
  sentinel. **Do not build the type layer as if this is settled.**
- **Local run defaults (Q4_K_M, first run):** `max_steps=48 t=[0.4,0.8]
  entropy_bound=0.1 confidence=0.005 canvas_length=256`. Pass `-ngl 99` (+
  `-cmoe`/`--n-cpu-moe` for overflow) or MoE experts spill to CPU (24 tok/s vs.
  456 tok/s in-step parallel).
