# ADR-CDG-001 — Define native socket types instead of reusing ComfyUI's SIGMAS / LATENT

**Status**: accepted (implementation pending)
**Date**: 2026-06-30
**Related**: ADR-CDG-002 (access path), ADR-CDG-004 (drive-seam amendment;
confirms the "no MASK" claim below, 2026-07-05); supports the topology choice
in `../loose-ends.md`

---

## Context

DiffusionGemma generates text by **uniform-state discrete diffusion**: a fixed
256-token canvas of random vocabulary tokens, iteratively refined by an
entropy-bound sampler that commits the lowest-entropy positions under a budget
and re-noises the rest. There is **no sigma (noise-standard-deviation) schedule
and no latent space**. What plays the "schedule" role is a per-step
*temperature + entropy-budget* trajectory; the working state is a discrete token
canvas, not a continuous latent.

ComfyUI's entire sampling ecosystem — `KSampler`, `BasicScheduler`, the RES4LYF
pack, the exponential-integrator solver zoo — is built on the `SIGMAS` and
`LATENT` socket types, which assume continuous Gaussian diffusion. A
`BasicScheduler` emits a `SIGMAS` float tensor; a `KSampler` consumes
`(model, SIGMAS, LATENT, CONDITIONING)`. DiffusionGemma's loop has no input of
that shape.

## Decision

Define new socket types and interconnect the node family through them:

1. **`DGEMMA_MODEL`** — loaded model + processor handle.
2. **`ENTROPY_SCHEDULE`** — per-step `(temperature, entropy_budget)` trajectory
   plus the stop criterion. *Not* a sigma tensor.
3. **`CANVAS_STATE`** — a (possibly partial) canvas, for chaining/infilling.
4. **`CANVAS_TRACE`** — per-step canvas + per-slot entropy + commit set, for
   instrumentation/visualization.
5. **`CONSTRAINTS`** — pinned tokens / masked regions fed to the sampler.

The pack deliberately does **not** plug into the `KSampler` family. Its nodes
connect to each other.

## Rationale

### Positive Consequences
- Payloads mean what they say. An entropy budget is never silently mangled by a
  `SIGMAS`-math node (e.g. multiply-sigmas).
- The node graph reflects the real substrate, which makes the pack legible and
  teachable rather than a disguise.
- Enables entropy-native manipulation nodes that have no sigma analog — e.g. a
  tangent-shaped *entropy-budget* schedule (the honest reincarnation of
  `bong_tangent`).

### Negative Consequences
- The pack does **not** compose with the mature image-side scheduler/sampler
  ecosystem. We forgo reuse of all those nodes.
- More socket types to define, document, and keep stable.
- Isolation is real: only our nodes talk to our nodes. (This is correct — the
  image ecosystem genuinely cannot process an entropy budget — but it is a cost.)

## Alternatives Considered

### Option A: Reuse `SIGMAS` / `LATENT` (the "lying sigmas" path)

Emit the entropy/temperature schedule disguised as a `SIGMAS` tensor so the pack
plugs into existing scheduler nodes and KSampler.

**Why rejected:** The values don't denote noise standard deviation. Any
downstream sigma-aware node would corrupt them, and the disguise would mislead
every reader of every workflow. RES4LYF can honestly reuse these types because
it *is* genuinely sigma/latent-based; this pack is not. Reusing them here would
be a literal instance of the trap RES4LYF jokingly named ("lying sigmas"), but
unintentional and load-bearing.

## Resolution Note (2026-07-05)

The "no MASK" characterization above — random-vocabulary renoise, not an
absorbing `[MASK]` token — was carried as an assumption pending ADR-CDG-002's
`mask_token=4` open question. That question is now resolved documentarily
(ADR-CDG-002's Open Questions, corroborated in ADR-CDG-004): DiffusionGemma
runs pure uniform-state renoise, `mask_token_id=None`. This confirms `CANVAS_STATE`
needs no mask sentinel value. Empirical corroboration lands in Phase 3;
until then this is a documentary confirmation, not yet a runtime observation.

## Open Questions

- [ ] Should we later ship an optional `CANVAS_STATE → IMAGE` adapter node so the
      trace heatmap can compose with image-side preview/compositing nodes?
      **Resolution trigger:** revisit in Phase 4+ if visualization demand wants it.

## Supersession Relationships

**Supersedes:** none
**Superseded by:** TBD
