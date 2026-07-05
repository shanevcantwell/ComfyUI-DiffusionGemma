# ADR-CDG-005 — CANVAS_STATE is a complete resumable save-state, not a display snapshot

**Status**: accepted (implementation pending)
**Date**: 2026-07-05
**Related**: ADR-CDG-001 (socket types — this ADR is the full contract for the
`CANVAS_STATE` type that ADR-CDG-001 introduced but left partially specified),
ADR-CDG-004 (drive seam — the pipeline/scheduler/generator objects this ADR's
state fields are drawn from)

---

## Context

ADR-CDG-001 defined `CANVAS_STATE` as "a (possibly partial) canvas, for
chaining/infilling" without settling what fields it actually carries. Two
questions force the issue now: what does a downstream node need to *resume* a
denoise trajectory from an arbitrary step, and what does it need to *branch*
it (change something at step k and diverge)? Both require a "sufficient
statistic" for the trajectory — enough state that resuming from it is
indistinguishable from having run continuously — not just enough state to
render the canvas as text.

This is a domain luxury unavailable to image diffusion or video: per-step
state here is genuinely small. The canvas is `gen_length` integers, any masks
are boolean vectors, and RNG state is on the order of 5 KB. There is no
MPEG-style keyframe-vs-delta trade-off to make — every step can afford to be
a keyframe. That changes what "the state" even means: instead of picking a
minimal delta encoding, the question is just which fields belong in the
keyframe.

Three sources of state were checked against the actual drive seam
(ADR-CDG-004, installed diffusers 0.39.0) rather than assumed:

- **Scheduler commit state is not uniform across schedulers.**
  `BlockRefinementScheduler` holds persistent `self._committed`
  (`scheduling_block_refinement.py:81`), a boolean mask that only accumulates
  within a block and is reset once at `step_index == 0`
  (`scheduling_block_refinement.py:266`). Its commit quota is keyed on an
  internal counter, `steps_done = step_index + 1`
  (`scheduling_block_refinement.py:272-273`), used to compute how many more
  positions are "owed" this step. A resumed run that doesn't restore
  `_committed` re-derives quota from a fresh `steps_done` sequence starting
  over — silent quota desync, not a crash. By contrast,
  `EntropyBoundScheduler` holds no such field at all (checked: only
  `num_inference_steps`/`timesteps` on `self`,
  `scheduling_entropy_bound.py:81-91`) — "committed" there is a per-step
  reading recomputed from confidence, nothing to restore.
- **A single `torch.Generator` is threaded through the entire trajectory.**
  The pipeline's canvas init (`pipeline_diffusion_gemma.py:346-348`) and every
  scheduler `.step()` call (e.g. `pipeline_diffusion_gemma.py:399`) consume
  the same `generator` object. Its state at any point is what makes the *rest*
  of an untouched run deterministic. Capturing it is what distinguishes a
  true rewind (replay bit-for-bit from step k onward) from a rerun (replay
  the same knobs but a different random draw from k onward) — those are
  different operations and the save-state must be able to do the first.
- **KV cache is deliberately excluded.** It is recomputable from the
  committed token prefix with one prefill pass — the only cost a resume pays
  is that one forward, the "run to the keyframe" cost MPEG keyframing exists
  to amortize. Carrying it in the payload would bloat `CANVAS_STATE` by
  orders of magnitude for a cost that's cheap to pay on demand instead.

## Decision

`CANVAS_STATE` is a sufficient statistic for resuming or branching a denoise
trajectory at any step — emulator-save-state semantics, not display-snapshot
semantics. Every field answers "what would be lost if this were omitted, and
what does that loss look like on resume":

1. **Canvas token ids** — the `gen_length`-length integer tensor. Without
   this there's nothing to resume.
2. **`(step_idx, t, temperature)`** — the position in the schedule. Needed
   because per-step temperature is a function of position
   (`scheduling_entropy_bound.py:153-155`), not carried state; a resume needs
   to know where in the anneal it is.
3. **Scheduler identity + config + scheduler commit state.** Identity and
   config (which scheduler class, what its `__init__` args were) plus
   whatever persistent state that class holds — `_committed` for
   `BlockRefinementScheduler`, nothing for `EntropyBoundScheduler`/
   `DiscreteDDIMScheduler` (both stateless per-step). Omitting commit state
   for a stateful scheduler is a **lying save-state**: it loads, it runs, and
   it silently diverges from what continuing the original run would have
   produced (ADR-CDG-001's addendum on scheduler-relative commit semantics
   is the same principle applied to the trace payload; this is its resume-time
   twin).
4. **`torch.Generator` state** — captured via the generator's own state dict,
   not reseeded from a stored integer seed (a fresh seed reproduces the
   *distribution*, not the *exact draw sequence* already consumed by prior
   steps). This is what makes "rewind" a real operation distinct from
   "rerun."

**Deliberately excluded: KV cache.** Recomputable from the committed prefix
with one prefill pass; not part of the sufficient statistic, just an
optimization payload that isn't worth the size cost here.

Phase 1 ships a subset of these fields as a validity readout (`converged`,
`committed_fraction`, `steps_used` — see `plan.md` P1), not the full resumable
contract. That is a scoping choice, not a contradiction: the fields grow
additively as phases land. This ADR fixes the contract's *direction* — what a
complete `CANVAS_STATE` looks like when the resumable/branchable feature set
is built — not P1's scope.

## Rationale

### Positive Consequences
- **Rewind to any step** becomes a real, cheap operation — the domain
  affords a keyframe-per-step economy that ruled-out approaches (image/video
  diffusion) don't have.
- **Branch-on-intervention** falls out for free: change a pin or a knob at
  step k, resume from the saved state, and the trajectory diverges from
  exactly that point — no separate "branching" mechanism needed beyond the
  save-state being complete.
- Enables a future `DGemmaStepSampler` node (`CANVAS_STATE` in, `CANVAS_STATE`
  out) that is **loop-mechanism-agnostic**: manual requeue-driven stepping,
  a third-party ComfyUI loop pack, or an eventual own For/While pair are all
  equally valid drivers of the same node, because the state *is* the
  contract, not the loop that produces it (IDENTITY⊥ENVELOPE applied to
  iteration — the envelope that steps the loop is free to vary because the
  identity that crosses each step boundary is fixed).

### Negative Consequences
- `CANVAS_STATE` grows a scheduler-identity + scheduler-config + scheduler-state
  sub-payload that a naive "just the canvas" design wouldn't need — more
  surface to keep honest per scheduler class.
- Every scheduler this pack supports needs its persistent-state fields
  enumerated explicitly (what to save, what's stateless) — a new scheduler
  added later must be checked against this question, not assumed compatible.

## Alternatives Considered

### Option A: Display-snapshot semantics (canvas + step index only)

Cheap, matches what P1's validity readout already needs, and is enough to
*render* a trace frame.

**Why rejected:** Not enough to *resume* — a display snapshot answers "what
does this step look like," not "what would continuing from here produce." It
would silently work for `EntropyBoundScheduler` (stateless) and silently
misbehave for `BlockRefinementScheduler` (stateful commit ratchet), which is
exactly the kind of scheduler-relative footgun ADR-CDG-001's addendum already
flagged for the trace payload. Same failure mode, resume side.

### Option B: Carry the KV cache too, for a truly zero-cost resume

Would make resume free of even the one prefill pass.

**Why rejected:** The prefill-pass cost is cheap and bounded (one forward
over the committed prefix); the KV cache size cost is not bounded the same
way and scales with context length. Not worth the payload bloat for a cost
this domain can already afford to re-pay on demand.

## Open Questions

- [ ] Does every future scheduler this pack adds need its own persistent-state
      audit before `CANVAS_STATE` can honestly claim completeness for it?
      **Resolution trigger:** whenever a new scheduler is wired in (Phase 4+),
      repeat the `_committed`-style check done here before assuming the
      existing `CANVAS_STATE` shape covers it.
- [ ] External stepping (a future `DGemmaStepSampler`) drives
      `BlockRefinementScheduler`'s internal `steps_done` counter
      (`scheduling_block_refinement.py:272-273`) from outside its own loop —
      does graph-side stepping desync that quota even with `_committed`
      faithfully restored? **Resolution trigger:** `EntropyBoundScheduler`
      (stateless, no quota counter) is the first target for external
      stepping precisely to sidestep this; revisit `BlockRefinement` support
      only after that's proven.

## Supersession Relationships

**Supersedes:** none (fills in a contract ADR-CDG-001 introduced but left
open)
**Superseded by:** TBD

## References

- `scheduling_block_refinement.py:81,266,272-273,280-287` (installed diffusers
  0.39.0, `.venv/lib/python3.12/site-packages/diffusers/schedulers/`)
- `scheduling_entropy_bound.py:81-91,153-155` (same package)
- `pipeline_diffusion_gemma.py:346-348,399` (same package)
