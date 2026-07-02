# ADR-CDG-003 — Split thin node adapters from a ComfyUI-agnostic engine

**Status**: accepted (implementation pending)
**Date**: 2026-06-30
**Related**: ADR-CDG-001 (socket types are the engine's public face), ADR-CDG-002 (access path)

---

## Context

A ComfyUI node is deliberately anemic: an `INPUT_TYPES` classmethod, a
`RETURN_TYPES` tuple, a `FUNCTION` name, a `CATEGORY`, and a method that takes
inputs as kwargs and returns a tuple. There is no signature architecture to
design, and most small nodepacks correctly inline their logic straight into the
node method.

This pack cannot. Its entire reason to exist is **per-step instrumentation** —
capturing the canvas, per-slot entropy, and commit set at each denoising step
(Phase 3). That loop must be developed, run, and tested from a bare script and a
`pytest`, with **no ComfyUI process alive**. You cannot iterate on a denoising
loop from inside a node function. The instrumentation goal — not taste — forces
a seam.

## Decision

Two packages:

- **`nodes/`** — thin adapters. Each node is declarations
  (`INPUT_TYPES`/`RETURN_TYPES`/`FUNCTION`/`CATEGORY`) plus a method that does
  exactly three moves: **unpack kwargs → call one `dgemma.*` function → wrap the
  result in a tuple.** No logic.
- **`dgemma/`** — the engine. Plain Python, imports and runs with zero ComfyUI
  present. Owns the model, the types, the schedule builders, the loop, and the
  sampling policy.

**Invariant (the one-line test):** if a node method body ever contains a `for`
loop over denoising steps, logic has leaked into the wrong layer. Nodes
translate; `dgemma/` computes.

The ADR-CDG-001 socket types are simply the thin public face of the engine's
dataclasses — a custom ComfyUI socket type is just a string ComfyUI matches by
equality, and the object riding it is the corresponding `dgemma/types.py`
dataclass, passed through untouched.

## Rationale

### Positive Consequences
- The denoising loop is testable headless — the precondition for Phase 3 existing
  at all.
- Node code stays trivial and uniform, so the node layer is nearly reviewable at
  a glance.
- Socket types stay honest (ADR-CDG-001) because they wrap engine objects rather
  than smuggling logic.

### Negative Consequences
- Indirection the idiomatic inline-everything pattern avoids.
- Two import surfaces to keep coherent.

## Alternatives Considered

### Option A: Inline logic in node methods (the idiomatic small-pack pattern)

**Why rejected:** It makes the denoising loop impossible to run or test outside a
live ComfyUI process, which blocks the instrumentation work that is the whole
point of the pack. Fine for packs whose logic is a single transform; wrong for
one whose value is watching a loop.

## Open Questions

- [ ] Do the `DGemmaOptions_*` nodes mutate an options object threaded through the
      sampler, or return a fresh merged one? **Resolution trigger:** decide when
      the options chain is built in Phase 5; prefer immutable-merge if it doesn't
      hurt ergonomics.

## Supersession Relationships

**Supersedes:** none
**Superseded by:** TBD
