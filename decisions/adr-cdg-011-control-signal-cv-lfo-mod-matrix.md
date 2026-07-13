# ADR-CDG-011 — Control signals as CV/LFO: declarative payloads, units-at-binding, `scheduler.config`-only mutation

**Status**: accepted (ratified 2026-07-13, PR #43)
**Date**: 2026-07-13
**Related**: ADR-CDG-004 (drive seam — `run_diffusion`'s signature is what
this ADR widens), ADR-CDG-008 (MCP-center topology — Correction 1's
`STATELESS-CORE` posture the same-in/same-out clause enforces), ADR-CDG-010
(constraint composite — the ordered composite slot this ADR's walker shares;
§"Cross-references" below), Issue #35 (architecture review — the drafting
spec both this ADR and ADR-CDG-010 transcribe), Issue #23 (per-step control
grounding — the `scheduler.config` fresh-read mechanism, the one-shot-executor
constraint, and the CV/LFO reframe this ADR's clauses transcribe), Issue #20
(closed — the `num_inference_steps` desync mechanism this ADR's ingress
reject forecloses by design)

---

## Context

Issue #35's grounding pass (2026-07-13) resolved three open questions about
per-step control in sequence, each comment correcting the shape of the last:

1. **The mechanism already exists.** `EntropyBoundScheduler.step()` reads
   `entropy_bound`, `t_min`, `t_max` fresh from `self.config` on **every**
   call (`scheduling_entropy_bound.py:148-149,154`) — nothing is baked into
   arrays at `set_timesteps` time. A step-end callback that changes
   `pipe.scheduler.config` values changes what the *next* step's call reads
   (mechanism note, post-grounding: the write is `register_to_config`
   whole-dict replacement, not an in-place set — see Decision 4). Setting
   `t_min = t_max = T_desired` each step degenerates the anneal formula to an
   exact per-step temperature — this is the mechanism, not a proposal for a
   new one.
2. **The socket must be declarative, not a live graph value.** ComfyUI's
   executor is one-shot topological (`execution.py:774-781`): a node runs
   once, its outputs cache, then its consumers run. No mechanism lets a graph
   value flow into an in-progress sampling run. The `SIGMAS` prior art
   confirms the shape: `BasicScheduler` returns a full precomputed 1-D tensor
   (`comfy_extras/nodes_custom_sampler.py:29-38`) that the sampling loop walks
   by index; all genuine per-step control in ComfyUI core is a code-level
   closure via `model_options` (`model_patcher.py:604`), invoked from inside
   the loop, never re-entering the graph. A control-signal socket must
   therefore be a **complete precomputed schedule**, not a live value.
3. **The socket is a unitless CV/LFO signal, units declared at the binding.**
   The operator's reframe: rather than a bespoke entropy-schedule type, the
   socket is a generic per-step control-voltage waveform (sine/triangle/ramp/
   decay/sample-and-hold generators, all emitting one type). A separate
   binding stage maps signal → target knob with explicit range/polarity
   (`→ entropy_bound [0.02, 0.3]`, `→ temperature [0.4, 0.8]`). This
   supersedes the earlier `ENTROPY_SCHEDULE`-as-bespoke-type framing and
   folds the "reuse `SIGMAS`?" debate into one answer: a `SIGMAS` curve is
   just another unitless signal source, normalized to shape at the same
   explicit adapter every other signal source uses.

Separately, issue #20 (closed, verified benign today but latent) found that
`anneal_temperature` reconstructs its schedule denominator from the
user-supplied `num_inference_steps` rather than the scheduler's own effective
`num_inference_steps` after `set_timesteps` — a desync that is dormant only
because the sole scheduler in use has no corrector steps. This ADR's
`num_inference_steps` non-mutable clause forecloses the mechanism that would
turn this bug live: if `num_inference_steps` could be mutated mid-run through
the same door the walker uses for other knobs, the scheduler's cached
`predictor_steps`/`_num_timesteps` (`pipeline_diffusion_gemma.py:297,299`)
would desync from the mutation exactly as #20 describes.

## Decision

1. **The control-signal socket carries a declarative payload only —
   `control_signals=` on `run_diffusion` — never a surface-built closure or
   hook.** Same rule-7 posture as ADR-CDG-010's `constraints=`: ingress
   validates the payload (schedule length == `num_inference_steps`; control
   values within the declared binding's range; fail on unknown target) and
   turns it into an engine-built walker participant. A surface never passes
   a callable; the engine never accepts one for this purpose.

2. **A control signal is a unitless, precomputed per-step tensor — one type
   for every generator shape.** Sine, triangle, ramp, decay,
   sample-and-hold, or a `SIGMAS`-style curve normalized to shape are all the
   same socket type at this layer: a flat sequence of `num_inference_steps`
   unitless values. Step count is known before the run starts (ComfyUI's
   one-shot topological executor requires this — Grounding finding 2), so
   this is exactly the shape a declarative payload can carry.

3. **Units are declared at the binding, not carried by the signal (the CV
   principle).** A binding stage maps one control signal to one scheduler
   knob with an explicit range and polarity
   (e.g. `signal → entropy_bound, range=[0.02, 0.3]`). The binding is where
   parse-at-the-door validation happens: an out-of-declared-range value, or a
   target knob name outside the in-vocab set, is rejected at ingress, not
   silently clamped or passed through. This is `EMIT-CANONICAL /
   PARSE-AT-THE-DOOR` applied to a control value the way ADR-CDG-001 applies
   it to a socket payload: the raw signal means nothing until the binding
   gives it units, so the binding — not the generator node — is the
   enforcement point.

4. **The engine walker changes only `scheduler.config` values that `step()`
   reads fresh — applied via `register_to_config(**kwargs)` whole-dict
   replacement, never an in-place attribute set — and it never touches
   `num_inference_steps`.** Grounded directly in the #35 comment:
   `entropy_bound`, `t_min`, `t_max` are read fresh from `self.config` on
   every `step()` call, so per-step replacement of exactly those values is
   what makes per-step control possible with no scheduler change. The write
   mechanism is pinned by PR #44's gate verification against the installed
   `diffusers==0.39.0` sources: `ConfigMixin.config` is a `FrozenDict` whose
   `__setattr__`/`__setitem__` raise once frozen
   (`configuration_utils.py:77-85`); the only real write path is
   `register_to_config(**kwargs)`, which rebuilds the internal dict
   wholesale as a new `FrozenDict` (`configuration_utils.py:143-158`) — or
   an equivalent engine-owned write that `step()` reads fresh. The intent is
   unchanged (per-step values are read fresh at step time); only the
   mechanism is now stated honestly.
   `num_inference_steps` is excluded by design and rejected at ingress if a
   binding names it as a target: mutating it mid-run would desync the
   scheduler's cached `predictor_steps`/`_num_timesteps`
   (`pipeline_diffusion_gemma.py:297,299`) from the value `step()`'s anneal
   formula divides by — issue #20's exact mechanism, currently dormant only
   because no corrector scheduler is in use. This clause keeps it dormant by
   construction rather than by accident of today's scheduler choice.

5. **`t_min = t_max = v` is the exact-per-step-temperature mechanism.** No
   new scheduler parameter is needed for an exact (non-annealed) per-step
   temperature: binding a control signal to both `t_min` and `t_max` with the
   same target value each step degenerates
   `anneal_temperature`'s formula (`t_min + (t_max - t_min) * t`) to that
   exact value, since `t_max - t_min = 0`. This is recorded as the sanctioned
   way to get exact temperature control — not a workaround, the mechanism.

6. **The walker prepares the next step; capture records the finished step.**
   The walker's `scheduler.config` mutation for step `k+1` happens after step
   `k`'s forward pass and capture have completed, so a captured frame's
   telemetry always reflects the config values that produced *that* frame,
   never the next one. This composes with ADR-CDG-010's composite ordering
   (capture before any canvas-writer): the walker is a config-mutator, not a
   canvas-writer, so it does not need a position in that ADR's β-rebuild/pin
   ordering — but it must still run after capture reads the current step's
   effective knobs, or capture would report the *next* step's config as if it
   were the current step's.

7. **Effective-knob telemetry rides the frame — the values the scheduler
   actually read, not the binding's static target.** Per #23's grounding,
   the trace must record what `step()` actually consumed each call, not the
   user-authored control-signal curve. This is the same honesty requirement
   ADR-CDG-001's addendum states for commit semantics, applied to control
   values: a `DiffusionFrame`'s `entropy_bound`/`t_min`/`t_max` (or
   `temperature`, already present per `dgemma/types.py`) must be read off the
   scheduler post-mutation, at the same callback point the walker just wrote
   through, not reconstructed from the binding's declared curve.

8. **Same-in/same-out is the statelessness enforcement (F5).** Two identical
   `run_diffusion` calls — same prompt, same seed, same `control_signals=`
   schedule, one loaded model — must yield identical effective-knob
   telemetry. This is `STATELESS-CORE` (ARCHITECTURE.md rule 6) applied to
   the walker and its config mutations specifically: the walker's per-run
   state (which step index it is on, what it last wrote) must not survive
   past the call that created it, exactly parallel to ADR-CDG-010's pin-state
   clause. ADR-CDG-008's MCP Phase-2 state manager must never cache a
   scheduler across calls — a cached scheduler carrying a prior run's
   mutated `config` forward is the same class of bug as the observed
   25-vs-29 heatmap frame-count mismatch (a cached scheduler's stale
   dimensions), just on `config` values instead of frame count.

## Rationale

### Positive Consequences

- **No vendoring required.** All three grounding passes (#20-mechanism read,
  #23, #35) confirm the live-mutation mechanism already exists in the
  installed `diffusers` 0.39.0 `EntropyBoundScheduler` — this ADR names an
  ingress/binding layer around an existing capability, not new scheduler
  code.
- **One signal type serves every generator shape.** Because units live at
  the binding and not the signal, a sine LFO, a decay envelope, and a
  `SIGMAS`-shaped curve are the same socket — new generator nodes are new
  producers of the one type, not new socket types requiring new consumer
  wiring.
- **`num_inference_steps` desync (#20) is foreclosed by construction, not by
  luck.** Today's dormancy depends on nobody using a corrector scheduler;
  excluding `num_inference_steps` from the walker's mutable-target vocabulary
  at ingress means the desync mechanism has no path to fire even after a
  corrector scheduler is adopted.
- **Telemetry honesty is testable, not just asserted.** Recording the
  scheduler's actually-read values (clause 7) rather than the binding's
  static curve means a walker bug that silently fails to write through is
  visible in the trace, rather than papered over by a trace that just
  echoes what the user asked for.

### Negative Consequences

- **A precomputed-only signal cannot express a responsive/adaptive control**
  (e.g. temperature driven by the live entropy field, an envelope-follower/
  sidechain shape). ComfyUI's one-shot topological executor makes this
  unreachable as a socket — the operator's grounding names this explicitly
  as engine-only, closure-territory future work, never a socket. This ADR
  does not solve that case; it is out of scope by the same execution-model
  constraint that shapes clause 2.
- **Binding-stage validation adds a real ingress surface to get right.**
  Range/polarity checking per target knob means every new bindable knob
  needs its declared range recorded somewhere central — an omission there is
  a silent-pass-through risk, not a crash, so the binding registry itself
  needs the same "fail on unknown" discipline the sockets get.
- **The walker and ADR-CDG-010's constraint composite must be reasoned about
  together even though they are separate ADRs.** A reader who only reads one
  of the two could miss that the walker's config mutation and the
  constraint composite's canvas writes are ordered relative to each other
  (clause 6); the cross-reference sections in both ADRs exist specifically
  to prevent that gap.

## Enforcement surfaces (per clause)

| Clause | Invariant | Enforcement surface | Status |
|---|---|---|---|
| 1 — declarative payloads only | `run_diffusion(control_signals=...)` accepts a payload, not a callable | Ingress type/shape validation (schedule length == steps; fail on unknown) | `NOT-YET-IMPLEMENTED` — ADR-CDG-010/011 ingress clause, ARCHITECTURE.md enforcement-surface table |
| 1 — schedule length == steps | a control signal shorter/longer than `num_inference_steps` is rejected, not truncated/padded | Ingress length check at `run_diffusion` entry | `NOT-YET-IMPLEMENTED` — #35 declarative-payload ingress clause |
| 3 — units at binding, in-range | an out-of-declared-range control value, or an unknown target-knob name, is rejected at ingress | Binding-stage validation (range/polarity check per target; fail on unknown) | `NOT-YET-IMPLEMENTED` — ADR-CDG-011 binding clause |
| 4 — `scheduler.config`-only mutation, `num_inference_steps` excluded | the walker's mutable-target vocabulary never includes `num_inference_steps`; an attempt to bind it is rejected at ingress | Ingress reject test naming `num_inference_steps` explicitly, regression-anchored to #20 | `NOT-YET-IMPLEMENTED` — #35 R5, ADR-CDG-011; regression coverage for #20 |
| 5 — `t_min=t_max=v` exact-temperature mechanism | binding one signal to both `t_min` and `t_max` produces an exact per-step temperature with no anneal drift | Unit test over `anneal_temperature`/`EntropyBoundScheduler.step()` asserting `temperature == v` when `t_min == t_max == v` | `NOT-YET-IMPLEMENTED` — rides R1's composite work |
| 6 — walker prepares next step, capture records finished step | a captured frame's telemetry reflects the config that produced it, never the next step's | Ordered-composite test (shared fixture, R4) asserting capture reads pre-walker-mutation state each step | `NOT-YET-IMPLEMENTED` — #35 R1 (over R4's fixture); shared with ADR-CDG-010 Decision 3 |
| 7 — effective-knob telemetry | `DiffusionFrame` fields reflect scheduler-read values, not the binding's static curve | Telemetry-honesty test: mutate `scheduler.config` via a fake walker, assert the captured frame matches the mutated (not original) value | `NOT-YET-IMPLEMENTED` — #35 R6 (rides `DiffusionFrame` extension discipline) |
| 8 — same-in/same-out statelessness (F5) | two identical `run_diffusion(control_signals=...)` calls on one loaded model yield identical effective-knob telemetry | Same-in/same-out test on one loaded model (shared with ADR-CDG-010's pin-state clause) | `NOT-YET-IMPLEMENTED` — #35 R5/F5; ADR-CDG-008 Correction 1 — CDG-008 Phase-2 MCP state manager must never cache a scheduler |

## Alternatives Considered

### Option A: A bespoke `ENTROPY_SCHEDULE` type (temperature + entropy-budget pair, purpose-built)

**Why rejected:** This was the framing before the operator's CV/LFO reframe
(#23's third comment). A purpose-built pair-type would need its own
generator-node family (one set of sine/triangle/decay nodes per target pair)
and would re-litigate the `SIGMAS`-reuse question every time a new bindable
knob appeared. The CV/LFO reframe collapses this: one unitless signal type,
N generator nodes, and a binding stage that declares units per target —
adding a new bindable knob is a new binding entry, not a new socket type or a
new generator-node family.

### Option B: Reuse ComfyUI's `SIGMAS` type directly for control signals

**Why rejected:** Directly reusing `SIGMAS` would either mislabel a unitless
control curve as a noise-schedule tensor (ADR-CDG-001's "lying sigmas"
prohibition) or require every consumer to know it might secretly not be
sigmas. The CV/LFO framing resolves this without reuse: a `SIGMAS`-shaped
curve is welcome as *one source* feeding the same unitless signal type every
other generator produces, normalized to shape at the binding — the honest
version of the reuse impulse, not the disguise ADR-CDG-001 rejected.

### Option C: Let the walker mutate any scheduler attribute, including `num_inference_steps`, with no exclusion list

**Why rejected:** Issue #20 is closed as benign-today/latent precisely
because no live path currently lets `num_inference_steps` diverge from the
scheduler's own effective step count mid-run. An unrestricted walker would
open exactly that path the moment a corrector scheduler is adopted,
resurrecting a closed bug through the very feature meant to add capability.
Excluding it at ingress costs one extra check and forecloses a whole bug
class permanently.

### Option D: Support responsive/adaptive control signals (envelope-follower/sidechain) as a socket

**Why rejected:** ComfyUI's executor is one-shot topological — verified
directly against the installed checkout (`execution.py:774-781`) — with no
mechanism for a live value to flow back into an in-progress run. A
responsive signal needs the current run's state (e.g. live entropy) to
decide its next value, which only an engine-internal closure can read. This
is named as future, engine-only work, not a rejected-forever idea: it simply
cannot be a **socket** under today's execution model.

## Open Questions

- [ ] **Where does the binding-stage's per-target range/polarity table
      live?** This ADR requires units-at-binding and in-vocab rejection but
      does not name the module that owns the target-knob registry.
      **Resolution trigger:** settle alongside R2's socket-type mint module
      — the binding registry is plausibly a sibling table in the same mint
      module, but this ADR does not decide that placement.
- [ ] **Exact `control_signals=` payload shape (single signal vs a named
      dict of signal→binding pairs).** This ADR fixes the semantics (unitless
      precomputed signal, units at binding) but not the wire shape.
      **Resolution trigger:** settle during R1/R2 implementation planning,
      before a generator-node UI is wired.
- [ ] **Does the walker's composite position ever need to be configurable
      relative to ADR-CDG-010's β-rebuild/pin steps** (e.g. a future
      responsive-adjacent case that reads post-pin canvas state)? Clause 6
      assumes the walker is purely a config-mutator with no canvas
      dependency. **Resolution trigger:** revisit only if a future control
      signal needs canvas state as an input, which Option D's rejection
      currently forecloses for anything socket-shaped.

**Resolution plan:** all three are resolved during R1/R2 implementation
planning; none blocks recording the CV/LFO semantics, units-at-binding rule,
or the `num_inference_steps` exclusion, and none should be silently decided
by implementation ahead of that pass.

## Cross-references

- **ADR-CDG-010** owns the ordered composite (β-rebuild, pin, capture) this
  ADR's walker shares a run with. The walker is a `scheduler.config`-mutator,
  not a canvas-writer, so it does not occupy a slot in ADR-CDG-010's
  fixed canvas-write order — but clause 6 above fixes its timing relative to
  capture, and that timing must be read together with ADR-CDG-010's
  Decision 3 to reason about one step's full effect. Where a future
  participant needs to be both a config-mutator and a canvas-writer, its
  ordering must satisfy both ADRs simultaneously; neither ADR alone is
  sufficient to reason about that hybrid case, which is why it is called
  out here rather than assumed.
- **ADR-CDG-008** Correction 1 (`STATELESS-CORE`) is the invariant clause 8
  applies to walker state; the same same-in/same-out test (F5) is shared
  with ADR-CDG-010's pin-state clause — a single test surface can plausibly
  cover both, but this ADR does not mandate that the tests be merged, only
  that both invariants hold.
- **ARCHITECTURE.md** "The step-end intervention architecture" section
  states the target this ADR formalizes; its enforcement-surface table's
  same-in/same-out and effective-knob-telemetry rows point back at this ADR
  by name.

## Supersession Relationships

**Supersedes:** none.
**Superseded by:** TBD.

## References

- Issue #35 — architecture review; F5, R1, R3, R5, R6; the delta comment's
  same-in/same-out F5 handle.
- Issue #23 — per-step scheduling mechanism grounding
  (`scheduler.config` read-fresh-per-call; `num_inference_steps` guard rail;
  effective-telemetry requirement); its "Interface grounding" comment
  (ComfyUI execution-model: one-shot topological executor,
  `SIGMAS`/`model_options` prior art); its CV/LFO reframe comment
  (2026-07-13, "Primitive reframe").
- Issue #20 (closed) — `anneal_temperature` schedule-denominator desync
  mechanism; the regression this ADR's `num_inference_steps` exclusion
  forecloses by construction.
- PR #44 (merged) — FrozenDict ground truth for Decision 4's write
  mechanism, verified against the installed `diffusers==0.39.0` sources:
  `ConfigMixin.config` is a `FrozenDict` whose `__setattr__`/`__setitem__`
  raise once frozen (`configuration_utils.py:77-85`); the only write path is
  `register_to_config(**kwargs)` whole-dict replacement
  (`configuration_utils.py:143-158`). Mirrored by `tests/conftest.py`'s
  `FakeFrozenConfig`.
- ADR-CDG-001 — native socket types; the "lying sigmas" prohibition Option B
  is checked against.
- ADR-CDG-004 — drive seam; `run_diffusion` signature this ADR widens.
- ADR-CDG-008 — MCP-center topology, Correction 1 (`STATELESS-CORE` applied
  to the surface lifecycle object).
- `dgemma/loop.py:465,477` — `run_diffusion` signature; `anneal_temperature`
  (`dgemma/loop.py`, near top) — the formula this ADR's exact-temperature
  mechanism (clause 5) degenerates.
- `ARCHITECTURE.md` — "The step-end intervention architecture" section; rule
  7; the enforcement-surface table.
