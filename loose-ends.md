# Loose Ends Log

Tactical decisions that didn't qualify for ADR treatment but are worth
remembering. See `decisions/` for full ADRs.

**Created:** 2026-06-30
**Last updated:** 2026-07-05

---

## 2026-06-30 — Modular RES4LYF-shaped node topology

**Category:** single-option (idiomatic)
**Related ADR:** ADR-CDG-001 (socket types), ADR-CDG-002 (access path)
**Graduation trigger:** If `CANVAS_STATE` chaining between nodes proves to need
tight coupling (shared mutable state that leaks across node boundaries),
revisit node-boundary design as an ADR.

### Context
Deciding how to decompose the pack into nodes. RES4LYF (and ComfyUI's
CustomSampler convention generally) splits into a schedule node, a
guides/conditioning node, a sampler node, and a chain of options nodes.

### Decision
Mirror that decomposition: **EntropySchedule → Constraints → Sampler**, with a
chainable `Options_*` family feeding an `options` socket — rather than one
monolithic node. (Payloads are the ADR-CDG-001 types, not `SIGMAS`/`LATENT`/`GUIDES`.)

### Why Not an ADR?
- [ ] Hard to reverse? → Somewhat, but it's a structural convention, not tech lock-in.
- [x] Surprising without context? → No. Modular CustomSampler-shaped nodes are the
      idiomatic ComfyUI pattern; a reader expects this, not the reverse.
- [x] Real trade-off? → Weak. Monolithic is possible but non-idiomatic; modular is
      the obvious platform-native choice, especially given the dev's deep ComfyUI
      familiarity and the goal of "insane combinations."

### Implementation Notes
- Node families: `DGemmaLoader`, `DGemmaEntropySchedule`, `DGemmaConstraints`,
  `DGemmaSampler`, `DGemmaTrace`, `DGemmaOptions_*`.
- `bongmath`-equivalent toggle on the sampler is `self_conditioning` (the
  documented loop feeds output logits back as self-conditioning for the next step).

---

## 2026-07-05 — DGemmaRenoise (text-to-text variation, SDEdit analogue): feasible-soft, priced

**Category:** feasibility ceiling (accepted)
**Related ADR:** ADR-CDG-004 (drive seam); ADR-CDG-001 addendum (scheduler-relative commit semantics)
**Graduation trigger:** If renoise becomes a headline node (not just an
experiment), it gets an ADR — the pipeline-subclass surface and strength
semantics are real design surface once something is built on them.

### Context
A text-to-text variation node — inject existing text into the canvas at some
corruption "strength" and diffuse from there, the SDEdit analogue for
discrete text diffusion. Grounded against the installed diffusers 0.39.0
pipeline: no seeding parameter exists on `DiffusionGemmaPipeline` — canvas
init is a hardcoded `torch.randint` with no injection point
(`pipeline_diffusion_gemma.py:346-348`).

### Decision
The soft version is feasible and priced: a ~50-100-line pipeline subclass
corrupting the user's injected text by `strength` for the canvas init.
**Hard-lock semantics are declined against a grounded ceiling, not a guess:**
injected text can only ever be evidence, not a hard constraint, because
`BlockRefinementScheduler` unconditionally resets `_committed` at step 0 of
every block (`scheduling_block_refinement.py:266`), and
`EntropyBoundScheduler` / `DiscreteDDIMScheduler` hold no commit state at all
to lock against. True hard-lock (injected tokens that cannot be renoised
away under any scheduler) would require vendoring or monkeypatching
third-party scheduler internals — declined. The soft version plus per-step
callback re-assertion (the same mechanism as P5 hard pinning,
`pipeline_diffusion_gemma.py:407`) covers the intended "variation" use case.

### Why Not an ADR?
- [ ] Hard to reverse? → No — a pipeline subclass, swappable/discardable.
- [ ] Surprising without context? → Somewhat, but this entry carries the
      grounding; no separate record needed while it's unbuilt.
- [x] Real trade-off? → Yes (soft vs. hard lock), but not yet load-bearing —
      nothing is built on it yet.

### Implementation Notes
- Candidate node: `DGemmaRenoise`. Not yet built.
- Graduates to an ADR when it ships as a headline node (the subclass surface
  and strength semantics need a decision record at that point).

---

## 2026-07-05 — Analyzer mode (entropy map of existing text): near-free, ~15-20 lines

**Category:** feasibility ceiling (accepted)
**Related ADR:** ADR-CDG-004 (drive seam)
**Graduation trigger:** Wire as a `DGemmaTrace` input mode or a tiny
standalone node once Phase 3's trace plumbing exists (P3-adjacent); promote to
an ADR only if it grows scheduler-touching logic of its own.

### Context
Running the entropy/temperature view over *existing* text (no diffusion
loop) — an analyzer, not a generator. Grounded against the installed
diffusers 0.39.0 pipeline: the loop body minus the loop — encoder KV populate
+ mask build + one forward (`pipeline_diffusion_gemma.py:318-343,364-371`) —
needs no scheduler at all, and the per-step temperature view is replicable
standalone from the anneal formula (`scheduling_entropy_bound.py:153-155`).

### Decision
Near-free: ~15-20 lines, no scheduler dependency. One caveat, flagged
UNGROUNDED during this pass rather than assumed: `modeling_diffusion_gemma.py`'s
encoder/mask-build call ordering was not independently verified — a quick
check before wiring is warranted.

### Why Not an ADR?
- [ ] Hard to reverse? → No — a small, isolated function.
- [ ] Surprising without context? → No, once this entry exists.
- [ ] Real trade-off? → None found; the only open cost is the ordering check
      above.

### Implementation Notes
- Candidate: a `DGemmaTrace` input mode, or a tiny separate node. P3-adjacent.
- Before wiring: verify `modeling_diffusion_gemma.py`'s encoder → mask-build
  call ordering (flagged ungrounded during this pass, not yet checked).

---

## 2026-07-05 — In-node live text view mechanism: send_sync + WEB_DIRECTORY (grounded, not yet built)

**Category:** grounded mechanism, not yet built
**Related ADR:** none directly; the mechanism this entry grounds is the (a)
LIVE-view half of `plan.md`'s Phase 3 split (the (b) ANALYSIS half is
`DGemmaTrace`, unaffected by this entry).
**Graduation trigger:** When P3 builds this, if the frontend idiom turns out
to need more than `addEventListener` + `setDirtyCanvas`, revisit — this entry
assumes that's sufficient based on the shim-level bundle read, not a
worked first-party example.

### Context
ComfyUI's execution model gives a node's outputs to downstream sockets only
once its `FUNCTION` returns — there is no way for a node to stream per-step
frames to another node's input live. A live denoise view therefore has to be
a feature of the *sampling* node's own body, not a downstream consumer.

### Decision
`DGemmaSampler`'s sync `FUNCTION` calls
`PromptServer.instance.send_sync("<custom_event>", payload)` once per step.
Grounded:
- `send_sync` is thread-safe by construction: it does
  `self.loop.call_soon_threadsafe(self.messages.put_nowait, (event, data, sid))`
  (`server.py:1374-1376`), so calling it from a sync function running off the
  asyncio loop's own thread is safe.
- There is no event-name whitelist on the receiving side: `send` routes
  anything that isn't a binary preview type to `send_json`, which just
  wraps `{"type": event, "data": data}` with no name check
  (`server.py:1364-1372`, dispatch at `server.py:1272-1281`). A custom event
  name is free to use.
- The frontend listener is a `WEB_DIRECTORY`-registered JS extension:
  `nodes.py:2269-2272` checks `module.WEB_DIRECTORY` and mounts it into
  `EXTENSION_WEB_DIRS`; `server.py:1225-1226` serves it as a static route
  (`/extensions/<name>`).

**Named trap:** do not smuggle this through `ProgressBar`'s `preview=` slot.
That path is structurally image-typed all the way down —
`comfy/utils.py`'s `ProgressBar.update_absolute` → the global hook installed
in `main.py` → `server.send_image` (`server.py:1293-1301`), which calls
`image.save(bytesIO, format=image_type, ...)` on whatever it's handed. A
string payload throws there; it is not a generic preview channel.

**Named residuals:**
- No in-tree precedent for per-step *text* push: `comfy_extras/*.py` has no
  `send_sync` usage to copy from. This pack establishes the pattern rather
  than following one.
- The frontend `addEventListener`-on-a-custom-event idiom is confirmed only
  at the shim/minified-bundle level, not walked through in a worked
  first-party JS example — verify the actual listener API against the live
  frontend when P3 builds this, not just against the shim.

### Why Not an ADR?
- [ ] Hard to reverse? → No — a JS extension file and one `send_sync` call
      site, both swappable without touching the engine.
- [x] Surprising without context? → Somewhat, hence this entry carrying the
      grounding rather than leaving it implicit.
- [ ] Real trade-off? → None found; the ComfyUI execution model leaves only
      one mechanism for a live in-node view, so there was nothing to choose
      between.
