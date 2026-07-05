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

---

## 2026-07-05 — bitsandbytes cannot quantize DiffusionGemma's MoE experts: NF4 load does NOT fit the 48GB box

**Category:** grounded capability wall (blocks P1 integration test on this hardware)
**Related ADR:** ADR-CDG-002/004 (load seam); CLAUDE.md grounded facts (the
"quantized loading is identical on both paths" note still holds — the wall is
bnb-vs-architecture, not transformers-vs-diffusers)
**Graduation trigger:** choosing the replacement quantization/offload strategy
is an ADR — it moves the load seam's dependency set and the P2+ performance
envelope.

### Context
First real integration run (weights cached, `DGEMMA_INTEGRATION=1`,
2026-07-05). The working assumption "NF4 of the 26B ≈ 14GB, fits 48GB with
headroom" failed on grounding. Evidence chain, all against installed
transformers 5.13.0 + bitsandbytes 0.49.2:

- bnb module replacement only touches `Conv1D` / exactly-`nn.Linear`
  (`transformers/integrations/bitsandbytes.py:189`).
- DiffusionGemma's MoE expert weights are fused 3D `nn.Parameter`s on
  `DiffusionGemmaTextExperts` (`modeling_diffusion_gemma.py:560-569`:
  `gate_up_proj [128, 2*704, 2816]`, `down_proj [128, 2816, 704]`) — never
  replaced, never quantized.
- Config arithmetic: 128 experts x 30 layers → **22.84B expert params =
  42.5 GiB at bf16, unquantized**, + ~1.4 GiB embeddings (also non-Linear,
  262144 x 2816, tied lm_head) + vision tower. Only the attention/dense
  `nn.Linear`s (~1B params) actually go NF4.
- Observed: with `device_map={"": 0}` the load OOMs at
  `caching_allocator_warmup` (`modeling_utils.py:5107`), trying to allocate
  46.06 GiB against 45.05 GiB free — and that estimate is
  quantization-aware (`param_element_size`, `quantizer_bnb_4bit.py:83-95`),
  i.e. honest. With `device_map="auto"`, accelerate spills to CPU and the
  bnb 4-bit guard rejects (`quantizer_bnb_4bit.py:70-81`). Neither path
  loads; the model genuinely does not fit under bnb quantization.

### Decision
None yet — recorded as a wall, not routed around. Candidate directions for
the follow-up decision (each has real trade-offs, none P1-unilateral):
CPU-offload of experts via `llm_int8_enable_fp32_cpu_offload` + custom
device_map (known-slow: the llama.cpp analogue measured 24 tok/s spilled vs.
456 in-step); a quantization backend that handles fused 3D experts (survey
needed — torchao/HQQ/GPTQ-class support for `nn.Parameter` experts is
unverified); or accepting GGUF/llama.cpp graduation earlier than planned
(ADR-CDG-002 gates this on need — this may be the need).

### Implementation Notes
- `dgemma/model.py` keeps `device_map={"": 0}` for quantized loads (pinning
  is still correct for any checkpoint that fits; it skips accelerate's
  conservative placement and the bnb CPU-spill rejection).
- `tests/test_integration.py` stays as-is: it is the detector that caught
  this, and goes green the moment a working load path exists.

---

## 2026-07-05 — ComfyUI loader context broke bare `dgemma` imports in nodes/ (observed violation)

**Category:** observed violation (ComfyUI loader context) — upgrades
ADR-CDG-003's greenfield anticipated-failure ("two import surfaces to keep
coherent") to an observed one. Detector: the downstream **graph smoke test**
against real ComfyUI (custom-node import failed; unit suite alone had never
exercised the real loader context).
**Related ADR:** ADR-CDG-003 (node/engine seam — the two import surfaces)
**Graduation trigger:** none expected — the enforcement test now holds the
line; graduate only if a third import context appears.

### Context
`nodes/loader.py`/`nodes/sampler.py` imported the engine as bare
`from dgemma...`. Every in-repo test passed (pytest puts the repo root on
sys.path), but ComfyUI's loader (`/srv/dev/ComfyUI/nodes.py:2226-2246`) puts
`custom_nodes/` — never the pack root — on sys.path and loads the pack as a
package named after its directory path (`:2233,2241`). Result in production:
`ModuleNotFoundError: No module named 'dgemma'`, IMPORT FAILED for the whole
pack.

### Decision
Dual-context imports via an **explicit package-depth gate** in each nodes/
module (`if __package__ and "." in __package__:` → relative `..dgemma`,
else absolute `dgemma`), matching the root `__init__.py`'s `__package__`
gate discipline. Blanket `try/except ImportError` was rejected for the same
reason it was killed in review at the root: it masks real dependency
failures (and can shadow-import ComfyUI's own top-level `nodes.py`).

### Enforcement surface
`tests/test_comfyui_loader_context.py` — replays ComfyUI's exact load
mechanics (path-derived module name, `spec_from_file_location` on the pack's
`__init__.py`, sys.modules registration before exec) in a fresh interpreter
with the repo root stripped from sys.path — the condition pytest's own
environment always masked. Verified to bite: pre-fix it reproduced the
production failure verbatim (`ModuleNotFoundError: No module named 'dgemma'`).

### Implementation Notes
- `dgemma/` itself needed no change (all-relative internally) — the seam
  held; only the adapter layer's imports were context-fragile.

## 2026-07-05 — Dead frontend interactivity in Ubuntu Chrome on this box (unexplained, STILL BROKEN — workaround: browse from another machine)

*(Corrected same day: the first version of this entry claimed a reboot
cleared it. Wrong — operator readback: the PASS session was browsed **from
Windows on another machine** against the server bound `0.0.0.0`. Ubuntu
Chrome on the workstation remains broken, reboot notwithstanding.)*

### Context
First live-GUI session: canvas rendered but right-click/add-node was dead,
with **no UI error surfaced** — only console noise (userdata 404s, legacy-menu
warning, `graph accessed before initialization`). An A/B ladder exonerated
everything we control: server healthy (`/object_info` 793 nodes), pack schema
clean (tuple/mapping/case audit), stock-only instance on a virgin origin
(port 8189) **still broken** → not the pack, not client site-state, not the
install. The same server then ran the P1 graph to PASS from a **Windows
browser over LAN** — isolating the fault to Ubuntu Chrome (or its
display-stack interaction) on this box specifically.

### Decision
Recorded as environment caution, not repaired: root cause unknown, confined
to Ubuntu Chrome on this workstation. Workaround for GUI sessions: bind the
server to `0.0.0.0` and browse from another machine/browser. Do **not**
burn session time re-running the diagnostic ladder against Ubuntu Chrome —
it is already exonerated down to the browser/display layer.

### Why Not an ADR?
No design decision — a host-environment anomaly with an unexplained root
cause. Belongs in the log precisely so the next session doesn't re-derive
the (expensive) exoneration chain.

### Implementation Notes
- Diagnostic ladder that produced the exoneration, for reuse: backend
  `/object_info` count → schema audit (string-vs-tuple, mappings, case) →
  `localStorage.clear()` → userdata-dir 200s → stock-only on fresh port/origin.
- The `nf4` OOM found the same session is **not** this — that one is
  explained and banked on issue #4.
