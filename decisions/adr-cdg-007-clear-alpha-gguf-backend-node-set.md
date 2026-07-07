# ADR-CDG-007 — Clear-alpha architecture: GGUF/llama.cpp-fork backend, three-node set, steering-vs-illumination socket rule

**Status**: rejected (2026-07-06) — the GGUF/fork alpha is **not adopted for 0.1.0**; the pack ships on the `transformers`-bf16 path (ADR-CDG-002). See "Decision reversal" below. The GGUF design is preserved as the record of the considered-and-set-aside alpha.
**Date**: 2026-07-06
**Related**: ADR-CDG-002 (access path — this ADR *amends* it for the alpha: it flips 002's
GGUF-as-primary rejection for the alpha scope), ADR-CDG-004 (drive seam — this ADR defers the
diffusers/transformers path to milestone-2), ADR-CDG-001 (socket types — this ADR states the
*when-bespoke* rule 001 left implicit), ADR-CDG-006 (advanced step-window sampler — reclassified
as milestone-2, gated on native per-step stepping this backend does not provide)

Source of the decision arc: `handoffs/2026-07-05-flipbook-and-sampler-theory.md` (committed).
Sampler *mechanics* (anneal formula, renoise, early-stop discontinuity, failure-mode theory) stay
in that handoff and `loose-ends.md`; this ADR captures only the *architecture decisions* they motivate.

---

## Decision reversal (2026-07-06) — this proposal is REJECTED for 0.1.0

**What changed.** This ADR *proposed* shipping the alpha on the GGUF/llama.cpp-fork backend. That
proposal is **not adopted for 0.1.0.** 0.1.0 ships on the **`transformers`-bf16 path** (ADR-CDG-002
load seam) — the runtime with two verified integration PASSes on the 48 GB RTX-8000.

**Why the reversal (decisive → supporting):**

1. **Fork dependency is disqualifying for a public pack.** The GGUF backend is a *private, unpublished*
   llama.cpp fork (`/srv/dev/llama.cpp-diffusiongemma`). This ADR's own **Open Question 1** flagged the
   fork's public/buildable status as a **blocking prerequisite, not a follow-up.** A ComfyUI-Manager
   listing cannot depend on a fork a user cannot obtain; the transformers path depends only on public HF
   weights + public `transformers`. **OQ1 is hereby resolved in the honest direction:** rather than block
   0.1.0 on publishing/upstreaming a fork, ship on the public in-torch path.
2. **The transformers path works today.** bf16 + `device_map="auto"` CPU-spill, two verified PASSes
   (`dgemma/model.py`) — ships now, no fork gate.
3. **No in-torch quant rescues the memory fit — and that is inherent, not a missing flag.** The
   ~42.5 GiB of fused 3D MoE experts (`DiffusionGemmaTextExperts`) are not `nn.Linear`/`Conv1D`, so every
   stock quantizer (bnb nf4/int8, torchao/quanto **fp8**, AWQ) skips them — fp8 would shrink only the
   ~1 B of Linear params. This is **module coverage, not a hardware cast**: nf4 already dequantizes to
   fp16 on Turing with no Blackwell needed. GGUF fit *only* because llama.cpp quantizes the experts
   natively in the conversion.

**The honest envelope of 0.1.0.** The shipped path needs a **large-VRAM GPU (≈48 GB+)** and CPU-spills
the unquantizable experts (~24 tok/s). It does **not** fit consumer 16–24 GB cards. 0.1.0 is an
*experimental instrumentation pack* — its README/requirements must state this envelope plainly.
Ampere/bf16-tensor-core validation and any smaller-checkpoint path remain **issue #16** (`pri:later`).

**GGUF is deferred, not cancelled.** It remains the route to consumer-fittable memory (Q4_K_M ~15.6 GB,
experts quantized). Revive when the fork is public/upstreamed. Tracked: **issue #15** (`pri:later`).

**Status bookkeeping.** Because this ADR is **rejected**, its proposed amendments/deferrals do **not**
take effect: ADR-CDG-002 stands as the load seam; ADR-CDG-004 and ADR-CDG-006 are **not** deferred-by-007
and retain their own independent status.

---

## Context

The pack's grounded access-path ADRs (002 → 004) target a `transformers`-load + `diffusers`-drive
pipeline, chosen for its native per-step commit mask and mid-loop constraint injection. That path
has since hit a wall that only became visible when the quant options were exhausted this session:

- The model is `google/diffusiongemma-26B-A4B-it`, ~53.6 GB safetensors (bf16). The model card
  wants ≥60 GB GPU memory for a bf16 load; the 48 GB RTX-8000 dev box (and any consumer card)
  cannot hold it un-quantized.
- Every third-party quant that would make the diffusers path fit is dead or walled on the current
  modeling revision: `cyankiwi/...-AWQ-INT4` fails on `transformers==5.13.0` with both a
  `param_element_size` gap **and** an arch-key **revision mismatch** (`model.decoder.layers.*` vs
  `model.encoder.language_model.layers.*`). That mismatch is not AWQ-specific — it transfers to any
  same-era third-party quant, MXFP4 included. `bitsandbytes` is walled and NVFP4 is Blackwell-only.

Meanwhile a different runtime **is proven running this session**: the DiffusionGemma **llama.cpp
fork** (`/srv/dev/llama.cpp-diffusiongemma`, CLI `build/bin/llama-diffusion-cli`, entropy-bound
decoder in `examples/diffusion/diffusion.cpp`). Q4_K_M GGUF is ~15.6 GB — consumer-Blackwell-fittable,
with A4B-MoE `-ngl`/`-cmoe` offload covering a 16 GB card. `tools/flipbook/flipbook.py` already drives
it into a navigable per-step flip-book. The alpha can ship on a runtime that *works today* instead of
one that is quant-blocked. ADR-CDG-002 rejected GGUF-as-primary only because per-step loop exposure was
*unconfirmed*; the flip-book confirms whole-run capture is enough for the alpha's viz goal.

The forcing question the alpha must answer: **what is the smallest node set that delivers the
DiffusionGemma-as-instrumentable-graph value on a runtime that runs on consumer hardware now?**

## Decision

Ship the alpha as a **three-node ComfyUI set over the GGUF/llama.cpp-fork backend**, deferring the
diffusers/native-stepping path to milestone-2.

1. **Backend = GGUF via the DiffusionGemma llama.cpp fork** (not `transformers`/`diffusers`) for the
   alpha. Q4_K_M (~15.6 GB), whole-run inference through `llama-diffusion-cli`.
2. **Node set (three nodes):**
   - **Loader** — styled after ComfyUI's *Load Diffusion Model* (the DiT loader; class `UNETLoader`,
     where "UNET" is a legacy envelope name), **not** *Load CLIP*. Lists model **directories** under
     `models/diffusion_models/` and loads via `from_pretrained(local_files_only=True)` — no `hf_hub`
     network fetch. Emits bespoke `DGEMMA_MODEL`.
   - **`SIGMAS→heat` translation node** — consumes a stock `SIGMAS` schedule (from `BasicScheduler`,
     RES4LYF, the solver zoo) and emits DiffusionGemma **heat** (the temperature/entropy trajectory),
     with its own tuning. It is the explicit `PARSE-AT-THE-DOOR` boundary: `SIGMAS` stays `SIGMAS`
     until this node translates it — it does **not** relabel a sigma tensor as heat (that would be
     ADR-CDG-001's "lying sigmas").
   - **Run + flip-book node** — drives the CLI over the canvas and emits per-step frames.
3. **Socket-type rule (states ADR-CDG-001's implicit *when*):** ComfyUI has **no runtime type
   enforcement** — the only "typing" is frontend link-illumination (`AlwaysEqualProxy`/`AnyType`
   wildcards compose with anything). Therefore choose the socket type by intent, not safety:
   - **Bespoke type** when a value should reach a **narrow** set of CDG sockets — *steering*: only
     compatible inputs illuminate, quietly guiding correct wiring. → `DGEMMA_MODEL`, **heat**,
     resume-state.
   - **Stock type** when a value should **compose broadly** — *illumination*: let it batch and preview
     with the whole ecosystem. → **stock `IMAGE`/`STRING`** for flip-book frames.
4. **Default `--diffusion-eb-confidence 0.0`** — run the full anneal to `max_steps` instead of the
   entropy early-stop, which (per the handoff findings) truncates the anneal mid-curve and ships
   under-annealed "quit-hot" output. Other defaults unchanged: `max_steps=48`, `t=[0.4, 0.8]`,
   `entropy_bound=0.1`, `canvas_length=256`.
5. **One bounded fork extension:** the CLI must accept a **per-step heat array** (e.g.
   `--eb-temp-schedule <file>`). Without it the `SIGMAS→heat` node can set only the anneal
   *endpoints* and loses the curve — which is the point of translating a schedule at all.
6. **Milestone-2 (deferred, not cancelled):** MXFP4-on-diffusers → native per-step stepping + the rich
   For-loop / `canvas_state` node family (ADR-CDG-004, ADR-CDG-006) — **if** a quant loads against the
   current modeling revision.

### Component design & data flow (alpha)

```
[Load DiffusionGemma (GGUF)]                    [BasicScheduler / RES4LYF] --(SIGMAS)-->
      | (DGEMMA_MODEL, bespoke — steering)                                              |
      v                                                                                 v
      +----------------------------------------------+   <--(heat, bespoke)--  [SIGMAS -> heat]
      |            Run + Flip-book (CLI)             |
      +----------------------------------------------+
             |                         |
    (IMAGE frames, stock)     (STRING text, stock)   --> batch/preview with the whole ecosystem
```

- **State / persistence.** The alpha is **stateless across runs** — whole-run inference, no
  cross-step resume. The CLI owns the canvas for the duration of a run; frames land on disk
  (`tools/flipbook/out/<slug>/`, gitignored, regenerable) and cross the node boundary as stock
  `IMAGE`/`STRING`. There is no `CANVAS_STATE`/resume payload in the alpha — that is milestone-2
  (ADR-CDG-006), and its absence here is deliberate, not an omission: the fork gives whole-run
  inference with no native per-step stepping to resume *from*.
- **Model identity.** `DGEMMA_MODEL` carries the loaded-model handle (GGUF path + CLI config).
  Loader emits it canonically; downstream nodes parse it at the door — no relabeling.
- **Error / failure paths.** (a) Loader over a directory with no readable GGUF → hard error at
  `from_pretrained(local_files_only=True)`, surfaced as node failure (never a silent network fetch —
  `local_files_only` forbids it). (b) CLI non-zero exit / OOM under `-ngl`/`-cmoe` overflow → run node
  fails with the captured CLI stderr, no partial-frame batch emitted as if complete. (c) Missing
  `--eb-temp-schedule` support in the installed fork build → `SIGMAS→heat` degrades to endpoints-only
  and must **declare** it (a `STRING`/log field), not silently drop the curve — the honest-payload
  discipline of ADR-CDG-001 applied to the time axis.

## Rationale

### Positive Consequences
- **Ships now.** The alpha runs on a proven, consumer-fittable runtime instead of waiting on a quant
  that may never load against the current revision.
- **Honest boundary preserved.** `SIGMAS→heat` makes the type translation an explicit, visible node
  rather than a silent relabel — ADR-CDG-001's principle, realized as a first-class bridge into the
  mature scheduler ecosystem instead of an isolation wall.
- **Steering-vs-illumination is the right axis.** Bespoke where narrow-steering pays (model/heat/
  resume), stock where broad composition pays (frames) — captures the actual UX value of ComfyUI's
  link-illumination without paying the bespoke-type cost where it buys nothing.
- **Milestone-2 is preserved, not burned.** The diffusers path (native stepping, the ADR-006 sampler)
  remains the prize; this ADR reclassifies it, it does not reject it.

### Negative Consequences
- **No native per-step stepping in the alpha.** Whole-run inference only; the rich For-loop /
  `canvas_state` / step-window-resume node family (ADR-CDG-006) cannot exist on this backend. The
  flip-book is a post-hoc capture of a completed run, not a resumable step machine.
- **Distribution cost.** The backend is a *fork*, not stock llama.cpp. The alpha's shippability now
  depends on that fork being public/buildable by a user (open question 1) — a dependency stock-runtime
  alphas don't carry.
- **A fork change is on the critical path.** The `--eb-temp-schedule` extension is required before the
  translation node delivers curves; until it lands the node is endpoints-only.
- **Two backends to reason about.** The alpha (GGUF/fork) and milestone-2 (diffusers) diverge in load
  seam, drive seam, and state model — the pack must keep both mental models straight during the
  transition.

## Alternatives Considered

### Option A — diffusers/transformers backend with an AWQ/MXFP4 quant (the ADR-002/004 path) for the alpha
The originally-decided path: native per-step commit mask, mid-loop constraint injection, the ADR-006
step-window sampler — all first-class.

**Why rejected for the alpha (deciding factor):** it is **quant-blocked on the current modeling
revision**. AWQ is dead (`param_element_size` gap + `decoder.layers` vs `encoder.language_model.layers`
revision mismatch); the mismatch transfers to MXFP4; bf16 won't fit 48 GB; `bitsandbytes` walled;
NVFP4 Blackwell-only. An alpha cannot ship on a runtime that does not load. **This is the milestone-2
prize, not a permanent rejection** — it is revivable the moment a quant loads against the current
revision (open question 2).

### Option B — *Load CLIP (gguf)* as the loader base
ComfyUI already ships a GGUF-capable CLIP loader; cloning it would inherit working GGUF plumbing.

**Why rejected (deciding factor, source-confirmed):** it is **arch-gated and the wrong interface**.
Stock arch detection returns `None` for DiffusionGemma, and the CLIP-loader path forces a
CLIP/text-encoder interface the model cannot present. The *Load Diffusion Model* (DiT) loader is the
right envelope — but even it needs a new node because stock detection returns `None` and the
monkeypatch path forces a sigma/latent `BaseModel` interface DiffusionGemma cannot honor. The case for
a bespoke loader is **mechanical**, not stylistic.

### Option C — whole-hog bespoke socket types everywhere, for safety
Type every CDG payload as bespoke so the graph is maximally self-documenting and "type-safe."

**Why rejected (deciding factor):** there is **no runtime type enforcement to make it safe** —
ComfyUI's `AlwaysEqualProxy`/`AnyType` means bespoke types buy *steering* (link-illumination), not
safety, and bespoke types carry a real UX cost (they don't batch/preview/compose with the ecosystem).
The actual axis is **steering vs composition**, not safety vs unsafety — so bespoke only where narrow
steering pays, stock (`IMAGE`/`STRING`) where broad composition pays.

## Open Questions

- [ ] **(1) Is the DiffusionGemma llama.cpp fork public/buildable by a user today, or does it need
  publishing?** The alpha's backend is `/srv/dev/llama.cpp-diffusiongemma`, a fork, not stock llama.cpp.
  **Resolution trigger — decides alpha shippability:** confirm the fork's public/buildable status
  before declaring the alpha shippable; if not public, publishing it (or upstreaming) is a blocking
  prerequisite, not a follow-up.
- [ ] **(2) Do MXFP4 checkpoints load against the *current* modeling revision?** The `decoder.layers`
  vs `encoder.language_model.layers` mismatch that killed AWQ transfers to any same-era third-party
  quant. **Resolution trigger — decides whether milestone-2 (diffusers/native-stepping, ADR-006) is
  revivable:** verify a candidate MXFP4 checkpoint loads against the installed modeling revision before
  betting milestone-2 on it.
- [ ] **(3) Does the fork accept a per-step heat array?** The `SIGMAS→heat` node delivers curves only
  if the CLI takes a per-step schedule (e.g. `--eb-temp-schedule <file>`). **Resolution trigger —
  needed before the translation node is more than endpoints-only:** land the bounded fork extension (or
  confirm an existing flag) before the translation node's curve output is claimed as working.

**Resolution plan:** (1) and (3) are prerequisites tracked against the alpha build; (2) is a
milestone-2 go/no-go gate. All three are grounding reads (fork status, checkpoint load, CLI flag), not
design decisions — each resolves by observation, not deliberation.

## Supersession Relationships

**Supersedes:** none outright. **Amends (on acceptance):** ADR-CDG-002 — flips its GGUF-as-primary
*rejection* for the **alpha scope** (002 rejected GGUF-as-primary because per-step exposure was
unconfirmed; the flip-book confirms whole-run capture suffices for the alpha). **Defers:** ADR-CDG-004's
diffusers drive seam and ADR-CDG-006's step-window sampler to **milestone-2**, gated on a quant that
loads (open question 2). These are **scope reclassifications, not rejections** — the diffusers path
stays the prize.

*Bidirectional bookkeeping — resolved by rejection (2026-07-06):* this ADR never reached `accepted`,
so none of its proposed amendments/deferrals fired. ADR-CDG-002 / -004 / -006 status lines are **left
untouched** — a `rejected` ADR moves no other ADR's status. The GGUF-alpha consideration is retained
above as durable record so the quant dead-ends and the fork-dependency reasoning are not re-burned.

**Superseded by:** TBD (milestone-2's native-stepping ADR may supersede the backend decision if/when
the diffusers path loads).

## References

- `handoffs/2026-07-05-flipbook-and-sampler-theory.md` — the decision arc (committed)
- `tools/flipbook/flipbook.py` — proven whole-run per-step capture over the fork CLI
- `/srv/dev/llama.cpp-diffusiongemma` — the fork; CLI `build/bin/llama-diffusion-cli`; entropy-bound
  decoder `examples/diffusion/diffusion.cpp`
- ADR-CDG-001 (socket types), ADR-CDG-002 (access path), ADR-CDG-004 (drive seam), ADR-CDG-006
  (advanced step-window sampler)
- Quant dead-ends re-confirmed in the handoff's "Quant dead-ends" section (do not re-burn)
