# ADR-CDG-020 — GGUF engine sourcing: consumer of a pinned upstream PR branch, protocol-bounded, not an owned fork

**Status**: accepted (2026-08-06) — ratified by independent Opus design-gate (fourth 2026-08-06 run) after the #131 rung-1 line closed: the line-120 acceptance gate ((A) selects a pin, (B) confirms example-layer implementability, (C) shows a usable envelope) is satisfied. **Pin SELECTED: [ggml-org/llama.cpp#24427](https://github.com/ggml-org/llama.cpp/pull/24427) @ `dd0cf04459b0c4f43aa6667dbc0879ac0cd50323`.** Operator-vetoable per the standing #131 gate. Gate adjudication: https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131#issuecomment-5212054211 — resolved open questions recorded as decisions below (§Open Questions, §Decision 2, §Implementation Notes). Proposed 2026-08-04; the pre-ratification gating history is preserved in the §Open Questions resolution records.
**Date**: 2026-08-04 (proposed) / 2026-08-06 (accepted)
**Related**: ADR-CDG-002 (access path — GGUF parked as graduation-triggered inference-only backend; this ADR graduates it), ADR-CDG-004 (diffusers drive seam — the transformers-primary path this remains secondary to), ADR-CDG-007 (**superseded by this ADR** — its preserved three-node GGUF design is the substrate this decision re-homes onto the pinned-PR posture), ADR-CDG-014 (`DiffusionFrame` wire-format contract — the abandonability seam this ADR leans on), ADR-CDG-018 (`loop.py` decomposition — maps the loop/interface delta this engine wraps), issues #131 (the engine thread), #15 (the 2026-07-06 no-fork ruling), #223 (the divergence this ADR resolves), #16 (audience/quant matrix)

Grounding reads (verify the current-system claims by following these):
- #131 rung-0 build readback: pin candidate `c3fb97241295c196e09b783e705e84b96cd1bd74` = tip of upstream draft PR [ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423); CPU build 1m27s, CUDA sm_75 build 9m46s, both exit-0 on the RTX-8000 host.
- #131 conformance-matrix comment (2026-08-03): core semantics identical across CDG / #24423 / #24427; divergence axis = self-conditioning signal fidelity.
- #15 operator ruling (2026-07-06): "never a fork this project compiles and hosts."
- `decisions/adr-cdg-007-*.md:82-145` — the preserved three-node GGUF design + its error/state/failure-path model.
- `https://github.com/shanevcantwell/design-docs/blob/main/experiments/ComfyUI-DiffusionGemma/handoffs/2026-08-03-v0.5.0-released-next-kv-or-gguf.md:21` — the three packaging design questions.
- `ROADMAP.md:51-56` — the "rung 4" shorthand this ADR replaces with the #131 thread sequence.

---

## Context

The pack's primary runtime is `transformers`-load + `diffusers`-drive (ADR-CDG-002 → -004), which needs ≈48 GB-class VRAM and CPU-spills the unquantizable MoE experts (~24 tok/s). That envelope reaches ~nobody in the ComfyUI community. GGUF Q4-class (~14–16 GB, arch-agnostic kernels) is the only path with a real user population — the Unsloth GGUF card reports ~52.7k monthly downloads (#131), a population currently self-serving by `gh pr checkout 24423` because DiffusionGemma is **not in llama.cpp master** (zero code-search hits) and both competing draft PRs have had **zero maintainer engagement since day one**. There is no visible upstream clock.

Two forces collide here, and this ADR exists to reconcile them in one signed place (the disposition #223 asks for):

1. **The operator's #15 ruling** — "never a fork this project compiles and hosts." A public ComfyUI pack cannot depend on code a user cannot obtain, and owning a fork means owning its drift against ggml master.
2. **The 2026-08-03 named reopen** (#131) — reopened that ruling in a *narrower* form: not a from-scratch engine, but consuming a community-hardened PR branch, because the pack's actual product (the instrument panel — per-step frames, canvas state) is a visibility surface **neither upstream PR exposes** (#24423 streams argmax-only; #24427 built an entropy-readback seam then left it dark, passing `nullptr` on all paths).

The forcing question: **what engine-sourcing posture gives the ~52.7k-download population the pack's actual instrumented product, without acquiring an owned-fork maintenance liability the operator has ruled against?**

ADR-CDG-007 proposed the GGUF backend a year-scale-ago-in-repo-time and was *rejected* for 0.1.0 — but rejected on the *fork-obtainability* prerequisite (its Open Question 1), not on the node design. Its three-node set (loader / SIGMAS→heat / run+flipbook) and its state/error model are sound substrate. That rejected ADR is now being treated as revived substrate in the working narrative while its formal record still reads `rejected / Superseded by: TBD` — the divergence #223 names. This ADR closes it.

## Decision

### 1. Engine sourcing posture — consumer of a pinned upstream PR branch, not an owned fork, not wait-for-merge

The pack sources its GGUF engine as a **pinned reference to an upstream draft-PR commit** (`c3fb972`-class SHA), consumed at a **protocol boundary**, never vendored or forked into an owned, hosted, project-compiled codebase. *"Compiles" in the #15 ruling's sense* means **owns-and-maintains a forked codebase** — carrying patches, rebasing against master, owning CI and backports; that is what is forbidden. It does **not** mean running a **reproducible CI build of a named upstream commit**, which is the provenance mechanism (a checksummed artifact of code we do not own or patch), not a fork. Building `c3fb972` in CI is the former only if we start carrying our own patches on top of it — which INV-1's protocol boundary and Decision 3c's re-pin-not-rebase maintenance scope explicitly forbid.

- **Rejected: owned fork.** Directly contradicts the #15 ruling. Owning a fork means owning its rebase drift against ggml master, its CI, its security backports, and its provenance — the maintenance liability the operator forbade.
- **Rejected: wait-for-upstream-merge.** There is no visible clock: #24423 draft since 2026-06-10; #24427 stale ~11 days as of 2026-08-03 — zero maintainer engagement on either. Waiting strands the ~52.7k-download population indefinitely on self-serve `gh pr checkout`.
- **Chosen: pinned-PR consumer.** The pack's build/packaging step checks out a *named upstream SHA* and builds `llama-diffusion-cli` from it (rung-0-proven at `c3fb972`, CPU + CUDA sm_75). The node consumes the engine through a **protocol** (frame format + step events), never engine internals.

**Named honestly — this is not owned provenance.** We own a *SHA reference*, not the code. "Owned pin" (as ROADMAP.md:55 phrases it) overstates the ground and must be corrected: the provenance is upstream's; our owned artifact is the *checksummed build output of a named upstream commit* plus our packaging around it. What we genuinely own and must maintain is **the pin's drift exposure**: when upstream force-pushes the draft PR, abandons it, or ggml master moves under it, our pin either goes stale (buildable but diverging from community heads) or unbuildable (if the branch is deleted). That cost is real and is the price of not-forking; §Risk quantifies it and §Decision 5 bounds the blast radius to the protocol seam.

### 2. Pin choice framework — criteria fixed here, selection deferred to rung-1 readback

The choice between pin candidates **#24423** (danielhanchen/Unsloth, `llama-diffusion-cli` + `--diffusion-visual`, the Unsloth-card-endorsed known-good path, rung-0-green at `c3fb972`) and **#24427** (lnigam/NVIDIA-orbit, adds an OpenAI-compatible `llama-diffusion-gemma-server`, cleaner kernels, a *built-but-dark* entropy-readback seam) is **deferred to the rung-1 probe** (Open Question A). The criteria by which it resolves are fixed now, ordered by weight:

1. **Frame-contract reachability (decisive).** Which branch lets the pack emit the `DiffusionFrame` Tier-0 contract (ADR-CDG-014) with the least engine-internal reach? #24423's visual server already streams per-step frames over stdio (argmax-only — delta = *widen the stream schema*, example-layer). #24427's dark entropy seam is a cleaner kernel base with no serving loop (delta = *build the driver*, example-layer). Both deltas live in the example layer, not kernel work — H0 in §Open Questions.
2. **Buildability & pin stability on the target hosts.** Rung-0 already green on #24423 at `c3fb972`/sm_75. #24427 is unmeasured on this host.
3. **Self-conditioning fidelity's effect on convergence.** The one semantic divergence (#24427 top-k-truncates the self-cond signal; #24423 passes full; CDG passes temp-scaled full). H0: truncation does not materially alter convergence — either answer is upstream-grade evidence, neither disqualifies.
4. **Serving-loop shape.** #24427's OpenAI-compatible server is closer to a persistent-load model but is *more* surface to track; #24423's CLI is thinner. Weighed *against* pin-stability, not for feature richness.

**Resolution trigger:** the #131 rung-1 measurement plan (visibility tax on/off; #24423 visual-frame capture vs Tier-0 contract; three-path conformance run; determinism probes; Q4_K_M tok/s on sm_75). The pin is selected when rung-1 returns; until then the pack builds against `c3fb972` (#24423) as the rung-0-de-risked default, **not** as the ratified choice.

> **RESOLVED (2026-08-06, ratification gate) — pin = [#24427](https://github.com/ggml-org/llama.cpp/pull/24427) @ `dd0cf04459b0c4f43aa6667dbc0879ac0cd50323`.** The decisive criterion (1, frame-contract reachability) is **REACHED** on #24427: #277 built and measured a Tier-0 `DiffusionFrame` stream on #24427's example layer (`--diffusion-frame-stream`, serialize-only, no kernel reach; Tier-0 core `{canvas_idx, step_idx, t, temperature, entropy}` exact-match, 48/48 lines × 3 seeds, ~0.11% emission tax), while #24423 stays `--diffusion-visual` argmax-only. Criterion 3 (self-cond fidelity, = OQ-D) was probed to closure: #278 disconfirmed self-cond width as the cause of #24427's default-config non-convergence, and the device-path early-stop mechanism probe ([comment 5212022958](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131#issuecomment-5212022958): matrix 22 runs, 16/16 `gpuOFF` early-stop 16–45/48, 6/6 `gpuON` ceiling 48/48, zero crossover, `--diffusion-gpu-sampling` isolated as sole causal flag, device-denoise-loop byte-identical/inert) returned **mechanism = TOGGLEABLE, not intrinsic**. So #24427 *is* capable of entropy-bound early-stop; criterion 3 returns to its ADR-authored "neither disqualifies" posture and no longer counterweights toward #24423. Both live criteria (1 decisive, 3 non-disqualifying) favor #24427. **Implementation-note carried (not a gate blocker):** frame-emission and early-stop both live on #24427's host path (`--no-diffusion-gpu-sampling`) at ~5.4 tok/s with a wide stop distribution (16–45/48 vs #24423 18–21/48 and CDG bf16 20–22/48) and 1/8 seeds degenerate; the fast path (GPU-sampling on, 30.4 tok/s sm_75) blocks both frames and early-stop ("reachability equals emission", #277). This GPU-sampling ⟂ {frame-emission, early-stop} coupling is a downstream knob-tuning / envelope-quality surface, consistent with INV-5 (GGUF inference-only-secondary; the whole-run fast path is the default audience product, per-step instrumentation the secondary mode at its measured cost) — see §Implementation Notes.

### 3. Packaging shape — the three handoff questions, decided where rung-1-independent

From `https://github.com/shanevcantwell/design-docs/blob/main/experiments/ComfyUI-DiffusionGemma/handoffs/2026-08-03-v0.5.0-released-next-kv-or-gguf.md:21`:

- **(a) Platform matrix — DECIDED (rung-1-independent).** Ship for the target population's center of mass: **CUDA on sm_75+ (Turing and up)**, source-built in CI from the pin. Rung-0 proved sm_75 CUDA build-green in 9m46s; the audience table (#131) centers on 3090/4090/4080-16GB (Ampere/Ada, sm_86/sm_89). CPU build is the fallback/CI-smoke tier (rung-0-green, 1m27s). Apple Metal and ROCm are **out of the initial matrix** — deferred to observed demand, named as an anticipated-failure boundary in §5, not built speculatively.
- **(b) Release-asset vs registry packaging — DECIDED (rung-1-independent): release-asset, not registry-bundled.** The binary is a **checksummed GitHub Release artifact** the pack *fetches and verifies*, not a blob committed to the repo or the ComfyUI registry. Reasoning: (i) the registry/pip path is source-distribution-shaped — a ~140 KB CLI is fine but its ~15 GB model dependency and platform-specific CUDA binary are not registry idioms; (ii) a Release artifact carries its own checksum and provenance line (this is the mechanism that makes the completion-bell "replicated beyond this working tree" property true, per the curator note); (iii) it keeps the engine *fetchable-and-verifiable* rather than *vendored*, which is the §5 no-fork invariant in packaging form. The pack degrades loud if the artifact is absent or checksum-mismatched (never a silent source-build fallback that pulls unpinned upstream).
- **(c) Maintenance commitment — DECIDED (rung-1-independent), scoped by the no-fork posture.** The commitment is bounded to: (i) re-pinning to a newer upstream SHA when the current pin goes stale/unbuildable — a *re-pin*, an atomic SHA bump + rebuild + checksum, **not** a merge/rebase of owned patches; (ii) the packaging glue (fetch/verify/invoke) and the protocol adapter (§5); (iii) **explicitly NOT** kernel maintenance, security backports, or ggml-master tracking — those belong to upstream, and if upstream dies the pack's exit is to re-pin to whatever the community's live head becomes, or (last resort) to let the pin freeze with a loud staleness declaration. The maintenance surface is deliberately the *protocol adapter + a SHA string*, nothing deeper.

### 4. Supersession — ADR-CDG-007's TBD is flipped to point here

This ADR **supersedes ADR-CDG-007**. Per the writing-adrs bidirectional-supersession convention, ADR-CDG-007's header `Superseded by: TBD` is edited to `Superseded by: ADR-CDG-020` (the one companion edit the convention itself authorizes; see §Implementation Notes). ADR-CDG-007's *node design* (three-node set, steering-vs-illumination socket rule, state/error/failure-path model at its lines 82–145) is **carried forward as substrate**, not discarded — what changes is the *engine-sourcing basis*: CDG-007 assumed a private local fork (`<dev-root>/llama.cpp-diffusiongemma`) and was rejected on that fork's un-obtainability (its OQ1); this ADR re-homes the same node design onto the **pinned-upstream-PR** posture, which is the obtainable-by-a-user answer OQ1 lacked.

### 5. Greenfield-invariant discipline — every invariant names its anticipated failure

Per the repo CLAUDE.md greenfield convention (harness-tools#18), each invariant this ADR introduces names the failure it prevents:

- **INV-1 — Protocol boundary, not engine internals.** The node consumes the engine only through the `DiffusionFrame` wire format (ADR-CDG-014) + step events; it never reaches into engine internals, kernel structs, or CLI-private state. *Anticipated failure prevented:* a node coupled to `c3fb972`-internal layouts breaks on every re-pin and makes the abandonability property (§Decision 1) false — the pack would own the engine's drift in practice even while disclaiming it in principle. `IDENTITY⊥ENVELOPE` applied to our own pin: any future engine speaking the protocol replaces this one with zero node changes.
- **INV-2 — Checksummed, verified, fail-loud fetch.** The release-asset binary is checksum-verified before invocation; absence or mismatch is a hard, declared node failure. *Anticipated failure prevented:* a silent fallback to an unpinned source-build (or an unaudited third-party prebuilt — declined per #131) would ship un-provenanced code under the pack's name, exactly the trust-endorsement this repo refuses to back.
- **INV-3 — Pin is a named SHA with recorded provenance, never "latest."** The build pins an exact upstream commit recorded in the packaging manifest, never a moving branch ref. *Anticipated failure prevented:* an unpinned checkout silently changes engine semantics between pack releases, breaking conformance reproducibility (the three-path conformance run's whole premise) and making bug reports irreproducible.
- **INV-4 — Staleness is declared, not hidden.** When the pin diverges from the community's live head (or upstream abandons it), the pack surfaces the staleness as a visible state, not a silent freeze. *Anticipated failure prevented:* a quietly-frozen pin lets users run months-diverged engine semantics believing they track the community, reproducing the #223-class divergence (working narrative vs recorded reality) at the binary layer.
- **INV-5 — GGUF stays inference-only-secondary to the transformers-primary path.** This ADR graduates GGUF from "deferred" to "shipped-secondary"; it does **not** demote the transformers/diffusers path (ADR-CDG-002/-004) from primary. *Anticipated failure prevented:* the instrument-panel research surface (resumable `CANVAS_STATE`, mid-loop constraint injection, ADR-CDG-006 step-window sampler) depends on the diffusers native-stepping drive seam; treating GGUF as the new primary would strand those research capabilities the GGUF whole-run path cannot express.

## Rationale

### Positive Consequences
- **The download population gets the pack's actual product.** ~52.7k monthly GGUF users currently self-serve a plain-generation CLI; this posture wraps it into the instrumented graph they lack, at their hardware's center of mass (16–24 GB).
- **The #15 ruling is honored in substance.** No compiled-and-hosted owned fork; a SHA reference + protocol adapter + checksummed build artifact. The maintenance surface is a string and a thin adapter.
- **Abandonability is a designed property, not a hope.** INV-1 (protocol boundary) means any engine speaking the frame protocol — including a future upstream-consolidated one — replaces the pin with zero node churn.
- **N=2 engines sharpen the `dgemma/` contract.** Running the pinned engine beside the transformers reference is the strongest conformance test of the frame contract — divergence is either a bug or an underspecified contract (a value channel #131 records).
- **Upstream-grade evidence falls out.** The conformance findings, knob science, and observable-surface design are RFC-grade evidence for the general-diffusion-server question upstream opened — a value channel available *because* the pack is the only party with real downstream consumers of per-step signals.

### Negative Consequences
- **Pin drift is a standing, unpaid-by-upstream cost.** Re-pinning is manual, triggered by staleness/unbuildability with no upstream clock. This is the honest price of not-forking (§Risk).
- **Two engines, two mental models.** The transformers-primary and GGUF-secondary paths diverge in load seam, drive seam, and — critically — state model (GGUF whole-run has no resumable `CANVAS_STATE`; ADR-CDG-007's state note at :130-135 carries forward). The pack keeps both straight.
- **Platform matrix is narrow at ship.** sm_75+ CUDA only; Metal/ROCm/older-arch users are out until demand is observed (§Risk names this).
- **The pin is upstream draft code.** `c3fb972`'s message is "Add tool calling" — an in-flight commit on an unmerged PR, not a release. Its stability is the community's hardening, not a maintainer's ratification.

## Alternatives Considered

### Option A — Owned fork the project compiles and hosts (the ADR-CDG-007-original posture)
Vendor a DiffusionGemma llama.cpp fork the pack builds and ships from its own repo/CI.

**Why rejected (deciding factor):** directly violates the #15 operator ruling ("never a fork this project compiles and hosts"), and acquires the full liability the ruling exists to prevent — rebase drift against ggml master, owned CI, security backports, provenance. ADR-CDG-007 was itself rejected on the obtainability half of this (its OQ1); this ADR does not re-open the losing side of that.

### Option B — Wait for upstream merge, ship nothing until then
Consume only mainline llama.cpp; ship GGUF support when/if DiffusionGemma merges to master.

**Why rejected (deciding factor):** there is no visible clock. #24423 has been draft since 2026-06-10; #24427 is stale ~11 days as of 2026-08-03 — zero maintainer engagement on either; the consolidation assumption from #131's earlier comment is explicitly *not holding*. Waiting strands the ~52.7k-download population on self-serve `gh pr checkout` for an unbounded interval — abdicating the exact on-ramp this issue exists to build.

### Option C — Registry/pip-bundled engine binary
Package the built CLI as part of the pack's registry/pip distribution.

**Why rejected:** the registry/pip idiom is source-distribution-shaped and does not carry a ~15 GB model dependency or a platform-specific CUDA binary well; a checksummed Release artifact (Decision 3b) carries provenance and checksum natively and keeps the engine fetchable-and-verifiable rather than vendored (which would re-approach the Option-A fork liability by another door).

### Option D — Endorse third-party prebuilt binaries (gbuznote-beep / WayneTechLab trackers)
Point users at existing community prebuilt-binary trackers instead of building.

**Why rejected (deciding factor):** endorsing an unaudited third-party binary under the pack's name is a trust endorsement this repo cannot back (#131 declined this explicitly). CI-building from a named SHA (Decision 1) gives provenance the pack *can* stand behind; a checksum on someone else's binary certifies only that we downloaded what we downloaded.

## Open Questions

- [x] **(A) Pin selection: #24423 vs #24427.** **RESOLVED (2026-08-06 ratification gate) — pin = #24427 @ `dd0cf04459b0c4f43aa6667dbc0879ac0cd50323`.** Criterion 1 (decisive, frame-contract reachability) REACHED on #24427 (#277 Tier-0 frame stream, example-layer, ~0.11% tax, no kernel reach) vs #24423 argmax-only; criterion 3's counterweight toward #24423 falsified as a disqualifier by the device-path mechanism probe ([5212022958](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131#issuecomment-5212022958): TOGGLEABLE, 16/16 vs 6/6, zero crossover). Both live criteria favor #24427. Adjudication: [5212054211](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131#issuecomment-5212054211). See §Decision 2 resolution record.
- [x] **(B) H0: the visibility delta (Tier-0 `DiffusionFrame` emission) is implementable in the example/serving layer without kernel work.** **CONFIRMED (#277).** A per-step frame stream built and measured on #24427's example layer emitting the host-resident `entropy` array: 48/48 × 3 seeds, Tier-0 core exact-match, ~0.11% emission tax (noise floor at whole-process granularity), no kernel reach. Contract is PARTIAL — three fields carry right content under a different name/shape (fail-loud KeyError, **not** the ADR-CDG-001 lying-payload mode), two + `pinned_mask` absent — every gap an example-layer adapter/naming concern. The falsification branch (kernel work → scope widens → return to operator) is foreclosed by the built artifact. The naming-seam adapter is a downstream implementation note.
- [x] **(C) H0: an unoptimized, visible engine reaches usable tok/s on 3090-class hardware.** **CONFIRMED.** On-host Q4_K_M / sm_75 (Turing) #24427 fast path 30.4 tok/s ([5210392752](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131#issuecomment-5210392752) carried; corroborated by the mechanism-probe defaults rows at 30.3–30.5 tok/s); a 3090 / sm_86 at Q4_K_M clears "usable" a fortiori. The shipped fast-path envelope is a real product.
- [x] **(D) H0: #24427's top-k self-cond truncation does not materially alter convergence.** **RESOLVED as input to (A) criterion 3.** #278's 21-run self-cond sweep (k ∈ {1,8,32,128,256,131072,262144} × 3 seeds) showed all 48/48, zero early-stops at any width including k=1 — **disconfirming self-cond width as the cause** of #24427's default-config non-convergence. The residual early-stop question was then resolved by the device-path mechanism probe (TOGGLEABLE), not by self-cond width. Feeds (A) per the 2026-08-04 gate-findings clause 2.
- [ ] **(E) Platform-matrix expansion (Metal/ROCm/older-arch).** **Resolution:** deferred to observed demand post-ship; re-open as a packaging follow-up when the audience table shows a non-CUDA population, not speculatively.

**Resolution plan:** the ratification gate is **three** named questions, not four — (D) is not a separate acceptance gate but feeds **into (A)** as the empirical input to §Decision-2 pin-selection criterion 3 (self-cond fidelity), so a cold reader should read (A)/(B)/(C) as the gate and (D) as an input to (A). All resolve on the single GPU-window-gated rung-1 probe (operator-scheduled): this ADR moves `proposed → accepted` when rung-1 returns and (A) selects a pin (informed by (D)), (B) confirms example-layer implementability, and (C) shows a usable envelope. (E) is a post-acceptance packaging trigger. All decisions above rung 0–1 execution remain operator-gated per #131.

> **GATE MET (2026-08-06).** (A) RESOLVED (pin #24427 @ `dd0cf04`), (B) CONFIRMED, (C) CONFIRMED — the three-question line-120 gate is satisfied and Status is flipped `proposed → accepted`, operator-vetoable per the standing #131 gate. (D) resolved as input to (A). (E) remains a post-acceptance packaging trigger.

## Supersession Relationships

**Supersedes:** ADR-CDG-007 (the GGUF/fork alpha node design) — carried forward as substrate (three-node set, steering-vs-illumination socket rule, state/error/failure-path model), re-homed from its rejected private-fork basis onto the pinned-upstream-PR posture. ADR-CDG-007's header is edited to `Superseded by: ADR-CDG-020` per the bidirectional convention (§Implementation Notes).

**Amends (on acceptance):** ADR-CDG-002 — graduates its "GGUF as graduation-triggered inference-only backend" from deferred to shipped-secondary; does **not** flip its transformers-primary posture (INV-5). ADR-CDG-004's diffusers drive seam remains the primary/native-stepping path.

**Superseded by:** TBD — a future upstream-consolidated engine, if DiffusionGemma merges to ggml master, would supersede the *pin* (not the node design or the protocol boundary, which are engine-agnostic by INV-1). That supersession is a SHA-to-mainline swap behind the protocol seam, not a redesign.

## Implementation Notes

This ADR authorizes exactly two file touches at authoring time — itself, and the one bidirectional-supersession edit the writing-adrs convention builds into the ADR act. All packaging/build/adapter changes below are *described here for downstream execution*, not performed by this design act; each is gated on ratification and (per repo waterfall) an independent plan.

| File | Change Type | Description |
|------|-------------|-------------|
| `decisions/adr-cdg-020-gguf-engine-sourcing-pinned-pr-branch.md` | Created | This decision record |
| `decisions/adr-cdg-007-*.md` | Modified (supersession only) | Header `Superseded by: TBD` → `Superseded by: ADR-CDG-020`; status line annotated that its node design is carried forward as substrate by ADR-CDG-020 |
| `ROADMAP.md:51-56` | Deferred (NOT this act) | Replace "rung 4 / owned pin / owned provenance" shorthand with the #131 rung sequence and the corrected "pinned upstream SHA, not owned provenance" framing — a downstream doc edit, per #223's disposition, gated on ratification |

### Ratification implementation-note (added 2026-08-06 on acceptance) — the pin #24427 GPU-sampling coupling

The device-path early-stop mechanism probe ([5212022958](https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131#issuecomment-5212022958)) surfaced a three-way knob interaction on the selected pin (#24427 @ `dd0cf04`) that the downstream packaging/adapter work must carry — it is a tuning surface, not a design defect, and it does not reopen (A)/(B)/(C):

- **GPU-sampling ON (default, fast path):** ~30.4 tok/s on sm_75; per-step frames blocked (the on-device denoise loop `continue`s past the emission site — "reachability equals emission", #277); **no** entropy-bound early-stop (48/48 ceiling).
- **GPU-sampling OFF (`--no-diffusion-gpu-sampling`, host path):** frames **emit** and early-stop **fires**, but ~5.4 tok/s (>5× slower), a wide/unstable stop distribution (16–45/48 vs #24423 18–21/48 and CDG bf16 20–22/48, both ~3–4-step spread), and 1/8 seeds degenerate (seed13, 16/48, repeated `* * *` noise).

Packaging consequence, consistent with **INV-5**: the pack's **default GGUF product is the whole-run fast path** (audience mode, no per-step instrumentation); the **instrumented per-step / early-stop mode runs the host path at its measured cost**, and is the secondary surface — the instrument-panel research surface (resumable `CANVAS_STATE`, mid-loop constraint injection) stays on the transformers/diffusers primary path, not GGUF. The naming-seam adapter for the three PARTIAL-contract fields (`canvas`→`canvas_ids`, `committed_fraction_per_example`→`committed_fraction`, `effective_entropy_bound`→`entropy_bound`; #277) is a downstream adapter task, tracked with the packaging plan.

## References

- #131 (llama.cpp engine thread; rung-0 readback, conformance matrix, seam map, rung-1 plan) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/131
- #15 (the 2026-07-06 no-fork ruling) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/15
- #223 (the divergence this ADR resolves) — https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/223
- [ggml-org/llama.cpp#24423](https://github.com/ggml-org/llama.cpp/pull/24423) (pin candidate, `c3fb972`), [#24427](https://github.com/ggml-org/llama.cpp/pull/24427) (competing candidate)
- ADR-CDG-007 (preserved node design), ADR-CDG-002/-004 (transformers-primary posture), ADR-CDG-014 (`DiffusionFrame` wire-format / abandonability seam), ADR-CDG-018 (`loop.py` decomposition)
- `https://github.com/shanevcantwell/design-docs/blob/main/experiments/ComfyUI-DiffusionGemma/handoffs/2026-08-03-v0.5.0-released-next-kv-or-gguf.md:21` (the three packaging questions)


## Gate findings resolved (2026-08-04)

Independent Opus design-gate review returned **PASS_WITH_FINDINGS** (zero blocking; the no-fork axis ruled honest and enforced, supersession executed correctly). Three recommendations resolved into this artifact:

1. **Date provenance corrected (Decision 1 rejected-wait bullet; Option B).** "Both PRs draft/stale since 2026-06-10" over-attributed #24423's open date to #24427. Restated: #24423 draft since 2026-06-10; #24427 stale ~11 days as of 2026-08-03; zero maintainer engagement on either.
2. **Ratification gate reads as three, not four (Open Questions resolution plan).** Added a clause: (D) (the #24427 self-cond-truncation H0) resolves *into* (A) as §Decision-2 pin-selection criterion 3's empirical input, not as a separate acceptance gate — the gate is (A)/(B)/(C).
3. **"Compiles" defined (Decision 1).** Clarified the #15 ruling's sense: owns-and-maintains-a-forked-codebase (forbidden) vs. a reproducible CI build of a named upstream commit (the provenance mechanism, not a fork) — the latter becomes the former only if we carry patches, which INV-1 and Decision 3c forbid.

---
