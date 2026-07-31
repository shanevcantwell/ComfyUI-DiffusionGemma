# Handoff: 2026-07-30 night — hands-on shakeout banked; NEXT SESSION: examples + workflow polish → 0.5.0 ship

**Supersedes** `2026-07-30-late-183-autoround-fixed-go-restored.md` (same day, later). **Machine state unchanged: `pre-0.5.0-release` @ a68e29d, run-4 gate PASS, 0.5.0 machine-GO.** The operator is sleeping on the ship call and has scoped ONE more session before it.

## The next session's mandate (operator, verbatim intent)

"I want a decent workflow, and a couple good examples" for the 0.5.0 release. Concretely:

1. **Ship-quality example workflows** — api.json + ui.json pairs, both conformance-clean (#181's live-validation rule: required widgets present regardless of defaults) and AESTHETICALLY decent (deliberate node layout, groups, no spaghetti — the operator judges visually and holds the veto; draft → screenshot → operator review). Candidate set: (a) basic bf16 generation + trace display; (b) INT4 autoround generation (the new 0.5.0 headline — works whole-fit, 33GiB); (c) the two-node kv-cache path **bf16 only** (INT4+Encode is broken, #187, 0.5.1). Closing the #174 .ui.json-staleness residual IS this work.
2. Known-good widget values for example authoring (live-proven tonight): kv-cache pair = Encode.text "Why do birds suddenly appear?" → Denoise.prompt "whenever they are near", seed 7, steps 48, t 0.4/0.8, entropy_bound 0.1, confidence 0.005, gen_length 256 (examples/kv-cache-tier1.api.json is the wiring reference). Local run defaults for INT4: entropy_bound 0.05 ran 48/48 correct on a count-table problem (see #186).
3. **Then, on operator GO: 0.5.0 ship mechanics** — CHANGELOG (two banked release-note inputs on #145: the 5090 "probably, no guarantees" posture line and the #187 known-issue line) · version bump (run-log header currently misreports 0.4.0, #175 wart) · PR #135 disposition (likely superseded) · merge `pre-0.5.0-release` → `main` (carries ARCHITECTURE.md trim → closes #134/#137; #183 fix) · tag · registry publish · re-check 0.4.1 registry activation (`Pending`) · drop stash@{0} · close #169, #183, #145.
4. **#175 scope call rides the sleep**: minimal tooltips in 0.5.0 or 0.5.1 (four loader warts + the tooltip ask are all banked on #175).

## Tonight's hands-on shakeout (operator drove the merged line live; everything banked)

- **INT4 field-proven by operator hand**: 48-step run @ entropy_bound 0.05, 2.03 s/it, output VERIFIED CORRECT on a global-consistency count-table problem (#28-class). Run JSONL: `/srv/dev/DG-runs` (a FILE — path-coercion wart, #175). Convergence anomaly (48 steps vs historical 20–30) + INT4-vs-bf16 near-parity s/it → banked as #40 amendments + **#186** (new: quantifying quantization via the Tier-2 capture surface — first customer for P-C).
- **#187 (bug, 0.5.1)**: DGemmaEncode crashes under whole-fit loads — `kv_cache.py:306/309` mint CPU tensors, no hooks to move them in the no-spill regime. INT4+Encode was unreachable pre-#183. bf16 kv-cache is the supported 0.5.0 combination (live-proven runs 3+4). Operator scoping: "autoround and split sampler are different releases."
- **#188 (bug, 0.5.1)**: DGemmaDenoise live view — Python emits correctly on `dgemma.denoise.step`; `live_view.js` hardcodes the sampler's event name + node class. One-mint miss, JS-side; fix = capability single-sourced.
- **Loader UI warts (all on #175)**: combo has no unset state → HF-default path unreachable from UI (workaround: bf16 snapshot symlinked at `models/text_encoders/diffusiongemma-26B-A4B-it-bf16` on the dev host, operator-suggested); trailing-slash path coerced to file silently; version misreport.
- **Post-0.5.0 research scoping**: Ampere/gptqmodel kernel test → #16 lane @ 0.5.1/0.6.0 (broad Turing-kernel-wall H0 falsified by tonight's runs); 5090 rental deferred, trigger = adoption (#131); metric discipline for all perf claims: delivered tok/s, decomposed into s/it × steps-to-freeze (#40).

## Standing state

Rig `/tmp/smoke-050` + run1–4 evidence VOLATILE until reboot (durable readbacks on #145). `comfyui_detail.log` untracked on main (pre-session, undispositioned). PR #185 branch retained. 0.5.1 pile forming: #187, #188, #175, #119, #184 candidate. Doctrine banked today: OD#41 (schema+DRY into physics), #167 schema-first signal, HT#225 instances 3–4.
