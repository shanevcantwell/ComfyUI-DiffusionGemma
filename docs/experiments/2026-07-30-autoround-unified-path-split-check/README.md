# Experiment — autoround unified-path split-check probe

**Status:** H0 FALSIFIED (split-fails outcome) — observed 2026-07-31 UTC,
after the Part-1 checkpoint blocker was cleared by the operator/orchestrator.
**Issue:** #183. **Plan:** signed `plan-autoround-placement.md` (ratified
2026-07-30, Opus gate PASS, second cycle). **Host:** `inference-host`, Quadro
RTX 8000 48GB; kept-resident llama-server (embeddinggemma) holding 1412 MiB
as recorded baseline, not killed. **Raw artifacts:** [`runs/`](runs/) · probe
script: [`probe_split_check.py`](probe_split_check.py).

This file is chronological: H0 (stated before observation), then Part 1 (the
first attempt's BLOCKED exit — checkpoint cache incomplete), then Part 2 (the
observation after the unblock, and the fork it resolves).

## H0 (stated before observation, per #101 floor)

> The accelerate-dispatched, cross-device forward pass through an
> AutoRound-quantized (`.qweight`) module that `device_map="auto"` has placed
> on CPU (i.e., under an artificially forced split on this 48GB host, since the
> INT4 checkpoint fits whole naturally) is numerically correct — matches a
> fits-whole baseline generation, within tolerance.

*Falsified if:* the forced-split forward errors, or its output diverges from
the fits-whole baseline beyond tolerance, or `_retie_lm_head`/
`_assert_no_meta_tensors` cannot reconcile the tied `lm_head.weight` under the
forced-split `hf_device_map`.

## PART 1 — first attempt (2026-07-30/31): BLOCKED — checkpoint precondition not met, H0 not observed
(historical record; resolved by the unblock recorded in Part 2)

Per the ground brief: *"The INT4 checkpoint ... was loaded successfully on
this host 2026-07-23 — it should be in the HF cache; verify before assuming
(local_files_only probe or huggingface-cli scan). If absent, that is a BLOCKED
exit, not a download decision."* Verification was run before any probe script
or source edit, per the plan's own MANDATORY-before-source-edit sequencing.

### Verification method

1. `local_files_only=True` resolution attempt via `huggingface_hub.snapshot_download`
   (ComfyUI's own Python env, `/tmp/smoke-050/venv`, `huggingface_hub==1.26.0`):

   ```
   EXCEPTION TYPE: IncompleteSnapshotError
   EXCEPTION: The cached snapshot for 'Intel/diffusiongemma-26B-A4B-it-int4-AutoRound'
   (revision 'main', commit 63a0dca884ce803e03567fd9d57359064104c471) is incomplete:
   14 file(s) are missing (.gitattributes, README.md, chat_template.jinja, ...
   (11 more)). Outgoing traffic is disabled ('local_files_only=True'). Re-run the
   download with network access to complete the snapshot.
   ```

2. `huggingface_hub.utils.scan_cache_dir()` authoritative readback of what is
   actually materialized on disk for this repo/revision:

   ```
   repo_id: Intel/diffusiongemma-26B-A4B-it-int4-AutoRound
   size on disk: 4.1G
    revision: 63a0dca884ce803e03567fd9d57359064104c471 complete: 4 files, size: 4.1G
     - config.json                          10,778 B
     - model-00006-of-00006.safetensors 2,657,733,096 B
     - model.safetensors.index.json      7,288,449 B
     - model_extra_tensors.safetensors 1,476,400,856 B
   ```

   The `model.safetensors.index.json` weight_map declares 6 shards
   (`model-00001-of-00006.safetensors` through `model-00006-of-00006.safetensors`,
   total_size 30,576,322,776 bytes per its own metadata) plus
   `model_extra_tensors.safetensors`. **Shards 1 through 5 are entirely absent**
   from the snapshot — only shard 6 and the extra-tensors file are present.
   `du -sh` on the repo's cache directory reports ~30G, but that figure is
   dominated by orphaned `.incomplete` partial-download blobs from prior failed
   fetch attempts (18 distinct `.incomplete` files found, several multi-GB,
   under `blobs/`), not by usable weight data — a `du`-only check would have
   been misleading; `scan_cache_dir()` was used instead to ground the claim in
   what `from_pretrained` can actually resolve.

### Disposition

This is the plan's named bounce condition: *"Checkpoint absent from cache →
BLOCKED (download is an operator/orchestrator call)."* An incomplete cache
missing 5 of 6 model shards is functionally absent for both
`from_pretrained(local_files_only=True)` and for this probe's own load step —
neither the bare-script mechanism probe (H0) nor the in-rig capacity leg (F5)
can run without the full weight set. Per the plan's own evidence-tier
discipline (F5), no bare-script or in-rig run may substitute for the missing
weights; there is no honest way to ground H0 without them.

**Note on the 2026-07-23 "loaded successfully" record cited in the ground
brief:** issue #183's own ledger (operator finding, 2026-07-30) already
identifies that record as bare-process-only evidence ("the '~30GB fits' claim
was grounded bare-process only ... load + single-forward from a script on an
idle card (2026-07-23 record)") — i.e., even if that load succeeded at the
time, it was never an in-rig (#163/#59 driver) run, and the plan's F5 finding
already discounts it as the wrong evidence tier for a fits/works verdict. Its
current absence from cache is consistent with either post-hoc cache eviction/
cleanup or a partial/interrupted state at the time; this experiment does not
attempt to adjudicate which — only that the cache is not usable *now*.

## No H0 observation was made; no source edit was attempted

Per the plan's explicit sequencing ("Probe before any source edit") and the
execution contract's bounce condition, this is a typed **BLOCKED** exit —
external precondition, resumable once the checkpoint is re-fetched with
network access. This is an operator/orchestrator download decision, not
something resolved by this session.

## What would unblock

Re-run `snapshot_download('Intel/diffusiongemma-26B-A4B-it-int4-AutoRound')`
with network access (outside `local_files_only`) to complete the snapshot
(shards 1-5 + the 14 other missing files: `.gitattributes`, `README.md`,
`chat_template.jinja`, tokenizer files, etc.), then re-run this probe.

## Measurement discipline for the re-run (banked ahead of the unblock, per
operator/orchestrator addendum 2026-07-31 — not yet exercised, no data below
is fabricated)

The bf16 instrument-trap record (`docs/experiments/bf16-fit-mechanism/README.md`,
"Three instrument traps recorded") is binding on this probe's re-run and on
step 9's in-rig leg. Read it before instrumenting either. Specifically:

1. **Load-end `nvidia-smi` is NOT the capacity datapoint.** This load path is
   lazy: `device_map="auto"` implies meta-device skeleton construction plus
   direct-to-placement streaming; peak VRAM appears during the forward pass /
   denoise steps, not at `from_pretrained` return. Use
   `torch.cuda.max_memory_allocated()` (reset via
   `torch.cuda.reset_peak_memory_stats()` before the run) plus `nvidia-smi`
   sampled continuously across the live steps, not just at load-end. Report
   both numbers.
2. **The mmap-backed CPU/disk spill is RSS-invisible.** The bf16 probe found
   only a ~0.6 GiB `MemAvailable` dip for a 10.25 GiB spill — offloaded params
   report device `meta`, not `cpu`, in `named_parameters()`, and are paged in
   per-forward-pass against the safetensors file rather than materialized as
   anonymous RAM. Do NOT use RSS/`MemAvailable` to infer spill size for the
   INT4 checkpoint either. Read `hf_device_map` / offload accounting directly
   and bank the device_map JSON verbatim, exactly as
   `docs/experiments/bf16-fit-mechanism/runs/device_map.json` did.
3. **The forced-split leg's `max_memory` cap governs WEIGHT placement only,
   not activations/onload buffers.** The bf16 record shows peak allocation
   (44.43 GiB) *exceeding* the `max_memory` GPU cap (44 GiB) it was run
   under. This probe's split verdict (H0 confirmed/falsified) must therefore
   come from observing whether the cross-device forward pass completes
   correctly or errors/diverges — never from "peak stayed under the cap,"
   which the bf16 record already shows is not a safe inference.
4. **A successful load with no forward pass proves nothing.** This is the
   exact gap in the 2026-07-23 record this plan exists to close — see
   `docs/handoffs/2026-07-23-int4-autoround-loaded.md`: that record documents
   "load + single forward pass verified" on what was then a complete
   checkpoint, never an in-rig/multi-step generation run, and the checkpoint
   it was run against is not the one present in cache today (see "Outcome"
   above — current cache is missing shards 1-5). Every green claim in this
   experiment's eventual re-run entry must name the forward-pass evidence
   (shape, step count, coherence check) behind it, not load success alone.

The orchestrator/operator seat may independently poll `nvidia-smi` at its own
cadence during the re-run for observability; that poll is not this probe's
instrument of record — the in-process `torch.cuda.max_memory_allocated()` +
continuous in-run sampling above is primary, per point 1.

---

## PART 2 — probe run (2026-07-31 UTC, after unblock)

### Unblock provenance (recorded verbatim from the coordinator)

The checkpoint was re-provisioned by the operator at the canonical path
`/srv/dev/ComfyUI/models/text_encoders/diffusiongemma-26B-A4B-it-int4-AutoRound/`
(plain directory — loaded via `from_pretrained` on that directory, NOT via HF
hub cache resolution). Coordinator-verified provenance: all 7 binary payloads
(6 shards + `model_extra_tensors`) byte-exact vs the HF manifest;
`tokenizer.json` byte-exact; 8 small text/JSON files are CRLF variants
(content-identical after `\r` strip — Windows hand-copy transport artifact;
JSON-legal, safetensors header parse all green: 14400/14452/14451/14451/
13376/362/2 tensors); `.gitattributes` absent. Byte-canonicalization pending
an operator chmod — cosmetic, does not gate the probe. The model dir is
read-only to this seat; no writes were attempted.

### Environment (both legs)

`/tmp/smoke-050/venv` (ComfyUI's own env): Python 3.12.3, torch 2.13.0+cu130,
transformers 5.13.0, auto_round 0.14.2. Quadro RTX 8000 48GB; co-resident
llama-server holding 1412 MiB at leg start (`nvidia-smi`: 1641 MiB used,
46758 MiB free card-wide) — baseline recorded, process untouched. Generation
args grounded in `tests/test_integration.py`: seed=0, num_inference_steps=8,
gen_length=64, prompt "Why is the sky blue?". Legs ran as separate processes
(clean CUDA state each); a 1 s `nvidia-smi` sampler logged the card-wide view
per leg (`runs/nvidia_smi_*.log`).

### Leg 1 — baseline (natural, fits-whole): COMPLETED

Unified shape under test: `device_map="auto"`, `dtype="auto"`,
`local_files_only=True` — the exact kwargs the fix ships, no `max_memory`.
Readback (`runs/report_baseline.json`, `runs/device_map_baseline.json`):

- **Composition `dtype='auto'` ∘ `device_map="auto"` works in transformers
  5.13.0** — the named bounce condition did not fire.
- load 28.0 s; 28.55 GiB allocated post-load.
- `hf_device_map` came back **empty** on the fits-whole INT4 load —
  accelerate placed everything on cuda:0 without recording a per-module map.
  `_resolve_device` correctly fell through to its first-parameter fallback
  (`cuda:0`). Instrument note for future code: do not assume a populated
  `hf_device_map` under `device_map="auto"` when the model fits whole.
- `_retie_lm_head`: `lm_head.weight` real on cuda:0 after retie;
  `_assert_no_meta_tensors` passed. The prior-art machinery holds unchanged
  on the unified path in the fits-whole regime.
- generation: 8/8 steps, 24.6 s wall (3.07 s/step including the 10.8 s
  warm-up first step; steady-state ~2.2 s/step), committed_fraction 0.5742,
  process-wide CUDA peak **30.67 GiB** at forward-touch — the load-end
  number (28.55 GiB) understates by 2.1 GiB, exactly the lazy-load
  instrument trap the discipline section above predicted.
- output text coherent modulo uncommitted-position noise (committed_fraction
  0.57 at 8 probe steps — the short-step probe regime, not a defect signal).

### Leg 2 — forced-split (`max_memory={0: "24GiB", "cpu": "40GiB"}`): ERRORED AT LOAD

Cap chosen to force ~4.5 GiB of the 28.55 GiB weight set off-GPU. The leg
**crashed inside `from_pretrained` itself** — before any forward pass could
exist (`runs/report_forced-split.json`, full traceback banked there):

```
ValueError: weight is on the meta device, we need a `value` to put in on 0.
  at accelerate/utils/modeling.py:299 set_module_tensor_to_device
  via accelerate/hooks.py:305 init_hook <- attach_align_device_hook_on_blocks
  via transformers/integrations/accelerate.py:404 accelerate_dispatch
  via transformers/modeling_utils.py:4388 from_pretrained
```

### Verdict: H0 FALSIFIED — one probe pass, unambiguous

H0's falsification criterion fired a fortiori: not only does no correct
cross-device forward exist — the *load* itself errors during accelerate's
align-device hook initialization when dispatch tries to split the AutoRound
model. Discrimination is clean: the same unified shape + `max_memory` cap
pattern loads and generates green for the bf16 checkpoint on this host
(`docs/experiments/bf16-fit-mechanism/`: 44 GiB cap, 13 modules offloaded,
2.2 s/step), so the crash is specific to the INT4-quantized checkpoint under
a split. This formalizes #183's field-failure family ("index and next blocks
being on different devices"): **a CPU/GPU split of this INT4 checkpoint does
not merely degrade — it cannot load.**

### Fork applied (per the plan's total mapping — pre-authorized, no orchestrator return)

**Split-fails arm ships:**

1. The unified path still lands (`device_map="auto"`, dtype-only per-quant
   residue) — correct and field-proven for bf16, correct for INT4 wherever
   the checkpoint fits whole (leg 1).
2. PLUS a fail-loud pre-load VRAM precondition for `quant="autoround"` only:
   before `from_pretrained`, query free VRAM and compare against the
   checkpoint's measured whole-fit floor (weights 28.55 GiB measured in leg
   1; floor set at 30 GiB to cover accelerate's dispatch margin), raising
   `RuntimeError` naming both numbers and the remedy (free VRAM, or use
   `quant='none'`, whose spill path is field-proven) rather than letting
   dispatch attempt the split that leg 2 proves cannot load.
3. Split-capable INT4 (block-wise onload — the richer remedy) stays banked
   as deferred future work, per the plan's F4 disposition under this outcome.

### Evidence-tier note (F5, binding)

This bare-script probe grounds MECHANISM ONLY: (a) the unified shape
composes and loads the INT4 checkpoint whole; (b) a forced split crashes at
dispatch. The fits/works capacity verdict for the unified path in ComfyUI
comes from the in-rig leg (S-A-autoround through the #163 rig / #59 driver),
banked separately — see the PR body for which run discharges F5.

---

## PART 3 — in-rig capacity leg (F5 discharge), 2026-07-31 UTC

The run that grounds the fits/works capacity claim at the correct evidence
tier (F5: in-rig only). Vehicle: `examples/smoke-tests/ping-smoke-autoround.api.json`
(S-A-autoround — `DGemmaLoader` quant='autoround' via the local-folders path →
`DGemmaSampler` 48-step budget → `PreviewAny`), submitted through the
black-box driver (`tests/e2e/driver.py`) against the standing `/tmp/smoke-050`
rig, ComfyUI headless on 127.0.0.1:8188, rig pack checkout at this branch
(`508b93c`). Model resolved via a symlink
`ComfyUI/models/text_encoders/diffusiongemma-26B-A4B-it-int4-AutoRound` →
the canonical read-only dir (loader union scan, #150; real-dir symlink
followed, #182). Co-resident llama-server baseline: first sampler reading
1805 MiB used card-wide pre-load (1641 MiB at teardown readback).

**Result: GREEN.**

- Prompt executed in **66.78 s** end-to-end (includes the in-ComfyUI model
  load — weights streamed in-process; `[INFO] ... model loaded on cuda:0`).
- Generation: converged/committed at **22/48 steps**, steady-state
  **1.53–1.54 s/step** (first step 10.1 s warm-up), coherent output for the
  "ping" prompt: `"I'm here! How can I help you today?"` — contrast the
  ~10x-slow field report this issue opened with (#183).
- **Peak VRAM 33,831 MiB card-wide (33.04 GiB)** across 162 one-second
  samples (`runs/inrig_gpu_sample.csv`) — ≈31.3 GiB net of the co-resident
  baseline, consistent with the bare-probe's 30.67 GiB in-process peak. No
  overflow, no stranded-meta raise, no error in the ComfyUI log.
- History entry banked verbatim: `runs/inrig_s-a-autoround-entry.json`.
- Teardown: ComfyUI SIGTERM'd; card back to exact 1641 MiB baseline.

**This run is the in-rig autoround analog of bf16's 42.4GiB/2.2s-step
datapoint: ~33.0 GiB card-wide peak (~31.3 GiB net), ~1.54 s/step, through
the real unified load path in ComfyUI.** It discharges F5 for the "unified
path works" claim (step 1's pre-PR leg). The #163 release-gate smoke re-run
(post-Opus-PASS, step 9's live leg) remains the release gate's own,
separately-scheduled confirmation.
