# bf16 fit mechanism on inference-host — lazy-load (mmap) offload quantified, 52.7 GiB checkpoint on a 48 GB card

**Status:** H0 falsified; alternative confirmed (with magnitude correction) · **date:** 2026-07-30
**Host identity (per the #101 floor amendment, 2026-07-30):** inference-host, Quadro RTX 8000
48 GB, Turing sm_75, Ubuntu, 62 GB RAM. torch 2.12.1+cu130, transformers 5.13.0, repo `.venv`.
**Raw artifacts:** [`runs/`](runs/)

## Reasoning at decision time

`load_model(quant='none')` loads the ~53.6 GB bf16 `google/diffusiongemma-26B-A4B-it`
checkpoint on a 48 GB card. Naive arithmetic says it cannot fit GPU-resident, yet historical
integration runs on this box read back coherent, reasonably fast generation with **no visible
CPU-RAM spill** from the operator's vantage — while `dgemma/model.py`'s own docstring
(`model.py:218-243`) documents the load as CPU-spilling under `device_map="auto"`. These two
first-hand accounts of the same code path disagree, and this experiment exists to resolve
which is right — or whether both are, from different instruments.

## H0 (stated before observation, 2026-07-30 session)

`load_model(quant='none')` places all layers GPU-resident (no CPU offload) despite naive
arithmetic; the record's "spills to CPU" claim is wrong.

**Alternative:** placement includes CPU layers; spill is real but mmap-invisible (i.e. it
never shows up as anonymous RSS growth, so RAM-based instruments miss it).

*Falsified if* the device map shows any non-`cuda:0` placement for real weight tensors.
*Alternative confirmed if* CPU/meta placement is present but does not register as
`/proc/meminfo` MemAvailable pressure.

## Method

Instrumented load replicating `load_model`'s exact `quant='none'` kwargs (`device_map="auto"`,
`dtype=torch.bfloat16`, `local_files_only`), with one deliberate deviation: a
`max_memory={0: "44GiB", "cpu": "40GiB"}` hard cap added for safety (`load_model` itself has
no `max_memory` passthrough — this is a probe-only addition, not a claim about production
behavior). 1s-interval samplers polled `nvidia-smi` and `/proc/meminfo` across the load and
the run. `hf_device_map` was dumped in full. A live `run_diffusion` followed (seed=0, 8 steps,
`gen_length=64` — args grounded in `tests/test_integration.py:54-60`).

**Safety rationale.** On this host, VRAM overflow can hard-reboot the box. The caps make the
only failure mode a clean refusal rather than a crash. First attempt honestly refused at a
≥46 GiB preflight gate (45.66 GiB free at check time, against a resident llama-server); the
gate was lowered to 45 GiB with the cap left unchanged, and the probe rerun.

## Result: H0 FALSIFIED; alternative CONFIRMED with magnitude correction

Load succeeded in 30.5 s under the cap.

**Placement** (`runs/device_map.json`, `runs/load_report.json`):

| device | modules | size | params |
|---|---|---|---|
| `cuda:0` | 57 | 42.42 GiB | 22.78 B |
| offloaded (`meta`) | 13 | 10.25 GiB | 5.50 B |

Sum: 52.67 GiB — the full 28.3 B-param bf16 checkpoint. Offloaded modules: encoder LM layers
27–29, decoder layers 27–29, both final norms, both rotary embeddings, `vision_tower`,
`embed_vision`, decoder `self_conditioning`.

`/proc/meminfo` MemAvailable dipped only ~0.6 GiB across the **entire** load (`runs/mem.log`)
— the offload is mmap-backed against the safetensors file, never materialized as anonymous
RAM. This is why RSS/RAM inspection sees no spill even though 10.25 GiB of weights are not
GPU-resident.

**The mechanism is lazy-load, in two distinct senses (operator's term, both grounded here):**

1. **The offloaded slice is lazy-loaded.** The 10.25 GiB of CPU-offloaded weights are never
   materialized as resident RAM — they are mmap-backed against the safetensors files and
   paged in per forward pass, on demand. This is why MemAvailable moved only ~0.6 GiB across
   the whole load. Device `meta` (not `cpu`) in `named_parameters()` is the visible signature
   of this laziness — see instrument trap 2 below.
2. **The load path itself is lazy.** `device_map="auto"` implies meta-device skeleton
   construction followed by direct-to-placement weight streaming — the checkpoint is never
   fully materialized in host RAM at any single point during load. Consistent with the 30.5 s
   load time and the flat MemAvailable trace across it.

**Live run** (`runs/run_report.json`): 8/8 steps, 17.7 s (2.2 s/step), `committed_fraction`
0.980, coherent output — consistent with historical ~2.6 s/step integration readbacks.

## Three instrument traps recorded (verbatim-grounded)

1. transformers itself reports *"Some parameters are on the meta device because they were
   offloaded to the cpu"* — an RSS-only inspection sees no spill and would miss this entirely.
2. Offloaded params report device **`meta`**, not `cpu`, in `named_parameters()` — a check for
   `p.device == 'cpu'` counts zero and wrongly concludes an all-GPU placement.
3. Run-time **peak allocation** (44.43 GiB) **exceeds** the `max_memory` GPU cap (44 GiB):
   `max_memory` governs weight placement only, not activations + onload buffers. Card-wide
   peak during the run: 47319 MiB used / 1080 MiB free.

## Analysis / consequences

- **(a)** The historical record's two accounts — operator's "no visible spill" and
  `dgemma/model.py`'s docstring "CPU-spill" — described the same regime from two different
  vantage points. Both are right; reconciled here.
- **(b)** Uncapped historical loads ran within ~1 GB of the VRAM ceiling. Refined
  reboot-hazard model: the cliff is **VRAM margin under activation peaks + allocator spikes**
  (cf. the 46 GB `caching_allocator_warmup` spike patched 2026-07-23), **not** host-RAM
  exhaustion — the mmap observation above exonerates host RAM as the hazard surface.
- **(c)** Two-box context: on shane-pc (RTX-3090 24 GB, sm_86, Windows/StabilityMatrix) the
  same mechanism spills ~30 GiB, landing in the historically-observed **slow** regime.
  Magnitude, not mechanism, differs across the two boxes.
- **(d)** Informs #160 (lazy materialization / never-hit-VRAM mode) and #163 (smoke-battery
  memory discipline: explicit caps + preflight; headroom math must budget activations
  *above* the placement cap, not just against it). This grounds issue #160 (operator
  directive: lazy weight materialization + never-hit-VRAM offload mode): the current
  default path **already** exhibits lazy materialization for the offloaded slice — #160's
  remaining scope is making the laziness deliberate and budgetable (explicit caps,
  never-hit-VRAM mode) rather than an incidental byproduct of accelerate's auto-dispatch.

## Cross-refs

#160, #163, #101 (floor), #16 (shane-pc identity), 2026-07-23 handoff
(`docs/handoffs/2026-07-23-int4-autoround-loaded.md` — its bf16 "spills to CPU / ~3min" line
is now understood as an untagged cross-host conflation), `dgemma/model.py:267-274`.

## Follow-up experiment queued (named, not run)

AutoRound INT4 sampler viability on Turing sm_75. Operator H0: kernel wall — INT4 runs on
shane-pc/Ampere but not on this box's Turing card. The 2026-07-23 handoff proved load +
single-forward only; sampler-loop viability on sm_75 remains unrun.
