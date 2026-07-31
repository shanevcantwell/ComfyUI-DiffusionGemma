# Experiment — autoround unified-path split-check probe (BLOCKED before H0 observation)

**Date:** 2026-07-30/31 (UTC). **Issue:** #183. **Plan:** signed
`plan-autoround-placement.md` (ratified 2026-07-30, Opus gate PASS, second
cycle). **Host:** `inference-host`, Quadro RTX 8000, 47.5GiB free at probe
attempt time; kept-resident llama-server (embeddinggemma) holding 1.4GiB noted
as baseline, not killed.

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

## Outcome: BLOCKED — checkpoint precondition not met, H0 not observed

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
