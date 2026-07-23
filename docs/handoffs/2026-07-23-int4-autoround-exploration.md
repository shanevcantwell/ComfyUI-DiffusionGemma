# Handoff: 2026-07-23 — AutoRound INT4 exploration

## Session summary

Explored Intel's AutoRound W4A16 quantized DiffusionGemma checkpoint as a path to consumer GPU support (3090, 4080 class). Verified config loads and MoE expert preparation succeeds. Full weight load timed out on the 48GB box — needs device_map tuning.

## What landed

- **#124** closed — `debug_log_path` directory bug fixed (was writing to dir path as a file)
- **#126** filed + patched — timestamp added so repeated runs don't collide (`dgemma_run_log_20260722T092745.jsonl`)
- **Examples cleanup** (#127 merged) — flagship renamed to `flipbook-annotated.ui.json`, phase artifacts banked to `examples/smoke-tests/`
- **v0.4.0 published** to ComfyUI registry, tagged on GitHub at `b4c5eca`
- **pyproject.toml** updated with `[quant]` optional extra (`auto-round>=0.5`) — PR pending merge

## What's in flight

### #128 — Loader dtype changes (filed, auto:draft)
`dgemma/model.py:load_model()` hardcodes `dtype=torch.bfloat16`. For pre-quantized checkpoints this must be `"auto"` so transformers reads the quantization config. Three sub-tasks:
1. Conditional dtype in load_kwargs (always "auto" vs conditional on checkpoint type)
2. Expand `_QUANT_CHOICES` beyond `("none",)`
3. Loader node UI — add Intel INT4 repo as an option

### #4 — Load-path strategy (updated with AutoRound findings)
AutoRound W4A16 is now the lead candidate over bnb/AWQ. Full weight load timed out on 48GB box under `device_map="auto"` — config + MoE prep succeeded, actual tensor loading hung. Local checkpoint at `/mnt/storage/LLMs/intel/diffusiongemma-26B-A4B-it-int4-AutoRound/` (58GB sharded).

## Ground truth for next session

**Checkpoint location:** `/mnt/storage/LLMs/intel/diffusiongemma-26B-A4B-it-int4-AutoRound/`
- 6 safetensors shards + config, verified present
- `quantization_config`: W4A16, group_size=64, sym=True, router proj at fp16

**Load test result:** Config loads, MoE experts unfused (60 modules), weight load timed out after 300s. Likely needs device_map tuning or tied weights fix (#119 territory).

**auto-round installed:** v0.14.2 in container Python env (`pip install --break-system-packages`)

## Open questions

1. Does `dtype="auto"` work for both bf16 and INT4 checkpoints, or does it need conditional logic?
2. Is the load timeout a memory issue (58GB sharded on 48GB box) or a device_map bug?
3. Should the loader node expose repo selection as dropdown vs free-text HF identifier?

## Branch state

Several local branches remain unmerged (already merged to main via PRs but not cleaned up locally):
- `docs/adr-cdg-017-neighborhood-remelt` — merged #116
- `fix/119-tied-weights-device-map-guard` — merged #119
- `fix/e2e-pack-identity-gate` — merged #123
- `pi-changes` — merged via cherry-picks
- `release/0.4.0`, `release/0.4.0-patch*` — merged, should be pruned

**Action:** `git branch --merged main | grep -v main | xargs git branch -d` to clean up.
