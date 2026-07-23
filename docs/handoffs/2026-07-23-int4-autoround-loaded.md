# Handoff: 2026-07-23 — AutoRound INT4 loaded, loader node behavior TBD

## Session summary

AutoRound W4A16 INT4 checkpoint loads successfully via `quant='autoround'`. Three transformers/auto-round patches handle the incompatibilities. Forward pass verified. Targeting v0.5.0.

## What landed

- **`193edd3`** — `feat(loader): AutoRound INT4 checkpoint support via quant='autoround' (#128)`
  - `_QUANT_CHOICES = ('none', 'autoround')` in `dgemma/model.py`
  - `_apply_autoround_patches()` patches three transformers/auto-round issues at load time
  - Loader node UI exposes `['none', 'autoround']` dropdown, defaults to `'none'` (bf16)
- **Issue #128** — updated with resolution comment and label change

## Verified results (48GB RTX-8000)

| Metric | INT4 autoround | bf16 none |
|---|---|---|
| Load time | 27s | ~3min w/ CPU spill |
| VRAM | 30.7 GB | ~53 GB (spills to CPU) |
| Forward pass | ✅ `[1, 256, 262144]` | ✅ same shape |

## Three patches applied at load time

1. **auto-round regex pre-compilation** — `skip_not_convert_modules` was O(N×M) recompiles (~120 patterns × ~7K modules), pinning one CPU core at 100%. Pre-compile once now.
2. **KV-cache warmup bypass** — `caching_allocator_warmup` pre-allocated bf16-sized buffer (46GB) before knowing weights are INT4, causing OOM on consumer GPUs. Skipped entirely.
3. **Tied-weight finalization guard** — `mark_tied_weights_as_initialized` and `tie_weights` crashed when lm_head.weight tied to quantized embed_tokens (no `.weight`, only `.qweight`). Catches AttributeError/NotImplementedError gracefully.

## What's next — loader node behavior

The loader node currently accepts `quant='autoround'` but has no auto-detection. Open questions for fresh context:

1. **Auto-detect INT4 checkpoints** — should the loader read `config.json` and detect `quantization_config.quant_method == 'auto-round'`, then set `quant='autoround'` automatically? Or keep it explicit (user must choose)?
2. **Default repo_id for autoround** — add Intel's INT4 repo as a dropdown option alongside Google's bf16? Free-text HF identifier is already supported via the `repo_id` field.
3. **Error messaging when auto-round missing** — if user selects `quant='autoround'` but doesn't have `[quant]` extra installed, surface an actionable message (currently falls through to raw ImportError from transformers).
4. **Tied-weight patch is a workaround** — ideally transformers handles quantized tied weights natively. Monitor for upstream fix; the current guard catches both AttributeError and NotImplementedError.

## Branch state

- 6 unmerged remote branches (pre-existing, banked on #114): `docs/adr-cdg-017-neighborhood-remelt`, `fix/119-tied-weights-device-map-guard`, `fix/e2e-pack-identity-gate`, `pi-changes`, `release/0.4.0-patch`, `salvage/s5-beta-rebuild-crash-wip`
- PR #121 (tied-weights guard) and #125 (debug_log_path fix) still open, targeting release/0.4.0-patch

## Target version

**0.5.0** — this is a new feature (INT4 support), not a patch to 0.4.0. The bf16 path is unchanged and backward-compatible.
