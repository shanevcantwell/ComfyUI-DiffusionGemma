# Test Coverage Plan — baseline 2026-07-05 (branch p2-expose-knobs @ c10ced0)

**Baseline:** `pytest --cov=dgemma --cov=nodes --cov-report=term-missing` → **87% total**,
54 passed / 1 skipped (integration, env-gated behind `DGEMMA_INTEGRATION=1`).

| module | cover | missing lines | what they are |
|---|---|---|---|
| `dgemma/model.py` | 30% | 50-58, 78-80, 94-100, 120-135 | `_quantization_config` branches; `_device_map` branches; `_resolve_device`; `load_model` body |
| `nodes/loader.py` | 93% | 19 | dual-context relative-import branch (ComfyUI package context) |
| `nodes/sampler.py` | 93% | 22 | same dual-context branch |
| `dgemma/loop.py` | 99% | 361 | `_decode_ids` eos-absent branch |
| `dgemma/types.py`, `dgemma/__init__.py`, `nodes/__init__.py` | 100% | — | — |

**Closed-issue cross-reference:** only closed issue is #2 (ADR-CDG-002 caveat, documentation,
labeled `verified`) — no code path, no regression test implied. Nothing missing on this axis.

## Phase 1 — `dgemma/model.py` unit coverage (the 30% module; no weights, no GPU)

All four gaps are unit-testable with fakes/monkeypatch:

1. `_quantization_config`: parametrize `nf4` (asserts 4-bit + fp16 compute dtype — the
   sm_75 grounded fact), `int8`, `none` → `None`, invalid → `ValueError`.
2. `_device_map`: `nf4`/`int8` with CUDA available (monkeypatch `torch.cuda.is_available`)
   → `{"": 0}`; `none` → `"auto"`; no-CUDA → `"auto"`.
3. `_resolve_device`: fake objects with `hf_device_map` variants — int GPU entry → `cuda:N`;
   CPU-spill map (first param off-GPU, later int entry) → still the accelerator;
   all-cpu/disk map → falls back to first parameter's device; no `hf_device_map` → fallback.
4. `load_model`: monkeypatch `DiffusionGemmaForBlockDiffusion.from_pretrained` +
   `AutoProcessor.from_pretrained`; assert kwargs shape per quant (`quantization_config`
   present iff quantized; `dtype=torch.bfloat16` iff `none`; `device_map` from `_device_map`)
   and the returned `DGemmaModel` fields (`dtype` label, `repo_id`, resolved device).

Target: model.py ≥ 95%.

## Phase 2 — branch-line sweep (micro)

- `dgemma/loop.py:361` — `_decode_ids` with `eos_token_id=None` / eos not in ids
  (parametrize onto the existing decode tests).
- `nodes/loader.py:19` + `nodes/sampler.py:22` — the ComfyUI-package-context import branch.
  `tests/test_comfyui_loader_context.py` already enforces the top-level context; attempt the
  dotted-package context via a synthetic package import (importlib, temp package dir).
  If that proves structurally awkward under pytest, mark the two lines `# pragma: no cover`
  with a comment citing this plan — the branch is enforced behaviorally by the live ComfyUI
  load (P1 PASS evidence), not left silently untested.

Target: total ≥ 97%.

## Open Questions

- **Dual-context import branch**: synthetic-package coverage vs `pragma: no cover` with the
  live-load evidence cited — implementer's judgment which is less contrived (see Phase 2).
- **`tests/test_integration.py` (env-gated skip)**: stays out of coverage scoring by design —
  it exists for real-weights runs, not CI. No action; named so the 1-skip readback stays legible.
