# Handoff — GGUF alpha rejected, ship path = idiomatic transformers loader

**Date:** 2026-07-06 · **From:** orchestrator (curator seat) · **HEAD at handoff:**
`b30ef1f` on `p3-instrumentation`. Cold-start: `/orient`. The record is authoritative
(README → plan.md → `decisions/` → issues → this file). Supersedes
`2026-07-05-p3-publish-armed.md` for the backend question; that handoff's publish-arming
facts (registry, secret, Action) still hold and are restated below.

## State in one line

The backend fork is closed: **0.1.0 ships on the `transformers`-bf16 path, GGUF deferred**
(ADR-CDG-007 rejected, `b30ef1f`). The merge/publish gate (#12 eyeball) is **cleared**, so
`p3-instrumentation → main` is publishable *today* — but the one thing worth doing first is
making the **loader idiomatic** (it still forces a free-text HF repo id + network pull), so
0.1.0 lands on Manager as a proper citizen rather than a rough first impression.

## What this session decided (durable, committed)

1. **GGUF/llama.cpp-fork alpha REJECTED for 0.1.0** — ADR-CDG-007 flipped `proposed →
   rejected` (`b30ef1f`, pushed to `origin/p3-instrumentation`). Deciding reason: the fork
   (`/srv/dev/llama.cpp-diffusiongemma`) is **private/unpublished**, which ADR-007's own
   Open Question 1 flagged as a *blocking prerequisite* — a public Manager pack can't depend
   on a fork users can't obtain. The transformers path uses only public HF weights + public
   `transformers`. The GGUF design is preserved in-ADR as the considered-and-set-aside record
   (don't re-burn the quant dead-ends).
2. **fp8/nf4 can't rescue in-torch memory fit — and it's inherent, not a missing flag.** The
   ~42.5 GiB of fused 3D MoE experts (`DiffusionGemmaTextExperts`) aren't `nn.Linear`/`Conv1D`,
   so *every* stock quantizer (bnb, torchao/quanto fp8, AWQ) skips them (fp8 would shrink only
   ~1 B of Linear params). This is **module coverage, not a hardware cast**: nf4 already
   dequantizes to fp16 on Turing — no Blackwell needed. GGUF fit only because llama.cpp
   quantizes the experts natively.
3. **Honest 0.1.0 hardware envelope:** needs **≈48 GB+ VRAM**, CPU-spills the experts
   (~24 tok/s), does **not** fit consumer 16–24 GB cards. This is in-genre for an experimental
   pack — state it plainly in README/requirements, don't apologize for it.
4. **Loader model-dir decision = scan BOTH folders (#2).** The loader lists the union of
   `models/diffusion_models/` **and** `models/text_encoders/` — found-wherever-you-put-it, no
   symlink lore. Rationale: DiffusionGemma is a transformer-denoiser (peer of Flux/SD3 DiTs →
   `diffusion_models/` is the role-correct home), but its Gemma lineage means users may already
   keep it under `text_encoders/`. Scanning both dissolves the ambiguity. (Amends ADR-007's
   `diffusion_models/`-only note, which is moot now that 007 is rejected.)
5. **Issue re-tiers:** #15 (GGUF backend) → `pri:later`. #16 (RTX-3090) anchored to the
   reversal — its role is native bf16-tensor-core validation (Ampere has it, the RTX-8000
   doesn't) + a smaller-checkpoint path, **not** running the 26 B on 24 GB (won't fit).

## The remaining ship path to 0.1.0-on-Manager

1. **[NEXT] Idiomatic loader retrofit** — tracked as a new issue (see dashboard). Spec below.
2. **README/requirements** — add the honest ≈48 GB envelope note (self-selection for Manager users).
3. **Merge `--no-ff` = publish 0.1.0.** The merge fires `.github/workflows/publish_action.yml`
   → publishes to `registry.comfy.org` under publisher `reflectiveattention` → appears in
   ComfyUI Manager within ~1–24 h. **Merging *now* would publish the non-idiomatic loader** —
   so step 1 is a deliberate publish-quality gate, not ceremony.

## Loader retrofit spec (for the next session's build)

Current state — `nodes/loader.py` + `dgemma/model.py`:
- `DGemmaLoader.INPUT_TYPES` (`nodes/loader.py:27-37`) offers a **free-text `repo_id` STRING**
  + a `quant` dropdown. `dgemma/model.py:load_model` (`:103`) calls
  `DiffusionGemmaForBlockDiffusion.from_pretrained(repo_id, …)` — **pulls from HF cache/network**.
- `folder_paths` and `comfy.model_management` are used **nowhere** in the pack (grep-confirmed).

Target (idiomatic, transformers backend):
- Replace the free-text `repo_id` with a **dropdown** built from
  `folder_paths.get_filename_list(...)` **unioned over `diffusion_models` + `text_encoders`**
  (decision #4). Resolve the chosen name back to a full path via `folder_paths.get_full_path`
  against each key (try both).
- Load via `from_pretrained(<resolved local path>, local_files_only=True)` — no network fetch;
  a directory with no readable weights is a hard node error, never a silent pull.
- Note the idiom-bend: transformers weights are a **shard directory** (config.json + safetensors),
  not a single file. Standard Comfy loaders list *files*. Decide at build time whether to list
  weight **directories** or a sentinel file (e.g. `config.json`) within them — either works; the
  dir-listing is the ADR-007-era intent.
- `comfy.model_management` integration (device/offload) is a **tier-2 nice-to-have**, not required
  for the retrofit — the bespoke `_device_map`/`_resolve_device` in `model.py` already handles the
  CPU-spill correctly (2 verified PASSes). Don't gold-plate it into the MVP.
- `DGEMMA_MODEL` socket + the `quant` widget are unchanged.

Enforcement/tests: extend `tests/` (the dual-context import test already guards loader import);
add coverage that the dropdown resolves a path and `local_files_only` is honored.

## Environment standing state (not derivable from the repo)

- **ComfyUI is UP on `0.0.0.0:8189`** (my copy — bounced this session), with **four packs loaded
  clean**: ComfyUI-Manager (V3.41), ComfyUI-Crystools (v1.27.4), ComfyUI-Easy-Use (v1.3.7), and
  ComfyUI-DiffusionGemma. **8188 is the operator's — leave it free.** A root-owned `python main.py`
  (PID from ~00:33 UTC) is not answering on either port; left untouched. GPU ~empty at bounce
  (1.6/49 GB) — a queued run has full headroom.
- The three new packs were `git clone --depth 1` into `/srv/dev/ComfyUI/custom_nodes/` and their
  requirements installed into `/srv/dev/ComfyUI/.venv` with **zero drift** to the guarded ML stack
  (transformers 5.13.0, diffusers 0.39.0, torch 2.12.1, numpy 2.5.1 all unchanged).
- **Publish wiring (from 2026-07-05, still valid):** publisher `reflectiveattention` minted
  (immutable); repo secret `REGISTRY_ACCESS_TOKEN` set operator-side;
  `.github/workflows/publish_action.yml` armed (fires on `pyproject.toml` change on `main` +
  manual dispatch). pyproject is at `0.1.0`. **Merge = publish.**
- **Merge gate CLEARED:** #12 (+#13) closed 2026-07-06 00:49 UTC, #12 `verified`. The
  operator-eyeball gate the prior handoff named is satisfied.

## Open decisions (operator)

1. **#1** (`user:gate`, `pri:now`, mask_token) — close-recommendation banked since 2026-07-05;
   doctrine grounded-facts confirm `mask_token_id=None`. One click to close.
2. **Loader retrofit vs publish-now** — you can publish 0.1.0 today (gate cleared) with the rough
   loader, or gate publish on the retrofit. This session's read: retrofit first (quality gate).
3. **`comfy.model_management`** integration — deferred as tier-2; revisit if VRAM UX complaints
   arrive post-publish.
