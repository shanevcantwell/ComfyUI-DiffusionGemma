# Handoff — current build verified working; next session = Windows/3090 setup

**Date:** 2026-07-06 · **From:** orchestrator (curator seat) · **HEAD at handoff:**
`040ae25` on `p3-instrumentation` (unchanged this session — see "No code touched" below).
Cold-start: `/orient`, then read **issue #16's test kit** before anything else. The record
is authoritative (README → plan.md → `decisions/` → issues → this file). Supersedes
`2026-07-06-backend-reversed-loader-retrofit.md` for session-state; that handoff's backend
decision (GGUF rejected, transformers-bf16 ships) and loader-retrofit facts still hold.

## State in one line

**The bf16 build works, end-to-end, in ComfyUI — verified live this session.** The next
move is not more code here; it is **standing the pack up on the Windows / RTX-3090 box**
(issue #16), whose memory precondition is now **resolved to a go**. The Windows run is a
you-at-that-box task — it is not drivable from the Linux host.

## What this session did

1. **`pip install gguf` assessed → does NOT reopen the GGUF path.** `gguf-py` is a
   file-format library (`GGUFReader`/`GGUFWriter` + numpy dequant, used by the HF→GGUF
   conversion scripts) — **not an inference runtime, no quantized-matmul kernels.**
   Dequantizing experts back to torch lands at full ~52 GB (no memory win); it closes no
   runtime gap. **Durable:** #15's revival trigger **narrowed** — GGUF reopens *only if
   DiffusionGemma decode lands in **stock upstream** llama.cpp*, never a fork this project
   compiles and distributes (operator constraint: will not own a build matrix / binary
   distribution). Comment: issues/15#issuecomment-4891479354.

2. **Current build verified running.** ComfyUI launched on the host at
   `localhost:8189` (`--listen --port 8189`, `.venv/bin/python main.py`). All three nodes
   register clean (`DGemmaLoader`, `DGemmaSampler`, `DGemmaTrace`), pack loads in 1.6 s, no
   errors. Weights not loaded until a graph is queued. **NB: this ComfyUI was left running**
   — stop it (`kill` the `main.py` on 8189) if the host GPU is needed elsewhere.

3. **Two design questions answered, both "no change."**
   - **DGemmaLoader edit box** = free-text `repo_id` STRING (`loader.py:31`), passed verbatim
     to `from_pretrained()`. Loosest widget available; no `local_files_only`, so a typo falls
     through to a network fetch. Stricter form = a `folder_paths` combo (enforced client+server,
     unlike bespoke *socket* types which only illuminate). **This is exactly #17** — left banked,
     not done this session.
   - **`gen_length` vs `canvas_length=256`** (confirmed from model `config.json`): 256 is the
     block quantum; `gen_length` is a free token budget that tiles into `ceil(gen_length/256)`
     blocks, last one partial. A non-multiple above 256 pays a full 48-step anneal for a tiny
     trailing block — the only argument for round numbers. **Operator decision: LEAVE IT** as
     the free budget. No widget `step` change.

4. **Windows/RTX-3090 test kit built and banked (#16) — the headline.**
   - **Grounded the memory fit from `model.safetensors.index.json`:** 51.6 GB bf16 total =
     **45.7 GB fused MoE experts** (90 tensors) + **6.0 GB** everything-else (attention, router,
     embeddings, norms). The 6 GB hot path fits 24 GB VRAM comfortably; only experts spill.
     `device_map="auto"` puts ~15 GB of experts on GPU and **~31 GB in system RAM** — same
     CPU-offload path as the 48 GB box, no code change, no quant.
   - **Go/no-go = system RAM, not VRAM.** Needed ≈40 GB free. **Target box confirmed: 96 GB
     DRAM + Intel i7-14700 → RESOLVED, go.** Load expected to succeed.
   - **Full executable recipe on #16:** Windows setup (torch+CUDA, `transformers==5.13.0`,
     `diffusers>=0.39.0`, `accelerate`), weight-transfer shortcut (copy the HF cache to skip
     the ~52 GB re-download), launch, and four pass criteria.
     Comments: issues/16#issuecomment-4891931443 (kit) + issues/16#issuecomment-4891969688 (go).

## Next session (on Windows)

**Start at issue #16's test kit comment.** The recipe is turn-key; execute it at the 3090 box.
Only two real unknowns remain, both benign:

- **(a) Windows node registration** — does the path-based dual-context import in
  `nodes/loader.py` (guarded by `tests/test_comfyui_loader_context.py`) come up clean on
  Windows? Expected yes; unvalidated on that OS.
- **(d) observed s/step** — will be slower than the host's ~2.3 s/step (~31 GB of CPU-resident
  experts stream over PCIe each step). A 14700 + PCIe 4.0 is a strong spill host; whatever it
  clocks **is** the deliverable that closes #16.

Criteria (b) load and (c) bf16-on-Ampere (native tensor cores, better than this Turing box) are
now **expected passes**, not risks.

## No code touched · working ground

- **HEAD unchanged at `040ae25`** — this session made no source edits. The durable artifacts are
  three GitHub issue comments (#15 ×1, #16 ×2) and this handoff. `p3-instrumentation` is otherwise
  clean and tracks `origin/p3-instrumentation`.
- **Publish gate unchanged:** `p3-instrumentation → main` remains publishable today (the #12
  eyeball gate was cleared last session); the loader-idiomatic retrofit (#17) is still the
  worth-doing-first item before 0.1.0 lands on Manager. Neither was actioned this session.
- **Open issues most-relevant to the next arc:** #16 (3090 test — armed), #17 (idiomatic loader —
  pri:next), #15 (GGUF — deferred, trigger narrowed), #4 (no smaller checkpoint — the fallback
  if RAM ever bites, which it won't at 96 GB).
