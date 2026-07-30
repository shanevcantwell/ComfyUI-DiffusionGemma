"""bf16 fit probe — replicates dgemma.model.load_model(quant='none') kwargs
PLUS a hard max_memory cap (DEVIATION: load_model has no max_memory
passthrough, so the from_pretrained call is replicated here verbatim from
dgemma/model.py lines 264-274 with the cap added and local_files_only=True).

Caps: max_memory={0: "44GiB", "cpu": "40GiB"} — never loosened.
Run args mirror tests/test_integration.py (seed=0, steps=8, gen_length=64).
"""
import json
import sys
import time
import traceback
from collections import Counter, defaultdict

sys.path.insert(0, "/srv/dev/shanevcantwell/ComfyUI-DiffusionGemma")

import torch  # noqa: E402

OUT = "/tmp/bf16-fit-probe"

from dgemma.model import DEFAULT_REPO_ID, _resolve_device  # noqa: E402
from dgemma.types import DGemmaModel  # noqa: E402
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion  # noqa: E402

# --- load_model's exact quant='none' kwargs + the hard cap (the one deviation) ---
load_kwargs = {
    "device_map": "auto",          # as in load_model
    "dtype": torch.bfloat16,       # as in load_model (quant='none' branch)
    "local_files_only": True,      # load_model forwards this; True here: no downloads
    "max_memory": {0: "44GiB", "cpu": "40GiB"},  # DEVIATION: hard cap, not in load_model
}

print(f"[probe] starting load at {time.strftime('%H:%M:%S')} UTC", flush=True)
t0 = time.perf_counter()
try:
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(DEFAULT_REPO_ID, **load_kwargs)
except Exception as exc:
    elapsed = time.perf_counter() - t0
    with open(f"{OUT}/load_error.txt", "w") as f:
        f.write(f"elapsed={elapsed:.1f}s\n")
        traceback.print_exc(file=f)
    print(f"[probe] LOAD REFUSED/FAILED after {elapsed:.1f}s: "
          f"{type(exc).__name__}: {exc}", flush=True)
    sys.exit(2)
load_s = time.perf_counter() - t0
print(f"[probe] LOAD OK in {load_s:.1f}s", flush=True)

# --- placement dump ---
dm = getattr(model, "hf_device_map", None) or {}
with open(f"{OUT}/device_map.json", "w") as f:
    json.dump({k: str(v) for k, v in dm.items()}, f, indent=2)

entry_counts = Counter(str(v) for v in dm.values())

bytes_by_dev = defaultdict(int)
params_by_dev = defaultdict(int)
seen = set()
for _name, p in model.named_parameters():
    if id(p) in seen:  # tied weights counted once
        continue
    seen.add(id(p))
    d = str(p.device)
    bytes_by_dev[d] += p.numel() * p.element_size()
    params_by_dev[d] += p.numel()
buf_bytes_by_dev = defaultdict(int)
for _name, b in model.named_buffers():
    if id(b) in seen:
        continue
    seen.add(id(b))
    buf_bytes_by_dev[str(b.device)] += b.numel() * b.element_size()

report = {
    "load_seconds": round(load_s, 1),
    "device_map_entry_counts": dict(entry_counts),
    "param_bytes_by_device": dict(bytes_by_dev),
    "param_GiB_by_device": {k: round(v / 2**30, 2) for k, v in bytes_by_dev.items()},
    "params_by_device": dict(params_by_dev),
    "buffer_bytes_by_device": dict(buf_bytes_by_dev),
    "cuda_memory_allocated_GB": round(torch.cuda.memory_allocated() / 1e9, 2),
    "cuda_max_memory_allocated_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
}
print("[probe] placement report:\n" + json.dumps(report, indent=2), flush=True)
with open(f"{OUT}/load_report.json", "w") as f:
    json.dump(report, f, indent=2)

disk_entries = [k for k, v in dm.items() if str(v) == "disk"]
if disk_entries:
    print(f"[probe] ANOMALY: disk-offloaded entries: {disk_entries}", flush=True)

# --- short generation, args grounded in tests/test_integration.py ---
processor = AutoProcessor.from_pretrained(DEFAULT_REPO_ID, local_files_only=True)
dg = DGemmaModel(
    model=model,
    processor=processor,
    device=_resolve_device(model),
    dtype="bfloat16",
    repo_id=DEFAULT_REPO_ID,
    quant="none",
)
print(f"[probe] execution device resolved: {dg.device}", flush=True)

from dgemma.loop import run_diffusion  # noqa: E402

t1 = time.perf_counter()
try:
    text, canvas_state, _trace = run_diffusion(
        dg,
        "Why is the sky blue?",
        seed=0,
        num_inference_steps=8,
        gen_length=64,
    )
    gen_s = time.perf_counter() - t1
    run_report = {
        "generate_seconds": round(gen_s, 1),
        "steps_used": canvas_state.steps_used,
        "seconds_per_step": round(gen_s / max(canvas_state.steps_used, 1), 1),
        "committed_fraction": canvas_state.committed_fraction,
        "converged": canvas_state.converged,
        "text_head": text[:300],
        "cuda_max_memory_allocated_GB_after_run": round(
            torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    print("[probe] RUN OK:\n" + json.dumps(run_report, indent=2), flush=True)
    with open(f"{OUT}/run_report.json", "w") as f:
        json.dump(run_report, f, indent=2)
except Exception as exc:
    gen_s = time.perf_counter() - t1
    with open(f"{OUT}/run_error.txt", "w") as f:
        f.write(f"elapsed={gen_s:.1f}s\n")
        traceback.print_exc(file=f)
    print(f"[probe] RUN FAILED after {gen_s:.1f}s: {type(exc).__name__}: {exc}",
          flush=True)

# --- cleanup ---
del dg, model, processor
torch.cuda.empty_cache()
print(f"[probe] CLEANUP done; cuda allocated now "
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB", flush=True)
