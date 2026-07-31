#!/usr/bin/env python
"""Forced-split mechanism probe for #183 — H0 stated in README.md BEFORE observation.

Two legs, run as SEPARATE processes (clean CUDA state each):

  --leg baseline      unified load_kwargs shape, natural conditions (INT4 fits whole
                      on this 48GB host) — the fits-whole reference generation.
  --leg forced-split  same shape + probe-only max_memory={0: "<cap>GiB", "cpu": "40GiB"}
                      forcing at least one module onto cpu — the split-inducing leg.

The load_kwargs under test are the UNIFIED shape the fix will ship
(device_map="auto", dtype="auto" as the checkpoint-identity residue,
local_files_only) — NOT load_model()'s current divergent autoround branch.
`_apply_autoround_patches` / `_retie_lm_head` / `_assert_no_meta_tensors` /
`_resolve_device` are imported from dgemma.model unchanged (the prior art the
unified path inherits). Generation args ground in tests/test_integration.py:
seed=0, num_inference_steps=8, gen_length=64, prompt "Why is the sky blue?".

Measurement discipline (bf16-fit-mechanism instrument traps, binding):
- peak VRAM via torch.cuda.max_memory_allocated() ACROSS THE FORWARD/DENOISE
  RUN, not load-end nvidia-smi (lazy load: peak appears at forward-touch);
- hf_device_map dumped verbatim (mmap spill is RSS-invisible; params report
  device `meta`, not `cpu`);
- the forced cap governs weight placement only, NOT activation peaks — the
  split verdict comes from whether the cross-device forward completes
  correctly, never from "peak stayed under the cap";
- a 1s nvidia-smi sampler records the card-wide view (includes the ~1.4GiB
  co-resident llama-server baseline) alongside the in-process numbers.

Bare-script probe grounds MECHANISM ONLY (F5): the fits/works capacity verdict
for the unified path comes from the in-rig leg, not from this script.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path("/tmp/wt-183")
sys.path.insert(0, str(REPO))

MODEL_DIR = (
    "/srv/dev/ComfyUI/models/text_encoders/"
    "diffusiongemma-26B-A4B-it-int4-AutoRound"
)

# Grounded in tests/test_integration.py (INTEGRATION_NUM_STEPS / _GEN_LENGTH / seed)
PROMPT = "Why is the sky blue?"
SEED = 0
NUM_STEPS = 8
GEN_LENGTH = 64

GIB = 1024**3


def _nvidia_smi_sampler(stop: threading.Event, out_path: Path, interval: float = 1.0):
    with open(out_path, "w") as f:
        while not stop.is_set():
            r = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=timestamp,memory.used,memory.free",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            f.write(r.stdout)
            f.flush()
            stop.wait(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leg", choices=["baseline", "forced-split"], required=True)
    ap.add_argument("--gpu-cap-gib", type=int, default=24)
    ap.add_argument(
        "--out-dir", default=str(Path(__file__).resolve().parent / "runs")
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.leg

    import torch
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    from dgemma.loop import run_diffusion
    from dgemma.model import (
        _apply_autoround_patches,
        _assert_no_meta_tensors,
        _resolve_device,
        _retie_lm_head,
    )
    from dgemma.types import DGemmaModel

    # THE UNIFIED SHAPE UNDER TEST — identical to the bf16 exemplar branch
    # modulo dtype (checkpoint identity), per the signed plan's step 2.
    load_kwargs: dict = {
        "device_map": "auto",
        "dtype": "auto",  # read the checkpoint's own quantization_config
        "local_files_only": True,
    }
    if args.leg == "forced-split":
        # PROBE-ONLY addition (split-inducing, never a production kwarg):
        # cap small enough to force >=1 module to cpu. Banked verbatim.
        load_kwargs["max_memory"] = {0: f"{args.gpu_cap_gib}GiB", "cpu": "40GiB"}

    report: dict = {
        "leg": tag,
        "model_dir": MODEL_DIR,
        "load_kwargs": {k: str(v) for k, v in load_kwargs.items()},
        "gen_args": {
            "prompt": PROMPT,
            "seed": SEED,
            "num_inference_steps": NUM_STEPS,
            "gen_length": GEN_LENGTH,
        },
        "env": {
            "torch": torch.__version__,
            "python": sys.version.split()[0],
        },
    }
    import transformers

    report["env"]["transformers"] = transformers.__version__
    try:
        import auto_round

        report["env"]["auto_round"] = auto_round.__version__
    except ImportError:
        report["env"]["auto_round"] = None

    stop = threading.Event()
    sampler = threading.Thread(
        target=_nvidia_smi_sampler,
        args=(stop, out_dir / f"nvidia_smi_{tag}.log"),
        daemon=True,
    )
    sampler.start()

    status = "unknown"
    try:
        t0 = time.perf_counter()
        with _apply_autoround_patches():
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(
                MODEL_DIR, **load_kwargs
            )
        report["load_seconds"] = round(time.perf_counter() - t0, 1)

        device_map = getattr(model, "hf_device_map", None) or {}
        report["hf_device_map_size"] = len(device_map)
        offloaded = {k: str(v) for k, v in device_map.items() if str(v) in ("cpu", "disk")}
        report["offloaded_modules"] = offloaded
        report["n_offloaded_modules"] = len(offloaded)
        with open(out_dir / f"device_map_{tag}.json", "w") as f:
            json.dump({k: str(v) for k, v in device_map.items()}, f, indent=2)

        if args.leg == "forced-split" and not offloaded:
            report["error"] = (
                "forced-split leg induced NO cpu/disk placement — cap too high; "
                "split verdict cannot be grounded from this run"
            )
            status = "invalid"
            return 2

        # Prior art under test-by-inheritance (unchanged code):
        _retie_lm_head(model)
        lm_head_w = getattr(getattr(model, "lm_head", None), "weight", None)
        report["lm_head_weight_device_after_retie"] = (
            str(lm_head_w.device) if lm_head_w is not None else "ABSENT"
        )
        _assert_no_meta_tensors(model)
        report["assert_no_meta_tensors"] = "passed"

        processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
        device = _resolve_device(model)
        report["resolved_device"] = device

        report["vram_allocated_after_load_gib"] = round(
            torch.cuda.memory_allocated() / GIB, 2
        )
        report["vram_peak_after_load_gib"] = round(
            torch.cuda.max_memory_allocated() / GIB, 2
        )

        dmodel = DGemmaModel(
            model=model,
            processor=processor,
            device=device,
            dtype="int4",
            repo_id=MODEL_DIR,
            quant="autoround",
        )

        # NO reset_peak_memory_stats between load and run: process-wide peak
        # is the honest number; per-phase numbers above bracket it.
        t1 = time.perf_counter()
        text, canvas_state, canvas_trace = run_diffusion(
            dmodel,
            PROMPT,
            seed=SEED,
            num_inference_steps=NUM_STEPS,
            gen_length=GEN_LENGTH,
        )
        gen_s = time.perf_counter() - t1

        report["generate_seconds"] = round(gen_s, 1)
        report["steps_used"] = canvas_state.steps_used
        report["seconds_per_step"] = round(gen_s / max(canvas_state.steps_used, 1), 2)
        report["committed_fraction"] = round(canvas_state.committed_fraction, 4)
        report["converged"] = canvas_state.converged
        report["text"] = text
        # Peak at forward-touch — THE capacity-relevant in-process number
        # (lazy-load discipline: load-end numbers understate).
        report["vram_peak_process_gib"] = round(
            torch.cuda.max_memory_allocated() / GIB, 2
        )
        status = "completed"
        return 0
    except Exception as exc:  # bank the full traceback — red readbacks banked too
        import traceback

        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
        status = "errored"
        return 1
    finally:
        stop.set()
        sampler.join(timeout=5)
        report["status"] = status
        with open(out_dir / f"report_{tag}.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"[probe] leg={tag} status={status} -> {out_dir}/report_{tag}.json")


if __name__ == "__main__":
    sys.exit(main())
