"""§2 fixed-seed encoder-text liveness sweep — runs against UNMODIFIED main
(door-invariance check). Per q2-smoke-protocol.md §2.

Loads the real model ONCE (bf16, device_map=auto, CPU-spill), mints a donor
cache per (seed, text) via encode_sequence, then calls run_diffusion(...,
kv_cache=<that cache>) and records (status, exception_type_or_None) plus a
sha256 of that tuple. All 12 must produce the identical hash for PASS.

Run from the MAIN repo checkout (sys.path[0] = main repo root), NOT the
worktree — this sweep's whole point is confirming the pre-rewrite door.
"""
import hashlib
import json
import sys
import time
import traceback

MAIN_REPO = "/srv/dev/shanevcantwell/ComfyUI-DiffusionGemma"
sys.path.insert(0, MAIN_REPO)

import torch  # noqa: E402

from dgemma.model import load_model  # noqa: E402
from dgemma.loop import run_diffusion  # noqa: E402
from dgemma.kv_cache import encode_sequence  # noqa: E402

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sec2_results.json"

SEEDS = [7, 13, 21]
TEXTS = [
    "The lighthouse keeper counted every wave before dawn.",
    "",
    "Quarterly revenue exceeded forecast by twelve percent in the northern region.",
    "xyzzy plugh",
]

KNOBS = dict(
    gen_length=256,
    num_inference_steps=48,
    entropy_bound=0.1,
    t_min=0.4,
    t_max=0.8,
    confidence=0.005,
    thinking=False,
)


def main():
    t_load0 = time.monotonic()
    dgemma_model = load_model(quant="none")  # bf16, DEFAULT_REPO_ID, device_map=auto
    t_load1 = time.monotonic()
    load_wall_s = t_load1 - t_load0

    results = []
    for seed in SEEDS:
        for text_idx, text in enumerate(TEXTS, start=1):
            row = {"seed": seed, "text_idx": text_idx, "text": text}
            t0 = time.monotonic()
            try:
                tokenizer = getattr(dgemma_model.processor, "tokenizer", dgemma_model.processor)
                # add_special_tokens=False: raw content ids only, no BOS/EOS
                # padding that would obscure what "empty string" (text #2)
                # actually mints — confirmed pre-flight (tokenizer-only load,
                # no weights) that "" -> [] with this call shape.
                ids = tokenizer(text, add_special_tokens=False)["input_ids"]
                cache = encode_sequence(dgemma_model, ids)
                status = None
                exc_type = None
                try:
                    run_diffusion(dgemma_model, text, seed=seed, kv_cache=cache, **KNOBS)
                    status = "success"
                except Exception as exc:  # noqa: BLE001
                    status = "raised"
                    exc_type = type(exc).__name__
                    row["exception_message"] = str(exc)
                    row["traceback"] = traceback.format_exc()
                row["status"] = status
                row["exception_type"] = exc_type
            except Exception as mint_exc:  # noqa: BLE001
                # A failure minting the donor cache itself (not the
                # run_diffusion call under test) — record distinctly so it
                # isn't confused with the door's own raise.
                row["status"] = "mint_error"
                row["exception_type"] = type(mint_exc).__name__
                row["exception_message"] = str(mint_exc)
                row["traceback"] = traceback.format_exc()
            t1 = time.monotonic()
            row["wall_clock_s"] = t1 - t0
            h = hashlib.sha256(
                json.dumps([row["status"], row["exception_type"]]).encode("utf-8")
            ).hexdigest()
            row["status_exc_sha256"] = h
            results.append(row)
            print(f"[sec2] seed={seed} text#{text_idx} -> status={row['status']} exc={row['exception_type']} sha={h[:12]} ({row['wall_clock_s']:.2f}s)", flush=True)

    hashes = {r["status_exc_sha256"] for r in results}
    summary = {
        "load_wall_clock_s": load_wall_s,
        "n_calls": len(results),
        "n_distinct_hashes": len(hashes),
        "all_identical": len(hashes) == 1,
        "results": results,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[sec2] wrote {OUT_PATH}; n_distinct_hashes={len(hashes)} all_identical={summary['all_identical']}")


if __name__ == "__main__":
    main()
