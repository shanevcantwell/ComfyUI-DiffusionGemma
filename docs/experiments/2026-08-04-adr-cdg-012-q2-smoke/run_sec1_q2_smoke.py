"""§1 Q-2 real-weights smoke — runs against the SKELETON worktree
(scratch/q2-skeleton-2026-08-04, /srv/dev/shanevcantwell/.worktrees/
cdg-q2-skeleton). Per q2-smoke-protocol.md §1.

3 seeds x 2 arms (with-cache, without-cache) = 6 run_diffusion calls, plus
3 mint calls (encode_sequence(donor_prompt) once per seed, reused across
both arms at that seed).

Loads the model ONCE (bf16, device_map=auto, CPU-spill).
"""
import hashlib
import json
import sys
import time
import traceback

SKELETON_ROOT = "/srv/dev/shanevcantwell/.worktrees/cdg-q2-skeleton"
sys.path.insert(0, SKELETON_ROOT)

from dgemma.model import load_model  # noqa: E402
from dgemma.loop import run_diffusion  # noqa: E402
from dgemma.kv_cache import encode_sequence  # noqa: E402

OUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sec1_results.json"

SEEDS = [7, 13, 21]
DONOR_PROMPT = "The lighthouse keeper counted every wave before dawn."

KNOBS = dict(
    gen_length=256,
    num_inference_steps=48,
    entropy_bound=0.1,
    t_min=0.4,
    t_max=0.8,
    confidence=0.005,
    thinking=False,
)


def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, default=str, sort_keys=True).encode("utf-8")).hexdigest()


def provenance_dict(p):
    if p is None:
        return None
    return {
        "minting_sequence": list(p.minting_sequence) if p.minting_sequence is not None else None,
        "edit_script": list(p.edit_script),
        "model_repo_id": p.model_repo_id,
        "tokenizer_fingerprint": p.tokenizer_fingerprint,
    }


def run_one(dgemma_model, seed, kv_cache):
    t0 = time.monotonic()
    row = {"seed": seed, "arm": "with-cache" if kv_cache is not None else "no-cache"}
    try:
        text, canvas_state, canvas_trace = run_diffusion(
            dgemma_model, DONOR_PROMPT, seed=seed, kv_cache=kv_cache, **KNOBS
        )
        row["status"] = "success"
        row["string"] = text
        row["string_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row["string_len"] = len(text)
        row["canvas_state"] = {
            "converged": canvas_state.converged,
            "committed_fraction": canvas_state.committed_fraction,
            "steps_used": canvas_state.steps_used,
        }
        row["injected_cache_provenance"] = provenance_dict(canvas_trace.injected_cache_provenance)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "raised"
        row["exception_type"] = type(exc).__name__
        row["exception_message"] = str(exc)
        row["traceback"] = traceback.format_exc()
    t1 = time.monotonic()
    row["wall_clock_s"] = t1 - t0
    return row


def main():
    t_load0 = time.monotonic()
    dgemma_model = load_model(quant="none")
    t_load1 = time.monotonic()
    load_wall_s = t_load1 - t_load0

    tokenizer = getattr(dgemma_model.processor, "tokenizer", dgemma_model.processor)
    donor_ids = tokenizer(DONOR_PROMPT, add_special_tokens=False)["input_ids"]

    mint_rows = []
    caches = {}
    for seed in SEEDS:
        t0 = time.monotonic()
        row = {"seed": seed, "call": "encode_sequence(mint)"}
        try:
            cache = encode_sequence(dgemma_model, donor_ids)
            caches[seed] = cache
            row["status"] = "success"
            row["provenance"] = provenance_dict(cache.provenance)
            row["cumulative_length"] = list(cache.cumulative_length)
        except Exception as exc:  # noqa: BLE001
            row["status"] = "raised"
            row["exception_type"] = type(exc).__name__
            row["exception_message"] = str(exc)
            row["traceback"] = traceback.format_exc()
        t1 = time.monotonic()
        row["wall_clock_s"] = t1 - t0
        mint_rows.append(row)
        print(f"[sec1] mint seed={seed} -> {row['status']} ({row['wall_clock_s']:.2f}s)", flush=True)

    run_rows = []
    for seed in SEEDS:
        # No-cache arm first (isolates seed variance per the protocol's own
        # framing — "same seed, same prompt, no kv_cache=").
        row_nc = run_one(dgemma_model, seed, None)
        run_rows.append(row_nc)
        print(f"[sec1] seed={seed} arm=no-cache -> {row_nc['status']} ({row_nc['wall_clock_s']:.2f}s)", flush=True)

        cache = caches.get(seed)
        row_wc = run_one(dgemma_model, seed, cache)
        run_rows.append(row_wc)
        print(f"[sec1] seed={seed} arm=with-cache -> {row_wc['status']} ({row_wc['wall_clock_s']:.2f}s)", flush=True)

    # Gating predicate pre-check (informational only here; the human/typed
    # verdict is assembled after reading this file, per protocol PASS (a)).
    gating_notes = []
    by_seed = {}
    for r in run_rows:
        by_seed.setdefault(r["seed"], {})[r["arm"]] = r
    for seed, arms in by_seed.items():
        nc = arms.get("no-cache")
        wc = arms.get("with-cache")
        if nc is None or wc is None:
            gating_notes.append(f"seed={seed}: missing an arm")
            continue
        if wc["status"] != "success":
            gating_notes.append(f"seed={seed}: with-cache raised ({wc.get('exception_type')})")
            continue
        if nc["status"] != "success":
            gating_notes.append(f"seed={seed}: no-cache raised ({nc.get('exception_type')})")
            continue
        same_string = wc["string_sha256"] == nc["string_sha256"]
        empty = wc["string_len"] == 0
        prov_ok = wc["injected_cache_provenance"] is not None
        gating_notes.append(
            f"seed={seed}: byte_identical_to_no_cache={same_string} empty={empty} "
            f"provenance_present={prov_ok}"
        )

    summary = {
        "load_wall_clock_s": load_wall_s,
        "donor_prompt": DONOR_PROMPT,
        "donor_ids": donor_ids,
        "mint_calls": mint_rows,
        "run_calls": run_rows,
        "gating_notes": gating_notes,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[sec1] wrote {OUT_PATH}")
    for note in gating_notes:
        print(f"[sec1] gating: {note}")


if __name__ == "__main__":
    main()
