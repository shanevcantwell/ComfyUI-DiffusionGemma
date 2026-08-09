#!/usr/bin/env python3
"""run_batch.py — TC-001's batch driver (issue #286): 10 premises x 2
conditions = 20 runs through the ComfyUI HTTP API, one at a time (one
model, one GPU, never concurrent), each banking a JSONL run-log
(`DGemmaRunLogWriter`, wired in `tc001_graph.api.json`) plus one line in a
batch manifest for resumability and downstream judging.

Orchestration-plane only: this script and its sibling harness files touch
no `dgemma/`, `surfaces/`, or `consumers/` code (ARCHITECTURE.md rule 8) —
it drives the already-shipped ComfyUI HTTP API exactly the way
`tests/e2e/driver.py` does for the E2E battery, with a pre-built graph and
the protocol's pre-registered premises/seeds substituted in per run.

Attribution: `load_workflow`, `submit_prompt`, and `poll_history` below are
copied verbatim from `tests/e2e/driver.py` (ADR-CDG-013's proven black-box
plumbing) rather than imported, because that module's own
`EXAMPLES_DIR` constant is hardcoded to `examples/smoke-tests/` — importing
it as-is would silently point `load_workflow` at the wrong directory for
this harness's own `tc001_graph.api.json`. Copying the three functions (no
behavior changes) avoids monkeypatching a module constant from outside;
`test_e2e_import_guard.py`'s independence invariant (stdlib + requests
only, no `dgemma`/`surfaces`/`consumers` import) is preserved by
construction since this file has the same import list as the original.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import requests

HARNESS_DIR = Path(__file__).resolve().parent
# harness -> tc-001-structural-conditioning -> experiments -> docs -> repo root (4 levels).
REPO_ROOT = HARNESS_DIR.parent.parent.parent.parent
GRAPH_PATH = HARNESS_DIR / "tc001_graph.api.json"

sys.path.insert(0, str(HARNESS_DIR))
from premises import (  # noqa: E402  (see sys.path insert above)
    CONDITIONS,
    PREMISES,
    SEEDS,
    condition_a_prompt,
    condition_b_prompt,
)

# Node ids in tc001_graph.api.json (see that file / this harness's own
# conformance test for the full wiring).
SAMPLER_NODE_ID = "73"
STRING_PREVIEW_NODE_ID = "74"
RUN_LOG_WRITER_NODE_ID = "75"

DEFAULT_BASE_URL = "http://127.0.0.1:8199"
DEFAULT_GEN_LENGTH = 1024
# A run is ~7 minutes per the operator's own estimate (task contract) —
# generous timeout, not a tight one.
DEFAULT_POLL_TIMEOUT_S = 1200.0

MANIFEST_FILENAME = "manifest.jsonl"


# --- tests/e2e/driver.py plumbing, copied verbatim (see module docstring for why) ---


def load_workflow(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def submit_prompt(base_url: str, client_id: str, workflow: dict[str, Any]) -> str:
    resp = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    assert "prompt_id" in body, f"/prompt response missing prompt_id: {body}"
    return body["prompt_id"]


def poll_history(
    base_url: str,
    prompt_id: str,
    timeout_s: float = 300.0,
    interval_s: float = 1.0,
    _sleep=time.sleep,
    _monotonic=time.monotonic,
) -> dict[str, Any]:
    deadline = _monotonic() + timeout_s
    while _monotonic() < deadline:
        resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        body = resp.json()
        if prompt_id in body:
            return body[prompt_id]
        _sleep(interval_s)
    raise TimeoutError(
        f"/history/{prompt_id} did not report a result within {timeout_s}s"
    )


# --- TC-001-specific batch plan + driver ---


def run_key(premise_idx: int, condition: str, seed: int) -> str:
    """`tc001_p{idx:02d}_{condition}_s{seed}` — also the graph's
    filename_prefix (task contract example: `tc001_p03_B_s103`)."""
    return f"tc001_p{premise_idx:02d}_{condition}_s{seed}"


def build_plan() -> list[dict[str, Any]]:
    """The 20 pre-registered runs: 10 premises x 2 conditions, paired seeds
    (protocol.md §Design — "seed i pairs premise i across both
    conditions"). Premise/condition indices are 1-based in the run key
    (matches the task contract's own `tc001_p03_...` example), 0-based
    against the PREMISES/SEEDS lists internally."""
    plan = []
    for i, (premise, seed) in enumerate(zip(PREMISES, SEEDS), start=1):
        for condition in CONDITIONS:
            prompt = condition_a_prompt(premise) if condition == "A" else condition_b_prompt(premise)
            plan.append(
                {
                    "key": run_key(i, condition, seed),
                    "premise_index": i,
                    "premise": premise,
                    "condition": condition,
                    "seed": seed,
                    "prompt": prompt,
                }
            )
    return plan


def mutate_graph(graph: dict[str, Any], run: dict[str, Any], runs_dir: Path) -> dict[str, Any]:
    """Deep-copies the base graph and sets this run's prompt/seed on the
    sampler and filename_prefix/debug_log_path on the run-log writer. All
    other knobs stay at the graph's repo-default values (steps 48,
    entropy_bound 0.1, t_min 0.4, t_max 0.8, confidence 0.005,
    thinking false, gen_length 1024) — untouched by this function."""
    mutated = copy.deepcopy(graph)
    mutated[SAMPLER_NODE_ID]["inputs"]["prompt"] = run["prompt"]
    mutated[SAMPLER_NODE_ID]["inputs"]["seed"] = run["seed"]
    mutated[SAMPLER_NODE_ID]["inputs"]["gen_length"] = DEFAULT_GEN_LENGTH
    mutated[RUN_LOG_WRITER_NODE_ID]["inputs"]["filename_prefix"] = run["key"]
    mutated[RUN_LOG_WRITER_NODE_ID]["inputs"]["debug_log_path"] = str(runs_dir)
    return mutated


def find_run_log_path(runs_dir: Path, filename_prefix: str) -> str | None:
    """`DGemmaRunLogWriter` names its file `{filename_prefix}_{timestamp}.jsonl`
    (surfaces/comfyui/run_log_writer.py:_resolve_output_path) — the
    timestamp isn't known ahead of the run, so the driver globs for it
    after the run completes rather than predicting it. Returns the most
    recently modified match (there should be exactly one per run key in
    the resumable/non-overlapping-key design this harness uses), or None
    if the writer did not produce a file (e.g. a failed run)."""
    matches = sorted(
        runs_dir.glob(f"{filename_prefix}_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(matches[-1]) if matches else None


def load_manifest_keys(manifest_path: Path) -> dict[str, dict[str, Any]]:
    """Existing manifest lines keyed by run key, for the resumable skip
    check. A run key present with `status == "ok"` is skipped; anything
    else (missing, or present but not ok) is re-run."""
    if not manifest_path.exists():
        return {}
    entries: dict[str, dict[str, Any]] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            entries[record["key"]] = record
    return entries


def append_manifest_line(manifest_path: Path, record: dict[str, Any]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def extract_final_text(history_entry: dict[str, Any]) -> str:
    """The sampler's `text` output, rendered by the `PreviewAny` at node 74
    (`{"text": [value, ...]}` shape — comfy_extras/nodes_preview_any.py)."""
    outputs = history_entry.get("outputs", {})
    preview = outputs.get(STRING_PREVIEW_NODE_ID, {})
    text_values = preview.get("text", [])
    return "".join(str(v) for v in text_values)


def run_one(
    base_url: str,
    graph: dict[str, Any],
    run: dict[str, Any],
    runs_dir: Path,
    poll_timeout_s: float,
) -> dict[str, Any]:
    """POSTs one run, polls to completion, and returns its manifest
    record. Never raises for a run-level failure (timeout, non-success
    status) — those are recorded as manifest entries with a non-"ok"
    status so the batch continues and a later invocation can retry them."""
    mutated = mutate_graph(graph, run, runs_dir)
    client_id = str(uuid.uuid4())

    record: dict[str, Any] = {
        "key": run["key"],
        "premise_index": run["premise_index"],
        "condition": run["condition"],
        "seed": run["seed"],
    }

    try:
        prompt_id = submit_prompt(base_url, client_id, mutated)
    except Exception as exc:  # noqa: BLE001 — a submit failure is a manifest outcome, not a crash
        record.update({"prompt_id": None, "status": "submit_error", "error": str(exc)})
        return record

    record["prompt_id"] = prompt_id

    try:
        history_entry = poll_history(base_url, prompt_id, timeout_s=poll_timeout_s)
    except TimeoutError as exc:
        record.update({"status": "timeout", "error": str(exc)})
        return record

    status = history_entry.get("status", {})
    status_str = status.get("status_str")
    final_text = extract_final_text(history_entry)
    run_log_path = find_run_log_path(runs_dir, run["key"])

    record.update(
        {
            "status": "ok" if status_str == "success" else f"comfyui_{status_str}",
            "run_log_path": run_log_path,
            "final_text_sha256": hashlib.sha256(final_text.encode("utf-8")).hexdigest(),
            "final_text_first_120": final_text[:120],
        }
    )
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--runs-dir", required=True, type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the 20 mutated prompts/seeds without POSTing anything.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run this harness dir's conformance test (pytest) against tc001_graph.api.json and exit.",
    )
    parser.add_argument(
        "--poll-timeout-s",
        type=float,
        default=DEFAULT_POLL_TIMEOUT_S,
        help=f"Per-run /history poll timeout in seconds (default {DEFAULT_POLL_TIMEOUT_S}, ~1200s = a run is ~7 min).",
    )
    args = parser.parse_args(argv)

    if args.validate:
        import os
        import subprocess

        # Run from REPO_ROOT with REPO_ROOT forced onto PYTHONPATH: the
        # conformance test's own `from __init__ import NODE_CLASS_MAPPINGS`
        # needs the repo root importable. `cwd=` alone is not reliable here
        # — pytest's `--import-mode=importlib` (this repo's pyproject.toml
        # config) does not consistently insert the invocation cwd onto
        # sys.path the way running `pytest` directly from a shell at repo
        # root happens to (observed: identical `subprocess.run(...,
        # cwd=REPO_ROOT)` call resolves fine standalone but fails once this
        # very script is itself invoked as `python run_batch.py`, which
        # changes the *parent* process's own `sys.path[0]` to HARNESS_DIR —
        # an ambient-environment dependency this explicit PYTHONPATH removes).
        env = dict(os.environ)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            str(REPO_ROOT) if not existing_pythonpath else f"{REPO_ROOT}{os.pathsep}{existing_pythonpath}"
        )
        result = subprocess.run(
            [sys.executable, "-m", "pytest", str(HARNESS_DIR / "test_tc001_graph_conformance.py"), "-v"],
            cwd=str(REPO_ROOT),
            env=env,
        )
        return result.returncode

    plan = build_plan()
    assert len(plan) == 20, f"expected 20 runs (10 premises x 2 conditions), got {len(plan)}"

    runs_dir: Path = args.runs_dir
    manifest_path = runs_dir / MANIFEST_FILENAME

    if args.dry_run:
        for run in plan:
            print(
                json.dumps(
                    {
                        "key": run["key"],
                        "premise_index": run["premise_index"],
                        "condition": run["condition"],
                        "seed": run["seed"],
                        "gen_length": DEFAULT_GEN_LENGTH,
                        "prompt": run["prompt"],
                    }
                )
            )
        return 0

    runs_dir.mkdir(parents=True, exist_ok=True)
    graph = load_workflow(GRAPH_PATH)
    existing = load_manifest_keys(manifest_path)

    # Smoke gate (issue #286 liveness ruling): the live HTTP submit/poll path
    # is first exercised by the batch itself. The FIRST freshly-executed run
    # is the smoke — if it does not land a valid run-log (`status == "ok"` AND
    # a `run_log_path` on disk), abort before spending the GPU on the other 19
    # runs. This is deliberately the first *executed* run, not a hardcoded key:
    # on a resume, an already-ok pair-1 is skipped and the gate falls to the
    # first run this invocation actually drives, so the smoke is always a live
    # end-to-end submit→poll→log-landing round-trip made this session.
    smoke_pending = True

    for run in plan:
        prior = existing.get(run["key"])
        if prior is not None and prior.get("status") == "ok":
            print(f"[skip] {run['key']} already ok in manifest", file=sys.stderr)
            continue

        print(f"[run]  {run['key']} (condition {run['condition']}, seed {run['seed']})", file=sys.stderr)
        record = run_one(args.base_url, graph, run, runs_dir, args.poll_timeout_s)
        append_manifest_line(manifest_path, record)
        print(f"[done] {run['key']} -> {record['status']}", file=sys.stderr)

        if smoke_pending:
            smoke_pending = False
            landed = record.get("status") == "ok" and record.get("run_log_path")
            if not landed:
                print(
                    f"[abort] smoke gate failed on first live run {run['key']}: "
                    f"status={record.get('status')!r}, "
                    f"run_log_path={record.get('run_log_path')!r}. "
                    "The live submit/poll/log-landing path did not round-trip; "
                    "not spending the GPU on the remaining runs. Fix the live "
                    "path (ComfyUI server, graph, node wiring) and re-invoke — "
                    "the manifest makes this resumable.",
                    file=sys.stderr,
                )
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
