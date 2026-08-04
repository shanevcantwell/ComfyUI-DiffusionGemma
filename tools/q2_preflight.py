#!/usr/bin/env python3
"""q2_preflight.py — push-button precondition check for the Q-2 GPU window
(Amendment 1 re-run route).

Standalone dev tool. Does NOT import into, or get imported by, the
ComfyUI-DiffusionGemma node pack (same discipline as `tools/flipbook/`).

Runsheet SSoT: the 2026-08-04 "Amendment 1" comment on issue #62
(https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62),
§3-amended ("Runsheet deltas for the ComfyUI-server lane"). Read it with:

    gh issue view 62 --comments

This tool checks the Amendment-1 §3-amended setup preconditions mechanically
(GPU headroom + tenancy, weights cached, skeleton branch present at origin,
fixture present, port free) and banks environment provenance (#228's
discipline: gate run 4's versions were never recorded; this run's are).

It does NOT drive the GPU window itself (no ComfyUI subprocess is launched,
no `run_diffusion`/`encode_sequence` call is made) — it is the go/no-go check
run *before* paying for that window.

Exit code 0 iff every check PASSes (warnings do not fail the run). Non-zero
otherwise.

Usage:
    python3 tools/q2_preflight.py --label 2026-08-0X-q2-rerun
    python3 tools/q2_preflight.py --label smoke-test --out /tmp/q2-preflight-smoke-test.json
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# --- Grounded defaults (Amendment-1 §3-amended, base-protocol §3) ----------

# #145 waiver line ("known-resident services <=2 GiB with >=35 GiB free"),
# carried into Amendment-1 §3-amended step 3a verbatim.
MIN_FREE_GIB = 35.0
RESIDENT_TENANT_MAX_MIB = 2048.0

# The desktop-compositor baseline that is present in every banked "GPU clean"
# reading across #59/#226/#62 (Xorg ~110 MiB, gnome-shell ~9 MiB) — part of
# the accepted idle floor, not a tenant to name as "other." Matched by
# process-name substring, case-sensitive to the banked evidence's own
# process names.
DESKTOP_BASELINE_NAME_SUBSTRINGS = ("Xorg", "gnome-shell")
RESIDENT_SERVER_NAME_SUBSTRING = "llama-server"

# Base protocol §0 / CLAUDE.md "Weights": google/diffusiongemma-26B-A4B-it.
# Banked evidence gives slightly different figures across runs/tools
# (~51.7-53.6G per #59/#226 comments; scan_cache_dir() itself measures
# 48.13 GiB on this host, verified this session — safetensors-on-disk vs.
# reported-model-size accounting differ, and that's expected, not a bug).
# The floor is set at 45 GiB: comfortably below the real observed complete
# cache (48.13 GiB) so it doesn't false-fail on that accounting gap, while
# still catching a genuinely partial/truncated cache (this repo id has no
# legitimate reason to be a fraction of that size).
WEIGHTS_REPO_ID = "google/diffusiongemma-26B-A4B-it"
WEIGHTS_MIN_GIB = 45.0

# Amendment-1 §0-amended / §3-amended step 3b: the skeleton drive body this
# window's §1 arm needs, confirmed present at this exact commit.
SKELETON_BRANCH = "scratch/q2-skeleton-2026-08-04"
SKELETON_EXPECTED_SHA = "d67e62f9c88c9dd83877742cf231f5b8226640e6"

# Amendment-1 §3-amended "Fixture lineage": the tier-1 fixture the with-cache
# arm's graph is submitted from.
FIXTURE_RELPATH = Path("examples/smoke-tests/kv-cache-tier1.api.json")

# Base protocol §3 setup step 2 / Amendment-1 §3-amended: the isolated
# headless-ComfyUI port, distinct from the operator's interactive 8188.
PREFLIGHT_PORT = 8199
PREFLIGHT_HOST = "127.0.0.1"

RUNSHEET_URL = "https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62"


class CheckResult(NamedTuple):
    name: str
    passed: bool
    detail: str
    warning: bool = False  # True: reported but does not fail the aggregate


# ---------------------------------------------------------------------------
# 1. GPU: nvidia-smi reachable, headroom, tenancy
# ---------------------------------------------------------------------------


def _run(cmd: list[str], timeout: float = 15.0) -> subprocess.CompletedProcess:
    """Thin subprocess.run wrapper — the one seam the unit test monkeypatches."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def check_gpu(run=_run) -> list[CheckResult]:
    """nvidia-smi reachable; >=35 GiB free; only the resident embedding
    server (~1.4 GiB) present. Other tenants: warn (name them), don't fail —
    per this task's own line 1."""
    results: list[CheckResult] = []

    try:
        proc = run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        results.append(
            CheckResult("gpu.nvidia_smi_reachable", False, f"nvidia-smi invocation failed: {exc}")
        )
        return results

    if proc.returncode != 0:
        results.append(
            CheckResult(
                "gpu.nvidia_smi_reachable",
                False,
                f"nvidia-smi exited {proc.returncode}: {proc.stderr.strip()}",
            )
        )
        return results

    results.append(CheckResult("gpu.nvidia_smi_reachable", True, "nvidia-smi responded"))

    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        results.append(
            CheckResult(
                "gpu.memory_query_parseable", False, f"unexpected nvidia-smi output: {line!r}"
            )
        )
        return results

    try:
        used_mib, free_mib, total_mib = (float(p) for p in parts)
    except ValueError:
        results.append(
            CheckResult(
                "gpu.memory_query_parseable", False, f"non-numeric nvidia-smi fields: {line!r}"
            )
        )
        return results

    free_gib = free_mib / 1024.0
    results.append(
        CheckResult(
            "gpu.free_memory",
            free_gib >= MIN_FREE_GIB,
            f"{free_gib:.2f} GiB free (need >= {MIN_FREE_GIB:.0f} GiB); "
            f"used={used_mib:.0f} MiB total={total_mib:.0f} MiB",
        )
    )

    # Tenancy: who's holding VRAM.
    try:
        apps_proc = run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.SubprocessError) as exc:
        results.append(
            CheckResult(
                "gpu.tenancy",
                True,
                f"could not enumerate compute-apps ({exc}); free-memory check above stands alone",
                warning=True,
            )
        )
        return results

    if apps_proc.returncode != 0:
        results.append(
            CheckResult(
                "gpu.tenancy",
                True,
                f"nvidia-smi --query-compute-apps exited {apps_proc.returncode}: "
                f"{apps_proc.stderr.strip()}",
                warning=True,
            )
        )
        return results

    tenants = []
    for row in apps_proc.stdout.strip().splitlines():
        if not row.strip():
            continue
        fields = [f.strip() for f in row.split(",")]
        if len(fields) != 3:
            continue
        pid, mem_mib_str, name = fields
        try:
            mem_mib = float(mem_mib_str)
        except ValueError:
            continue
        tenants.append((pid, mem_mib, name))

    def _is_accepted_baseline(name: str, mem_mib: float) -> bool:
        # Desktop compositor processes (Xorg/gnome-shell) are accepted
        # unconditionally — they are not a GPU-window "tenant" in the sense
        # #145's waiver means, and are present in every banked clean baseline.
        if any(sub in name for sub in DESKTOP_BASELINE_NAME_SUBSTRINGS):
            return True
        # The resident embedding server: accepted only if it looks like
        # llama-server AND stays under the waiver ceiling. An oversized
        # llama-server is NOT silently accepted just because of its name.
        if RESIDENT_SERVER_NAME_SUBSTRING in name and mem_mib <= RESIDENT_TENANT_MAX_MIB:
            return True
        return False

    non_resident = [
        (pid, mem_mib, name)
        for pid, mem_mib, name in tenants
        if not _is_accepted_baseline(name, mem_mib)
    ]
    named_other_tenants = [
        f"{name} (pid {pid}, {mem_mib:.0f} MiB)" for pid, mem_mib, name in non_resident
    ]

    if named_other_tenants:
        results.append(
            CheckResult(
                "gpu.tenancy",
                True,
                "other GPU tenants present (named, not blocking): " + "; ".join(named_other_tenants),
                warning=True,
            )
        )
    else:
        resident_desc = "; ".join(f"{name} (pid {pid}, {mem_mib:.0f} MiB)" for pid, mem_mib, name in tenants) or "none"
        results.append(
            CheckResult(
                "gpu.tenancy",
                True,
                f"only the resident embedding server present: {resident_desc}",
            )
        )

    return results


# ---------------------------------------------------------------------------
# 2. Weights: DGemma HF cache present
# ---------------------------------------------------------------------------


def check_weights() -> CheckResult:
    """~49G at the hub path for google/diffusiongemma-26B-A4B-it, per
    `tests/e2e/conftest.py`'s `_weights_cached` convention (scan_cache_dir)."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return CheckResult(
            "weights.hf_cache_present",
            False,
            "huggingface_hub not importable in this interpreter — cannot check the HF cache "
            f"for {WEIGHTS_REPO_ID}",
        )

    try:
        cache_info = scan_cache_dir()
    except Exception as exc:  # noqa: BLE001 - degrade to FAIL, name the cause
        return CheckResult(
            "weights.hf_cache_present", False, f"scan_cache_dir() raised: {exc!r}"
        )

    for repo in cache_info.repos:
        if repo.repo_id == WEIGHTS_REPO_ID:
            size_gib = repo.size_on_disk / (1024.0**3)
            ok = size_gib >= WEIGHTS_MIN_GIB
            return CheckResult(
                "weights.hf_cache_present",
                ok,
                f"{WEIGHTS_REPO_ID} cached at {size_gib:.1f} GiB "
                f"(need >= {WEIGHTS_MIN_GIB:.0f} GiB) at {repo.repo_path}",
            )

    return CheckResult(
        "weights.hf_cache_present", False, f"{WEIGHTS_REPO_ID} not found in the local HF cache"
    )


# ---------------------------------------------------------------------------
# 3. Skeleton branch on origin at the expected commit
# ---------------------------------------------------------------------------


def check_skeleton_branch(run=_run) -> CheckResult:
    """`scratch/q2-skeleton-2026-08-04` exists on origin at d67e62f
    (Amendment-1 §0-amended / §3-amended step 3b) — the §1 with-cache arm
    needs this worktree."""
    try:
        proc = run(
            ["git", "ls-remote", "origin", f"refs/heads/{SKELETON_BRANCH}"],
            timeout=30.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return CheckResult(
            "skeleton.branch_at_origin", False, f"git ls-remote failed: {exc}"
        )

    if proc.returncode != 0:
        return CheckResult(
            "skeleton.branch_at_origin",
            False,
            f"git ls-remote exited {proc.returncode}: {proc.stderr.strip()}",
        )

    line = proc.stdout.strip()
    if not line:
        return CheckResult(
            "skeleton.branch_at_origin",
            False,
            f"origin has no ref refs/heads/{SKELETON_BRANCH}",
        )

    sha = line.split()[0]
    ok = sha == SKELETON_EXPECTED_SHA
    return CheckResult(
        "skeleton.branch_at_origin",
        ok,
        f"origin/{SKELETON_BRANCH} @ {sha}"
        + ("" if ok else f" (expected {SKELETON_EXPECTED_SHA})"),
    )


# ---------------------------------------------------------------------------
# 4. Fixture present
# ---------------------------------------------------------------------------


def check_fixture(repo_root: Path = REPO_ROOT) -> CheckResult:
    """examples/smoke-tests/kv-cache-tier1.api.json — the with-cache arm's
    graph template (Amendment-1 §3-amended "Fixture lineage")."""
    path = repo_root / FIXTURE_RELPATH
    if not path.is_file():
        return CheckResult("fixture.kv_cache_tier1_present", False, f"missing: {path}")
    return CheckResult("fixture.kv_cache_tier1_present", True, f"present: {path}")


# ---------------------------------------------------------------------------
# 5. Port 8199 free
# ---------------------------------------------------------------------------


def check_port_free(host: str = PREFLIGHT_HOST, port: int = PREFLIGHT_PORT) -> CheckResult:
    """The runsheet's isolated ComfyUI port (base §3 step 2 / `tests/e2e/
    conftest.py`'s `_free_port`) — a stale prior server would otherwise make
    the readiness poll pass against the wrong process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(2.0)
        in_use = s.connect_ex((host, port)) == 0
    return CheckResult(
        "port.preflight_port_free",
        not in_use,
        f"{host}:{port} " + ("already in use" if in_use else "free"),
    )


# ---------------------------------------------------------------------------
# 6. Environment provenance banking (#228's discipline)
# ---------------------------------------------------------------------------


def collect_environment_provenance(run=_run) -> dict:
    """torch/CUDA/driver/transformers/diffusers versions. #228: gate run 4's
    versions were never banked, closing the #226 environment-differential
    question off by construction; this run's are, unconditionally (banked
    even on FAIL/BLOCKED, since a blocked run's environment is itself
    diagnostic data per Amendment-1's OOM-typed-outcome note)."""
    provenance: dict[str, str | None] = {
        "driver_version": None,
        "cuda_version_nvidia_smi": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "transformers_version": None,
        "diffusers_version": None,
    }

    try:
        proc = run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=15.0,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            provenance["driver_version"] = proc.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        proc = run(["nvidia-smi"], timeout=15.0)
        if proc.returncode == 0:
            match = re.search(r"CUDA Version:\s*([\d.]+)", proc.stdout)
            if match:
                provenance["cuda_version_nvidia_smi"] = match.group(1)
    except (OSError, subprocess.SubprocessError):
        pass

    try:
        import torch  # type: ignore

        provenance["torch_version"] = torch.__version__
        provenance["torch_cuda_version"] = torch.version.cuda
    except ImportError:
        pass

    try:
        import transformers  # type: ignore

        provenance["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    try:
        import diffusers  # type: ignore

        provenance["diffusers_version"] = diffusers.__version__
    except ImportError:
        pass

    return provenance


# ---------------------------------------------------------------------------
# Aggregation + report
# ---------------------------------------------------------------------------


def run_all_checks(repo_root: Path = REPO_ROOT, run=_run) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(check_gpu(run=run))
    results.append(check_weights())
    results.append(check_skeleton_branch(run=run))
    results.append(check_fixture(repo_root=repo_root))
    results.append(check_port_free())
    return results


def aggregate_pass(results: list[CheckResult]) -> bool:
    """Exit 0 (True) iff every non-warning check passed. A `warning=True`
    result never fails the aggregate, per line 1's "warn, don't fail"."""
    return all(r.passed for r in results if not r.warning)


def format_report(
    results: list[CheckResult], provenance: dict, label: str
) -> str:
    lines = [f"Q-2 window preflight — label: {label}", f"Runsheet SSoT: {RUNSHEET_URL} (Amendment 1, §3-amended)", ""]
    for r in results:
        if r.warning:
            tag = "WARN"
        else:
            tag = "PASS" if r.passed else "FAIL"
        lines.append(f"[{tag}] {r.name}: {r.detail}")

    lines.append("")
    lines.append("Environment provenance (#228):")
    for key, value in provenance.items():
        lines.append(f"  {key}: {value if value is not None else '(unavailable)'}")

    lines.append("")
    overall = aggregate_pass(results)
    lines.append(f"Overall: {'PASS' if overall else 'FAIL'}")
    return "\n".join(lines)


def build_report_dict(results: list[CheckResult], provenance: dict, label: str) -> dict:
    return {
        "label": label,
        "runsheet_url": RUNSHEET_URL,
        "overall_pass": aggregate_pass(results),
        "checks": [
            {
                "name": r.name,
                "status": "WARN" if r.warning else ("PASS" if r.passed else "FAIL"),
                "passed": r.passed,
                "warning": r.warning,
                "detail": r.detail,
            }
            for r in results
        ],
        "environment_provenance": provenance,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="q2_preflight.py",
        description=(
            "Push-button precondition check for the Q-2 GPU window "
            "(Amendment 1 re-run route). Runsheet SSoT: the 2026-08-04 "
            f"'Amendment 1' comment on issue #62 ({RUNSHEET_URL}), "
            "§3-amended ('Runsheet deltas for the ComfyUI-server lane'). "
            "Read it with: gh issue view 62 --comments"
        ),
    )
    parser.add_argument(
        "--label",
        required=True,
        help=(
            "Run label used to name the --out file when --out is not given "
            "explicitly (no timestamp is auto-generated — pass a label that "
            "identifies this run, e.g. the date/window it's checking)."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path to write the environment-provenance + check report as JSON. "
            "Defaults to /tmp/q2-preflight-<label>.json."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_path = Path(args.out) if args.out else Path(f"/tmp/q2-preflight-{args.label}.json")

    if not shutil.which("git"):
        print("FAIL: git not found on PATH — cannot run the skeleton-branch check", file=sys.stderr)

    results = run_all_checks()
    provenance = collect_environment_provenance()

    print(format_report(results, provenance, args.label))

    report = build_report_dict(results, provenance, args.label)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nReport written to {out_path}")

    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
