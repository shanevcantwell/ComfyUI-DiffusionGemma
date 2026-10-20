#!/usr/bin/env python3
"""Plan harness work and validate banked phase receipts; never run inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class GateError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def canonical_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def phase_map(flow: dict) -> dict:
    require(isinstance(flow, dict), "Workflow must be an object")
    require(flow.get("schema") == 1, "Unsupported workflow schema")
    require(isinstance(flow.get("phases"), list), "Missing phases")
    phases: dict = {}
    for phase in flow["phases"]:
        pid = phase["id"]
        require(pid not in phases, "Duplicate phase: " + pid)
        require(set(phase["depends"]) <= set(phases), "Dependencies must precede " + pid)
        require(bool(phase["tests"]), "Empty test contract: " + pid)
        require(len(phase["tests"]) == len(set(phase["tests"])), "Duplicate test ID")
        phases[pid] = phase
    return phases


def cumulative_tests(flow: dict, pid: str) -> set[str]:
    phases = phase_map(flow)
    require(pid in phases, "Unknown phase: " + pid)
    result = set(phases[pid]["tests"])
    for parent in phases[pid]["depends"]:
        result.update(cumulative_tests(flow, parent))
    return result


def safe_path(root: Path, name: str) -> Path:
    require(isinstance(name, str) and bool(name), "Empty artifact path")
    part = PurePosixPath(name)
    require(not part.is_absolute() and ".." not in part.parts, "Unsafe artifact path")
    require("\\" not in name and ":" not in name and "\0" not in name, "Unsafe artifact path")
    require(str(part) == name and name != ".", "Noncanonical artifact path")
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Artifact escapes evidence root")
    require(path.is_file(), "Missing artifact: " + name)
    return path


def validate_receipt(flow: dict, requirements: dict, receipt: dict,
                     root: Path, expected_commit: str) -> dict:
    require(isinstance(receipt, dict), "Receipt must be an object")
    require(isinstance(expected_commit, str), "Candidate commit must be a string")
    require(flow.get("enabled") is True, "Workflow disabled; planning is not implementation authority")
    phases = phase_map(flow)
    pid = receipt.get("phase")
    require(pid in phases, "Unknown receipt phase")
    phase = phases[pid]
    require(receipt.get("schema") == 1, "Unsupported receipt schema")
    require(receipt.get("kind") == "engine-phase-evidence", "Not an engine phase receipt")
    require(receipt.get("flow") == flow["flow"], "Wrong workflow")
    require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "Invalid candidate commit")
    require(receipt.get("commit") == expected_commit, "Stale or wrong candidate commit")
    require(receipt.get("workflow_sha256") == canonical_hash(flow), "Workflow changed")
    require(receipt.get("requirements_sha256") == canonical_hash(requirements), "Requirements changed")
    require(receipt.get("blockers") == [], "Unresolved or undeclared blockers")

    implementer = receipt.get("implementer_run")
    verifier = receipt.get("verifier_run")
    require(isinstance(implementer, str) and bool(implementer.strip()), "Missing implementer run")
    require(isinstance(verifier, str) and bool(verifier.strip()), "Missing verifier run")
    require(implementer != verifier, "Verifier is implementer")

    artifacts: dict = {}
    require(isinstance(receipt.get("artifacts"), list), "Missing artifacts")
    for artifact in receipt["artifacts"]:
        name = artifact["path"]
        require(name not in artifacts, "Duplicate artifact: " + name)
        path = safe_path(root, name)
        require(path.stat().st_size > 0, "Empty evidence: " + name)
        require(file_hash(path) == artifact["sha256"], "Evidence hash mismatch: " + name)
        artifacts[name] = path

    tests: dict = {}
    require(isinstance(receipt.get("tests"), list), "Missing tests")
    for test in receipt["tests"]:
        tid = test["id"]
        require(tid not in tests, "Duplicate test group: " + tid)
        require(test.get("status") == "PASS", "Nonpassing test group: " + tid)
        require(type(test.get("executed")) is int and test["executed"] > 0, "No executed tests: " + tid)
        for field in ("failed", "errors", "skipped"):
            require(type(test.get(field)) is int and test[field] == 0, field + " cases in " + tid)
        require(test.get("log") in artifacts, "Unbanked test log: " + tid)
        tests[tid] = test
    missing = cumulative_tests(flow, pid) - set(tests)
    require(not missing, "Missing cumulative test groups: " + ", ".join(sorted(missing)))

    scopes: set[str] = set()
    runs = {implementer, verifier}
    require(isinstance(receipt.get("reviews"), list), "Missing reviews")
    for review in receipt["reviews"]:
        scope = review["scope"]
        require(scope not in scopes, "Duplicate review scope")
        require(review.get("status") == "PASS", "Nonpassing review")
        require(review.get("commit") == expected_commit, "Review covers different code")
        run = review.get("run")
        require(isinstance(run, str) and bool(run.strip()) and run not in runs, "Review session is not separate")
        require(review.get("report") in artifacts, "Unbanked review report")
        scopes.add(scope)
        runs.add(run)
    require(scopes == set(flow["required_review_scopes"]), "Missing or unexpected review scopes")

    previous: dict = {}
    require(isinstance(receipt.get("prerequisites"), list), "Missing prerequisite receipts")
    for prerequisite in receipt["prerequisites"]:
        parent = prerequisite["phase"]
        require(parent in phase["depends"] and parent not in previous, "Wrong or duplicate prerequisite")
        name = prerequisite["receipt"]
        require(name in artifacts, "Unbanked prerequisite receipt")
        prior = read_json(artifacts[name])
        require(isinstance(prior, dict), "Prerequisite receipt must be an object")
        require(prior.get("phase") == parent, "Prerequisite phase mismatch")
        validate_receipt(flow, requirements, prior, root, prior.get("commit", ""))
        previous[parent] = prior["commit"]
    require(set(previous) == set(phase["depends"]), "Missing prerequisite phase")

    release = phase.get("release_candidate") is True
    if release:
        nodes = receipt.get("nodes")
        require(isinstance(nodes, list), "Missing release node coverage")
        require(len(nodes) == len(set(nodes)) and set(nodes) == set(requirements["nodes"]), "Incomplete release node coverage")
    return {"gate": "PASS", "phase": pid, "commit": expected_commit,
            "release_eligible": release, "publication_authorized": False,
            "prerequisite_commits": previous}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "hashes", "packet", "check"):
        command = sub.add_parser(name)
        command.add_argument("flow", choices=("text", "image"))
        if name in ("packet", "check"):
            command.add_argument("phase")
        if name == "check":
            command.add_argument("receipt", type=Path)
            command.add_argument("--evidence-dir", required=True, type=Path)
            command.add_argument("--commit", required=True)
    args = parser.parse_args(argv)
    try:
        flow = read_json(ROOT / "workflows" / (args.flow + ".json"))
        requirements = read_json(ROOT / "requirements.json")
        phases = phase_map(flow)
        if args.command == "plan":
            print(json.dumps(flow, indent=2))
        elif args.command == "hashes":
            print(json.dumps({"workflow_sha256": canonical_hash(flow),
                              "requirements_sha256": canonical_hash(requirements)}, indent=2))
        elif args.command == "packet":
            require(args.phase in phases, "Unknown phase")
            phase = phases[args.phase]
            print("# " + phase["id"] + ": " + phase["title"])
            print("\nImplementation enabled: " + str(flow["enabled"]))
            print("Read sources.md, architecture.md, decisions.md, verification.md and workflows/README.md.")
            print("Separate roles: Explore, implementer, verifier, model reviewer, systems reviewer.")
            for task in phase["tasks"]:
                print("\n" + task)
            print("\nDeliverables: " + ", ".join(phase["deliverables"]))
            print("Cumulative tests: " + ", ".join(sorted(cumulative_tests(flow, args.phase))))
            print("Bank exact-candidate evidence. Never promote skipped tests or self-review.")
        else:
            receipt = read_json(args.receipt)
            require(isinstance(receipt, dict), "Receipt must be an object")
            require(receipt.get("phase") == args.phase, "Requested phase differs from receipt")
            result = validate_receipt(flow, requirements, receipt, args.evidence_dir, args.commit)
            print(json.dumps(result, indent=2))
    except (GateError, KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        print("BLOCKED: " + str(error), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
