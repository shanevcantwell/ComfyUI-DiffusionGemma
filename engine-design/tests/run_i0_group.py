#!/usr/bin/env python3
"""Run an I0 group, enforce nonempty counts, and fail closed on prerequisites."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import unittest

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

from i0_evidence import (  # noqa: E402
    EvidenceBlocked,
    assert_oracle_suite_integrity,
    validate_autoround_distribution_policy,
    validate_unresolved_blocker_model,
)
from i0_testlib import (  # noqa: E402
    ENGINE_DESIGN,
    gate_categories_complete,
    read_json,
    scope_is_complete,
    verify_base02_gate,
    verify_evidence_scope,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("BASE-01", "BASE-02"))
    parser.add_argument("--evidence-root", required=True)
    args = parser.parse_args(argv)
    os.environ["DG_ENGINE_I0_EVIDENCE_ROOT"] = args.evidence_root

    pattern = "test_base_01*.py" if args.group == "BASE-01" else "test_base_02*.py"
    suite = unittest.defaultTestLoader.discover(str(TESTS), pattern=pattern)
    collected = suite.countTestCases()
    if collected <= 0:
        print(f"FAILED: {args.group}: collected=0", file=sys.stderr)
        return 1
    result = unittest.TextTestRunner(verbosity=2, stream=sys.stdout).run(suite)
    skipped = len(result.skipped)
    failed = len(result.failures)
    errors = len(result.errors)
    unexpected = len(result.unexpectedSuccesses)
    expected_failures = len(result.expectedFailures)
    print(
        f"SOURCE_SUBTESTS: group={args.group} collected={collected} executed={result.testsRun} "
        f"failed={failed} errors={errors} skipped={skipped} "
        f"expected_failures={expected_failures} unexpected_successes={unexpected}"
    )
    if result.testsRun != collected or failed or errors or skipped or expected_failures or unexpected:
        print(f"FAILED: {args.group}: source/synthetic subtests are incomplete", file=sys.stderr)
        return 1
    try:
        assert_oracle_suite_integrity()
    except BaseException as error:
        print(f"FAILED: {args.group}: final oracle suite integrity: {error}", file=sys.stderr)
        return 1
    print(f"ORACLE_FINAL_INTEGRITY: group={args.group} status=PASS")

    lock = read_json(ENGINE_DESIGN / "reference.lock.json")
    try:
        validate_autoround_distribution_policy(lock)
        policy_blockers = validate_unresolved_blocker_model(lock)
    except EvidenceBlocked as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2
    print("POLICY_PREREQUISITE: autoround_native_distribution=EXCLUDED status=PASS")
    print(
        "BLOCKER_CONTRACT: complete=PASS "
        f"phase_policy_ids={','.join(item['id'] for item in policy_blockers)}"
    )
    source_scope = "BASE-01-source" if args.group == "BASE-01" else "BASE-02-source"
    source_executed, source_blockers = verify_evidence_scope(lock, source_scope)
    print(
        f"SOURCE_EVIDENCE: group={args.group} executed={source_executed} "
        f"blocked={len(source_blockers)}"
    )
    if not scope_is_complete(source_executed, source_blockers):
        if source_executed <= 0:
            print(f"BLOCKED: I0-BLOCK-EMPTY-SOURCE-SCOPE: {source_scope}: executed=0", file=sys.stderr)
        for blocker in source_blockers:
            print(f"BLOCKED: {blocker}", file=sys.stderr)
        return 2

    if args.group == "BASE-02":
        gate = verify_base02_gate(lock)
        category_text = ",".join(f"{key}={value}" for key, value in sorted(gate.categories.items()))
        print(
            f"GATE_EVIDENCE: group=BASE-02 executed={gate.artifact_executed} "
            f"lineage_executed={gate.lineage_executed} categories={category_text} "
            f"artifact_or_lineage_blocked={len(gate.blockers)} policy_blocked={len(policy_blockers)}"
        )
        gate_scope_ok = (
            gate.artifact_executed > 0
            and gate.lineage_executed > 0
            and gate_categories_complete(lock, gate)
        )
        if gate.blockers or policy_blockers or not gate_scope_ok:
            seen: set[str] = set()
            for blocker in gate.blockers:
                key = f"{blocker.blocker}:{blocker.artifact_id}"
                if key not in seen:
                    print(f"BLOCKED: {blocker}", file=sys.stderr)
                    seen.add(key)
            for blocker in policy_blockers:
                if blocker["id"] not in seen:
                    print(f"BLOCKED: {blocker['id']}: {blocker['reason']}", file=sys.stderr)
                    seen.add(blocker["id"])
            if not gate_scope_ok:
                print(
                    "BLOCKED: I0-BLOCK-EMPTY-OR-INCOMPLETE-GATE-SCOPE: "
                    "prospective PASS requires nonzero artifact and lineage execution plus every required category.",
                    file=sys.stderr,
                )
            print("BLOCKED: BASE-02 is not PASS; source/synthetic subtests do not satisfy the gate.", file=sys.stderr)
            return 2

    print(f"GROUP_RESULT: {args.group}=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
