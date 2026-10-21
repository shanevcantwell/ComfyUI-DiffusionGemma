"""Real-process security regressions for the committed I0 launcher boundary."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest import mock

import i0_evidence
import run_i0_group


class WorkerBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = os.environ.get("DG_ENGINE_I0_TEST_EVIDENCE_ROOT")
        if not root: raise RuntimeError("DG_ENGINE_I0_TEST_EVIDENCE_ROOT is required")
        cls.evidence_root = Path(root).resolve(strict=True)
        if os.environ.get("DG_ENGINE_I0_BOUNDARY_TEST") != "1":
            raise RuntimeError("dedicated boundary-test marker is required")

    @staticmethod
    def fd_snapshot() -> set[str]: return set(os.listdir("/proc/self/fd"))

    def assert_blocked(self, call, pattern: str):
        before = self.fd_snapshot(); started = time.monotonic()
        with self.assertRaisesRegex(run_i0_group.WorkerBlocked, pattern): call()
        self.assertLess(time.monotonic() - started, 30 * 60)
        self.assertEqual(self.fd_snapshot(), before, "blocked launcher leaked a descriptor")

    def fault(self, name: str, *, group: str = "BASE-01", timeout: float = 120):
        return run_i0_group.launch_worker_test_fault(group, self.evidence_root, timeout=timeout, fault=name)

    def test_00_production_launcher_accepts_the_clean_committed_candidate(self):
        code, value = run_i0_group.launch_worker("BASE-01", self.evidence_root, timeout=240)
        self.assertEqual((code, value["disposition"]), (0, "PASS"))
        self.assertEqual(value["tests"]["count"], 50)

    def test_parent_imports_no_oracle_or_product_module_and_context_api_is_deleted(self):
        contaminated = [name for name in sys.modules if name.partition(".")[0] in
                        {"transformers", "diffusers", "dgemma", "surfaces"}]
        self.assertEqual(contaminated, [])
        with self.assertRaisesRegex(i0_evidence.EvidenceBlocked, "dedicated one-shot worker"):
            i0_evidence.load_authenticated_cdg_package({})

    def test_dirty_candidate_and_correspondingly_modified_worktree_lock_are_rejected(self):
        parent = run_i0_group.Path(run_i0_group.__file__)
        lock_path = run_i0_group.HARNESS_LOCK_PATH
        original_parent = parent.read_bytes(); original_lock = lock_path.read_bytes()
        try:
            changed = original_parent + b"\n"
            parent.write_bytes(changed)
            lock = json.loads(original_lock)
            source = next(item for item in lock["sources"] if item["id"] == "parent-runner")
            source["size"] = len(changed); source["sha256"] = hashlib.sha256(changed).hexdigest()
            lock_path.write_bytes(json.dumps(lock, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n")
            self.assert_blocked(
                lambda: run_i0_group.launch_worker("BASE-01", self.evidence_root, timeout=5),
                "production candidate is not clean",
            )
        finally:
            parent.write_bytes(original_parent); lock_path.write_bytes(original_lock)

    def test_extra_module_pth_and_sitecustomize_injection_is_rejected(self):
        root = Path("/tmp/dg-engine-i0-oracle-environment/site-packages")
        for name in ("injected_module.py", "injected.pth", "sitecustomize.py"):
            path = root / name
            try:
                path.write_text("raise RuntimeError('injected')\n")
                run_i0_group._TEST_ENV_CACHE = None
                self.assert_blocked(
                    lambda: run_i0_group.launch_worker("BASE-01", self.evidence_root, timeout=10),
                    "extras/missing|forbidden startup injection",
                )
            finally:
                path.unlink(missing_ok=True); run_i0_group._TEST_ENV_CACHE = None

    def test_real_worker_crash_timeout_and_fork_setsid_descendant_are_contained(self):
        for fault, timeout in (("crash", 10), ("hang", .3), ("fork-setsid-descendant", .5)):
            with self.subTest(fault=fault):
                self.assert_blocked(lambda f=fault,t=timeout: self.fault(f, timeout=t), "without a result|timeout")

    def test_real_worker_malformed_truncated_oversized_and_multiple_results_are_blocked(self):
        cases = {"malformed-result":"malformed JSON", "truncated-result":"malformed JSON",
                 "multiple-results":"malformed JSON|not one exact canonical", "oversized-result":"exceeded"}
        for fault, pattern in cases.items():
            with self.subTest(fault=fault): self.assert_blocked(lambda f=fault:self.fault(f, timeout=20), pattern)

    def test_real_worker_direct_and_buffered_output_is_parent_visible_blocked(self):
        for fault, stream in (("stdout-output","stdout"),("stderr-output","stderr"),("buffered-print","stdout")):
            with self.subTest(fault=fault):
                self.assert_blocked(lambda f=fault:self.fault(f, timeout=20), f"emitted a byte on {stream}")

    def test_each_child_identity_argument_check_is_reached(self):
        cases = {"wrong-worker-argument":"without a result", "wrong-environment-root":"without a result",
                 "wrong-interpreter-image":"without a result", "wrong-nonce-argument":"without a result"}
        for fault, pattern in cases.items():
            with self.subTest(fault=fault): self.assert_blocked(lambda f=fault:self.fault(f, timeout=20), pattern)

    def test_worker_and_bound_source_substitution_are_blocked(self):
        for fault in ("worker-substitution", "worker-fd-substitution"):
            with self.subTest(fault=fault):
                self.assert_blocked(lambda f=fault:self.fault(f, timeout=20), "without a result|emitted a byte")

    def test_result_fd_type_mode_and_duplication_are_blocked(self):
        for fault in ("result-fd-read-only", "result-fd-regular", "result-fd-duplicate"):
            with self.subTest(fault=fault):
                self.assert_blocked(lambda f=fault:self.fault(f, timeout=20), "without a result")

    def test_forged_counts_ids_blockers_hashes_nonce_paths_and_disposition_are_blocked(self):
        cases = {"wrong-candidate":"identities differ", "wrong-tree":"identities differ",
                 "wrong-environment":"identities differ", "wrong-worker-hash":"identities differ",
                 "wrong-interpreter-hash":"identities differ", "wrong-reference-lock-hash":"identities differ",
                 "wrong-harness-lock-hash":"identities differ", "wrong-evidence-hash":"identities differ",
                 "forged-counts":"counts are forged", "forged-test-ids":"test IDs/count/hash differ",
                 "forged-blockers":"blocker multiset", "wrong-disposition":"BASE-01 result",
                 "wrong-nonce-result":"nonce/group", "wrong-paths":"paths differ"}
        for fault, pattern in cases.items():
            with self.subTest(fault=fault): self.assert_blocked(lambda f=fault:self.fault(f, timeout=240), pattern)

    def test_forged_base02_gate_categories_are_blocked(self):
        self.assert_blocked(
            lambda: self.fault("forged-gate", group="BASE-02", timeout=700),
            "gate categories differ",
        )

    def test_authenticated_truncation_multiple_records_and_exit_mismatch_are_blocked(self):
        cases = {"truncated-authenticated-result":"malformed JSON|not one exact canonical",
                 "multiple-authenticated-results":"malformed JSON|not one exact canonical",
                 "wrong-exit":"exit code .* differs"}
        for fault, pattern in cases.items():
            with self.subTest(fault=fault): self.assert_blocked(lambda f=fault:self.fault(f, timeout=240), pattern)

    def test_preload_thread_and_final_oracle_mutations_emit_no_result(self):
        for fault, group in (("preload","BASE-01"),("extra-thread","BASE-01"),
                             ("oracle-mutation","BASE-02"),("oracle-dependency-mutation","BASE-02")):
            with self.subTest(fault=fault):
                self.assert_blocked(
                    lambda f=fault,g=group:self.fault(f, group=g, timeout=700),
                    "without a result|left a live descendant",
                )

    def test_spawn_failure_closes_all_descriptors_without_factory_injection(self):
        self.assert_blocked(lambda:self.fault("spawn-failure", timeout=10), "spawn failed: boundary injected")

    def test_parent_environment_poison_is_not_inherited(self):
        poisoned = {"PATH":"/untrusted", "PYTHONPATH":"/untrusted", "PYTHONUSERBASE":"/untrusted",
                    "PYTHONHOME":"/untrusted", "LD_PRELOAD":"/untrusted.so", "LD_LIBRARY_PATH":"/untrusted"}
        with mock.patch.dict(os.environ, poisoned):
            code, value = run_i0_group.launch_worker("BASE-01", self.evidence_root, timeout=240)
        self.assertEqual((code, value["disposition"]), (0, "PASS"))

    def test_runner_under_wrong_python_or_without_no_site_is_blocked(self):
        for flags in (("-I","-S","-B","-u"),("-I","-B","-u")):
            completed = subprocess.run(
                ["/usr/bin/python3.12", *flags, str(run_i0_group.__file__), "BASE-01", "--evidence-root",
                 str(self.evidence_root), "--worker-timeout", "1"], cwd=run_i0_group.REPO_ROOT, env={},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            self.assertEqual(completed.returncode, 2)
            self.assertEqual(completed.stdout, "")
            self.assertRegex(completed.stderr, "not locked launcher|requires exact Python")


if __name__ == "__main__": unittest.main()
