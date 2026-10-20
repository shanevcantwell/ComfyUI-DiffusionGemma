"""Synthetic receipt tests only; no test here executes DiffusionGemma."""
from __future__ import annotations

import copy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("workflow", ROOT / "tools/workflow.py")
workflow = importlib.util.module_from_spec(spec)
spec.loader.exec_module(workflow)


class GateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.bank = Path(self.temp.name)
        self.flow = workflow.read_json(ROOT / "workflows/text.json")
        self.req = workflow.read_json(ROOT / "requirements.json")
        self.commit = "a" * 40
        self.receipts = {}
        for phase in self.flow["phases"]:
            pid = phase["id"]
            artifacts = []
            for name in (pid + "-tests.log", pid + "-model.md", pid + "-systems.md"):
                path = self.bank / name
                path.write_text("Synthetic policy-test fixture; not engine evidence.\n")
                artifacts.append({"path": name, "sha256": workflow.file_hash(path)})
            prerequisites = []
            for dep in phase["depends"]:
                name = dep + ".json"
                artifacts.append({"path": name, "sha256": workflow.file_hash(self.bank / name)})
                prerequisites.append({"phase": dep, "receipt": name})
            receipt = {
                "schema": 1, "kind": "engine-phase-evidence", "flow": "text", "phase": pid,
                "commit": self.commit, "workflow_sha256": workflow.canonical_hash(self.flow),
                "requirements_sha256": workflow.canonical_hash(self.req),
                "implementer_run": "author-" + pid, "verifier_run": "verifier-" + pid,
                "blockers": [], "prerequisites": prerequisites, "artifacts": artifacts,
                "tests": [{"id": tid, "status": "PASS", "executed": 1, "failed": 0,
                           "errors": 0, "skipped": 0, "log": pid + "-tests.log"}
                          for tid in sorted(workflow.cumulative_tests(self.flow, pid))],
                "reviews": [{"scope": scope, "run": scope + "-review-" + pid,
                             "status": "PASS", "commit": self.commit, "report": pid + "-" + scope + ".md"}
                            for scope in ("model", "systems")],
                "nodes": self.req["nodes"]}
            self.receipts[pid] = receipt
            (self.bank / (pid + ".json")).write_text(json.dumps(receipt))
        self.receipt = copy.deepcopy(self.receipts["I0"])

    def check(self, receipt=None, flow=None):
        return workflow.validate_receipt(flow or self.flow, self.req,
                                         self.receipt if receipt is None else receipt,
                                         self.bank, self.commit)

    def rejected(self):
        with self.assertRaises(workflow.GateError):
            self.check()

    def test_valid_early_phase_not_release(self):
        result = self.check()
        self.assertEqual(result["gate"], "PASS")
        self.assertFalse(result["release_eligible"])
        self.assertFalse(result["publication_authorized"])

    def test_valid_cumulative_release_still_not_publication(self):
        result = self.check(self.receipts["I5"])
        self.assertTrue(result["release_eligible"])
        self.assertFalse(result["publication_authorized"])

    def test_wrong_commit(self):
        self.receipt["commit"] = "b" * 40
        self.rejected()

    def test_invalid_candidate_hash(self):
        with self.assertRaises(workflow.GateError):
            workflow.validate_receipt(self.flow, self.req, self.receipt, self.bank, "HEAD")

    def test_changed_workflow(self):
        self.receipt["workflow_sha256"] = "0" * 64
        self.rejected()

    def test_changed_requirements(self):
        self.receipt["requirements_sha256"] = "0" * 64
        self.rejected()

    def test_blocker(self):
        self.receipt["blockers"] = ["cache mutation"]
        self.rejected()

    def test_missing_blocker_declaration(self):
        del self.receipt["blockers"]
        self.rejected()

    def test_same_verifier(self):
        self.receipt["verifier_run"] = self.receipt["implementer_run"]
        self.rejected()

    def test_missing_test(self):
        self.receipt["tests"].pop()
        self.rejected()

    def test_duplicate_test(self):
        self.receipt["tests"].append(self.receipt["tests"][0])
        self.rejected()

    def test_test_failure(self):
        self.receipt["tests"][0]["failed"] = 1
        self.rejected()

    def test_test_error(self):
        self.receipt["tests"][0]["errors"] = 1
        self.rejected()

    def test_skipped_tests(self):
        self.receipt["tests"][0]["skipped"] = 1
        self.rejected()

    def test_empty_collection(self):
        self.receipt["tests"][0]["executed"] = 0
        self.rejected()

    def test_bool_is_not_count(self):
        self.receipt["tests"][0]["executed"] = True
        self.rejected()

    def test_not_run(self):
        self.receipt["tests"][0]["status"] = "NOT_RUN"
        self.rejected()

    def test_missing_log(self):
        self.receipt["tests"][0]["log"] = "not-banked.log"
        self.rejected()

    def test_tampered_artifact(self):
        (self.bank / "I0-tests.log").write_text("changed")
        self.rejected()

    def test_empty_artifact(self):
        (self.bank / "I0-tests.log").write_text("")
        self.rejected()

    def test_missing_artifact(self):
        (self.bank / "I0-tests.log").unlink()
        self.rejected()

    def test_duplicate_artifact(self):
        self.receipt["artifacts"].append(self.receipt["artifacts"][0])
        self.rejected()

    def test_traversal(self):
        self.receipt["artifacts"][0]["path"] = "../outside"
        self.rejected()

    def test_absolute_path(self):
        self.receipt["artifacts"][0]["path"] = "/etc/passwd"
        self.rejected()

    def test_windows_path(self):
        self.receipt["artifacts"][0]["path"] = "C:\\evidence.log"
        self.rejected()

    def test_symlink_escape(self):
        path = self.bank / "escape"
        path.symlink_to("/etc/passwd")
        self.receipt["artifacts"][0]["path"] = "escape"
        self.rejected()

    def test_missing_review(self):
        self.receipt["reviews"].pop()
        self.rejected()

    def test_review_wrong_commit(self):
        self.receipt["reviews"][0]["commit"] = "b" * 40
        self.rejected()

    def test_self_review(self):
        self.receipt["reviews"][0]["run"] = self.receipt["implementer_run"]
        self.rejected()

    def test_duplicate_review_session(self):
        self.receipt["reviews"][1]["run"] = self.receipt["reviews"][0]["run"]
        self.rejected()

    def test_duplicate_scope(self):
        self.receipt["reviews"][1]["scope"] = "model"
        self.rejected()

    def test_unbanked_review(self):
        self.receipt["reviews"][0]["report"] = "unknown.md"
        self.rejected()

    def test_missing_prerequisite(self):
        self.receipt = copy.deepcopy(self.receipts["I1"])
        self.receipt["prerequisites"] = []
        self.rejected()

    def test_invalid_prerequisite_receipt(self):
        self.receipt = copy.deepcopy(self.receipts["I1"])
        prior = copy.deepcopy(self.receipts["I0"])
        prior["blockers"] = ["unresolved"]
        path = self.bank / "I0.json"
        path.write_text(json.dumps(prior))
        for item in self.receipt["artifacts"]:
            if item["path"] == "I0.json":
                item["sha256"] = workflow.file_hash(path)
        self.rejected()

    def test_missing_release_node(self):
        self.receipt = copy.deepcopy(self.receipts["I5"])
        self.receipt["nodes"].pop()
        self.rejected()

    def test_duplicate_release_node(self):
        self.receipt = copy.deepcopy(self.receipts["I5"])
        self.receipt["nodes"].append(self.receipt["nodes"][0])
        self.rejected()

    def test_disabled_image_gate(self):
        image = workflow.read_json(ROOT / "workflows/image.json")
        with self.assertRaises(workflow.GateError):
            self.check(flow=image)

    def test_cumulative_tests_include_baseline(self):
        self.assertIn("BASE-01", workflow.cumulative_tests(self.flow, "I5"))
        self.assertIn("NODE-08", workflow.cumulative_tests(self.flow, "I5"))

    def test_unknown_phase(self):
        with self.assertRaises(workflow.GateError):
            workflow.cumulative_tests(self.flow, "I99")

    def test_workflow_forward_dependency(self):
        flow = copy.deepcopy(self.flow)
        flow["phases"][0]["depends"] = ["I5"]
        with self.assertRaises(workflow.GateError):
            workflow.phase_map(flow)

    def test_non_object_receipt(self):
        with self.assertRaises(workflow.GateError):
            self.check(receipt=[])

    def test_cli_non_object_receipt(self):
        path = self.bank / "bad.json"
        path.write_text("[]")
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            code = workflow.main(["check", "text", "I0", str(path),
                                  "--evidence-dir", str(self.bank), "--commit", self.commit])
        self.assertEqual(code, 2)
        self.assertIn("Receipt must be an object", errors.getvalue())

    def test_phase_scope_not_premature(self):
        basic = workflow.cumulative_tests(self.flow, "I2")
        self.assertIn("BASIC-01", basic)
        self.assertNotIn("NODE-02", basic)
        self.assertNotIn("SECURITY-01", basic)
        complete = workflow.cumulative_tests(self.flow, "I4")
        self.assertTrue(set("NODE-%02d" % i for i in range(1, 9)) <= complete)

    def test_canonical_json_whitespace_independent(self):
        self.assertEqual(workflow.canonical_hash(self.flow),
                         workflow.canonical_hash(json.loads(json.dumps(self.flow, indent=2))))


if __name__ == "__main__":
    unittest.main()
