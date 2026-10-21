"""BASE-02 native-distribution policy prerequisites and fail-closed exclusions."""
from __future__ import annotations

import copy
import hashlib
import unittest

from jsonschema import ValidationError

import i0_evidence
from i0_evidence import (
    EvidenceBlocked,
    validate_autoround_distribution_policy,
    validate_json_data,
    validate_unresolved_blocker_model,
)
from i0_testlib import ENGINE_DESIGN, FIXTURES, read_json, verify_base02_gate


class Base02AutoRoundDistributionPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        cls.defects = read_json(FIXTURES / "baseline-defects.json")

    @staticmethod
    def manifest_entries(lock: dict) -> list[dict]:
        fields = ("id", "path", "size", "sha256", "status", "required_for", "kind", "blocker")
        entries = []
        for identity in lock["evidence_root_contract"]["artifacts"]:
            descriptor = {key: copy.deepcopy(identity[key]) for key in fields if key in identity}
            lineage = identity.get("lineage_lock")
            if isinstance(lineage, dict) and lineage.get("status") == "BOUND":
                descriptor["derivation"] = {
                    "record": copy.deepcopy(lineage["record"]),
                    "command_log": copy.deepcopy(lineage["command_log"]),
                }
            entries.append(descriptor)
        return entries

    def manifest(self, entries: list[dict]) -> dict:
        return {
            "$schema": "urn:comfyui-diffusiongemma:dg-engine:i0:evidence-manifest:v1",
            "schema": 1,
            "kind": "dg-engine-i0-evidence-manifest",
            "contract_artifact_set_sha256": self.lock["evidence_root_contract"]["artifact_set_sha256"],
            "artifacts": entries,
        }

    def assert_structural_exclusion(self, lock: dict, entries: list[dict] | None = None) -> None:
        with self.assertRaisesRegex(
            EvidenceBlocked,
            "reviewed lock/schema changes|reviewed lock/schema update",
        ):
            validate_autoround_distribution_policy(lock, entries)

    def test_declared_broken_inventory_is_preserved_but_native_distribution_is_excluded(self):
        derivative = self.lock["autoround_derivative"]
        self.assertEqual(derivative["mode_status"], "declared-but-broken")
        self.assertIsNone(derivative["license"]["spdx"])
        self.assertEqual(
            derivative["distribution_disposition"],
            "EXCLUDED_FROM_NATIVE_ENGINE_DISTRIBUTION_AND_RELEASE_QUALIFICATION",
        )
        self.assertEqual(derivative["native_engine_distribution"], {
            "status": "EXCLUDED",
            "artifact_inclusion": "FORBIDDEN",
            "release_qualification": "NOT_RELIED_ON",
            "evidence_artifact_ids": [],
            "future_reentry_requirement": "EXPLICIT_SCOPE_AND_LICENSE_DECISION_REQUIRED",
        })
        validate_autoround_distribution_policy(self.lock)

    def test_exact_broken_mode_reproduction_and_rejection_evidence_remain_frozen(self):
        case = next(item for item in self.defects["cases"] if item["id"] == "AUTOROUND-DECLARED-BROKEN")
        self.assertEqual(case["classification"], "declared-but-broken-mode")
        self.assertIn("qweight but no weight", case["reproduction"])
        self.assertIn("current end-to-end mode is broken", case["observed"])
        self.assertIn("never silently map autoround to GGUF", case["frozen_native_expectation"])
        self.assertTrue(case["license_disposition"].startswith("EXCLUDED:"))
        self.assertIn("future explicit scope and license decision", case["license_disposition"])

    def test_autoround_license_is_not_a_phase_blocker_while_excluded(self):
        blocker_ids = {item["id"] for item in self.lock["unresolved_blockers"]}
        self.assertNotIn("I0-BLOCK-AUTOROUND-LICENSE", blocker_ids)
        self.assertNotIn("I0-BLOCK-INDEPENDENT-ROLES", blocker_ids)
        self.assertEqual(
            [item["id"] for item in validate_unresolved_blocker_model(self.lock)],
            ["I0-BLOCK-ADR-CDG-020"],
        )

    def test_complete_unresolved_blocker_model_rejects_set_and_classification_mutations(self):
        mutations = {}
        added = copy.deepcopy(self.lock)
        added["unresolved_blockers"].append({
            "id": "I0-BLOCK-UNDECLARED",
            "classification": "phase-policy",
            "reason": "hostile addition",
        })
        mutations["added"] = added
        removed = copy.deepcopy(self.lock)
        removed["unresolved_blockers"].pop()
        mutations["removed"] = removed
        renamed = copy.deepcopy(self.lock)
        renamed["unresolved_blockers"][-1]["id"] = "I0-BLOCK-ADR-RENAMED"
        mutations["renamed"] = renamed
        misclassified = copy.deepcopy(self.lock)
        misclassified["unresolved_blockers"][-1]["classification"] = "evidence"
        mutations["misclassified"] = misclassified
        for mutation, lock in mutations.items():
            with self.subTest(mutation=mutation), self.assertRaises(EvidenceBlocked):
                validate_unresolved_blocker_model(lock)

    def test_non_gguf_derivation_and_optional_descriptor_fields_are_closed(self):
        entries = self.manifest_entries(self.lock)
        source = next(item for item in entries if item["id"] == "comfy.server")
        source["derivation"] = {
            "record": {
                "id": "gguf.bf16.derivation-record",
                "path": "neutral/record.json",
                "size": 1,
                "sha256": "0" * 64,
            },
            "command_log": {
                "id": "gguf.bf16.command-log",
                "path": "neutral/log.json",
                "size": 1,
                "sha256": "1" * 64,
            },
        }
        with self.assertRaises(ValidationError):
            validate_json_data(
                self.manifest(entries),
                ENGINE_DESIGN / "schemas/i0/evidence-manifest.schema.json",
            )
        with self.assertRaisesRegex(EvidenceBlocked, "derivation|not bound"):
            i0_evidence._validate_manifest_against_lock(self.lock, self.manifest(entries))

        entries = self.manifest_entries(self.lock)
        next(item for item in entries if item["id"] == "comfy.server")["blocker"] = "I0-BLOCK-NEUTRAL"
        with self.assertRaisesRegex(EvidenceBlocked, "blocker differs"):
            i0_evidence._validate_manifest_against_lock(self.lock, self.manifest(entries))

        entries = self.manifest_entries(self.lock)
        gguf = next(item for item in entries if item["id"] == "gguf.bf16")
        gguf["derivation"] = copy.deepcopy(source["derivation"])
        with self.assertRaisesRegex(EvidenceBlocked, "UNBOUND"):
            i0_evidence._validate_manifest_against_lock(self.lock, self.manifest(entries))

    def test_gguf_nested_derivation_schema_closes_every_optional_field(self):
        entries = self.manifest_entries(self.lock)
        gguf = next(item for item in entries if item["id"] == "gguf.bf16")
        gguf["status"] = "present"
        gguf.pop("blocker")
        gguf["derivation"] = {
            "record": {
                "id": "gguf.bf16.derivation-record",
                "path": "gguf/bf16.record.json",
                "size": 1,
                "sha256": "0" * 64,
                "neutral_extra": "forbidden",
            },
            "command_log": {
                "id": "gguf.bf16.command-log",
                "path": "gguf/bf16.log.json",
                "size": 1,
                "sha256": "1" * 64,
            },
        }
        with self.assertRaises(ValidationError):
            validate_json_data(
                self.manifest(entries),
                ENGINE_DESIGN / "schemas/i0/evidence-manifest.schema.json",
            )

    def test_reference_contract_cannot_add_autoround_release_artifact(self):
        mutated = copy.deepcopy(self.lock)
        mutated["evidence_root_contract"]["artifacts"].append({
            "id": "release.autoround-int4",
            "path": "release/Intel-diffusiongemma-AutoRound.safetensors",
            "size": 1,
            "sha256": "0" * 64,
            "required_for": ["BASE-02-gate"],
        })
        with self.assertRaisesRegex(EvidenceBlocked, "future explicit scope and license decision"):
            validate_autoround_distribution_policy(mutated)

    def test_manifest_cannot_add_autoround_package_artifact(self):
        attempted = [{
            "id": "native-package.model-int4",
            "status": "present",
            "path": "assets/Intel/diffusiongemma-26B-A4B-it-int4-AutoRound/model.safetensors",
            "size": 1,
            "sha256": "0" * 64,
        }]
        with self.assertRaisesRegex(EvidenceBlocked, "future explicit scope and license decision"):
            validate_autoround_distribution_policy(self.lock, attempted)

    def test_inclusion_requires_code_and_lock_policy_change_not_manifest_only(self):
        mutated = copy.deepcopy(self.lock)
        mutated["autoround_derivative"]["native_engine_distribution"]["status"] = "INCLUDED"
        with self.assertRaisesRegex(EvidenceBlocked, "exclusion policy changed"):
            validate_autoround_distribution_policy(mutated)

    def test_marker_free_contract_and_manifest_addition_is_rejected(self):
        mutated = copy.deepcopy(self.lock)
        neutral = {
            "id": "native-package.model-int4",
            "path": "assets/model.safetensors",
            "size": 17,
            "sha256": hashlib.sha256(b"neutral model bytes").hexdigest(),
            "status": "present",
            "required_for": ["BASE-02-source"],
            "kind": "reference-source",
        }
        mutated["evidence_root_contract"]["artifacts"].append(copy.deepcopy(neutral))
        entries = self.manifest_entries(mutated)
        self.assert_structural_exclusion(mutated, entries)

    def test_marker_free_extra_manifest_artifact_is_rejected(self):
        entries = self.manifest_entries(self.lock)
        entries.append({
            "id": "supplemental.payload",
            "path": "assets/payload.bin",
            "size": 7,
            "sha256": hashlib.sha256(b"payload").hexdigest(),
            "status": "present",
            "required_for": ["BASE-02-source"],
            "kind": "reference-source",
        })
        self.assert_structural_exclusion(self.lock, entries)

    def test_neutral_source_and_release_gate_tagging_cannot_expand_contract(self):
        for required_for, kind in (
            (["BASE-02-source"], "reference-source"),
            (["BASE-02-gate"], "gguf-model"),
        ):
            with self.subTest(required_for=required_for):
                mutated = copy.deepcopy(self.lock)
                neutral = {
                    "id": "candidate.payload",
                    "path": "assets/candidate.bin",
                    "size": 9,
                    "sha256": hashlib.sha256(b"candidate").hexdigest(),
                    "status": "present",
                    "required_for": required_for,
                    "kind": kind,
                }
                mutated["evidence_root_contract"]["artifacts"].append(neutral)
                self.assert_structural_exclusion(mutated, self.manifest_entries(mutated))

    def test_allowed_id_cannot_be_repurposed_with_excluded_model_bytes_or_provenance(self):
        mutated = copy.deepcopy(self.lock)
        identity = next(
            item for item in mutated["evidence_root_contract"]["artifacts"]
            if item["id"] == "dg27.converter"
        )
        excluded_bytes = b"simulated bytes from the excluded quantized derivative"
        derivative = mutated["autoround_derivative"]
        identity.update({
            "path": "assets/model.bin",
            "size": len(excluded_bytes),
            "sha256": hashlib.sha256(excluded_bytes).hexdigest(),
            "source_identity": {
                "repository": f"https://huggingface.co/{derivative['repository']}",
                "revision": derivative["revision"],
                "path": "model.safetensors",
                "git_blob_sha1": "2" * 40,
            },
        })
        entries = self.manifest_entries(mutated)
        self.assert_structural_exclusion(mutated, entries)

    def test_omitted_and_substituted_expected_ids_are_rejected(self):
        for mutation in ("omitted", "substituted"):
            with self.subTest(mutation=mutation):
                mutated = copy.deepcopy(self.lock)
                artifacts = mutated["evidence_root_contract"]["artifacts"]
                target = next(item for item in artifacts if item["id"] == "dg27.converter")
                if mutation == "omitted":
                    artifacts.remove(target)
                else:
                    target["id"] = "neutral.converter"
                self.assert_structural_exclusion(mutated, self.manifest_entries(mutated))

    def test_exact_id_set_and_future_inclusion_require_frozen_contract_update(self):
        entries = self.manifest_entries(self.lock)
        validate_autoround_distribution_policy(self.lock, entries)
        future = copy.deepcopy(self.lock)
        future["evidence_root_contract"]["artifacts"].append({
            "id": "future.reviewed.payload",
            "path": "artifacts/future/payload.bin",
            "size": 1,
            "sha256": "0" * 64,
            "status": "absent",
            "required_for": ["BASE-02-source"],
            "kind": "reference-source",
            "blocker": "I0-BLOCK-FUTURE-EVIDENCE",
        })
        # Updating only the mutable lock fingerprint is insufficient: the
        # reviewed code/schema contract fingerprint remains the authority.
        future["evidence_root_contract"]["artifact_set_sha256"] = "f" * 64
        self.assert_structural_exclusion(future, self.manifest_entries(future))

    def test_expected_gate_blockers_are_fourteen_bytes_plus_three_lineages(self):
        result = verify_base02_gate(self.lock)
        byte_blockers = [item for item in result.blockers if item.blocker in {
            "I0-BLOCK-CHECKPOINT-EVIDENCE", "I0-BLOCK-GGUF-EVIDENCE"
        }]
        lineage_blockers = [item for item in result.blockers if item.blocker.startswith("I0-BLOCK-GGUF-LINEAGE-")]
        self.assertEqual(len(byte_blockers), 14)
        self.assertEqual(len(lineage_blockers), 3)
        self.assertEqual(len(result.blockers), 17)


if __name__ == "__main__":
    unittest.main(verbosity=2)
