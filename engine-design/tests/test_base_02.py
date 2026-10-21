"""BASE-02 source subtests; the group runner separately enforces BLOCKED gate inputs."""
from __future__ import annotations

import ast
import gc
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest import mock

import torch

import i0_evidence
from i0_testlib import (
    ENGINE_DESIGN,
    FIXTURES,
    REPO_ROOT,
    EvidenceBlocked,
    checked_distribution_bytes,
    decoder_visible_positions,
    entropy_bound_accept,
    inverse_cdf,
    authenticated_worker_package,
    read_authenticated_cdg_source,
    read_evidence_artifact,
    read_json,
    retained_prefix_positions,
    sha256,
    stable_softmax,
    top_k_contract,
    verify_base02_gate,
    verify_evidence_scope,
)


def class_source(source: str, class_name: str) -> str:
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError(f"no source segment for {class_name}")
    return segment


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return __import__("hashlib").sha1(header + data).hexdigest()


class Base02LocksAndSourceBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        # The first oracle context owns the clean import boundary.  Tests may
        # inspect dependency modules only after that independent snapshot is
        # established; preloading them would now fail closed.
        cls.pack, cls.pack_loader = authenticated_worker_package()
        cls.oracle_dependency_snapshot = {}
        for package in cls.lock["oracle_packages"]:
            for identity in package["files"]:
                relative = identity["path"].removeprefix(f"src/{package['name']}/").removesuffix(".py")
                module_name = package["name"] + "." + relative.replace("/", ".")
                module = importlib.import_module(module_name)
                origin = Path(module.__spec__.origin).resolve()
                cls.oracle_dependency_snapshot[module_name] = {
                    "module": module,
                    "origin": origin,
                    "version": importlib.metadata.version(package["name"]),
                    "sha256": sha256(origin),
                    "expected_sha256": identity["sha256"],
                }
        cls.semantic = read_json(FIXTURES / "semantics.json")
        cls.defects = read_json(FIXTURES / "baseline-defects.json")

    def test_source_evidence_scope_is_nonempty_and_fully_authenticated(self):
        executed, blockers = verify_evidence_scope(self.lock, "BASE-02-source")
        self.assertGreater(executed, 0)
        self.assertEqual(blockers, [])

    def test_every_interpreted_cdg_source_is_bound_to_baseline_bytes(self):
        identities = self.lock["cdg_source_identities"]
        self.assertGreaterEqual(len(identities), 34)
        for identity in identities:
            with self.subTest(path=identity["path"]):
                self.assertEqual(identity["revision"], self.lock["product_baseline"])
                data = read_authenticated_cdg_source(self.lock, identity["path"])
                self.assertEqual(len(data), identity["size"])
                self.assertEqual(__import__("hashlib").sha256(data).hexdigest(), identity["sha256"])

    def test_product_and_upstream_authorities_resolve_to_bytes(self):
        baseline = self.lock["product_baseline"]
        subprocess.run(["git", "cat-file", "-e", baseline + "^{commit}"], cwd=REPO_ROOT, check=True)
        license_bytes = read_evidence_artifact(self.lock, "llama.license")
        llama = next(item for item in self.lock["source_baselines"] if item["id"] == "LLAMA")
        self.assertEqual(__import__("hashlib").sha256(license_bytes).hexdigest(), llama["license"]["sha256"])
        self.assertEqual(llama["license"]["evidence"]["git_blob_sha1"], "e7dca554bcb802f98408383a864404e3aa4eacca")
        self.assertFalse(next(item for item in self.lock["source_baselines"] if item["id"] == "DG23")["correctness_authority"])
        self.assertFalse(next(item for item in self.lock["source_baselines"] if item["id"] == "DG27")["correctness_authority"])

    def test_preloaded_oracle_dependencies_keep_exact_identity_origin_version_and_source_hash(self):
        expected_versions = {package["name"]: package["version"] for package in self.lock["oracle_packages"]}
        expected_hashes = self.lock["oracle_environment"]["installed_source_sha256"]
        self.assertEqual(len(self.oracle_dependency_snapshot), 6)
        for module_name, snapshot in self.oracle_dependency_snapshot.items():
            package_name = module_name.partition(".")[0]
            relative = module_name.removeprefix(package_name + ".").replace(".", "/") + ".py"
            hash_key = f"{package_name}:{Path(relative).name}"
            with self.subTest(module=module_name):
                self.assertIs(sys.modules[module_name], snapshot["module"])
                self.assertEqual(Path(sys.modules[module_name].__spec__.origin).resolve(), snapshot["origin"])
                self.assertEqual(importlib.metadata.version(package_name), snapshot["version"])
                self.assertEqual(snapshot["version"], expected_versions[package_name])
                self.assertEqual(sha256(snapshot["origin"]), snapshot["sha256"])
                self.assertEqual(snapshot["sha256"], snapshot["expected_sha256"])
                self.assertEqual(snapshot["sha256"], expected_hashes[hash_key])

    def test_product_captures_exact_trusted_package_root_class_objects(self):
        loop_module = self.pack_loader.import_module("dgemma.loop")
        model_module = self.pack_loader.import_module("dgemma.model")
        expected_products = {
            "DiffusionGemmaPipeline": loop_module,
            "EntropyBoundScheduler": loop_module,
            "AutoProcessor": model_module,
            "DiffusionGemmaForBlockDiffusion": model_module,
        }
        for export, product_module in expected_products.items():
            with self.subTest(export=export):
                expected = self.pack_loader.oracle_object(export)
                self.assertIs(getattr(product_module, export), expected)
                exact_module = importlib.import_module(expected.__module__)
                package_root = importlib.import_module(expected.__module__.partition(".")[0])
                self.assertIs(getattr(exact_module, export), expected)
                self.assertIs(getattr(package_root, export), expected)
                self.assertIsInstance(exact_module.__spec__.loader, i0_evidence._AuthenticatedOracleFinder)
                self.assertIsInstance(package_root.__spec__.loader, i0_evidence._AuthenticatedOracleFinder)
                self.assertIs(exact_module.__spec__.loader, package_root.__spec__.loader)
                source = Path(__import__("inspect").getsourcefile(expected)).resolve()
                self.assertEqual(source, Path(exact_module.__spec__.origin).resolve())
        self.pack_loader.assert_oracle_bindings()

    def test_exact_oracle_packages_match_installed_and_evidence_bytes(self):
        self.assertEqual(importlib.metadata.version("transformers"), "5.13.0")
        self.assertEqual(importlib.metadata.version("diffusers"), "0.39.0")
        import diffusers
        import transformers

        roots = {"transformers": Path(transformers.__file__).parent, "diffusers": Path(diffusers.__file__).parent}
        for package in self.lock["oracle_packages"]:
            root_identity = package["root_source"]
            root_installed = roots[package["name"]] / "__init__.py"
            self.assertEqual(root_identity["path"], f"src/{package['name']}/__init__.py")
            self.assertEqual(sha256(root_installed), root_identity["sha256"])
            for identity in package["files"]:
                relative = identity["path"].removeprefix(f"src/{package['name']}/")
                installed = roots[package["name"]] / relative
                artifact_id = f"package.{package['name']}.{Path(relative).name}"
                evidence = read_evidence_artifact(self.lock, artifact_id)
                with self.subTest(package=package["name"], source=relative):
                    self.assertEqual(sha256(installed), identity["sha256"])
                    self.assertEqual(__import__("hashlib").sha256(evidence).hexdigest(), identity["sha256"])
                    self.assertEqual(installed.read_bytes(), evidence)
                    if "git_blob_sha1" in identity:
                        self.assertEqual(git_blob_sha1(evidence), identity["git_blob_sha1"])

    def test_selected_converter_bytes_and_mapping_are_exact(self):
        converter = self.lock["converter"]
        source = read_evidence_artifact(self.lock, "dg27.converter").decode("utf-8")
        converter_bytes = source.encode()
        self.assertEqual(__import__("hashlib").sha256(converter_bytes).hexdigest(), converter["sha256"])
        self.assertEqual(git_blob_sha1(converter_bytes), converter["git_blob_sha1"])
        tree = ast.parse(source)
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DiffusionGemmaModel")
        rendered = ast.dump(cls)
        for required in ("ENC_LAYER_OUT_SCALE", "SC_PRE_NORM", "SC_GATE", "SC_UP", "SC_DOWN"):
            self.assertIn(required, rendered)
        self.assertIn("add_diffusion_canvas_length", source)
        self.assertIn("model.encoder.layers.", source)
        self.assertEqual(converter["selection_relation_to_adr_cdg_020"], "consistent-with-accepted-pin")
        self.assertFalse(converter["correctness_authority"])

    def test_official_config_tokenizer_template_and_sampler_bytes(self):
        model = self.lock["official_model"]
        identities = {item["path"]: item for item in model["small_files"]}
        observed = {}
        for path, identity in identities.items():
            artifact_id = "official." + path.replace("/", ".")
            data = read_evidence_artifact(self.lock, artifact_id)
            observed[path] = data
            self.assertEqual(len(data), identity["size"])
            self.assertEqual(__import__("hashlib").sha256(data).hexdigest(), identity["sha256"])
            if identity["git_blob_kind"] == "content":
                self.assertEqual(git_blob_sha1(data), identity["git_blob_sha1"])
            else:
                self.assertEqual(identity["git_blob_kind"], "lfs-pointer")
                self.assertNotEqual(git_blob_sha1(data), identity["git_blob_sha1"])
        config = __import__("json").loads(observed["config.json"])
        generation = __import__("json").loads(observed["generation_config.json"])
        self.assertEqual(config["canvas_length"], self.semantic["oracle_profile"]["canvas_length"])
        self.assertEqual(config["text_config"]["vocab_size"], self.semantic["oracle_profile"]["vocab_size"])
        self.assertEqual(generation["max_denoising_steps"], self.semantic["oracle_profile"]["num_inference_steps"])
        self.assertEqual(generation["max_new_tokens"], 256)
        self.assertIn(b"<|think|>", observed["chat_template.jinja"])
        tokenizer = __import__("json").loads(observed["tokenizer.json"])
        self.assertIsInstance(tokenizer, dict)
        self.assertGreater(len(observed["tokenizer.json"]), 30_000_000)

    def test_every_reconnaissance_identity_hashes_exact_bytes(self):
        recon = self.lock["reconnaissance"]
        groups = (
            recon["ten_current_model_reads"],
            recon["three_recent_merged_model_additions"],
            recon["developer_document_reads"],
        )
        self.assertEqual([len(group) for group in groups], [10, 3, 8])
        seen = set()
        for group in groups:
            for identity in group:
                with self.subTest(artifact=identity["artifact_id"]):
                    self.assertNotIn(identity["artifact_id"], seen)
                    seen.add(identity["artifact_id"])
                    data = read_evidence_artifact(self.lock, identity["artifact_id"])
                    self.assertEqual(len(data), identity["size"])
                    self.assertEqual(__import__("hashlib").sha256(data).hexdigest(), identity["sha256"])
                    self.assertEqual(git_blob_sha1(data), identity["git_blob_sha1"])
                    self.assertEqual(len(data.decode("utf-8").splitlines()), identity["lines"])
                    self.assertTrue(identity["read_in_full"])

    def test_fixture_hashes_are_bound_to_public_bytes(self):
        for identity in self.lock["fixtures"]:
            with self.subTest(fixture=identity["path"]):
                self.assertEqual(sha256(ENGINE_DESIGN / identity["path"]), identity["sha256"])

    def test_checkpoint_identities_are_exact_gate_requirements(self):
        gate = [
            identity
            for identity in self.lock["evidence_root_contract"]["artifacts"]
            if identity["id"].startswith("official.checkpoint.")
        ]
        self.assertEqual(len(gate), 11)
        self.assertTrue(all(item["blocker"] == "I0-BLOCK-CHECKPOINT-EVIDENCE" for item in gate))
        self.assertEqual(
            [(item["size"], item["sha256"]) for item in gate],
            [(item["size"], item["lfs_sha256"]) for item in self.lock["official_model"]["checkpoint_shards"]],
        )

    def test_gguf_byte_identities_are_known_but_lineage_is_unbound(self):
        gate = [
            identity
            for identity in self.lock["evidence_root_contract"]["artifacts"]
            if identity["id"].startswith("gguf.")
        ]
        self.assertEqual(len(gate), 3)
        self.assertTrue(all(item["identity_status"] == "KNOWN_BYTES" for item in gate))
        self.assertTrue(all(item["lineage_status"] == "UNBOUND" for item in gate))
        self.assertTrue(all("lineage" not in item for item in gate))
        self.assertTrue(all(item["lineage_lock"] == {
            "status": "UNBOUND", "derivation_id": None, "record": None, "command_log": None,
        } for item in gate))
        remote = self.lock["gguf_remote_reference"]
        self.assertEqual(remote["lineage_status"], "UNBOUND")
        self.assertTrue(all(item["lineage_claim"] is None for item in remote["artifacts"]))
        mismatch = self.lock["rejected_artifacts"][0]
        self.assertNotEqual(mismatch["observed_sha256"], mismatch["required_sha256"])
        self.assertEqual(mismatch["disposition"], "REJECTED_AS_SUBSTITUTE")

    def test_gguf_lineage_requires_independent_authenticated_derivations(self):
        gate = verify_base02_gate(self.lock)
        lineage = [item for item in gate.blockers if item.blocker.startswith("I0-BLOCK-GGUF-LINEAGE-")]
        self.assertEqual(len(lineage), 3)
        self.assertEqual(gate.lineage_executed, 0)
        requirements = [
            item["derivation_requirement"]
            for item in self.lock["evidence_root_contract"]["artifacts"]
            if item["id"].startswith("gguf.")
        ]
        self.assertTrue(all(item["independent_from_manifest"] for item in requirements))
        self.assertTrue(all(item["record_schema"].endswith("derivation-record.schema.json") for item in requirements))
        self.assertTrue(all(item["command_log_schema"].endswith("command-log.schema.json") for item in requirements))
        self.assertTrue(all(item["trust_root"] == "Git-controlled reference.lock.json lineage_lock" for item in requirements))
        self.assertIn("I2 GRAPH-04", self.lock["phase_scope_note"])
