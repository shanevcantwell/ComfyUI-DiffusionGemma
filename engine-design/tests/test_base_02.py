"""BASE-02 source subtests; the group runner separately enforces BLOCKED gate inputs."""
from __future__ import annotations

import ast
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import subprocess
import sys
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
    load_authenticated_cdg_package,
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
        cls.pack, cls.pack_loader = cls.enterClassContext(
            load_authenticated_cdg_package(cls.lock, verify_oracle_bindings=True)
        )
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



class Base02OracleMutationRejections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.class_lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        if i0_evidence._ORACLE_TRUST_SNAPSHOT is None:
            with load_authenticated_cdg_package(
                cls.class_lock,
                "_dg_engine_i0_oracle_test_bootstrap",
                verify_oracle_bindings=True,
            ) as (_package, loader):
                loader.assert_oracle_bindings()

    def setUp(self):
        self.lock = self.class_lock

    def assert_namespace_absent(self, package_name: str) -> None:
        prefix = package_name + "."
        self.assertFalse(any(name == package_name or name.startswith(prefix) for name in sys.modules))

    @staticmethod
    def oracle_global_namespace(snapshot, export: str, global_name: str) -> dict:
        for _role, function in i0_evidence._class_functions(snapshot.object_for(export)):
            if global_name in function.__code__.co_names and global_name in function.__globals__:
                return function.__globals__
        raise AssertionError(f"{export} does not resolve module global {global_name}")

    @staticmethod
    def closure_global_namespace(snapshot, export: str, global_name: str) -> dict:
        pending = [function for _role, function in i0_evidence._class_functions(snapshot.object_for(export))]
        seen = set()
        while pending:
            function = pending.pop()
            if id(function) in seen:
                continue
            seen.add(id(function))
            if global_name in function.__code__.co_names and global_name in function.__globals__:
                return function.__globals__
            for cell in function.__closure__ or ():
                try:
                    held = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(held, types.FunctionType):
                    pending.append(held)
        raise AssertionError(f"{export} closure does not resolve module global {global_name}")

    def test_coordinated_root_and_submodule_spoof_with_trusted_module_name_is_rejected(self):
        bindings = (
            ("diffusers", "diffusers.pipelines.diffusion_gemma.pipeline_diffusion_gemma", "DiffusionGemmaPipeline"),
            ("diffusers", "diffusers.schedulers.scheduling_entropy_bound", "EntropyBoundScheduler"),
            ("transformers", "transformers.models.auto.processing_auto", "AutoProcessor"),
            (
                "transformers",
                "transformers.models.diffusion_gemma.modeling_diffusion_gemma",
                "DiffusionGemmaForBlockDiffusion",
            ),
        )
        for index, (package_name, module_name, export) in enumerate(bindings):
            with self.subTest(export=export):
                exact_module = importlib.import_module(module_name)
                package_root = importlib.import_module(package_name)
                trusted = getattr(exact_module, export)
                spoof = type(
                    f"Spoof{export}",
                    (trusted,),
                    {"__module__": module_name},
                )
                reserved = f"_dg_engine_i0_oracle_double_spoof_{index}"
                with mock.patch.object(package_root, export, spoof), mock.patch.object(exact_module, export, spoof):
                    with self.assertRaisesRegex(EvidenceBlocked, "export identity changed"):
                        with load_authenticated_cdg_package(
                            self.lock,
                            reserved,
                            verify_oracle_bindings=True,
                            acquisition_timeout=0.5,
                        ):
                            self.fail("coordinated root/submodule spoof must not authenticate")
                self.assertIs(getattr(package_root, export), trusted)
                self.assertIs(getattr(exact_module, export), trusted)
                self.assert_namespace_absent(reserved)
                self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)

    def test_first_trust_initialization_rejects_preloaded_target_modules(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        self.assertIsNotNone(snapshot)
        i0_evidence._ORACLE_TRUST_SNAPSHOT = None
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "preloaded before trust initialization"):
                with load_authenticated_cdg_package(
                    self.lock,
                    "_dg_engine_i0_preloaded_oracle",
                    verify_oracle_bindings=True,
                    acquisition_timeout=0.5,
                ):
                    self.fail("preloaded oracle modules must fail closed")
        finally:
            i0_evidence._ORACLE_TRUST_SNAPSHOT = snapshot
        self.assert_namespace_absent("_dg_engine_i0_preloaded_oracle")
        self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)

    def test_failed_mid_establishment_rolls_back_and_allows_clean_retry(self):
        original_snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        original_modules = {
            name: module
            for name, module in tuple(sys.modules.items())
            if name == "diffusers" or name.startswith("diffusers.")
            or name == "transformers" or name.startswith("transformers.")
        }
        baseline_meta_path = tuple(sys.meta_path)
        for name, module in original_modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
        i0_evidence._ORACLE_TRUST_SNAPSHOT = None
        real_import_module = importlib.import_module
        failed = False
        foreign_name = "foreign_oracle_establishment_change"
        foreign_module = types.ModuleType(foreign_name)

        def fail_mid_establishment(name, package=None):
            nonlocal failed
            if name == "diffusers.schedulers.scheduling_entropy_bound" and not failed:
                failed = True
                sys.modules[foreign_name] = foreign_module
                raise RuntimeError("injected mid-establishment failure")
            return real_import_module(name, package)

        try:
            with mock.patch.object(importlib, "import_module", side_effect=fail_mid_establishment):
                with self.assertRaises(BaseException) as raised:
                    with load_authenticated_cdg_package(
                        self.lock,
                        "_dg_engine_i0_failed_establishment",
                        verify_oracle_bindings=True,
                    ):
                        self.fail("injected oracle establishment failure must propagate")

            def leaves(error):
                if isinstance(error, BaseExceptionGroup):
                    return [leaf for child in error.exceptions for leaf in leaves(child)]
                return [error]

            rollback_errors = leaves(raised.exception)
            self.assertTrue(any(
                isinstance(error, RuntimeError) and "mid-establishment" in str(error)
                for error in rollback_errors
            ))
            self.assertTrue(any(
                "preserved and reported" in str(error) for error in rollback_errors
            ))
            self.assertTrue(failed)
            self.assertIs(sys.modules.get(foreign_name), foreign_module)
            self.assertFalse(any(
                name == "diffusers" or name.startswith("diffusers.")
                or name == "transformers" or name.startswith("transformers.")
                for name in sys.modules
            ))
            self.assertTrue(i0_evidence._identity_sequence_matches(sys.meta_path, baseline_meta_path))
            self.assertIsNone(i0_evidence._ORACLE_TRUST_SNAPSHOT)
            self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)

            with load_authenticated_cdg_package(
                self.lock,
                "_dg_engine_i0_retry_establishment",
                verify_oracle_bindings=True,
            ) as (_package, loader):
                loader.assert_oracle_bindings()
            self.assertIsNotNone(i0_evidence._ORACLE_TRUST_SNAPSHOT)
        finally:
            current_modules = [
                name for name in tuple(sys.modules)
                if name == "diffusers" or name.startswith("diffusers.")
                or name == "transformers" or name.startswith("transformers.")
            ]
            for name in current_modules:
                sys.modules.pop(name, None)
            sys.modules.update(original_modules)
            if sys.modules.get(foreign_name) is foreign_module:
                del sys.modules[foreign_name]
            i0_evidence._ORACLE_TRUST_SNAPSHOT = original_snapshot
            # A successful retry can create only authenticated six finders;
            # remove exact additions before restoring the old trust world.
            if not i0_evidence._identity_sequence_matches(sys.meta_path, baseline_meta_path):
                for finder in tuple(sys.meta_path):
                    if not any(finder is item for item in baseline_meta_path):
                        i0_evidence._remove_identity_from_meta_path(finder)
        self.assertTrue(i0_evidence._identity_sequence_matches(sys.meta_path, baseline_meta_path))
        original_snapshot.assert_integrity()

    def test_later_context_reuses_the_exact_process_trust_snapshot(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        package_name = "_dg_engine_i0_shared_oracle_snapshot"
        with load_authenticated_cdg_package(
            self.lock, package_name, verify_oracle_bindings=True
        ) as (_package, loader):
            self.assertIs(i0_evidence._ORACLE_TRUST_SNAPSHOT, snapshot)
            for export in snapshot.exports:
                self.assertIs(loader.oracle_object(export), snapshot.object_for(export))
            loader.assert_oracle_bindings()
        self.assertIs(i0_evidence._ORACLE_TRUST_SNAPSHOT, snapshot)
        self.assert_namespace_absent(package_name)

    def test_each_monkeypatched_root_export_after_preload_blocks_authenticated_context(self):
        bindings = (
            ("diffusers", "diffusers.pipelines.diffusion_gemma.pipeline_diffusion_gemma", "DiffusionGemmaPipeline"),
            ("diffusers", "diffusers.schedulers.scheduling_entropy_bound", "EntropyBoundScheduler"),
            ("transformers", "transformers.models.auto.processing_auto", "AutoProcessor"),
            (
                "transformers",
                "transformers.models.diffusion_gemma.modeling_diffusion_gemma",
                "DiffusionGemmaForBlockDiffusion",
            ),
        )
        for index, (package_name, module_name, export) in enumerate(bindings):
            with self.subTest(export=export):
                exact_module = importlib.import_module(module_name)
                package_root = importlib.import_module(package_name)
                trusted = getattr(exact_module, export)
                self.assertIs(getattr(package_root, export), trusted)
                reserved = f"_dg_engine_i0_oracle_negative_{index}"
                before_import = __import__("builtins").__import__
                before_finders = tuple(sys.meta_path)
                with mock.patch.object(package_root, export, object()):
                    with self.assertRaisesRegex(EvidenceBlocked, "package-root export"):
                        with load_authenticated_cdg_package(
                            self.lock,
                            reserved,
                            verify_oracle_bindings=True,
                            acquisition_timeout=0.5,
                        ):
                            self.fail("monkeypatched oracle root export must not authenticate")
                self.assertIs(getattr(package_root, export), trusted)
                self.assertIs(__import__("builtins").__import__, before_import)
                self.assertEqual(tuple(sys.meta_path), before_finders)
                self.assert_namespace_absent(reserved)
                self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)

    def test_same_class_method_replacement_is_rejected_for_all_oracle_classes(self):
        cases = (
            ("DiffusionGemmaPipeline", "check_inputs"),
            ("EntropyBoundScheduler", "step"),
            ("AutoProcessor", "register"),
            ("DiffusionGemmaForBlockDiffusion", "forward"),
        )
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        for index, (export, method_name) in enumerate(cases):
            with self.subTest(export=export):
                trusted_class = snapshot.object_for(export)
                original = trusted_class.__dict__[method_name]

                def replacement(*_args, **_kwargs):
                    return "hostile replacement"

                setattr(trusted_class, method_name, replacement)
                try:
                    with self.assertRaisesRegex(EvidenceBlocked, "class contents changed"):
                        with load_authenticated_cdg_package(
                            self.lock,
                            f"_dg_engine_i0_class_method_mutation_{index}",
                            verify_oracle_bindings=True,
                        ):
                            self.fail("same-object class method mutation must not authenticate")
                finally:
                    setattr(trusted_class, method_name, original)
                snapshot.assert_integrity()
                self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)

    def test_in_place_mutable_class_attributes_are_rejected_where_present(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        cases = (
            ("DiffusionGemmaPipeline", "_callback_tensor_inputs", "append"),
            ("DiffusionGemmaForBlockDiffusion", "_tied_weights_keys", "dict"),
        )
        for index, (export, attribute, mutation) in enumerate(cases):
            with self.subTest(export=export):
                trusted_class = snapshot.object_for(export)
                mutable = trusted_class.__dict__[attribute]
                original = mutable.copy()
                if mutation == "append":
                    mutable.append("hostile-in-place-value")
                else:
                    mutable["hostile.in_place"] = "mutation"
                try:
                    with self.assertRaisesRegex(EvidenceBlocked, "class contents changed"):
                        with load_authenticated_cdg_package(
                            self.lock,
                            f"_dg_engine_i0_class_attribute_mutation_{index}",
                            verify_oracle_bindings=True,
                        ):
                            self.fail("in-place mutable class attribute mutation must not authenticate")
                finally:
                    mutable.clear()
                    mutable.update(original) if isinstance(mutable, dict) else mutable.extend(original)
                snapshot.assert_integrity()

    def test_late_module_global_binding_replacement_is_rejected_for_every_oracle_class(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        cases = (
            ("DiffusionGemmaPipeline", "torch"),
            ("EntropyBoundScheduler", "ConfigMixin"),
            ("AutoProcessor", "processor_class_from_name"),
            ("DiffusionGemmaForBlockDiffusion", "DiffusionGemmaModel"),
        )
        for export, global_name in cases:
            with self.subTest(export=export, global_name=global_name):
                namespace = self.oracle_global_namespace(snapshot, export, global_name)
                original = namespace[global_name]
                namespace[global_name] = object()
                try:
                    with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                        snapshot.assert_integrity()
                finally:
                    namespace[global_name] = original
                snapshot.assert_integrity()

    def test_late_in_place_module_dependency_mutation_is_rejected_for_every_oracle_class(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        cases = (
            ("DiffusionGemmaPipeline", "torch", "ones_like"),
            ("EntropyBoundScheduler", "inspect", "signature"),
            ("AutoProcessor", "json", "load"),
            ("DiffusionGemmaForBlockDiffusion", "nn", "Linear"),
        )
        for export, global_name, attribute_name in cases:
            with self.subTest(export=export, dependency=f"{global_name}.{attribute_name}"):
                namespace = self.oracle_global_namespace(snapshot, export, global_name)
                dependency = namespace[global_name]
                original = dependency.__dict__[attribute_name]
                dependency.__dict__[attribute_name] = object()
                try:
                    with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                        snapshot.assert_integrity()
                finally:
                    dependency.__dict__[attribute_name] = original
                snapshot.assert_integrity()

    def test_late_in_place_helper_and_class_dependency_mutations_are_rejected(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        auto_globals = self.oracle_global_namespace(snapshot, "AutoProcessor", "processor_class_from_name")
        helper = auto_globals["processor_class_from_name"]
        helper.__dict__["_dg_i0_hostile"] = object()
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            del helper.__dict__["_dg_i0_hostile"]
        snapshot.assert_integrity()

        # AutoProcessor resolves imported cached_file, which in turn resolves
        # the module helper cached_files and its own global dependency.  A late
        # replacement two helper levels away must remain inside the fingerprint.
        cached_file = self.oracle_global_namespace(snapshot, "AutoProcessor", "cached_file")["cached_file"]
        cached_files = cached_file.__globals__["cached_files"]
        nested_globals = cached_files.__globals__
        nested_original = nested_globals["hf_hub_download"]
        nested_globals["hf_hub_download"] = object()
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            nested_globals["hf_hub_download"] = nested_original
        snapshot.assert_integrity()

        cases = (
            ("EntropyBoundScheduler", "ConfigMixin", None),
            ("AutoProcessor", "AutoConfig", "from_pretrained"),
            ("DiffusionGemmaForBlockDiffusion", "DiffusionGemmaModel", "__init__"),
        )
        for export, global_name, selected_method in cases:
            with self.subTest(export=export, class_dependency=global_name):
                namespace = self.oracle_global_namespace(snapshot, export, global_name)
                dependency = namespace[global_name]
                if selected_method is None:
                    setattr(dependency, "_dg_i0_hostile", object())
                    restore = lambda: delattr(dependency, "_dg_i0_hostile")
                else:
                    owner = next(item for item in dependency.__mro__ if selected_method in item.__dict__)
                    original = owner.__dict__[selected_method]
                    setattr(owner, selected_method, lambda *_args, **_kwargs: None)
                    restore = lambda: setattr(owner, selected_method, original)
                try:
                    if selected_method is None:
                        # Unselected mutable/base-class cache state is outside
                        # the finite behavior manifest and must not false-block.
                        snapshot.assert_integrity()
                    else:
                        with self.assertRaisesRegex(EvidenceBlocked, "class contents changed|dependency graph changed"):
                            snapshot.assert_integrity()
                finally:
                    restore()
                snapshot.assert_integrity()

    def test_closure_global_inherited_method_registry_and_full_attribute_chain_mutations_are_rejected(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT

        pipeline_globals = self.closure_global_namespace(snapshot, "DiffusionGemmaPipeline", "DynamicCache")
        original_dynamic_cache = pipeline_globals["DynamicCache"]
        pipeline_globals["DynamicCache"] = object()
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            pipeline_globals["DynamicCache"] = original_dynamic_cache
        snapshot.assert_integrity()

        pipeline_class = snapshot.object_for("DiffusionGemmaPipeline")
        register_owner = next(owner for owner in pipeline_class.__mro__ if "register_modules" in owner.__dict__)
        original_register_modules = register_owner.__dict__["register_modules"]
        register_owner.register_modules = lambda *_args, **_kwargs: None
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            register_owner.register_modules = original_register_modules
        snapshot.assert_integrity()

        inherited_code = original_register_modules.__code__
        inherited_globals = original_register_modules.__globals__
        inherited_helper = inherited_globals["_fetch_class_library_tuple"]
        inherited_globals["_fetch_class_library_tuple"] = object()
        try:
            self.assertIs(original_register_modules.__code__, inherited_code)
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            inherited_globals["_fetch_class_library_tuple"] = inherited_helper
        snapshot.assert_integrity()

        auto_globals = self.oracle_global_namespace(snapshot, "AutoProcessor", "processor_class_from_name")
        mapping = auto_globals["PROCESSOR_MAPPING"]
        extra_content = mapping._extra_content
        marker = "_dg_i0_hostile_processor"
        self.assertNotIn(marker, extra_content)
        extra_content[marker] = object()
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            del extra_content[marker]
        snapshot.assert_integrity()

        torch_module = pipeline_globals["torch"]
        original_pad = torch_module.nn.functional.pad
        torch_module.nn.functional.pad = object()
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            torch_module.nn.functional.pad = original_pad
        snapshot.assert_integrity()

    def test_oracle_dependency_traversal_limits_are_deterministic_cycle_safe_and_fail_closed(self):
        self.assertEqual(i0_evidence.MAX_ORACLE_DEPENDENCY_DEPTH, 96)
        self.assertEqual(i0_evidence.MAX_ORACLE_DEPENDENCY_NODES, 100_000)
        self.assertEqual(i0_evidence.MAX_ORACLE_SINGLE_CONTAINER_ITEMS, 20_000)
        self.assertEqual(i0_evidence.MAX_ORACLE_CONTAINER_ITEMS, 200_000)
        self.assertEqual(i0_evidence.MAX_ORACLE_ENCODED_BYTES, 32 * 1024 * 1024)
        self.assertEqual(i0_evidence.MAX_ORACLE_SINGLE_CODE_BYTES, 1024 * 1024)
        self.assertEqual(i0_evidence.MAX_ORACLE_BYTECODE_STATES, 100_000)
        self.assertEqual(i0_evidence.MAX_ORACLE_FINGERPRINT_STABILIZATION_PASSES, 4)
        self.assertEqual(i0_evidence.ORACLE_DEEP_MUTABLE_DEPENDENCIES, frozenset({
            (
                "transformers.models.auto.processing_auto",
                "PROCESSOR_MAPPING",
                ("_extra_content",),
            ),
            (
                "transformers.models.auto.processing_auto",
                "PROCESSOR_MAPPING_NAMES",
                (),
            ),
        }))
        self.assertEqual(i0_evidence.ORACLE_CONSTRUCTOR_DEPENDENCIES, frozenset({
            ("transformers.cache_utils", "DynamicCache"),
            ("transformers.cache_utils", "StaticCache"),
            (
                "transformers.models.diffusion_gemma.modeling_diffusion_gemma",
                "DiffusionGemmaModel",
            ),
            (
                "diffusers.schedulers.scheduling_entropy_bound",
                "EntropyBoundSchedulerOutput",
            ),
        }))

        cyclic = []
        cursor = cyclic
        for _ in range(40):
            child = []
            cursor.append(child)
            cursor = child
        cursor.append(cyclic)
        digests = []
        for _ in range(2):
            encoder = i0_evidence._ReferencedGlobalIntegrityEncoder({}, "test", artifact_id="deep-cycle")
            encoder.add_binding(cyclic)
            digests.append(encoder.digest.hexdigest())
        self.assertEqual(digests[0], digests[1])

        nested = None
        for _ in range(8):
            nested = [nested]
        depth_budget = i0_evidence._OracleTraversalBudget(max_depth=4)
        depth_encoder = i0_evidence._ReferencedGlobalIntegrityEncoder(
            {}, "test", depth_budget, "depth-budget"
        )
        with self.assertRaisesRegex(EvidenceBlocked, "depth=4"):
            depth_encoder.add_binding(nested)

        node_budget = i0_evidence._OracleTraversalBudget(max_nodes=3)
        node_encoder = i0_evidence._ReferencedGlobalIntegrityEncoder({}, "test", node_budget, "node-budget")
        with self.assertRaisesRegex(EvidenceBlocked, "node-count=3"):
            node_encoder.add_binding([1, 2, 3, 4])

        item_budget = i0_evidence._OracleTraversalBudget(max_container_items=2)
        item_encoder = i0_evidence._ReferencedGlobalIntegrityEncoder({}, "test", item_budget, "item-budget")
        with self.assertRaisesRegex(EvidenceBlocked, "container-items=2"):
            item_encoder.add_binding([1, 2, 3])

        byte_budget = i0_evidence._OracleTraversalBudget(max_encoded_bytes=16)
        byte_encoder = i0_evidence._ReferencedGlobalIntegrityEncoder({}, "test", byte_budget, "byte-budget")
        with self.assertRaisesRegex(EvidenceBlocked, "encoded-bytes=16"):
            byte_encoder.add_binding("a long encoded value")

        oversized = [None] * (i0_evidence.MAX_ORACLE_SINGLE_CONTAINER_ITEMS + 1)
        size_encoder = i0_evidence._ReferencedGlobalIntegrityEncoder({}, "test", artifact_id="single-container")
        with self.assertRaisesRegex(EvidenceBlocked, "single-container-items"):
            size_encoder.add_binding(oversized)

    def test_dataflow_covers_load_deref_builtin_inherited_globals_and_unaliased_full_chains(self):
        def digest(function, *, receiver=None):
            encoder = i0_evidence._ReferencedGlobalIntegrityEncoder(
                {}, function.__module__, artifact_id=function.__qualname__
            )
            encoder.add_referenced_function(function, bound_receiver=receiver)
            return encoder.digest.hexdigest()

        class ClosureDependency:
            def __init__(self):
                self.state = 1

            def selected(self):
                return self.state

        dependency = ClosureDependency()

        def closure_factory(captured):
            def selected():
                alias = captured
                return alias.selected()
            return selected

        closure_function = closure_factory(dependency)
        before = digest(closure_function)
        dependency.state = 2
        self.assertNotEqual(before, digest(closure_function))
        dependency.state = 1

        builtin_name = "_dg_i0_builtin_fallback_probe"
        namespace = {"__builtins__": vars(__import__("builtins")), "__name__": "dg_i0_builtin_probe"}
        exec(f"def selected():\n    return {builtin_name}()\n", namespace)
        builtin_function = namespace["selected"]
        builtins_module = __import__("builtins")
        setattr(builtins_module, builtin_name, lambda: 1)
        try:
            before = digest(builtin_function)
            setattr(builtins_module, builtin_name, lambda: 2)
            self.assertNotEqual(before, digest(builtin_function))
        finally:
            delattr(builtins_module, builtin_name)

        chain_root = types.ModuleType("dg_i0_chain_root")
        chain_root.level_one = types.SimpleNamespace(
            level_two=types.SimpleNamespace(value=1)
        )
        chain_namespace = {"__builtins__": vars(builtins_module), "__name__": "dg_i0_chain", "chain_root": chain_root}
        exec("def selected():\n    return chain_root.level_one.level_two.value\n", chain_namespace)
        chain_function = chain_namespace["selected"]
        before = digest(chain_function)
        chain_root.level_one.level_two.value = 2
        self.assertNotEqual(before, digest(chain_function))

        inherited_namespace = {
            "__builtins__": vars(builtins_module),
            "__name__": "dg_i0_inherited",
            "HELPER": lambda: 1,
        }
        exec(
            "class Base:\n"
            "    def inherited(self):\n"
            "        return HELPER()\n"
            "class Child(Base):\n"
            "    pass\n",
            inherited_namespace,
        )
        child = inherited_namespace["Child"]
        inherited = inherited_namespace["Base"].__dict__["inherited"]
        unchanged_code = inherited.__code__
        before = digest(inherited, receiver=child)
        inherited_namespace["HELPER"] = lambda: 2
        self.assertIs(inherited.__code__, unchanged_code)
        self.assertNotEqual(before, digest(inherited, receiver=child))

    def test_processor_mapping_names_in_place_mutation_is_rejected(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        namespace = self.oracle_global_namespace(snapshot, "AutoProcessor", "processor_class_from_name")
        registry = namespace["processor_class_from_name"].__globals__["PROCESSOR_MAPPING_NAMES"]
        marker = "dg-i0-hostile-mapping-name"
        self.assertNotIn(marker, registry)
        registry[marker] = "HostileProcessor"
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "module-global dependency graph changed"):
                snapshot.assert_integrity()
        finally:
            del registry[marker]
        snapshot.assert_integrity()

    def test_realistic_large_values_classes_containers_and_code_fail_before_encoding_work(self):
        bytearray_value = bytearray(i0_evidence.MAX_ORACLE_ENCODED_BYTES + 1)
        encoder = i0_evidence._ClassIntegrityEncoder(artifact_id="large-bytearray")
        with self.assertRaisesRegex(EvidenceBlocked, "encoded-bytes"):
            encoder.add(bytearray_value)

        string_value = "x" * (i0_evidence.MAX_ORACLE_ENCODED_BYTES // 4 + 1)
        encoder = i0_evidence._ClassIntegrityEncoder(artifact_id="large-string")
        with self.assertRaisesRegex(EvidenceBlocked, "encoded-bytes"):
            encoder.add(string_value)

        large_class = type(
            "LargeClass",
            (),
            {f"entry_{index:05d}": None for index in range(i0_evidence.MAX_ORACLE_SINGLE_CONTAINER_ITEMS + 1)},
        )
        with self.assertRaisesRegex(EvidenceBlocked, "single-container-items"):
            i0_evidence._class_integrity_fingerprint(large_class)

        large_container = {index: None for index in range(i0_evidence.MAX_ORACLE_SINGLE_CONTAINER_ITEMS + 1)}
        encoder = i0_evidence._ClassIntegrityEncoder(artifact_id="large-container")
        with self.assertRaisesRegex(EvidenceBlocked, "single-container-items"):
            encoder.add(large_container)

        def tiny():
            return None
        oversized_code = tiny.__code__.replace(
            co_code=b"\x09\x00" * (i0_evidence.MAX_ORACLE_SINGLE_CODE_BYTES // 2 + 1)
        )
        encoder = i0_evidence._ClassIntegrityEncoder(artifact_id="large-code")
        with self.assertRaisesRegex(EvidenceBlocked, "single-code-bytes"):
            encoder.add(oversized_code)

    def test_cleanup_never_calls_foreign_finder_invalidate_caches(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        trusted = snapshot.object_for("EntropyBoundScheduler")

        class MutatingFinder:
            called = False

            def find_spec(self, _fullname, _path=None, _target=None):
                return None

            def invalidate_caches(self):
                self.called = True
                trusted._dg_i0_cleanup_mutation = object()

        finder = MutatingFinder()
        sys.meta_path.append(finder)
        package_name = "_dg_engine_i0_cleanup_finder"
        try:
            with load_authenticated_cdg_package(
                self.lock, package_name, verify_oracle_bindings=True
            ):
                pass
            self.assertFalse(finder.called)
            self.assertFalse(hasattr(trusted, "_dg_i0_cleanup_mutation"))
        finally:
            if hasattr(trusted, "_dg_i0_cleanup_mutation"):
                del trusted._dg_i0_cleanup_mutation
            sys.meta_path[:] = [item for item in sys.meta_path if item is not finder]
        self.assert_namespace_absent(package_name)
        snapshot.assert_integrity()

    def test_late_oracle_source_fifo_directory_symlink_and_oversized_replacements_fail_promptly(self):
        snapshot = i0_evidence._ORACLE_TRUST_SNAPSHOT
        source = snapshot.snapshots["DiffusionGemmaPipeline"]["origin"]
        backup = source.with_name(source.name + ".dg-i0-backup")
        self.assertFalse(backup.exists())
        for kind in ("fifo", "directory", "symlink", "oversized-regular"):
            with self.subTest(kind=kind):
                source.rename(backup)
                try:
                    if kind == "fifo":
                        os.mkfifo(source)
                    elif kind == "directory":
                        source.mkdir()
                    elif kind == "symlink":
                        source.symlink_to(backup)
                    else:
                        with source.open("wb") as replacement:
                            replacement.truncate(i0_evidence.MAX_ORACLE_SOURCE_BYTES + 1)
                    started = time.monotonic()
                    with self.assertRaises(EvidenceBlocked):
                        snapshot.assert_integrity()
                    self.assertLess(time.monotonic() - started, 1.0)
                finally:
                    if source.is_dir() and not source.is_symlink():
                        source.rmdir()
                    else:
                        source.unlink(missing_ok=True)
                    backup.rename(source)
                snapshot.assert_integrity()

    def test_post_context_oracle_integrity_check_detects_late_export_mutation_and_cleans(self):
        import diffusers

        package_name = "_dg_engine_i0_oracle_late_mutation"
        trusted = diffusers.EntropyBoundScheduler
        context = load_authenticated_cdg_package(
            self.lock, package_name, verify_oracle_bindings=True
        )
        patcher = mock.patch.object(diffusers, "EntropyBoundScheduler", object())
        try:
            with self.assertRaisesRegex(EvidenceBlocked, "changed after trust initialization"):
                with context as (_package, loader):
                    self.assertIs(loader.oracle_object("EntropyBoundScheduler"), trusted)
                    patcher.start()
        finally:
            patcher.stop()
        self.assertIs(diffusers.EntropyBoundScheduler, trusted)
        self.assert_namespace_absent(package_name)
        self.assertIsNone(context.loader)
        self.assertIsNone(context._oracle_guard)
        self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)


if __name__ == "__main__":
    unittest.main(verbosity=2)
