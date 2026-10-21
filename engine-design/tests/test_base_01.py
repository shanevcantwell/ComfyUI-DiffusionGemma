"""BASE-01: source-bound eight-node and complete /object_info freeze."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest

from i0_testlib import (
    ENGINE_DESIGN,
    FIXTURES,
    REPO_ROOT,
    SCHEMAS,
    derive_object_info,
    authenticated_worker_package,
    read_authenticated_cdg_source,
    read_evidence_artifact,
    read_json,
    repository_files_under,
    sha256,
    validate_json,
)

BASELINE = "fed377afc7b54f03cb7faa4dd798c80c15279d8a"
EXPECTED_KEYS = [
    "DGemmaLoader",
    "DGemmaSampler",
    "DGemmaTrace",
    "DGemmaTallyAudit",
    "DGemmaRunLogWriter",
    "DGemmaEncode",
    "DGemmaDenoise",
    "DGemmaTokenTrace",
]


def baseline_bytes(path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{BASELINE}:{path}"], cwd=REPO_ROOT)


class Base01InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inventory = read_json(ENGINE_DESIGN / "node-modes.json")
        cls.lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        cls.pack, cls.pack_loader = authenticated_worker_package()
        loader_module = sys.modules[cls.pack.NODE_CLASS_MAPPINGS["DGemmaLoader"].__module__]
        loader_module.list_local_model_dirs = lambda: ["<runtime-local-model-directory>"]
        for node_class in cls.pack.NODE_CLASS_MAPPINGS.values():
            node_class.RELATIVE_PYTHON_MODULE = "custom_nodes.ComfyUI-DiffusionGemma"
        cls.by_key = {node["mapping_key"]: node for node in cls.inventory["nodes"]}

    def test_all_new_json_schemas_and_records_validate(self):
        from jsonschema import Draft202012Validator, ValidationError

        records = [
            (ENGINE_DESIGN / "reference.lock.json", SCHEMAS / "reference-lock.schema.json"),
            (ENGINE_DESIGN / "i0-harness.lock.json", SCHEMAS / "harness-lock.schema.json"),
            (ENGINE_DESIGN / "node-modes.json", SCHEMAS / "node-modes.schema.json"),
            (FIXTURES / "semantics.json", SCHEMAS / "semantics.schema.json"),
            (FIXTURES / "baseline-defects.json", SCHEMAS / "baseline-defects.schema.json"),
        ]
        for schema_path in repository_files_under(SCHEMAS, ".schema.json"):
            Draft202012Validator.check_schema(read_json(schema_path))
        for record_path, schema_path in records:
            with self.subTest(record=record_path.name):
                validate_json(record_path, schema_path)
                schema = read_json(schema_path)
                record = read_json(record_path)
                missing = copy.deepcopy(record)
                del missing["kind"]
                with self.assertRaises(ValidationError):
                    Draft202012Validator(schema).validate(missing)
                wrong_type = copy.deepcopy(record)
                wrong_type["schema"] = "1"
                with self.assertRaises(ValidationError):
                    Draft202012Validator(schema).validate(wrong_type)
        root = __import__("os").environ["DG_ENGINE_I0_EVIDENCE_ROOT"]
        manifest_path = __import__("pathlib").Path(root) / "i0-evidence.json"
        manifest_schema = read_json(SCHEMAS / "evidence-manifest.schema.json")
        validate_json(manifest_path, SCHEMAS / "evidence-manifest.schema.json")
        manifest = read_json(manifest_path)
        del manifest["artifacts"][0]["sha256"]
        with self.assertRaises(ValidationError):
            Draft202012Validator(manifest_schema).validate(manifest)

    def test_schema_and_exact_registration(self):
        self.assertEqual(self.inventory["registered_node_count"], 8)
        self.assertEqual(list(self.pack.NODE_CLASS_MAPPINGS), EXPECTED_KEYS)
        self.assertEqual(list(self.pack.NODE_DISPLAY_NAME_MAPPINGS), EXPECTED_KEYS)
        self.assertEqual(list(self.by_key), EXPECTED_KEYS)
        self.assertEqual(len(self.by_key), 8)
        for key in EXPECTED_KEYS:
            self.assertEqual(self.pack.NODE_CLASS_MAPPINGS[key].__name__, self.by_key[key]["registered_class"])
            self.assertEqual(self.pack.NODE_DISPLAY_NAME_MAPPINGS[key], self.by_key[key]["display_name"])

    def test_baseline_and_checkout_node_sources_are_byte_identical(self):
        paths = ["__init__.py", *dict.fromkeys(node["source"] for node in self.inventory["nodes"])]
        for relative in paths:
            with self.subTest(path=relative):
                authenticated = read_authenticated_cdg_source(self.lock, relative)
                self.assertEqual(authenticated, baseline_bytes(relative))
                self.assertEqual((REPO_ROOT / relative).read_bytes(), authenticated)
        self.assertEqual(self.inventory["baseline_delta"]["status"], "NO_PRODUCT_SOURCE_DELTA")
        self.assertEqual(self.inventory["baseline_delta"]["changed_preexisting_paths"], [])

    def test_pinned_comfyui_projection_source_is_authenticated(self):
        server = read_evidence_artifact(self.lock, "comfy.server").decode("utf-8")
        nodes = read_evidence_artifact(self.lock, "comfy.nodes").decode("utf-8")
        for clause in (
            "info['input'] = obj_class.INPUT_TYPES()",
            "info['input_order'] = {key: list(value.keys())",
            "info['output_is_list']",
            "info['output_name']",
            "info['python_module'] = getattr(obj_class, \"RELATIVE_PYTHON_MODULE\", \"nodes\")",
            "info['has_intermediate_output']",
            "info['search_aliases']",
            "info['essentials_category']",
        ):
            self.assertIn(clause, server)
        self.assertIn('node_cls.RELATIVE_PYTHON_MODULE = "{}.{}".format(module_parent, get_module_name(module_path))', nodes)
        contract = self.inventory["object_info_contract"]
        self.assertEqual(contract["comfyui_revision"], "7cf4e7833519cbe9eff0e45767d01fd365d12a86")
        self.assertEqual(contract["source"]["server.py"]["sha256"], sha256(
            __import__("pathlib").Path(__import__("os").environ["DG_ENGINE_I0_EVIDENCE_ROOT"]) / "artifacts/comfy/server.py"
        ))

    def test_all_complete_object_info_projections_match_pinned_algorithm(self):
        required = set(self.inventory["object_info_contract"]["always_present"])
        allowed_conditional = set(self.inventory["object_info_contract"]["conditional"])
        for key in EXPECTED_KEYS:
            with self.subTest(node=key):
                node_class = self.pack.NODE_CLASS_MAPPINGS[key]
                actual = derive_object_info(key, node_class, self.pack.NODE_DISPLAY_NAME_MAPPINGS[key])
                expected = self.by_key[key]["object_info_projection"]
                self.assertEqual(actual, expected)
                self.assertEqual(set(expected) - allowed_conditional, required)
                self.assertEqual(expected["python_module"], "custom_nodes.ComfyUI-DiffusionGemma")
                self.assertIsInstance(expected["input"], dict)
                self.assertEqual(len(expected["output"]), len(expected["output_is_list"]))
                self.assertEqual(len(expected["output"]), len(expected["output_name"]))

    def test_every_input_choice_optional_hidden_socket_and_return_is_concrete(self):
        loader = self.by_key["DGemmaLoader"]["object_info_projection"]["input"]
        self.assertEqual(loader["required"]["quant"][0], ["none", "autoround"])
        self.assertEqual(loader["optional"]["local_model_dir"][0], ["<runtime-local-model-directory>"])
        sampler = self.by_key["DGemmaSampler"]["object_info_projection"]
        self.assertEqual(sampler["input"]["hidden"], {"unique_id": "UNIQUE_ID", "dgemma_live_view": "DGEMMA_LIVE_VIEW"})
        denoise = self.by_key["DGemmaDenoise"]["object_info_projection"]
        self.assertEqual(denoise["input"]["optional"]["kv_cache"][0], "DGEMMA_KV_CACHE")
        self.assertEqual(sampler["output_is_list"], [False, False, False, True, False, False])

    def test_mode_statuses_preserve_broken_and_fail_loud_distinctions(self):
        modes = {mode["name"]: mode for mode in self.by_key["DGemmaLoader"]["modes"]}
        self.assertTrue(modes["transformers-bf16"]["implemented"])
        self.assertEqual(modes["transformers-autoround-int4"]["implemented"], "load code exists")
        self.assertIn("known non-functional", modes["transformers-autoround-int4"]["status"])
        self.assertEqual(modes["transformers-autoround-int4"]["native_engine_distribution"], "excluded")
        self.assertIn("transformers:autoround", self.by_key["DGemmaLoader"]["backend_modes"])
        self.assertIn("autoround end-to-end", self.by_key["DGemmaLoader"]["capabilities"]["declared_but_broken"])
        denoise_modes = {mode["name"]: mode for mode in self.by_key["DGemmaDenoise"]["modes"]}
        self.assertFalse(denoise_modes["multi-block-composed"]["implemented"])
        self.assertIn("#263", denoise_modes["multi-block-composed"]["status"])

    def test_every_loaded_product_module_was_authenticated_before_execution(self):
        declared = {item["path"] for item in self.lock["cdg_source_identities"]}
        self.assertTrue(self.pack_loader.loaded_paths)
        self.assertEqual(self.pack_loader.loaded_paths - declared, set())
        guard = self.pack_loader._import_guard
        self.assertIsNotNone(guard)
        self.assertIn(guard, sys.meta_path)
        self.assertLess(sys.meta_path.index(guard), sys.meta_path.index(self.pack_loader))
        self.assertEqual(self.pack_loader.bare_product_import_requests, ())
        self.assertFalse(any(
            name.partition(".")[0] in {"dgemma", "surfaces"} for name in sys.modules
        ))
        self.pack_loader.assert_namespace_bound()
        prefix = self.pack.__name__ + "."
        for name, module in tuple(sys.modules.items()):
            if name == self.pack.__name__ or name.startswith(prefix):
                with self.subTest(module=name):
                    self.assertIs(module.__loader__, self.pack_loader)
                    self.assertTrue(module.__spec__.origin.startswith("authenticated-cdg://"))

    def test_inventory_source_hashes_are_current_bytes_and_baseline_blobs(self):
        for node in self.inventory["nodes"]:
            with self.subTest(node=node["mapping_key"]):
                self.assertEqual(sha256(REPO_ROOT / node["source"]), node["source_sha256"])
                blob = subprocess.check_output(
                    ["git", "rev-parse", f"{BASELINE}:{node['source']}"], cwd=REPO_ROOT, text=True
                ).strip()
                self.assertEqual(blob, node["baseline_git_blob_sha1"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
