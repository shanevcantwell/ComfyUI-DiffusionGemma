"""Focused negative tests for I0 schemas, evidence paths, and fail-closed scopes."""
from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
import unittest
from unittest import mock

from jsonschema import Draft202012Validator, ValidationError

import i0_evidence
from i0_testlib import (
    ENGINE_DESIGN,
    FIXTURES,
    SCHEMAS,
    EvidenceBlocked,
    GateVerification,
    gate_categories_complete,
    load_authenticated_cdg_package,
    read_evidence_artifact,
    read_json,
    scope_is_complete,
    verify_base02_gate,
    verify_gguf_derivation,
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def descriptor(path: str, data: bytes, descriptor_id: str) -> dict:
    return {"id": descriptor_id, "path": path, "size": len(data), "sha256": digest(data)}


def padded_evidence_identities(artifacts: list[dict]) -> list[dict]:
    rendered = copy.deepcopy(artifacts)
    existing = {entry["id"] for entry in rendered}
    index = 0
    while len(rendered) < 91:
        artifact_id = f"synthetic.padding.{index:03d}"
        index += 1
        if artifact_id in existing:
            continue
        existing.add(artifact_id)
        rendered.append({
            "id": artifact_id,
            "path": f"artifacts/tests/{artifact_id}",
            "size": 1,
            "sha256": "0" * 64,
            "status": "absent",
            "required_for": ["BASE-01-source"],
            "kind": "reference-source",
            "blocker": "TEST-PADDING-ABSENT",
        })
    return rendered


def write_manifest(root: Path, artifacts: list[dict], lock: dict | None = None) -> None:
    supplied = {entry["id"]: entry for entry in artifacts}
    rendered: list[dict] = []
    identities = (
        lock["evidence_root_contract"]["artifacts"]
        if lock is not None
        else padded_evidence_identities(artifacts)
    )
    for identity in identities:
        supplied_entry = supplied.get(identity["id"], {})
        entry = {
            field: copy.deepcopy(supplied_entry.get(field, identity[field]))
            for field in i0_evidence.EVIDENCE_DESCRIPTOR_FIELDS
            if field in supplied_entry or field in identity
        }
        entry.setdefault("path", f"artifacts/tests/{entry['id']}")
        entry.setdefault("size", 1)
        entry.setdefault("sha256", "0" * 64)
        entry.setdefault("required_for", ["BASE-01-source"])
        entry.setdefault("kind", "reference-source")
        if "blocker" in supplied_entry or "blocker" in identity:
            entry["blocker"] = supplied_entry.get("blocker", identity.get("blocker"))
        if "derivation" in supplied_entry:
            entry["derivation"] = copy.deepcopy(supplied_entry["derivation"])
        rendered.append(entry)
    (root / "i0-evidence.json").write_text(json.dumps({
        "$schema": "urn:comfyui-diffusiongemma:dg-engine:i0:evidence-manifest:v1",
        "schema": 1,
        "kind": "dg-engine-i0-evidence-manifest",
        "contract_artifact_set_sha256": (
            lock["evidence_root_contract"]["artifact_set_sha256"]
            if lock is not None
            else i0_evidence.I0_EVIDENCE_ARTIFACT_SET_SHA256
        ),
        "artifacts": rendered,
    }))


class EvidenceReaderNegativeTests(unittest.TestCase):
    def setUp(self):
        self.lock = read_json(ENGINE_DESIGN / "reference.lock.json")

    def synthetic_cdg_fixture(
        self, parent: Path, sources: dict[str, bytes]
    ) -> tuple[Path, Path, dict]:
        checkout = parent / "checkout"
        evidence = parent / "evidence"
        checkout.mkdir()
        evidence.mkdir()
        for relative, data in sources.items():
            path = checkout / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        subprocess.run(["git", "init", "-q"], cwd=checkout, check=True)
        subprocess.run(["git", "add", "."], cwd=checkout, check=True)
        subprocess.run(
            ["git", "-c", "user.name=I0 Test", "-c", "user.email=i0@example.invalid", "commit", "-qm", "fixture"],
            cwd=checkout,
            check=True,
        )
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        lock = copy.deepcopy(self.lock)
        lock["product_baseline"] = revision
        identities = []
        contracts = []
        manifest = []
        for index, (relative, data) in enumerate(sources.items()):
            artifact_id = f"synthetic.source.{index}"
            blob = subprocess.check_output(
                ["git", "rev-parse", f"{revision}:{relative}"], cwd=checkout, text=True
            ).strip()
            staged = f"authenticated/{index}.py"
            staged_path = evidence / staged
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.write_bytes(data)
            source_identity = {
                "repository": "https://example.invalid/authenticated-cdg-fixture",
                "revision": revision,
                "path": relative,
                "git_blob_sha1": blob,
            }
            identities.append({
                "artifact_id": artifact_id,
                **source_identity,
                "size": len(data),
                "sha256": digest(data),
            })
            contracts.append({
                "id": artifact_id,
                "path": staged,
                "size": len(data),
                "sha256": digest(data),
                "status": "present",
                "required_for": ["BASE-02-source"],
                "kind": "product-source",
                "source_identity": source_identity,
            })
            manifest.append({
                "id": artifact_id,
                "status": "present",
                "path": staged,
                "size": len(data),
                "sha256": digest(data),
                "required_for": ["BASE-02-source"],
                "kind": "product-source",
            })
        lock["cdg_source_identities"] = identities
        lock["evidence_root_contract"]["artifacts"] = padded_evidence_identities(contracts)
        write_manifest(evidence, manifest, lock)
        return checkout, evidence, lock

    @staticmethod
    def assert_namespace_absent(package_name: str) -> None:
        prefix = package_name + "."
        stale = [name for name in sys.modules if name == package_name or name.startswith(prefix)]
        if stale:
            raise AssertionError(f"stale synthetic modules: {stale}")

    def assert_context_owner_released(self) -> None:
        self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_THREAD)
        self.assertIsNone(i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN)
        self.assertTrue(i0_evidence._AUTHENTICATED_CONTEXT_LOCK.acquire(blocking=False))
        i0_evidence._AUTHENTICATED_CONTEXT_LOCK.release()

    def synthetic_lock(
        self,
        artifact_id: str,
        data: bytes,
        *,
        path: str = "artifact",
        status: str = "present",
        blocker: str | None = None,
    ) -> dict:
        lock = copy.deepcopy(self.lock)
        identity = {
            "id": artifact_id,
            "path": path,
            "size": len(data),
            "sha256": digest(data),
            "status": status,
            "required_for": ["BASE-01-source"],
            "kind": "reference-source",
        }
        if blocker is not None:
            identity["blocker"] = blocker
        lock["evidence_root_contract"]["artifacts"] = padded_evidence_identities([identity])
        return lock

    def test_unsafe_or_empty_path_parts_are_rejected_as_evidence_blocked(self):
        data = b"abc"
        unsafe_paths = (
            "", ".", "/absolute", "../escape", "dir/../escape", "dir\\escape",
            "dir//file", "./file", "dir/./file", "dir/",
        )
        for unsafe in unsafe_paths:
            with self.subTest(path=unsafe), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock = self.synthetic_lock("test.path", data, path=unsafe)
                write_manifest(root, [{
                    "id": "test.path", "status": "present", "path": unsafe,
                    "size": len(data), "sha256": digest(data),
                }])
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaises(EvidenceBlocked):
                        read_evidence_artifact(lock, "test.path")

    def test_path_schemas_reject_dot_and_empty_parts(self):
        manifest_schema = read_json(SCHEMAS / "evidence-manifest.schema.json")
        derivation_schema = read_json(SCHEMAS / "derivation-record.schema.json")
        relative_schemas = (manifest_schema["$defs"]["relativePath"], derivation_schema["$defs"]["relativePath"])
        for schema in relative_schemas:
            for unsafe in ("", ".", "./x", "x/./y", "x//y", "x/"):
                with self.subTest(path=unsafe):
                    with self.assertRaises(ValidationError):
                        Draft202012Validator(schema).validate(unsafe)

    def test_symlink_artifact_and_symlink_parent_are_rejected(self):
        for parent_symlink in (False, True):
            with self.subTest(parent_symlink=parent_symlink), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                outside = root.parent / f"outside-{root.name}"
                outside.write_bytes(b"abc")
                try:
                    if parent_symlink:
                        real = root / "real"
                        real.mkdir()
                        (real / "file").write_bytes(b"abc")
                        (root / "linkdir").symlink_to(real, target_is_directory=True)
                        relative = "linkdir/file"
                    else:
                        (root / "link").symlink_to(outside)
                        relative = "link"
                    lock = self.synthetic_lock("test.symlink", b"abc", path=relative)
                    write_manifest(root, [{
                        "id": "test.symlink", "status": "present", "path": relative,
                        "size": 3, "sha256": digest(b"abc"),
                    }])
                    with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                        with self.assertRaisesRegex(EvidenceBlocked, "safe open failed"):
                            read_evidence_artifact(lock, "test.symlink")
                finally:
                    outside.unlink(missing_ok=True)

    def test_fifo_and_directory_are_promptly_rejected_before_read(self):
        for kind in ("fifo", "directory"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                special = root / "special"
                if kind == "fifo":
                    os.mkfifo(special)
                else:
                    special.mkdir()
                lock = self.synthetic_lock("test.special", b"x", path="special")
                write_manifest(root, [{
                    "id": "test.special", "status": "present", "path": "special",
                    "size": 1, "sha256": digest(b"x"),
                }])
                started = time.monotonic()
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaisesRegex(EvidenceBlocked, "not a regular file"):
                        read_evidence_artifact(lock, "test.special")
                self.assertLess(time.monotonic() - started, 1.0)

    def test_stream_authentication_uses_bounded_reads_without_retaining_bytes(self):
        data = b"z" * (i0_evidence.STREAM_CHUNK_BYTES * 2 + 12345)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "multi-chunk").write_bytes(data)
            lock = self.synthetic_lock("test.stream", data, path="multi-chunk")
            write_manifest(root, [{
                "id": "test.stream", "status": "present", "path": "multi-chunk",
                "size": len(data), "sha256": digest(data),
            }])
            real_read = os.read
            requests: list[int] = []

            def tracked_read(fd: int, count: int) -> bytes:
                requests.append(count)
                return real_read(fd, count)

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}), mock.patch(
                "i0_evidence.os.read", side_effect=tracked_read
            ):
                with i0_evidence._evidence_state(lock) as state:
                    authenticated = i0_evidence._authenticate_evidence_artifact(
                        state, "test.stream", capture_limit=None
                    )
            self.assertIsNone(authenticated.data)
            self.assertEqual(authenticated.size, len(data))
            self.assertEqual(authenticated.sha256, digest(data))
            self.assertGreaterEqual(len(requests), 5)  # manifest + three data reads + EOF
            self.assertTrue(all(size <= i0_evidence.STREAM_CHUNK_BYTES for size in requests))

    def test_byte_return_limit_is_enforced_before_sparse_artifact_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            size = i0_evidence.MAX_RETURNED_ARTIFACT_BYTES + 1
            (root / "sparse").touch()
            os.truncate(root / "sparse", size)
            fake_hash = "0" * 64
            lock = copy.deepcopy(self.lock)
            lock["evidence_root_contract"]["artifacts"] = padded_evidence_identities([{
                "id": "test.sparse", "path": "sparse", "size": size, "sha256": fake_hash,
                "status": "present", "required_for": ["BASE-01-source"], "kind": "reference-source",
            }])
            write_manifest(root, [{
                "id": "test.sparse", "status": "present", "path": "sparse",
                "size": size, "sha256": fake_hash,
            }])
            real_read = os.read
            target_reads = 0

            def tracked_read(fd: int, count: int) -> bytes:
                nonlocal target_reads
                if os.fstat(fd).st_size == size:
                    target_reads += 1
                return real_read(fd, count)

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}), mock.patch(
                "i0_evidence.os.read", side_effect=tracked_read
            ):
                with self.assertRaisesRegex(EvidenceBlocked, "bounded byte-read limit"):
                    read_evidence_artifact(lock, "test.sparse")
            self.assertEqual(target_reads, 0)

    def test_duplicate_manifest_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.synthetic_lock("duplicate", b"x", path="artifacts/tests/duplicate", status="absent", blocker="TEST-BLOCK")
            entry = {"id": "duplicate", "status": "absent", "blocker": "TEST-BLOCK"}
            write_manifest(root, [entry, copy.deepcopy(entry)])
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                with self.assertRaisesRegex(EvidenceBlocked, "duplicate artifact id|non-unique"):
                    read_evidence_artifact(lock, "duplicate")

    def test_duplicate_manifest_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = self.synthetic_lock("duplicate.key", b"x", path="artifacts/tests/duplicate.key", status="absent", blocker="TEST-BLOCK")
            manifest = {
                "$schema": "urn:comfyui-diffusiongemma:dg-engine:i0:evidence-manifest:v1",
                "schema": 1,
                "kind": "dg-engine-i0-evidence-manifest",
                "artifacts": [{"id": "duplicate.key", "status": "absent", "blocker": "TEST-BLOCK"}],
            }
            encoded = json.dumps(manifest, separators=(",", ":")).encode()
            (root / "i0-evidence.json").write_bytes(b'{"schema":1,' + encoded[1:])
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                with self.assertRaisesRegex(EvidenceBlocked, "duplicate JSON key"):
                    read_evidence_artifact(lock, "duplicate.key")

    def test_wrong_manifest_hash_and_size_are_rejected(self):
        for field, value in (("size", 4), ("sha256", "0" * 64)):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "artifact").write_bytes(b"abc")
                lock = self.synthetic_lock("test.identity", b"abc", path="artifact")
                entry = {
                    "id": "test.identity", "status": "present", "path": "artifact",
                    "size": 3, "sha256": digest(b"abc"),
                }
                entry[field] = value
                write_manifest(root, [entry])
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaises(EvidenceBlocked):
                        read_evidence_artifact(lock, "test.identity")

    def test_secure_open_flags_are_mandatory(self):
        for flag in ("O_NOFOLLOW", "O_NONBLOCK"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (root / "artifact").write_bytes(b"abc")
                lock = self.synthetic_lock("test.flags", b"abc", path="artifact")
                write_manifest(root, [{
                    "id": "test.flags", "status": "present", "path": "artifact",
                    "size": 3, "sha256": digest(b"abc"),
                }])
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}), mock.patch.object(
                    os, flag, 0
                ):
                    with self.assertRaisesRegex(EvidenceBlocked, "required secure open flags"):
                        read_evidence_artifact(lock, "test.flags")

    def test_root_replacement_keeps_anchored_reads_but_reports_blocked(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "evidence"
            root.mkdir()
            (root / "artifact").write_bytes(b"trusted")
            lock = self.synthetic_lock("test.root", b"trusted", path="artifact")
            write_manifest(root, [{
                "id": "test.root", "status": "present", "path": "artifact",
                "size": 7, "sha256": digest(b"trusted"),
            }])
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                state = i0_evidence._evidence_state(lock)
                try:
                    moved = parent / "anchored"
                    root.rename(moved)
                    root.mkdir()
                    (root / "artifact").write_bytes(b"hostile")
                    authenticated = i0_evidence._read_evidence_artifact(state, "test.root")
                    self.assertEqual(authenticated, b"trusted")
                    with self.assertRaisesRegex(EvidenceBlocked, "root path was replaced"):
                        state.check_root_consistency()
                finally:
                    state.close()

    def test_intermediate_rename_uses_already_open_directory_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            nested = root / "nested"
            nested.mkdir()
            (nested / "artifact").write_bytes(b"trusted")
            lock = self.synthetic_lock("test.rename", b"trusted", path="nested/artifact")
            write_manifest(root, [{
                "id": "test.rename", "status": "present", "path": "nested/artifact",
                "size": 7, "sha256": digest(b"trusted"),
            }])
            real_open = os.open
            renamed = False

            def rename_after_open(path, flags, *args, **kwargs):
                nonlocal renamed
                fd = real_open(path, flags, *args, **kwargs)
                if path == "nested" and not renamed:
                    renamed = True
                    nested.rename(root / "moved")
                    nested.mkdir()
                    (nested / "artifact").write_bytes(b"hostile")
                return fd

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}), mock.patch(
                "i0_evidence.os.open", side_effect=rename_after_open
            ):
                self.assertEqual(read_evidence_artifact(lock, "test.rename"), b"trusted")
            self.assertTrue(renamed)

    def test_package_loader_executes_retained_relative_and_nested_bytes_after_live_rename(self):
        package_name = "_dg_engine_i0_toctou"
        sources = {
            "__init__.py": b"from .dgemma import VALUE\n",
            "dgemma/__init__.py": b"from .nested.module import VALUE\n",
            "dgemma/nested/__init__.py": b"# authenticated nested package\n",
            "dgemma/nested/module.py": b"VALUE = 'authenticated-retained-bytes'\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            real_authenticate = i0_evidence.authenticated_current_cdg_sources
            moved = checkout / "dgemma.authenticated-original"

            def replace_after_authentication(value):
                retained = real_authenticate(value)
                (checkout / "dgemma").rename(moved)
                hostile = checkout / "dgemma/nested"
                hostile.mkdir(parents=True)
                (checkout / "dgemma/__init__.py").write_text("from .nested.module import VALUE\n")
                (hostile / "__init__.py").write_text("# hostile package\n")
                (hostile / "module.py").write_text("VALUE = 'hostile-live-path'\n")
                return retained

            try:
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ), mock.patch.object(
                    i0_evidence, "authenticated_current_cdg_sources", side_effect=replace_after_authentication
                ):
                    with load_authenticated_cdg_package(lock, package_name) as (package, loader):
                        nested = loader.import_module("dgemma.nested.module")
                        self.assertEqual(package.VALUE, "authenticated-retained-bytes")
                        self.assertEqual(nested.VALUE, "authenticated-retained-bytes")
                        self.assertEqual(
                            loader.loaded_paths,
                            {"__init__.py", "dgemma/__init__.py", "dgemma/nested/__init__.py", "dgemma/nested/module.py"},
                        )
                        self.assertTrue(all(
                            module.__spec__.origin.startswith("authenticated-cdg://")
                            for module in loader.loaded_modules.values()
                        ))
            finally:
                if moved.exists():
                    shutil.rmtree(checkout / "dgemma", ignore_errors=True)
                    moved.rename(checkout / "dgemma")
            self.assertNotIn(loader, sys.meta_path)
            self.assert_namespace_absent(package_name)

    def test_package_loader_rejects_prepopulated_namespace_and_preserves_unrelated_modules(self):
        package_name = "_dg_engine_i0_preexisting"
        sources = {
            "__init__.py": b"from .dgemma import VALUE\n",
            "dgemma/__init__.py": b"VALUE = 'authenticated'\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            occupied = ModuleType(package_name + ".dgemma")
            unrelated = ModuleType("dgemma")
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                sys.modules[package_name + ".dgemma"] = occupied
                sys.modules["dgemma"] = unrelated
                try:
                    before_finders = tuple(sys.meta_path)
                    with self.assertRaisesRegex(EvidenceBlocked, "synthetic namespace is already occupied"):
                        with load_authenticated_cdg_package(lock, package_name):
                            self.fail("occupied namespace must not execute")
                    self.assertIs(sys.modules[package_name + ".dgemma"], occupied)
                    self.assertIs(sys.modules["dgemma"], unrelated)
                    self.assertEqual(tuple(sys.meta_path), before_finders)
                    del sys.modules[package_name + ".dgemma"]
                    with load_authenticated_cdg_package(lock, package_name) as (package, loader):
                        self.assertEqual(package.VALUE, "authenticated")
                        self.assertIsNot(loader.import_module("dgemma"), unrelated)
                    self.assertIs(sys.modules["dgemma"], unrelated)
                finally:
                    if sys.modules.get(package_name + ".dgemma") is occupied:
                        del sys.modules[package_name + ".dgemma"]
                    if sys.modules.get("dgemma") is unrelated:
                        del sys.modules["dgemma"]
            self.assert_namespace_absent(package_name)

    def test_retained_bare_product_import_is_blocked_with_preloaded_top_level_module(self):
        package_name = "_dg_engine_i0_bare_import"
        cases = (
            ("dgemma.preloaded", "builtins.__import__", "import dgemma.preloaded\n"),
            (
                "surfaces.preloaded",
                "meta_path",
                "import importlib\nVALUE = importlib.import_module('surfaces.preloaded')\n",
            ),
        )
        for requested, mechanism, source in cases:
            with self.subTest(requested=requested), tempfile.TemporaryDirectory() as temporary:
                sources = {"__init__.py": source.encode()}
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                root_name = requested.partition(".")[0]
                preloaded_root = ModuleType(root_name)
                preloaded_child = ModuleType(requested)
                sys.modules[root_name] = preloaded_root
                sys.modules[requested] = preloaded_child
                context = load_authenticated_cdg_package(lock, package_name)
                try:
                    with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                        i0_evidence, "REPO_ROOT", checkout
                    ):
                        with self.assertRaisesRegex(EvidenceBlocked, "bare product import is forbidden"):
                            with context:
                                self.fail("retained bare import must not resolve a preloaded product module")
                    audited_name = requested if mechanism == "builtins.__import__" else requested.partition(".")[0]
                    self.assertEqual(context.closed_import_audit, ((audited_name, mechanism),))
                    self.assertIs(sys.modules[root_name], preloaded_root)
                    self.assertIs(sys.modules[requested], preloaded_child)
                    self.assertIsNone(context.loader)
                    self.assertIsNone(context.module)
                    self.assert_namespace_absent(package_name)
                finally:
                    if sys.modules.get(requested) is preloaded_child:
                        del sys.modules[requested]
                    if sys.modules.get(root_name) is preloaded_root:
                        del sys.modules[root_name]

    def test_package_loader_rejects_undeclared_module_without_source_or_bytecode_fallback(self):
        package_name = "_dg_engine_i0_missing_dependency"
        sources = {
            "__init__.py": b"from .dgemma import VALUE\n",
            "dgemma/__init__.py": b"from .undeclared import VALUE\n",
        }
        for fallback in ("source", "bytecode"):
            with self.subTest(fallback=fallback), tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                undeclared = checkout / "dgemma/undeclared.py"
                undeclared.write_text("VALUE = 'hostile-fallback'\n")
                if fallback == "bytecode":
                    py_compile.compile(str(undeclared), cfile=str(checkout / "dgemma/undeclared.pyc"), doraise=True)
                    undeclared.unlink()
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    finders_before = tuple(sys.meta_path)
                    context = load_authenticated_cdg_package(lock, package_name)
                    with self.assertRaisesRegex(ModuleNotFoundError, "not in the authenticated CDG source set"):
                        with context:
                            self.fail("undeclared module must not execute")
                    self.assertEqual(tuple(sys.meta_path), finders_before)
                    self.assertIsNone(context.loader)
                    self.assertIsNone(context.module)
                self.assert_namespace_absent(package_name)

    def test_foreign_synthetic_replacement_and_injection_are_unconditionally_removed(self):
        package_name = "_dg_engine_i0_foreign_cleanup"
        sources = {
            "__init__.py": b"from .dgemma import VALUE\n",
            "dgemma/__init__.py": b"VALUE = 'authenticated'\n",
        }
        unrelated_name = package_name + "_unrelated"
        unrelated = ModuleType(unrelated_name)
        sys.modules[unrelated_name] = unrelated
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    with context as (_package, loader):
                        foreign_root = ModuleType(package_name)
                        foreign_child = ModuleType(package_name + ".injected")
                        sys.modules[package_name] = foreign_root
                        sys.modules[package_name + ".injected"] = foreign_child
                        with self.assertRaisesRegex(EvidenceBlocked, "not bound"):
                            loader.assert_namespace_bound()
                    self.assert_namespace_absent(package_name)
                    self.assertIs(sys.modules[unrelated_name], unrelated)
                    self.assertEqual(loader.loaded_modules, {})
                    self.assertEqual(loader.loaded_paths, set())
                    self.assertEqual(loader._source_by_module, {})
                    self.assertIsNone(context.loader)
                    self.assertIsNone(context.module)
        finally:
            if sys.modules.get(unrelated_name) is unrelated:
                del sys.modules[unrelated_name]

    def test_package_loader_repeated_success_and_body_exception_leave_no_residue(self):
        package_name = "_dg_engine_i0_cleanup"
        sources = {
            "__init__.py": b"from .dgemma import VALUE\n",
            "dgemma/__init__.py": b"VALUE = 'authenticated'\n",
        }
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                baseline_finders = tuple(sys.meta_path)
                baseline_import = builtins.__import__
                for _ in range(3):
                    context = load_authenticated_cdg_package(lock, package_name)
                    with context as (package, loader):
                        self.assertEqual(package.VALUE, "authenticated")
                        self.assertIn(loader, sys.meta_path)
                        self.assertEqual(loader.bare_product_import_requests, ())
                    self.assertEqual(tuple(sys.meta_path), baseline_finders)
                    self.assertIs(builtins.__import__, baseline_import)
                    self.assert_namespace_absent(package_name)
                    self.assertEqual(loader.loaded_modules, {})
                    self.assertEqual(loader.loaded_paths, set())
                    self.assertEqual(loader._source_by_module, {})
                    self.assertIsNone(context.loader)
                    self.assertIsNone(context.module)
                exception_context = load_authenticated_cdg_package(lock, package_name)
                with self.assertRaisesRegex(RuntimeError, "injected test-body failure"):
                    with exception_context as (_package, loader):
                        raise RuntimeError("injected test-body failure")
                self.assertNotIn(loader, sys.meta_path)
                self.assertEqual(loader.loaded_modules, {})
                self.assertEqual(loader.loaded_paths, set())
                self.assertEqual(loader._source_by_module, {})
                self.assertIsNone(exception_context.loader)
                self.assertIsNone(exception_context.module)
                self.assertEqual(tuple(sys.meta_path), baseline_finders)
                self.assertIs(builtins.__import__, baseline_import)
                self.assert_namespace_absent(package_name)
                self.assert_context_owner_released()

    def test_authenticated_contexts_serialize_across_threads(self):
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            first_entered = threading.Event()
            release_first = threading.Event()
            second_started = threading.Event()
            second_entered = threading.Event()
            errors: list[BaseException] = []

            def first() -> None:
                try:
                    with load_authenticated_cdg_package(lock, "_dg_engine_i0_thread_a"):
                        first_entered.set()
                        if not release_first.wait(3):
                            raise RuntimeError("test did not release first context")
                except BaseException as error:  # pragma: no cover - asserted below
                    errors.append(error)

            def second() -> None:
                try:
                    second_started.set()
                    with load_authenticated_cdg_package(
                        lock, "_dg_engine_i0_thread_b", acquisition_timeout=2.0
                    ):
                        second_entered.set()
                except BaseException as error:  # pragma: no cover - asserted below
                    errors.append(error)

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                thread_a = threading.Thread(target=first)
                thread_b = threading.Thread(target=second)
                thread_a.start()
                self.assertTrue(first_entered.wait(2))
                thread_b.start()
                self.assertTrue(second_started.wait(1))
                self.assertFalse(second_entered.wait(0.1))
                release_first.set()
                self.assertTrue(second_entered.wait(2))
                thread_a.join(2)
                thread_b.join(2)
            self.assertFalse(thread_a.is_alive() or thread_b.is_alive())
            self.assertEqual(errors, [])
            self.assert_namespace_absent("_dg_engine_i0_thread_a")
            self.assert_namespace_absent("_dg_engine_i0_thread_b")
            self.assert_context_owner_released()

    def test_authenticated_context_overlap_times_out_and_same_thread_nesting_fails_promptly(self):
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            held = threading.Event()
            release = threading.Event()
            holder_errors: list[BaseException] = []

            def holder() -> None:
                try:
                    with load_authenticated_cdg_package(lock, "_dg_engine_i0_timeout_holder"):
                        held.set()
                        release.wait(3)
                except BaseException as error:  # pragma: no cover - asserted below
                    holder_errors.append(error)

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                thread = threading.Thread(target=holder)
                thread.start()
                self.assertTrue(held.wait(2))
                started = time.monotonic()
                with self.assertRaisesRegex(EvidenceBlocked, "remained active"):
                    with load_authenticated_cdg_package(
                        lock, "_dg_engine_i0_timeout_waiter", acquisition_timeout=0.05
                    ):
                        self.fail("overlapping context must not enter")
                self.assertLess(time.monotonic() - started, 0.5)
                release.set()
                thread.join(2)
                self.assertEqual(holder_errors, [])

                with load_authenticated_cdg_package(lock, "_dg_engine_i0_outer"):
                    started = time.monotonic()
                    with self.assertRaisesRegex(EvidenceBlocked, "same-thread nested"):
                        with load_authenticated_cdg_package(
                            lock, "_dg_engine_i0_inner", acquisition_timeout=2.0
                        ):
                            self.fail("same-thread nested context must not enter")
                    self.assertLess(time.monotonic() - started, 0.5)
            self.assert_context_owner_released()

    def test_non_owner_close_cannot_mutate_or_release_and_third_thread_stays_blocked(self):
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            context = load_authenticated_cdg_package(lock, "_dg_engine_i0_owner_a")
            entered = threading.Event()
            try_close = threading.Event()
            close_rejected = threading.Event()
            release_owner = threading.Event()
            third_started = threading.Event()
            third_entered = threading.Event()
            errors: list[BaseException] = []
            owner_ids: list[int] = []

            def owner() -> None:
                try:
                    context.__enter__()
                    owner_ids.append(threading.get_ident())
                    entered.set()
                    if not release_owner.wait(3):
                        raise RuntimeError("owner was not released")
                    context.close()
                    context.close()  # same-owner double close is idempotent
                except BaseException as error:  # pragma: no cover - asserted below
                    errors.append(error)

            def intruder() -> None:
                try_close.wait(2)
                try:
                    context.close()
                except EvidenceBlocked as error:
                    if "only the thread" not in error.detail:
                        errors.append(error)
                    close_rejected.set()
                except BaseException as error:  # pragma: no cover - asserted below
                    errors.append(error)

            def third() -> None:
                third_started.set()
                try:
                    with load_authenticated_cdg_package(
                        lock, "_dg_engine_i0_owner_c", acquisition_timeout=2.0
                    ):
                        third_entered.set()
                except BaseException as error:  # pragma: no cover - asserted below
                    errors.append(error)

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                thread_a = threading.Thread(target=owner)
                thread_b = threading.Thread(target=intruder)
                thread_c = threading.Thread(target=third)
                thread_a.start()
                thread_b.start()
                self.assertTrue(entered.wait(2))
                self.assertEqual(context._owner_thread_id, owner_ids[0])
                owned_loader = context.loader
                owned_guard = context._guard
                try_close.set()
                self.assertTrue(close_rejected.wait(1))
                self.assertIs(context.loader, owned_loader)
                self.assertIn(owned_loader, sys.meta_path)
                self.assertIn(owned_guard, sys.meta_path)
                self.assertFalse(context._closed)
                thread_c.start()
                self.assertTrue(third_started.wait(1))
                self.assertFalse(third_entered.wait(0.1))
                release_owner.set()
                self.assertTrue(third_entered.wait(2))
                for thread in (thread_a, thread_b, thread_c):
                    thread.join(2)
                    self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assert_namespace_absent("_dg_engine_i0_owner_a")
            self.assert_namespace_absent("_dg_engine_i0_owner_c")
            self.assert_context_owner_released()

    def test_simultaneous_non_owner_and_owner_close_rejects_intruder_and_owner_double_close_is_safe(self):
        package_name = "_dg_engine_i0_simultaneous_close"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            context = load_authenticated_cdg_package(lock, package_name)
            start = threading.Event()
            attempted = threading.Event()
            intruder_errors: list[BaseException] = []

            def intruder() -> None:
                start.wait(2)
                try:
                    context.close()
                except BaseException as error:
                    intruder_errors.append(error)
                finally:
                    attempted.set()

            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                context.__enter__()
                thread = threading.Thread(target=intruder)
                thread.start()
                start.set()
                context.close()
                context.close()
                self.assertTrue(attempted.wait(1))
                thread.join(2)
            self.assertEqual(len(intruder_errors), 1)
            self.assertIsInstance(intruder_errors[0], EvidenceBlocked)
            self.assertIn("only the thread", str(intruder_errors[0]))
            self.assert_namespace_absent(package_name)
            self.assert_context_owner_released()

    def test_token_mismatch_never_releases_lock_and_owner_can_retry_after_repair(self):
        package_name = "_dg_engine_i0_token_mismatch"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            context = load_authenticated_cdg_package(lock, package_name)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                context.__enter__()
                loader = context.loader
                real_token = i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN
                i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN = object()
                try:
                    with self.assertRaisesRegex(EvidenceBlocked, "token no longer matches"):
                        context.close()
                    self.assertFalse(context._closed)
                    self.assertIs(context.loader, loader)
                    self.assertIn(loader, sys.meta_path)
                    self.assertFalse(i0_evidence._AUTHENTICATED_CONTEXT_LOCK.acquire(blocking=False))
                finally:
                    i0_evidence._AUTHENTICATED_CONTEXT_OWNER_TOKEN = real_token
                context.close()
            self.assert_namespace_absent(package_name)
            self.assert_context_owner_released()

    def test_body_and_cleanup_failures_are_both_preserved(self):
        package_name = "_dg_engine_i0_mixed_failure"
        bare_name = "dgemma.mixed_cleanup_conflict"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        foreign = ModuleType(bare_name)
        previous = sys.modules.pop(bare_name, None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    with self.assertRaises(ExceptionGroup) as raised:
                        with context:
                            sys.modules[bare_name] = foreign
                            raise RuntimeError("body-marker")
                rendered = repr(raised.exception)
                nested = list(raised.exception.exceptions)
                self.assertTrue(any(isinstance(item, RuntimeError) and str(item) == "body-marker" for item in nested))
                self.assertTrue(any("foreign bare product module" in str(item) for item in nested), rendered)
                self.assertIs(sys.modules.get(bare_name), foreign)
                self.assert_context_owner_released()
        finally:
            if sys.modules.get(bare_name) is foreign:
                del sys.modules[bare_name]
            if previous is not None:
                sys.modules[bare_name] = previous

    def test_foreign_meta_path_before_actual_loader_import_is_blocked_and_preserved(self):
        package_name = "_dg_engine_i0_meta_before_import"
        sources = {
            "__init__.py": b"VALUE = 'root'\n",
            "child.py": b"VALUE = 'child'\n",
        }

        class ForeignFinder:
            def find_spec(self, fullname, path=None, target=None):
                return None

        foreign = ForeignFinder()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    context.__enter__()
                    loader = context.loader
                    sys.meta_path.append(foreign)
                    with self.assertRaisesRegex(EvidenceBlocked, "before loader.import_module"):
                        loader.import_module("child")
                    with self.assertRaisesRegex(EvidenceBlocked, "sys.meta_path"):
                        context.close()
                self.assertIn(foreign, sys.meta_path)
                self.assertNotIn(loader, sys.meta_path)
                self.assert_namespace_absent(package_name)
                self.assert_context_owner_released()
        finally:
            while foreign in sys.meta_path:
                sys.meta_path.remove(foreign)

    def test_foreign_meta_path_mutation_during_actual_module_exec_blocks_and_leaves_no_stale_namespace(self):
        package_name = "_dg_engine_i0_meta_during_import"
        sources = {
            "__init__.py": b"VALUE = 'root'\n",
            "child.py": (
                b"import sys\n"
                b"class ForeignFinder:\n"
                b"    def find_spec(self, fullname, path=None, target=None): return None\n"
                b"FOREIGN = ForeignFinder()\n"
                b"sys.meta_path.append(FOREIGN)\n"
                b"VALUE = 'must-not-be-trusted'\n"
            ),
        }
        baseline = tuple(sys.meta_path)
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            context = load_authenticated_cdg_package(lock, package_name)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                context.__enter__()
                loader = context.loader
                with self.assertRaisesRegex(EvidenceBlocked, "after executing"):
                    loader.import_module("child")
                foreign = next(item for item in sys.meta_path if item not in baseline and item is not loader and item is not context._guard)
                with self.assertRaisesRegex(EvidenceBlocked, "sys.meta_path"):
                    context.close()
            self.assertIn(foreign, sys.meta_path)
            self.assert_namespace_absent(package_name)
            self.assert_context_owner_released()
            sys.meta_path.remove(foreign)

    def test_module_local_import_guard_avoids_global_mutation_and_foreign_hooks_block_cleanup(self):
        package_name = "_dg_engine_i0_hook_ownership"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        baseline_import = builtins.__import__

        class ForeignFinder:
            def find_spec(self, fullname, path=None, target=None):
                return None

        foreign_finder = ForeignFinder()
        foreign_import = None
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    with self.assertRaises(BaseException) as raised:
                        with context as (package, loader):
                            guard = loader._import_guard
                            captured_owned_hook = package.__dict__["__builtins__"]["__import__"]
                            self.assertIs(captured_owned_hook, loader._guarded_import)
                            self.assertIs(builtins.__import__, baseline_import)

                            def installed_import(name, globals=None, locals=None, fromlist=(), level=0):
                                return baseline_import(name, globals, locals, fromlist, level)

                            foreign_import = installed_import
                            builtins.__import__ = foreign_import
                            sys.meta_path.append(foreign_finder)
                    self.assertIn("cleanup", str(raised.exception).lower())
                    self.assertIs(builtins.__import__, foreign_import)
                    self.assertIn(foreign_finder, sys.meta_path)
                    self.assertIsNone(captured_owned_hook._guard)
                    self.assertFalse(guard.active)
                    self.assertNotIn(guard, sys.meta_path)
                    self.assertNotIn(loader, sys.meta_path)
                    self.assertTrue(any(
                        "builtins.__import__" in item for item in context.closed_cleanup_audit
                    ))
                    self.assertTrue(any("sys.meta_path" in item for item in context.closed_cleanup_audit))
                    self.assertIsNone(context.loader)
                    self.assertIsNone(context._guard)
        finally:
            builtins.__import__ = baseline_import
            while foreign_finder in sys.meta_path:
                sys.meta_path.remove(foreign_finder)
        self.assert_namespace_absent(package_name)
        self.assert_context_owner_released()

    def test_coordinated_insertions_at_restoration_boundaries_preserve_bare_and_remove_synthetic(self):
        package_name = "_dg_engine_i0_restoration_boundary"
        bare_name = "dgemma.restoration_boundary"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        foreign = ModuleType(bare_name)
        previous = sys.modules.pop(bare_name, None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    context.__enter__()
                    real_pop = context._pop_module
                    real_restore = context._restore_bare_modules
                    injected_synthetic = False

                    def pop_with_boundary_insertion(name: str):
                        nonlocal injected_synthetic
                        removed = real_pop(name)
                        if name == package_name and not injected_synthetic:
                            injected_synthetic = True
                            sys.modules[package_name + ".late"] = ModuleType(package_name + ".late")
                        return removed

                    def restore_with_foreign(failures):
                        sys.modules[bare_name] = foreign
                        real_restore(failures)

                    context._pop_module = pop_with_boundary_insertion
                    context._restore_bare_modules = restore_with_foreign
                    with self.assertRaisesRegex(EvidenceBlocked, "foreign bare product module"):
                        context.close()
                self.assertTrue(injected_synthetic)
                self.assertIs(sys.modules.get(bare_name), foreign)
                self.assert_namespace_absent(package_name)
                self.assert_context_owner_released()
        finally:
            if sys.modules.get(bare_name) is foreign:
                del sys.modules[bare_name]
            if previous is not None:
                sys.modules[bare_name] = previous

    def test_product_builtins_mutation_is_confined_to_safe_proxy(self):
        package_name = "_dg_engine_i0_builtins_during_import"
        sources = {
            "__init__.py": b"VALUE = 'root'\n",
            "child.py": (
                b"import builtins\n"
                b"ORIGINAL = builtins.__import__\n"
                b"def foreign_import(name, globals=None, locals=None, fromlist=(), level=0):\n"
                b"    return ORIGINAL(name, globals, locals, fromlist, level)\n"
                b"builtins.__import__ = foreign_import\n"
            ),
        }
        baseline_import = builtins.__import__
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    context.__enter__()
                    loader = context.loader
                    child = loader.import_module("child")
                    self.assertIs(builtins.__import__, baseline_import)
                    self.assertIsNot(child.builtins.__import__, baseline_import)
                    context.close()
                self.assertIs(builtins.__import__, baseline_import)
                self.assert_namespace_absent(package_name)
                self.assert_context_owner_released()
        finally:
            builtins.__import__ = baseline_import

    def test_concurrent_bare_module_insertion_is_preserved_and_blocks_cleanup_audit(self):
        package_name = "_dg_engine_i0_concurrent_bare"
        bare_name = "dgemma.concurrent_foreign"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        previous = sys.modules.pop(bare_name, None)
        foreign = ModuleType(bare_name)
        insert = threading.Event()
        inserted = threading.Event()

        def inserter() -> None:
            if insert.wait(2):
                sys.modules[bare_name] = foreign
                inserted.set()

        thread = threading.Thread(target=inserter)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    with self.assertRaisesRegex(EvidenceBlocked, "foreign bare product module"):
                        with context as (_package, loader):
                            guard = loader._import_guard
                            insert.set()
                            self.assertTrue(inserted.wait(1))
                thread.join(2)
                self.assertIs(sys.modules.get(bare_name), foreign)
                self.assertNotIn(loader, sys.meta_path)
                self.assertNotIn(guard, sys.meta_path)
                self.assertFalse(guard.active)
                self.assertEqual(loader.loaded_modules, {})
                self.assertEqual(loader._source_by_module, {})
                self.assertIsNone(context.loader)
                self.assertIsNone(context._guard)
                self.assertTrue(any(bare_name in item for item in context.closed_cleanup_audit))
                self.assert_namespace_absent(package_name)
                self.assert_context_owner_released()
        finally:
            insert.set()
            thread.join(2)
            if sys.modules.get(bare_name) is foreign:
                del sys.modules[bare_name]
            if previous is not None:
                sys.modules[bare_name] = previous

    @unittest.skipUnless(Path("/proc/self/fd").is_dir(), "requires Linux /proc descriptor inventory")
    def test_exception_paths_close_all_owned_descriptors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "nested").mkdir()
            (root / "nested/artifact").write_bytes(b"trusted")
            lock = self.synthetic_lock("test.close", b"trusted", path="nested/artifact")
            write_manifest(root, [{
                "id": "test.close", "status": "present", "path": "nested/artifact",
                "size": 7, "sha256": "0" * 64,
            }])
            before = len(list(Path("/proc/self/fd").iterdir()))
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                with self.assertRaises(EvidenceBlocked):
                    read_evidence_artifact(lock, "test.close")
            self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)

            # Restore a descriptor exactly bound to the synthetic lock before
            # exercising the lower-level close failure path.
            write_manifest(root, [{
                "id": "test.close", "status": "present", "path": "nested/artifact",
                "size": 7, "sha256": digest(b"trusted"),
            }], lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                state = i0_evidence._evidence_state(lock)
                try:
                    before = len(list(Path("/proc/self/fd").iterdir()))
                    real_close = os.close
                    raised = False

                    def raise_once(fd: int) -> None:
                        nonlocal raised
                        if not raised:
                            raised = True
                            raise OSError("injected old-directory close failure")
                        real_close(fd)

                    with mock.patch("i0_evidence.os.close", side_effect=raise_once):
                        with self.assertRaisesRegex(EvidenceBlocked, "safe open failed"):
                            i0_evidence._authenticate_confined_file(
                                state.root_fd,
                                "nested/artifact",
                                "test.close",
                                expected_size=7,
                                expected_sha256=digest(b"trusted"),
                                capture_limit=64,
                            )
                    self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), before)
                finally:
                    state.close()


    def test_oracle_retained_loader_executes_authenticated_bytes_after_path_replacement(self):
        package_name = "_dg_engine_i0_oracle_retained"
        module_name = package_name + ".target"
        with tempfile.TemporaryDirectory() as temporary:
            root_path = Path(temporary) / "__init__.py"
            target_path = Path(temporary) / "target.py"
            trusted_root = b"from .target import VALUE\n"
            trusted_target = b"VALUE = 'locked-authenticated-bytes'\n"
            root_path.write_bytes(trusted_root)
            target_path.write_bytes(trusted_target)
            retained_root = i0_evidence._read_authenticated_oracle_bytes(
                root_path, digest(trusted_root), package_name
            )
            retained_target = i0_evidence._read_authenticated_oracle_bytes(
                target_path, digest(trusted_target), module_name
            )
            # This is the critical interval: both pathname sources are replaced
            # after authentication but before the package-relative import.
            root_path.write_bytes(b"from .target import VALUE\nVALUE = 'hostile-root'\n")
            target_path.write_bytes(b"VALUE = 'hostile-replacement-bytes'\n")
            finder = i0_evidence._AuthenticatedOracleFinder({
                package_name: (root_path, retained_root, True),
                module_name: (target_path, retained_target, False),
            })
            sys.meta_path.insert(0, finder)
            try:
                package = __import__(package_name, fromlist=("VALUE",))
                module = sys.modules[module_name]
                self.assertEqual(package.VALUE, "locked-authenticated-bytes")
                self.assertEqual(module.VALUE, "locked-authenticated-bytes")
                self.assertIs(package.__spec__.loader, finder)
                self.assertIs(module.__spec__.loader, finder)
                self.assertIs(finder.executed[package_name], package)
                self.assertIs(finder.executed[module_name], module)
            finally:
                i0_evidence._remove_identity_from_meta_path(finder)
                sys.modules.pop(module_name, None)
                sys.modules.pop(package_name, None)
                finder.release()

    def test_meta_path_identity_checks_reject_equality_spoofs_without_calling_eq(self):
        package_name = "_dg_engine_i0_eq_spoof"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}

        class EqualitySpoof:
            def __init__(self, raises: bool):
                self.raises = raises
                self.eq_calls = 0

            def __eq__(self, _other):
                self.eq_calls += 1
                if self.raises:
                    raise RuntimeError("finder equality must never run")
                return True

            def find_spec(self, fullname, path=None, target=None):
                return None

        for raises in (False, True):
            with self.subTest(raises=raises), tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                spoof = EqualitySpoof(raises)
                context = load_authenticated_cdg_package(lock, package_name + str(int(raises)))
                try:
                    with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                        i0_evidence, "REPO_ROOT", checkout
                    ):
                        context.__enter__()
                        sys.meta_path.append(spoof)
                        with self.assertRaisesRegex(EvidenceBlocked, "sys.meta_path"):
                            context.close()
                    self.assertEqual(spoof.eq_calls, 0)
                    self.assert_context_owner_released()
                finally:
                    i0_evidence._remove_identity_from_meta_path(spoof)

    def test_fake_six_finder_is_preserved_and_blocked(self):
        package_name = "_dg_engine_i0_fake_six"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        FakeSix = type(
            "_SixMetaPathImporter",
            (),
            {
                "__module__": "six",
                "find_spec": lambda self, fullname, path=None, target=None: None,
            },
        )
        fake = FakeSix()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    context.__enter__()
                    sys.meta_path.append(fake)
                    with self.assertRaisesRegex(EvidenceBlocked, "sys.meta_path"):
                        context.close()
                self.assertTrue(any(item is fake for item in sys.meta_path))
                self.assert_context_owner_released()
        finally:
            i0_evidence._remove_identity_from_meta_path(fake)

    def test_foreign_invalidate_caches_callback_is_never_invoked_and_next_context_acquires(self):
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}

        class RaisingFinder:
            called = False

            def find_spec(self, fullname, path=None, target=None):
                return None

            def invalidate_caches(self):
                self.called = True
                raise RuntimeError("injected invalidate_caches failure")

        foreign = RaisingFinder()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    context = load_authenticated_cdg_package(lock, "_dg_engine_i0_raising_invalidate")
                    context.__enter__()
                    sys.meta_path.append(foreign)
                    with self.assertRaises(BaseException):
                        context.close()
                    self.assertFalse(foreign.called)
                    self.assertFalse(any(
                        "invalidate_caches" in item for item in context.closed_cleanup_audit
                    ))
                    self.assert_context_owner_released()
                    i0_evidence._remove_identity_from_meta_path(foreign)
                    with load_authenticated_cdg_package(lock, "_dg_engine_i0_after_raising_invalidate"):
                        pass
                    self.assert_context_owner_released()
        finally:
            i0_evidence._remove_identity_from_meta_path(foreign)

    def test_owned_synthetic_module_alias_is_removed_and_reported(self):
        package_name = "_dg_engine_i0_owned_alias"
        alias = "foreign_authenticated_alias"
        sources = {"__init__.py": b"VALUE = 'authenticated'\n"}
        previous = sys.modules.pop(alias, None)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
                context = load_authenticated_cdg_package(lock, package_name)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                    i0_evidence, "REPO_ROOT", checkout
                ):
                    with self.assertRaisesRegex(EvidenceBlocked, "alias"):
                        with context as (package, _loader):
                            sys.modules[alias] = package
                self.assertNotIn(alias, sys.modules)
                self.assert_namespace_absent(package_name)
                self.assert_context_owner_released()
        finally:
            sys.modules.pop(alias, None)
            if previous is not None:
                sys.modules[alias] = previous

    def test_product_import_builtins_gets_safe_proxy_and_cannot_cache_global_importer(self):
        package_name = "_dg_engine_i0_safe_builtins"
        sources = {"__init__.py": (
            b"import builtins\n"
            b"import sys\n"
            b"import types\n"
            b"CACHED = builtins.__import__\n"
            b"IS_GLOBAL = CACHED is sys.modules['builtins'].__import__\n"
            b"sys.modules['dgemma.transient_bypass'] = types.ModuleType('dgemma.transient_bypass')\n"
            b"try:\n"
            b"    CACHED('dgemma.transient_bypass')\n"
            b"except BaseException:\n"
            b"    BLOCKED = True\n"
            b"finally:\n"
            b"    sys.modules.pop('dgemma.transient_bypass', None)\n"
        )}
        with tempfile.TemporaryDirectory() as temporary:
            checkout, evidence, lock = self.synthetic_cdg_fixture(Path(temporary), sources)
            context = load_authenticated_cdg_package(lock, package_name)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(evidence)}), mock.patch.object(
                i0_evidence, "REPO_ROOT", checkout
            ):
                with self.assertRaisesRegex(EvidenceBlocked, "bare product"):
                    with context:
                        self.fail("cached importer must not bypass the authenticated import guard")
            self.assertIn(("dgemma.transient_bypass", "builtins.__import__"), context.closed_import_audit)
            self.assertNotIn("dgemma.transient_bypass", sys.modules)
            self.assert_context_owner_released()

    def test_oracle_binding_guard_requires_active_owner_token(self):
        with self.assertRaisesRegex(EvidenceBlocked, "owner token"):
            i0_evidence.OracleBindingGuard(self.lock, lambda _stage: None, object())


class DerivationSecurityTests(unittest.TestCase):
    GGUF_IDS = ("gguf.bf16", "gguf.q8_0", "gguf.q4_k_m")
    ROLES = ("output", "record", "command_log")

    def setUp(self):
        self.lock = read_json(ENGINE_DESIGN / "reference.lock.json")

    @staticmethod
    def identity(lock: dict, artifact_id: str) -> dict:
        return next(item for item in lock["evidence_root_contract"]["artifacts"] if item["id"] == artifact_id)

    @staticmethod
    def log_from_record(record: dict) -> dict:
        shared = {
            key: copy.deepcopy(record[key])
            for key in (
                "artifact_id", "derivation_id", "artifact_kind", "official_checkpoint_inputs",
                "additional_inputs", "converter", "conversion", "tool_binary", "quantization", "output",
            )
        }
        return {
            "$schema": "urn:comfyui-diffusiongemma:dg-engine:i0:command-log:v1",
            "schema": 1,
            "kind": "dg-engine-i0-command-log",
            **shared,
            "completion": {"status": "completed", "exit_code": 0},
        }

    @staticmethod
    def _anchor(file_descriptor: dict) -> dict:
        return {key: file_descriptor[key] for key in ("id", "path", "size", "sha256")}

    def pin_lineage(self, lock: dict, artifact_id: str, derivation_id: str, record_desc: dict, log_desc: dict) -> None:
        identity = self.identity(lock, artifact_id)
        identity["lineage_status"] = "BOUND"
        identity["lineage_lock"] = {
            "status": "BOUND",
            "derivation_id": derivation_id,
            "record": self._anchor(record_desc),
            "command_log": self._anchor(log_desc),
        }
        lock["gguf_lineage_policy"]["status"] = "BOUND"
        lock["gguf_remote_reference"]["lineage_status"] = "BOUND"
        remote = next(
            item for item in lock["gguf_remote_reference"]["artifacts"]
            if item["lfs_sha256"] == identity["sha256"]
        ) if any(
            item["lfs_sha256"] == identity["sha256"] for item in lock["gguf_remote_reference"]["artifacts"]
        ) else lock["gguf_remote_reference"]["artifacts"][("gguf.bf16", "gguf.q8_0", "gguf.q4_k_m").index(artifact_id)]
        remote["lineage_status"] = "BOUND"
        remote["lineage_claim"] = derivation_id

    def write_derivation(self, root: Path, lock: dict, artifact_id: str) -> tuple[dict, dict, dict]:
        suffix = artifact_id.removeprefix("gguf.")
        derivation_id = f"gguf.{suffix}.fixture-derivation"
        output = ("gguf-" + suffix).encode()
        identity = self.identity(lock, artifact_id)
        identity["size"] = len(output)
        identity["sha256"] = digest(output)
        remote = next(
            item for item in lock["gguf_remote_reference"]["artifacts"]
            if item["path"].endswith({
                "gguf.bf16": "BF16.gguf",
                "gguf.q8_0": "Q8_0.gguf",
                "gguf.q4_k_m": "Q4_K_M.gguf",
            }[artifact_id])
        )
        # Remote and gate identities are separate trust-root records.  Build
        # both explicitly, then require equality in the verifier.
        remote["size"] = len(bytes(output))
        remote["lfs_sha256"] = digest(bytes(output))
        checkpoint_inputs = [
            {"path": item["path"], "size": item["size"], "sha256": item["lfs_sha256"]}
            for item in lock["official_model"]["checkpoint_shards"]
        ]
        converter = lock["converter"]
        kind = identity["artifact_kind"]
        bf16_remote = next(
            item for item in lock["gguf_remote_reference"]["artifacts"] if item["path"].endswith("BF16.gguf")
        )
        bf16_gate = self.identity(lock, "gguf.bf16")
        additional = [] if kind == "GGUF_BF16" else [{
            "path": bf16_remote["path"], "size": bf16_gate["size"], "sha256": bf16_gate["sha256"],
        }]
        role = "converter-runtime" if kind == "GGUF_BF16" else "llama-quantize"
        llama = next(item for item in lock["source_baselines"] if item["id"] == "LLAMA")
        tool_repository = converter["repository"] if kind == "GGUF_BF16" else llama["repository"]
        tool_revision = converter["revision"] if kind == "GGUF_BF16" else llama["revision"]
        output_path = f"artifacts/{suffix}.gguf"
        identity["path"] = output_path
        identity["status"] = "present"
        input_paths = [item["path"] for item in checkpoint_inputs] if kind == "GGUF_BF16" else [additional[0]["path"]]
        command_argv = [
            "python3" if kind == "GGUF_BF16" else "llama-quantize",
            converter["path"] if kind == "GGUF_BF16" else input_paths[0],
            *([] if kind == "GGUF_BF16" else [output_path, kind.removeprefix("GGUF_")]),
            *(input_paths if kind == "GGUF_BF16" else []),
            *( ["--outfile", output_path, "--outtype", "bf16"] if kind == "GGUF_BF16" else []),
        ]
        configuration = {
            "mode": "checkpoint-to-gguf" if kind == "GGUF_BF16" else "gguf-quantization",
            "outtype": kind,
            "threads": 8,
        }
        tool = {
            "role": role,
            "repository": tool_repository,
            "revision": tool_revision,
            "source_tree_sha256": digest((tool_repository + tool_revision).encode()),
            "build": {
                "build_id": f"fixture-build-{suffix}",
                "compiler": "gcc",
                "compiler_version": "14.2.0",
                "build_type": "Release",
                "options": ["GGML_CUDA=OFF", "BUILD_SHARED_LIBS=OFF"],
            },
            "filename": "python3" if kind == "GGUF_BF16" else "llama-quantize",
            "size": 12345,
            "sha256": digest(("tool-" + suffix).encode()),
        }
        output_identity = {"id": artifact_id, "size": len(output), "sha256": digest(output)}
        record = {
            "$schema": "urn:comfyui-diffusiongemma:dg-engine:i0:derivation-record:v1",
            "schema": 1,
            "kind": "dg-engine-i0-gguf-derivation",
            "artifact_id": artifact_id,
            "derivation_id": derivation_id,
            "artifact_kind": kind,
            "official_checkpoint_inputs": checkpoint_inputs,
            "additional_inputs": additional,
            "converter": {key: converter[key] for key in ("repository", "revision", "path", "git_blob_sha1", "sha256")},
            "conversion": {"command_argv": command_argv, "configuration": configuration},
            "tool_binary": tool,
            "quantization": {"type": kind, "parameters": {"outtype": kind, "threads": 8}},
            "output": output_identity,
        }
        log = self.log_from_record(record)
        record_bytes = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        log_bytes = json.dumps(log, sort_keys=True, separators=(",", ":")).encode()
        record_desc = descriptor(f"records/{suffix}.json", record_bytes, f"{artifact_id}.derivation-record")
        log_desc = descriptor(f"logs/{suffix}.json", log_bytes, f"{artifact_id}.command-log")
        for relative, data in ((output_path, output), (record_desc["path"], record_bytes), (log_desc["path"], log_bytes)):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        entry = {
            "id": artifact_id,
            "status": "present",
            "path": output_path,
            "size": len(output),
            "sha256": digest(output),
            "derivation": {"record": record_desc, "command_log": log_desc},
        }
        self.pin_lineage(lock, artifact_id, derivation_id, record_desc, log_desc)
        return entry, record, log

    def rewrite_record(self, root: Path, lock: dict, entry: dict, record: dict, *, pin: bool = True) -> None:
        data = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
        old = entry["derivation"]["record"]
        updated = descriptor(old["path"], data, old["id"])
        (root / old["path"]).write_bytes(data)
        entry["derivation"]["record"] = updated
        if pin:
            self.identity(lock, entry["id"])["lineage_lock"]["record"] = self._anchor(updated)

    def rewrite_log(self, root: Path, lock: dict, entry: dict, log: dict, *, pin: bool = True) -> None:
        data = json.dumps(log, sort_keys=True, separators=(",", ":")).encode()
        old = entry["derivation"]["command_log"]
        updated = descriptor(old["path"], data, old["id"])
        (root / old["path"]).write_bytes(data)
        entry["derivation"]["command_log"] = updated
        if pin:
            self.identity(lock, entry["id"])["lineage_lock"]["command_log"] = self._anchor(updated)

    def write_raw_role(self, root: Path, lock: dict, entry: dict, role: str, data: bytes, *, pin: bool) -> None:
        old = entry["derivation"][role]
        updated = descriptor(old["path"], data, old["id"])
        (root / old["path"]).write_bytes(data)
        entry["derivation"][role] = updated
        if pin:
            self.identity(lock, entry["id"])["lineage_lock"][role] = self._anchor(updated)

    def three_derivations(self, root: Path) -> tuple[dict, list[dict], list[dict], list[dict]]:
        lock = copy.deepcopy(self.lock)
        entries, records, logs = [], [], []
        for artifact_id in self.GGUF_IDS:
            entry, record, log = self.write_derivation(root, lock, artifact_id)
            entries.append(entry)
            records.append(record)
            logs.append(log)
        return lock, entries, records, logs

    def test_trust_root_valid_fixture_pins_record_and_log_and_authenticates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, entries, _records, _logs = self.three_derivations(root)
            Draft202012Validator(read_json(SCHEMAS / "reference-lock.schema.json")).validate(lock)
            write_manifest(root, entries, lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                result = verify_base02_gate(lock, _freeze_contract=False)
            self.assertEqual(result.lineage_executed, 3)
            self.assertEqual(result.categories["gguf_authenticated_lineage"], 3)
            for entry in entries:
                identity = self.identity(lock, entry["id"])
                remote = next(
                    item for item in lock["gguf_remote_reference"]["artifacts"]
                    if item["lineage_claim"] == identity["lineage_lock"]["derivation_id"]
                )
                self.assertIsNot(identity, remote)
                self.assertEqual((remote["size"], remote["lfs_sha256"]), (identity["size"], identity["sha256"]))
                anchor = identity["lineage_lock"]
                self.assertEqual(anchor["record"], self._anchor(entry["derivation"]["record"]))
                self.assertEqual(anchor["command_log"], self._anchor(entry["derivation"]["command_log"]))

    def test_duplicate_derivation_ids_across_lineages_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, entries, _records, _logs = self.three_derivations(root)
            bf16_lineage = self.identity(lock, "gguf.bf16")["lineage_lock"]
            q8_lineage = self.identity(lock, "gguf.q8_0")["lineage_lock"]
            q8_lineage["derivation_id"] = bf16_lineage["derivation_id"]
            q8_remote = next(
                item for item in lock["gguf_remote_reference"]["artifacts"]
                if item["path"].endswith("Q8_0.gguf")
            )
            q8_remote["lineage_claim"] = bf16_lineage["derivation_id"]
            self.assertNotEqual(entries[0]["path"], entries[1]["path"])
            self.assertNotEqual(entries[0]["sha256"], entries[1]["sha256"])
            with self.assertRaises(ValidationError):
                Draft202012Validator(read_json(SCHEMAS / "reference-lock.schema.json")).validate(lock)
            write_manifest(root, entries, lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                result = verify_base02_gate(lock, _freeze_contract=False)
            self.assertEqual(result.lineage_executed, 0)
            self.assertTrue(any("derivation IDs must be globally unique" in item.detail for item in result.blockers))

    def test_schema_rejects_wrong_case_unicode_wrong_artifact_and_lineage_cardinality(self):
        reference_schema = Draft202012Validator(read_json(SCHEMAS / "reference-lock.schema.json"))
        record_schema = Draft202012Validator(read_json(SCHEMAS / "derivation-record.schema.json"))
        log_schema = Draft202012Validator(read_json(SCHEMAS / "command-log.schema.json"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid, _entries, records, logs = self.three_derivations(root)
            reference_schema.validate(valid)
            for bad_id in ("GGUF.BF16.fixture-derivation", "gguf.bf16.Ｆixture-derivation"):
                with self.subTest(form=bad_id):
                    changed = copy.deepcopy(valid)
                    lineage = self.identity(changed, "gguf.bf16")["lineage_lock"]
                    lineage["derivation_id"] = bad_id
                    with self.assertRaises(ValidationError):
                        reference_schema.validate(changed)
                    changed_record = copy.deepcopy(records[0])
                    changed_record["derivation_id"] = bad_id
                    with self.assertRaises(ValidationError):
                        record_schema.validate(changed_record)
                    changed_log = copy.deepcopy(logs[0])
                    changed_log["derivation_id"] = bad_id
                    with self.assertRaises(ValidationError):
                        log_schema.validate(changed_log)

            wrong_artifact = copy.deepcopy(valid)
            self.identity(wrong_artifact, "gguf.q8_0")["lineage_lock"]["derivation_id"] = (
                "gguf.bf16.other-derivation"
            )
            with self.assertRaises(ValidationError):
                reference_schema.validate(wrong_artifact)
            wrong_record = copy.deepcopy(records[1])
            wrong_record["derivation_id"] = "gguf.bf16.other-derivation"
            with self.assertRaises(ValidationError):
                record_schema.validate(wrong_record)
            wrong_log = copy.deepcopy(logs[1])
            wrong_log["derivation_id"] = "gguf.bf16.other-derivation"
            with self.assertRaises(ValidationError):
                log_schema.validate(wrong_log)

            omitted = copy.deepcopy(valid)
            omitted["evidence_root_contract"]["artifacts"] = [
                item for item in omitted["evidence_root_contract"]["artifacts"] if item["id"] != "gguf.q4_k_m"
            ]
            with self.assertRaises(ValidationError):
                reference_schema.validate(omitted)
            extra = copy.deepcopy(valid)
            duplicate_anchor = copy.deepcopy(self.identity(extra, "gguf.q4_k_m"))
            duplicate_anchor["size"] += 1
            extra["evidence_root_contract"]["artifacts"].append(duplicate_anchor)
            with self.assertRaises(ValidationError):
                reference_schema.validate(extra)

    def test_runtime_descriptor_id_registry_rejects_global_collision_directly(self):
        aliases = i0_evidence._DerivationAliases()
        blocker = "I0-BLOCK-GGUF-LINEAGE-BF16"
        aliases.claim_descriptor_id(
            "gguf.bf16.derivation-record", "gguf.bf16:record", blocker, "gguf.bf16"
        )
        with self.assertRaisesRegex(EvidenceBlocked, "descriptor IDs must be globally unique"):
            aliases.claim_descriptor_id(
                "gguf.bf16.derivation-record", "gguf.q8_0:record", blocker, "gguf.q8_0"
            )
        with self.assertRaisesRegex(EvidenceBlocked, "canonical lowercase ASCII"):
            aliases.claim_descriptor_id(
                "GGUF.BF16.DERIVATION-RECORD", "gguf.q8_0:record", blocker, "gguf.q8_0"
            )

    def _assert_descriptor_id_collision_rejected(
        self,
        source_artifact: int,
        source_role: str,
        target_artifact: int,
        target_role: str,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, entries, _records, _logs = self.three_derivations(root)
            source = entries[source_artifact]["derivation"][source_role]
            target = entries[target_artifact]["derivation"][target_role]
            self.assertNotEqual(source["path"], target["path"])
            self.assertNotEqual(source["sha256"], target["sha256"])
            target["id"] = source["id"]
            target_identity = self.identity(lock, entries[target_artifact]["id"])
            target_identity["lineage_lock"][target_role]["id"] = source["id"]
            reference_schema = read_json(SCHEMAS / "reference-lock.schema.json")
            with self.assertRaises(ValidationError):
                Draft202012Validator(reference_schema).validate(lock)
            write_manifest(root, entries, lock)
            manifest = read_json(root / "i0-evidence.json")
            with self.assertRaises(ValidationError):
                Draft202012Validator(read_json(SCHEMAS / "evidence-manifest.schema.json")).validate(manifest)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                result = verify_base02_gate(lock, _freeze_contract=False)
            self.assertEqual(result.lineage_executed, 0)
            self.assertTrue(any(
                "role-qualified" in item.detail or "globally unique" in item.detail
                for item in result.blockers
            ), result.blockers)

    def test_same_record_and_command_log_id_within_one_lineage_is_rejected(self):
        self._assert_descriptor_id_collision_rejected(0, "record", 0, "command_log")

    def test_cross_lineage_record_record_id_collision_is_rejected(self):
        self._assert_descriptor_id_collision_rejected(0, "record", 1, "record")

    def test_cross_lineage_log_log_id_collision_is_rejected(self):
        self._assert_descriptor_id_collision_rejected(0, "command_log", 1, "command_log")

    def test_cross_lineage_record_log_id_collision_is_rejected(self):
        self._assert_descriptor_id_collision_rejected(0, "record", 1, "command_log")

    def test_bound_policy_requires_exactly_three_bound_lineages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, entries, _records, _logs = self.three_derivations(root)
            q4 = self.identity(lock, "gguf.q4_k_m")
            q4["lineage_status"] = "UNBOUND"
            q4["lineage_lock"] = {
                "status": "UNBOUND", "derivation_id": None, "record": None, "command_log": None,
            }
            write_manifest(root, entries, lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                result = verify_base02_gate(lock, _freeze_contract=False)
            self.assertEqual(result.lineage_executed, 0)
            self.assertTrue(any("exactly three independently bound" in item.detail for item in result.blockers))

    def test_each_remote_gate_gguf_identity_mismatch_is_rejected(self):
        for artifact_id in self.GGUF_IDS:
            with self.subTest(artifact=artifact_id), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock, entries, _records, _logs = self.three_derivations(root)
                identity = self.identity(lock, artifact_id)
                remote = next(
                    item for item in lock["gguf_remote_reference"]["artifacts"]
                    if item["lineage_claim"] == identity["lineage_lock"]["derivation_id"]
                )
                remote["lfs_sha256"] = "f" * 64 if identity["sha256"] != "f" * 64 else "e" * 64
                write_manifest(root, entries, lock)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    result = verify_base02_gate(lock, _freeze_contract=False)
                self.assertTrue(
                    any(blocker.artifact_id == artifact_id and "remote GGUF size/SHA-256" in blocker.detail
                        for blocker in result.blockers),
                    result.blockers,
                )
                self.assertLess(result.lineage_executed, 3)

    def test_quantized_derivation_with_wrong_bf16_gate_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, entries, records, logs = self.three_derivations(root)
            q8_entry, q8_record, q8_log = entries[1], records[1], logs[1]
            q8_record["additional_inputs"][0]["sha256"] = "a" * 64
            q8_log = self.log_from_record(q8_record)
            self.rewrite_record(root, lock, q8_entry, q8_record, pin=True)
            self.rewrite_log(root, lock, q8_entry, q8_log, pin=True)
            write_manifest(root, entries, lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                result = verify_base02_gate(lock, _freeze_contract=False)
            self.assertTrue(
                any(blocker.artifact_id == "gguf.q8_0" and "authenticated BF16 gate identity" in blocker.detail
                    for blocker in result.blockers),
                result.blockers,
            )
            self.assertEqual(result.lineage_executed, 2)

    def test_quantized_lineages_cannot_survive_a_broken_bf16_derivation_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock, entries, records, _logs = self.three_derivations(root)
            bf16_entry, bf16_record = entries[0], records[0]
            bf16_record["official_checkpoint_inputs"][0]["sha256"] = "b" * 64
            bf16_log = self.log_from_record(bf16_record)
            self.rewrite_record(root, lock, bf16_entry, bf16_record, pin=True)
            self.rewrite_log(root, lock, bf16_entry, bf16_log, pin=True)
            write_manifest(root, entries, lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                result = verify_base02_gate(lock, _freeze_contract=False)
            self.assertEqual(result.lineage_executed, 0)
            self.assertEqual(result.categories["gguf_authenticated_lineage"], 0)
            self.assertTrue(any("official checkpoint/input" in blocker.detail for blocker in result.blockers))

    def test_current_unbound_lock_cannot_authenticate_even_valid_external_lineage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = copy.deepcopy(self.lock)
            entry, _record, _log = self.write_derivation(root, lock, "gguf.bf16")
            identity = self.identity(lock, "gguf.bf16")
            identity["lineage_status"] = "UNBOUND"
            identity["lineage_lock"] = {
                "status": "UNBOUND", "derivation_id": None, "record": None, "command_log": None,
            }
            write_manifest(root, [entry], lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                with self.assertRaisesRegex(EvidenceBlocked, "lineage is UNBOUND"):
                    verify_gguf_derivation(lock, "gguf.bf16")

    def test_every_nested_derivation_descriptor_field_is_bound_to_the_lineage_lock(self):
        for role in ("record", "command_log"):
            for field in ("id", "path", "size", "sha256", "extra"):
                with self.subTest(role=role, field=field), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    lock = copy.deepcopy(self.lock)
                    entry, _record, _log = self.write_derivation(root, lock, "gguf.bf16")
                    nested = entry["derivation"][role]
                    if field == "id":
                        nested[field] = nested[field] + ".forged"
                    elif field == "path":
                        nested[field] = "neutral/renamed.json"
                    elif field == "size":
                        nested[field] += 1
                    elif field == "sha256":
                        nested[field] = "f" * 64
                    else:
                        nested[field] = "forbidden"
                    write_manifest(root, [entry], lock)
                    manifest = read_json(root / "i0-evidence.json")
                    with self.assertRaisesRegex(EvidenceBlocked, "does not match the Git-controlled"):
                        i0_evidence._validate_manifest_against_lock(lock, manifest)

    def test_synchronized_manifest_record_and_log_tampering_fails_against_unchanged_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = copy.deepcopy(self.lock)
            entry, record, _log = self.write_derivation(root, lock, "gguf.q8_0")
            original_lock = copy.deepcopy(self.identity(lock, "gguf.q8_0")["lineage_lock"])
            record["tool_binary"]["build"]["build_id"] = "forged-build"
            record["conversion"]["command_argv"] = ["forged", "command"]
            log = self.log_from_record(record)
            self.rewrite_record(root, lock, entry, record, pin=False)
            self.rewrite_log(root, lock, entry, log, pin=False)
            self.assertEqual(self.identity(lock, "gguf.q8_0")["lineage_lock"], original_lock)
            write_manifest(root, [entry], lock)
            with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                with self.assertRaisesRegex(EvidenceBlocked, "does not match the Git-controlled"):
                    verify_gguf_derivation(lock, "gguf.q8_0")

    def test_every_command_log_binding_class_is_cross_checked(self):
        mutations = {
            "artifact_id": lambda value: value.__setitem__("artifact_id", "gguf.q4_k_m"),
            "derivation_id": lambda value: value.__setitem__("derivation_id", "other-derivation"),
            "artifact_kind": lambda value: value.__setitem__("artifact_kind", "GGUF_Q4_K_M"),
            "official_inputs": lambda value: value["official_checkpoint_inputs"][0].__setitem__("sha256", "a" * 64),
            "additional_inputs": lambda value: value["additional_inputs"][0].__setitem__("sha256", "b" * 64),
            "converter_repository": lambda value: value["converter"].__setitem__("repository", "https://example.com/forged"),
            "converter_revision": lambda value: value["converter"].__setitem__("revision", "c" * 40),
            "converter_path": lambda value: value["converter"].__setitem__("path", "forged/converter.py"),
            "converter_blob": lambda value: value["converter"].__setitem__("git_blob_sha1", "d" * 40),
            "converter_sha": lambda value: value["converter"].__setitem__("sha256", "e" * 64),
            "command_argv": lambda value: value["conversion"].__setitem__("command_argv", ["forged"]),
            "conversion_configuration": lambda value: value["conversion"]["configuration"].__setitem__("threads", 99),
            "tool_repository": lambda value: value["tool_binary"].__setitem__("repository", "https://example.com/tool"),
            "tool_revision": lambda value: value["tool_binary"].__setitem__("revision", "f" * 40),
            "tool_source_tree": lambda value: value["tool_binary"].__setitem__("source_tree_sha256", "1" * 64),
            "tool_binary_sha": lambda value: value["tool_binary"].__setitem__("sha256", "2" * 64),
            "tool_binary_size": lambda value: value["tool_binary"].__setitem__("size", 999),
            "tool_build_id": lambda value: value["tool_binary"]["build"].__setitem__("build_id", "forged"),
            "tool_build_options": lambda value: value["tool_binary"]["build"].__setitem__("options", ["FORGED=ON"]),
            "quantization_type": lambda value: value["quantization"].__setitem__("type", "GGUF_Q4_K_M"),
            "quantization_parameters": lambda value: value["quantization"]["parameters"].__setitem__("threads", 99),
            "output_id": lambda value: value["output"].__setitem__("id", "gguf.q4_k_m"),
            "output_size": lambda value: value["output"].__setitem__("size", 999),
            "output_sha": lambda value: value["output"].__setitem__("sha256", "3" * 64),
            "completion_status": lambda value: value["completion"].__setitem__("status", "failed"),
            "completion_exit": lambda value: value["completion"].__setitem__("exit_code", 1),
        }
        for name, mutate in mutations.items():
            with self.subTest(binding=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock, entries, _records, logs = self.three_derivations(root)
                entry, log = entries[1], logs[1]
                mutate(log)
                self.rewrite_log(root, lock, entry, log, pin=True)
                write_manifest(root, entries, lock)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaises(EvidenceBlocked):
                        verify_gguf_derivation(lock, "gguf.q8_0")

    def test_extra_and_duplicate_json_keys_are_rejected(self):
        for target, nested in (("record", False), ("record", True), ("command_log", False), ("command_log", True)):
            with self.subTest(target=target, nested=nested), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock = copy.deepcopy(self.lock)
                entry, record, log = self.write_derivation(root, lock, "gguf.bf16")
                value = record if target == "record" else log
                if nested:
                    value["tool_binary"]["unexpected"] = True
                else:
                    value["unexpected"] = True
                if target == "record":
                    self.rewrite_record(root, lock, entry, value, pin=True)
                else:
                    self.rewrite_log(root, lock, entry, value, pin=True)
                write_manifest(root, [entry], lock)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaises(EvidenceBlocked):
                        verify_gguf_derivation(lock, "gguf.bf16")

        for target, duplicate in (("record", b'"artifact_id":"gguf.bf16",'), ("command_log", b'"artifact_id":"gguf.bf16",')):
            with self.subTest(target=target, duplicate=True), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock = copy.deepcopy(self.lock)
                entry, _record, _log = self.write_derivation(root, lock, "gguf.bf16")
                original = (root / entry["derivation"][target]["path"]).read_bytes()
                forged = b"{" + duplicate + original[1:]
                self.write_raw_role(root, lock, entry, target, forged, pin=True)
                write_manifest(root, [entry], lock)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaisesRegex(EvidenceBlocked, "duplicate JSON key"):
                        verify_gguf_derivation(lock, "gguf.bf16")

    def test_arbitrary_and_oversized_command_logs_are_rejected(self):
        cases = (
            ("arbitrary", b"nonempty but unstructured converter output"),
            ("oversized", b"x" * (i0_evidence.MAX_COMMAND_LOG_BYTES + 1)),
        )
        for name, log_data in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                lock = copy.deepcopy(self.lock)
                entry, _record, _log = self.write_derivation(root, lock, "gguf.bf16")
                self.write_raw_role(root, lock, entry, "command_log", log_data, pin=True)
                write_manifest(root, [entry], lock)
                with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                    with self.assertRaises(EvidenceBlocked):
                        verify_gguf_derivation(lock, "gguf.bf16")

    @staticmethod
    def role_path(entry: dict, role: str) -> str:
        return entry["path"] if role == "output" else entry["derivation"][role]["path"]

    def alias_target(self, root: Path, lock: dict, source: dict, source_role: str, target: dict, target_role: str, mode: str) -> None:
        source_path = self.role_path(source, source_role)
        target_path = self.role_path(target, target_role)
        source_data = (root / source_path).read_bytes()
        if mode == "path":
            replacement_path = source_path
        else:
            (root / target_path).unlink()
            os.link(root / source_path, root / target_path)
            replacement_path = target_path
        if target_role == "output":
            target["path"] = replacement_path
            target["size"] = len(source_data)
            target["sha256"] = digest(source_data)
            identity = self.identity(lock, target["id"])
            identity["path"] = replacement_path
            identity["size"] = len(source_data)
            identity["sha256"] = digest(source_data)
            remote = next(
                item for item in lock["gguf_remote_reference"]["artifacts"]
                if item["lineage_claim"] == identity["lineage_lock"]["derivation_id"]
            )
            remote["size"] = len(source_data)
            remote["lfs_sha256"] = digest(source_data)
        else:
            old = target["derivation"][target_role]
            updated = descriptor(replacement_path, source_data, old["id"])
            target["derivation"][target_role] = updated
            self.identity(lock, target["id"])["lineage_lock"][target_role] = self._anchor(updated)

    def test_all_cross_gguf_path_and_inode_role_aliases_are_rejected(self):
        for mode in ("path", "inode"):
            for source_role in self.ROLES:
                for target_role in self.ROLES:
                    with self.subTest(mode=mode, source=source_role, target=target_role), tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        lock, entries, _records, _logs = self.three_derivations(root)
                        self.alias_target(root, lock, entries[0], source_role, entries[1], target_role, mode)
                        write_manifest(root, entries, lock)
                        with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                            result = verify_base02_gate(lock, _freeze_contract=False)
                        expected = "pairwise distinct" if mode == "path" else "inode-distinct"
                        self.assertTrue(any(expected in blocker.detail for blocker in result.blockers), result.blockers)
                        self.assertLess(result.lineage_executed, 3)

    def test_intra_derivation_all_cross_role_aliases_are_rejected(self):
        pairs = (("output", "record"), ("output", "command_log"), ("record", "command_log"))
        for mode in ("path", "inode"):
            for source_role, target_role in pairs:
                with self.subTest(mode=mode, source=source_role, target=target_role), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    lock = copy.deepcopy(self.lock)
                    entry, _record, _log = self.write_derivation(root, lock, "gguf.bf16")
                    self.alias_target(root, lock, entry, source_role, entry, target_role, mode)
                    write_manifest(root, [entry], lock)
                    with mock.patch.dict(os.environ, {"DG_ENGINE_I0_EVIDENCE_ROOT": str(root)}):
                        with self.assertRaises(EvidenceBlocked):
                            verify_gguf_derivation(lock, "gguf.bf16")



class SchemaAndScopeNegativeTests(unittest.TestCase):
    def test_evidence_schemas_require_exact_91_unique_entries(self):
        reference_validator = Draft202012Validator(
            read_json(SCHEMAS / "reference-lock.schema.json")
        )
        manifest_validator = Draft202012Validator(
            read_json(SCHEMAS / "evidence-manifest.schema.json")
        )
        lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        reference_validator.validate(lock)
        self.assertEqual(len(lock["evidence_root_contract"]["artifacts"]), 91)
        for mutate in (
            lambda entries: entries.pop(),
            lambda entries: entries.append(copy.deepcopy(entries[-1])),
        ):
            invalid = copy.deepcopy(lock)
            mutate(invalid["evidence_root_contract"]["artifacts"])
            with self.assertRaises(ValidationError):
                reference_validator.validate(invalid)

        manifest = read_json(ENGINE_DESIGN.parent / "build" / "i0-evidence-source" / "i0-evidence.json")
        manifest_validator.validate(manifest)
        self.assertEqual(len(manifest["artifacts"]), 91)
        for mutate in (
            lambda entries: entries.pop(),
            lambda entries: entries.append(copy.deepcopy(entries[-1])),
        ):
            invalid = copy.deepcopy(manifest)
            mutate(invalid["artifacts"])
            with self.assertRaises(ValidationError):
                manifest_validator.validate(invalid)

    def test_missing_semantic_section_is_rejected(self):
        schema = read_json(SCHEMAS / "semantics.schema.json")
        fixture = read_json(FIXTURES / "semantics.json")
        del fixture["trace"]
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(fixture)

    def test_wrong_node_count_and_malformed_nested_projection_are_rejected(self):
        schema = read_json(SCHEMAS / "node-modes.schema.json")
        inventory = read_json(ENGINE_DESIGN / "node-modes.json")
        wrong_count = copy.deepcopy(inventory)
        wrong_count["registered_node_count"] = 7
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(wrong_count)
        malformed = copy.deepcopy(inventory)
        malformed["nodes"][0]["object_info_projection"]["output_is_list"] = ["false"]
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(malformed)
        malformed_input = copy.deepcopy(inventory)
        malformed_input["nodes"][0]["object_info_projection"]["input"]["required"]["quant"] = {"unchecked": True}
        with self.assertRaises(ValidationError):
            Draft202012Validator(schema).validate(malformed_input)

    def test_reference_lock_nested_wrong_types_nulls_and_extras_are_rejected(self):
        schema = read_json(SCHEMAS / "reference-lock.schema.json")
        lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        mutations = (
            lambda value: value["official_model"].__setitem__("architecture_facts", None),
            lambda value: value["official_model"]["architecture_facts"].__setitem__("layers", "30"),
            lambda value: value["official_model"].__setitem__("sampler_profile", "invalid"),
            lambda value: value["official_model"]["sampler_profile"].__setitem__("entropy_bound", None),
            lambda value: value["official_model"]["sampler_profile"].__setitem__("unchecked", True),
            lambda value: next(
                item for item in value["evidence_root_contract"]["artifacts"] if item["id"] == "gguf.bf16"
            )["lineage_lock"].__setitem__(
                "record", {"id": "forged", "size": 1, "sha256": "0" * 64}
            ),
            lambda value: next(
                item for item in value["evidence_root_contract"]["artifacts"] if item["id"] == "gguf.bf16"
            ).__setitem__("lineage_status", "BOUND"),
        )
        Draft202012Validator(schema).validate(lock)
        for mutate in mutations:
            invalid = copy.deepcopy(lock)
            mutate(invalid)
            with self.assertRaises(ValidationError):
                Draft202012Validator(schema).validate(invalid)

    def test_reference_lock_rejects_extra_keys_at_multiple_nesting_depths(self):
        schema = read_json(SCHEMAS / "reference-lock.schema.json")
        lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        mutations = {
            "source-envelope": lambda value: value["source_baselines"][0].__setitem__("unexpected", True),
            "source-license-evidence": lambda value: value["source_baselines"][1]["license"]["evidence"].__setitem__(
                "unexpected", True
            ),
            "package-envelope": lambda value: value["oracle_packages"][0].__setitem__("unexpected", True),
            "package-pypi-file": lambda value: value["oracle_packages"][0]["pypi"]["wheel"].__setitem__(
                "unexpected", True
            ),
            "file-identity": lambda value: value["official_model"]["small_files"][0].__setitem__(
                "unexpected", True
            ),
            "recon-identity": lambda value: value["reconnaissance"]["ten_current_model_reads"][0].__setitem__(
                "unexpected", True
            ),
            "remote-license": lambda value: value["gguf_remote_reference"]["license"].__setitem__(
                "unexpected", True
            ),
            "remote-artifact": lambda value: value["gguf_remote_reference"]["artifacts"][0].__setitem__(
                "unexpected", True
            ),
            "gate-category": lambda value: value["evidence_root_contract"]["required_gate_categories"][
                "checkpoint_identity"
            ].__setitem__("unexpected", True),
            "gate-identity": lambda value: value["evidence_root_contract"]["artifacts"][0].__setitem__(
                "unexpected", True
            ),
            "checkpoint-lineage": lambda value: next(
                item for item in value["evidence_root_contract"]["artifacts"]
                if item["id"] == "official.checkpoint.00001-of-00011"
            )["lineage"].__setitem__("unexpected", True),
            "lineage-anchor": lambda value: next(
                item for item in value["evidence_root_contract"]["artifacts"] if item["id"] == "gguf.bf16"
            )["lineage_lock"].__setitem__("unexpected", True),
        }
        Draft202012Validator(schema).validate(lock)
        for name, mutate in mutations.items():
            with self.subTest(depth=name):
                invalid = copy.deepcopy(lock)
                mutate(invalid)
                with self.assertRaises(ValidationError):
                    Draft202012Validator(schema).validate(invalid)

    def test_i0_gate_contract_cannot_be_downgraded_by_lock_mutations(self):
        original = read_json(ENGINE_DESIGN / "reference.lock.json")

        def gate_artifacts(value):
            return value["evidence_root_contract"]["artifacts"]

        mutations = {
            "retagged": lambda value: next(
                item for item in gate_artifacts(value) if item["id"] == "official.checkpoint.00001-of-00011"
            ).__setitem__("required_for", ["BASE-02-source"]),
            "lowered-count": lambda value: value["evidence_root_contract"]["required_gate_categories"][
                "checkpoint_identity"
            ].__setitem__("expected", 1),
            "prefix-mismatch": lambda value: value["evidence_root_contract"]["required_gate_categories"][
                "gguf_identity"
            ].__setitem__("id_prefix", "forged."),
            "omitted": lambda value: gate_artifacts(value).remove(next(
                item for item in gate_artifacts(value) if item["id"] == "gguf.q4_k_m"
            )),
            "extra": lambda value: gate_artifacts(value).append({
                **copy.deepcopy(next(
                    item for item in gate_artifacts(value)
                    if item["id"] == "official.checkpoint.00011-of-00011"
                )),
                "id": "official.checkpoint.00012-of-00011",
            }),
            "substituted": lambda value: next(
                item for item in gate_artifacts(value) if item["id"] == "gguf.q8_0"
            ).__setitem__("id", "gguf.q8_substitute"),
            "wrong-kind": lambda value: next(
                item for item in gate_artifacts(value) if item["id"] == "gguf.q8_0"
            ).__setitem__("artifact_kind", "GGUF_Q4_K_M"),
            "extra-lineage-lock": lambda value: next(
                item for item in gate_artifacts(value) if item["id"] == "official.config.json"
            ).__setitem__("lineage_lock", {
                "status": "UNBOUND", "derivation_id": None, "record": None, "command_log": None,
            }),
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                invalid = copy.deepcopy(original)
                mutate(invalid)
                result = verify_base02_gate(invalid)
                self.assertEqual(result.artifact_executed, 0)
                self.assertEqual(result.lineage_executed, 0)
                self.assertEqual(len(result.blockers), 1)
                self.assertEqual(result.blockers[0].blocker, "I0-BLOCK-GATE-CONTRACT")
                self.assertFalse(gate_categories_complete(invalid, result))

    def test_empty_source_and_prospective_gate_scopes_fail_closed(self):
        self.assertFalse(scope_is_complete(0, []))
        lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        empty_gate = GateVerification(0, 0, {name: 0 for name in lock["evidence_root_contract"]["required_gate_categories"]}, [])
        self.assertFalse(gate_categories_complete(lock, empty_gate))
        self.assertEqual(empty_gate.artifact_executed, 0)
        self.assertEqual(empty_gate.lineage_executed, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
