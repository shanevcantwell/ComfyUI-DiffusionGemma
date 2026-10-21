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
