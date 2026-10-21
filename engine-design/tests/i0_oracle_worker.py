#!/usr/bin/env python3
"""Single-use retained-byte I0 oracle worker; never imported by the parent."""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.abc
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import threading
import time
from types import ModuleType
import unittest

MAX_RESULT_BYTES = 32 * 1024
RESULT_SCHEMA = 3
TARGET_ROOTS = ("transformers", "diffusers", "dgemma", "surfaces", "_dg_engine_i0_pack")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _hash_fd(fd: int) -> tuple[int, str]:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256(); size = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk: break
        size += len(chunk); digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return size, digest.hexdigest()


def _read_fd(fd: int, maximum: int, role: str) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks = []; size = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, maximum + 1 - size))
        if not chunk: break
        chunks.append(chunk); size += len(chunk)
        if size > maximum: raise ValueError(f"{role} exceeds bound")
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _strict_json(data: bytes, role: str, canonical: bool = True) -> dict:
    def reject(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError(f"duplicate {role} key")
            value[key] = item
        return value
    value = json.loads(data, object_pairs_hook=reject,
                       parse_constant=lambda token: (_ for _ in ()).throw(ValueError(f"nonfinite {token}")))
    if type(value) is not dict: raise ValueError(f"{role} root is not object")
    if canonical and data != _canonical_bytes(value) + b"\n": raise ValueError(f"{role} is not canonical")
    return value


def _exact_keys(value, keys: set[str], role: str) -> dict:
    if type(value) is not dict or set(value) != keys: raise ValueError(f"{role} fields are not exact")
    return value


def _canonical_relative(value: object, role: str) -> str:
    if type(value) is not str or not value: raise ValueError(f"{role} is not string")
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise ValueError(f"{role} is not canonical relative")
    return path.as_posix()


def _source_map(harness: dict) -> dict[str, dict]:
    result = {}
    for item in harness["sources"]:
        _exact_keys(item, {"id", "path", "size", "sha256", "groups", "kind"}, "source")
        if item["id"] in result: raise ValueError("duplicate source")
        result[item["id"]] = item
    return result


def _inventory_hash(ids: list[str]) -> str:
    return _sha256_bytes(json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--group", choices=("BASE-01", "BASE-02"), required=True)
    parser.add_argument("--result-fd", type=int, required=True)
    parser.add_argument("--interpreter-fd", type=int, required=True)
    parser.add_argument("--harness-fd", type=int, required=True)
    parser.add_argument("--bound-file", action="append", default=[])
    parser.add_argument("--nonce", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--source-set-sha256", required=True)
    parser.add_argument("--environment-merkle", required=True)
    parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--test-fault", default="none")
    return parser.parse_args()


def _bound_fd_map(values: list[str]) -> dict[str, int]:
    result = {}
    for value in values:
        source_id, separator, fd_text = value.partition(":")
        if not separator or not source_id or source_id in result: raise ValueError("bound file argument invalid")
        fd = int(fd_text)
        if fd < 3 or fd > 4096: raise ValueError("bound file fd invalid")
        result[source_id] = fd
    return result


def _validate_result_fd(result_fd: int, reserved: set[int]) -> None:
    if result_fd < 3 or result_fd > 4096 or result_fd in reserved: raise ValueError("result fd identity invalid")
    observed = os.fstat(result_fd)
    if not stat.S_ISFIFO(observed.st_mode): raise ValueError("result fd is not a pipe")
    if fcntl.fcntl(result_fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_WRONLY:
        raise ValueError("result fd is not write-only")
    for text in os.listdir("/proc/self/fd"):
        if not text.isdigit(): continue
        fd = int(text)
        if fd == result_fd: continue
        try: other = os.fstat(fd)
        except OSError: continue
        if stat.S_ISFIFO(other.st_mode) and (other.st_dev, other.st_ino) == (observed.st_dev, observed.st_ino):
            raise ValueError("result pipe has a duplicate descriptor")
    os.set_inheritable(result_fd, False)


def _authenticate_image(args: argparse.Namespace, harness: dict) -> str:
    image = _exact_keys(harness["interpreter"]["image"], {"path", "dev", "ino", "size", "sha256"}, "image")
    opened = os.fstat(args.interpreter_fd); running = os.stat("/proc/self/exe")
    size, digest = _hash_fd(args.interpreter_fd)
    expected = (image["dev"], image["ino"], image["size"], image["sha256"])
    if (opened.st_dev, opened.st_ino, size, digest) != expected:
        raise ValueError("opened interpreter image differs")
    if (running.st_dev, running.st_ino, running.st_size) != expected[:3]:
        raise ValueError("/proc/self/exe identity differs")
    exe_fd = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
    try:
        if _hash_fd(exe_fd)[1] != image["sha256"]: raise ValueError("/proc/self/exe hash differs")
    finally: os.close(exe_fd)
    launcher = Path(args.repo_root) / _canonical_relative(harness["interpreter"]["launcher_path"], "launcher")
    if args.launcher.absolute() != launcher.absolute() or launcher.resolve(strict=True) != Path(image["path"]):
        raise ValueError("launcher argument differs from interpreter lock")
    return image["sha256"]


class _BoundLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, modules: dict[str, tuple[str, bytes]], repo_root: Path) -> None:
        self.modules = modules; self.repo_root = repo_root; self.executed: set[str] = set()
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.modules: return None
        spec = importlib.util.spec_from_loader(fullname, self, origin=f"candidate-blob://{self.modules[fullname][0]}")
        if spec is not None: spec.has_location = True
        return spec
    def create_module(self, spec): return None
    def exec_module(self, module: ModuleType) -> None:
        relative, data = self.modules[module.__name__]
        display_path = str(self.repo_root / relative)
        module.__file__ = display_path
        exec(compile(data, display_path, "exec", dont_inherit=True), module.__dict__)
        self.executed.add(module.__name__)


def _target_preloads() -> list[str]:
    return sorted(name for name in sys.modules if type(name) is str and
                  any(name == root or name.startswith(root + ".") for root in TARGET_ROOTS))


def _flatten(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite): yield from _flatten(item)
        else: yield item


def _blocker(error: object, category: str) -> dict[str, str]:
    return {"category": category, "blocker": str(getattr(error, "blocker", "I0-BLOCK-INTERNAL"))[:128],
            "artifact_id": str(getattr(error, "artifact_id", "oracle-worker"))[:256],
            "detail": str(getattr(error, "detail", error))[:512]}


class _FailOnWrite(io.TextIOBase):
    def __init__(self, name: str) -> None: self.name_for_error = name
    def writable(self) -> bool: return True
    def write(self, text: str) -> int: raise RuntimeError(f"write to {self.name_for_error} is forbidden")
    def flush(self) -> None: return None


def _install_at_fork(result_fd: int) -> None:
    def child_close() -> None:
        for fd in (result_fd, 1, 2):
            try: os.close(fd)
            except OSError: pass
    os.register_at_fork(after_in_child=child_close)


def _early_fault(args: argparse.Namespace) -> None:
    fault = args.test_fault
    if fault == "crash": os._exit(94)
    if fault == "hang": time.sleep(3600)
    if fault == "fork-setsid-descendant":
        child = os.fork()
        if child == 0:
            os.setsid(); time.sleep(3600); os._exit(0)
        time.sleep(3600)
    if fault == "stdout-output": os.write(1, b"unexpected-stdout"); time.sleep(3600)
    if fault == "stderr-output": os.write(2, b"unexpected-stderr"); time.sleep(3600)
    if fault == "buffered-print": print("unexpected-buffered-print"); time.sleep(3600)
    if fault == "malformed-result": os.write(args.result_fd, b"{"); os._exit(0)
    if fault == "truncated-result": os.write(args.result_fd, b'{"schema":3'); os._exit(0)
    if fault == "multiple-results": os.write(args.result_fd, b"{}\n{}\n"); os._exit(0)
    if fault == "oversized-result":
        chunk = b"x" * 4096
        for _ in range(MAX_RESULT_BYTES // len(chunk) + 2): os.write(args.result_fd, chunk)
        os._exit(0)


def _main() -> None:
    args = _parse()
    try:
        if not (sys.flags.isolated and sys.flags.ignore_environment and sys.flags.no_site and
                sys.flags.safe_path and sys.dont_write_bytecode and
                getattr(sys.stdout, "write_through", False) and getattr(sys.stderr, "write_through", False)):
            raise ValueError("worker requires exact -I -S -B -u")
        if len(threading.enumerate()) != 1: raise ValueError("worker has extra startup thread")
        bound_fds = _bound_fd_map(args.bound_file)
        _validate_result_fd(args.result_fd, {args.interpreter_fd, args.harness_fd, *bound_fds.values()})
        for fd in (args.interpreter_fd, args.harness_fd, *bound_fds.values()): os.set_inheritable(fd, False)
        harness_bytes = _read_fd(args.harness_fd, 4 * 1024 * 1024, "harness")
        harness = _strict_json(harness_bytes, "harness")
        _exact_keys(harness, {"schema", "kind", "git_tool", "interpreter", "environment", "candidate", "sources",
                              "test_groups", "boundary_test", "result_contract"}, "harness")
        if harness["schema"] != 2 or harness["kind"] != "dg-engine-i0-harness-lock": raise ValueError("harness identity")
        source_by_id = _source_map(harness)
        if set(bound_fds) != set(source_by_id): raise ValueError("bound source fd inventory differs from harness")
        source_bytes = {}
        for source_id, identity in source_by_id.items():
            fd = bound_fds[source_id]
            if not stat.S_ISREG(os.fstat(fd).st_mode): raise ValueError(f"bound source {source_id} is not regular")
            data = _read_fd(fd, identity["size"], source_id)
            if len(data) != identity["size"] or _sha256_bytes(data) != identity["sha256"]:
                raise ValueError(f"bound source {source_id} differs from harness")
            source_bytes[source_id] = data
        source_digest = _sha256_bytes(_canonical_bytes([
            {"id": source_id, "sha256": source_by_id[source_id]["sha256"]} for source_id in sorted(source_by_id)
        ]))
        if source_digest != args.source_set_sha256: raise ValueError("source-set hash differs from parent")
        environment = harness["environment"]
        if args.environment_merkle != environment["merkle_sha256"]: raise ValueError("environment Merkle differs")
        manifest_path = Path(environment["manifest_path"])
        if not manifest_path.is_absolute() or manifest_path.resolve(strict=True) != manifest_path:
            raise ValueError("environment manifest path is not exact")
        if args.environment_manifest.resolve(strict=True) != manifest_path:
            raise ValueError("environment manifest argument differs")
        if environment["sys_path"] != ["/usr/lib/python3.12", "/usr/lib/python3.12/lib-dynload",
                                       "/tmp/dg-engine-i0-oracle-environment/site-packages"]:
            raise ValueError("environment sys.path differs")
        image_hash = _authenticate_image(args, harness)
        boundary_name = harness["boundary_test"]["environment"]
        allowed = {"DG_ENGINE_I0_ORACLE_WORKER", "LC_CTYPE"}
        if args.test_fault != "none": allowed.add(boundary_name)
        if set(os.environ) - allowed: raise ValueError("worker environment has non-allowlisted variables")
        if os.environ.get("DG_ENGINE_I0_ORACLE_WORKER") != args.nonce: raise ValueError("worker nonce marker differs")
        if args.test_fault != "none":
            if os.environ.get(boundary_name) != "1" or args.test_fault not in harness["boundary_test"]["faults"]:
                raise ValueError("fault is outside boundary contract")
        elif boundary_name in os.environ: raise ValueError("boundary marker in production worker")
        if Path(args.repo_root).resolve(strict=True) != args.repo_root: raise ValueError("repo-root argument differs")
        if not (len(args.candidate_commit) == len(args.candidate_tree) == 40): raise ValueError("candidate identity malformed")

        reference_bytes = source_bytes["reference-lock"]
        reference_lock = _strict_json(reference_bytes, "reference lock", canonical=False)
        group_contract = harness["test_groups"][args.group]
        required = set(group_contract["source_ids"])
        if args.test_fault != "none": required.add(harness["boundary_test"]["source_id"])
        if not required <= set(source_bytes): raise ValueError("group source set is incomplete")

        for fd in (args.interpreter_fd, args.harness_fd, *bound_fds.values()):
            try: os.close(fd)
            except OSError: pass
        os.set_inheritable(1, False); os.set_inheritable(2, False)
        _install_at_fork(args.result_fd)
        _early_fault(args)
        sys.stdout = _FailOnWrite("stdout"); sys.stderr = _FailOnWrite("stderr")

        if args.test_fault == "preload": sys.modules["transformers"] = ModuleType("transformers")
        if args.test_fault == "extra-thread":
            event = threading.Event(); threading.Thread(target=event.wait, daemon=True).start()
        if _target_preloads() or len(threading.enumerate()) != 1: raise ValueError("worker startup contamination")

        sys.path[:] = list(environment["sys_path"])
        modules: dict[str, tuple[str, bytes]] = {}
        for source_id, identity in source_by_id.items():
            if identity["kind"] != "python" or source_id in ("parent-runner", "oracle-worker", "boundary-tests"): continue
            modules[Path(identity["path"]).stem] = (identity["path"], source_bytes[source_id])
        loader = _BoundLoader(modules, args.repo_root); sys.meta_path.insert(0, loader)
        import i0_evidence
        repository_bytes = {identity["path"]: source_bytes[source_id] for source_id, identity in source_by_id.items()
                            if identity["kind"] in ("json", "schema")}
        repository_bytes["engine-design/i0-harness.lock.json"] = harness_bytes
        i0_evidence.install_bound_repository_files(args.repo_root, repository_bytes)
        suite = unittest.TestSuite()
        for source_id in group_contract["test_module_ids"]:
            module_name = Path(source_by_id[source_id]["path"]).stem
            module = __import__(module_name)
            suite.addTests(unittest.defaultTestLoader.loadTestsFromModule(module))
        test_ids = [test.id() for test in _flatten(suite)]
        if test_ids != group_contract["test_ids"] or len(test_ids) != group_contract["count"] or \
                _inventory_hash(test_ids) != group_contract["inventory_sha256"]:
            raise ValueError("loaded retained test IDs/count/hash differ")
        expected_modules = {Path(source_by_id[item]["path"]).stem for item in group_contract["test_module_ids"]} | {"i0_evidence", "i0_testlib"}
        if not expected_modules <= loader.executed: raise ValueError("support/test module did not execute retained bytes")
        if _target_preloads() or len(threading.enumerate()) != 1: raise ValueError("discovery contaminated worker")

        evidence_root = args.evidence_root.resolve(strict=True)
        os.environ["DG_ENGINE_I0_EVIDENCE_ROOT"] = str(evidence_root)
        i0_evidence.validate_autoround_distribution_policy(reference_lock)
        policy = i0_evidence.validate_unresolved_blocker_model(reference_lock)
        scope = "BASE-01-source" if args.group == "BASE-01" else "BASE-02-source"
        source_executed, source_blockers = i0_evidence.verify_evidence_scope(reference_lock, scope)
        if source_executed <= 0 or source_blockers: os._exit(92)
        gate_data = None; blockers = []
        if args.group == "BASE-02":
            gate = i0_evidence.verify_base02_gate(reference_lock)
            gate_data = {"artifact_executed": gate.artifact_executed, "lineage_executed": gate.lineage_executed,
                         "categories": gate.categories, "blocked": len(gate.blockers)}
            for item in gate.blockers:
                category = "lineage" if item.blocker.startswith("I0-BLOCK-GGUF-LINEAGE-") else "evidence"
                blockers.append(_blocker(item, category))
            blockers.extend({"category": "phase-policy", "blocker": item["id"], "artifact_id": item["id"],
                             "detail": item["reason"][:512]} for item in policy)
        session = i0_evidence.initialize_authenticated_worker(
            reference_lock, nonce=args.nonce, worker_path=args.repo_root / source_by_id["oracle-worker"]["path"],
            repo_root=args.repo_root, tests_root=args.repo_root / "engine-design/tests",
            lock_path=args.repo_root / "engine-design/reference.lock.json", evidence_path=evidence_root,
            interpreter_path=Path(harness["interpreter"]["image"]["path"]), interpreter_sha256=image_hash,
        )
        test_output = io.StringIO()
        result = unittest.TextTestRunner(verbosity=0, stream=test_output).run(suite)
        counts = {"collected": len(test_ids), "executed": result.testsRun, "failed": len(result.failures),
                  "errors": len(result.errors), "skipped": len(result.skipped),
                  "expected_failures": len(result.expectedFailures), "unexpected_successes": len(result.unexpectedSuccesses)}
        if counts != {"collected": len(test_ids), "executed": len(test_ids), "failed": 0, "errors": 0, "skipped": 0,
                      "expected_failures": 0, "unexpected_successes": 0}:
            os.write(2, test_output.getvalue().encode("utf-8", "replace")[:4096])
            os._exit(91)
        disposition = "BLOCKED" if blockers else "PASS"; exit_code = 2 if blockers else 0
        manifest = evidence_root / reference_lock["evidence_root_contract"]["manifest"]
        manifest_fd = os.open(manifest, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try: evidence_manifest_hash = _hash_fd(manifest_fd)[1]
        finally: os.close(manifest_fd)
        payload = {"schema": RESULT_SCHEMA, "kind": "dg-engine-i0-oracle-worker-result", "nonce": args.nonce,
                   "group": args.group, "disposition": disposition, "counts": counts,
                   "tests": {"ids": test_ids, "count": len(test_ids), "inventory_sha256": _inventory_hash(test_ids)},
                   "source": {"executed": source_executed, "blocked": len(source_blockers)}, "gate": gate_data,
                   "blockers": blockers,
                   "hashes": {"candidate_commit": args.candidate_commit, "candidate_tree": args.candidate_tree,
                    "reference_lock_sha256": _sha256_bytes(reference_bytes), "harness_lock_sha256": _sha256_bytes(harness_bytes),
                    "evidence_manifest_sha256": evidence_manifest_hash,
                    "evidence_artifact_set_sha256": reference_lock["evidence_root_contract"]["artifact_set_sha256"],
                    "environment_merkle_sha256": args.environment_merkle, "interpreter_sha256": image_hash,
                    "worker_sha256": source_by_id["oracle-worker"]["sha256"], "test_source_set_sha256": source_digest},
                   "paths": {"launcher": str(args.launcher.absolute()), "interpreter_image": harness["interpreter"]["image"]["path"],
                    "repo_root": str(args.repo_root), "evidence_root": str(evidence_root),
                    "environment_manifest": str(args.environment_manifest)}}
        fault = args.test_fault
        if fault == "wrong-candidate": payload["hashes"]["candidate_commit"] = "0" * 40
        elif fault == "wrong-tree": payload["hashes"]["candidate_tree"] = "0" * 40
        elif fault == "wrong-environment": payload["hashes"]["environment_merkle_sha256"] = "0" * 64
        elif fault == "wrong-worker-hash": payload["hashes"]["worker_sha256"] = "0" * 64
        elif fault == "wrong-interpreter-hash": payload["hashes"]["interpreter_sha256"] = "0" * 64
        elif fault == "wrong-reference-lock-hash": payload["hashes"]["reference_lock_sha256"] = "0" * 64
        elif fault == "wrong-harness-lock-hash": payload["hashes"]["harness_lock_sha256"] = "0" * 64
        elif fault == "wrong-evidence-hash": payload["hashes"]["evidence_manifest_sha256"] = "0" * 64
        elif fault == "forged-counts": payload["counts"]["collected"] += 1
        elif fault == "forged-test-ids": payload["tests"]["ids"] = payload["tests"]["ids"][:-1]
        elif fault == "forged-blockers": payload["blockers"] = [{"category":"evidence","blocker":"FORGED","artifact_id":"x","detail":"x"}]
        elif fault == "forged-gate": payload["gate"]["categories"]["checkpoint_identity"] = 1
        elif fault == "wrong-disposition": payload["disposition"] = "BLOCKED" if disposition == "PASS" else "PASS"
        elif fault == "wrong-nonce-result": payload["nonce"] = "0" * 64
        elif fault == "wrong-paths": payload["paths"]["repo_root"] = str(args.repo_root / "engine-design")
        elif fault == "oracle-mutation": setattr(session.oracle_guard.object_for("EntropyBoundScheduler"), "_fault", True)
        elif fault == "oracle-dependency-mutation":
            sys.modules["transformers.models.auto.processing_auto"].PROCESSOR_MAPPING_NAMES["_fault"] = "Injected"
        wire = _canonical_bytes(payload) + b"\n"
        if fault == "truncated-authenticated-result": wire = wire[:len(wire)//2]
        elif fault == "multiple-authenticated-results": wire += b"{}\n"
        if len(wire) > MAX_RESULT_BYTES: os._exit(93)
        i0_evidence.assert_oracle_suite_integrity(dependency_closure=args.group == "BASE-02")
        if os.write(args.result_fd, wire) != len(wire): os._exit(95)
        if fault == "wrong-exit": os._exit(7)
        os._exit(exit_code)
    except BaseException:
        os._exit(96)


if __name__ == "__main__": _main()
