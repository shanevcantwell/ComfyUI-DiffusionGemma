#!/usr/bin/env python3
"""Fail-closed parent for one committed I0 candidate and one oracle worker.

Production evidence has one entry: :func:`launch_worker`.  It accepts no process
factory, worker path, or fault knob.  The boundary suite uses the separate
:func:`launch_worker_test_fault`, whose contract requires an enumerated fault
and can never return a production success.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
import csv
import ctypes
import errno
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import selectors
import signal
import stat
import subprocess
import sys
import time
from typing import Any

TESTS = Path(__file__).resolve().parent
ENGINE_DESIGN = TESTS.parent
REPO_ROOT = ENGINE_DESIGN.parent
HARNESS_REL = "engine-design/i0-harness.lock.json"
HARNESS_LOCK_PATH = REPO_ROOT / HARNESS_REL
REFERENCE_LOCK_PATH = ENGINE_DESIGN / "reference.lock.json"
WORKER_PATH = TESTS / "i0_oracle_worker.py"
GIT_PATH = Path("/usr/bin/git")
MAX_RESULT_BYTES = 32 * 1024
MAX_SIDE_BYTES = 4 * 1024
POST_KILL_SECONDS = 3.0
RESULT_SCHEMA = 3
PR_SET_CHILD_SUBREAPER = 36
_TEST_ENV_CACHE: tuple[str, dict[str, Any]] | None = None


class WorkerBlocked(RuntimeError):
    """The candidate or subprocess boundary could not authenticate a result."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _hash_fd(fd: int) -> tuple[int, str]:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    size = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        size += len(chunk)
        digest.update(chunk)
    os.lseek(fd, 0, os.SEEK_SET)
    return size, digest.hexdigest()


def _read_fd(fd: int, maximum: int, role: str) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = os.read(fd, min(1024 * 1024, maximum + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
        if size > maximum:
            raise WorkerBlocked(f"{role} exceeds its byte bound")
    os.lseek(fd, 0, os.SEEK_SET)
    return b"".join(chunks)


def _open_regular(path: Path, role: str) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise WorkerBlocked(f"cannot open {role}: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise WorkerBlocked(f"{role} is not a regular file")
        return fd
    except BaseException:
        os.close(fd)
        raise


def _strict_json_bytes(data: bytes, role: str, *, canonical: bool = True) -> dict[str, Any]:
    if not data:
        raise WorkerBlocked(f"{role} is empty")

    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WorkerBlocked(f"duplicate {role} key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            object_pairs_hook=reject,
            parse_constant=lambda token: (_ for _ in ()).throw(
                WorkerBlocked(f"non-finite {role} number: {token}")
            ),
        )
    except WorkerBlocked:
        raise
    except Exception as error:
        raise WorkerBlocked(f"{role} is malformed JSON: {error}") from error
    if type(value) is not dict:
        raise WorkerBlocked(f"{role} root is not an object")
    if canonical and data != _canonical_bytes(value) + b"\n":
        raise WorkerBlocked(f"{role} is not one exact canonical JSON record")
    return value


def _exact_keys(value: object, keys: set[str], role: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise WorkerBlocked(f"{role} fields are not exact")
    return value


def _exact_string(value: object, role: str, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise WorkerBlocked(f"{role} is not a bounded nonempty string")
    return value


def _canonical_relative(value: object, role: str) -> str:
    text = _exact_string(value, role)
    path = Path(text)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise WorkerBlocked(f"{role} is not a canonical relative path")
    return path.as_posix()


def _git(args: list[str], *, maximum: int = 16 * 1024 * 1024, check_stderr: bool = True) -> bytes:
    try:
        completed = subprocess.run(
            [str(GIT_PATH), *args], cwd=REPO_ROOT, env={}, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise WorkerBlocked(f"candidate Git read failed: {error}") from error
    if completed.returncode != 0 or (check_stderr and completed.stderr) or len(completed.stdout) > maximum:
        detail = completed.stderr[:512].decode("utf-8", "replace")
        raise WorkerBlocked(f"candidate Git read was not clean and bounded: {detail}")
    return completed.stdout


def _git_blob(commit: str, path: str, maximum: int) -> bytes:
    return _git(["cat-file", "blob", f"{commit}:{path}"], maximum=maximum, check_stderr=True)


def _inventory_hash(ids: list[str]) -> str:
    return _sha256_bytes(json.dumps(ids, separators=(",", ":"), ensure_ascii=True).encode("ascii"))


def _candidate_identity() -> tuple[str, str]:
    lines = _git(["rev-parse", "HEAD", "HEAD^{tree}"], maximum=256).decode("ascii", "strict").splitlines()
    if len(lines) != 2 or any(len(item) != 40 or set(item) - set("0123456789abcdef") for item in lines):
        raise WorkerBlocked("candidate commit/tree identity is malformed")
    dirty = _git(
        ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
        maximum=1024 * 1024,
    )
    if dirty:
        first = dirty.split(b"\0", 1)[0].decode("utf-8", "replace")
        raise WorkerBlocked(f"production candidate is not clean: {first}")
    return lines[0], lines[1]


def _load_harness_from_candidate(commit: str) -> tuple[dict[str, Any], bytes]:
    data = _git_blob(commit, HARNESS_REL, 4 * 1024 * 1024)
    value = _strict_json_bytes(data, "candidate harness lock")
    return value, data


def _source_map(harness: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    sources = harness.get("sources")
    if type(sources) is not list or not sources:
        raise WorkerBlocked("harness source set is empty")
    for index, item in enumerate(sources):
        _exact_keys(item, {"id", "path", "size", "sha256", "groups", "kind"}, f"source {index}")
        source_id = _exact_string(item["id"], f"source {index} id", 128)
        if source_id in result:
            raise WorkerBlocked(f"duplicate source id: {source_id}")
        _canonical_relative(item["path"], f"source {source_id} path")
        if type(item["size"]) is not int or item["size"] <= 0:
            raise WorkerBlocked(f"source {source_id} size is invalid")
        if type(item["sha256"]) is not str or len(item["sha256"]) != 64:
            raise WorkerBlocked(f"source {source_id} hash is invalid")
        if item["kind"] not in ("python", "json", "schema"):
            raise WorkerBlocked(f"source {source_id} kind is invalid")
        if type(item["groups"]) is not list or not item["groups"]:
            raise WorkerBlocked(f"source {source_id} groups are invalid")
        result[source_id] = item
    return result


def _validate_harness(harness: dict[str, Any], candidate_commit: str) -> None:
    _exact_keys(
        harness,
        {"schema", "kind", "git_tool", "interpreter", "environment", "candidate", "sources",
         "test_groups", "boundary_test", "result_contract"},
        "harness lock",
    )
    if harness["schema"] != 2 or harness["kind"] != "dg-engine-i0-harness-lock":
        raise WorkerBlocked("harness lock identity is not exact")
    git = _exact_keys(harness["git_tool"], {"path", "size", "sha256"}, "Git identity")
    if git["path"] != str(GIT_PATH):
        raise WorkerBlocked("Git path is not the compiled bootstrap path")
    git_fd = _open_regular(GIT_PATH, "Git tool")
    try:
        if _hash_fd(git_fd) != (git["size"], git["sha256"]):
            raise WorkerBlocked("Git tool differs from the candidate lock")
    finally:
        os.close(git_fd)
    source_by_id = _source_map(harness)
    candidate = _exact_keys(harness["candidate"], {"relevant_paths", "inventory_sha256"}, "candidate contract")
    paths = candidate["relevant_paths"]
    if type(paths) is not list or paths != sorted(paths) or len(paths) != len(set(paths)):
        raise WorkerBlocked("candidate relevant path inventory is not exact/sorted")
    if any(_canonical_relative(path, "candidate path") != path for path in paths):
        raise WorkerBlocked("candidate path is not canonical")
    expected_paths = sorted([HARNESS_REL, *(item["path"] for item in source_by_id.values())])
    if paths != expected_paths or candidate["inventory_sha256"] != _inventory_hash(paths):
        raise WorkerBlocked("candidate path inventory differs from source/harness lock")
    tree_paths = _git(
        ["ls-tree", "-r", "--name-only", candidate_commit, "--", *paths], maximum=2 * 1024 * 1024
    ).decode("utf-8", "strict").splitlines()
    if tree_paths != paths:
        raise WorkerBlocked("candidate committed tree does not contain the exact locked path inventory")
    groups = _exact_keys(harness["test_groups"], {"BASE-01", "BASE-02"}, "test groups")
    for group, contract in groups.items():
        _exact_keys(contract, {"source_ids", "test_module_ids", "test_ids", "count", "inventory_sha256"}, group)
        if any(source_id not in source_by_id for source_id in contract["source_ids"]):
            raise WorkerBlocked(f"{group} contains an unknown source")
        if any(source_by_id[source_id]["kind"] != "python" for source_id in contract["test_module_ids"]):
            raise WorkerBlocked(f"{group} test module source is not Python")
        ids = contract["test_ids"]
        if type(ids) is not list or not ids or len(ids) != len(set(ids)):
            raise WorkerBlocked(f"{group} test IDs are invalid")
        if contract["count"] != len(ids) or contract["inventory_sha256"] != _inventory_hash(ids):
            raise WorkerBlocked(f"{group} test inventory differs from IDs")
    boundary = _exact_keys(harness["boundary_test"], {"source_id", "environment", "faults"}, "boundary")
    if boundary["source_id"] not in source_by_id or type(boundary["faults"]) is not list:
        raise WorkerBlocked("boundary contract is invalid")
    if "none" in boundary["faults"] or len(boundary["faults"]) != len(set(boundary["faults"])):
        raise WorkerBlocked("boundary fault inventory is invalid")
    if harness["result_contract"] != {
        "schema": RESULT_SCHEMA, "max_bytes": MAX_RESULT_BYTES, "max_side_bytes": MAX_SIDE_BYTES
    }:
        raise WorkerBlocked("result contract differs from parent implementation")


def _open_candidate_sources(harness: dict[str, Any], candidate_commit: str) -> tuple[dict[str, int], dict[str, bytes], str]:
    fds: dict[str, int] = {}
    blobs: dict[str, bytes] = {}
    try:
        for source_id, identity in _source_map(harness).items():
            relative = identity["path"]
            blob = _git_blob(candidate_commit, relative, max(identity["size"] + 1, 1024))
            if len(blob) != identity["size"] or _sha256_bytes(blob) != identity["sha256"]:
                raise WorkerBlocked(f"candidate blob {source_id} differs from harness lock")
            fd = _open_regular(REPO_ROOT / relative, f"worktree source {source_id}")
            if _hash_fd(fd) != (identity["size"], identity["sha256"]):
                os.close(fd)
                raise WorkerBlocked(f"worktree source {source_id} differs from candidate blob/lock")
            fds[source_id] = fd
            blobs[source_id] = blob
        aggregate = _sha256_bytes(_canonical_bytes([
            {"id": source_id, "sha256": _source_map(harness)[source_id]["sha256"]}
            for source_id in sorted(fds)
        ]))
        return fds, blobs, aggregate
    except BaseException:
        for fd in fds.values():
            os.close(fd)
        raise


def _verify_environment(harness: dict[str, Any], *, cache_for_test: bool) -> dict[str, Any]:
    global _TEST_ENV_CACHE
    contract = _exact_keys(
        harness["environment"],
        {"manifest_path", "manifest_sha256", "merkle_sha256", "roots", "sys_path", "record_policy"},
        "environment contract",
    )
    cache_key = _sha256_bytes(_canonical_bytes(contract))
    if cache_for_test and _TEST_ENV_CACHE is not None and _TEST_ENV_CACHE[0] == cache_key:
        return _TEST_ENV_CACHE[1]
    manifest_path = Path(_exact_string(contract["manifest_path"], "environment manifest path", 4096))
    if not manifest_path.is_absolute() or manifest_path.resolve(strict=True) != manifest_path:
        raise WorkerBlocked("environment manifest path is not exact canonical absolute")
    fd = _open_regular(manifest_path, "external environment manifest")
    try:
        data = _read_fd(fd, 16 * 1024 * 1024, "environment manifest")
    finally:
        os.close(fd)
    if _sha256_bytes(data) != contract["manifest_sha256"]:
        raise WorkerBlocked("external environment manifest differs from the Git-pinned hash")
    manifest = _strict_json_bytes(data, "environment manifest")
    _exact_keys(manifest, {"schema", "kind", "roots", "merkle_sha256"}, "environment manifest")
    if manifest["schema"] != 1 or manifest["kind"] != "dg-engine-i0-environment-manifest":
        raise WorkerBlocked("environment manifest identity is invalid")
    if manifest["merkle_sha256"] != contract["merkle_sha256"]:
        raise WorkerBlocked("environment Merkle differs from Git lock")
    locked_roots = contract["roots"]
    if type(locked_roots) is not list or len(locked_roots) != len(manifest["roots"]):
        raise WorkerBlocked("environment root count differs")
    root_by_id: dict[str, Path] = {}
    site_entries: dict[str, dict[str, Any]] = {}
    site_record_bytes: dict[str, bytes] = {}
    for locked, declared in zip(locked_roots, manifest["roots"], strict=True):
        _exact_keys(locked, {"id", "path", "entry_count", "merkle_sha256"}, "locked environment root")
        _exact_keys(declared, {"id", "path", "entries", "entry_count", "merkle_sha256"}, "manifest root")
        if {key: declared[key] for key in ("id", "path", "entry_count", "merkle_sha256")} != locked:
            raise WorkerBlocked("environment root identity differs from Git lock")
        root = Path(declared["path"])
        if not root.is_absolute() or root.resolve(strict=True) != root or not root.is_dir():
            raise WorkerBlocked("environment root path is not exact/canonical")
        entries = declared["entries"]
        if type(entries) is not list or len(entries) != declared["entry_count"]:
            raise WorkerBlocked("environment root entry count is invalid")
        if _sha256_bytes(_canonical_bytes(entries)) != declared["merkle_sha256"]:
            raise WorkerBlocked("environment root manifest Merkle is invalid")
        declared_names = [entry.get("path") for entry in entries if type(entry) is dict]
        observed_names = {
            path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()
        }
        if len(declared_names) != len(entries) or observed_names != set(declared_names):
            extra = sorted(observed_names - set(declared_names))
            missing = sorted(set(declared_names) - observed_names)
            raise WorkerBlocked(f"environment root has extras/missing files: extra={extra[:1]} missing={missing[:1]}")
        expected_names: set[str] = set()
        for entry in entries:
            if type(entry) is not dict or set(entry) not in (
                {"path", "type", "size", "sha256"}, {"path", "type", "target", "size", "sha256"}
            ):
                raise WorkerBlocked("environment entry fields are invalid")
            relative = _canonical_relative(entry["path"], "environment entry path")
            if relative in expected_names:
                raise WorkerBlocked("duplicate environment path")
            expected_names.add(relative)
            path = root / relative
            if entry["type"] == "file":
                file_fd = _open_regular(path, f"environment file {relative}")
                try:
                    observed = _hash_fd(file_fd)
                    if declared["id"] == "site-packages" and relative.endswith(".dist-info/RECORD"):
                        site_record_bytes[relative] = _read_fd(file_fd, 8 * 1024 * 1024, f"package RECORD {relative}")
                finally: os.close(file_fd)
            elif entry["type"] == "symlink":
                if not path.is_symlink() or os.readlink(path) != entry["target"]:
                    raise WorkerBlocked(f"environment symlink {relative} differs")
                resolved = path.resolve(strict=True)
                file_fd = _open_regular(resolved, f"environment symlink target {relative}")
                try: observed = _hash_fd(file_fd)
                finally: os.close(file_fd)
            else:
                raise WorkerBlocked("environment entry type is invalid")
            if observed != (entry["size"], entry["sha256"]):
                raise WorkerBlocked(f"environment file {relative} differs from manifest")
        root_by_id[declared["id"]] = root
        if declared["id"] == "site-packages":
            site_entries = {item["path"]: item for item in entries}
    site_root = root_by_id.get("site-packages")
    if site_root is None:
        raise WorkerBlocked("site-packages environment root is absent")
    forbidden = sorted(
        name for name in site_entries
        if name.endswith(".pth") or Path(name).name in ("sitecustomize.py", "usercustomize.py")
    )
    if forbidden:
        raise WorkerBlocked(f"environment contains forbidden startup injection: {forbidden[0]}")
    policy = _exact_keys(contract["record_policy"], {"verify_all", "allow_omitted"}, "RECORD policy")
    if policy["verify_all"] is not True or policy["allow_omitted"] != ["*.pth"]:
        raise WorkerBlocked("package RECORD policy is not exact")
    record_count = 0
    for relative, identity in sorted(site_entries.items()):
        if not relative.endswith(".dist-info/RECORD") or relative.count("/") != 1:
            continue
        record_count += 1
        rows = csv.reader(io.StringIO(site_record_bytes[relative].decode("utf-8", "strict")))
        for row in rows:
            if len(row) != 3:
                raise WorkerBlocked(f"package RECORD row is malformed: {relative}")
            record_path, encoded_hash, size_text = row
            normalized = Path(record_path)
            if normalized.is_absolute():
                raise WorkerBlocked(f"package RECORD absolute path: {relative}")
            if ".." in normalized.parts:
                continue  # console scripts are outside the permitted import root
            name = normalized.as_posix()
            present = site_entries.get(name)
            if present is None:
                if name.endswith(".pth"):
                    continue
                raise WorkerBlocked(f"package RECORD file missing from closed root: {name}")
            if encoded_hash:
                algorithm, token = encoded_hash.split("=", 1)
                if algorithm != "sha256":
                    raise WorkerBlocked(f"package RECORD uses non-SHA256 hash: {name}")
                observed_token = base64.urlsafe_b64encode(bytes.fromhex(present["sha256"])).rstrip(b"=").decode("ascii")
                if token != observed_token or int(size_text) != present["size"]:
                    raise WorkerBlocked(f"package RECORD identity differs: {name}")
    if record_count < 2:
        raise WorkerBlocked("package RECORD coverage is empty")
    expected_sys_path = contract["sys_path"]
    if expected_sys_path != ["/usr/lib/python3.12", "/usr/lib/python3.12/lib-dynload", str(site_root)]:
        raise WorkerBlocked("environment sys.path contract is not exact")
    result = {"merkle": manifest["merkle_sha256"], "sys_path": expected_sys_path, "manifest_path": str(manifest_path)}
    if cache_for_test:
        _TEST_ENV_CACHE = (cache_key, result)
    return result


def _authenticate_interpreter(harness: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    interpreter = _exact_keys(harness["interpreter"], {"launcher_path", "image"}, "interpreter")
    launcher = REPO_ROOT / _canonical_relative(interpreter["launcher_path"], "launcher path")
    if Path(sys.executable).absolute() != launcher.absolute():
        raise WorkerBlocked(f"parent interpreter is not locked launcher: {sys.executable}")
    if not (sys.flags.isolated and sys.flags.ignore_environment and sys.flags.no_site and sys.flags.safe_path and sys.dont_write_bytecode):
        raise WorkerBlocked("parent requires exact Python -I -S -B mode")
    image = _exact_keys(interpreter["image"], {"path", "dev", "ino", "size", "sha256"}, "image")
    path = Path(image["path"])
    if launcher.resolve(strict=True) != path or path.resolve(strict=True) != path:
        raise WorkerBlocked("launcher does not resolve to exact interpreter image")
    fd = _open_regular(path, "interpreter image")
    try:
        observed = os.fstat(fd)
        size, digest = _hash_fd(fd)
    finally:
        os.close(fd)
    if (observed.st_dev, observed.st_ino, size, digest) != (
        image["dev"], image["ino"], image["size"], image["sha256"]
    ):
        raise WorkerBlocked("interpreter image differs from lock")
    return launcher.absolute(), image


def _memfd(name: str, data: bytes) -> int:
    if not hasattr(os, "memfd_create"):
        raise WorkerBlocked("Linux memfd_create is required")
    fd = os.memfd_create(name, os.MFD_CLOEXEC | getattr(os, "MFD_ALLOW_SEALING", 0))
    try:
        if os.write(fd, data) != len(data):
            raise WorkerBlocked(f"short write creating {name}")
        os.lseek(fd, 0, os.SEEK_SET)
        seals = 0
        for seal in ("F_SEAL_SEAL", "F_SEAL_SHRINK", "F_SEAL_GROW", "F_SEAL_WRITE"):
            seals |= getattr(fcntl, seal, 0)
        if seals:
            fcntl.fcntl(fd, fcntl.F_ADD_SEALS, seals)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _expected_blockers(lock: dict[str, Any], group: str) -> Counter[tuple[str, str, str]]:
    if group == "BASE-01":
        return Counter()
    expected: Counter[tuple[str, str, str]] = Counter()
    for identity in lock["evidence_root_contract"]["artifacts"]:
        if "BASE-02-gate" not in identity.get("required_for", []):
            continue
        if identity["status"] != "present":
            expected[("evidence", identity["blocker"], identity["id"])] += 1
        if identity["id"].startswith("gguf."):
            expected[("lineage", identity["derivation_requirement"]["blocker"], identity["id"])] += 1
    for item in lock["unresolved_blockers"]:
        if item["classification"] == "phase-policy":
            expected[("phase-policy", item["id"], item["id"])] += 1
    return expected


def _validate_result(
    value: dict[str, Any], *, group: str, reference_lock: dict[str, Any], harness: dict[str, Any],
    expected: dict[str, str], evidence_root: Path, nonce: str,
) -> int:
    _exact_keys(
        value,
        {"schema", "kind", "nonce", "group", "disposition", "counts", "tests", "source", "gate",
         "blockers", "hashes", "paths"},
        "worker result",
    )
    if value["schema"] != RESULT_SCHEMA or value["kind"] != "dg-engine-i0-oracle-worker-result":
        raise WorkerBlocked("worker result schema/kind is not exact")
    if value["nonce"] != nonce or value["group"] != group:
        raise WorkerBlocked("worker result nonce/group differs from parent contract")
    locked = harness["test_groups"][group]
    tests = _exact_keys(value["tests"], {"ids", "count", "inventory_sha256"}, "tests")
    if tests != {"ids": locked["test_ids"], "count": locked["count"], "inventory_sha256": locked["inventory_sha256"]}:
        raise WorkerBlocked("worker test IDs/count/hash differ from exact lock")
    counts = _exact_keys(
        value["counts"], {"collected", "executed", "failed", "errors", "skipped", "expected_failures", "unexpected_successes"},
        "counts",
    )
    exact = locked["count"]
    if counts != {"collected": exact, "executed": exact, "failed": 0, "errors": 0, "skipped": 0,
                   "expected_failures": 0, "unexpected_successes": 0}:
        raise WorkerBlocked("worker test counts are forged, incomplete, skipped, or failing")
    scope = "BASE-01-source" if group == "BASE-01" else "BASE-02-source"
    expected_source = sum(1 for item in reference_lock["evidence_root_contract"]["artifacts"]
                          if scope in item.get("required_for", []) and item["status"] == "present")
    if value["source"] != {"executed": expected_source, "blocked": 0} or expected_source <= 0:
        raise WorkerBlocked("source evidence counts differ")
    blockers = value["blockers"]
    if type(blockers) is not list or len(blockers) > 64:
        raise WorkerBlocked("blocker list is not bounded")
    observed: Counter[tuple[str, str, str]] = Counter()
    for item in blockers:
        _exact_keys(item, {"category", "blocker", "artifact_id", "detail"}, "blocker")
        observed[(item["category"], item["blocker"], item["artifact_id"])] += 1
    expected_blockers = _expected_blockers(reference_lock, group)
    if observed != expected_blockers:
        raise WorkerBlocked("worker blocker multiset/categories differ from lock")
    hashes = _exact_keys(
        value["hashes"], {"candidate_commit", "candidate_tree", "reference_lock_sha256", "harness_lock_sha256",
        "evidence_manifest_sha256", "evidence_artifact_set_sha256", "environment_merkle_sha256",
        "interpreter_sha256", "worker_sha256", "test_source_set_sha256"}, "hashes")
    expected_hashes = {
        "candidate_commit": expected["candidate_commit"], "candidate_tree": expected["candidate_tree"],
        "reference_lock_sha256": expected["reference_lock_sha256"], "harness_lock_sha256": expected["harness_lock_sha256"],
        "evidence_manifest_sha256": expected["evidence_manifest_sha256"],
        "evidence_artifact_set_sha256": reference_lock["evidence_root_contract"]["artifact_set_sha256"],
        "environment_merkle_sha256": expected["environment_merkle_sha256"],
        "interpreter_sha256": expected["interpreter_sha256"], "worker_sha256": expected["worker_sha256"],
        "test_source_set_sha256": expected["test_source_set_sha256"],
    }
    if hashes != expected_hashes:
        raise WorkerBlocked("worker candidate/environment/interpreter/source identities differ")
    paths = _exact_keys(value["paths"], {"launcher", "interpreter_image", "repo_root", "evidence_root", "environment_manifest"}, "paths")
    if paths != {"launcher": expected["launcher"], "interpreter_image": expected["interpreter_image"],
                 "repo_root": str(REPO_ROOT.resolve()), "evidence_root": str(evidence_root),
                 "environment_manifest": expected["environment_manifest"]}:
        raise WorkerBlocked("worker exact paths differ from parent contract")
    if group == "BASE-01":
        if value["gate"] is not None or value["disposition"] != "PASS" or blockers:
            raise WorkerBlocked("BASE-01 result is not exact PASS")
        return 0
    gate = _exact_keys(value["gate"], {"artifact_executed", "lineage_executed", "categories", "blocked"}, "gate")
    category_counts = Counter(category for category, _blocker, _artifact in expected_blockers.elements())
    expected_gate = {"artifact_executed": 0, "lineage_executed": 0,
                     "categories": {"checkpoint_identity": 0, "gguf_identity": 0, "gguf_authenticated_lineage": 0},
                     "blocked": 17}
    if gate != expected_gate or category_counts != Counter({"evidence": 14, "lineage": 3, "phase-policy": 1}):
        raise WorkerBlocked("BASE-02 gate categories differ from exact 14+3+1 scope")
    if value["disposition"] != "BLOCKED":
        raise WorkerBlocked("BASE-02 disposition is not BLOCKED")
    return 2


def _set_subreaper() -> None:
    try:
        if ctypes.CDLL(None, use_errno=True).prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_CHILD_SUBREAPER)")
    except (AttributeError, OSError) as error:
        raise WorkerBlocked(f"platform process-tree containment unavailable: {error}") from error


def _proc_ppid(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError):
        return None
    return None


def _discover_descendants(root_pid: int, tracked: dict[int, int]) -> None:
    ppid: dict[int, int] = {}
    for item in Path("/proc").iterdir():
        if item.name.isdigit():
            parent = _proc_ppid(int(item.name))
            if parent is not None:
                ppid[int(item.name)] = parent
    queue = deque([root_pid])
    descendants: set[int] = set()
    while queue:
        parent = queue.popleft()
        for pid, observed_parent in ppid.items():
            if observed_parent == parent and pid not in descendants:
                descendants.add(pid); queue.append(pid)
    for pid in {root_pid, *descendants}:
        if pid not in tracked:
            try: tracked[pid] = os.pidfd_open(pid, 0)
            except ProcessLookupError: pass
            except OSError as error: raise WorkerBlocked(f"pidfd containment unavailable: {error}") from error


def _kill_tracked(tracked: dict[int, int]) -> None:
    for pidfd in tuple(tracked.values()):
        try: signal.pidfd_send_signal(pidfd, signal.SIGKILL)
        except ProcessLookupError: pass
    deadline = time.monotonic() + POST_KILL_SECONDS
    while time.monotonic() < deadline:
        reaped = False
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                pid = 0
            if pid <= 0: break
            reaped = True
        alive = []
        for pidfd in tracked.values():
            poller = selectors.DefaultSelector()
            try:
                poller.register(pidfd, selectors.EVENT_READ)
                if not poller.select(0): alive.append(pidfd)
            finally: poller.close()
        if not alive: break
        if not reaped: time.sleep(0.01)
    for pidfd in tracked.values():
        try: os.close(pidfd)
        except OSError: pass
    tracked.clear()


def _drain_process(process: subprocess.Popen[bytes], result_fd: int, timeout: float) -> tuple[int, bytes]:
    selector = selectors.DefaultSelector()
    streams = {result_fd: "result", process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}  # type: ignore[union-attr]
    for fd, role in streams.items():
        os.set_blocking(fd, False); selector.register(fd, selectors.EVENT_READ, role)
    buffers = {role: bytearray() for role in ("result", "stdout", "stderr")}
    tracked: dict[int, int] = {}
    deadline = time.monotonic() + timeout
    error: str | None = None
    try:
        while selector.get_map() or process.poll() is None:
            _discover_descendants(process.pid, tracked)
            if time.monotonic() >= deadline:
                error = error or f"oracle worker timeout after {timeout:g} seconds"
                break
            for key, _mask in selector.select(min(0.05, deadline - time.monotonic())):
                try: chunk = os.read(key.fd, 4096)
                except BlockingIOError: continue
                role = key.data
                if not chunk:
                    selector.unregister(key.fd)
                    continue
                limit = MAX_RESULT_BYTES if role == "result" else MAX_SIDE_BYTES
                buffers[role].extend(chunk[: max(0, limit + 1 - len(buffers[role]))])
                if role == "result" and len(buffers[role]) > MAX_RESULT_BYTES:
                    error = error or "oracle worker result exceeded its parent-side bound"
                elif role != "result":
                    snippet = bytes(buffers[role][:MAX_SIDE_BYTES]).decode("utf-8", "replace")
                    error = error or f"oracle worker emitted a byte on {role}: {snippet}"
                if error: break
            if error: break
            if process.poll() is not None and not selector.get_map(): break
        if error:
            _kill_tracked(tracked)
        else:
            process.wait(timeout=POST_KILL_SECONDS)
            _discover_descendants(process.pid, tracked)
            # A successful direct child is not allowed to leave any live child.
            live_descendants = [pid for pid in tracked if pid != process.pid]
            if live_descendants:
                error = "oracle worker left a live descendant"
                _kill_tracked(tracked)
            else:
                _kill_tracked(tracked)
    finally:
        for key in list(selector.get_map().values()):
            try: selector.unregister(key.fd)
            except Exception: pass
        selector.close()
        for stream in (process.stdout, process.stderr):
            if stream is not None: stream.close()
        try: os.close(result_fd)
        except OSError: pass
    if error: raise WorkerBlocked(error)
    assert process.returncode is not None
    return process.returncode, bytes(buffers["result"])


def _close(fd: int | None) -> None:
    if fd is not None:
        try: os.close(fd)
        except OSError: pass


def _launch(group: str, evidence_root: Path, timeout: float, *, test_fault: str | None) -> tuple[int, dict[str, Any]]:
    if group not in ("BASE-01", "BASE-02") or timeout <= 0:
        raise WorkerBlocked("invalid group/timeout")
    if os.environ.get("DG_ENGINE_I0_ORACLE_WORKER") is not None:
        raise WorkerBlocked("nested oracle worker marker is forbidden")
    commit, tree = _candidate_identity()
    harness, harness_bytes = _load_harness_from_candidate(commit)
    _validate_harness(harness, commit)
    boundary = test_fault is not None
    boundary_name = harness["boundary_test"]["environment"]
    if boundary:
        if os.environ.get(boundary_name) != "1" or test_fault not in harness["boundary_test"]["faults"]:
            raise WorkerBlocked("test fault is outside the dedicated boundary-test contract")
    source_fds, source_blobs, source_digest = _open_candidate_sources(harness, commit)
    try:
        reference_bytes = source_blobs["reference-lock"]
        reference_lock = _strict_json_bytes(reference_bytes, "reference lock", canonical=False)
        evidence_root = evidence_root.resolve(strict=True)
        if not evidence_root.is_dir():
            raise WorkerBlocked("evidence root is not a directory")
        launcher, image = _authenticate_interpreter(harness)
        environment = _verify_environment(harness, cache_for_test=boundary)
        evidence_manifest = evidence_root / reference_lock["evidence_root_contract"]["manifest"]
        evidence_manifest_fd = _open_regular(evidence_manifest, "evidence manifest")
        try: evidence_manifest_hash = _hash_fd(evidence_manifest_fd)[1]
        finally: os.close(evidence_manifest_fd)
    except BaseException:
        for fd in source_fds.values(): _close(fd)
        source_fds.clear()
        raise
    nonce = secrets.token_hex(32)
    image_fd = harness_fd = read_fd = write_fd = duplicate_fd = regular_fd = None
    process: subprocess.Popen[bytes] | None = None
    try:
        image_fd = _open_regular(Path(image["path"]), "interpreter image for spawn")
        image_stat = os.fstat(image_fd); image_size, image_hash = _hash_fd(image_fd)
        if (image_stat.st_dev, image_stat.st_ino, image_size, image_hash) != (
            image["dev"], image["ino"], image["size"], image["sha256"]
        ): raise WorkerBlocked("spawn image differs from lock")
        harness_fd = _memfd("dg-i0-harness", harness_bytes)
        read_fd, write_fd = os.pipe2(getattr(os, "O_CLOEXEC", 0))
        child_result_fd = write_fd
        pass_fds = [image_fd, harness_fd, write_fd, *source_fds.values()]
        command = [f"/proc/self/fd/{image_fd}", "-I", "-S", "-B", "-u", f"/proc/self/fd/{source_fds['oracle-worker']}",
                   "--group", group, "--result-fd", str(child_result_fd), "--interpreter-fd", str(image_fd),
                   "--harness-fd", str(harness_fd), "--nonce", nonce, "--repo-root", str(REPO_ROOT.resolve()),
                   "--evidence-root", str(evidence_root), "--candidate-commit", commit, "--candidate-tree", tree,
                   "--source-set-sha256", source_digest, "--environment-merkle", environment["merkle"],
                   "--environment-manifest", environment["manifest_path"], "--launcher", str(launcher),
                   "--test-fault", test_fault or "none"]
        for source_id, fd in sorted(source_fds.items()): command.extend(["--bound-file", f"{source_id}:{fd}"])
        env = {"DG_ENGINE_I0_ORACLE_WORKER": nonce}
        if boundary: env[boundary_name] = "1"
        # Parent-owned faults exercise spawn/fd argument boundaries without an unrestricted factory.
        if test_fault == "spawn-failure":
            raise OSError("boundary injected spawn failure")
        if test_fault == "worker-substitution":
            command[5] = f"/proc/self/fd/{source_fds['reference-lock']}"
        elif test_fault == "worker-fd-substitution":
            for index in range(len(command) - 1):
                if command[index] == "--bound-file" and command[index + 1].startswith("oracle-worker:"):
                    command[index + 1] = f"oracle-worker:{source_fds['base-01-tests']}"
                    break
        elif test_fault == "wrong-worker-argument": command[command.index("--repo-root") + 1] = str(TESTS)
        elif test_fault == "wrong-environment-root": command[command.index("--environment-manifest") + 1] = str(REFERENCE_LOCK_PATH)
        elif test_fault == "wrong-interpreter-image": command[command.index("--launcher") + 1] = str(GIT_PATH)
        elif test_fault == "wrong-nonce-argument": command[command.index("--nonce") + 1] = "0" * 64
        elif test_fault == "result-fd-read-only":
            child_result_fd = read_fd
            command[command.index("--result-fd") + 1] = str(child_result_fd)
            pass_fds.append(read_fd)
        elif test_fault == "result-fd-duplicate":
            duplicate_fd = os.dup(write_fd); pass_fds.append(duplicate_fd)
        elif test_fault == "result-fd-regular":
            regular_fd = os.open("/dev/null", os.O_WRONLY | os.O_CLOEXEC)
            command[command.index("--result-fd") + 1] = str(regular_fd); pass_fds.append(regular_fd)
        if _candidate_identity() != (commit, tree):
            raise WorkerBlocked("candidate HEAD/tree or cleanliness changed before spawn")
        _set_subreaper()
        try:
            process = subprocess.Popen(
                command, executable=f"/proc/self/fd/{image_fd}", cwd=REPO_ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=tuple(dict.fromkeys(pass_fds)), close_fds=True, start_new_session=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise WorkerBlocked(f"oracle worker spawn failed: {error}") from error
        for fd in [image_fd, harness_fd, write_fd, duplicate_fd, regular_fd, *source_fds.values()]: _close(fd)
        image_fd = harness_fd = write_fd = duplicate_fd = regular_fd = None
        source_fds.clear()
        code, wire = _drain_process(process, read_fd, timeout)
        read_fd = None
    except OSError as error:
        if test_fault == "spawn-failure":
            raise WorkerBlocked(f"oracle worker spawn failed: {error}") from error
        raise
    finally:
        for fd in [image_fd, harness_fd, read_fd, write_fd, duplicate_fd, regular_fd, *source_fds.values()]: _close(fd)
        if process is not None and process.poll() is None:
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            try: process.wait(timeout=POST_KILL_SECONDS)
            except subprocess.TimeoutExpired: pass
    if not wire: raise WorkerBlocked(f"oracle worker exited {code} without a result")
    if len(wire) > MAX_RESULT_BYTES: raise WorkerBlocked("oracle worker result exceeded its parent-side bound")
    value = _strict_json_bytes(wire, "worker result")
    expected = {"candidate_commit": commit, "candidate_tree": tree,
                "reference_lock_sha256": _sha256_bytes(reference_bytes), "harness_lock_sha256": _sha256_bytes(harness_bytes),
                "evidence_manifest_sha256": evidence_manifest_hash,
                "environment_merkle_sha256": environment["merkle"], "interpreter_sha256": image["sha256"],
                "worker_sha256": _source_map(harness)["oracle-worker"]["sha256"], "test_source_set_sha256": source_digest,
                "launcher": str(launcher), "interpreter_image": image["path"], "environment_manifest": environment["manifest_path"]}
    expected_exit = _validate_result(value, group=group, reference_lock=reference_lock, harness=harness,
                                     expected=expected, evidence_root=evidence_root, nonce=nonce)
    if code != expected_exit:
        raise WorkerBlocked(f"oracle worker exit code {code} differs from authenticated disposition {expected_exit}")
    return code, value


def launch_worker(group: str, evidence_root: Path, *, timeout: float) -> tuple[int, dict[str, Any]]:
    """Run the only production launcher: committed candidate, direct Popen, no faults."""
    return _launch(group, evidence_root, timeout, test_fault=None)


def launch_worker_test_fault(group: str, evidence_root: Path, *, timeout: float, fault: str) -> None:
    """Run one enumerated boundary fault; this entry can never certify success."""
    if not fault or fault == "none":
        raise WorkerBlocked("test-only launcher requires a non-production fault")
    try:
        _launch(group, evidence_root, timeout, test_fault=fault)
    except WorkerBlocked:
        raise
    raise WorkerBlocked("test-only launcher unexpectedly reached a production-valid result")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group", choices=("BASE-01", "BASE-02"))
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--worker-timeout", type=float)
    args = parser.parse_args(argv)
    timeout = args.worker_timeout if args.worker_timeout is not None else (180.0 if args.group == "BASE-01" else 3300.0)
    try:
        code, result = launch_worker(args.group, args.evidence_root, timeout=timeout)
    except (WorkerBlocked, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"BLOCKED: {args.group}: oracle worker boundary: {error}", file=sys.stderr)
        return 2
    counts = result["counts"]
    print(f"SOURCE_SUBTESTS: group={args.group} collected={counts['collected']} executed={counts['executed']} "
          "failed=0 errors=0 skipped=0 expected_failures=0 unexpected_successes=0")
    print(f"WORKER_BOUNDARY: group={args.group} candidate_tree=PASS environment=PASS retained_bytes=PASS "
          f"dedicated_pipes=PASS disposition={result['disposition']}")
    if args.group == "BASE-02":
        gate = result["gate"]
        print(f"GATE_EVIDENCE: group=BASE-02 executed={gate['artifact_executed']} lineage_executed={gate['lineage_executed']} "
              f"artifact_or_lineage_blocked={gate['blocked']} expected_scope=14+3+1 policy_blocked=1")
        for blocker in result["blockers"]:
            print(f"BLOCKED: {blocker['category']}: {blocker['blocker']}: {blocker['artifact_id']}: {blocker['detail']}", file=sys.stderr)
        print("BLOCKED: BASE-02 is not PASS; source/synthetic subtests do not satisfy the gate.", file=sys.stderr)
    else:
        print("GROUP_RESULT: BASE-01=PASS")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
