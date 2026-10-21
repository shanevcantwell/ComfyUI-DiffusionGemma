"""Pure helpers and fail-closed evidence readers for I0 baseline groups."""
from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass, field
import builtins
import dis
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import threading
from types import BuiltinFunctionType, BuiltinMethodType, CodeType, FunctionType, MethodType, ModuleType
import unicodedata
from typing import Any, Callable

ENGINE_DESIGN = Path(__file__).resolve().parents[1]
REPO_ROOT = ENGINE_DESIGN.parent
FIXTURES = ENGINE_DESIGN / "fixtures" / "i0"
SCHEMAS = ENGINE_DESIGN / "schemas" / "i0"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STREAM_CHUNK_BYTES = 1024 * 1024
# The largest byte-returning source is the pinned 32.2 MB tokenizer.  A 64 MiB
# ceiling admits that source while making accidental checkpoint/GGUF allocation
# structurally impossible; large identities use capture_limit=None and stream.
MAX_RETURNED_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_DERIVATION_RECORD_BYTES = 1024 * 1024
MAX_COMMAND_LOG_BYTES = 1024 * 1024
MAX_ORACLE_SOURCE_BYTES = 64 * 1024 * 1024
# Oracle closure walks are process-local integrity evidence, not an unbounded
# object-graph claim.  Every structural/global walk shares these deterministic
# hard limits and converts exhaustion to I0-BLOCK-ORACLE-BINDING.
MAX_ORACLE_DEPENDENCY_DEPTH = 96
MAX_ORACLE_DEPENDENCY_NODES = 100_000
MAX_ORACLE_SINGLE_CONTAINER_ITEMS = 20_000
MAX_ORACLE_CONTAINER_ITEMS = 200_000
MAX_ORACLE_ENCODED_BYTES = 32 * 1024 * 1024
MAX_ORACLE_SINGLE_CODE_BYTES = 1024 * 1024
MAX_ORACLE_BYTECODE_STATES = 100_000
MAX_ORACLE_FINGERPRINT_STABILIZATION_PASSES = 4
# Finite source-derived deep-mutable manifest.  Module/class/function bindings
# and full bytecode attribute chains are always fingerprinted; only these
# selected mutable chain values are recursively content-addressed rather than
# identity-bound.  Additions require an explicit reviewed code/test update.
ORACLE_DEEP_MUTABLE_DEPENDENCIES = frozenset({
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
})
# Selected inherited implementations reached by authenticated product/source
# calls but not necessarily by a trusted class's own receiver bytecode.  This
# is deliberately finite: broad base-class dictionaries contain mutable
# framework caches that are not oracle behavior.  Each selected implementation
# is dependency-closed (code/defaults/closure/globals and selected receiver
# paths) by the same bytecode walker as directly declared methods.
ORACLE_SELECTED_INHERITED_METHOD_DEPENDENCIES = frozenset({
    (
        "diffusers.pipelines.diffusion_gemma.pipeline_diffusion_gemma",
        "DiffusionGemmaPipeline",
        "register_modules",
    ),
    (
        "diffusers.pipelines.diffusion_gemma.pipeline_diffusion_gemma",
        "DiffusionGemmaPipeline",
        "progress_bar",
    ),
    (
        "transformers.models.diffusion_gemma.modeling_diffusion_gemma",
        "DiffusionGemmaForBlockDiffusion",
        "from_pretrained",
    ),
})
ORACLE_CONSTRUCTOR_DEPENDENCIES = frozenset({
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
})

I0_CHECKPOINT_IDS = tuple(
    f"official.checkpoint.{index:05d}-of-00011" for index in range(1, 12)
)
I0_GGUF_KINDS = {
    "gguf.bf16": "GGUF_BF16",
    "gguf.q8_0": "GGUF_Q8_0",
    "gguf.q4_k_m": "GGUF_Q4_K_M",
}
I0_GGUF_REMOTE_SUFFIXES = {
    "gguf.bf16": "BF16.gguf",
    "gguf.q8_0": "Q8_0.gguf",
    "gguf.q4_k_m": "Q4_K_M.gguf",
}
I0_GATE_CATEGORIES = {
    "checkpoint_identity": {"expected": 11, "id_prefix": "official.checkpoint."},
    "gguf_identity": {"expected": 3, "id_prefix": "gguf."},
    "gguf_authenticated_lineage": {"expected": 3, "id_prefix": "gguf."},
}
I0_GATE_REQUIRED_FOR = ["BASE-02-gate"]
I0_EVIDENCE_ARTIFACT_SET_SHA256 = "b66b5081df7e9a2ccc58dcdd00008e8a7c051021548fe76b7bd6f966426dbfbe"
EVIDENCE_DESCRIPTOR_FIELDS = (
    "id",
    "path",
    "size",
    "sha256",
    "status",
    "required_for",
    "kind",
)
AUTOROUND_DISTRIBUTION_POLICY = {
    "status": "EXCLUDED",
    "artifact_inclusion": "FORBIDDEN",
    "release_qualification": "NOT_RELIED_ON",
    "evidence_artifact_ids": [],
    "future_reentry_requirement": "EXPLICIT_SCOPE_AND_LICENSE_DECISION_REQUIRED",
}
AUTOROUND_DISTRIBUTION_DISPOSITION = (
    "EXCLUDED_FROM_NATIVE_ENGINE_DISTRIBUTION_AND_RELEASE_QUALIFICATION"
)
I0_UNRESOLVED_BLOCKER_CLASSIFICATIONS = {
    "I0-BLOCK-CHECKPOINT-EVIDENCE": "evidence",
    "I0-BLOCK-GGUF-EVIDENCE": "evidence",
    "I0-BLOCK-ADR-CDG-020": "phase-policy",
}
I0_PHASE_POLICY_BLOCKERS = frozenset(("I0-BLOCK-ADR-CDG-020",))
CANONICAL_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}", re.ASCII)
I0_DERIVATION_ID_PREFIXES = {
    "gguf.bf16": "gguf.bf16.",
    "gguf.q8_0": "gguf.q8_0.",
    "gguf.q4_k_m": "gguf.q4_k_m.",
}
BARE_PRODUCT_ROOTS = frozenset(("dgemma", "surfaces"))
AUTHENTICATED_CONTEXT_ACQUIRE_TIMEOUT = 5.0
_AUTHENTICATED_CONTEXT_LOCK = threading.Lock()
_AUTHENTICATED_CONTEXT_STATE_LOCK = threading.Lock()
_AUTHENTICATED_CONTEXT_OWNER_THREAD: int | None = None
_AUTHENTICATED_CONTEXT_OWNER_TOKEN: object | None = None
_ORACLE_TRUST_SNAPSHOT: _OracleTrustSnapshot | None = None
_MISSING = object()


def _identity_sequence_matches(observed: object, expected: tuple[Any, ...]) -> bool:
    """Compare an import-state sequence without invoking finder equality hooks."""
    try:
        if len(observed) != len(expected):  # type: ignore[arg-type]
            return False
        return all(observed[index] is item for index, item in enumerate(expected))  # type: ignore[index]
    except (IndexError, TypeError):
        return False


def _remove_identity_from_meta_path(target: object) -> None:
    """Remove every exact target occurrence without calling any finder.__eq__."""
    sys.meta_path[:] = [finder for finder in sys.meta_path if finder is not target]

# Every export below is imported by the authenticated product at module scope.
# The exact source module is part of the trust statement: package-root lazy
# exports are not sufficient unless they resolve to these same class objects.
ORACLE_EXPORT_BINDINGS = (
    {
        "package": "diffusers",
        "export": "DiffusionGemmaPipeline",
        "module": "diffusers.pipelines.diffusion_gemma.pipeline_diffusion_gemma",
        "source": "src/diffusers/pipelines/diffusion_gemma/pipeline_diffusion_gemma.py",
        "product_module": "dgemma.loop",
    },
    {
        "package": "diffusers",
        "export": "EntropyBoundScheduler",
        "module": "diffusers.schedulers.scheduling_entropy_bound",
        "source": "src/diffusers/schedulers/scheduling_entropy_bound.py",
        "product_module": "dgemma.loop",
    },
    {
        "package": "transformers",
        "export": "AutoProcessor",
        "module": "transformers.models.auto.processing_auto",
        "source": "src/transformers/models/auto/processing_auto.py",
        "product_module": "dgemma.model",
    },
    {
        "package": "transformers",
        "export": "DiffusionGemmaForBlockDiffusion",
        "module": "transformers.models.diffusion_gemma.modeling_diffusion_gemma",
        "source": "src/transformers/models/diffusion_gemma/modeling_diffusion_gemma.py",
        "product_module": "dgemma.model",
    },
)


def _canonical_id(value: object, blocker: str, artifact_id: str, role: str) -> str:
    """Return an ASCII-lowercase identifier or fail before alias comparison."""
    if not isinstance(value, str) or CANONICAL_ID_RE.fullmatch(value) is None:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            f"{role} must use canonical lowercase ASCII [a-z0-9][a-z0-9._-]* form",
        )
    normalized = unicodedata.normalize("NFKC", value).casefold()
    if value != normalized:
        raise EvidenceBlocked(blocker, artifact_id, f"{role} is not normalization-canonical")
    return normalized


class EvidenceBlocked(RuntimeError):
    """A required evidence artifact is absent or cannot be authenticated."""

    def __init__(self, blocker: str, artifact_id: str, detail: str) -> None:
        self.blocker = blocker
        self.artifact_id = artifact_id
        self.detail = detail
        super().__init__(f"{blocker}: {artifact_id}: {detail}")


@dataclass
class EvidenceState:
    root: Path
    root_fd: int
    root_identity: tuple[int, int]
    expected: dict[str, dict[str, Any]]
    observed: dict[str, dict[str, Any]]

    def check_root_consistency(self) -> None:
        """Diagnose configured-path replacement without abandoning the anchored root fd."""
        directory_flags, nofollow, _nonblock = _required_open_flags()
        diagnostic_fd = -1
        try:
            diagnostic_fd = os.open(self.root, directory_flags | nofollow)
            current = os.fstat(diagnostic_fd)
            if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != self.root_identity:
                raise EvidenceBlocked(
                    "I0-BLOCK-EVIDENCE-ROOT",
                    "evidence-root",
                    "configured root path was replaced during verification; anchored reads were retained",
                )
        except EvidenceBlocked:
            raise
        except OSError as error:
            raise EvidenceBlocked(
                "I0-BLOCK-EVIDENCE-ROOT",
                "evidence-root",
                f"configured root path became unavailable during verification: {error.strerror}",
            ) from error
        finally:
            _close_quietly(diagnostic_fd)

    def close(self) -> None:
        if self.root_fd >= 0:
            root_fd, self.root_fd = self.root_fd, -1
            os.close(root_fd)

    def __enter__(self) -> EvidenceState:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True)
class AuthenticatedFile:
    path: str
    file_identity: tuple[int, int]
    size: int
    sha256: str
    data: bytes | None


@dataclass
class _DerivationAliases:
    paths: dict[str, str] = field(default_factory=dict)
    files: dict[tuple[int, int], str] = field(default_factory=dict)
    descriptor_ids: dict[str, str] = field(default_factory=dict)

    def claim_descriptor_id(self, descriptor_id: str, owner: str, blocker: str, artifact_id: str) -> None:
        canonical = _canonical_id(descriptor_id, blocker, artifact_id, "evidence descriptor ID")
        previous = self.descriptor_ids.get(canonical)
        if previous is not None:
            raise EvidenceBlocked(
                blocker,
                artifact_id,
                f"evidence descriptor IDs must be globally unique; {owner} aliases {previous}",
            )
        self.descriptor_ids[canonical] = owner

    def claim_path(self, path: str, owner: str, blocker: str, artifact_id: str) -> None:
        _canonical_parts(path, artifact_id)
        previous = self.paths.get(path)
        if previous is not None:
            raise EvidenceBlocked(
                blocker,
                artifact_id,
                f"derivation paths must be pairwise distinct across roles; {owner} aliases {previous}",
            )
        self.paths[path] = owner

    def claim_file(self, authenticated: AuthenticatedFile, owner: str, blocker: str, artifact_id: str) -> None:
        previous = self.files.get(authenticated.file_identity)
        if previous is not None:
            raise EvidenceBlocked(
                blocker,
                artifact_id,
                f"derivation files must be inode-distinct across roles; {owner} aliases {previous}",
            )
        self.files[authenticated.file_identity] = owner


@dataclass(frozen=True)
class GateVerification:
    artifact_executed: int
    lineage_executed: int
    categories: dict[str, int]
    blockers: list[EvidenceBlocked]


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_loads(data: bytes | str) -> Any:
    return json.loads(data, object_pairs_hook=_reject_duplicate_pairs)


def read_json(path: Path) -> Any:
    return _strict_json_loads(path.read_bytes())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_json_data(instance: Any, schema_path: Path) -> None:
    from jsonschema import Draft202012Validator

    schema = read_json(schema_path)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(instance)


def validate_json(instance_path: Path, schema_path: Path) -> None:
    """Validate exactly the in-memory bytes read from instance_path."""
    validate_json_data(_strict_json_loads(instance_path.read_bytes()), schema_path)


def _artifact_set_sha256(entries: list[dict[str, Any]]) -> str:
    """Hash the complete, ID-sorted Git-controlled evidence identity set."""
    encoded = json.dumps(
        sorted(entries, key=lambda entry: entry["id"]),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _evidence_identity_map(
    entries: object, blocker: str, artifact_id: str
) -> dict[str, dict[str, Any]]:
    if not isinstance(entries, list):
        raise EvidenceBlocked(blocker, artifact_id, "evidence identities must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise EvidenceBlocked(blocker, artifact_id, "every evidence identity must have a string id")
        if entry["id"] in by_id:
            raise EvidenceBlocked(blocker, entry["id"], "duplicate artifact id")
        by_id[entry["id"]] = entry
    return by_id


def _validate_frozen_evidence_contract(lock: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Require the reviewed I0 identity set, not a caller-coordinated replacement."""
    blocker = "I0-BLOCK-EVIDENCE-CONTRACT"
    artifact_id = "evidence-contract"
    try:
        contract = lock["evidence_root_contract"]
        entries = contract["artifacts"]
        declared_digest = contract["artifact_set_sha256"]
    except (KeyError, TypeError) as error:
        raise EvidenceBlocked(blocker, artifact_id, f"frozen contract field is missing: {error}") from error
    expected = _evidence_identity_map(entries, blocker, artifact_id)
    try:
        observed_digest = _artifact_set_sha256(entries)
    except (KeyError, TypeError, ValueError) as error:
        raise EvidenceBlocked(blocker, artifact_id, f"contract identity set is malformed: {error}") from error
    if declared_digest != I0_EVIDENCE_ARTIFACT_SET_SHA256 or observed_digest != declared_digest:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "evidence identity set or immutable descriptor changed; a reviewed lock/schema update is required",
        )
    return expected


def _validate_manifest_against_lock(
    lock: dict[str, Any], manifest: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Bind every manifest descriptor to exactly one complete lock identity."""
    blocker = "I0-BLOCK-EVIDENCE-MANIFEST"
    artifact_id = "evidence-manifest"
    contract = lock["evidence_root_contract"]
    expected = _evidence_identity_map(contract["artifacts"], "I0-BLOCK-EVIDENCE-CONTRACT", "evidence-contract")
    observed = _evidence_identity_map(manifest.get("artifacts"), blocker, artifact_id)
    expected_ids, observed_ids = set(expected), set(observed)
    if observed_ids != expected_ids:
        missing = sorted(expected_ids - observed_ids)
        extra = sorted(observed_ids - expected_ids)
        detail = f"artifact ID set differs from the lock (missing={missing}, extra={extra})"
        raise EvidenceBlocked(blocker, artifact_id, detail)
    if manifest.get("contract_artifact_set_sha256") != contract.get("artifact_set_sha256"):
        raise EvidenceBlocked(blocker, artifact_id, "manifest contract fingerprint differs from the lock")
    for name in sorted(expected):
        identity = expected[name]
        descriptor = observed[name]
        is_gguf = name in I0_GGUF_KINDS
        # This rejection deliberately precedes all content/keyword policy
        # checks.  A neutral-named nested payload on an allowed source ID is
        # still undeclared evidence and cannot inherit that ID's authority.
        if not is_gguf and "derivation" in descriptor:
            raise EvidenceBlocked(
                blocker,
                name,
                "derivation is forbidden for every non-GGUF evidence identity",
            )
        allowed_fields = {*EVIDENCE_DESCRIPTOR_FIELDS, "blocker"}
        if is_gguf:
            allowed_fields.add("derivation")
        extra_fields = sorted(set(descriptor) - allowed_fields)
        if extra_fields:
            raise EvidenceBlocked(
                blocker,
                name,
                f"manifest descriptor has fields not bound by the lock: {extra_fields}",
            )
        for field_name in EVIDENCE_DESCRIPTOR_FIELDS:
            if descriptor.get(field_name, _MISSING) != identity.get(field_name, _MISSING):
                raise EvidenceBlocked(
                    blocker,
                    name,
                    f"manifest {field_name} differs from the immutable lock identity",
                )
        if descriptor.get("blocker", _MISSING) != identity.get("blocker", _MISSING):
            raise EvidenceBlocked(blocker, name, "manifest blocker differs from the immutable lock identity")
        if is_gguf:
            lineage_lock = identity.get("lineage_lock")
            lineage_bound = (
                identity.get("lineage_status") == "BOUND"
                and isinstance(lineage_lock, dict)
                and lineage_lock.get("status") == "BOUND"
            )
            has_derivation = "derivation" in descriptor
            if has_derivation != lineage_bound:
                detail = (
                    "manifest derivation is present while the GGUF lineage is UNBOUND"
                    if not lineage_bound
                    else "manifest derivation is absent while the GGUF lineage is BOUND"
                )
                raise EvidenceBlocked(blocker, name, detail)
            if lineage_bound:
                expected_derivation = {
                    "record": lineage_lock.get("record"),
                    "command_log": lineage_lock.get("command_log"),
                }
                if descriptor["derivation"] != expected_derivation:
                    raise EvidenceBlocked(
                        blocker,
                        name,
                        "manifest derivation descriptor does not match the Git-controlled immutable GGUF lineage lock",
                    )
    return expected, observed


def validate_unresolved_blocker_model(lock: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the complete I0 blocker taxonomy and return phase-policy blockers."""
    blocker = "I0-BLOCK-UNRESOLVED-BLOCKER-CONTRACT"
    artifact_id = "unresolved-blockers"
    entries = lock.get("unresolved_blockers")
    if not isinstance(entries, list):
        raise EvidenceBlocked(blocker, artifact_id, "unresolved_blockers must be an array")
    observed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvidenceBlocked(blocker, artifact_id, "every unresolved blocker must be an object")
        if set(entry) != {"id", "classification", "reason"}:
            raise EvidenceBlocked(blocker, artifact_id, "unresolved blocker fields are not exact")
        blocker_id = entry.get("id")
        if not isinstance(blocker_id, str) or blocker_id in observed:
            raise EvidenceBlocked(blocker, artifact_id, "unresolved blocker IDs must be unique strings")
        if not isinstance(entry.get("reason"), str) or not entry["reason"]:
            raise EvidenceBlocked(blocker, blocker_id, "unresolved blocker reason must be nonempty")
        observed[blocker_id] = entry
    if set(observed) != set(I0_UNRESOLVED_BLOCKER_CLASSIFICATIONS):
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "complete unresolved blocker ID set differs from the authoritative I0 model",
        )
    for blocker_id, classification in I0_UNRESOLVED_BLOCKER_CLASSIFICATIONS.items():
        if observed[blocker_id]["classification"] != classification:
            raise EvidenceBlocked(
                blocker,
                blocker_id,
                f"unresolved blocker classification must be {classification}",
            )
    policy = [observed[blocker_id] for blocker_id in sorted(I0_PHASE_POLICY_BLOCKERS)]
    if {item["id"] for item in policy} != set(I0_PHASE_POLICY_BLOCKERS):
        raise EvidenceBlocked(blocker, artifact_id, "phase-policy blocker set is not exact")
    return policy


def validate_autoround_distribution_policy(
    lock: dict[str, Any],
    evidence_entries: list[dict[str, Any]] | None = None,
    *,
    _freeze_contract: bool = True,
) -> None:
    """Require inventory-only AutoRound and reject it from native release evidence."""
    blocker = "I0-BLOCK-AUTOROUND-DISTRIBUTION-POLICY"
    artifact_id = "autoround-distribution-policy"
    try:
        derivative = lock["autoround_derivative"]
        policy = derivative["native_engine_distribution"]
        unresolved_ids = {item["id"] for item in lock["unresolved_blockers"]}
        contract_entries = lock["evidence_root_contract"]["artifacts"]
    except (KeyError, TypeError) as error:
        raise EvidenceBlocked(blocker, artifact_id, f"required exclusion field is missing: {error}") from error

    if _freeze_contract:
        try:
            _validate_frozen_evidence_contract(lock)
            if evidence_entries is not None:
                _validate_manifest_against_lock(lock, {
                    "contract_artifact_set_sha256": lock["evidence_root_contract"]["artifact_set_sha256"],
                    "artifacts": evidence_entries,
                })
        except EvidenceBlocked as error:
            raise EvidenceBlocked(
                blocker,
                error.artifact_id,
                f"{error.detail}; a future explicit scope and license decision requires reviewed lock/schema changes",
            ) from error

    if derivative.get("mode_status") != "declared-but-broken":
        raise EvidenceBlocked(blocker, artifact_id, "AutoRound must remain inventoried as declared-but-broken")
    if derivative.get("distribution_disposition") != AUTOROUND_DISTRIBUTION_DISPOSITION:
        raise EvidenceBlocked(blocker, artifact_id, "native distribution disposition is not EXCLUDED")
    if policy != AUTOROUND_DISTRIBUTION_POLICY:
        raise EvidenceBlocked(blocker, artifact_id, "native distribution/release exclusion policy changed")
    if "I0-BLOCK-AUTOROUND-LICENSE" in unresolved_ids:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "license uncertainty cannot block this phase while AutoRound model/assets are excluded",
        )

    # Marker detection is intentionally secondary.  Exact ID-set and immutable
    # descriptor binding above are the authorization boundary; these markers
    # only improve diagnostics for conspicuous attempts.
    repository = str(derivative.get("repository", "")).casefold()
    revision = str(derivative.get("revision", "")).casefold()
    forbidden_markers = ("autoround", repository, revision)
    for source, entries in (("reference lock", contract_entries), ("evidence manifest", evidence_entries or [])):
        for entry in entries:
            rendered = json.dumps(entry, sort_keys=True).casefold()
            if any(marker and marker in rendered for marker in forbidden_markers):
                raise EvidenceBlocked(
                    blocker,
                    str(entry.get("id", "unknown-artifact")),
                    f"{source} attempts to include the excluded AutoRound model/asset; "
                    "a future explicit scope and license decision is required",
                )


def evidence_root() -> Path:
    value = os.environ.get("DG_ENGINE_I0_EVIDENCE_ROOT", "")
    if not value:
        raise EvidenceBlocked("I0-BLOCK-EVIDENCE-ROOT", "evidence-root", "DG_ENGINE_I0_EVIDENCE_ROOT is unset")
    # Keep the configured path spelling (made absolute, not symlink-resolved) so
    # O_NOFOLLOW and the end check diagnose replacement of that actual root path.
    return Path(value).absolute()


def _required_open_flags() -> tuple[int, int, int]:
    missing = [name for name in ("O_NOFOLLOW", "O_NONBLOCK") if not getattr(os, name, 0)]
    if missing:
        raise EvidenceBlocked(
            "I0-BLOCK-EVIDENCE-PLATFORM",
            "secure-open-flags",
            f"required secure open flags are unavailable: {', '.join(missing)}",
        )
    try:
        directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    except AttributeError as error:
        raise EvidenceBlocked(
            "I0-BLOCK-EVIDENCE-PLATFORM",
            "secure-open-flags",
            f"required directory open flag is unavailable: {error}",
        ) from error
    return directory_flags, os.O_NOFOLLOW, os.O_NONBLOCK


def _close_quietly(fd: int) -> None:
    if fd >= 0:
        try:
            os.close(fd)
        except OSError:
            pass


def _canonical_parts(name: str, artifact_id: str) -> tuple[str, ...]:
    part = PurePosixPath(name)
    if (
        not name
        or not part.parts
        or part.is_absolute()
        or ".." in part.parts
        or "." in part.parts
        or "\\" in name
        or str(part) != name
        or any(not value for value in part.parts)
    ):
        raise EvidenceBlocked("I0-BLOCK-EVIDENCE-PATH", artifact_id, "unsafe or noncanonical relative path")
    return part.parts


def _authenticate_confined_file(
    root_fd: int,
    name: str,
    artifact_id: str,
    *,
    expected_size: int | None = None,
    expected_sha256: str | None = None,
    capture_limit: int | None = None,
) -> AuthenticatedFile:
    """Hash one descriptor in bounded chunks, optionally retaining only explicitly bounded bytes."""
    parts = _canonical_parts(name, artifact_id)
    directory_flags, nofollow, nonblock = _required_open_flags()
    current_fd = -1
    file_fd = -1
    try:
        current_fd = os.dup(root_fd)
        for component in parts[:-1]:
            next_fd = os.open(component, directory_flags | nofollow, dir_fd=current_fd)
            try:
                os.close(current_fd)
            except BaseException:
                _close_quietly(next_fd)
                raise
            current_fd = next_fd
        file_fd = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | nofollow | nonblock,
            dir_fd=current_fd,
        )
        try:
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise EvidenceBlocked("I0-BLOCK-EVIDENCE-PATH", artifact_id, "artifact is not a regular file")
            if expected_size is not None and before.st_size != expected_size:
                raise EvidenceBlocked("I0-BLOCK-EVIDENCE-SIZE", artifact_id, "descriptor/file size differs")
            if capture_limit is not None and before.st_size > capture_limit:
                raise EvidenceBlocked(
                    "I0-BLOCK-EVIDENCE-SIZE",
                    artifact_id,
                    f"artifact exceeds bounded byte-read limit ({capture_limit} bytes)",
                )

            digest = hashlib.sha256()
            count = 0
            captured = bytearray() if capture_limit is not None else None
            while True:
                chunk = os.read(file_fd, STREAM_CHUNK_BYTES)
                if not chunk:
                    break
                count += len(chunk)
                digest.update(chunk)
                if captured is not None:
                    if count > capture_limit:
                        raise EvidenceBlocked(
                            "I0-BLOCK-EVIDENCE-SIZE", artifact_id, "artifact grew beyond bounded byte-read limit"
                        )
                    captured.extend(chunk)

            after = os.fstat(file_fd)
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if before_identity != after_identity or count != before.st_size:
                raise EvidenceBlocked("I0-BLOCK-EVIDENCE-TOCTOU", artifact_id, "artifact changed while being read")
            observed_hash = digest.hexdigest()
            if expected_sha256 is not None and observed_hash != expected_sha256:
                raise EvidenceBlocked("I0-BLOCK-EVIDENCE-HASH", artifact_id, "descriptor hash differs from bytes")
            return AuthenticatedFile(
                path=name,
                file_identity=(before.st_dev, before.st_ino),
                size=count,
                sha256=observed_hash,
                data=None if captured is None else bytes(captured),
            )
        finally:
            if file_fd >= 0:
                closing_fd, file_fd = file_fd, -1
                os.close(closing_fd)
    except EvidenceBlocked:
        raise
    except OSError as error:
        raise EvidenceBlocked("I0-BLOCK-EVIDENCE-PATH", artifact_id, f"safe open failed: {error.strerror}") from error
    finally:
        _close_quietly(file_fd)
        _close_quietly(current_fd)


def _parse_and_validate(data: bytes, schema_path: Path, blocker: str, artifact_id: str) -> Any:
    try:
        instance = _strict_json_loads(data)
        validate_json_data(instance, schema_path)
        return instance
    except Exception as error:
        raise EvidenceBlocked(blocker, artifact_id, f"JSON/schema validation failed: {error}") from error


def _evidence_state(lock: dict[str, Any]) -> EvidenceState:
    # Reader-level validation binds the complete observed manifest to the lock
    # supplied by its caller.  Promotion gates separately require that lock to
    # match the frozen reviewed I0 contract.
    validate_autoround_distribution_policy(lock, _freeze_contract=False)
    root = evidence_root()
    directory_flags, nofollow, _nonblock = _required_open_flags()
    root_fd = -1
    try:
        root_fd = os.open(root, directory_flags | nofollow)
        root_stat = os.fstat(root_fd)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise EvidenceBlocked("I0-BLOCK-EVIDENCE-ROOT", "evidence-root", "configured root is not a directory")
        root_identity = (root_stat.st_dev, root_stat.st_ino)
        contract = lock["evidence_root_contract"]
        manifest_file = _authenticate_confined_file(
            root_fd,
            contract["manifest"],
            "evidence-manifest",
            capture_limit=MAX_MANIFEST_BYTES,
        )
        assert manifest_file.data is not None
        manifest = _parse_and_validate(
            manifest_file.data,
            ENGINE_DESIGN / contract["manifest_schema"],
            "I0-BLOCK-EVIDENCE-MANIFEST",
            "evidence-manifest",
        )
        expected, observed = _validate_manifest_against_lock(lock, manifest)
        validate_autoround_distribution_policy(
            lock,
            manifest["artifacts"],
            _freeze_contract=False,
        )
        return EvidenceState(root, root_fd, root_identity, expected, observed)
    except OSError as error:
        _close_quietly(root_fd)
        raise EvidenceBlocked(
            "I0-BLOCK-EVIDENCE-ROOT",
            "evidence-root",
            f"configured evidence root could not be opened securely: {error.strerror}",
        ) from error
    except Exception:
        _close_quietly(root_fd)
        raise


def _read_descriptor(
    state: EvidenceState,
    descriptor: dict[str, Any],
    artifact_id: str,
    *,
    capture_limit: int | None,
) -> AuthenticatedFile:
    return _authenticate_confined_file(
        state.root_fd,
        descriptor["path"],
        artifact_id,
        expected_size=descriptor["size"],
        expected_sha256=descriptor["sha256"],
        capture_limit=capture_limit,
    )


def _authenticate_evidence_artifact(
    state: EvidenceState,
    artifact_id: str,
    *,
    capture_limit: int | None,
) -> AuthenticatedFile:
    if artifact_id not in state.expected:
        raise EvidenceBlocked("I0-BLOCK-EVIDENCE-CONTRACT", artifact_id, "identity is not declared")
    identity = state.expected[artifact_id]
    entry = state.observed.get(artifact_id)
    blocker = identity.get("blocker", "I0-BLOCK-SOURCE-EVIDENCE")
    if entry is None:
        raise EvidenceBlocked(blocker, artifact_id, "manifest entry is missing")
    if entry["status"] != "present":
        raise EvidenceBlocked(entry.get("blocker", blocker), artifact_id, "artifact is declared absent")
    if entry["size"] != identity["size"]:
        raise EvidenceBlocked("I0-BLOCK-EVIDENCE-SIZE", artifact_id, "manifest size differs from immutable identity")
    if entry["sha256"] != identity["sha256"]:
        raise EvidenceBlocked("I0-BLOCK-EVIDENCE-HASH", artifact_id, "manifest hash differs from immutable identity")
    return _read_descriptor(state, entry, artifact_id, capture_limit=capture_limit)


def _read_evidence_artifact(state: EvidenceState, artifact_id: str) -> bytes:
    authenticated = _authenticate_evidence_artifact(
        state,
        artifact_id,
        capture_limit=MAX_RETURNED_ARTIFACT_BYTES,
    )
    assert authenticated.data is not None
    return authenticated.data


def read_evidence_artifact(lock: dict[str, Any], artifact_id: str) -> bytes:
    with _evidence_state(lock) as state:
        data = _read_evidence_artifact(state, artifact_id)
        state.check_root_consistency()
        return data


def verify_evidence_scope(lock: dict[str, Any], scope: str) -> tuple[int, list[EvidenceBlocked]]:
    blockers: list[EvidenceBlocked] = []
    executed = 0
    try:
        _validate_frozen_evidence_contract(lock)
        expected = [
            entry for entry in lock["evidence_root_contract"]["artifacts"]
            if scope in entry["required_for"]
        ]
        state = _evidence_state(lock)
    except EvidenceBlocked as error:
        return 0, [error]
    with state:
        for identity in expected:
            try:
                _authenticate_evidence_artifact(state, identity["id"], capture_limit=None)
                executed += 1
            except EvidenceBlocked as error:
                blockers.append(error)
        try:
            state.check_root_consistency()
        except EvidenceBlocked as error:
            blockers.append(error)
    return executed, blockers


def _expected_checkpoint_inputs(lock: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"path": item["path"], "size": item["size"], "sha256": item["lfs_sha256"]}
        for item in lock["official_model"]["checkpoint_shards"]
    ]


def _validate_i0_gate_contract(
    lock: dict[str, Any], *, require_independent_bound_lineages: bool = False
) -> dict[str, dict[str, Any]]:
    """Validate the immutable schema-1 I0 gate shape before reading any gate bytes."""
    blocker = "I0-BLOCK-GATE-CONTRACT"
    artifact_id = "BASE-02-gate"
    validate_autoround_distribution_policy(lock, _freeze_contract=False)
    try:
        contract = lock["evidence_root_contract"]
        artifacts = contract["artifacts"]
        categories = contract["required_gate_categories"]
        shards = lock["official_model"]["checkpoint_shards"]
    except (KeyError, TypeError) as error:
        raise EvidenceBlocked(blocker, artifact_id, f"I0 gate contract field is missing: {error}") from error

    if lock.get("schema") != 1:
        raise EvidenceBlocked(blocker, artifact_id, "unsupported reference-lock schema for the frozen I0 gate")
    if categories != I0_GATE_CATEGORIES:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "required gate categories, counts, and prefixes differ from the immutable I0 contract",
        )
    if not isinstance(artifacts, list):
        raise EvidenceBlocked(blocker, artifact_id, "evidence artifacts must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for identity in artifacts:
        if not isinstance(identity, dict) or not isinstance(identity.get("id"), str):
            raise EvidenceBlocked(blocker, artifact_id, "every evidence identity must have a string id")
        if identity["id"] in by_id:
            raise EvidenceBlocked(blocker, identity["id"], "duplicate evidence identity id")
        by_id[identity["id"]] = identity

    if len(shards) != len(I0_CHECKPOINT_IDS):
        raise EvidenceBlocked(blocker, artifact_id, "I0 requires exactly 11 official checkpoint shards")
    official = lock["official_model"]
    for index, (checkpoint_id, shard) in enumerate(zip(I0_CHECKPOINT_IDS, shards, strict=True), start=1):
        expected_path = f"model-{index:05d}-of-00011.safetensors"
        identity = by_id.get(checkpoint_id)
        expected_lineage = {
            "repository": official["repository"],
            "revision": official["revision"],
            "path": expected_path,
        }
        if shard.get("path") != expected_path:
            raise EvidenceBlocked(blocker, checkpoint_id, "checkpoint shard role/path differs from I0 numbering")
        if identity is None:
            raise EvidenceBlocked(blocker, checkpoint_id, "required checkpoint gate identity is missing")
        if identity.get("required_for") != I0_GATE_REQUIRED_FOR:
            raise EvidenceBlocked(blocker, checkpoint_id, "checkpoint required_for differs from BASE-02-gate")
        if identity.get("blocker") != "I0-BLOCK-CHECKPOINT-EVIDENCE":
            raise EvidenceBlocked(blocker, checkpoint_id, "checkpoint blocker role differs from the I0 contract")
        if identity.get("lineage") != expected_lineage:
            raise EvidenceBlocked(blocker, checkpoint_id, "checkpoint lineage role differs from the official shard")
        if (identity.get("size"), identity.get("sha256")) != (shard.get("size"), shard.get("lfs_sha256")):
            raise EvidenceBlocked(blocker, checkpoint_id, "checkpoint gate bytes differ from the official shard")

    checkpoint_prefixed = {name for name in by_id if name.startswith("official.checkpoint.")}
    if checkpoint_prefixed != set(I0_CHECKPOINT_IDS):
        raise EvidenceBlocked(blocker, artifact_id, "checkpoint identity set is omitted, extra, or substituted")

    for gguf_id, kind in I0_GGUF_KINDS.items():
        identity = by_id.get(gguf_id)
        if identity is None:
            raise EvidenceBlocked(blocker, gguf_id, "required named GGUF gate identity is missing")
        if identity.get("required_for") != I0_GATE_REQUIRED_FOR:
            raise EvidenceBlocked(blocker, gguf_id, "GGUF required_for differs from BASE-02-gate")
        if identity.get("artifact_kind") != kind:
            raise EvidenceBlocked(blocker, gguf_id, "GGUF kind differs from its immutable I0 role")
        if "derivation_requirement" not in identity or "lineage_lock" not in identity:
            raise EvidenceBlocked(blocker, gguf_id, "GGUF lineage contract is missing")

    gguf_prefixed = {name for name in by_id if name.startswith("gguf.")}
    if gguf_prefixed != set(I0_GGUF_KINDS):
        raise EvidenceBlocked(blocker, artifact_id, "GGUF identity set is omitted, extra, or substituted")

    expected_gate_ids = set(I0_CHECKPOINT_IDS) | set(I0_GGUF_KINDS)
    tagged_gate_ids = {
        identity["id"]
        for identity in artifacts
        if isinstance(identity.get("required_for"), list) and "BASE-02-gate" in identity["required_for"]
    }
    if tagged_gate_ids != expected_gate_ids:
        raise EvidenceBlocked(blocker, artifact_id, "BASE-02-gate membership is omitted, extra, retagged, or substituted")
    lineage_ids = {name for name, identity in by_id.items() if "lineage_lock" in identity}
    if lineage_ids != set(I0_GGUF_KINDS):
        raise EvidenceBlocked(blocker, artifact_id, "I0 requires exactly three corresponding GGUF lineage locks")

    bound_lineages = [by_id[name] for name in I0_GGUF_KINDS if by_id[name]["lineage_lock"]["status"] == "BOUND"]
    if require_independent_bound_lineages and (
        bound_lineages or lock.get("gguf_lineage_policy", {}).get("status") == "BOUND"
    ):
        if len(bound_lineages) != 3:
            raise EvidenceBlocked(blocker, artifact_id, "I0 requires exactly three independently bound GGUF lineages")
        derivation_ids = [
            _canonical_id(
                identity["lineage_lock"]["derivation_id"],
                blocker,
                identity["id"],
                "derivation ID",
            )
            for identity in bound_lineages
        ]
        if len(set(derivation_ids)) != 3:
            raise EvidenceBlocked(blocker, artifact_id, "BF16, Q8_0, and Q4_K_M derivation IDs must be globally unique")
        descriptor_ids: list[str] = []
        for name in I0_GGUF_KINDS:
            lineage = by_id[name]["lineage_lock"]
            derivation_id = _canonical_id(lineage["derivation_id"], blocker, name, "derivation ID")
            if not derivation_id.startswith(I0_DERIVATION_ID_PREFIXES[name]):
                raise EvidenceBlocked(
                    blocker,
                    name,
                    "derivation ID must be canonical and artifact-qualified for its GGUF lineage",
                )
            expected_record_id = f"{name}.derivation-record"
            expected_log_id = f"{name}.command-log"
            if lineage["record"]["id"] != expected_record_id:
                raise EvidenceBlocked(blocker, name, "derivation record ID must be artifact- and role-qualified")
            if lineage["command_log"]["id"] != expected_log_id:
                raise EvidenceBlocked(blocker, name, "command-log ID must be artifact- and role-qualified")
            descriptor_ids.extend((lineage["record"]["id"], lineage["command_log"]["id"]))
        if len(set(descriptor_ids)) != 6:
            raise EvidenceBlocked(blocker, artifact_id, "all six lineage evidence descriptor IDs must be globally unique")
    return by_id


def _remote_gguf_identity(lock: dict[str, Any], artifact_id: str) -> dict[str, Any]:
    suffix = I0_GGUF_REMOTE_SUFFIXES[artifact_id]
    matches = [item for item in lock["gguf_remote_reference"]["artifacts"] if item["path"].endswith(suffix)]
    if len(matches) != 1:
        raise EvidenceBlocked(
            "I0-BLOCK-GATE-CONTRACT",
            artifact_id,
            "remote GGUF role must resolve to exactly one named artifact",
        )
    return matches[0]


def _descriptor_matches_lock(
    descriptor: dict[str, Any], anchor: dict[str, Any], blocker: str, artifact_id: str, role: str
) -> None:
    observed = {key: descriptor.get(key) for key in ("id", "path", "size", "sha256")}
    if observed != anchor:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            f"external {role} descriptor does not match the Git-controlled lineage lock anchor",
        )


def _validate_command_log(
    data: bytes,
    record: dict[str, Any],
    schema_path: Path,
    blocker: str,
    artifact_id: str,
) -> None:
    observed = _parse_and_validate(data, schema_path, blocker, artifact_id)
    expected = {
        "$schema": "urn:comfyui-diffusiongemma:dg-engine:i0:command-log:v1",
        "schema": 1,
        "kind": "dg-engine-i0-command-log",
        "artifact_id": record["artifact_id"],
        "derivation_id": record["derivation_id"],
        "artifact_kind": record["artifact_kind"],
        "official_checkpoint_inputs": record["official_checkpoint_inputs"],
        "additional_inputs": record["additional_inputs"],
        "converter": record["converter"],
        "conversion": record["conversion"],
        "tool_binary": record["tool_binary"],
        "quantization": record["quantization"],
        "output": record["output"],
        "completion": {"status": "completed", "exit_code": 0},
    }
    if observed != expected:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "command/tool log must exactly cross-bind the complete authenticated derivation context",
        )


def _verify_gguf_derivation(
    lock: dict[str, Any],
    state: EvidenceState,
    artifact_id: str,
    *,
    aliases: _DerivationAliases | None = None,
    output_authenticated: AuthenticatedFile | None = None,
    authenticated_outputs: dict[str, AuthenticatedFile] | None = None,
    authenticated_lineages: set[str] | None = None,
) -> tuple[str, str]:
    gate_identities = _validate_i0_gate_contract(lock)
    if artifact_id not in I0_GGUF_KINDS:
        raise EvidenceBlocked("I0-BLOCK-GATE-CONTRACT", artifact_id, "artifact is not a named I0 GGUF identity")
    identity = gate_identities[artifact_id]
    requirement = identity["derivation_requirement"]
    blocker = requirement["blocker"]
    lineage_lock = identity.get("lineage_lock")
    if lock["gguf_lineage_policy"].get("status") != "BOUND":
        raise EvidenceBlocked(blocker, artifact_id, "GGUF lineage policy remains UNBOUND in the Git-controlled lock")
    if (
        identity.get("lineage_status") != "BOUND"
        or not isinstance(lineage_lock, dict)
        or lineage_lock.get("status") != "BOUND"
        or not lineage_lock.get("derivation_id")
        or not isinstance(lineage_lock.get("record"), dict)
        or not isinstance(lineage_lock.get("command_log"), dict)
    ):
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "lineage is UNBOUND: non-null record and command-log anchors require reviewed Git lock source",
        )
    remote = _remote_gguf_identity(lock, artifact_id)
    remote_bytes = (remote.get("size"), remote.get("lfs_sha256"))
    gate_bytes = (identity.get("size"), identity.get("sha256"))
    if remote_bytes != gate_bytes:
        raise EvidenceBlocked(
            blocker,
            artifact_id,
            "remote GGUF size/SHA-256 differs from the independently locked BASE-02 gate identity",
        )
    if (
        lock["gguf_remote_reference"].get("lineage_status") != "BOUND"
        or remote.get("lineage_status") != "BOUND"
        or remote.get("lineage_claim") != lineage_lock["derivation_id"]
    ):
        raise EvidenceBlocked(blocker, artifact_id, "remote GGUF lineage claim differs from the Git lineage lock")

    entry = state.observed.get(artifact_id)
    if entry is None or entry.get("status") != "present":
        raise EvidenceBlocked(blocker, artifact_id, "authenticated derivation is absent with the GGUF artifact")
    derivation = entry.get("derivation")
    if not isinstance(derivation, dict):
        raise EvidenceBlocked(blocker, artifact_id, "independent derivation descriptors are missing")
    record_descriptor = derivation.get("record")
    log_descriptor = derivation.get("command_log")
    if not isinstance(record_descriptor, dict) or not isinstance(log_descriptor, dict):
        raise EvidenceBlocked(blocker, artifact_id, "record and command-log descriptors are mandatory")
    _descriptor_matches_lock(record_descriptor, lineage_lock["record"], blocker, artifact_id, "record")
    _descriptor_matches_lock(log_descriptor, lineage_lock["command_log"], blocker, artifact_id, "command-log")
    if record_descriptor["id"] != f"{artifact_id}.derivation-record":
        raise EvidenceBlocked(blocker, artifact_id, "derivation record ID must be artifact- and role-qualified")
    if log_descriptor["id"] != f"{artifact_id}.command-log":
        raise EvidenceBlocked(blocker, artifact_id, "command-log ID must be artifact- and role-qualified")

    aliases = aliases or _DerivationAliases()
    output_owner = f"{artifact_id}.output"
    record_owner = f"{artifact_id}.derivation-record"
    log_owner = f"{artifact_id}.command-log"
    aliases.claim_descriptor_id(record_descriptor["id"], record_owner, blocker, artifact_id)
    aliases.claim_descriptor_id(log_descriptor["id"], log_owner, blocker, artifact_id)
    aliases.claim_path(entry["path"], output_owner, blocker, artifact_id)
    aliases.claim_path(record_descriptor["path"], record_owner, blocker, artifact_id)
    aliases.claim_path(log_descriptor["path"], log_owner, blocker, artifact_id)

    if output_authenticated is None:
        output_authenticated = _authenticate_evidence_artifact(state, artifact_id, capture_limit=None)
    aliases.claim_file(output_authenticated, output_owner, blocker, artifact_id)

    record_file = _read_descriptor(
        state,
        record_descriptor,
        record_owner,
        capture_limit=MAX_DERIVATION_RECORD_BYTES,
    )
    aliases.claim_file(record_file, record_owner, blocker, artifact_id)
    assert record_file.data is not None
    record = _parse_and_validate(
        record_file.data,
        ENGINE_DESIGN / requirement["record_schema"],
        blocker,
        artifact_id,
    )
    if record["artifact_id"] != artifact_id or record["derivation_id"] != lineage_lock["derivation_id"]:
        raise EvidenceBlocked(blocker, artifact_id, "derivation record artifact/derivation ID differs from lock")
    if record["official_checkpoint_inputs"] != _expected_checkpoint_inputs(lock):
        raise EvidenceBlocked(blocker, artifact_id, "official checkpoint/input digest set is incomplete or mismatched")
    converter = lock["converter"]
    expected_converter = {
        "repository": converter["repository"],
        "revision": converter["revision"],
        "path": converter["path"],
        "git_blob_sha1": converter["git_blob_sha1"],
        "sha256": converter["sha256"],
    }
    if record["converter"] != expected_converter:
        raise EvidenceBlocked(blocker, artifact_id, "selected converter identity is mismatched")
    expected_output = {"id": artifact_id, "size": identity["size"], "sha256": identity["sha256"]}
    if record["output"] != expected_output:
        raise EvidenceBlocked(blocker, artifact_id, "derivation output identity does not match GGUF bytes")
    if record["artifact_kind"] != identity["artifact_kind"] or record["quantization"]["type"] != identity["artifact_kind"]:
        raise EvidenceBlocked(blocker, artifact_id, "derivation artifact/quantization kind is mismatched")
    if identity["artifact_kind"] == "GGUF_BF16":
        if record["additional_inputs"] != [] or record["tool_binary"]["role"] != "converter-runtime":
            raise EvidenceBlocked(
                blocker, artifact_id, "BF16 derivation must identify the converter runtime and no quantized intermediary"
            )
    else:
        if authenticated_lineages is None:
            authenticated_lineages = set()
        if "gguf.bf16" not in authenticated_lineages:
            bf16_output = None if authenticated_outputs is None else authenticated_outputs.get("gguf.bf16")
            _verify_gguf_derivation(
                lock,
                state,
                "gguf.bf16",
                aliases=aliases,
                output_authenticated=bf16_output,
                authenticated_outputs=authenticated_outputs,
                authenticated_lineages=authenticated_lineages,
            )
        bf16_identity = gate_identities["gguf.bf16"]
        remote_bf16 = _remote_gguf_identity(lock, "gguf.bf16")
        if (remote_bf16.get("size"), remote_bf16.get("lfs_sha256")) != (
            bf16_identity.get("size"),
            bf16_identity.get("sha256"),
        ):
            raise EvidenceBlocked(
                blocker,
                artifact_id,
                "remote BF16 identity differs from the independently locked BF16 gate trust root",
            )
        if authenticated_outputs is None:
            authenticated_outputs = {}
        authenticated_bf16 = authenticated_outputs.get("gguf.bf16")
        if authenticated_bf16 is None:
            authenticated_bf16 = _authenticate_evidence_artifact(state, "gguf.bf16", capture_limit=None)
            authenticated_outputs["gguf.bf16"] = authenticated_bf16
        if (authenticated_bf16.size, authenticated_bf16.sha256) != (
            bf16_identity["size"],
            bf16_identity["sha256"],
        ):
            raise EvidenceBlocked(blocker, artifact_id, "authenticated BF16 bytes differ from the BF16 gate trust root")
        expected_input = [{
            "path": remote_bf16["path"],
            "size": bf16_identity["size"],
            "sha256": bf16_identity["sha256"],
        }]
        if record["additional_inputs"] != expected_input or record["tool_binary"]["role"] not in {
            "llama-quantize",
            "quantizer",
        }:
            raise EvidenceBlocked(
                blocker,
                artifact_id,
                "quantized derivation must bind the authenticated BF16 gate identity and quantizer binary",
            )

    log_file = _read_descriptor(
        state,
        log_descriptor,
        log_owner,
        capture_limit=MAX_COMMAND_LOG_BYTES,
    )
    aliases.claim_file(log_file, log_owner, blocker, artifact_id)
    assert log_file.data is not None
    _validate_command_log(
        log_file.data,
        record,
        ENGINE_DESIGN / requirement["command_log_schema"],
        blocker,
        artifact_id,
    )
    if authenticated_lineages is not None:
        authenticated_lineages.add(artifact_id)
    return record_descriptor["path"], log_descriptor["path"]


def verify_gguf_derivation(lock: dict[str, Any], artifact_id: str) -> tuple[str, str]:
    with _evidence_state(lock) as state:
        result = _verify_gguf_derivation(lock, state, artifact_id)
        state.check_root_consistency()
        return result


def verify_base02_gate(
    lock: dict[str, Any], *, _freeze_contract: bool = True
) -> GateVerification:
    categories = {name: 0 for name in I0_GATE_CATEGORIES}
    blockers: list[EvidenceBlocked] = []
    artifact_executed = 0
    lineage_executed = 0
    try:
        gate_identities = _validate_i0_gate_contract(lock, require_independent_bound_lineages=True)
        if _freeze_contract:
            _validate_frozen_evidence_contract(lock)
        state = _evidence_state(lock)
    except EvidenceBlocked as error:
        return GateVerification(0, 0, categories, [error])
    aliases = _DerivationAliases()
    authenticated_outputs: dict[str, AuthenticatedFile] = {}
    authenticated_lineages: set[str] = set()
    gate_ids = (*I0_CHECKPOINT_IDS, *I0_GGUF_KINDS)
    with state:
        for artifact_id in gate_ids:
            try:
                authenticated = _authenticate_evidence_artifact(state, artifact_id, capture_limit=None)
                authenticated_outputs[artifact_id] = authenticated
                artifact_executed += 1
                category = "checkpoint_identity" if artifact_id in I0_CHECKPOINT_IDS else "gguf_identity"
                categories[category] += 1
            except EvidenceBlocked as error:
                blockers.append(error)

        # Lineage order is frozen: BF16 is established before either quantized
        # output can consume its authenticated gate identity as a trust root.
        for artifact_id in I0_GGUF_KINDS:
            identity = gate_identities[artifact_id]
            output_authenticated = authenticated_outputs.get(artifact_id)
            blocker = identity["derivation_requirement"]["blocker"]
            if output_authenticated is None:
                blockers.append(
                    EvidenceBlocked(blocker, artifact_id, "GGUF output bytes were not authenticated for derivation")
                )
                continue
            try:
                _verify_gguf_derivation(
                    lock,
                    state,
                    artifact_id,
                    aliases=aliases,
                    output_authenticated=output_authenticated,
                    authenticated_outputs=authenticated_outputs,
                    authenticated_lineages=authenticated_lineages,
                )
                lineage_executed += 1
                categories["gguf_authenticated_lineage"] += 1
            except EvidenceBlocked as error:
                blockers.append(error)
        try:
            state.check_root_consistency()
        except EvidenceBlocked as error:
            blockers.append(error)
    return GateVerification(artifact_executed, lineage_executed, categories, blockers)


def scope_is_complete(executed: int, blockers: list[EvidenceBlocked]) -> bool:
    return executed > 0 and not blockers


def gate_categories_complete(lock: dict[str, Any], result: GateVerification) -> bool:
    try:
        _validate_i0_gate_contract(lock, require_independent_bound_lineages=True)
        _validate_frozen_evidence_contract(lock)
    except EvidenceBlocked:
        return False
    return result.categories == {
        name: detail["expected"] for name, detail in I0_GATE_CATEGORIES.items()
    }


def _read_authenticated_cdg_source(
    lock: dict[str, Any], relative_path: str, state: EvidenceState
) -> bytes:
    matches = [item for item in lock["cdg_source_identities"] if item["path"] == relative_path]
    if len(matches) != 1:
        raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", relative_path, "source identity is missing or duplicated")
    identity = matches[0]
    if identity["revision"] != lock["product_baseline"]:
        raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", relative_path, "source is not bound to product baseline")
    contract = next(
        (item for item in lock["evidence_root_contract"]["artifacts"] if item["id"] == identity["artifact_id"]),
        None,
    )
    expected_source = {
        "repository": identity["repository"],
        "revision": identity["revision"],
        "path": identity["path"],
        "git_blob_sha1": identity["git_blob_sha1"],
    }
    if contract is None or contract.get("source_identity") != expected_source:
        raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", relative_path, "durable evidence contract identity is mismatched")
    data = _read_evidence_artifact(state, identity["artifact_id"])
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    revision_blob = subprocess.check_output(
        ["git", "rev-parse", f"{identity['revision']}:{relative_path}"], cwd=REPO_ROOT, text=True
    ).strip()
    if blob != identity["git_blob_sha1"] or revision_blob != identity["git_blob_sha1"]:
        raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", relative_path, "staged bytes do not match baseline Git blob")
    return data


def read_authenticated_cdg_source(lock: dict[str, Any], relative_path: str) -> bytes:
    with _evidence_state(lock) as state:
        data = _read_authenticated_cdg_source(lock, relative_path, state)
        state.check_root_consistency()
        return data


def authenticated_current_cdg_sources(lock: dict[str, Any]) -> dict[str, bytes]:
    """Return the complete executable source set authenticated under one root descriptor.

    Checkout comparison happens before this returns, but callers execute only
    the returned immutable bytes.  The live product path is never an execution
    input after authentication.
    """
    sources: dict[str, bytes] = {}
    with _evidence_state(lock) as state:
        for identity in lock["cdg_source_identities"]:
            relative_path = identity["path"]
            if relative_path in sources:
                raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", relative_path, "source identity is duplicated")
            authenticated = _read_authenticated_cdg_source(lock, relative_path, state)
            current = (REPO_ROOT / relative_path).read_bytes()
            if current != authenticated:
                raise EvidenceBlocked(
                    "I0-BLOCK-CDG-SOURCE", relative_path, "checkout bytes differ from authenticated baseline"
                )
            sources[relative_path] = authenticated
        state.check_root_consistency()
    return sources


def authenticate_current_cdg_sources(lock: dict[str, Any]) -> None:
    """Authenticate all executable BASE source without executing pathname bytes."""
    authenticated_current_cdg_sources(lock)


class _BareProductImportGuard(importlib.abc.MetaPathFinder):
    """Record and reject ordinary imports of product top-level packages."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, str]] = []
        self.active = True

    @staticmethod
    def is_bare_product(fullname: str) -> bool:
        root = fullname.partition(".")[0]
        return root in BARE_PRODUCT_ROOTS

    def reject(self, fullname: str, mechanism: str) -> None:
        if not self.active or not self.is_bare_product(fullname):
            return
        self.requests.append((fullname, mechanism))
        raise EvidenceBlocked(
            "I0-BLOCK-CDG-SOURCE",
            fullname,
            "bare product import is forbidden while authenticated product context is active",
        )

    def find_spec(
        self,
        fullname: str,
        _path: object = None,
        _target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        self.reject(fullname, "meta_path")
        return None

    def clear(self) -> None:
        self.active = False
        self.requests.clear()


class _OwnedImportHook:
    """Detachable import hook so a foreign wrapper cannot retain live context state."""

    def __init__(self, guard: _BareProductImportGuard, upstream: Any) -> None:
        self._guard: _BareProductImportGuard | None = guard
        self._upstream: Any | None = upstream
        self.safe_builtins = _SafeBuiltinsProxy(self)

    def __call__(
        self,
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        guard = self._guard
        upstream = self._upstream
        if guard is None or upstream is None:
            raise EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE", name, "detached authenticated importer cannot be reused"
            )
        if level == 0:
            guard.reject(name, "builtins.__import__")
            if name == "builtins":
                return self.safe_builtins
        return upstream(name, globals, locals, fromlist, level)

    def detach(self) -> None:
        self._guard = None
        self._upstream = None


class _AuthenticatedOracleFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Load selected oracle modules only from retained, already authenticated bytes."""

    def __init__(self, sources: dict[str, tuple[Path, bytes, bool]]) -> None:
        self._sources = {
            name: (Path(path), bytes(data), is_package)
            for name, (path, data, is_package) in sources.items()
        }
        self.executed: dict[str, ModuleType] = {}

    def find_spec(
        self,
        fullname: str,
        _path: object = None,
        _target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        mapped = self._sources.get(fullname)
        if mapped is None:
            return None
        path, _data, is_package = mapped
        spec = importlib.util.spec_from_loader(fullname, self, origin=str(path), is_package=is_package)
        if spec is None:
            raise ImportError(f"could not construct authenticated oracle spec for {fullname}")
        spec.has_location = True
        if is_package:
            spec.submodule_search_locations = [str(path.parent)]
        return spec

    def create_module(self, _spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        mapped = self._sources.get(module.__name__)
        if mapped is None:
            raise ImportError(f"retained oracle source disappeared for {module.__name__}")
        path, data, _is_package = mapped
        code = compile(data, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
        # Hugging Face package roots deliberately replace themselves with a
        # trusted lazy-module object during __init__ execution.  Record the
        # exact retained-byte result visible to importers, not the temporary
        # bootstrap module passed to exec_module.
        self.executed[module.__name__] = sys.modules.get(module.__name__, module)

    def release(self) -> None:
        self._sources.clear()
        self.executed.clear()


def _authenticate_oracle_source(
    path: Path,
    expected_sha256: str | None,
    artifact_id: str,
    *,
    capture: bool,
    expected_size: int | None = None,
) -> bytes | None:
    """Authenticate one regular source through a no-follow, nonblocking descriptor.

    Establishment captures only bounded source bytes for compilation.  Recurring
    checks use the same descriptor policy but stream directly into SHA-256, so a
    late FIFO/device/symlink replacement cannot block or trigger an unbounded
    allocation.
    """
    flags = os.O_RDONLY | os.O_CLOEXEC
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not nofollow or not nonblock:
        raise EvidenceBlocked(
            "I0-BLOCK-ORACLE-BINDING",
            artifact_id,
            "O_NOFOLLOW and O_NONBLOCK are required for oracle source authentication",
        )
    fd = -1
    try:
        fd = os.open(path, flags | nofollow | nonblock)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise EvidenceBlocked("I0-BLOCK-ORACLE-BINDING", artifact_id, "oracle source is not a regular file")
        if before.st_size > MAX_ORACLE_SOURCE_BYTES:
            raise EvidenceBlocked(
                "I0-BLOCK-ORACLE-BINDING",
                artifact_id,
                f"oracle source exceeds the {MAX_ORACLE_SOURCE_BYTES}-byte source limit",
            )
        if expected_size is not None and before.st_size != expected_size:
            raise EvidenceBlocked(
                "I0-BLOCK-ORACLE-BINDING",
                artifact_id,
                "installed oracle source size differs from the authenticated snapshot",
            )
        chunks: list[bytes] | None = [] if capture else None
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(fd, STREAM_CHUNK_BYTES)
            if not chunk:
                break
            if chunks is not None:
                chunks.append(chunk)
            digest.update(chunk)
            count += len(chunk)
            if count > MAX_ORACLE_SOURCE_BYTES:
                raise EvidenceBlocked(
                    "I0-BLOCK-ORACLE-BINDING", artifact_id, "oracle source grew beyond the source limit"
                )
        after = os.fstat(fd)
        identities = lambda value: (  # noqa: E731 - compact immutable descriptor tuple
            value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns
        )
        if identities(before) != identities(after) or count != before.st_size:
            raise EvidenceBlocked(
                "I0-BLOCK-ORACLE-BINDING", artifact_id, "oracle source changed while authenticated bytes were read"
            )
        observed = digest.hexdigest()
        if expected_sha256 is not None and observed != expected_sha256:
            raise EvidenceBlocked(
                "I0-BLOCK-ORACLE-BINDING", artifact_id, "installed oracle source bytes differ from the lock"
            )
        return b"".join(chunks) if chunks is not None else None
    except EvidenceBlocked:
        raise
    except OSError as error:
        raise EvidenceBlocked(
            "I0-BLOCK-ORACLE-BINDING", artifact_id, f"oracle source safe open failed: {error.strerror}"
        ) from error
    finally:
        _close_quietly(fd)


def _read_authenticated_oracle_bytes(path: Path, expected_sha256: str | None, artifact_id: str) -> bytes:
    """Read and retain exactly the bounded installed bytes that were hashed."""
    data = _authenticate_oracle_source(path, expected_sha256, artifact_id, capture=True)
    assert data is not None
    return data


def _assert_authenticated_oracle_source(
    path: Path,
    expected_sha256: str,
    artifact_id: str,
    expected_size: int,
) -> None:
    """Stream-authenticate exact-size installed source during recurring checks."""
    _authenticate_oracle_source(
        path,
        expected_sha256,
        artifact_id,
        capture=False,
        expected_size=expected_size,
    )


@dataclass
class _OracleTraversalBudget:
    """Shared hard limits for one deterministic oracle dependency walk."""

    max_depth: int = MAX_ORACLE_DEPENDENCY_DEPTH
    max_nodes: int = MAX_ORACLE_DEPENDENCY_NODES
    max_container_items: int = MAX_ORACLE_CONTAINER_ITEMS
    max_encoded_bytes: int = MAX_ORACLE_ENCODED_BYTES
    nodes: int = 0
    container_items: int = 0
    encoded_bytes: int = 0

    def _blocked(self, artifact_id: str, limit: str) -> EvidenceBlocked:
        return EvidenceBlocked(
            "I0-BLOCK-ORACLE-BINDING",
            artifact_id,
            f"oracle dependency traversal exceeded {limit} limit",
        )

    def visit(self, artifact_id: str, depth: int) -> None:
        if depth > self.max_depth:
            raise self._blocked(artifact_id, f"depth={self.max_depth}")
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise self._blocked(artifact_id, f"node-count={self.max_nodes}")

    def add_container_items(self, artifact_id: str, count: int) -> None:
        if count > MAX_ORACLE_SINGLE_CONTAINER_ITEMS:
            raise self._blocked(
                artifact_id,
                f"single-container-items={MAX_ORACLE_SINGLE_CONTAINER_ITEMS}",
            )
        self.container_items += count
        if self.container_items > self.max_container_items:
            raise self._blocked(artifact_id, f"container-items={self.max_container_items}")

    def ensure_encoded_allocation(self, artifact_id: str, count: int) -> None:
        """Reject a prospective allocation before bytes are copied/encoded."""
        if count < 0 or count > self.max_encoded_bytes - self.encoded_bytes:
            raise self._blocked(artifact_id, f"encoded-bytes={self.max_encoded_bytes}")

    def add_encoded_bytes(self, artifact_id: str, count: int) -> None:
        self.ensure_encoded_allocation(artifact_id, count)
        self.encoded_bytes += count

    def remaining_encoded_bytes(self) -> int:
        return self.max_encoded_bytes - self.encoded_bytes


class _ClassIntegrityEncoder:
    """Bounded canonical process-local encoder for mutable class structure."""

    def __init__(
        self,
        budget: _OracleTraversalBudget | None = None,
        artifact_id: str = "oracle-class",
    ) -> None:
        self.digest = hashlib.sha256()
        self._seen: dict[int, int] = {}
        # Retain every identity key for the walk so short-lived bound-method
        # wrappers cannot be collected and have their id reused mid-fingerprint.
        self._seen_values: list[object] = []
        self._budget = budget or _OracleTraversalBudget()
        self._artifact_id = artifact_id

    def _field(self, tag: bytes, data: bytes = b"") -> None:
        encoded_size = 12 + len(tag) + len(data)
        self._budget.add_encoded_bytes(self._artifact_id, encoded_size)
        self.digest.update(len(tag).to_bytes(4, "big"))
        self.digest.update(tag)
        self.digest.update(len(data).to_bytes(8, "big"))
        self.digest.update(data)

    def _encoded_text(
        self,
        value: str,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> bytes:
        """Encode only after a conservative no-allocation budget precheck."""
        if encoding == "ascii":
            worst_case = len(value)
        elif errors == "backslashreplace":
            worst_case = len(value) * 10
        else:
            worst_case = len(value) * 4
        self._budget.ensure_encoded_allocation(self._artifact_id, worst_case)
        return value.encode(encoding, errors)

    def _field_text(
        self,
        tag: bytes,
        value: str,
        encoding: str = "utf-8",
        errors: str = "strict",
    ) -> None:
        self._field(tag, self._encoded_text(value, encoding, errors))

    def _copied_bytes(self, value: bytearray | memoryview) -> bytes:
        byte_count = value.nbytes if isinstance(value, memoryview) else len(value)
        self._budget.ensure_encoded_allocation(self._artifact_id, byte_count)
        return bytes(value)

    def _add_code(self, value: CodeType, depth: int) -> None:
        """Fingerprint code structurally without an unbounded marshal temporary."""
        code_size = len(value.co_code)
        if code_size > MAX_ORACLE_SINGLE_CODE_BYTES:
            raise self._budget._blocked(
                self._artifact_id,
                f"single-code-bytes={MAX_ORACLE_SINGLE_CODE_BYTES}",
            )
        sequences = (
            value.co_consts,
            value.co_names,
            value.co_varnames,
            value.co_freevars,
            value.co_cellvars,
        )
        for sequence in sequences:
            self._budget.add_container_items(self._artifact_id, len(sequence))
        self._field(b"code-start")
        for number in (
            value.co_argcount,
            value.co_posonlyargcount,
            value.co_kwonlyargcount,
            value.co_nlocals,
            value.co_stacksize,
            value.co_flags,
            value.co_firstlineno,
        ):
            self._field(b"code-int", str(number).encode("ascii"))
        self._field(b"code-bytes", value.co_code)
        for text in (value.co_name, value.co_qualname, value.co_filename):
            self._field(b"code-text", self._encoded_text(text, "utf-8", "surrogatepass"))
        for label, sequence in (
            (b"consts", value.co_consts),
            (b"names", value.co_names),
            (b"varnames", value.co_varnames),
            (b"freevars", value.co_freevars),
            (b"cellvars", value.co_cellvars),
        ):
            self._field(label, str(len(sequence)).encode("ascii"))
            for item in sequence:
                self.add(item, depth + 1)
        self._field(b"linetable", value.co_linetable)
        self._field(b"exceptiontable", value.co_exceptiontable)
        self._field(b"code-end")

    def _reference(self, value: object) -> bool:
        identity = id(value)
        reference = self._seen.get(identity)
        if reference is not None:
            self._field(b"graph-reference", str(reference).encode("ascii"))
            return True
        self._seen[identity] = len(self._seen)
        self._seen_values.append(value)
        return False

    def _sort_token(self, value: object) -> tuple[str, str, bytes]:
        type_name = f"{type(value).__module__}:{type(value).__qualname__}"
        if value is None:
            payload = b"None"
        elif isinstance(value, str):
            payload = self._encoded_text(value, "utf-8", "surrogatepass")
        elif isinstance(value, bytes):
            self._budget.ensure_encoded_allocation(self._artifact_id, len(value))
            payload = value
        elif isinstance(value, (bool, int, float, complex)):
            if isinstance(value, int) and value.bit_length() > self._budget.remaining_encoded_bytes() * 3:
                raise self._budget._blocked(self._artifact_id, f"encoded-bytes={self._budget.max_encoded_bytes}")
            payload = self._encoded_text(repr(value), "ascii")
        elif isinstance(value, (FunctionType, type)):
            payload = self._encoded_text(
                f"{getattr(value, '__module__', '')}:{getattr(value, '__qualname__', '')}:{id(value)}",
                "utf-8",
                "backslashreplace",
            )
        else:
            payload = str(id(value)).encode("ascii")
        return type(value).__module__, type_name, payload

    def add(self, value: object, depth: int = 0) -> None:
        self._budget.visit(self._artifact_id, depth)
        if value is None:
            self._field(b"NoneType")
            return
        if isinstance(value, str):
            self._field(b"str", self._encoded_text(value, "utf-8", "surrogatepass"))
            return
        if isinstance(value, bytes):
            self._budget.ensure_encoded_allocation(self._artifact_id, len(value))
            self._field(b"bytes", value)
            return
        if isinstance(value, (bool, int, float, complex)):
            if isinstance(value, int) and value.bit_length() > self._budget.remaining_encoded_bytes() * 3:
                raise self._budget._blocked(self._artifact_id, f"encoded-bytes={self._budget.max_encoded_bytes}")
            self._field(type(value).__name__.encode(), self._encoded_text(repr(value), "ascii"))
            return
        if isinstance(value, (bytearray, memoryview)):
            self._field(type(value).__name__.encode(), self._copied_bytes(value))
            return
        if isinstance(value, CodeType):
            self._add_code(value, depth)
            return
        if isinstance(value, MethodType):
            if self._reference(value):
                return
            self._field(b"bound-method")
            self.add(value.__func__, depth + 1)
            self.add(value.__self__, depth + 1)
            return
        if isinstance(value, FunctionType):
            if self._reference(value):
                return
            self._field_text(
                b"function", f"{value.__module__}:{value.__qualname__}:{id(value)}", "utf-8", "backslashreplace"
            )
            self.add(value.__code__, depth + 1)
            self.add(value.__defaults__, depth + 1)
            self.add(value.__kwdefaults__, depth + 1)
            self.add(value.__annotations__, depth + 1)
            self.add(value.__dict__, depth + 1)
            if value.__closure__ is None:
                self._field(b"closure-none")
            else:
                self._budget.add_container_items(self._artifact_id, len(value.__closure__))
                self._field(b"closure", str(len(value.__closure__)).encode())
                for cell in value.__closure__:
                    try:
                        self.add(cell.cell_contents, depth + 1)
                    except ValueError:
                        self._field(b"empty-cell")
            return
        if isinstance(value, (staticmethod, classmethod)):
            self._field(type(value).__name__.encode())
            self.add(value.__func__, depth + 1)
            return
        if isinstance(value, property):
            self._field(b"property")
            self.add(value.fget, depth + 1)
            self.add(value.fset, depth + 1)
            self.add(value.fdel, depth + 1)
            self.add(value.__doc__, depth + 1)
            return
        if isinstance(value, (tuple, list)):
            if self._reference(value):
                return
            self._budget.add_container_items(self._artifact_id, len(value))
            self._field(type(value).__name__.encode(), str(len(value)).encode())
            for item in value:
                self.add(item, depth + 1)
            return
        if isinstance(value, dict):
            if self._reference(value):
                return
            self._budget.add_container_items(self._artifact_id, len(value))
            self._field(b"dict", str(len(value)).encode())
            items = [
                (self._sort_token(key), key, item_value)
                for key, item_value in list(dict.items(value))
            ]
            for (module_name, type_name, payload), key, item_value in sorted(items, key=lambda item: item[0]):
                self._field(b"key-module", self._encoded_text(module_name, "utf-8", "backslashreplace"))
                self._field(b"key-type", self._encoded_text(type_name, "utf-8", "backslashreplace"))
                self._field(b"key-payload", payload)
                self.add(item_value, depth + 1)
            return
        if isinstance(value, (set, frozenset)):
            if self._reference(value):
                return
            self._budget.add_container_items(self._artifact_id, len(value))
            children: list[tuple[tuple[str, str, bytes], object]] = [
                (self._sort_token(item), item) for item in value
            ]
            self._field(type(value).__name__.encode(), str(len(children)).encode())
            for _sort_key, item in sorted(children, key=lambda pair: pair[0]):
                self.add(item, depth + 1)
            return
        type_name = f"{type(value).__module__}:{type(value).__qualname__}"
        self._field_text(
            b"identity",
            f"{type_name}:{getattr(value, '__name__', '')}:"
            f"{getattr(value, '__qualname__', '')}:{id(value)}",
            "utf-8",
            "backslashreplace",
        )


def _class_integrity_fingerprint(trusted_class: type) -> tuple[str, tuple[type, ...]]:
    """Fingerprint the trusted class itself plus MRO identities.

    Base-class dictionaries are intentionally not walked wholesale: framework
    bases own unrelated mutable caches.  Inherited behavior is selected and
    dependency-closed precisely by the bytecode/source-derived method policy in
    ``_module_global_integrity_fingerprint``.
    """
    artifact_id = f"{trusted_class.__module__}.{trusted_class.__qualname__}"
    budget = _OracleTraversalBudget()
    encoder = _ClassIntegrityEncoder(budget, artifact_id)
    mro = tuple(trusted_class.__mro__)
    budget.add_container_items(artifact_id, len(mro))
    encoder._field(b"class", encoder._encoded_text(
        f"{trusted_class.__module__}:{trusted_class.__qualname__}", "utf-8", "surrogatepass"
    ))
    try:
        for owner in mro:
            encoder._field(b"mro-owner", encoder._encoded_text(
                f"{owner.__module__}:{owner.__qualname__}:{id(owner)}", "utf-8", "backslashreplace"
            ))
        namespace = trusted_class.__dict__
        budget.add_container_items(artifact_id, len(namespace))
        names: list[str] = []
        for name in namespace:
            # Charge/check the class entry count before appending to a sortable
            # temporary.  ``len`` above rejects oversized mappings before work;
            # this loop also makes the materialization boundary explicit.
            names.append(name)
        names.sort()
        encoder._field(b"trusted-dict-size", str(len(names)).encode("ascii"))
        for name in names:
            encoder._field(b"attribute", encoder._encoded_text(name, "utf-8", "surrogatepass"))
            # The trusted class's own dictionary is authenticated behavior, so
            # its mutable members remain content-addressed.  Only unrelated
            # mutable dictionaries on base classes are excluded above.
            encoder.add(namespace[name], 1)
    except RecursionError as error:
        raise EvidenceBlocked(
            "I0-BLOCK-ORACLE-BINDING",
            artifact_id,
            "oracle class traversal reached Python recursion before its declared depth limit",
        ) from error
    return encoder.digest.hexdigest(), mro


def _class_functions(
    trusted_class: type,
    *,
    include_inherited: bool = False,
) -> tuple[tuple[str, FunctionType], ...]:
    """Return Python implementations selected through the class's MRO.

    With ``include_inherited``, each attribute name is taken from the first MRO
    dictionary that defines it, exactly matching normal class lookup.  Thus an
    inherited implementation such as ``DiffusionPipeline.register_modules`` is
    fingerprinted while an overridden base implementation is not mislabeled as
    live behavior.
    """
    functions: list[tuple[str, FunctionType]] = []
    seen_names: set[str] = set()
    owners = trusted_class.__mro__ if include_inherited else (trusted_class,)
    for owner in owners:
        namespace = owner.__dict__
        if len(namespace) > MAX_ORACLE_SINGLE_CONTAINER_ITEMS:
            raise EvidenceBlocked(
                "I0-BLOCK-ORACLE-BINDING",
                f"{trusted_class.__module__}.{trusted_class.__qualname__}",
                f"oracle dependency traversal exceeded single-container-items={MAX_ORACLE_SINGLE_CONTAINER_ITEMS} limit",
            )
        for attribute_name in sorted(namespace):
            if attribute_name in seen_names:
                continue
            seen_names.add(attribute_name)
            value = owner.__dict__[attribute_name]
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            role = f"{owner.__module__}.{owner.__qualname__}.{attribute_name}"
            if isinstance(value, FunctionType):
                functions.append((role, value))
            elif isinstance(value, property):
                for accessor, function in (("get", value.fget), ("set", value.fset), ("del", value.fdel)):
                    if isinstance(function, FunctionType):
                        functions.append((f"{role}.{accessor}", function))
    return tuple(functions)


class _ReferencedGlobalIntegrityEncoder(_ClassIntegrityEncoder):
    """Bounded closure of behavior-relevant bindings named by Python bytecode.

    Roots are implementations in each trusted class dictionary plus the finite
    source-confirmed inherited-method manifest.  A bounded abstract bytecode
    walk follows globals/builtin fallback, closure cells, receiver/local aliases,
    attributes/methods and relevant subscripts.  Functions include code,
    defaults, closure cells, their own globals, and selected bound-receiver
    state; helper functions recurse by the same rule.  Mutable deep traversal
    and direct constructor traversal remain explicit finite manifests, avoiding
    any claim over unrelated mutable framework base-class caches.
    """

    def __init__(
        self,
        locked_sources: dict[str, tuple[str, str]],
        root_module: str,
        budget: _OracleTraversalBudget | None = None,
        artifact_id: str | None = None,
        trusted_class: type | None = None,
    ) -> None:
        super().__init__(budget, artifact_id or root_module)
        self._locked_sources = locked_sources
        self._root_module = root_module
        self._trusted_class = trusted_class

    @staticmethod
    def _static_attribute(value: object, name: str) -> tuple[bool, object]:
        """Resolve without executing arbitrary descriptor or ``__getattr__`` code."""
        if isinstance(value, ModuleType):
            namespace = value.__dict__
            return (name in namespace, namespace.get(name))
        if isinstance(value, type):
            for owner in value.__mro__:
                if name in owner.__dict__:
                    member = owner.__dict__[name]
                    if isinstance(member, staticmethod):
                        return True, member.__func__
                    if isinstance(member, classmethod):
                        return True, MethodType(member.__func__, value)
                    return True, member
            return False, None
        try:
            namespace = object.__getattribute__(value, "__dict__")
        except (AttributeError, TypeError):
            namespace = None
        if isinstance(namespace, dict) and name in namespace:
            return True, dict.__getitem__(namespace, name)
        for owner in type(value).__mro__:
            if name not in owner.__dict__:
                continue
            member = owner.__dict__[name]
            if isinstance(member, staticmethod):
                return True, member.__func__
            if isinstance(member, classmethod):
                return True, MethodType(member.__func__, type(value))
            if isinstance(member, FunctionType):
                return True, MethodType(member, value)
            return True, member
        return False, None

    @dataclass(frozen=True)
    class _FlowValue:
        kind: str
        root_kind: str = ""
        root_name: str = ""
        path: tuple[tuple[str, object], ...] = ()
        const_index: int = -1
        alternatives: tuple["_ReferencedGlobalIntegrityEncoder._FlowValue", ...] = ()

    @dataclass(frozen=True)
    class _FlowState:
        stack: tuple["_ReferencedGlobalIntegrityEncoder._FlowValue", ...]
        locals: tuple[tuple[str, "_ReferencedGlobalIntegrityEncoder._FlowValue"], ...]

    def _bytecode_dependencies(
        self,
        value: FunctionType,
        receiver: object | None,
    ) -> tuple[
        tuple[tuple[str, str, tuple[tuple[str, object], ...], bool], ...],
        dict[tuple[str, str], object],
        tuple[object, ...],
    ]:
        """Bounded deterministic abstract interpretation of dependency loads.

        The walker follows globals (including Python's builtin fallback),
        closure cells, receiver/local aliases, attributes/methods, and constant
        or dynamic subscripts across control-flow edges.  It never executes a
        descriptor.  Every emitted path is later either resolved and explicitly
        fingerprinted or evidence-blocked.
        """
        code = value.__code__
        if len(code.co_code) > MAX_ORACLE_SINGLE_CODE_BYTES:
            raise self._budget._blocked(
                self._artifact_id, f"single-code-bytes={MAX_ORACLE_SINGLE_CODE_BYTES}"
            )
        self._budget.add_container_items(self._artifact_id, len(code.co_consts))
        instructions = tuple(dis.get_instructions(code))
        self._budget.add_container_items(self._artifact_id, len(instructions))
        if not instructions:
            return (), {}, code.co_consts

        roots: dict[tuple[str, str], object] = {}
        closure_values: dict[str, object] = {}
        if value.__closure__ is not None:
            for name, cell in zip(code.co_freevars, value.__closure__, strict=True):
                try:
                    closure_values[name] = cell.cell_contents
                except ValueError:
                    continue
        initial_locals: dict[str, _ReferencedGlobalIntegrityEncoder._FlowValue] = {}
        if receiver is not None and code.co_argcount:
            receiver_name = code.co_varnames[0]
            roots[("receiver", receiver_name)] = receiver
            initial_locals[receiver_name] = self._FlowValue("ref", "receiver", receiver_name)

        unknown = self._FlowValue("unknown")
        null = self._FlowValue("null")
        dependencies: set[tuple[str, str, tuple[tuple[str, object], ...], bool]] = set()
        offset_to_index = {instruction.offset: index for index, instruction in enumerate(instructions)}
        initial_state = self._FlowState((), tuple(sorted(initial_locals.items())))
        pending: list[int] = [0]
        accepted: dict[int, _ReferencedGlobalIntegrityEncoder._FlowState] = {0: initial_state}
        processed_states = 0
        processed_by_offset: dict[int, int] = {}

        def alternatives(symbol: _ReferencedGlobalIntegrityEncoder._FlowValue) -> tuple[_ReferencedGlobalIntegrityEncoder._FlowValue, ...]:
            if symbol.kind == "multi":
                return symbol.alternatives
            if symbol.kind == "ref":
                return (symbol,)
            return ()

        def union_value(
            left: _ReferencedGlobalIntegrityEncoder._FlowValue,
            right: _ReferencedGlobalIntegrityEncoder._FlowValue,
        ) -> _ReferencedGlobalIntegrityEncoder._FlowValue:
            if left == right:
                return left
            refs = set((*alternatives(left), *alternatives(right)))
            if not refs:
                return unknown
            ordered_refs = tuple(sorted(refs, key=lambda item: (item.root_kind, item.root_name, repr(item.path))))
            return ordered_refs[0] if len(ordered_refs) == 1 else self._FlowValue("multi", alternatives=ordered_refs)

        def merge_state(
            left: _ReferencedGlobalIntegrityEncoder._FlowState,
            right: _ReferencedGlobalIntegrityEncoder._FlowState,
        ) -> _ReferencedGlobalIntegrityEncoder._FlowState:
            width = max(len(left.stack), len(right.stack))
            left_stack = (unknown,) * (width - len(left.stack)) + left.stack
            right_stack = (unknown,) * (width - len(right.stack)) + right.stack
            merged_stack = tuple(union_value(a, b) for a, b in zip(left_stack, right_stack, strict=True))
            left_locals = dict(left.locals)
            right_locals = dict(right.locals)
            merged_locals = {
                name: union_value(left_locals.get(name, unknown), right_locals.get(name, unknown))
                for name in sorted(set(left_locals) | set(right_locals))
            }
            merged_locals = {name: item for name, item in merged_locals.items() if item.kind in {"ref", "multi"}}
            return self._FlowState(merged_stack, tuple(merged_locals.items()))

        def enqueue(index: int, state: _ReferencedGlobalIntegrityEncoder._FlowState) -> None:
            previous = accepted.get(index)
            merged = state if previous is None else merge_state(previous, state)
            if previous != merged:
                accepted[index] = merged
                pending.append(index)

        def ref_dependency(symbol: _ReferencedGlobalIntegrityEncoder._FlowValue, deep: bool = False) -> None:
            for reference in alternatives(symbol):
                dependencies.add((reference.root_kind, reference.root_name, reference.path, deep))

        def path_can_continue(reference: _ReferencedGlobalIntegrityEncoder._FlowValue) -> bool:
            current = roots[(reference.root_kind, reference.root_name)]
            seen_objects = [current]
            for operation, operand in reference.path:
                if operation == "attr" and isinstance(operand, str):
                    found, current = self._static_attribute(current, operand)
                    if not found or isinstance(current, property):
                        return False
                elif operation == "subscript":
                    key = self._decode_subscript_key(operand)
                    if isinstance(current, dict) and dict.__contains__(current, key):
                        current = dict.__getitem__(current, key)
                    elif isinstance(current, tuple) and isinstance(key, int):
                        try:
                            current = tuple.__getitem__(current, key)
                        except IndexError:
                            return False
                    elif isinstance(current, list) and isinstance(key, int):
                        try:
                            current = list.__getitem__(current, key)
                        except IndexError:
                            return False
                    else:
                        return False
                else:
                    return False
                if current is None or isinstance(current, (bool, int, float, complex, str, bytes)):
                    return False
                if any(current is item for item in seen_objects):
                    return False
                seen_objects.append(current)
            return True

        while pending:
            index = pending.pop()
            state = accepted[index]
            processed_states += 1
            processed_by_offset[index] = processed_by_offset.get(index, 0) + 1
            if processed_states > MAX_ORACLE_BYTECODE_STATES:
                raise EvidenceBlocked(
                    "I0-BLOCK-ORACLE-BINDING",
                    self._artifact_id,
                    f"oracle dependency traversal exceeded bytecode-states={MAX_ORACLE_BYTECODE_STATES} "
                    f"limit in {value.__module__}.{value.__qualname__} "
                    f"({len(instructions)} instructions, {len(accepted)} offsets, "
                    f"hottest={max(processed_by_offset.values())})",
                )
            instruction = instructions[index]
            opname = instruction.opname
            stack = list(state.stack)
            local_values = dict(state.locals)

            def pop() -> _ReferencedGlobalIntegrityEncoder._FlowValue:
                return stack.pop() if stack else unknown

            if opname == "LOAD_GLOBAL" and isinstance(instruction.argval, str):
                name = instruction.argval
                if name in value.__globals__:
                    root_kind, binding = "global", value.__globals__[name]
                elif name in builtins.__dict__:
                    root_kind, binding = "builtin", builtins.__dict__[name]
                else:
                    root_kind, binding = "missing-global", _MISSING
                roots[(root_kind, name)] = binding
                symbol = self._FlowValue("ref", root_kind, name)
                stack.append(symbol)
                ref_dependency(symbol)
            elif opname == "LOAD_DEREF" and isinstance(instruction.argval, str):
                name = instruction.argval
                if name in closure_values:
                    roots[("closure", name)] = closure_values[name]
                    symbol = self._FlowValue("ref", "closure", name)
                else:
                    symbol = local_values.get(name, unknown)
                stack.append(symbol)
                ref_dependency(symbol)
            elif opname == "LOAD_FAST" and isinstance(instruction.argval, str):
                stack.append(local_values.get(instruction.argval, unknown))
            elif opname == "STORE_FAST" and isinstance(instruction.argval, str):
                stored = pop()
                if stored.kind in {"ref", "multi"}:
                    local_values[instruction.argval] = stored
                else:
                    # Non-reference locals cannot contribute a dependency path;
                    # dropping them prevents irrelevant branch values from
                    # multiplying abstract states at control-flow joins.
                    local_values.pop(instruction.argval, None)
            elif opname == "DELETE_FAST" and isinstance(instruction.argval, str):
                local_values.pop(instruction.argval, None)
            elif opname == "LOAD_CONST":
                stack.append(self._FlowValue("const", const_index=int(instruction.arg or 0)))
            elif opname == "PUSH_NULL":
                stack.append(null)
            elif opname in {"LOAD_ATTR", "LOAD_METHOD"} and isinstance(instruction.argval, str):
                base = pop()
                selected: list[_ReferencedGlobalIntegrityEncoder._FlowValue] = []
                for reference in alternatives(base):
                    if len(reference.path) >= self._budget.max_depth:
                        raise EvidenceBlocked(
                            "I0-BLOCK-ORACLE-BINDING",
                            self._artifact_id,
                            f"oracle dependency traversal exceeded depth={self._budget.max_depth} limit "
                            f"in {value.__module__}.{value.__qualname__} at {reference.root_name}",
                        )
                    selected.append(self._FlowValue(
                        "ref", reference.root_kind, reference.root_name,
                        (*reference.path, ("attr", instruction.argval)),
                    ))
                if selected:
                    propagated = [reference for reference in selected if path_can_continue(reference)]
                    base = selected[0]
                    for alternate in selected[1:]:
                        base = union_value(base, alternate)
                    ref_dependency(base)
                    if propagated:
                        base = propagated[0]
                        for alternate in propagated[1:]:
                            base = union_value(base, alternate)
                    else:
                        base = unknown
                stack.append(base)
            elif opname == "BINARY_SUBSCR":
                key = pop()
                base = pop()
                selected = []
                if key.kind == "const" and 0 <= key.const_index < len(code.co_consts):
                    constant = code.co_consts[key.const_index]
                    if not isinstance(constant, (str, bytes, int, bool, type(None))):
                        raise EvidenceBlocked(
                            "I0-BLOCK-ORACLE-BINDING",
                            self._artifact_id,
                            f"unsupported constant subscript type {type(constant).__qualname__}",
                        )
                    operation = ("subscript", (type(constant).__name__, constant))
                else:
                    operation = ("subscript-dynamic", "*")
                for reference in alternatives(base):
                    selected.append(self._FlowValue(
                        "ref", reference.root_kind, reference.root_name, (*reference.path, operation)
                    ))
                if selected:
                    base = selected[0]
                    for alternate in selected[1:]:
                        base = union_value(base, alternate)
                    ref_dependency(base, deep=operation[0] == "subscript-dynamic")
                stack.append(base)
            elif opname in {"GET_ITER", "CONTAINS_OP"}:
                if opname == "CONTAINS_OP":
                    pop()
                base = pop()
                ref_dependency(base, deep=True)
                stack.append(base if opname == "GET_ITER" else unknown)
            elif opname == "FOR_ITER":
                # Branch-specific stack effects are applied when successors are
                # enqueued below; keeping one shared effect would grow the stack
                # on every loop back-edge and defeat bounded convergence.
                pass
            elif opname == "COPY" and instruction.arg:
                stack.append(stack[-instruction.arg] if len(stack) >= instruction.arg else unknown)
            elif opname == "SWAP" and instruction.arg and len(stack) >= instruction.arg:
                stack[-1], stack[-instruction.arg] = stack[-instruction.arg], stack[-1]
            elif opname in {"CALL", "CALL_FUNCTION_EX"}:
                # All callable/argument dependencies have already been emitted.
                # CPython 3.11+ method-call stack layouts vary by LOAD_ATTR flag;
                # retaining a guessed residual stack leaks unrelated receivers
                # across control-flow joins and creates false dependency paths.
                # A call result is runtime data, so reset to one unknown value.
                stack[:] = [unknown]
            elif opname in {"BINARY_OP", "COMPARE_OP", "IS_OP"}:
                pop()
                pop()
                stack.append(unknown)
            elif opname.startswith("STORE_") or opname.startswith("DELETE_"):
                # Mutating bytecode consumes its operands; dependencies were
                # already emitted at their loads and selected paths.
                try:
                    effect = dis.stack_effect(instruction.opcode, instruction.arg)
                except ValueError:
                    effect = dis.stack_effect(instruction.opcode)
                for _ in range(min(len(stack), max(0, -effect))):
                    pop()
            elif opname not in {
                "CACHE", "RESUME", "NOP", "EXTENDED_ARG", "PRECALL",
                "RETURN_VALUE", "RETURN_CONST", "POP_TOP",
            }:
                try:
                    effect = dis.stack_effect(instruction.opcode, instruction.arg)
                except ValueError:
                    effect = dis.stack_effect(instruction.opcode)
                if effect < 0:
                    for _ in range(min(len(stack), -effect)):
                        pop()
                elif effect > 0:
                    stack.extend([unknown] * effect)
            elif opname == "POP_TOP":
                pop()

            next_state = self._FlowState(tuple(stack), tuple(sorted(local_values.items())))
            terminal = opname in {"RETURN_VALUE", "RETURN_CONST", "RAISE_VARARGS", "RERAISE"}
            target_index = (
                offset_to_index.get(instruction.argval)
                if isinstance(instruction.argval, int) and ("JUMP" in opname or opname == "FOR_ITER")
                else None
            )
            unconditional = opname.startswith("JUMP") and "IF" not in opname
            if opname == "FOR_ITER":
                if target_index is not None:
                    exhausted_stack = stack[:-1] if stack else []
                    enqueue(
                        target_index,
                        self._FlowState(tuple(exhausted_stack), tuple(sorted(local_values.items()))),
                    )
                if index + 1 < len(instructions):
                    enqueue(
                        index + 1,
                        self._FlowState(tuple((*stack, unknown)), tuple(sorted(local_values.items()))),
                    )
            else:
                successors: list[int] = []
                if target_index is not None:
                    successors.append(target_index)
                if not terminal and not unconditional and index + 1 < len(instructions):
                    successors.append(index + 1)
                for successor in reversed(sorted(set(successors))):
                    enqueue(successor, next_state)

        ordered = tuple(sorted(
            dependencies,
            key=lambda item: (item[0], item[1], repr(item[2]), item[3]),
        ))
        return ordered, roots, code.co_consts

    @staticmethod
    def _decode_subscript_key(token: object) -> object:
        if not isinstance(token, tuple) or len(token) != 2:
            raise TypeError("invalid subscript token")
        return token[1]

    def _add_resolved_dependency(
        self,
        root_kind: str,
        root_name: str,
        path: tuple[tuple[str, object], ...],
        force_deep: bool,
        roots: dict[tuple[str, str], object],
        depth: int,
    ) -> None:
        binding = roots[(root_kind, root_name)]
        self._field(b"dependency-root-kind", root_kind.encode("ascii"))
        self._field(b"dependency-root-name", self._encoded_text(root_name, "utf-8", "surrogatepass"))
        if root_kind == "missing-global":
            self._field(b"dependency-global-missing")
            return
        forced_paths = frozenset(
            dependency_path
            for module_name, global_name, dependency_path in ORACLE_DEEP_MUTABLE_DEPENDENCIES
            if module_name == self._root_module and global_name == root_name and root_kind == "global"
        )
        self.add_binding(binding, depth + 1, deep_container=() in forced_paths or (force_deep and not path))
        current = binding
        attribute_prefix: tuple[str, ...] = ()
        for operation_index, (operation, operand) in enumerate(path):
            is_last = operation_index == len(path) - 1
            if operation == "attr":
                assert isinstance(operand, str)
                attribute_prefix = (*attribute_prefix, operand)
                self._field(b"dependency-attr", self._encoded_text(operand, "utf-8", "surrogatepass"))
                found, selected = self._static_attribute(current, operand)
                if not found:
                    if root_kind == "receiver":
                        self._field(b"runtime-receiver-attribute")
                    else:
                        # Absence is itself an explicit endpoint fingerprint;
                        # adding the attribute later changes this marker to the
                        # resolved value and therefore blocks integrity.
                        self._field(b"dependency-attribute-missing")
                    return
                current = selected
            elif operation == "subscript":
                key = self._decode_subscript_key(operand)
                self._field(b"dependency-subscript")
                self.add(key, depth + 1)
                if isinstance(current, dict):
                    if not dict.__contains__(current, key):
                        self._field(b"dependency-subscript-missing")
                        return
                    current = dict.__getitem__(current, key)
                elif isinstance(current, (tuple, list)) and isinstance(key, int):
                    length = tuple.__len__(current) if isinstance(current, tuple) else list.__len__(current)
                    if not -length <= key < length:
                        self._field(b"dependency-subscript-missing")
                        return
                    current = (
                        tuple.__getitem__(current, key)
                        if isinstance(current, tuple)
                        else list.__getitem__(current, key)
                    )
                elif (
                    type(current).__module__ == "os"
                    and type(current).__qualname__ == "_Environ"
                    and isinstance(key, str)
                ):
                    # Resolve os.environ without invoking a replaceable mapping
                    # callback.  _Environ stores encoded keys/values in its
                    # private exact dict; selected keys are fingerprinted, not
                    # the unrelated process environment.
                    data = object.__getattribute__(current, "_data")
                    encoded_key = os.fsencode(key)
                    if encoded_key not in data:
                        self._field(b"dependency-subscript-missing")
                        return
                    current = dict.__getitem__(data, encoded_key)
                elif root_kind == "receiver":
                    self._field(b"runtime-receiver-subscript")
                    return
                elif self._static_attribute(current, "__getitem__")[0]:
                    self._field(b"dependency-custom-subscript")
                    self.add_binding(
                        self._static_attribute(current, "__getitem__")[1],
                        depth + 1,
                        deep_container=False,
                    )
                    return
                elif current is None or isinstance(current, (bool, int, float, complex, str, bytes)):
                    # Control-flow widening can pair a guarded subscript with an
                    # immutable non-container alternative.  Its exact value was
                    # fingerprinted at the preceding prefix, so record this
                    # explicit non-applicable endpoint rather than pretending an
                    # identity-only container was covered.
                    self._field(b"dependency-subscript-non-container")
                    return
                else:
                    raise EvidenceBlocked(
                        "I0-BLOCK-ORACLE-BINDING",
                        self._artifact_id,
                        f"unsupported static subscript on {type(current).__qualname__}",
                    )
            elif operation == "subscript-dynamic":
                self._field(b"dependency-dynamic-subscript")
                if isinstance(current, (dict, list, tuple, set, frozenset)):
                    self.add_binding(current, depth + 1, deep_container=True)
                    return
                if root_kind == "receiver":
                    self._field(b"runtime-receiver-dynamic-subscript")
                    return
                if isinstance(current, type):
                    found, implementation = self._static_attribute(current, "__class_getitem__")
                    if not found:
                        found, implementation = self._static_attribute(current, "__getitem__")
                    self._field(b"dependency-type-dynamic-subscript")
                    if found:
                        self.add_binding(implementation, depth + 1, deep_container=False)
                    return
                found, implementation = self._static_attribute(current, "__getitem__")
                if found:
                    self._field(b"dependency-custom-dynamic-subscript")
                    self.add_binding(implementation, depth + 1, deep_container=False)
                    return
                raise EvidenceBlocked(
                    "I0-BLOCK-ORACLE-BINDING",
                    self._artifact_id,
                    f"unsupported dynamic subscript on {type(current).__qualname__}",
                )
            else:
                raise EvidenceBlocked(
                    "I0-BLOCK-ORACLE-BINDING", self._artifact_id, f"unsupported dependency operation {operation}"
                )

            volatile_import_registry = (
                isinstance(binding, ModuleType)
                and binding.__name__ == "sys"
                and attribute_prefix[:1] in {("meta_path",), ("modules",), ("path_importer_cache",)}
            )
            deep = (attribute_prefix in forced_paths or (force_deep and is_last)) and not volatile_import_registry
            self.add_binding(current, depth + 1, deep_container=deep)
            if isinstance(current, property) and not is_last:
                # The descriptor implementation is fingerprinted above.  Its
                # runtime result cannot be resolved without executing arbitrary
                # code, so close the path explicitly here rather than treating
                # the property object as that result.
                self._field(b"runtime-descriptor-result")
                return

    def add_referenced_function(
        self,
        value: FunctionType,
        depth: int = 0,
        *,
        bound_receiver: object | None = None,
    ) -> None:
        self._budget.visit(self._artifact_id, depth)
        if self._reference(value):
            return
        self._field(
            b"referenced-function",
            self._encoded_text(
                f"{value.__module__}:{value.__qualname__}", "utf-8", "backslashreplace"
            ),
        )
        self._field(b"function-code")
        self._add_code(value.__code__, depth + 1)
        self.add_binding(value.__defaults__, depth + 1)
        self.add_binding(value.__kwdefaults__, depth + 1)
        self.add_binding(value.__annotations__, depth + 1)
        self.add_binding(value.__dict__, depth + 1)
        if value.__closure__ is None:
            self._field(b"closure-none")
        else:
            self._budget.add_container_items(self._artifact_id, len(value.__closure__))
            self._field(b"closure", str(len(value.__closure__)).encode("ascii"))
            for cell in value.__closure__:
                try:
                    self.add_binding(cell.cell_contents, depth + 1)
                except ValueError:
                    self._field(b"empty-cell")

        receiver = bound_receiver
        if receiver is None and self._trusted_class is not None and value.__code__.co_argcount:
            first_name = value.__code__.co_varnames[0]
            if first_name in {"self", "cls"}:
                receiver = self._trusted_class
        dependencies, roots, _constants = self._bytecode_dependencies(value, receiver)
        self._budget.add_container_items(self._artifact_id, len(dependencies))
        self._field(b"bytecode-dependency-count", str(len(dependencies)).encode("ascii"))
        for root_kind, root_name, path, force_deep in dependencies:
            self._add_resolved_dependency(
                root_kind, root_name, path, force_deep, roots, depth + 1
            )

    def add_module_binding(self, value: ModuleType, depth: int = 0) -> None:
        self._budget.visit(self._artifact_id, depth)
        if self._reference(value):
            return
        spec = getattr(value, "__spec__", None)
        origin = getattr(spec, "origin", None)
        loader = getattr(spec, "loader", None)
        self._field_text(
            b"module-binding",
            f"{value.__name__}:{id(value)}:{origin!r}:"
            f"{type(loader).__module__}:{type(loader).__qualname__}:{id(loader)}",
            "utf-8",
            "backslashreplace",
        )
        locked = self._locked_sources.get(value.__name__)
        if locked is not None:
            self._field_text(b"locked-module-source", f"{locked[0]}:{locked[1]}", "utf-8", "surrogatepass")

    def add_class_binding(self, value: type, depth: int = 0) -> None:
        self._budget.visit(self._artifact_id, depth)
        if self._reference(value):
            return
        self._field_text(
            b"class-binding",
            f"{value.__module__}:{value.__qualname__}:{id(value)}",
            "utf-8",
            "backslashreplace",
        )
        mro = tuple(value.__mro__)
        self._budget.add_container_items(self._artifact_id, len(mro))
        for owner in mro:
            self._field_text(
                b"class-mro",
                f"{owner.__module__}:{owner.__qualname__}:{id(owner)}",
                "utf-8",
                "backslashreplace",
            )
        # Referenced framework classes can own legitimate mutable registries on
        # unrelated attributes.  Bind MRO identities plus explicitly selected
        # constructors here; bytecode-selected methods are dependency-closed by
        # the data-flow walker without traversing unrelated base-class caches.
        if (value.__module__, value.__qualname__) in ORACLE_CONSTRUCTOR_DEPENDENCIES:
            for constructor_name in ("__new__", "__init__"):
                found, constructor = self._static_attribute(value, constructor_name)
                if not found:
                    continue
                self._field(b"class-constructor", constructor_name.encode("ascii"))
                if isinstance(constructor, FunctionType):
                    self.add_referenced_function(constructor, depth + 1)
                else:
                    self._field(
                        b"class-constructor-identity",
                        f"{type(constructor).__module__}:{type(constructor).__qualname__}:{id(constructor)}".encode(),
                    )

    def add_binding(
        self,
        value: object,
        depth: int = 0,
        *,
        deep_container: bool = True,
    ) -> None:
        self._budget.visit(self._artifact_id, depth)
        if value is None:
            self._field(b"NoneType")
            return
        if isinstance(value, str):
            self._field(b"str", self._encoded_text(value, "utf-8", "surrogatepass"))
            return
        if isinstance(value, bytes):
            self._budget.ensure_encoded_allocation(self._artifact_id, len(value))
            self._field(b"bytes", value)
            return
        if isinstance(value, (bool, int, float, complex)):
            if isinstance(value, int) and value.bit_length() > self._budget.remaining_encoded_bytes() * 3:
                raise self._budget._blocked(self._artifact_id, f"encoded-bytes={self._budget.max_encoded_bytes}")
            self._field(type(value).__name__.encode(), self._encoded_text(repr(value), "ascii"))
            return
        if isinstance(value, (bytearray, memoryview)):
            self._field(type(value).__name__.encode(), self._copied_bytes(value))
            return
        if isinstance(value, CodeType):
            self._add_code(value, depth)
            return
        if isinstance(value, MethodType):
            if self._reference(value):
                return
            self._field(b"bound-method")
            function = value.__func__
            if isinstance(function, FunctionType):
                self.add_referenced_function(function, depth + 1, bound_receiver=value.__self__)
            else:
                self._field(
                    b"bound-method-implementation",
                    f"{type(function).__module__}:{type(function).__qualname__}:{id(function)}".encode(),
                )
            self.add_binding(value.__self__, depth + 1, deep_container=False)
            return
        if isinstance(value, FunctionType):
            self.add_referenced_function(value, depth)
            return
        if isinstance(value, (BuiltinFunctionType, BuiltinMethodType)):
            receiver = getattr(value, "__self__", None)
            self._field(
                b"builtin-callable",
                self._encoded_text(
                    f"{getattr(value, '__module__', '')}:{getattr(value, '__qualname__', '')}:"
                    f"{getattr(value, '__name__', '')}:"
                    f"{id(receiver) if receiver is not None else 0}",
                    "utf-8",
                    "backslashreplace",
                ),
            )
            if receiver is not None and isinstance(receiver, (dict, list, set, bytearray)):
                self.add_binding(receiver, depth + 1, deep_container=deep_container)
            return
        if isinstance(value, type):
            self.add_class_binding(value, depth)
            return
        if isinstance(value, ModuleType):
            self.add_module_binding(value, depth)
            return
        if isinstance(value, (staticmethod, classmethod)):
            self._field(type(value).__name__.encode())
            function = value.__func__
            if isinstance(function, FunctionType):
                self.add_referenced_function(function, depth + 1)
            else:
                self._field(
                    b"descriptor-implementation",
                    f"{type(function).__module__}:{type(function).__qualname__}:{id(function)}".encode(),
                )
            return
        if isinstance(value, property):
            self._field(b"property")
            for function in (value.fget, value.fset, value.fdel):
                self.add_binding(function, depth + 1)
            return
        if isinstance(value, (tuple, list)):
            if self._reference(value):
                return
            if not deep_container:
                self._field(
                    b"external-container-identity",
                    f"{type(value).__module__}:{type(value).__qualname__}:{id(value)}".encode(),
                )
                return
            self._budget.add_container_items(self._artifact_id, len(value))
            self._field(type(value).__name__.encode(), str(len(value)).encode("ascii"))
            for item in value:
                self.add_binding(item, depth + 1, deep_container=True)
            return
        if isinstance(value, dict):
            if self._reference(value):
                return
            if not deep_container:
                self._field(
                    b"external-container-identity",
                    f"{type(value).__module__}:{type(value).__qualname__}:{id(value)}".encode(),
                )
                return
            self._budget.add_container_items(self._artifact_id, len(value))
            self._field(b"dict", str(len(value)).encode("ascii"))
            items = [
                (self._sort_token(key), key, item_value)
                for key, item_value in list(dict.items(value))
            ]
            for (module_name, type_name, payload), key, item_value in sorted(items, key=lambda item: item[0]):
                self._field(b"key-module", self._encoded_text(module_name, "utf-8", "backslashreplace"))
                self._field(b"key-type", self._encoded_text(type_name, "utf-8", "backslashreplace"))
                self._field(b"key-payload", payload)
                self.add_binding(item_value, depth + 1, deep_container=True)
            return
        if isinstance(value, (set, frozenset)):
            if self._reference(value):
                return
            if not deep_container:
                self._field(
                    b"external-container-identity",
                    f"{type(value).__module__}:{type(value).__qualname__}:{id(value)}".encode(),
                )
                return
            self._budget.add_container_items(self._artifact_id, len(value))
            children = sorted(((self._sort_token(item), item) for item in value), key=lambda pair: pair[0])
            self._field(type(value).__name__.encode(), str(len(children)).encode("ascii"))
            for _sort_key, item in children:
                self.add_binding(item, depth + 1)
            return

        # Callable objects are behavior endpoints.  Close their selected
        # ``__call__`` implementation and receiver paths rather than reducing
        # them to identity-only fingerprints.
        if callable(value):
            if self._reference(value):
                return
            is_lru_wrapper = (
                type(value).__module__ == "functools"
                and type(value).__qualname__ == "_lru_cache_wrapper"
            )
            wrapped = object.__getattribute__(value, "__wrapped__") if is_lru_wrapper else None
            if is_lru_wrapper and isinstance(wrapped, FunctionType):
                # The wrapper's identity/cache is recreated and warmed by
                # ordinary imports.  Its behavior is the wrapped Python
                # function, which is fully dependency-closed; cache contents
                # are deliberately not oracle state.
                self._field(b"lru-cache-wrapper")
                self.add_referenced_function(wrapped, depth + 1)
                return
            self._field(
                b"callable-object",
                self._encoded_text(
                    f"{type(value).__module__}:{type(value).__qualname__}:{id(value)}",
                    "utf-8",
                    "backslashreplace",
                ),
            )
            found, implementation = self._static_attribute(value, "__call__")
            if not found:
                raise EvidenceBlocked(
                    "I0-BLOCK-ORACLE-BINDING", self._artifact_id, "callable object has no static __call__"
                )
            if isinstance(implementation, MethodType) and isinstance(implementation.__func__, FunctionType):
                self.add_referenced_function(
                    implementation.__func__, depth + 1, bound_receiver=implementation.__self__
                )
            elif isinstance(implementation, FunctionType):
                self.add_referenced_function(implementation, depth + 1, bound_receiver=value)
            else:
                self.add_binding(implementation, depth + 1, deep_container=False)
            return

        # Non-callable framework objects may own unrelated mutable caches.  Bind
        # identity here; any bytecode-selected attribute/subscript endpoint was
        # separately and explicitly fingerprinted above.
        self._field(
            b"object-binding",
            self._encoded_text(
                f"{type(value).__module__}:{type(value).__qualname__}:{id(value)}",
                "utf-8",
                "backslashreplace",
            ),
        )


def _module_global_integrity_fingerprint(
    trusted_class: type,
    locked_sources: dict[str, tuple[str, str]],
) -> str:
    artifact_id = f"{trusted_class.__module__}.{trusted_class.__qualname__}"
    budget = _OracleTraversalBudget()
    encoder = _ReferencedGlobalIntegrityEncoder(
        locked_sources,
        trusted_class.__module__,
        budget,
        artifact_id,
        trusted_class,
    )
    encoder._field_text(
        b"trusted-class",
        f"{trusted_class.__module__}:{trusted_class.__qualname__}",
        "utf-8",
        "surrogatepass",
    )
    functions = _class_functions(trusted_class, include_inherited=False)
    selected_names = tuple(sorted(
        method_name
        for module_name, class_name, method_name in ORACLE_SELECTED_INHERITED_METHOD_DEPENDENCIES
        if module_name == trusted_class.__module__ and class_name == trusted_class.__qualname__
    ))
    budget.add_container_items(artifact_id, len(functions))
    budget.add_container_items(artifact_id, len(selected_names))
    encoder._field(b"trusted-method-count", str(len(functions)).encode("ascii"))
    encoder._field(b"selected-inherited-count", str(len(selected_names)).encode("ascii"))
    try:
        for role, function in functions:
            encoder._field(b"trusted-method", encoder._encoded_text(role, "utf-8", "surrogatepass"))
            encoder.add_referenced_function(function, 1)
        for method_name in selected_names:
            found, selected = encoder._static_attribute(trusted_class, method_name)
            if not found:
                raise EvidenceBlocked(
                    "I0-BLOCK-ORACLE-BINDING",
                    artifact_id,
                    f"selected inherited method {method_name!r} is absent",
                )
            encoder._field(
                b"selected-inherited-method",
                encoder._encoded_text(method_name, "utf-8", "surrogatepass"),
            )
            if isinstance(selected, MethodType) and isinstance(selected.__func__, FunctionType):
                encoder.add_referenced_function(
                    selected.__func__, 1, bound_receiver=selected.__self__
                )
            elif isinstance(selected, FunctionType):
                encoder.add_referenced_function(selected, 1, bound_receiver=trusted_class)
            else:
                encoder.add_binding(selected, 1, deep_container=False)
    except RecursionError as error:
        raise EvidenceBlocked(
            "I0-BLOCK-ORACLE-BINDING",
            artifact_id,
            "oracle dependency traversal reached Python recursion before its declared depth limit",
        ) from error
    return encoder.digest.hexdigest()


class _OracleTrustSnapshot:
    """Process-lifetime class identities minted at one clean authenticated import boundary."""

    def __init__(
        self,
        lock: dict[str, Any],
        assert_import_state: Callable[[str], None],
    ) -> None:
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.owned_side_effect_finders: tuple[Any, ...] = ()
        self.locked_sources: dict[str, tuple[str, str]] = {}
        self.lock_fingerprint = self._lock_fingerprint(lock)
        self._establish(lock, assert_import_state)

    @staticmethod
    def _blocked(export: str, detail: str) -> EvidenceBlocked:
        return EvidenceBlocked("I0-BLOCK-ORACLE-BINDING", export, detail)

    @staticmethod
    def _lock_fingerprint(lock: dict[str, Any]) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (package["name"], package["version"], identity["path"] + ":" + identity["sha256"])
            for package in lock["oracle_packages"]
            for identity in (package["root_source"], *package["files"])
        )

    def assert_lock(self, lock: dict[str, Any]) -> None:
        if self._lock_fingerprint(lock) != self.lock_fingerprint:
            raise self._blocked("oracle-lock", "later context does not share the established oracle trust lock")

    def discard_owned_import_side_effect_finders(self, baseline: tuple[Any, ...]) -> None:
        """Remove only exact six importers authenticated during first establishment."""
        for finder in self.owned_side_effect_finders:
            if not any(finder is item for item in baseline):
                _remove_identity_from_meta_path(finder)

    @staticmethod
    def _module_name(package_name: str, relative: Path) -> tuple[str, bool]:
        parts = relative.parts
        if len(parts) < 2 or parts[0] != "src" or parts[1] != package_name:
            raise EvidenceBlocked(
                "I0-BLOCK-ORACLE-BINDING", str(relative), "oracle source path is not canonical under src/package"
            )
        suffix = parts[2:]
        if not suffix:
            raise EvidenceBlocked("I0-BLOCK-ORACLE-BINDING", str(relative), "oracle source path is incomplete")
        if suffix[-1] == "__init__.py":
            return ".".join((package_name, *suffix[:-1])), True
        if not suffix[-1].endswith(".py"):
            raise EvidenceBlocked("I0-BLOCK-ORACLE-BINDING", str(relative), "oracle source is not Python")
        return ".".join((package_name, *suffix[:-1], suffix[-1][:-3])), False

    @staticmethod
    def _new_six_importers(baseline: tuple[Any, ...]) -> tuple[Any, ...]:
        candidates: list[Any] = []
        for module_name in ("six", "botocore.vendored.six"):
            module = sys.modules.get(module_name)
            candidate = getattr(module, "_importer", None) if module is not None else None
            if candidate is None or any(candidate is item for item in baseline):
                continue
            if any(candidate is finder for finder in sys.meta_path) and not any(
                candidate is item for item in candidates
            ):
                candidates.append(candidate)
        return tuple(candidates)

    def _rollback_failed_establishment(
        self,
        baseline_modules: dict[str, ModuleType],
        baseline_meta_path: tuple[Any, ...],
        finder: _AuthenticatedOracleFinder | None,
        owned_side_finders: tuple[Any, ...],
    ) -> list[BaseException]:
        failures: list[BaseException] = []
        if finder is not None:
            _remove_identity_from_meta_path(finder)
        for side_finder in owned_side_finders:
            _remove_identity_from_meta_path(side_finder)

        packages = {binding["package"] for binding in ORACLE_EXPORT_BINDINGS}
        for name, module in tuple(sys.modules.items()):
            if name in baseline_modules or not any(
                name == package or name.startswith(package + ".") for package in packages
            ):
                continue
            # All target-package names were absent at the clean boundary.  Pop
            # only the exact object introduced by this attempt.
            if sys.modules.get(name) is module:
                sys.modules.pop(name, None)
        preserved_additions = sorted(
            name
            for name in sys.modules
            if name not in baseline_modules
            and not any(name == package or name.startswith(package + ".") for package in packages)
        )
        if preserved_additions:
            failures.append(self._blocked(
                preserved_additions[0],
                "non-target sys.modules additions were preserved and reported during failed-establishment rollback",
            ))
        for name, original in baseline_modules.items():
            if sys.modules.get(name, _MISSING) is not original:
                failures.append(self._blocked(name, "foreign sys.modules mutation was preserved during rollback"))

        if not _identity_sequence_matches(sys.meta_path, baseline_meta_path):
            failures.append(self._blocked("sys.meta_path", "foreign finder mutation was preserved during rollback"))
        self.snapshots.clear()
        self.owned_side_effect_finders = ()
        if finder is not None:
            finder.release()
        return failures

    def _establish(self, lock: dict[str, Any], assert_import_state: Callable[[str], None]) -> None:
        package_names = {binding["package"] for binding in ORACLE_EXPORT_BINDINGS}
        preloaded = sorted(
            name
            for name in sys.modules
            if any(name == package or name.startswith(package + ".") for package in package_names)
        )
        if preloaded:
            raise self._blocked(
                preloaded[0],
                "target oracle package/module was preloaded before trust initialization",
            )

        baseline_modules = dict(sys.modules)
        baseline_meta_path = tuple(sys.meta_path)
        baseline_import = builtins.__import__
        finder: _AuthenticatedOracleFinder | None = None
        owned_side_finders: tuple[Any, ...] = ()
        direct_import_patch: tuple[ModuleType, Any] | None = None
        spec_from_file_patch: Any | None = None
        try:
            # Every lock-listed module is read once.  The retained bytes that
            # produced the locked digest are exactly the bytes later compiled.
            authenticated_paths: dict[tuple[str, str], Path] = {}
            controlled_sources: dict[str, tuple[Path, bytes, bool]] = {}
            locked_sources: dict[str, tuple[str, str]] = {}
            packages = {item["name"]: item for item in lock["oracle_packages"]}
            for package_name, package in packages.items():
                distribution = importlib.metadata.distribution(package_name)
                if distribution.version != package["version"]:
                    raise self._blocked(package_name, "installed package version differs from the reference lock")
                package_root = Path(distribution.locate_file(package_name)).resolve()
                root_identity = package["root_source"]
                if root_identity["path"] != f"src/{package_name}/__init__.py":
                    raise self._blocked(package_name, "package-root source lock path is not canonical")
                root_source = package_root / "__init__.py"
                root_bytes = _read_authenticated_oracle_bytes(
                    root_source, root_identity["sha256"], root_identity["path"]
                )
                controlled_sources[package_name] = (root_source, root_bytes, True)
                locked_sources[package_name] = (str(root_source), root_identity["sha256"])
                for identity in package["files"]:
                    relative = Path(identity["path"])
                    module_name, is_package = self._module_name(package_name, relative)
                    installed = Path(distribution.locate_file(Path(*relative.parts[1:]))).resolve()
                    data = _read_authenticated_oracle_bytes(installed, identity["sha256"], identity["path"])
                    authenticated_paths[(package_name, identity["path"])] = installed
                    controlled_sources[module_name] = (installed, data, is_package)
                    locked_sources[module_name] = (str(installed), identity["sha256"])
            self.locked_sources = locked_sources

            assert_import_state("before oracle imports")
            finder = _AuthenticatedOracleFinder(controlled_sources)
            sys.meta_path.insert(0, finder)
            expected_meta_path = (finder, *baseline_meta_path)
            # Import both exact package roots first.  Otherwise a dependency
            # probe nested inside one oracle can establish the other root via
            # an ordinary cached spec before its target binding is requested.
            for package_name in packages:
                if builtins.__import__ is not baseline_import or not _identity_sequence_matches(
                    sys.meta_path, expected_meta_path
                ):
                    raise self._blocked(package_name, "import state changed before controlled package-root import")
                root_module = importlib.import_module(package_name)
                newly_owned = self._new_six_importers(baseline_meta_path)
                for side_finder in newly_owned:
                    if not any(side_finder is item for item in owned_side_finders):
                        owned_side_finders = (*owned_side_finders, side_finder)
                    _remove_identity_from_meta_path(side_finder)
                root_spec = getattr(root_module, "__spec__", None)
                if (
                    getattr(root_spec, "loader", None) is not finder
                    or finder.executed.get(package_name) is not root_module
                    or Path(getattr(root_spec, "origin", "")).resolve() != controlled_sources[package_name][0]
                ):
                    raise self._blocked(package_name, "package root did not execute retained locked source bytes")

            # transformers.processing_utils intentionally bypasses meta_path via
            # direct_transformers_import() and a fresh SourceFileLoader.  Bind
            # that upstream escape hatch to this retained-byte loader while the
            # authenticated oracle graph is being established.
            import_utils_module = sys.modules.get("transformers.utils.import_utils")
            original_direct_import = getattr(import_utils_module, "direct_transformers_import", None)
            if not isinstance(import_utils_module, ModuleType) or not callable(original_direct_import):
                raise self._blocked("transformers", "direct import helper is unavailable for retained-byte binding")

            def retained_direct_transformers_import(path: str | Path, file: str = "__init__.py") -> ModuleType:
                expected_path, _data, _is_package = controlled_sources["transformers"]
                requested = (Path(path) / file).resolve()
                if requested != expected_path or file != "__init__.py":
                    raise self._blocked("transformers", "direct import requested an unlocked package-root source")
                spec = finder.find_spec("transformers")
                if spec is None:
                    raise self._blocked("transformers", "retained package-root spec disappeared")
                module = importlib.util.module_from_spec(spec)
                finder.exec_module(module)
                current = sys.modules.get("transformers")
                if not isinstance(current, ModuleType):
                    raise self._blocked("transformers", "retained direct import did not establish package root")
                return current

            import_utils_module.direct_transformers_import = retained_direct_transformers_import
            direct_import_patch = (import_utils_module, original_direct_import)
            original_spec_from_file_location = importlib.util.spec_from_file_location

            def retained_spec_from_file_location(name: str, location: object, *args: Any, **kwargs: Any) -> Any:
                expected_path = controlled_sources["transformers"][0]
                try:
                    requested = Path(os.fspath(location)).resolve()
                except TypeError:
                    requested = None
                if name == "transformers" and requested == expected_path:
                    return finder.find_spec("transformers")
                return original_spec_from_file_location(name, location, *args, **kwargs)

            importlib.util.spec_from_file_location = retained_spec_from_file_location
            spec_from_file_patch = original_spec_from_file_location
            for binding in ORACLE_EXPORT_BINDINGS:
                package_name = binding["package"]
                export = binding["export"]
                expected_origin = authenticated_paths[(package_name, binding["source"])]
                if builtins.__import__ is not baseline_import or not _identity_sequence_matches(
                    sys.meta_path, expected_meta_path
                ):
                    raise self._blocked(export, "import state changed before controlled oracle import")
                exact_module = importlib.import_module(binding["module"])
                newly_owned = self._new_six_importers(baseline_meta_path)
                for side_finder in newly_owned:
                    if not any(side_finder is item for item in owned_side_finders):
                        owned_side_finders = (*owned_side_finders, side_finder)
                    _remove_identity_from_meta_path(side_finder)
                if builtins.__import__ is not baseline_import or not _identity_sequence_matches(
                    sys.meta_path, expected_meta_path
                ):
                    raise self._blocked(export, "foreign import-state mutation detected during controlled oracle import")
                root_module = sys.modules.get(package_name)
                if root_module is None:
                    raise self._blocked(export, "controlled import did not establish the package root")
                root_spec = getattr(root_module, "__spec__", None)
                if (
                    getattr(root_spec, "loader", None) is not finder
                    or finder.executed.get(package_name) is not root_module
                    or Path(getattr(root_spec, "origin", "")).resolve() != controlled_sources[package_name][0]
                ):
                    raise self._blocked(export, "package root did not execute retained locked source bytes")
                spec = getattr(exact_module, "__spec__", None)
                origin_text = getattr(spec, "origin", None)
                if (
                    not origin_text
                    or Path(origin_text).resolve() != expected_origin
                    or getattr(spec, "loader", None) is not finder
                    or finder.executed.get(binding["module"]) is not exact_module
                ):
                    raise self._blocked(export, "controlled import did not execute retained authenticated bytes")
                try:
                    exact_object = exact_module.__dict__[export]
                    root_object = getattr(root_module, export)
                except (AttributeError, KeyError) as error:
                    raise self._blocked(export, f"required oracle export is absent: {error}") from error
                if root_object is not exact_object:
                    raise self._blocked(export, "package-root export differs from the freshly imported source class")
                if not inspect.isclass(exact_object) or exact_object.__module__ != binding["module"]:
                    raise self._blocked(export, "fresh source export is not the class declared by the trusted module")
                source_text = inspect.getsourcefile(exact_object)
                if not source_text or Path(source_text).resolve() != expected_origin:
                    raise self._blocked(export, "fresh class code is not bound to the authenticated source file")
                fingerprint, mro = _class_integrity_fingerprint(exact_object)
                locked_digest = next(
                    identity["sha256"]
                    for identity in packages[package_name]["files"]
                    if identity["path"] == binding["source"]
                )
                self.snapshots[export] = {
                    "binding": binding,
                    "root_module": root_module,
                    "exact_module": exact_module,
                    "object": exact_object,
                    "origin": expected_origin,
                    "source_artifact_id": binding["source"],
                    "sha256": locked_digest,
                    "source_size": len(controlled_sources[binding["module"]][1]),
                    "root_origin": controlled_sources[package_name][0],
                    "root_artifact_id": packages[package_name]["root_source"]["path"],
                    "root_sha256": packages[package_name]["root_source"]["sha256"],
                    "root_source_size": len(controlled_sources[package_name][1]),
                    "version": packages[package_name]["version"],
                    "class_fingerprint": fingerprint,
                    "mro": mro,
                }
            self.owned_side_effect_finders = owned_side_finders
            if direct_import_patch is not None:
                direct_import_patch[0].direct_transformers_import = direct_import_patch[1]
                direct_import_patch = None
            if spec_from_file_patch is not None:
                importlib.util.spec_from_file_location = spec_from_file_patch
                spec_from_file_patch = None
            _remove_identity_from_meta_path(finder)
            finder.release()
            finder = None
            assert_import_state("after oracle imports")
            # Fingerprint globals only after temporary retained-import shims are
            # restored.  Exact attribute chains may resolve lazy framework
            # exports, so require the complete four-class vector to reach a
            # fixed point.  This loop is explicitly bounded; a graph that keeps
            # mutating while inspected is evidence-blocked, never accepted.
            previous: dict[str, str] | None = None
            for _attempt in range(MAX_ORACLE_FINGERPRINT_STABILIZATION_PASSES):
                current = {
                    export: _module_global_integrity_fingerprint(snapshot["object"], self.locked_sources)
                    for export, snapshot in self.snapshots.items()
                }
                if current == previous:
                    for export, fingerprint_value in current.items():
                        self.snapshots[export]["module_global_fingerprint"] = fingerprint_value
                    break
                previous = current
            else:
                raise self._blocked(
                    "oracle-dependency-graph",
                    "oracle dependency fingerprint did not stabilize within the declared pass limit",
                )
            self.assert_integrity()
        except BaseException as error:
            if direct_import_patch is not None:
                direct_import_patch[0].direct_transformers_import = direct_import_patch[1]
                direct_import_patch = None
            if spec_from_file_patch is not None:
                importlib.util.spec_from_file_location = spec_from_file_patch
                spec_from_file_patch = None
            rollback_failures = self._rollback_failed_establishment(
                baseline_modules, baseline_meta_path, finder, owned_side_finders
            )
            if rollback_failures:
                raise BaseExceptionGroup(
                    "oracle trust establishment and rollback both failed", [error, *rollback_failures]
                ) from None
            raise

    @property
    def exports(self) -> tuple[str, ...]:
        return tuple(self.snapshots)

    def object_for(self, export: str) -> Any:
        return self.snapshots[export]["object"]

    def assert_integrity(self, loader: AuthenticatedCdgLoader | None = None) -> None:
        for export, snapshot in self.snapshots.items():
            binding = snapshot["binding"]
            package_name = binding["package"]
            root_module = snapshot["root_module"]
            exact_module = snapshot["exact_module"]
            trusted_object = snapshot["object"]
            if sys.modules.get(package_name) is not root_module:
                raise self._blocked(export, "package-root module identity changed after trust initialization")
            if sys.modules.get(binding["module"]) is not exact_module:
                raise self._blocked(export, "trusted source module identity changed after trust initialization")
            if getattr(root_module, export, None) is not trusted_object:
                raise self._blocked(export, "package-root export identity changed after trust initialization")
            if exact_module.__dict__.get(export) is not trusted_object:
                raise self._blocked(export, "trusted submodule class identity changed after trust initialization")
            origin_text = getattr(getattr(exact_module, "__spec__", None), "origin", None)
            if not origin_text or Path(origin_text).resolve() != snapshot["origin"]:
                raise self._blocked(export, "trusted source module origin changed after trust initialization")
            _assert_authenticated_oracle_source(
                snapshot["origin"],
                snapshot["sha256"],
                snapshot["source_artifact_id"],
                snapshot["source_size"],
            )
            _assert_authenticated_oracle_source(
                snapshot["root_origin"],
                snapshot["root_sha256"],
                snapshot["root_artifact_id"],
                snapshot["root_source_size"],
            )
            if importlib.metadata.version(package_name) != snapshot["version"]:
                raise self._blocked(export, "oracle distribution version changed after trust initialization")
            fingerprint, mro = _class_integrity_fingerprint(trusted_object)
            expected_mro = snapshot["mro"]
            if len(mro) != len(expected_mro) or any(
                observed is not expected for observed, expected in zip(mro, expected_mro, strict=True)
            ):
                raise self._blocked(export, "trusted class MRO identity changed after trust initialization")
            if fingerprint != snapshot["class_fingerprint"]:
                raise self._blocked(export, "trusted class contents changed after trust initialization")
            global_fingerprint = _module_global_integrity_fingerprint(trusted_object, self.locked_sources)
            if global_fingerprint != snapshot["module_global_fingerprint"]:
                raise self._blocked(export, "trusted module-global dependency graph changed after trust initialization")
            if loader is not None:
                product_module = loader.import_module(binding["product_module"])
                if getattr(product_module, export, None) is not trusted_object:
                    raise self._blocked(
                        export,
                        f"authenticated {binding['product_module']} did not capture the trusted class object",
                    )


class OracleBindingGuard:
    """Share and continuously verify the independently established oracle snapshot."""

    def __init__(
        self,
        lock: dict[str, Any],
        assert_import_state: Callable[[str], None],
        owner_token: object,
    ) -> None:
        global _ORACLE_TRUST_SNAPSHOT
        with _AUTHENTICATED_CONTEXT_STATE_LOCK:
            if (
                _AUTHENTICATED_CONTEXT_OWNER_THREAD != threading.get_ident()
                or _AUTHENTICATED_CONTEXT_OWNER_TOKEN is not owner_token
                or not _AUTHENTICATED_CONTEXT_LOCK.locked()
            ):
                raise EvidenceBlocked(
                    "I0-BLOCK-AUTHENTICATED-CONTEXT-OWNERSHIP",
                    "oracle-snapshot",
                    "OracleBindingGuard requires the authenticated context owner token and process lock",
                )
        if _ORACLE_TRUST_SNAPSHOT is None:
            _ORACLE_TRUST_SNAPSHOT = _OracleTrustSnapshot(lock, assert_import_state)
        else:
            _ORACLE_TRUST_SNAPSHOT.assert_lock(lock)
        self._snapshot: _OracleTrustSnapshot | None = _ORACLE_TRUST_SNAPSHOT
        self._snapshot.assert_integrity()

    @property
    def exports(self) -> tuple[str, ...]:
        assert self._snapshot is not None
        return self._snapshot.exports

    def object_for(self, export: str) -> Any:
        assert self._snapshot is not None
        return self._snapshot.object_for(export)

    def assert_integrity(self, loader: AuthenticatedCdgLoader | None = None) -> None:
        if self._snapshot is None:
            raise EvidenceBlocked("I0-BLOCK-ORACLE-BINDING", "oracle-snapshot", "oracle guard was released")
        self._snapshot.assert_integrity(loader)

    def release(self) -> None:
        # The independently established snapshot intentionally survives later
        # contexts; per-context guards drop only their own reference.
        self._snapshot = None


class _SafeBuiltinsProxy(ModuleType):
    """Private builtins view that never exposes the process-global importer."""

    def __init__(self, guarded_import: _OwnedImportHook) -> None:
        super().__init__("builtins")
        self.__dict__.update(vars(builtins))
        self.__dict__["__import__"] = guarded_import
        self.__dict__["__name__"] = "builtins"


class AuthenticatedCdgLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Import one synthetic package exclusively from pre-authenticated bytes."""

    def __init__(
        self,
        package_name: str,
        sources: dict[str, bytes],
        import_guard: _BareProductImportGuard | None = None,
        oracle_guard: OracleBindingGuard | None = None,
        assert_import_state: Callable[[str], None] | None = None,
        guarded_import: _OwnedImportHook | None = None,
    ) -> None:
        if not package_name or "." in package_name:
            raise ValueError("authenticated package name must be one top-level component")
        self.package_name = package_name
        self._source_by_module: dict[str, tuple[str, bytes, bool]] = {}
        self.loaded_paths: set[str] = set()
        self.loaded_modules: dict[str, ModuleType] = {}
        self._import_guard = import_guard
        self._oracle_guard = oracle_guard
        self._assert_import_state = assert_import_state
        self._guarded_import = guarded_import
        self._safe_builtins = guarded_import.safe_builtins if guarded_import is not None else None
        for path, data in sources.items():
            pure = PurePosixPath(path)
            if pure.name == "__init__.py":
                suffix = pure.parent.parts
                is_package = True
            elif pure.suffix == ".py":
                suffix = (*pure.parent.parts, pure.stem) if pure.parent != PurePosixPath(".") else (pure.stem,)
                is_package = False
            else:
                raise EvidenceBlocked(
                    "I0-BLOCK-CDG-SOURCE", path, "authenticated executable source must be a Python module"
                )
            module_name = ".".join((package_name, *suffix)) if suffix else package_name
            if module_name in self._source_by_module:
                raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", path, "multiple source paths map to one module")
            self._source_by_module[module_name] = (path, bytes(data), is_package)
        if package_name not in self._source_by_module:
            raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", "__init__.py", "authenticated package root is missing")

    @property
    def bare_product_import_requests(self) -> tuple[tuple[str, str], ...]:
        if self._import_guard is None:
            return ()
        return tuple(self._import_guard.requests)

    def find_spec(
        self,
        fullname: str,
        _path: object = None,
        _target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname != self.package_name and not fullname.startswith(self.package_name + "."):
            return None
        if self._assert_import_state is not None:
            self._assert_import_state(f"finder dispatch for {fullname}")
        mapped = self._source_by_module.get(fullname)
        if mapped is None:
            raise ModuleNotFoundError(
                f"executable product module {fullname!r} is not in the authenticated CDG source set",
                name=fullname,
            )
        relative_path, _data, is_package = mapped
        spec = importlib.util.spec_from_loader(fullname, self, is_package=is_package)
        if spec is None:
            raise ImportError(f"could not construct authenticated module spec for {fullname}")
        spec.origin = f"authenticated-cdg://{relative_path}"
        spec.has_location = False
        return spec

    def create_module(self, _spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        mapped = self._source_by_module.get(module.__name__)
        if mapped is None:
            raise ImportError(f"authenticated source bytes disappeared for {module.__name__}")
        if self._assert_import_state is not None:
            self._assert_import_state(f"before executing {module.__name__}")
        relative_path, data, _is_package = mapped
        self.loaded_modules[module.__name__] = module
        code = compile(data, f"authenticated-cdg://{relative_path}", "exec", dont_inherit=True)
        self.loaded_paths.add(relative_path)
        if self._guarded_import is None:
            raise EvidenceBlocked("I0-BLOCK-CDG-SOURCE", module.__name__, "module-local guarded importer is absent")
        module_builtins = dict(vars(builtins))
        module_builtins["__import__"] = self._guarded_import
        module.__dict__["__builtins__"] = module_builtins
        exec(code, module.__dict__)
        if self._assert_import_state is not None:
            self._assert_import_state(f"after executing {module.__name__}")
        self.assert_import_isolation()

    def assert_import_isolation(self) -> None:
        """Fail if any bare product module or request escaped the isolated namespace."""
        bare_modules = sorted(
            name for name in sys.modules if _BareProductImportGuard.is_bare_product(name)
        )
        if bare_modules:
            raise EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE",
                bare_modules[0],
                "bare product module appeared while authenticated product context was active",
            )
        if self.bare_product_import_requests:
            name, mechanism = self.bare_product_import_requests[0]
            raise EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE",
                name,
                f"bare product import request reached authenticated context via {mechanism}",
            )

    def assert_namespace_bound(self) -> None:
        """Fail if a synthetic product module did not come from this retained-byte loader."""
        self.assert_import_isolation()
        prefix = self.package_name + "."
        for name, module in tuple(sys.modules.items()):
            if name != self.package_name and not name.startswith(prefix):
                continue
            if self.loaded_modules.get(name) is not module or getattr(module, "__loader__", None) is not self:
                raise EvidenceBlocked(
                    "I0-BLOCK-CDG-SOURCE",
                    name,
                    "executed product module is not bound to the authenticated retained-byte loader",
                )
            spec = getattr(module, "__spec__", None)
            if spec is None or not str(spec.origin).startswith("authenticated-cdg://"):
                raise EvidenceBlocked(
                    "I0-BLOCK-CDG-SOURCE", name, "executed product module has an unauthenticated origin"
                )

    def import_module(self, relative_name: str) -> ModuleType:
        """Import one declared product module inside this loader's isolated namespace."""
        if not relative_name or relative_name.startswith(".") or relative_name.endswith("."):
            raise ValueError("relative product module name must be nonempty and canonical")
        fullname = f"{self.package_name}.{relative_name}"
        if self._assert_import_state is not None:
            self._assert_import_state(f"before loader.import_module({relative_name})")
        existing = sys.modules.get(fullname)
        if existing is not None and self.loaded_modules.get(fullname) is not existing:
            raise EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE", fullname, "synthetic product namespace was populated by another loader"
            )
        module = importlib.import_module(fullname)
        if self._assert_import_state is not None:
            self._assert_import_state(f"after loader.import_module({relative_name})")
        self.assert_namespace_bound()
        return module

    def assert_oracle_bindings(self) -> None:
        if self._oracle_guard is None:
            raise RuntimeError("oracle bindings were not requested for this authenticated package")
        self._oracle_guard.assert_integrity(self)

    def oracle_object(self, export: str) -> Any:
        if self._oracle_guard is None:
            raise RuntimeError("oracle bindings were not requested for this authenticated package")
        return self._oracle_guard.object_for(export)

    def release(self) -> None:
        """Drop retained source/module/finder references after global cleanup."""
        self._source_by_module.clear()
        self.loaded_modules.clear()
        self.loaded_paths.clear()
        self._import_guard = None
        self._oracle_guard = None
        self._assert_import_state = None
        self._guarded_import = None
        self._safe_builtins = None


class AuthenticatedCdgPackage(AbstractContextManager[tuple[ModuleType, AuthenticatedCdgLoader]]):
    """Own one authenticated package under exclusive process-global import state."""

    def __init__(
        self,
        lock: dict[str, Any],
        package_name: str,
        *,
        verify_oracle_bindings: bool = False,
        acquisition_timeout: float = AUTHENTICATED_CONTEXT_ACQUIRE_TIMEOUT,
    ) -> None:
        if not package_name or "." in package_name:
            raise ValueError("authenticated package name must be one top-level component")
        if acquisition_timeout <= 0:
            raise ValueError("authenticated context acquisition timeout must be positive")
        self.lock = lock
        self.package_name = package_name
        self.verify_oracle_bindings = verify_oracle_bindings
        self.acquisition_timeout = acquisition_timeout
        self.loader: AuthenticatedCdgLoader | None = None
        self.module: ModuleType | None = None
        self.closed_import_audit: tuple[tuple[str, str], ...] = ()
        self.closed_cleanup_audit: tuple[str, ...] = ()
        self.closed_oracle_audit: tuple[str, ...] = ()
        self._guard: _BareProductImportGuard | None = None
        self._oracle_guard: OracleBindingGuard | None = None
        self._oracle_bound = False
        self._original_import: Any = None
        self._guarded_import: _OwnedImportHook | None = None
        self._quarantined_bare_modules: dict[str, ModuleType] = {}
        self._bare_quarantine_active = False
        self._ownership_token = object()
        self._owner_thread_id: int | None = None
        self._owns_process_lock = False
        self._namespace_reserved = False
        self._namespace_ever_reserved = False
        self._baseline_meta_path: tuple[Any, ...] = ()
        self._active_meta_path: tuple[Any, ...] = ()
        self._owned_synthetic_modules: tuple[ModuleType, ...] = ()
        self._owned_import_side_effect_finders: tuple[Any, ...] = ()
        self._closed = False
        self._cleanup_complete = False

    @staticmethod
    def _failure_group(message: str, failures: list[BaseException]) -> BaseException:
        if len(failures) == 1:
            return failures[0]
        if all(isinstance(error, Exception) for error in failures):
            return ExceptionGroup(message, list(failures))  # type: ignore[arg-type]
        return BaseExceptionGroup(message, failures)

    def _blocked_ownership(self, detail: str) -> EvidenceBlocked:
        return EvidenceBlocked(
            "I0-BLOCK-AUTHENTICATED-CONTEXT-OWNERSHIP",
            self.package_name,
            detail,
        )

    def _acquire_process_ownership(self) -> None:
        global _AUTHENTICATED_CONTEXT_OWNER_THREAD, _AUTHENTICATED_CONTEXT_OWNER_TOKEN
        thread_id = threading.get_ident()
        with _AUTHENTICATED_CONTEXT_STATE_LOCK:
            if _AUTHENTICATED_CONTEXT_OWNER_THREAD == thread_id:
                raise EvidenceBlocked(
                    "I0-BLOCK-AUTHENTICATED-CONTEXT-OVERLAP",
                    self.package_name,
                    "same-thread nested authenticated context is forbidden",
                )
        if not _AUTHENTICATED_CONTEXT_LOCK.acquire(timeout=self.acquisition_timeout):
            raise EvidenceBlocked(
                "I0-BLOCK-AUTHENTICATED-CONTEXT-OVERLAP",
                self.package_name,
                f"another authenticated context remained active for {self.acquisition_timeout:g} seconds",
            )
        try:
            with _AUTHENTICATED_CONTEXT_STATE_LOCK:
                if (
                    _AUTHENTICATED_CONTEXT_OWNER_THREAD is not None
                    or _AUTHENTICATED_CONTEXT_OWNER_TOKEN is not None
                ):
                    raise RuntimeError("authenticated context ownership state is inconsistent")
                _AUTHENTICATED_CONTEXT_OWNER_THREAD = thread_id
                _AUTHENTICATED_CONTEXT_OWNER_TOKEN = self._ownership_token
                self._owner_thread_id = thread_id
                self._owns_process_lock = True
        except BaseException:
            _AUTHENTICATED_CONTEXT_LOCK.release()
            raise

    def _validate_close_ownership(self) -> None:
        thread_id = threading.get_ident()
        if self._owner_thread_id is not None and self._owner_thread_id != thread_id:
            raise self._blocked_ownership("only the thread that entered the context may close it")
        if not self._owns_process_lock:
            return
        with _AUTHENTICATED_CONTEXT_STATE_LOCK:
            if _AUTHENTICATED_CONTEXT_OWNER_THREAD != thread_id:
                raise self._blocked_ownership("process ownership thread identity no longer matches")
            if _AUTHENTICATED_CONTEXT_OWNER_TOKEN is not self._ownership_token:
                raise self._blocked_ownership("process ownership token no longer matches")

    def _release_process_ownership(self) -> None:
        global _AUTHENTICATED_CONTEXT_OWNER_THREAD, _AUTHENTICATED_CONTEXT_OWNER_TOKEN
        if not self._owns_process_lock:
            return
        self._validate_close_ownership()
        with _AUTHENTICATED_CONTEXT_STATE_LOCK:
            # Re-check under the state lock immediately before changing either
            # ownership field.  A mismatch never releases the primitive lock.
            if (
                _AUTHENTICATED_CONTEXT_OWNER_THREAD != threading.get_ident()
                or _AUTHENTICATED_CONTEXT_OWNER_TOKEN is not self._ownership_token
            ):
                raise self._blocked_ownership("ownership changed at release boundary")
            _AUTHENTICATED_CONTEXT_OWNER_THREAD = None
            _AUTHENTICATED_CONTEXT_OWNER_TOKEN = None
            self._owns_process_lock = False
            _AUTHENTICATED_CONTEXT_LOCK.release()

    def _assert_pristine_import_state(self, stage: str) -> None:
        if builtins.__import__ is not self._original_import:
            raise EvidenceBlocked(
                "I0-BLOCK-AUTHENTICATED-IMPORT-STATE",
                self.package_name,
                f"builtins.__import__ changed {stage}",
            )
        if not _identity_sequence_matches(sys.meta_path, self._baseline_meta_path):
            raise EvidenceBlocked(
                "I0-BLOCK-AUTHENTICATED-IMPORT-STATE",
                self.package_name,
                f"sys.meta_path changed {stage}",
            )

    def _assert_active_import_state(self, stage: str, *, verify_oracle: bool = False) -> None:
        # Only the exact six finder instances minted by authenticated oracle
        # establishment may be normalized.  Name-compatible foreign objects
        # are preserved and block below.
        if self._oracle_guard is not None and self._oracle_guard._snapshot is not None:
            self._oracle_guard._snapshot.discard_owned_import_side_effect_finders(self._active_meta_path)
        # BASE-01 may first import six through authenticated retained product
        # source without establishing the BASE-02 oracle.  Bind only six's
        # actual module-owned `_importer` object created during this import;
        # class-name lookalikes are never normalized.
        for finder in _OracleTrustSnapshot._new_six_importers(self._active_meta_path):
            if not any(finder is item for item in self._owned_import_side_effect_finders):
                self._owned_import_side_effect_finders = (*self._owned_import_side_effect_finders, finder)
            _remove_identity_from_meta_path(finder)
        if builtins.__import__ is not self._original_import:
            raise EvidenceBlocked(
                "I0-BLOCK-AUTHENTICATED-IMPORT-STATE",
                self.package_name,
                f"foreign builtins.__import__ replacement detected {stage}",
            )
        if not _identity_sequence_matches(sys.meta_path, self._active_meta_path):
            raise EvidenceBlocked(
                "I0-BLOCK-AUTHENTICATED-IMPORT-STATE",
                self.package_name,
                f"foreign sys.meta_path mutation detected {stage}",
            )
        if verify_oracle and self._oracle_guard is not None:
            self._oracle_guard.assert_integrity()

    @staticmethod
    def _pop_module(name: str) -> object:
        return sys.modules.pop(name, _MISSING)

    def __enter__(self) -> tuple[ModuleType, AuthenticatedCdgLoader]:
        if self._closed or self.loader is not None:
            raise RuntimeError("authenticated package context is single-use")
        self._acquire_process_ownership()
        try:
            self._original_import = builtins.__import__
            self._baseline_meta_path = tuple(sys.meta_path)
            self._assert_pristine_import_state("at context entry")
            prefix = self.package_name + "."
            conflicts = [name for name in sys.modules if name == self.package_name or name.startswith(prefix)]
            if conflicts:
                raise EvidenceBlocked(
                    "I0-BLOCK-CDG-SOURCE",
                    self.package_name,
                    f"synthetic namespace is already occupied: {', '.join(sorted(conflicts))}",
                )
            self._namespace_reserved = True
            self._namespace_ever_reserved = True
            oracle_guard = (
                OracleBindingGuard(self.lock, self._assert_pristine_import_state, self._ownership_token)
                if self.verify_oracle_bindings
                else None
            )
            self._oracle_guard = oracle_guard
            self._assert_pristine_import_state("after oracle trust initialization")
            sources = authenticated_current_cdg_sources(self.lock)
            self._assert_pristine_import_state("after product source authentication")
            guard = _BareProductImportGuard()
            self._guard = guard
            self._guarded_import = _OwnedImportHook(guard, self._original_import)
            loader = AuthenticatedCdgLoader(
                self.package_name,
                sources,
                guard,
                oracle_guard,
                self._assert_active_import_state,
                self._guarded_import,
            )
            self.loader = loader
            self._bare_quarantine_active = True
            self._quarantined_bare_modules = {
                name: module
                for name, module in tuple(sys.modules.items())
                if _BareProductImportGuard.is_bare_product(name)
            }
            for name, module in self._quarantined_bare_modules.items():
                removed = self._pop_module(name)
                if removed is not module:
                    if removed is not _MISSING:
                        sys.modules.setdefault(name, removed)  # never overwrite a newer insertion
                    raise EvidenceBlocked(
                        "I0-BLOCK-CDG-SOURCE", name, "bare product module changed during quarantine"
                    )
            sys.meta_path[0:0] = [guard, loader]
            self._active_meta_path = (guard, loader, *self._baseline_meta_path)
            self._assert_active_import_state("after owned finder installation", verify_oracle=True)
            self.module = importlib.import_module(self.package_name)
            self._assert_active_import_state("after authenticated package import", verify_oracle=True)
            loader.assert_namespace_bound()
            if oracle_guard is not None:
                oracle_guard.assert_integrity(loader)
                self._oracle_bound = True
            return self.module, loader
        except BaseException as body_error:
            try:
                self.close()
            except BaseException as cleanup_error:
                raise self._failure_group(
                    "authenticated context entry and cleanup both failed",
                    [body_error, cleanup_error],
                ) from None
            raise

    def _remove_owned_finders(self, guard: _BareProductImportGuard | None, loader: AuthenticatedCdgLoader | None) -> None:
        for finder in (guard, loader, *self._owned_import_side_effect_finders):
            if finder is not None:
                _remove_identity_from_meta_path(finder)

    def _is_owned_synthetic_module(self, value: object) -> bool:
        return any(value is module for module in self._owned_synthetic_modules)

    def _cleanup_synthetic_modules(self, cleanup_failures: list[BaseException]) -> None:
        if not self._namespace_reserved:
            return
        prefix = self.package_name + "."
        aliases: set[str] = set()
        # The prefix is uniquely reserved.  Exact owned module objects are also
        # removed under every alias in sys.modules, without touching unrelated
        # values or invoking value equality.
        for _attempt in range(8):
            found = False
            for name, module in tuple(sys.modules.items()):
                canonical = name == self.package_name or name.startswith(prefix)
                alias = not canonical and self._is_owned_synthetic_module(module)
                if not canonical and not alias:
                    continue
                found = True
                if alias:
                    aliases.add(name)
                if sys.modules.get(name, _MISSING) is module:
                    self._pop_module(name)
            if not found:
                break
        if aliases:
            cleanup_failures.append(EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE",
                ", ".join(sorted(aliases)),
                "foreign sys.modules alias to an owned synthetic module was removed",
            ))
        stale = sorted(
            name for name, module in tuple(sys.modules.items())
            if name == self.package_name
            or name.startswith(prefix)
            or self._is_owned_synthetic_module(module)
        )
        if stale:
            cleanup_failures.append(EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE",
                stale[0],
                "synthetic namespace or owned-object alias remained populated after cleanup",
            ))

    def _final_namespace_alias_audit(self, cleanup_failures: list[BaseException]) -> None:
        """Last process-global namespace audit immediately before lock release."""
        if not self._namespace_ever_reserved and not self._owned_synthetic_modules:
            return
        prefix = self.package_name + "."
        stale: list[str] = []
        for name, module in tuple(sys.modules.items()):
            if (
                name == self.package_name
                or name.startswith(prefix)
                or self._is_owned_synthetic_module(module)
            ):
                stale.append(name)
                if sys.modules.get(name, _MISSING) is module:
                    sys.modules.pop(name, None)
        if stale:
            cleanup_failures.append(EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE",
                ", ".join(sorted(stale)),
                "final synthetic namespace/alias audit removed late process-global residue",
            ))

    def _restore_bare_modules(self, cleanup_failures: list[BaseException]) -> None:
        if not self._bare_quarantine_active:
            return
        conflicts: set[str] = set()
        for name, module in self._quarantined_bare_modules.items():
            current = sys.modules.setdefault(name, module)
            if current is not module:
                conflicts.add(name)
        for name, module in tuple(sys.modules.items()):
            if not _BareProductImportGuard.is_bare_product(name):
                continue
            original = self._quarantined_bare_modules.get(name, _MISSING)
            if original is _MISSING or module is not original:
                conflicts.add(name)
        if conflicts:
            cleanup_failures.append(EvidenceBlocked(
                "I0-BLOCK-CDG-SOURCE",
                ", ".join(sorted(conflicts)),
                "foreign bare product module appeared while authenticated context was active; preserved",
            ))

    @staticmethod
    def _capture_cleanup_phase(
        cleanup_failures: list[BaseException],
        phase: str,
        operation: Callable[[], None],
    ) -> None:
        try:
            operation()
        except BaseException as error:
            cleanup_failures.append(error if isinstance(error, EvidenceBlocked) else EvidenceBlocked(
                "I0-BLOCK-AUTHENTICATED-CLEANUP",
                phase,
                f"cleanup phase raised {type(error).__name__}: {error}",
            ))

    def _perform_cleanup(self) -> list[BaseException]:
        loader = self.loader
        guard = self._guard
        oracle_guard = self._oracle_guard
        cleanup_failures: list[BaseException] = []
        owned: list[ModuleType] = []
        if loader is not None:
            for module in loader.loaded_modules.values():
                if not any(module is item for item in owned):
                    owned.append(module)
        self._owned_synthetic_modules = tuple(owned)

        try:
            if oracle_guard is not None and loader is not None and self._oracle_bound:
                def final_oracle() -> None:
                    self._assert_active_import_state("before final oracle integrity assertion")
                    oracle_guard.assert_integrity(loader)
                    self.closed_oracle_audit = oracle_guard.exports
                self._capture_cleanup_phase(cleanup_failures, "oracle-integrity", final_oracle)

            # Foreign global importer changes are preserved but invalidate the
            # context; the owned module-local hook is always detached below.
            if self._original_import is not None and builtins.__import__ is not self._original_import:
                cleanup_failures.append(EvidenceBlocked(
                    "I0-BLOCK-AUTHENTICATED-IMPORT-STATE",
                    self.package_name,
                    "preserved unrelated builtins.__import__ replacement",
                ))
            if self._guarded_import is not None:
                self._capture_cleanup_phase(cleanup_failures, "detach-import-hook", self._guarded_import.detach)

            self._capture_cleanup_phase(
                cleanup_failures, "remove-owned-finders", lambda: self._remove_owned_finders(guard, loader)
            )
            if not _identity_sequence_matches(sys.meta_path, self._baseline_meta_path):
                cleanup_failures.append(EvidenceBlocked(
                    "I0-BLOCK-AUTHENTICATED-IMPORT-STATE",
                    self.package_name,
                    "preserved foreign sys.meta_path mutation during cleanup",
                ))

            self._capture_cleanup_phase(
                cleanup_failures,
                "synthetic-module-cleanup",
                lambda: self._cleanup_synthetic_modules(cleanup_failures),
            )
            self._capture_cleanup_phase(
                cleanup_failures,
                "bare-module-restoration",
                lambda: self._restore_bare_modules(cleanup_failures),
            )
            # Do not call importlib.invalidate_caches() here.  Baseline finders
            # are foreign arbitrary objects; invoking their callbacks after an
            # oracle check would create an unauthenticated mutation boundary.

            if guard is not None:
                self.closed_import_audit = tuple(guard.requests)
            if loader is not None:
                self._capture_cleanup_phase(cleanup_failures, "loader-release", loader.release)
            if guard is not None:
                self._capture_cleanup_phase(cleanup_failures, "import-guard-clear", guard.clear)
        finally:
            # Plain ownership-field clearing is deliberately in the outermost
            # finally.  No cleanup phase, including a foreign finder callback,
            # can prevent close() from reaching lock release.
            self.module = None
            self.loader = None
            self._guard = None
            # Keep the plain per-context guard until close() performs the final
            # post-cleanup integrity assertion.  No foreign callback is invoked
            # after that assertion and before process-lock release.
            self._oracle_bound = False
            self._original_import = None
            self._guarded_import = None
            self._quarantined_bare_modules.clear()
            self._bare_quarantine_active = False
            self._namespace_reserved = False
            self._active_meta_path = ()
            self._owned_import_side_effect_finders = ()
            self._cleanup_complete = True
        return cleanup_failures

    def close(self) -> None:
        # This owner check precedes every mutation so an intruder can neither
        # clean resources nor release another thread's lock.
        self._validate_close_ownership()
        if self._closed:
            return
        cleanup_failures: list[BaseException] = []
        try:
            if not self._cleanup_complete:
                try:
                    cleanup_failures.extend(self._perform_cleanup())
                except BaseException as error:
                    cleanup_failures.append(error)
            self._capture_cleanup_phase(
                cleanup_failures,
                "final-namespace-alias-audit",
                lambda: self._final_namespace_alias_audit(cleanup_failures),
            )
            # This is the truly final mutation-capable operation before process
            # ownership is released.  All finder/module/hook cleanup has already
            # happened, and no foreign callback is invoked below this point.
            oracle_guard = self._oracle_guard
            if oracle_guard is not None:
                oracle_guard.assert_integrity()
                self.closed_oracle_audit = oracle_guard.exports
        except BaseException as error:
            # The pre-cleanup product-binding check and the mandatory final
            # snapshot check can report the same unchanged mutation.  Keep one
            # finding while still executing both boundaries.
            if not any(type(existing) is type(error) and str(existing) == str(error) for existing in cleanup_failures):
                cleanup_failures.append(error)
        finally:
            # release() only drops a plain reference; it invokes no foreign
            # callback and the process-lifetime snapshot remains globally held.
            if self._oracle_guard is not None:
                self._oracle_guard.release()
                self._oracle_guard = None
            try:
                self._release_process_ownership()
            except BaseException as ownership_error:
                cleanup_failures.append(ownership_error)
        if not self._owns_process_lock:
            self._closed = True
            self._owned_synthetic_modules = ()
            self._namespace_ever_reserved = False
        self.closed_cleanup_audit = tuple(str(error) for error in cleanup_failures)
        if cleanup_failures:
            raise self._failure_group("authenticated context cleanup failed", cleanup_failures) from None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        _traceback: object,
    ) -> bool:
        try:
            self.close()
        except BaseException as cleanup_error:
            if exc is not None:
                raise self._failure_group(
                    "authenticated context body and cleanup both failed",
                    [exc, cleanup_error],
                ) from None
            raise
        return False

def assert_oracle_suite_integrity() -> None:
    """Fail the group if the process-lifetime oracle changed after its final context."""
    if _ORACLE_TRUST_SNAPSHOT is not None:
        _ORACLE_TRUST_SNAPSHOT.assert_integrity()


def load_authenticated_cdg_package(
    lock: dict[str, Any],
    package_name: str = "_dg_engine_i0_pack",
    *,
    verify_oracle_bindings: bool = False,
    acquisition_timeout: float = AUTHENTICATED_CONTEXT_ACQUIRE_TIMEOUT,
) -> AuthenticatedCdgPackage:
    """Return a serialized retained-byte package context with owned cleanup."""
    return AuthenticatedCdgPackage(
        lock,
        package_name,
        verify_oracle_bindings=verify_oracle_bindings,
        acquisition_timeout=acquisition_timeout,
    )
