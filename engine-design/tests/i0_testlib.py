"""Semantic fixture helpers plus re-exported I0 evidence APIs."""
from __future__ import annotations

import math
from typing import Any

from i0_evidence import (
    ENGINE_DESIGN,
    FIXTURES,
    REPO_ROOT,
    SCHEMAS,
    AuthenticatedCdgLoader,
    EvidenceBlocked,
    EvidenceState,
    GateVerification,
    authenticate_current_cdg_sources,
    authenticated_current_cdg_sources,
    evidence_root,
    gate_categories_complete,
    load_authenticated_cdg_package,
    read_authenticated_cdg_source,
    read_evidence_artifact,
    read_json,
    scope_is_complete,
    sha256,
    validate_json,
    validate_json_data,
    verify_base02_gate,
    verify_evidence_scope,
    verify_gguf_derivation,
)


def jsonable(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    return value


def derive_object_info(node_name: str, node_class: type, display_name: str) -> dict[str, Any]:
    """Exact relevant projection of pinned ComfyUI server.py::node_info."""
    inputs = node_class.INPUT_TYPES()
    info: dict[str, Any] = {
        "input": jsonable(inputs),
        "input_order": {key: list(value.keys()) for key, value in node_class.INPUT_TYPES().items()},
        "is_input_list": getattr(node_class, "INPUT_IS_LIST", False),
        "output": jsonable(node_class.RETURN_TYPES),
        "output_is_list": jsonable(
            node_class.OUTPUT_IS_LIST
            if hasattr(node_class, "OUTPUT_IS_LIST")
            else [False] * len(node_class.RETURN_TYPES)
        ),
        "output_name": jsonable(
            node_class.RETURN_NAMES if hasattr(node_class, "RETURN_NAMES") else node_class.RETURN_TYPES
        ),
        "name": node_name,
        "display_name": display_name,
        "description": node_class.DESCRIPTION if hasattr(node_class, "DESCRIPTION") else "",
        "python_module": getattr(node_class, "RELATIVE_PYTHON_MODULE", "nodes"),
        "category": getattr(node_class, "CATEGORY", "sd"),
        "output_node": bool(getattr(node_class, "OUTPUT_NODE", False)),
        "has_intermediate_output": bool(getattr(node_class, "HAS_INTERMEDIATE_OUTPUT", False)),
    }
    if hasattr(node_class, "OUTPUT_TOOLTIPS"):
        info["output_tooltips"] = jsonable(node_class.OUTPUT_TOOLTIPS)
    for attribute, key in (("DEPRECATED", "deprecated"), ("EXPERIMENTAL", "experimental"), ("DEV_ONLY", "dev_only")):
        if getattr(node_class, attribute, False):
            info[key] = True
    if hasattr(node_class, "API_NODE"):
        info["api_node"] = jsonable(node_class.API_NODE)
    info["search_aliases"] = jsonable(getattr(node_class, "SEARCH_ALIASES", []))
    if hasattr(node_class, "ESSENTIALS_CATEGORY"):
        info["essentials_category"] = jsonable(node_class.ESSENTIALS_CATEGORY)
    return info


def entropy_bound_accept(entropies: list[float], bound: float) -> list[bool]:
    """Independent hand-calculation helper with explicit position tie-break."""
    if not math.isfinite(bound) or bound < 0:
        raise ValueError("entropy bound must be finite and nonnegative")
    if any(not math.isfinite(value) or value < 0 for value in entropies):
        raise ValueError("entropies must be finite and nonnegative")
    ordered = sorted(enumerate(entropies), key=lambda pair: (pair[1], pair[0]))
    cumulative = 0.0
    accepted = [False] * len(entropies)
    for position, entropy in ordered:
        cumulative += entropy
        accepted[position] = cumulative - entropy <= bound
    return accepted


def retained_prefix_positions(prefix_length: int, window: int) -> list[int]:
    if prefix_length < 0 or window <= 0:
        raise ValueError("invalid prefix/window")
    return list(range(max(0, prefix_length - window + 1), prefix_length))


def decoder_visible_positions(prefix_length: int, canvas_length: int, window: int, *, sliding: bool) -> list[int]:
    if canvas_length <= 0:
        raise ValueError("canvas length must be positive")
    prefix = retained_prefix_positions(prefix_length, window) if sliding else list(range(prefix_length))
    return prefix + list(range(prefix_length, prefix_length + canvas_length))


def stable_softmax(values: list[float | None]) -> list[float]:
    finite = [value for value in values if value is not None]
    if not finite:
        raise ValueError("softmax row has no finite value")
    maximum = max(finite)
    numerators = [0.0 if value is None else math.exp(value - maximum) for value in values]
    total = sum(numerators)
    return [value / total for value in numerators]


def top_k_contract(logits: list[float | None], k: int) -> tuple[list[int], list[float], float]:
    if not 0 < k <= len(logits):
        raise ValueError("k outside vocabulary")
    ranked = sorted(
        range(len(logits)),
        key=lambda index: (float("-inf") if logits[index] is None else logits[index], -index),
        reverse=True,
    )
    ids = ranked[:k]
    full = stable_softmax(logits)
    retained_mass = sum(full[index] for index in ids)
    return ids, [full[index] / retained_mass for index in ids], retained_mass


def inverse_cdf(probabilities: list[float], draw: float) -> int:
    if not 0.0 <= draw < 1.0 or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-12):
        raise ValueError("invalid categorical fixture")
    cumulative = 0.0
    for index, probability in enumerate(probabilities):
        cumulative += probability
        if draw < cumulative:
            return index
    return len(probabilities) - 1


def checked_distribution_bytes(shape: list[int], element_bytes: int, *, limit: int = (1 << 63) - 1) -> int:
    product = 1
    for dimension in shape:
        if type(dimension) is not int or dimension < 0:
            raise ValueError("invalid dimension")
        if dimension and product > limit // dimension:
            raise OverflowError("shape product overflow")
        product *= dimension
    if element_bytes <= 0 or (product and product > limit // element_bytes):
        raise OverflowError("byte count overflow")
    return product * element_bytes
