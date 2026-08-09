"""test_tc001_graph_conformance.py — conformance gate for
`tc001_graph.api.json` (issue #286), the TC-001 batch harness's API graph.

`tc001_graph.api.json` lives under `docs/experiments/`, outside
`tests/test_examples_conformance.py`'s `examples/**/*.api.json` glob (that
module's own coverage-guard test, `test_every_shipped_api_json_is_covered_
by_this_module`, pins its scope to `examples/`, so a harness graph living
in `docs/` is deliberately not swept in there). This module is the same
checks — class_type resolution, required-input presence, unknown-input
rejection, wired-link socket-type round-trip — applied to this one graph,
so the harness carries its own conformance coverage rather than silently
riding on an assumption that the shared suite reaches it.

Static-only: no GPU, no running ComfyUI server, no real weights — same
discipline as the module it mirrors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HARNESS_DIR = Path(__file__).resolve().parent
# harness -> tc-001-structural-conditioning -> experiments -> docs -> repo root (4 levels).
_REPO_ROOT = _HARNESS_DIR.parent.parent.parent.parent
_GRAPH_PATH = _HARNESS_DIR / "tc001_graph.api.json"

# The pack's NODE_CLASS_MAPPINGS lives at the repo root's __init__.py
# (ComfyUI's own discovery entry point) — same import this harness's test
# counterpart (tests/test_examples_conformance.py) makes, but reached via
# an explicit sys.path insert since this file is not itself under
# tests/ (which is where the repo's pyproject.toml `--import-mode=importlib`
# + `testpaths` normally resolve that bare `from __init__ import ...`).
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from __init__ import NODE_CLASS_MAPPINGS  # noqa: E402


def _load_graph() -> dict:
    return json.loads(_GRAPH_PATH.read_text())


def _node_input_types(class_type: str) -> dict:
    return NODE_CLASS_MAPPINGS[class_type].INPUT_TYPES()


def _node_return_types(class_type: str) -> tuple:
    return NODE_CLASS_MAPPINGS[class_type].RETURN_TYPES


def _required_input_names(input_types: dict) -> set:
    # See tests/test_examples_conformance.py's `_required_input_names`
    # docstring for the full grounding (execution.py's validate_inputs):
    # every `required` key must be present regardless of any INPUT_TYPES
    # default — this module applies the identical rule to this one graph.
    return set(input_types.get("required", {}))


def _all_declared_input_names(input_types: dict) -> set:
    return set(input_types.get("required", {})) | set(input_types.get("optional", {}))


def _socket_type_for_input(input_types: dict, name: str):
    for section in ("required", "optional"):
        spec = input_types.get(section, {}).get(name)
        if spec is not None:
            return spec[0]
    return None


def test_graph_file_exists_and_is_non_empty():
    assert _GRAPH_PATH.exists()
    graph = _load_graph()
    assert graph, f"{_GRAPH_PATH.name} parsed to an empty graph"


def test_every_pack_node_class_type_resolves():
    graph = _load_graph()
    for node_id, node in graph.items():
        class_type = node["class_type"]
        if class_type not in NODE_CLASS_MAPPINGS:
            continue  # out of this pack's contract (e.g. ComfyUI's own PreviewAny/PreviewImage)
        assert class_type in NODE_CLASS_MAPPINGS, (
            f"{_GRAPH_PATH.name} node {node_id}: class_type {class_type!r} "
            "does not resolve in NODE_CLASS_MAPPINGS"
        )


def test_every_pack_node_required_input_present():
    graph = _load_graph()
    for node_id, node in graph.items():
        class_type = node["class_type"]
        if class_type not in NODE_CLASS_MAPPINGS:
            continue
        input_types = _node_input_types(class_type)
        required = _required_input_names(input_types)
        given = set(node.get("inputs", {}).keys())
        missing = required - given
        assert not missing, (
            f"{_GRAPH_PATH.name} node {node_id} ({class_type}): missing required "
            f"input(s) {missing} — a live /prompt POST 400s on ANY required key "
            "absent from `inputs`, even one with an INPUT_TYPES-declared widget "
            "default (ComfyUI never backfills a default server-side for a raw "
            "API-JSON submission)"
        )


def test_every_pack_node_input_key_is_declared():
    graph = _load_graph()
    for node_id, node in graph.items():
        class_type = node["class_type"]
        if class_type not in NODE_CLASS_MAPPINGS:
            continue
        input_types = _node_input_types(class_type)
        declared = _all_declared_input_names(input_types)
        given = set(node.get("inputs", {}).keys())
        unknown = given - declared
        assert not unknown, (
            f"{_GRAPH_PATH.name} node {node_id} ({class_type}): unknown input "
            f"key(s) {unknown} — not in current INPUT_TYPES (required ∪ optional)"
        )


def test_every_wired_link_socket_type_matches():
    graph = _load_graph()
    for node_id, node in graph.items():
        class_type = node["class_type"]
        if class_type not in NODE_CLASS_MAPPINGS:
            continue
        input_types = _node_input_types(class_type)
        declared_names = _all_declared_input_names(input_types)
        for input_name, value in node.get("inputs", {}).items():
            if not (isinstance(value, list) and len(value) == 2):
                continue  # a literal widget value, not a link
            if input_name not in declared_names:
                continue
            source_node_id, output_idx = value
            source_node = graph.get(source_node_id)
            if source_node is None:
                continue
            source_class_type = source_node["class_type"]
            if source_class_type not in NODE_CLASS_MAPPINGS:
                continue
            source_socket = _node_return_types(source_class_type)[output_idx]
            target_socket = _socket_type_for_input(input_types, input_name)
            assert source_socket == target_socket, (
                f"{_GRAPH_PATH.name}: node {node_id} ({class_type}) input "
                f"{input_name!r} wired from node {source_node_id} "
                f"({source_class_type}) output {output_idx} ({source_socket!r}) "
                f"does not match the declared socket type {target_socket!r}"
            )


def test_sampler_prompt_and_seed_and_gen_length_are_mutable_widgets():
    """A driver-specific guard (not in the mirrored suite): `run_batch.py`
    mutates node 73's `prompt`/`seed`/`gen_length` and node 75's
    `filename_prefix`/`debug_log_path` per run — this pins those five keys
    as literal (non-link) values in the base graph, so a future edit to
    `tc001_graph.api.json` that accidentally wires one of them from another
    node fails here rather than silently breaking `mutate_graph`."""
    graph = _load_graph()
    sampler_inputs = graph["73"]["inputs"]
    for key in ("prompt", "seed", "gen_length"):
        value = sampler_inputs[key]
        assert not (isinstance(value, list) and len(value) == 2), (
            f"tc001_graph.api.json node 73 ({key!r}) is wired as a link, but "
            "run_batch.py's mutate_graph expects a literal widget value"
        )
    writer_inputs = graph["75"]["inputs"]
    for key in ("filename_prefix", "debug_log_path"):
        value = writer_inputs[key]
        assert not (isinstance(value, list) and len(value) == 2), (
            f"tc001_graph.api.json node 75 ({key!r}) is wired as a link, but "
            "run_batch.py's mutate_graph expects a literal widget value"
        )


def test_defaults_match_repo_defaults_per_task_contract():
    """The task contract's own pinned knob values (steps 48, entropy_bound
    0.1, t_min 0.4, t_max 0.8, confidence 0.005, thinking false) plus
    gen_length 1024 — this is the one place those seven numbers are
    asserted against the shipped base graph, not just eyeballed."""
    graph = _load_graph()
    sampler_inputs = graph["73"]["inputs"]
    assert sampler_inputs["num_inference_steps"] == 48
    assert sampler_inputs["t_min"] == 0.4
    assert sampler_inputs["t_max"] == 0.8
    assert sampler_inputs["entropy_bound"] == 0.1
    assert sampler_inputs["confidence"] == 0.005
    assert sampler_inputs["thinking"] is False
    assert sampler_inputs["gen_length"] == 1024
    assert graph["72"]["inputs"]["quant"] == "none"
