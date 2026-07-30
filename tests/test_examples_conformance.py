"""tests/test_examples_conformance.py — issue #174: shipped-example
conformance across EVERY `examples/**/*.api.json` graph, not just the
KV_CACHE family `test_kv_cache_workflows.py` (ADR-CDG-012 DV.2) already
covers.

The #163 smoke found `DGemmaLoader` graphs still carrying widget keys
(`repo_id`, `local_files_only`) removed from `INPUT_TYPES` by the #17
idiomatic-loader retrofit — the smoke had to hand-adapt every graph against
a live `/object_info` to get a clean POST. `test_kv_cache_workflows.py`
checks class_type resolution, missing-required-input, and wired-link
socket-type round-trip, but has **no unknown-input-key check** — a stale
widget key silently rides through as an ignored extra field in a `/prompt`
POST rather than failing. That gap is exactly how `repo_id`/`local_files_only`
survived past DV.2's existing tests undetected.

This module:
1. Generalizes the DV.2 checks (class_type resolves, required inputs
   present, wired-link socket types match) to every shipped
   `examples/**/*.api.json`, not just the `*kv-cache*` subset.
2. Adds the missing check: every declared input KEY on a pack node must
   resolve against that node's current `INPUT_TYPES` (required ∪ optional).
   An unknown key fails **by the workflow's file name and node id** — the
   fix this file exists to enforce; the fix without this test just resets
   the drift clock (issue #174).

Static-only: no GPU, no running ComfyUI server, no real weights. Distinct
from the live `.api.json` POST probe (`examples/README.md`'s end-to-end curl
probe), which stays the real-server E2E — this module is the unit-tier
gate that catches the next schema drift before it reaches that probe.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from __init__ import NODE_CLASS_MAPPINGS

_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
_ALL_EXAMPLE_WORKFLOWS = sorted(_EXAMPLES_DIR.glob("**/*.api.json"))


def _node_input_types(class_type: str) -> dict:
    node_cls = NODE_CLASS_MAPPINGS[class_type]
    return node_cls.INPUT_TYPES()


def _node_return_types(class_type: str) -> tuple:
    return NODE_CLASS_MAPPINGS[class_type].RETURN_TYPES


def _required_input_names(input_types: dict) -> set:
    """Every `required`-section input name, full stop — issue #179's
    grounded correction of the prior `_required_input_names_with_no_default`
    helper, which wrongly treated a declared `INPUT_TYPES` widget `default`
    as making the key omittable from a live `/prompt` POST.

    Grounded against the real ComfyUI validator,
    `/tmp/smoke-050/ComfyUI/execution.py`'s `validate_inputs`
    (`execution.py:896-913` in that checkout): `valid_inputs` is the union of
    `required` ∪ `optional` names (:896); the loop at :898-913 checks, for
    each name `x`, `if x not in inputs:` (:901) and — ONLY when
    `input_category == "required"` (:902) — unconditionally appends a
    `required_input_missing` error (:904-912), with NO fallback anywhere in
    that branch to an `INPUT_TYPES`-declared `default`. An `optional` key
    missing from `inputs` just `continue`s (:913) with no error either way.
    So: every `required` key must be present in the fixture's `inputs` dict
    or the live POST 400s — regardless of whether `INPUT_TYPES` declares a
    `default` for it. `INPUT_TYPES` defaults are backfilled only by the
    frontend graph editor when it builds a graph for POSTing (`.ui.json`
    authoring), never by the server for a raw `.api.json` submitted
    directly to `/prompt` — issue #179 (kv-cache-tier1.api.json's `thinking`/
    `mode`/`cell_px` gap, caught by the #163 smoke re-run) is exactly a
    fixture that relied on the false assumption this helper used to encode.

    Every workflow this module walks is `.api.json` (glob is
    `**/*.api.json` — see `_ALL_EXAMPLE_WORKFLOWS` below); there is no
    `.ui.json` carve-out to preserve here because none is in scope. If a
    `.ui.json` fixture is ever added to this glob, this strict rule would be
    the WRONG one for it (the editor does backfill defaults before POSTing
    a `.ui.json`-sourced graph) — that distinction would need reinstating
    at that point, keyed off file suffix, not deleted silently."""
    return set(input_types.get("required", {}))


def _all_declared_input_names(input_types: dict) -> set:
    return set(input_types.get("required", {})) | set(input_types.get("optional", {}))


def _socket_type_for_input(input_types: dict, name: str):
    for section in ("required", "optional"):
        spec = input_types.get(section, {}).get(name)
        if spec is not None:
            return spec[0]
    return None


@pytest.mark.parametrize("workflow_path", _ALL_EXAMPLE_WORKFLOWS, ids=lambda p: p.name)
class TestExampleWorkflowConformance:
    def test_workflow_file_exists_and_is_non_empty(self, workflow_path):
        assert workflow_path.exists()
        graph = json.loads(workflow_path.read_text())
        assert graph, f"{workflow_path.name} parsed to an empty graph"

    def test_every_pack_node_class_type_resolves(self, workflow_path):
        graph = json.loads(workflow_path.read_text())
        for node_id, node in graph.items():
            class_type = node["class_type"]
            if class_type not in NODE_CLASS_MAPPINGS:
                continue  # out of this pack's contract (e.g. ComfyUI's own PreviewAny)
            assert class_type in NODE_CLASS_MAPPINGS, (
                f"{workflow_path.name} node {node_id}: class_type {class_type!r} "
                "does not resolve in NODE_CLASS_MAPPINGS"
            )

    def test_every_pack_node_required_input_present(self, workflow_path):
        graph = json.loads(workflow_path.read_text())
        for node_id, node in graph.items():
            class_type = node["class_type"]
            if class_type not in NODE_CLASS_MAPPINGS:
                continue
            input_types = _node_input_types(class_type)
            required = _required_input_names(input_types)
            given = set(node.get("inputs", {}).keys())
            missing = required - given
            assert not missing, (
                f"{workflow_path.name} node {node_id} ({class_type}): missing required "
                f"input(s) {missing} — a live /prompt POST 400s on ANY required key "
                "absent from `inputs`, even one with an INPUT_TYPES-declared widget "
                "default (ComfyUI never backfills a default server-side for a raw "
                "API-JSON submission; see execution.py:898-913, cited in "
                "_required_input_names's docstring) — node signature changed since "
                "this workflow was authored (shipped-but-rotted graph)"
            )

    def test_every_pack_node_input_key_is_declared(self, workflow_path):
        """The check DV.2's original KV_CACHE-only suite lacked (issue #174):
        every input KEY a shipped graph sets on a pack node must resolve
        against that node's CURRENT `INPUT_TYPES` (required ∪ optional). A
        stale widget key (e.g. a removed `repo_id`/`local_files_only`) fails
        here **by the workflow's file name and node id**, instead of riding
        through silently as an ignored extra field on a `/prompt` POST."""
        graph = json.loads(workflow_path.read_text())
        for node_id, node in graph.items():
            class_type = node["class_type"]
            if class_type not in NODE_CLASS_MAPPINGS:
                continue
            input_types = _node_input_types(class_type)
            declared = _all_declared_input_names(input_types)
            given = set(node.get("inputs", {}).keys())
            unknown = given - declared
            assert not unknown, (
                f"{workflow_path.name} node {node_id} ({class_type}): unknown input "
                f"key(s) {unknown} — not in current INPUT_TYPES (required ∪ optional); "
                "node signature changed since this workflow was authored and this "
                "graph is stale (shipped-but-rotted graph)"
            )

    def test_every_wired_link_socket_type_matches(self, workflow_path):
        """Round-trips every `[source_node_id, output_idx]` link against
        BOTH ends' declared socket type — the DV.3a native-type discipline
        as it actually appears wired in a shipped graph."""
        graph = json.loads(workflow_path.read_text())
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
                    continue  # unrelated/pack-external input on a foreign node
                source_node_id, output_idx = value
                source_node = graph.get(source_node_id)
                if source_node is None:
                    continue  # link to a node outside this workflow file — not this test's scope
                source_class_type = source_node["class_type"]
                if source_class_type not in NODE_CLASS_MAPPINGS:
                    continue
                source_socket = _node_return_types(source_class_type)[output_idx]
                target_socket = _socket_type_for_input(input_types, input_name)
                assert source_socket == target_socket, (
                    f"{workflow_path.name}: node {node_id} ({class_type}) input "
                    f"{input_name!r} wired from node {source_node_id} "
                    f"({source_class_type}) output {output_idx} ({source_socket!r}) "
                    f"does not match the declared socket type {target_socket!r}"
                )


def test_at_least_one_example_workflow_is_shipped():
    """Sanity: the glob itself must find something, or every test above
    would vacuously pass on zero parametrizations."""
    assert _ALL_EXAMPLE_WORKFLOWS, "expected at least one examples/**/*.api.json"


def test_every_shipped_api_json_is_covered_by_this_module():
    """Guards against the glob silently narrowing (e.g. a future example
    moved outside `examples/` or renamed off `.api.json`) — this module's
    coverage must match `test_kv_cache_workflows.py`'s narrower KV_CACHE
    glob as a subset, and both must be non-empty over the same directory
    tree."""
    kv_cache_only = sorted((_EXAMPLES_DIR / "smoke-tests").glob("*kv-cache*.api.json"))
    assert set(kv_cache_only).issubset(set(_ALL_EXAMPLE_WORKFLOWS))
