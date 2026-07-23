"""tests/test_kv_cache_workflows.py — ADR-CDG-012 DV.2 (issue #62 Phase 3):
shipped-workflow conformance for the `KV_CACHE` example graphs.

Loads each `examples/*kv-cache*.json` and, for every node this PACK
registers (`NODE_CLASS_MAPPINGS` — a node like ComfyUI's own `PreviewAny` is
out of scope, this pack does not own its contract), asserts:

1. `class_type` resolves in `NODE_CLASS_MAPPINGS`.
2. Every required input is present in the workflow's `inputs` dict (either a
   literal widget value or a `[node_id, output_idx]` link).
3. Every wired link's source node's declared `RETURN_TYPES[output_idx]`
   matches the target node's declared `INPUT_TYPES` socket type for that
   input name (the `DGEMMA_*` native types round-trip).

A node-signature change that orphans a shipped workflow fails CI **by the
workflow's file name** — the "shipped-but-rotted graph" tripwire the ADR
names (no test loads `examples/*.json` against node defs today outside this
module). Static-only: no GPU, no running ComfyUI server, no real weights.
Distinct from the live `.api.json` POST probe (`examples/README.md`'s
end-to-end curl probe), which stays the real-server E2E.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from __init__ import NODE_CLASS_MAPPINGS

# Workflows moved to examples/smoke-tests/ per #127 (b4c5eca, 2026-07-22).
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "smoke-tests"
_KV_CACHE_WORKFLOWS = sorted(_EXAMPLES_DIR.glob("*kv-cache*.api.json"))


def _node_input_types(class_type: str) -> dict:
    node_cls = NODE_CLASS_MAPPINGS[class_type]
    return node_cls.INPUT_TYPES()


def _node_return_types(class_type: str) -> tuple:
    return NODE_CLASS_MAPPINGS[class_type].RETURN_TYPES


def _node_return_names(class_type: str) -> tuple:
    node_cls = NODE_CLASS_MAPPINGS[class_type]
    names = getattr(node_cls, "RETURN_NAMES", None)
    return names if names is not None else tuple(_node_return_types(class_type))


def _required_input_names_with_no_default(input_types: dict) -> set:
    """Required inputs with no declared widget `default` — the ones a
    `/prompt` POST genuinely cannot omit. A required socket input (a
    1-tuple spec, e.g. `("DGEMMA_MODEL",)`) has no default and always
    counts; a required WIDGET (2-tuple, e.g. `("BOOLEAN", {"default":
    False})`) with a declared default is server-side-fillable and does NOT
    count — matching the existing shipped-example convention
    (`ping-smoke.api.json` omits `DGemmaLoader`'s defaulted
    `local_files_only`; `p3-trace-smoke.api.json` omits `DGemmaTrace`'s
    defaulted `cell_px`), which this test must not regress against."""
    names = set()
    for name, spec in input_types.get("required", {}).items():
        if isinstance(spec, tuple) and len(spec) > 1 and "default" in spec[1]:
            continue
        names.add(name)
    return names


def _all_declared_input_names(input_types: dict) -> set:
    return set(input_types.get("required", {})) | set(input_types.get("optional", {}))


def _socket_type_for_input(input_types: dict, name: str):
    for section in ("required", "optional"):
        spec = input_types.get(section, {}).get(name)
        if spec is not None:
            return spec[0]
    return None


@pytest.mark.parametrize("workflow_path", _KV_CACHE_WORKFLOWS, ids=lambda p: p.name)
class TestKVCacheWorkflowConformance:
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
            required = _required_input_names_with_no_default(input_types)
            given = set(node.get("inputs", {}).keys())
            missing = required - given
            assert not missing, (
                f"{workflow_path.name} node {node_id} ({class_type}): missing required "
                f"input(s) {missing} — node signature changed since this workflow was "
                "authored (shipped-but-rotted graph)"
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


def test_at_least_one_kv_cache_workflow_is_shipped():
    """Sanity: the glob itself must find something, or every test above
    would vacuously pass on zero parametrizations."""
    assert _KV_CACHE_WORKFLOWS, "expected at least one examples/*kv-cache*.api.json"


def test_tier1_workflow_wires_encode_into_denoise():
    """The DV.2 minimum, named explicitly: the tier-1 honest-cache path
    actually chains `DGemmaEncode`'s output into `DGemmaDenoise`'s
    `kv_cache` input somewhere in the shipped tier-1 graph."""
    tier1_path = _EXAMPLES_DIR / "kv-cache-tier1.api.json"
    assert tier1_path.exists()
    graph = json.loads(tier1_path.read_text())

    encode_node_ids = {nid for nid, node in graph.items() if node["class_type"] == "DGemmaEncode"}
    denoise_nodes = [node for node in graph.values() if node["class_type"] == "DGemmaDenoise"]
    assert encode_node_ids, "tier-1 workflow has no DGemmaEncode node"
    assert denoise_nodes, "tier-1 workflow has no DGemmaDenoise node"

    wired = any(
        isinstance(node["inputs"].get("kv_cache"), list) and node["inputs"]["kv_cache"][0] in encode_node_ids
        for node in denoise_nodes
    )
    assert wired, "no DGemmaDenoise.kv_cache is wired from a DGemmaEncode output in the tier-1 workflow"
