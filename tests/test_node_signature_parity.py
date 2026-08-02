"""tests/test_node_signature_parity.py — executor-faithful declared-input
vs. FUNCTION-signature parity, for every registered node (issue #212).

**Why this exists (the load-bearing half of #212's fix).** ComfyUI's
executor calls a node's `FUNCTION` as `f(**inputs)`, where `inputs` is built
from the node's ENTIRE declared `INPUT_TYPES()` — required, optional, AND
hidden (`execution.py`'s `process_inputs`, `f(**inputs)` at line 304; hidden
values are resolved by `get_input_data`, `execution.py:156-224`). A node
whose `INPUT_TYPES()` declares a key its `FUNCTION` signature does not
accept is fatal at execution — and invisible to every existing test in this
suite, all of which call node bodies directly with hand-picked kwargs
(never derived from `INPUT_TYPES()` itself). That gap is exactly how #212's
live TypeError (`DGemmaDenoise.denoise() got an unexpected keyword argument
'dgemma_live_view'`) shipped through two green gates: `INPUT_TYPES()` and
`FUNCTION`'s signature drifted apart when issue #188's live-view mint added
`dgemma_live_view` to the `hidden` dict of both `DGemmaSampler` and
`DGemmaDenoise` without adding a matching parameter to either `sample()` or
`denoise()`.

**Mechanism, mirroring `execution.py` exactly:** for every class in
`NODE_CLASS_MAPPINGS`, build the full declared input set — required +
optional + hidden, EVERY key `INPUT_TYPES()` returns, exactly as the
executor's `f(**inputs)` would receive them — then assert
`inspect.signature(FUNCTION).bind(**inputs)` succeeds. `bind()` alone (no
`.apply_defaults()`, no call) is sufficient: it raises `TypeError` on
exactly the failure this issue reports (an unexpected keyword, or a missing
required one) without needing to execute any node body — matching the
acceptance criterion's "signature-binding verification is sufficient."
Placeholder VALUES below are chosen only for `bind()`-time innocuousness
(never for correctness of node behavior, never mocking away the signature
itself) — `bind()` does not type-check or invoke, so `None`/simple stand-ins
are enough; the thing under test is purely name/arity parity between
`INPUT_TYPES()` and `FUNCTION`, the same shape `test_socket_mint.py`/
`test_live_view_mint.py` already use for their own round-trip checks.

This test is intentionally generic over `NODE_CLASS_MAPPINGS` rather than
naming each node: a NEW node this pack ships in the future is covered with
zero changes here, the same "no hand-maintained list" property `live_view.py`
architects for the JS side.
"""
from __future__ import annotations

import inspect

from __init__ import NODE_CLASS_MAPPINGS

# Placeholder value chosen per ComfyUI socket/widget TYPE STRING as declared
# in a node's own `INPUT_TYPES()` tuple (`(type_string, options_dict)` or
# `(type_string,)`) — this is a lookup table over TYPES, not over per-node
# input names, so it stays correct as node signatures evolve. `bind()` never
# inspects these values' contents, only that a value was supplied for every
# named parameter the signature does not default — so "innocuous stand-in
# of roughly the right shape" is sufficient, never "behaviorally correct."
_PLACEHOLDER_BY_TYPE = {
    "STRING": "placeholder",
    "INT": 0,
    "FLOAT": 0.0,
    "BOOLEAN": False,
    "IMAGE": object(),
    "UNIQUE_ID": "1",
    "PROMPT": {},
    "DYNPROMPT": None,
    "EXTRA_PNGINFO": None,
    "AUTH_TOKEN_COMFY_ORG": None,
    "API_KEY_COMFY_ORG": None,
    "COMFY_USAGE_SOURCE": None,
}


def _placeholder_for(type_entry):
    """Resolve one `INPUT_TYPES()` entry (a `(type, {...})` tuple, or a bare
    list of choices for a combo widget) to a `bind()`-safe placeholder
    value. Falls back to a generic sentinel object for any `DGEMMA_*`
    socket type or unrecognized string — `bind()` only needs *a* value, not
    a type-correct one."""
    type_key = type_entry[0] if isinstance(type_entry, (tuple, list)) and type_entry else type_entry
    if isinstance(type_key, (list, tuple)):
        # Combo widget: a literal list/tuple of choices IS the type slot
        # (e.g. DGemmaLoader's `quant`, DGemmaTrace's `mode`). Any member is
        # a valid stand-in; fall back to a plain sentinel if empty.
        return type_key[0] if len(type_key) > 0 else object()
    return _PLACEHOLDER_BY_TYPE.get(type_key, object())


def _build_full_declared_inputs(node_cls) -> dict:
    """Build the complete declared-input kwarg set for `node_cls`, mirroring
    ComfyUI's own executor construction: every key across `required` +
    `optional` + `hidden` in `INPUT_TYPES()`, each mapped to a `bind()`-safe
    placeholder VALUE (never a mock of the signature/binding itself)."""
    spec = node_cls.INPUT_TYPES()
    inputs = {}
    for category in ("required", "optional", "hidden"):
        for name, type_entry in spec.get(category, {}).items():
            inputs[name] = _placeholder_for(type_entry)
    return inputs


def test_node_class_mappings_is_non_empty():
    """Sanity precondition: if this ever collapses to empty (e.g. an import
    ordering break), the parametrized test below would silently pass with
    zero cases — assert the registry itself is populated first."""
    assert len(NODE_CLASS_MAPPINGS) > 0


def test_every_node_function_accepts_its_full_declared_input_set():
    """The executor-faithful parity check (issue #212 acceptance criterion
    2): for EVERY node in `NODE_CLASS_MAPPINGS`, build the full declared
    input set from its own `INPUT_TYPES()` and assert
    `inspect.signature(FUNCTION).bind(**inputs)` succeeds — exactly mirroring
    `execution.py`'s `f(**inputs)`. A single assertion covering every node
    (rather than one test per node) so the failure message names every
    offender in one run, not just the first one pytest happens to collect."""
    failures: dict[str, str] = {}

    for class_type, node_cls in sorted(NODE_CLASS_MAPPINGS.items()):
        function_name = node_cls.FUNCTION
        bound_method = getattr(node_cls, function_name)
        inputs = _build_full_declared_inputs(node_cls)
        sig = inspect.signature(bound_method)
        try:
            # `self` is supplied implicitly by `getattr(node_cls, ...)`
            # resolving to an unbound function accessed via the class in
            # Python 3 — bind against the params minus `self` by binding on
            # an actual (cheaply constructible) instance's method instead,
            # so `self` never needs a placeholder.
            instance = node_cls.__new__(node_cls)
            instance_sig = inspect.signature(getattr(instance, function_name))
            instance_sig.bind(**inputs)
        except TypeError as exc:
            failures[class_type] = (
                f"{class_type}.{function_name}{sig} rejected its own full "
                f"declared INPUT_TYPES() set {sorted(inputs.keys())}: {exc}"
            )

    assert not failures, (
        "declared-input / FUNCTION-signature parity violation(s) — a node "
        "whose INPUT_TYPES() declares a key its FUNCTION does not accept is "
        "fatal at ComfyUI execution time (f(**inputs), execution.py:304):\n"
        + "\n".join(f"  - {msg}" for msg in failures.values())
    )
