"""tests/test_sampler_denoise_parity.py — issue #166 mechanical enforcement
surface: `DGemmaSampler`'s and `DGemmaDenoise`'s socket signatures (inputs
and outputs) must match, modulo an explicit NAMED-delta allowlist.

**Why this test exists (ratified scope, issue #166 decision comment,
2026-07-30):** the operator's scope decision adopts `DGemmaSampler`'s full
output signature as `DGemmaDenoise`'s "for this version" — copying the
proven contract rather than re-deriving a bespoke one. A test that merely
snapshots today's two signatures would rot the moment either node changes;
this test instead asserts EQUALITY modulo the allowlist below, so a future
sampler-class node that adds an output to one node and not the other fails
BY NAME (the assertion messages name the missing/extra socket), not by a
vague "these tuples differ" diff.

Named deltas (issue #166 decision comment, "legitimate deltas"):
- `kv_cache` (optional input, `DGEMMA_KV_CACHE`) — denoise-only, the whole
  reason `DGemmaDenoise` exists (IN-2). Not a sampler input.
- OUT-1 (a fourth `DGEMMA_KV_CACHE` output, ADR-CDG-012 §4/§D.2) — deferred
  to Phase 4 (issue #62 Q-2), NOT present on either node today. Named here
  so a reader knows this allowlist has room for it without the test
  needing to change first; there is currently nothing to allow because
  neither node emits it.
"""
from __future__ import annotations

from surfaces.comfyui.denoise import DGemmaDenoise
from surfaces.comfyui.sampler import DGemmaSampler
from surfaces.comfyui.socket_types import DGEMMA_KV_CACHE

# The ONE named input-side delta (issue #166 decision comment): denoise's
# optional `kv_cache` input has no sampler counterpart.
_DENOISE_ONLY_OPTIONAL_INPUTS = {"kv_cache"}

# The ONE named input-side delta going the other way: none today. Kept as
# an explicit (empty) set rather than omitted, so a future sampler-only
# widget is a deliberate allowlist edit, not a silent test change.
_SAMPLER_ONLY_OPTIONAL_INPUTS: set[str] = set()

# `thinking` was the #166 input-side parity fix itself — asserted to be on
# BOTH required-input sets below (not an allowlisted delta; its absence on
# either node would be exactly the regression this test exists to catch).
_SHARED_REQUIRED_WIDGETS = {
    "model",
    "prompt",
    "seed",
    "num_inference_steps",
    "t_min",
    "t_max",
    "entropy_bound",
    "confidence",
    "gen_length",
    "thinking",
}


def test_required_input_widgets_match_exactly():
    sampler_spec = DGemmaSampler.INPUT_TYPES()
    denoise_spec = DGemmaDenoise.INPUT_TYPES()

    sampler_required = set(sampler_spec["required"])
    denoise_required = set(denoise_spec["required"])

    assert sampler_required == _SHARED_REQUIRED_WIDGETS, (
        f"DGemmaSampler required inputs drifted from the shared widget set: "
        f"missing {_SHARED_REQUIRED_WIDGETS - sampler_required}, "
        f"extra {sampler_required - _SHARED_REQUIRED_WIDGETS}"
    )
    assert denoise_required == _SHARED_REQUIRED_WIDGETS, (
        f"DGemmaDenoise required inputs drifted from the shared widget set: "
        f"missing {_SHARED_REQUIRED_WIDGETS - denoise_required}, "
        f"extra {denoise_required - _SHARED_REQUIRED_WIDGETS}"
    )


def test_required_widget_socket_specs_match_field_for_field():
    """Not just the same NAMES — the same declared socket type + default
    for every shared required widget (a widget re-typed or re-defaulted on
    one node and not the other is exactly the drift this test guards)."""
    sampler_spec = DGemmaSampler.INPUT_TYPES()["required"]
    denoise_spec = DGemmaDenoise.INPUT_TYPES()["required"]

    mismatches = []
    for name in _SHARED_REQUIRED_WIDGETS:
        sampler_type = sampler_spec[name][0]
        denoise_type = denoise_spec[name][0]
        if sampler_type != denoise_type:
            mismatches.append(f"{name}: socket type {sampler_type!r} != {denoise_type!r}")
            continue
        sampler_default = sampler_spec[name][1].get("default") if len(sampler_spec[name]) > 1 else None
        denoise_default = denoise_spec[name][1].get("default") if len(denoise_spec[name]) > 1 else None
        if sampler_default != denoise_default:
            mismatches.append(f"{name}: default {sampler_default!r} != {denoise_default!r}")

    assert not mismatches, "sampler/denoise required-widget spec drift: " + "; ".join(mismatches)


def test_optional_inputs_match_modulo_the_named_delta_allowlist():
    sampler_optional = set(DGemmaSampler.INPUT_TYPES().get("optional", {}))
    denoise_optional = set(DGemmaDenoise.INPUT_TYPES().get("optional", {}))

    denoise_only = denoise_optional - sampler_optional
    sampler_only = sampler_optional - denoise_optional

    assert denoise_only == _DENOISE_ONLY_OPTIONAL_INPUTS, (
        f"DGemmaDenoise has an un-allowlisted optional input delta: "
        f"{denoise_only - _DENOISE_ONLY_OPTIONAL_INPUTS or denoise_only}"
    )
    assert sampler_only == _SAMPLER_ONLY_OPTIONAL_INPUTS, (
        f"DGemmaSampler has an un-allowlisted optional input delta: {sampler_only}"
    )


def test_kv_cache_optional_input_is_the_documented_denoise_delta():
    denoise_optional = DGemmaDenoise.INPUT_TYPES()["optional"]
    assert denoise_optional["kv_cache"][0] == DGEMMA_KV_CACHE


def test_hidden_inputs_match():
    """Issue #188: both nodes now also merge `live_view.LIVE_VIEW_HIDDEN_INPUT`
    (`dgemma_live_view`) into their `hidden` dict — the ONE-MINT sentinel the
    JS extension reads generically instead of a hardcoded node-type list."""
    sampler_hidden = set(DGemmaSampler.INPUT_TYPES().get("hidden", {}))
    denoise_hidden = set(DGemmaDenoise.INPUT_TYPES().get("hidden", {}))
    assert sampler_hidden == denoise_hidden == {"unique_id", "dgemma_live_view"}


def test_output_signature_matches_exactly():
    """The #166 ratified core: full connection parity — every output
    `DGemmaSampler` has, `DGemmaDenoise` has too, same order, same names,
    same list-ness. No allowlisted delta on the OUTPUT side today (OUT-1 is
    deferred on BOTH — see this module's docstring); a future divergence
    here is exactly what this test must catch by name."""
    assert DGemmaDenoise.RETURN_TYPES == DGemmaSampler.RETURN_TYPES, (
        f"RETURN_TYPES drifted: denoise={DGemmaDenoise.RETURN_TYPES} "
        f"sampler={DGemmaSampler.RETURN_TYPES}"
    )
    assert DGemmaDenoise.RETURN_NAMES == DGemmaSampler.RETURN_NAMES, (
        f"RETURN_NAMES drifted: denoise={DGemmaDenoise.RETURN_NAMES} "
        f"sampler={DGemmaSampler.RETURN_NAMES}"
    )
    assert DGemmaDenoise.OUTPUT_IS_LIST == DGemmaSampler.OUTPUT_IS_LIST, (
        f"OUTPUT_IS_LIST drifted: denoise={DGemmaDenoise.OUTPUT_IS_LIST} "
        f"sampler={DGemmaSampler.OUTPUT_IS_LIST}"
    )


def test_output_signature_names_the_full_sampler_output_set():
    """Pins the CONTENT of the shared contract, not just cross-node
    equality — a future PR that shrinks BOTH nodes' outputs in lockstep
    (still "equal" to each other) would pass the test above but silently
    regress the #166 ratified scope. This asserts the actual six names."""
    expected = ("text", "canvas_state", "canvas_trace", "frames", "images", "run_config")
    assert DGemmaSampler.RETURN_NAMES == expected
    assert DGemmaDenoise.RETURN_NAMES == expected
