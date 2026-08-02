"""Grep/identity-gate for the terms-and-units vocabulary ONE-MINT
(units-glossary-tooltips work item, `dgemma.loop.KNOB_DOCS`).

Mirrors `test_socket_mint.py`'s discipline: assert against the mint MODULE
OBJECT, never against a hardcoded copy of its text, so a future wording
tweak only touches `KNOB_DOCS` itself and every door that reads from it
follows for free. The enforcement surface this test provides: a future
edit that re-types a knob's tooltip/description text INSTEAD OF importing
`KNOB_DOCS` breaks the identity checks below — the dual-minting failure
mode this test exists to catch.
"""
from __future__ import annotations

from dgemma.loop import DEFAULT_GEN_LENGTH, KNOB_DOCS
from surfaces.comfyui.sampler import DGemmaSampler
from surfaces.mcp.commands import generate as generate_module

_SAMPLER_KNOBS = (
    "t_min",
    "t_max",
    "entropy_bound",
    "confidence",
    "num_inference_steps",
    "gen_length",
    "seed",
    "thinking",
)


def test_knob_docs_mint_covers_every_scoped_knob():
    """The mint names exactly the knobs this work item scoped — a knob
    missing from `KNOB_DOCS` would silently fall back to no tooltip/no
    description at the doors below instead of failing loud here."""
    for knob in _SAMPLER_KNOBS:
        assert knob in KNOB_DOCS
        assert isinstance(KNOB_DOCS[knob], str) and KNOB_DOCS[knob]


def test_sampler_widget_tooltips_are_the_mint_object_not_a_copy():
    """Every `DGemmaSampler.INPUT_TYPES()` widget tooltip for a scoped knob
    is the SAME STRING OBJECT as `KNOB_DOCS[knob]` (identity, `is`) — proof
    the widget spec reads from the mint at call time rather than embedding
    a text copy that could drift the moment either side is edited alone."""
    spec = DGemmaSampler.INPUT_TYPES()
    for knob in _SAMPLER_KNOBS:
        widget_options = spec["required"][knob][1]
        assert widget_options["tooltip"] is KNOB_DOCS[knob], (
            f"{knob}'s widget tooltip is not the KNOB_DOCS mint object — "
            "re-typed text instead of a shared reference"
        )


def test_mcp_generate_schema_descriptions_are_the_mint_object_not_a_copy():
    """Same identity check as the ComfyUI tooltips, for the MCP `generate`
    tool's JSON-schema `description`s — rule-8 parity BY CONSTRUCTION: two
    doors reading the same object can't independently drift."""
    tools = {t.name: t for t in generate_module.get_tools()}
    properties = tools["generate"].inputSchema["properties"]
    for knob in _SAMPLER_KNOBS:
        assert properties[knob]["description"] is KNOB_DOCS[knob], (
            f"{knob}'s MCP schema description is not the KNOB_DOCS mint "
            "object — re-typed text instead of a shared reference"
        )


def test_gen_length_doc_states_the_grounded_block_arithmetic():
    """Issue #175 minimal-phase grounding flag: the gen_length tooltip's
    blocks×steps arithmetic must be confirmed against the loop's actual
    block-size constant, not restated from the spec unverified. Grounded
    against `dgemma/loop.py`'s `DEFAULT_GEN_LENGTH = 256` (the per-block
    canvas length, confirmed byte-identical across all cached checkpoint
    variants per issue #165's enumeration close-out) — this is the divisor
    the doc's "gen_length 1024 -> 4 blocks" example computes against
    (ceil(1024 / 256) == 4), matching the spec's own worked example.

    The assertions below derive their expected substrings FROM
    `DEFAULT_GEN_LENGTH` (never restate a literal `256`/`4` that merely
    happens to agree with it): if the constant changes, the doc must state
    the new divisor and the new worked-example block count, or this test
    fails — the doc-to-constant tie is mechanical, not coincidental."""
    import math

    block_size = DEFAULT_GEN_LENGTH
    doc = KNOB_DOCS["gen_length"]
    # The doc must name the actual per-block canvas length (the divisor).
    assert str(block_size) in doc, (
        f"gen_length doc does not state the block-size divisor {block_size}"
    )
    assert "num_inference_steps" in doc
    # The worked example ("gen_length WORKED_EXAMPLE -> N blocks") must state
    # the block count computed from the real constant, not a pinned literal.
    worked_example = 1024
    expected_blocks = math.ceil(worked_example / block_size)
    assert str(worked_example) in doc, "gen_length doc dropped its worked example"
    assert f"{expected_blocks} blocks" in doc, (
        f"gen_length doc's worked example is not "
        f"ceil({worked_example} / {block_size}) == {expected_blocks} blocks"
    )


def test_sampler_and_mcp_agree_by_construction():
    """Direct door-to-door parity check (rule-8): the ComfyUI tooltip and
    the MCP description for every scoped knob are the IDENTICAL object —
    not merely equal strings that happen to match today."""
    spec = DGemmaSampler.INPUT_TYPES()
    tools = {t.name: t for t in generate_module.get_tools()}
    properties = tools["generate"].inputSchema["properties"]
    for knob in _SAMPLER_KNOBS:
        tooltip = spec["required"][knob][1]["tooltip"]
        description = properties[knob]["description"]
        assert tooltip is description, (
            f"{knob}'s ComfyUI tooltip and MCP description are not the same "
            "object — a dual-mint has crept in"
        )
