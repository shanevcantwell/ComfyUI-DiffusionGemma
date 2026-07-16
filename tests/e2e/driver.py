"""The black-box ComfyUI-API driver's request/response plumbing (ADR-CDG-013,
issue #59), factored out of `test_battery.py` so it is unit-testable against
canned `/history` payloads without a live server — the part of this tier
that genuinely IS exercisable pre-infra (`test_driver_unit.py`), distinct
from the scenario assertions in `test_battery.py` that need the real
subprocess.

Independence discipline: stdlib + `requests` only, same invariant
`test_e2e_import_guard.py` enforces for the rest of `tests/e2e/`.
"""
from __future__ import annotations

import json
import re
import struct
import time
from pathlib import Path
from typing import Any

import requests

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

# examples/README.md's own provenance table: node 74 previews the STRING
# text output, node 75 previews the CanvasState repr (validity readout).
STRING_PREVIEW_NODE_ID = "74"
CANVAS_STATE_PREVIEW_NODE_ID = "75"

# `p3-trace-smoke(.api|-thinking.api).json` (examples/README.md's P3
# provenance entry): node 76 is `DGemmaTrace`, node 77 previews its
# `heatmap` IMAGE output, node 78 previews its `summary` STRING output.
TRACE_HEATMAP_PREVIEW_NODE_ID = "77"
TRACE_SUMMARY_PREVIEW_NODE_ID = "78"

# `dgemma/loop.py`'s literal thought-channel delimiter tokens
# (THOUGHT_CHANNEL_START_TOKEN/_END_TOKEN) — a well-formed answer STRING
# must never contain these post `excise_thought_channel` (ADR-CDG-001
# payload-contamination discipline). S2's "clean of the thought frame"
# assertion checks for their literal absence, the black-box observable of
# that excision contract.
THOUGHT_CHANNEL_START_TOKEN = "<|channel>"
THOUGHT_CHANNEL_END_TOKEN = "<channel|>"

# `dgemma/types.py:CanvasState` is a plain `@dataclass` with no custom
# `__repr__` — its auto-generated repr is what a `PreviewAny` on the
# sampler's `canvas_state` output renders to `/history`'s
# `outputs[node_id]["text"]`. These patterns pull specific fields back out
# of that repr string without importing the dataclass itself (the e2e
# tier's independence invariant forbids importing `dgemma` at all) — a
# black-box reader is only allowed to parse the wire text, same as any
# other consumer of this payload.
_CANVAS_STATE_FIELD_PATTERNS = {
    "bool": r"{field}=(True|False)",
    "float": r"{field}=([0-9.eE+-]+)",
    "int": r"{field}=(\d+)",
}


def parse_canvas_state_field(repr_text: str, field: str, kind: str) -> Any:
    """Extract one field's value out of a `CanvasState` repr string.
    `kind` is `"bool"`, `"float"`, or `"int"` — selects the value pattern.
    Raises `AssertionError` (not a plain regex-miss) if the field is absent,
    since every scenario that calls this treats a missing field as a
    malformed/unexpected payload shape, not an optional read."""
    pattern = _CANVAS_STATE_FIELD_PATTERNS[kind].format(field=re.escape(field))
    match = re.search(pattern, repr_text)
    assert match, f"CanvasState repr missing field {field!r} ({kind}): {repr_text!r}"
    raw = match.group(1)
    if kind == "bool":
        return raw == "True"
    if kind == "float":
        return float(raw)
    return int(raw)


def png_dimensions(data: bytes) -> tuple[int, int]:
    """Pure-stdlib PNG width/height read from the IHDR chunk (bytes 16:24,
    big-endian `>II`) — the PNG signature is always 8 bytes and IHDR is
    always the first chunk immediately after it, so this needs no image
    library. Kept stdlib-only deliberately: ADR-CDG-013 Decision 1 names
    `requests`/`websocket-client` as the driver's only non-stdlib imports,
    and pulling in Pillow just to read two integers would be scope creep
    against that list."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG file (bad signature)"
    assert data[12:16] == b"IHDR", f"expected IHDR as first chunk, got {data[12:16]!r}"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def load_workflow(name: str) -> dict[str, Any]:
    path = EXAMPLES_DIR / name
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def submit_prompt(base_url: str, client_id: str, workflow: dict[str, Any]) -> str:
    resp = requests.post(
        f"{base_url}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    resp.raise_for_status()
    body = resp.json()
    assert "prompt_id" in body, f"/prompt response missing prompt_id: {body}"
    return body["prompt_id"]


def poll_history(
    base_url: str,
    prompt_id: str,
    timeout_s: float = 300.0,
    interval_s: float = 1.0,
    _sleep=time.sleep,
    _monotonic=time.monotonic,
) -> dict[str, Any]:
    """Poll `/history/{prompt_id}` until an entry appears, bounded by
    `timeout_s`. Returns the history entry (not the whole response dict).
    `_sleep`/`_monotonic` are injectable seams for the unit tests below —
    real callers never pass them."""
    deadline = _monotonic() + timeout_s
    while _monotonic() < deadline:
        resp = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
        resp.raise_for_status()
        body = resp.json()
        if prompt_id in body:
            return body[prompt_id]
        _sleep(interval_s)
    raise TimeoutError(
        f"/history/{prompt_id} did not report a result within {timeout_s}s"
    )


def _preview_any_text(outputs: dict[str, Any], node_id: str, label: str) -> str:
    """Shared `PreviewAny` unwrap: `outputs[node_id]` is the node's `ui`
    dict, `{"text": (value,)}` (`comfy_extras/nodes_preview_any.py:39`) —
    every scenario that reads a `PreviewAny`-previewed STRING/repr output
    goes through this one unwrap so the shape assumption lives in one
    place. Raises (does not return None) when the node id or its "text" key
    is missing — a scenario calling this always expects the value to be
    present; a missing preview is itself the failure being reported."""
    assert node_id in outputs, (
        f"expected {label} preview at node {node_id}, got outputs: "
        f"{sorted(outputs.keys())}"
    )
    text_values = outputs[node_id].get("text", [])
    assert text_values, f"{label} preview output was empty: {outputs[node_id]}"
    return "".join(str(v) for v in text_values)


def fetch_view_bytes(
    base_url: str, filename: str, subfolder: str = "", folder_type: str = "temp"
) -> bytes:
    """`GET /view?filename=&subfolder=&type=` — the standard ComfyUI
    convention for retrieving a saved output/temp asset by the
    filename/subfolder/type triple `/history`'s image-output dicts carry
    (`comfy_api/latest/_ui.py:PreviewImage.as_dict`'s `SavedResult` shape).
    Used by S4 to read the heatmap PNG's own pixel dimensions back — the
    one thing `/history` alone does not carry for an IMAGE output."""
    resp = requests.get(
        f"{base_url}/view",
        params={"filename": filename, "subfolder": subfolder, "type": folder_type},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.content


def assert_s1_minimal_generate_honest(history_entry: dict[str, Any]) -> str:
    """S1's observable assertion (#59 §1), factored out of the live test so
    it can run against a canned `/history` payload: `status.status_str ==
    "success"`, both PreviewAny outputs present, STRING preview non-blank.
    Returns the rendered STRING text on success; raises AssertionError
    otherwise — same contract a caller gets from a bare `assert`."""
    status = history_entry.get("status", {})
    assert status.get("status_str") == "success", (
        f"ComfyUI run did not report success: {status}"
    )

    outputs = history_entry.get("outputs", {})
    rendered_text = _preview_any_text(outputs, STRING_PREVIEW_NODE_ID, "STRING")
    assert CANVAS_STATE_PREVIEW_NODE_ID in outputs, (
        f"expected CanvasState preview at node {CANVAS_STATE_PREVIEW_NODE_ID}, got "
        f"outputs: {sorted(outputs.keys())}"
    )
    assert rendered_text.strip(), "STRING preview text was blank after stripping"
    return rendered_text


def assert_s2_full_knob_honest(history_entry: dict[str, Any]) -> tuple[str, str]:
    """S2's observable assertion (#59 §1): every sampler knob reaches the
    loop and the run is honestly reported. `p2-knobs-smoke.api.json` wires
    all eight `DGemmaSampler` widgets at their served `/object_info`
    defaults (issue #59's own provenance note) — a success here is the
    knob-plumbing regression floor: `status_str == "success"`, the STRING
    preview is clean of the raw thought-frame delimiter tokens (ADR-CDG-001
    payload-contamination discipline — `dgemma.loop.excise_thought_channel`
    must have run), and the `CanvasState` repr's `converged`/
    `committed_fraction`/`steps_used` triple is internally consistent
    (`committed_fraction == 1.0` iff `converged is True`, matching
    `dgemma/types.py:CanvasState.converged`'s own "honest, not
    aspirational" definition — a per-step reading of the *last captured
    frame*, so the two must agree on the same frame).

    Returns `(rendered_string_text, canvas_state_repr)` on success."""
    status = history_entry.get("status", {})
    assert status.get("status_str") == "success", (
        f"ComfyUI run did not report success: {status}"
    )

    outputs = history_entry.get("outputs", {})
    rendered_text = _preview_any_text(outputs, STRING_PREVIEW_NODE_ID, "STRING")
    canvas_state_text = _preview_any_text(
        outputs, CANVAS_STATE_PREVIEW_NODE_ID, "CanvasState"
    )

    assert THOUGHT_CHANNEL_START_TOKEN not in rendered_text, (
        f"STRING preview leaked the raw thought-channel start delimiter: {rendered_text!r}"
    )
    assert THOUGHT_CHANNEL_END_TOKEN not in rendered_text, (
        f"STRING preview leaked the raw thought-channel end delimiter: {rendered_text!r}"
    )

    converged = parse_canvas_state_field(canvas_state_text, "converged", "bool")
    committed_fraction = parse_canvas_state_field(
        canvas_state_text, "committed_fraction", "float"
    )
    steps_used = parse_canvas_state_field(canvas_state_text, "steps_used", "int")

    assert steps_used > 0, f"steps_used should be positive: {canvas_state_text!r}"
    if converged:
        assert committed_fraction == 1.0, (
            "converged=True must mean the last captured frame's "
            f"committed_fraction==1.0 (CanvasState.converged's own honesty "
            f"contract), got committed_fraction={committed_fraction} in "
            f"{canvas_state_text!r}"
        )
    else:
        assert committed_fraction < 1.0, (
            "committed_fraction==1.0 with converged=False is the same "
            f"internal contradiction the other direction: {canvas_state_text!r}"
        )

    return rendered_text, canvas_state_text


def assert_s3_thinking_toggle_honest(history_entry: dict[str, Any]) -> str:
    """S3's observable assertion (#59 §1) — the **#9 catcher**, the sharp
    scenario this phase is expected to run RED against until #9 is fixed.
    `p3-trace-smoke-thinking.api.json` runs with `thinking=True`, the
    configuration issue #9 reports can consume the whole canvas as
    thought, leaving an empty answer `STRING` while the `CanvasState`
    repr still claims full convergence.

    The assertion: run reports `success`, AND the specific contradiction
    #9 names must NOT hold — an empty (post-strip) STRING must not
    co-exist with `converged=True committed_fraction==1.0`. Per
    `dgemma/loop.py:derive_canvas_state`, that same empty-answer specimen
    always carries `turn_closed=False answer_tokens=0` (nothing to find
    EOS in) — checked as the legible corroborating signature, not a
    separate assertion, since `turn_closed`/`answer_tokens` are exactly
    the fields `CanvasState`'s own docstring names as the honest read for
    "did this run actually finish" when `converged` alone can't be
    trusted.

    This assertion is EXPECTED to fail (AssertionError) against the real
    model until #9 is fixed — that failure IS the finding (ADR-CDG-013 §4
    /issue #59 phase E3's until-green protocol); the calling test wraps
    it in the plan's documented expected-RED mechanism rather than
    letting it read as an infra break.

    Returns the rendered STRING text when the assertion holds (post-fix)."""
    status = history_entry.get("status", {})
    assert status.get("status_str") == "success", (
        f"ComfyUI run did not report success: {status}"
    )

    outputs = history_entry.get("outputs", {})
    rendered_text = _preview_any_text(outputs, STRING_PREVIEW_NODE_ID, "STRING")
    canvas_state_text = _preview_any_text(
        outputs, CANVAS_STATE_PREVIEW_NODE_ID, "CanvasState"
    )

    is_empty_answer = not rendered_text.strip()
    converged = parse_canvas_state_field(canvas_state_text, "converged", "bool")
    committed_fraction = parse_canvas_state_field(
        canvas_state_text, "committed_fraction", "float"
    )

    fully_converged = converged and committed_fraction == 1.0
    assert not (is_empty_answer and fully_converged), (
        "issue #9 contradiction: empty STRING co-exists with "
        f"converged=True committed_fraction=1.0 — {canvas_state_text!r} "
        f"(STRING={rendered_text!r})"
    )

    # Corroborating signature (#9's own specimen shape), checked only when
    # present — older CanvasState reprs (pre-#9-fields) would still have
    # failed the assertion above already if the contradiction held, so this
    # is a legibility aid on top of the real check, not a second gate.
    if "turn_closed" in canvas_state_text and is_empty_answer:
        turn_closed = parse_canvas_state_field(
            canvas_state_text, "turn_closed", "bool"
        )
        answer_tokens = parse_canvas_state_field(
            canvas_state_text, "answer_tokens", "int"
        )
        assert turn_closed is False and answer_tokens == 0, (
            "empty-answer specimen did not match issue #9's documented "
            f"signature (turn_closed=False answer_tokens=0): {canvas_state_text!r}"
        )

    return rendered_text


def assert_s4_trace_readout_honest(
    history_entry: dict[str, Any], cell_px: int = 6
) -> dict[str, Any]:
    """S4's observable assertion (#59 §1): the `DGemmaTrace` analysis
    channel wires end-to-end on real frames. `p3-trace-smoke.api.json`
    chains `DGemmaSampler -> DGemmaTrace` with the heatmap previewed via
    `PreviewImage` (node 77, `{"images": [...]}` shape — distinct from the
    `PreviewAny` `{"text": [...]}` shape every other preview in this tier
    uses) and the trace summary via `PreviewAny` (node 78).

    Checks: `success`; heatmap IMAGE present; trace summary present and
    its `steps={N}` line (`surfaces/comfyui/trace.py:_format_summary`) is
    internally consistent with the run's own `CanvasState.steps_used`
    (both describe the same run, so they must agree); the heatmap PNG's
    own pixel height equals `steps_used * cell_px` — `build_commit_heatmap`
    emits one row per frame, nearest-neighbor-upscaled by `cell_px`
    (`consumers/analysis.py:build_commit_heatmap`), so the heatmap's
    row count is the one place "the trace channel actually saw every
    frame" is independently verifiable from outside the process.

    `cell_px` defaults to 6, `p3-trace-smoke.api.json`'s wired
    `DGemmaTrace.cell_px` widget value — pass the workflow's actual value
    if a future scenario JSON changes it.

    Returns a dict of the parsed readout (`steps_used`, `summary_steps`,
    `heatmap_height`, `expected_height`) for a caller that wants to log or
    assert further on the extracted values."""
    status = history_entry.get("status", {})
    assert status.get("status_str") == "success", (
        f"ComfyUI run did not report success: {status}"
    )

    outputs = history_entry.get("outputs", {})
    assert CANVAS_STATE_PREVIEW_NODE_ID in outputs, (
        f"expected CanvasState preview at node {CANVAS_STATE_PREVIEW_NODE_ID}, "
        f"got outputs: {sorted(outputs.keys())}"
    )
    canvas_state_text = _preview_any_text(
        outputs, CANVAS_STATE_PREVIEW_NODE_ID, "CanvasState"
    )
    steps_used = parse_canvas_state_field(canvas_state_text, "steps_used", "int")

    assert TRACE_HEATMAP_PREVIEW_NODE_ID in outputs, (
        f"expected trace heatmap preview at node {TRACE_HEATMAP_PREVIEW_NODE_ID}, "
        f"got outputs: {sorted(outputs.keys())}"
    )
    heatmap_output = outputs[TRACE_HEATMAP_PREVIEW_NODE_ID]
    images = heatmap_output.get("images", [])
    assert images, f"trace heatmap PreviewImage output was empty: {heatmap_output}"
    image_ref = images[0]
    for required_key in ("filename", "type"):
        assert required_key in image_ref, (
            f"heatmap image reference missing {required_key!r}: {image_ref}"
        )

    summary_text = _preview_any_text(
        outputs, TRACE_SUMMARY_PREVIEW_NODE_ID, "trace summary"
    )
    match = re.search(r"^steps=(\d+)$", summary_text, flags=re.MULTILINE)
    assert match, f"trace summary missing a 'steps=N' line: {summary_text!r}"
    summary_steps = int(match.group(1))
    assert summary_steps == steps_used, (
        f"trace summary steps={summary_steps} disagrees with CanvasState "
        f"steps_used={steps_used} — same run, must agree: {canvas_state_text!r} / "
        f"{summary_text!r}"
    )

    return {
        "steps_used": steps_used,
        "summary_steps": summary_steps,
        "image_ref": image_ref,
        "expected_heatmap_height": summary_steps * cell_px,
    }
