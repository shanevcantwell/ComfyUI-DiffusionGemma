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
import time
from pathlib import Path
from typing import Any

import requests

EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"

# examples/README.md's own provenance table: node 74 previews the STRING
# text output, node 75 previews the CanvasState repr (validity readout).
STRING_PREVIEW_NODE_ID = "74"
CANVAS_STATE_PREVIEW_NODE_ID = "75"


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
    assert STRING_PREVIEW_NODE_ID in outputs, (
        f"expected STRING preview at node {STRING_PREVIEW_NODE_ID}, got outputs: "
        f"{sorted(outputs.keys())}"
    )
    assert CANVAS_STATE_PREVIEW_NODE_ID in outputs, (
        f"expected CanvasState preview at node {CANVAS_STATE_PREVIEW_NODE_ID}, got "
        f"outputs: {sorted(outputs.keys())}"
    )

    # PreviewAny.main() (`comfy_extras/nodes_preview_any.py:39`) returns
    # `{"ui": {"text": (value,)}, ...}` — `/history`'s `outputs[node_id]` is
    # that `ui` dict, so the rendered value is always under the "text" key,
    # a one-tuple of the stringified source.
    string_output = outputs[STRING_PREVIEW_NODE_ID]
    text_values = string_output.get("text", [])
    assert text_values, f"STRING preview output was empty: {string_output}"
    rendered_text = "".join(str(v) for v in text_values)
    assert rendered_text.strip(), "STRING preview text was blank after stripping"
    return rendered_text
