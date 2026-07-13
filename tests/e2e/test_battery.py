"""The black-box E2E battery (ADR-CDG-013, issue #59).

Phase E0/S1 (this module, first phase): the minimal-generate scenario.
Battery P2 (this module, second phase, issue #59 phasing E2 + the sharp E3
scenario S3): S2 (full-knob), S3 (thinking-toggle / #9 catcher), S4 (trace
readout). Drives a real, running ComfyUI instance purely through its own
HTTP API — `POST /prompt` + poll `GET /history/{id}` (+ `GET /view` for
S4's heatmap dimension check) — using the shipped `examples/*.api.json`
workflows. Zero imports from `dgemma`/`surfaces`/`consumers`; the
enforcement surface for that is `test_e2e_import_guard.py`, not this
docstring.

The request/response plumbing and every scenario's honesty assertion live
in `driver.py` (unit-tested against canned payloads in
`test_driver_unit.py`, no server required) — this module wires that
plumbing to the real, live-server fixture (`comfyui_server`,
`conftest.py`).

Every test in this module is marked `e2e` (excluded from the default fast
suite and from `-m live`; select with `pytest -m e2e`) and depends,
transitively via the `comfyui_server` fixture, on the three operator-
scheduled preconditions named in ADR-CDG-013/issue #59 §5 — none of which
are satisfied yet, so this module SKIPs end-to-end today. That is the
correct, mergeable state per the ratified design: the battery is built
skip-gated, not faked green.

S3 is marked `xfail(strict=True)` referencing issue #9 (#59 phase E3's
"expected RED until the underlying bug is fixed" convention): `strict=True`
means an unexpected PASS is itself reported as a failure, so the marker
cannot silently go stale once #9 is fixed — the flip to green forces the
marker's removal as part of that fix's own PR, banking the red-to-green
transition as the fix's live proof (ADR-CDG-013 §4/issue #59 §4).
"""
from __future__ import annotations

import pytest

from tests.e2e import driver
from tests.e2e.conftest import ComfyUIServer

pytestmark = pytest.mark.e2e


def test_s1_minimal_generate(comfyui_server: ComfyUIServer) -> None:
    """S1 (#59 §1): pack loads, model loads, one canvas converges — the
    regression floor. Reuses the shipped `ping-smoke.api.json` graph
    (DGemmaLoader -> DGemmaSampler -> two PreviewAny nodes) exactly as
    `examples/README.md`'s own curl-based E2E probe does, automated."""
    workflow = driver.load_workflow("ping-smoke.api.json")

    prompt_id = driver.submit_prompt(
        comfyui_server.base_url, comfyui_server.client_id, workflow
    )
    history_entry = driver.poll_history(comfyui_server.base_url, prompt_id)

    driver.assert_s1_minimal_generate_honest(history_entry)


def test_s2_full_knob_sampler(comfyui_server: ComfyUIServer) -> None:
    """S2 (#59 §1): every sampler widget reaches the loop. Reuses the
    shipped `p2-knobs-smoke.api.json` graph — all eight `DGemmaSampler`
    knobs wired at their served `/object_info` defaults (issue #59's own
    provenance note: derived mechanically from a live instance after the
    P2 commit). Asserts `success`, the STRING preview clean of the raw
    thought-frame delimiter tokens, and the `CanvasState` validity
    readout's `converged`/`committed_fraction`/`steps_used` triple
    internally consistent."""
    workflow = driver.load_workflow("p2-knobs-smoke.api.json")

    prompt_id = driver.submit_prompt(
        comfyui_server.base_url, comfyui_server.client_id, workflow
    )
    history_entry = driver.poll_history(comfyui_server.base_url, prompt_id)

    driver.assert_s2_full_knob_honest(history_entry)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "issue #9: thinking=True can consume the whole canvas, leaving an "
        "empty answer STRING while CanvasState still claims "
        "converged=True committed_fraction=1.0 — expected RED until #9 is "
        "fixed (issue #59 phase E3 convention); strict=True so an "
        "unexpected PASS fails loudly and forces this marker's removal as "
        "part of the #9 fix's own PR."
    ),
)
def test_s3_thinking_toggle_honest(comfyui_server: ComfyUIServer) -> None:
    """S3 (#59 §1) — the **#9 catcher**, one of the battery's "acceptance
    teeth" (#59's acceptance-sharpness note): a naive success-only battery
    would pass right over this scenario. Reuses the shipped
    `p3-trace-smoke-thinking.api.json` graph (`thinking=True`) and asserts
    `success` PLUS the specific #9 contradiction does NOT hold: an empty
    (post-strip) STRING must not co-exist with `converged=True
    committed_fraction=1.0`."""
    workflow = driver.load_workflow("p3-trace-smoke-thinking.api.json")

    prompt_id = driver.submit_prompt(
        comfyui_server.base_url, comfyui_server.client_id, workflow
    )
    history_entry = driver.poll_history(comfyui_server.base_url, prompt_id)

    driver.assert_s3_thinking_toggle_honest(history_entry)


def test_s4_trace_readout(comfyui_server: ComfyUIServer) -> None:
    """S4 (#59 §1): the `DGemmaTrace` analysis channel wires end-to-end on
    real frames. Reuses the shipped `p3-trace-smoke.api.json` graph
    (`DGemmaSampler -> DGemmaTrace`, heatmap + trace-summary previews).
    Asserts `success`, the heatmap IMAGE present, the trace summary's
    `steps={N}` line agreeing with `CanvasState.steps_used`, and — via a
    `GET /view` fetch of the saved heatmap PNG — that the heatmap's own
    pixel height equals `steps_used * cell_px`, confirming the channel
    actually saw every frame."""
    workflow = driver.load_workflow("p3-trace-smoke.api.json")

    prompt_id = driver.submit_prompt(
        comfyui_server.base_url, comfyui_server.client_id, workflow
    )
    history_entry = driver.poll_history(comfyui_server.base_url, prompt_id)

    readout = driver.assert_s4_trace_readout_honest(history_entry)

    image_ref = readout["image_ref"]
    png_bytes = driver.fetch_view_bytes(
        comfyui_server.base_url,
        filename=image_ref["filename"],
        subfolder=image_ref.get("subfolder", ""),
        folder_type=image_ref.get("type", "temp"),
    )
    _, heatmap_height = driver.png_dimensions(png_bytes)
    assert heatmap_height == readout["expected_heatmap_height"], (
        f"heatmap PNG height {heatmap_height} != steps_used*cell_px "
        f"{readout['expected_heatmap_height']} — the heatmap did not "
        "capture one row per frame"
    )
