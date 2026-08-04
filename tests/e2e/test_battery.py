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

Issue #228 part 2
(https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/228) adds
the standing Encode scenario, promoting the ad-hoc gate-run-4 "S-B" probe
(bare-module `encode_sequence` under bf16 CPU-spill, 2026-07-30, `a68e29d`;
issue #145) into this battery so encode liveness has a standing enforcement
surface rather than a result banked only in a handoff doc. Split across the
`dgemma/loop.py` `kv_cache` door's ADR-CDG-012 Phase 4 boundary
(`decisions/adr-cdg-012-mitm-seam-ar-diffusion-kv-cache.md`), same
`xfail(strict=True)` convention as S3: `test_encode_live` (stable before
and after Phase 4 — it never reaches the `kv_cache` door) and
`test_kv_door_contract` (asserts TODAY's fail-loud contract, marked to flip
when issue #62
(https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62)
Phase 4 lands).
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


# --- #228 part 2: standing Encode E2E scenario -------------------------------

# kv-cache-tier1-encode-only.api.json's OWN node id (examples/README.md's
# provenance entry for this fixture) — deliberately NOT one of this
# module's S1-S4 constants above (`driver.STRING_PREVIEW_NODE_ID` etc.,
# node ids 74/75/77/78 from the unrelated ping-smoke/p3-trace fixtures).
_ENCODE_KV_CACHE_PREVIEW_NODE_ID = "83"


def test_encode_live(comfyui_server: ComfyUIServer) -> None:
    """The standing Encode scenario (#228 part 2), promoting the ad-hoc
    gate-run-4 "S-B" probe (bare-module `encode_sequence` under bf16
    CPU-spill, 2026-07-30, `a68e29d`; issue #145) into this battery.
    Reuses `kv-cache-tier1-encode-only.api.json` — the minimal derivative
    that stops at `DGemmaLoader -> DGemmaEncode -> PreviewAny` on the raw
    `kv_cache` output, never reaching `DGemmaDenoise`'s `kv_cache` door.

    STABLE both before and after ADR-CDG-012 Phase 4: this scenario proves
    Encode liveness, a claim Phase 4 landing does not change. Asserts
    `success` and that the `KVCache` repr's `cumulative_length` parses to a
    non-empty tuple of strictly-positive per-layer lengths — the black-box
    signature that the encoder call actually advanced the cache."""
    workflow = driver.load_workflow("kv-cache-tier1-encode-only.api.json")

    prompt_id = driver.submit_prompt(
        comfyui_server.base_url, comfyui_server.client_id, workflow
    )
    history_entry = driver.poll_history(comfyui_server.base_url, prompt_id)

    driver.assert_encode_live_honest(history_entry, _ENCODE_KV_CACHE_PREVIEW_NODE_ID)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ADR-CDG-012 Phase 4 (issue #62 "
        "https://github.com/shanevcantwell/ComfyUI-DiffusionGemma/issues/62) "
        "is not yet landed: dgemma/loop.py's kv_cache-is-not-None block "
        "(currently lines 422-432) fails loud with NotImplementedError by "
        "design (issue #207) rather than silently ignoring an injected "
        "cache — expected RED until Phase 4's decoder-drive body lands "
        "(same issue #228 part 2 convention as test_s3_thinking_toggle_"
        "honest's issue #9 marker above); strict=True so an unexpected "
        "PASS (the door quietly starting to succeed) fails loudly and "
        "forces this marker's removal as part of Phase 4's own PR."
    ),
)
def test_kv_door_contract(comfyui_server: ComfyUIServer) -> None:
    """The KV-cache door's TODAY contract (#228 part 2): the full
    `kv-cache-tier1.api.json` graph (`DGemmaLoader -> DGemmaEncode ->
    DGemmaDenoise -> DGemmaTrace`) surfaces `dgemma/loop.py:422-432`'s
    fail-loud `NotImplementedError` as a failed execution — a well-formed
    injected `kv_cache` passes ingress validation (V1-V6) and is then
    rejected rather than silently run uninjected (issue #207's
    operator-ruled discipline). Asserts `status.status_str == "error"` with
    an `execution_error` status message naming `NotImplementedError` and
    Phase 4.

    THIS TEST FLIPS at ADR-CDG-012 Phase 4: once the decoder-drive body
    lands, this same graph reports `success` instead, and the `xfail`
    marker above must be removed as part of that fix's own PR — the same
    strict-xfail convention `test_s3_thinking_toggle_honest` uses for issue
    #9."""
    workflow = driver.load_workflow("kv-cache-tier1.api.json")

    prompt_id = driver.submit_prompt(
        comfyui_server.base_url, comfyui_server.client_id, workflow
    )
    history_entry = driver.poll_history(comfyui_server.base_url, prompt_id)

    driver.assert_kv_door_contract_honest(history_entry)
