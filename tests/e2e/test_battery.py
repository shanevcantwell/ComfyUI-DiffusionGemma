"""The black-box E2E battery (ADR-CDG-013, issue #59).

Phase E0/S1 (this module, first phase): the minimal-generate scenario.
Drives a real, running ComfyUI instance purely through its own HTTP API —
`POST /prompt` + poll `GET /history/{id}` — using the shipped
`examples/ping-smoke.api.json` workflow. Zero imports from
`dgemma`/`surfaces`/`consumers`; the enforcement surface for that is
`test_e2e_import_guard.py`, not this docstring.

The request/response plumbing and the S1 honesty assertion live in
`driver.py` (unit-tested against canned payloads in `test_driver_unit.py`,
no server required) — this module wires that plumbing to the real,
live-server fixture (`comfyui_server`, `conftest.py`).

Every test in this module is marked `e2e` (excluded from the default fast
suite and from `-m live`; select with `pytest -m e2e`) and depends,
transitively via the `comfyui_server` fixture, on the three operator-
scheduled preconditions named in ADR-CDG-013/issue #59 §5 — none of which
are satisfied yet, so this module SKIPs end-to-end today. That is the
correct, mergeable state per the ratified design: the battery is built
skip-gated, not faked green.
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
