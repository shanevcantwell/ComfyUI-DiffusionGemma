"""Unit tests for the E2E driver's own request/response plumbing
(`driver.py`) against canned `/history` payloads — the part of the black-box
battery that IS exercisable pre-infra, per ADR-CDG-013/issue #59's coverage
register: no live server, no GPU, no weights. These run in the **default
fast suite** (no `e2e`/`live` marker) precisely because they need none of
the three operator-scheduled preconditions — only `driver.py`'s parsing and
assertion logic, which is plain Python over JSON dicts.

HTTP calls are mocked with `unittest.mock` (stdlib) rather than a new test
dependency — the ratified design (ADR-CDG-013 Decision 1) names
`requests`/`websocket-client` as the driver's only non-stdlib imports;
adding an HTTP-mocking library here would be scope creep against that list
for a need stdlib already covers.

This is deliberately separate from `test_battery.py` (the `e2e`-marked
scenario tests, which DO need the real subprocess): here we prove the
driver's own honesty-assertion logic is correct against known-shape
payloads (mirroring the real `PreviewAny.main()` return shape confirmed
against `/srv/dev/ComfyUI/comfy_extras/nodes_preview_any.py`), independent
of whether a live ComfyUI is reachable.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from tests.e2e import driver


def _history_entry(
    status_str: str = "success",
    string_text: str = "thought\nPong! How can I help you today?",
    canvas_state_text: str = "CanvasState(converged=True, committed_fraction=1.0, steps_used=3)",
    include_string_output: bool = True,
    include_canvas_output: bool = True,
) -> dict:
    """Builds a canned `/history/{id}` entry shaped exactly like ComfyUI's
    real response for the `ping-smoke.api.json` graph: `status.status_str`
    plus `outputs["74"]`/`outputs["75"]` as `PreviewAny`'s real
    `{"ui": {"text": (value,)}}` return shape."""
    outputs = {}
    if include_string_output:
        outputs[driver.STRING_PREVIEW_NODE_ID] = {"text": [string_text]}
    if include_canvas_output:
        outputs[driver.CANVAS_STATE_PREVIEW_NODE_ID] = {"text": [canvas_state_text]}
    return {
        "status": {"status_str": status_str, "completed": True, "messages": []},
        "outputs": outputs,
    }


def _fake_response(json_body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    if status_code >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.side_effect = None
    return resp


class TestAssertS1MinimalGenerateHonest:
    def test_passes_on_a_well_formed_success_entry(self):
        entry = _history_entry()
        rendered = driver.assert_s1_minimal_generate_honest(entry)
        assert "Pong" in rendered

    def test_fails_when_status_is_not_success(self):
        entry = _history_entry(status_str="error")
        with pytest.raises(AssertionError, match="did not report success"):
            driver.assert_s1_minimal_generate_honest(entry)

    def test_fails_when_string_preview_output_missing(self):
        entry = _history_entry(include_string_output=False)
        with pytest.raises(AssertionError, match="expected STRING preview"):
            driver.assert_s1_minimal_generate_honest(entry)

    def test_fails_when_canvas_state_preview_output_missing(self):
        entry = _history_entry(include_canvas_output=False)
        with pytest.raises(AssertionError, match="expected CanvasState preview"):
            driver.assert_s1_minimal_generate_honest(entry)

    def test_fails_when_string_preview_output_has_no_text_values(self):
        entry = _history_entry()
        entry["outputs"][driver.STRING_PREVIEW_NODE_ID] = {"text": []}
        with pytest.raises(AssertionError, match="STRING preview output was empty"):
            driver.assert_s1_minimal_generate_honest(entry)

    def test_fails_when_string_preview_text_is_an_empty_string(self):
        # A one-tuple containing "" is truthy as a list (non-empty
        # container) but blank after stripping — this exercises the
        # *second* honesty check, not the "output was empty" one.
        entry = _history_entry(string_text="")
        with pytest.raises(AssertionError, match="STRING preview text was blank"):
            driver.assert_s1_minimal_generate_honest(entry)

    def test_fails_when_string_preview_text_is_whitespace_only(self):
        entry = _history_entry(string_text="   \n\t  ")
        with pytest.raises(AssertionError, match="STRING preview text was blank"):
            driver.assert_s1_minimal_generate_honest(entry)


class TestLoadWorkflow:
    def test_loads_the_shipped_ping_smoke_workflow(self):
        workflow = driver.load_workflow("ping-smoke.api.json")
        assert workflow["73"]["class_type"] == "DGemmaSampler"
        assert workflow["74"]["class_type"] == "PreviewAny"
        assert workflow["75"]["class_type"] == "PreviewAny"


class TestSubmitPrompt:
    def test_returns_prompt_id_from_response_body(self):
        fake_resp = _fake_response({"prompt_id": "abc-123", "number": 1, "node_errors": {}})
        with patch("tests.e2e.driver.requests.post", return_value=fake_resp) as mock_post:
            prompt_id = driver.submit_prompt(
                "http://127.0.0.1:8199", "test-client", {"1": {"class_type": "X"}}
            )
        assert prompt_id == "abc-123"
        mock_post.assert_called_once()
        _, kwargs = mock_post.call_args
        assert kwargs["json"]["client_id"] == "test-client"

    def test_raises_when_response_body_lacks_prompt_id(self):
        fake_resp = _fake_response({"error": {"type": "invalid_prompt"}, "node_errors": {}})
        with patch("tests.e2e.driver.requests.post", return_value=fake_resp):
            with pytest.raises(AssertionError, match="missing prompt_id"):
                driver.submit_prompt(
                    "http://127.0.0.1:8199", "test-client", {"1": {"class_type": "X"}}
                )

    def test_raises_on_http_error_status(self):
        fake_resp = _fake_response({"error": {}}, status_code=400)
        with patch("tests.e2e.driver.requests.post", return_value=fake_resp):
            with pytest.raises(Exception):
                driver.submit_prompt(
                    "http://127.0.0.1:8199", "test-client", {"1": {"class_type": "X"}}
                )


class TestPollHistory:
    def test_returns_entry_on_first_successful_poll(self):
        entry = _history_entry()
        fake_resp = _fake_response({"abc-123": entry})
        with patch("tests.e2e.driver.requests.get", return_value=fake_resp):
            result = driver.poll_history("http://127.0.0.1:8199", "abc-123")
        assert result == entry

    def test_retries_until_the_entry_appears(self):
        entry = _history_entry()
        responses_sequence = [
            _fake_response({}),
            _fake_response({}),
            _fake_response({"abc-123": entry}),
        ]
        fake_clock = {"t": 0.0}

        def _fake_monotonic():
            return fake_clock["t"]

        def _fake_sleep(seconds):
            fake_clock["t"] += seconds

        with patch("tests.e2e.driver.requests.get", side_effect=responses_sequence):
            result = driver.poll_history(
                "http://127.0.0.1:8199",
                "abc-123",
                timeout_s=10,
                interval_s=1,
                _sleep=_fake_sleep,
                _monotonic=_fake_monotonic,
            )
        assert result == entry

    def test_raises_timeout_error_when_entry_never_appears(self):
        fake_clock = {"t": 0.0}

        def _fake_monotonic():
            return fake_clock["t"]

        def _fake_sleep(seconds):
            fake_clock["t"] += seconds

        with patch(
            "tests.e2e.driver.requests.get", return_value=_fake_response({})
        ):
            with pytest.raises(TimeoutError, match="did not report a result"):
                driver.poll_history(
                    "http://127.0.0.1:8199",
                    "abc-123",
                    timeout_s=2,
                    interval_s=1,
                    _sleep=_fake_sleep,
                    _monotonic=_fake_monotonic,
                )
