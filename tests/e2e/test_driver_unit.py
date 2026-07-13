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

Battery P2 (issue #59, ADR-CDG-013) adds unit coverage for S2/S3/S4's
assertion helpers plus the two new stdlib-only primitives they lean on
(`parse_canvas_state_field`, `png_dimensions`) and the `fetch_view_bytes`
HTTP call — same canned-payload, no-server discipline as S1's register.
"""
from __future__ import annotations

import struct
import zlib
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


def _trace_history_entry(
    status_str: str = "success",
    canvas_state_text: str = "CanvasState(converged=True, committed_fraction=1.0, steps_used=15)",
    summary_text: str = (
        "scheduler=EntropyBoundScheduler config={}\n"
        "steps=15\n"
        "committed_fraction per step (block-local — resets near 0 at each "
        "canvas/block boundary; this is block advancement, not re-melt): "
        "0.0000, 0.5000, 1.0000\n"
        "mask-token corroboration: no fixed sentinel (uniform-state renoise supported)"
    ),
    image_ref: dict | None = None,
    include_canvas_output: bool = True,
    include_heatmap_output: bool = True,
    include_summary_output: bool = True,
) -> dict:
    """Builds a canned `/history/{id}` entry shaped like the
    `p3-trace-smoke.api.json` graph: node 75 CanvasState (PreviewAny),
    node 77 heatmap (PreviewImage, `{"images": [...]}`), node 78 trace
    summary (PreviewAny)."""
    if image_ref is None:
        image_ref = {"filename": "heatmap_00001_.png", "subfolder": "", "type": "temp"}
    outputs = {}
    if include_canvas_output:
        outputs[driver.CANVAS_STATE_PREVIEW_NODE_ID] = {"text": [canvas_state_text]}
    if include_heatmap_output:
        outputs[driver.TRACE_HEATMAP_PREVIEW_NODE_ID] = {"images": [image_ref]}
    if include_summary_output:
        outputs[driver.TRACE_SUMMARY_PREVIEW_NODE_ID] = {"text": [summary_text]}
    return {
        "status": {"status_str": status_str, "completed": True, "messages": []},
        "outputs": outputs,
    }


def _make_png_bytes(width: int, height: int) -> bytes:
    """Minimal valid greyscale PNG built from stdlib `struct`/`zlib` only —
    exactly enough structure (signature + IHDR + one IDAT + IEND) for
    `driver.png_dimensions` to read back the width/height it wrote,
    without pulling in an image library the driver itself is forbidden
    from depending on."""
    signature = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data))
        )

    ihdr = chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )  # 8-bit RGB, no interlace
    row = b"\x00" + b"\x00\x00\x00" * width  # filter-type-0 byte + RGB pixels
    idat = chunk(b"IDAT", zlib.compress(row * height))
    iend = chunk(b"IEND", b"")
    return signature + ihdr + idat + iend


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


class TestParseCanvasStateField:
    def test_extracts_bool_field(self):
        text = "CanvasState(converged=True, committed_fraction=1.0, steps_used=3)"
        assert driver.parse_canvas_state_field(text, "converged", "bool") is True

    def test_extracts_false_bool_field(self):
        text = "CanvasState(converged=False, committed_fraction=0.99, steps_used=48)"
        assert driver.parse_canvas_state_field(text, "converged", "bool") is False

    def test_extracts_float_field(self):
        text = "CanvasState(converged=True, committed_fraction=0.9961, steps_used=3)"
        value = driver.parse_canvas_state_field(text, "committed_fraction", "float")
        assert value == pytest.approx(0.9961)

    def test_extracts_int_field(self):
        text = "CanvasState(converged=True, committed_fraction=1.0, steps_used=48)"
        assert driver.parse_canvas_state_field(text, "steps_used", "int") == 48

    def test_extracts_field_from_full_repr_with_trailing_fields(self):
        # The real dataclass repr (dgemma/types.py:CanvasState) has more
        # fields than the trimmed fixtures elsewhere in this module use —
        # confirm the regex isn't anchored to end-of-string.
        text = (
            "CanvasState(text='hi', canvas_ids=[1, 2, 3], converged=True, "
            "committed_fraction=1.0, steps_used=3, thought=None, "
            "stray_thought_delimiter=False, turn_closed=False, answer_tokens=0)"
        )
        assert driver.parse_canvas_state_field(text, "turn_closed", "bool") is False
        assert driver.parse_canvas_state_field(text, "answer_tokens", "int") == 0

    def test_raises_when_field_is_absent(self):
        text = "CanvasState(converged=True, committed_fraction=1.0, steps_used=3)"
        with pytest.raises(AssertionError, match="missing field 'turn_closed'"):
            driver.parse_canvas_state_field(text, "turn_closed", "bool")


class TestPngDimensions:
    def test_reads_width_and_height_from_a_valid_png(self):
        png_bytes = _make_png_bytes(11, 36)
        width, height = driver.png_dimensions(png_bytes)
        assert (width, height) == (11, 36)

    def test_raises_on_bad_signature(self):
        with pytest.raises(AssertionError, match="bad signature"):
            driver.png_dimensions(b"not a png at all, just text padding out")

    def test_raises_when_first_chunk_is_not_ihdr(self):
        bad = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\x00" + b"XXXX" + b"\x00" * 8
        with pytest.raises(AssertionError, match="expected IHDR"):
            driver.png_dimensions(bad)


class TestFetchViewBytes:
    def test_requests_view_with_filename_subfolder_type(self):
        fake_resp = MagicMock()
        fake_resp.raise_for_status.side_effect = None
        fake_resp.content = b"\x89PNG\r\n\x1a\nfakebytes"
        with patch(
            "tests.e2e.driver.requests.get", return_value=fake_resp
        ) as mock_get:
            result = driver.fetch_view_bytes(
                "http://127.0.0.1:8199",
                filename="heatmap_00001_.png",
                subfolder="sub",
                folder_type="temp",
            )
        assert result == fake_resp.content
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert args[0] == "http://127.0.0.1:8199/view"
        assert kwargs["params"] == {
            "filename": "heatmap_00001_.png",
            "subfolder": "sub",
            "type": "temp",
        }

    def test_raises_on_http_error_status(self):
        fake_resp = MagicMock()
        fake_resp.raise_for_status.side_effect = Exception("HTTP 404")
        with patch("tests.e2e.driver.requests.get", return_value=fake_resp):
            with pytest.raises(Exception, match="HTTP 404"):
                driver.fetch_view_bytes("http://127.0.0.1:8199", filename="missing.png")


class TestAssertS2FullKnobHonest:
    def test_passes_on_a_well_formed_converged_entry(self):
        entry = _history_entry(
            string_text="Why do birds suddenly appear? Every time you are near.",
            canvas_state_text=(
                "CanvasState(converged=True, committed_fraction=1.0, steps_used=48)"
            ),
        )
        rendered_text, canvas_state_text = driver.assert_s2_full_knob_honest(entry)
        assert "birds" in rendered_text
        assert "steps_used=48" in canvas_state_text

    def test_passes_on_a_well_formed_non_converged_entry(self):
        # issue #22 honesty finding: converged=False is not itself a
        # failure — only converged=True paired with committed_fraction<1.0
        # (or the reverse) is the contradiction this assertion catches.
        entry = _history_entry(
            canvas_state_text=(
                "CanvasState(converged=False, committed_fraction=0.9961, steps_used=48)"
            )
        )
        driver.assert_s2_full_knob_honest(entry)

    def test_fails_when_status_is_not_success(self):
        entry = _history_entry(status_str="error")
        with pytest.raises(AssertionError, match="did not report success"):
            driver.assert_s2_full_knob_honest(entry)

    def test_fails_when_string_leaks_thought_channel_start_token(self):
        entry = _history_entry(string_text="<|channel>thought\nstray leak")
        with pytest.raises(AssertionError, match="leaked the raw thought-channel start"):
            driver.assert_s2_full_knob_honest(entry)

    def test_fails_when_string_leaks_thought_channel_end_token(self):
        entry = _history_entry(string_text="leaked content\n<channel|>")
        with pytest.raises(AssertionError, match="leaked the raw thought-channel end"):
            driver.assert_s2_full_knob_honest(entry)

    def test_fails_when_converged_true_but_committed_fraction_not_one(self):
        entry = _history_entry(
            canvas_state_text=(
                "CanvasState(converged=True, committed_fraction=0.98, steps_used=48)"
            )
        )
        with pytest.raises(AssertionError, match="converged=True must mean"):
            driver.assert_s2_full_knob_honest(entry)

    def test_fails_when_committed_fraction_one_but_converged_false(self):
        entry = _history_entry(
            canvas_state_text=(
                "CanvasState(converged=False, committed_fraction=1.0, steps_used=48)"
            )
        )
        with pytest.raises(AssertionError, match="same internal contradiction"):
            driver.assert_s2_full_knob_honest(entry)


class TestAssertS3ThinkingToggleHonest:
    def test_passes_when_answer_is_non_empty_and_converged(self):
        entry = _history_entry(
            string_text="A real answer.",
            canvas_state_text=(
                "CanvasState(converged=True, committed_fraction=1.0, steps_used=48)"
            ),
        )
        rendered = driver.assert_s3_thinking_toggle_honest(entry)
        assert rendered == "A real answer."

    def test_passes_when_answer_is_empty_but_not_fully_converged(self):
        # Empty answer alone isn't the #9 contradiction — only paired with
        # a fully-converged validity readout.
        entry = _history_entry(
            string_text="",
            canvas_state_text=(
                "CanvasState(converged=False, committed_fraction=0.4, steps_used=48)"
            ),
        )
        driver.assert_s3_thinking_toggle_honest(entry)

    def test_fails_on_the_number_9_contradiction_empty_string_fully_converged(self):
        entry = _history_entry(
            string_text="",
            canvas_state_text=(
                "CanvasState(converged=True, committed_fraction=1.0, steps_used=48)"
            ),
        )
        with pytest.raises(AssertionError, match="issue #9 contradiction"):
            driver.assert_s3_thinking_toggle_honest(entry)

    def test_fails_on_the_number_9_contradiction_whitespace_only_string(self):
        entry = _history_entry(
            string_text="   \n  ",
            canvas_state_text=(
                "CanvasState(converged=True, committed_fraction=1.0, steps_used=48)"
            ),
        )
        with pytest.raises(AssertionError, match="issue #9 contradiction"):
            driver.assert_s3_thinking_toggle_honest(entry)

    def test_corroborating_signature_matches_when_present(self):
        entry = _history_entry(
            string_text="A real answer.",
            canvas_state_text=(
                "CanvasState(text='hi', converged=True, committed_fraction=1.0, "
                "steps_used=48, turn_closed=True, answer_tokens=12)"
            ),
        )
        driver.assert_s3_thinking_toggle_honest(entry)

    def test_corroborating_signature_matches_the_number_9_empty_answer_specimen(self):
        # The actual issue #9 specimen shape this corroboration branch
        # exists to legibly confirm: empty answer, NOT fully converged
        # (so the primary assertion doesn't already catch it), and the
        # turn_closed=False/answer_tokens=0 signature present and matching.
        entry = _history_entry(
            string_text="",
            canvas_state_text=(
                "CanvasState(text='', converged=False, committed_fraction=0.4, "
                "steps_used=48, turn_closed=False, answer_tokens=0)"
            ),
        )
        driver.assert_s3_thinking_toggle_honest(entry)

    def test_corroborating_signature_mismatch_raises_when_empty_answer(self):
        # An empty answer whose turn_closed/answer_tokens don't match the
        # documented #9 signature (e.g. turn_closed=True despite no
        # answer) is itself an anomaly worth surfacing, distinct from the
        # primary #9 contradiction check.
        entry = _history_entry(
            string_text="",
            canvas_state_text=(
                "CanvasState(text='', converged=False, committed_fraction=0.4, "
                "steps_used=48, turn_closed=True, answer_tokens=5)"
            ),
        )
        with pytest.raises(AssertionError, match="did not match issue #9's documented"):
            driver.assert_s3_thinking_toggle_honest(entry)

    def test_fails_when_status_is_not_success(self):
        entry = _history_entry(status_str="error")
        with pytest.raises(AssertionError, match="did not report success"):
            driver.assert_s3_thinking_toggle_honest(entry)


class TestAssertS4TraceReadoutHonest:
    def test_passes_on_a_well_formed_consistent_entry(self):
        entry = _trace_history_entry()
        readout = driver.assert_s4_trace_readout_honest(entry, cell_px=6)
        assert readout["steps_used"] == 15
        assert readout["summary_steps"] == 15
        assert readout["expected_heatmap_height"] == 90
        assert readout["image_ref"]["filename"] == "heatmap_00001_.png"

    def test_fails_when_status_is_not_success(self):
        entry = _trace_history_entry(status_str="error")
        with pytest.raises(AssertionError, match="did not report success"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_heatmap_output_missing(self):
        entry = _trace_history_entry(include_heatmap_output=False)
        with pytest.raises(AssertionError, match="expected trace heatmap preview"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_heatmap_images_list_is_empty(self):
        entry = _trace_history_entry()
        entry["outputs"][driver.TRACE_HEATMAP_PREVIEW_NODE_ID] = {"images": []}
        with pytest.raises(AssertionError, match="PreviewImage output was empty"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_image_ref_missing_filename(self):
        entry = _trace_history_entry(image_ref={"type": "temp"})
        with pytest.raises(AssertionError, match="missing 'filename'"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_summary_output_missing(self):
        entry = _trace_history_entry(include_summary_output=False)
        with pytest.raises(AssertionError, match="expected trace summary preview"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_summary_has_no_steps_line(self):
        entry = _trace_history_entry(summary_text="scheduler=Foo config={}\nno steps line here")
        with pytest.raises(AssertionError, match="missing a 'steps=N' line"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_summary_steps_disagrees_with_canvas_state_steps_used(self):
        entry = _trace_history_entry(
            canvas_state_text=(
                "CanvasState(converged=True, committed_fraction=1.0, steps_used=48)"
            ),
            summary_text="scheduler=Foo config={}\nsteps=15\nmask-token corroboration: vacuous",
        )
        with pytest.raises(AssertionError, match="disagrees with CanvasState"):
            driver.assert_s4_trace_readout_honest(entry)

    def test_fails_when_canvas_state_output_missing(self):
        entry = _trace_history_entry(include_canvas_output=False)
        with pytest.raises(AssertionError, match="expected CanvasState preview"):
            driver.assert_s4_trace_readout_honest(entry)
