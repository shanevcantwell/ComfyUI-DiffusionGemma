"""Unit tests for `surfaces/mcp/commands/generate.py` — `generate` /
`cancel_run`.

Reuses `tests/test_run_diffusion_statelessness.py`'s fake-pipeline
installer (`_install_stateless_fakes`, `_fake_model`) — the same fake
`EntropyBoundScheduler`/`DGemmaPipeline` shape the engine-level
statelessness suite already validates against, so this module doesn't
reinvent a second fixture for the same contract.

Three concerns:

- Thin-adapter correctness: `generate` unpacks args, calls `run_diffusion`
  exactly once, wraps the 3-tuple into a JSON-safe dict — no logic beyond
  that (ARCHITECTURE.md surface-tier rules).
- The `run_id`/`cancel_run` wiring (issue #38's MCP remainder): a
  `cancel_run` call against a REGISTERED, still-running `run_id` sets the
  shared `threading.Event`; `should_cancel` (that event's `.is_set`) is what
  `run_diffusion` polls once per step. `cancel_run` against an unknown
  `run_id` reports `found=False` rather than raising.
- The registry never leaks: after a call completes (or is cancelled), its
  `run_id` is removed from `_active_runs` — checked directly, since a
  leaked entry would be exactly the kind of accreting cross-call state
  ADR-CDG-008 Correction 1 forbids (even though it lives in this module, not
  `StateManager`, the same discipline applies — see `generate.py`'s module
  docstring on why it's kept out of `StateManager` rather than merely
  "somewhere else").
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest

from surfaces.mcp.commands import generate as generate_module
from surfaces.mcp.state_manager import StateManager
from tests.test_run_diffusion_statelessness import _fake_model, _install_stateless_fakes


def _manager_with_fake_model() -> StateManager:
    manager = StateManager()
    manager._model = _fake_model()
    manager._repo_id = "fake/repo"
    manager._quant = "none"
    return manager


class TestGenerateThinAdapter:
    def test_generate_without_prompt_returns_structured_error(self):
        manager = _manager_with_fake_model()
        result = asyncio.run(generate_module.generate(manager, {}))
        assert "error" in result
        assert "prompt" in result["error"]

    def test_generate_without_loaded_model_raises_actionable_error(self):
        manager = StateManager()  # no model loaded
        with pytest.raises(RuntimeError, match="No DiffusionGemma model is loaded"):
            asyncio.run(generate_module.generate(manager, {"prompt": "hi"}))

    def test_generate_wraps_run_diffusion_result(self, monkeypatch):
        _install_stateless_fakes(monkeypatch, scheduler_registry=[], num_steps=2)
        manager = _manager_with_fake_model()

        result = asyncio.run(
            generate_module.generate(manager, {"prompt": "hi", "num_inference_steps": 2})
        )

        assert "text" in result
        assert set(result["canvas_state"]) == {
            "converged",
            "committed_fraction",
            "steps_used",
            "thought",
            "stray_thought_delimiter",
            "turn_closed",
            "answer_tokens",
            "finished_honestly",
        }
        assert set(result["trace_summary"]) == {"scheduler_name", "scheduler_config", "num_frames"}
        assert "frames" not in result["trace_summary"]  # include_frames defaults False

    def test_generate_include_frames_true_adds_frame_list(self, monkeypatch):
        _install_stateless_fakes(monkeypatch, scheduler_registry=[], num_steps=3)
        manager = _manager_with_fake_model()

        result = asyncio.run(
            generate_module.generate(
                manager, {"prompt": "hi", "num_inference_steps": 3, "include_frames": True}
            )
        )

        assert "frames" in result["trace_summary"]
        assert len(result["trace_summary"]["frames"]) == result["trace_summary"]["num_frames"]
        frame = result["trace_summary"]["frames"][0]
        assert set(frame) == {
            "canvas_idx",
            "step_idx",
            "t",
            "temperature",
            "committed_fraction_per_example",
        }

    def test_generate_schema_requires_only_prompt(self):
        tools = {t.name: t for t in generate_module.get_tools()}
        schema = tools["generate"].inputSchema
        assert schema["required"] == ["prompt"]


class TestCancelRunWiring:
    def test_cancel_run_without_run_id_returns_structured_error(self):
        manager = _manager_with_fake_model()
        result = asyncio.run(generate_module.cancel_run(manager, {}))
        assert "error" in result

    def test_cancel_run_against_unknown_run_id_reports_not_found(self):
        manager = _manager_with_fake_model()
        result = asyncio.run(generate_module.cancel_run(manager, {"run_id": "no-such-run"}))
        assert result == {"found": False, "run_id": "no-such-run"}

    def test_registry_never_leaks_a_completed_runs_run_id(self, monkeypatch):
        """After a `generate` call with a `run_id` completes, that id must
        no longer be in `_active_runs` — else a `cancel_run` against a
        long-finished run would falsely report `found=True`, and the
        registry would accrete one entry per past call forever (the exact
        cross-call-accretion shape ADR-CDG-008 Correction 1 forbids)."""
        _install_stateless_fakes(monkeypatch, scheduler_registry=[], num_steps=2)
        manager = _manager_with_fake_model()

        asyncio.run(
            generate_module.generate(
                manager, {"prompt": "hi", "num_inference_steps": 2, "run_id": "run-A"}
            )
        )

        assert "run-A" not in generate_module._active_runs
        result = asyncio.run(generate_module.cancel_run(manager, {"run_id": "run-A"}))
        assert result == {"found": False, "run_id": "run-A"}

    def test_cancel_run_sets_the_event_a_concurrent_generate_call_polls(self, monkeypatch):
        """End-to-end proof of the wiring: register a run_id, call
        `cancel_run` against it while it's "in flight" (simulated directly,
        without spinning up a real background thread — the unit under test
        is the registry + event plumbing, not asyncio.to_thread's own
        scheduling), and confirm `should_cancel` (the registered event's
        `.is_set`) flips to True.
        """
        event = generate_module._register_run("run-B")
        assert event.is_set() is False

        result = asyncio.run(generate_module.cancel_run(_manager_with_fake_model(), {"run_id": "run-B"}))

        assert result == {"found": True, "run_id": "run-B", "status": "cancel requested"}
        assert event.is_set() is True

        generate_module._unregister_run("run-B")
        assert "run-B" not in generate_module._active_runs

    def test_generate_honors_cancel_run_mid_flight(self, monkeypatch):
        """The real integration: a `generate` call with `run_id="run-C"` is
        started; while its fake pipeline is mid-step-loop, a concurrent
        `cancel_run("run-C")` call sets the event, and `run_diffusion`'s
        `should_cancel` (polled once per step by `StepEndComposite`) picks
        it up, truncating the run and returning the partial result (issue
        #38's "a cancelled experiment run is still data" clause) rather than
        running to completion."""
        scheduler_registry: list = []

        # A fake pipeline whose callback triggers cancellation from a
        # SEPARATE thread partway through its step loop — mirrors a real
        # client's concurrent cancel_run call racing the in-flight generate.
        cancelled_from_outside = threading.Event()

        def _install_cancel_triggering_fakes(monkeypatch, num_steps=5):
            import torch

            from dgemma.loop import DEFAULT_T_MAX, DEFAULT_T_MIN

            class FakeSchedulerOutput:
                def __init__(self, accepted):
                    self.accepted_index = torch.tensor(accepted, dtype=torch.bool)

            class FakePipelineOutput:
                def __init__(self, sequences):
                    self.sequences = sequences
                    self.texts = ["<<unused>>"]

            class FakeScheduler:
                def __init__(self, *, entropy_bound, t_max, t_min, num_inference_steps):
                    self.num_inference_steps = num_inference_steps
                    scheduler_registry.append(self)

            class FakePipeline:
                def __init__(self, model, scheduler, processor):
                    self.eos_token_id = 999

                def __call__(self, **kwargs):
                    callback = kwargs["callback_on_step_end"]
                    for step_idx in range(num_steps):
                        if step_idx == 1:
                            # Simulate a concurrent cancel_run call landing
                            # after step 0's frame is already captured.
                            asyncio.run(
                                generate_module.cancel_run(
                                    _manager_with_fake_model(), {"run_id": "run-C"}
                                )
                            )
                        callback_kwargs = {
                            "scheduler_output": FakeSchedulerOutput([[True]]),
                            "canvas": torch.tensor([[step_idx]]),
                        }
                        callback(self, step_idx, step_idx, callback_kwargs)
                    return FakePipelineOutput(sequences=[torch.tensor([num_steps], dtype=torch.long)])

            monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
            monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)

        _install_cancel_triggering_fakes(monkeypatch, num_steps=5)
        manager = _manager_with_fake_model()

        result = asyncio.run(
            generate_module.generate(
                manager,
                {"prompt": "hi", "num_inference_steps": 5, "run_id": "run-C", "include_frames": True},
            )
        )

        # Truncated well before the full 5 steps: capture-then-cancel means
        # the cancelled step's own frame IS included (ADR-CDG-010 amendment),
        # so exactly 2 frames survive (step 0 captured, step 1 triggers the
        # cancel-flag-set at the TOP of the loop body above but the callback
        # for step 1 still runs and captures before the composite's next
        # check trips on step 2 — the exact count is an implementation
        # detail of this fake; the load-bearing assertion is "fewer than
        # num_inference_steps", proving cancellation actually truncated the
        # run rather than completing it).
        assert result["trace_summary"]["num_frames"] < 5
        assert "run-C" not in generate_module._active_runs
