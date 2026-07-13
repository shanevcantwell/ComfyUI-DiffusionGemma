"""Same-in/same-out cross-run statelessness enforcement (#35 R5, F5;
ADR-CDG-008 Correction 1 `STATELESS-CORE`; ADR-CDG-010 Decision 7;
ADR-CDG-011 clause 8).

This IS the enforcement surface ARCHITECTURE.md's enforcement-surface table
names as "Same-in/same-out walker/pin statelessness (identical calls ->
identical effective-knob telemetry) | Same-in/same-out test on one loaded
model".

The invariant: two identical `run_diffusion` calls on one loaded (fake)
model must produce identical effective-knob telemetry and identical
results — because `run_diffusion` builds a fresh `EntropyBoundScheduler` /
`_FrameCollector` / `StepEndComposite` every call
(`dgemma/loop.py:run_diffusion`), never caching one across calls. Today that
freshness is enforced by the function body simply never reading a
module-level or `dgemma_model`-attached scheduler — this test module makes
that containment an assertion, not an implementation detail a future
refactor (e.g. CDG-008 Phase-2's MCP state manager memoizing a scheduler for
performance) could silently break.

Two tiers, matching the hook-lifecycle module's structure:

- `TestSchedulerFreshPerCall` — unit-level proof against the real
  `dgemma.loop.run_diffusion` body: each call constructs its OWN
  `EntropyBoundScheduler` instance (never the same object twice), so a
  mutation a future walker/pin participant makes to one call's scheduler
  config is structurally unreachable from the next call — there is no
  shared object for the mutation to survive on.
- `TestSameInSameOutTelemetry` — the behavioral half: two identical calls
  produce byte-identical `CanvasTrace.scheduler_config` and per-frame
  `t`/`temperature`/`committed_fraction_per_example` telemetry, and a
  deliberate `register_to_config` mutation performed *during* call 1 (the
  shape a future walker/pin participant would perform) is invisible to
  call 2 — proving the containment is real, not merely "nothing happens to
  mutate it today."
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.payloads import Constraints, Pin
from dgemma.types import DGemmaModel


class FakeSchedulerOutput:
    def __init__(self, accepted: list[list[bool]]):
        self.accepted_index = torch.tensor(accepted, dtype=torch.bool)


class FakePipelineOutput:
    def __init__(self, sequences: list[torch.Tensor]):
        self.sequences = sequences
        self.texts = ["<<unused>>"]


class FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _fake_model() -> DGemmaModel:
    return DGemmaModel(
        model=object(), processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


class _RecordingFrozenConfig:
    """Minimal `FrozenDict`-alike (mirrors `tests/conftest.py`'s
    `FakeFrozenConfig`): direct attribute-set raises; the only mutation path
    is `register_to_config`, which rebuilds the dict wholesale — the same
    resolution `tests/conftest.py`'s module docstring pins against the real
    `diffusers.configuration_utils.FrozenDict`."""

    def __init__(self, **kwargs):
        object.__setattr__(self, "_values", dict(kwargs))

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError(f"frozen — use register_to_config, not direct set of {name!r}")


def _install_stateless_fakes(monkeypatch, *, scheduler_registry: list, num_steps: int = 2):
    """Builds a REAL `EntropyBoundScheduler`-shaped fake with a genuine
    `register_to_config` mutation path (not just a static kwargs dict), so a
    test can mutate config mid-call and check whether that mutation survives
    into the next call's fresh instance. `scheduler_registry` collects every
    constructed scheduler instance in call order — the seam
    `TestSchedulerFreshPerCall` reads to prove distinct objects per call."""

    class FakeScheduler:
        def __init__(self, *, entropy_bound, t_max, t_min, num_inference_steps):
            self._config = _RecordingFrozenConfig(
                entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
            )
            self.num_inference_steps = num_inference_steps
            scheduler_registry.append(self)

        @property
        def config(self):
            return self._config

        def register_to_config(self, **kwargs):
            merged = dict(object.__getattribute__(self._config, "_values"))
            merged.update(kwargs)
            self._config = _RecordingFrozenConfig(**merged)

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self._scheduler = scheduler
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            for step_idx in range(num_steps):
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True]]),
                    "canvas": torch.tensor([[step_idx]]),
                }
                callback(self, step_idx, step_idx, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor([num_steps], dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


class TestSchedulerFreshPerCall:
    """`run_diffusion` must construct a brand-new scheduler every call —
    never reuse or cache one — so there is no shared object for cross-call
    mutation to ride on. This is the structural precondition
    `TestSameInSameOutTelemetry` depends on; if a future refactor made
    `run_diffusion` accept/reuse a caller-supplied scheduler, this test would
    catch it independent of whether any test happened to exercise a
    mutation."""

    def test_two_calls_construct_two_distinct_scheduler_objects(self, monkeypatch):
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry)

        run_diffusion(_fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2)
        run_diffusion(_fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2)

        assert len(registry) == 2
        assert registry[0] is not registry[1]

    def test_second_calls_scheduler_config_reflects_only_its_own_ctor_args(self, monkeypatch):
        """Even with identical ctor args, the second call's scheduler is a
        separate object whose config was built fresh from THIS call's
        arguments — never carrying over a mutation from the first call's
        instance (proven together with `TestSameInSameOutTelemetry` below,
        which performs the actual mutation)."""
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry)

        run_diffusion(_fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2)
        run_diffusion(_fake_model(), "hi", entropy_bound=0.2, t_min=0.4, t_max=0.8, num_inference_steps=2)

        assert registry[0].config.entropy_bound == 0.1
        assert registry[1].config.entropy_bound == 0.2


class TestSameInSameOutTelemetry:
    """The behavioral core of F5: identical `run_diffusion` calls yield
    identical effective-knob telemetry, and a mutation performed during one
    call cannot leak into the next."""

    def test_two_identical_calls_yield_identical_scheduler_config(self, monkeypatch):
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry)

        _, _, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )
        _, _, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )

        assert trace1.scheduler_config == trace2.scheduler_config

    def test_two_identical_calls_yield_identical_frame_telemetry(self, monkeypatch):
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry, num_steps=3)

        _, state1, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=3
        )
        _, state2, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=3
        )

        telemetry1 = [(f.t, f.temperature, f.committed_fraction_per_example) for f in trace1.frames]
        telemetry2 = [(f.t, f.temperature, f.committed_fraction_per_example) for f in trace2.frames]
        assert telemetry1 == telemetry2
        assert state1.committed_fraction == state2.committed_fraction
        assert state1.steps_used == state2.steps_used

    def test_mid_call_config_mutation_does_not_survive_into_the_next_call(self, monkeypatch):
        """Enforces containment rather than trusting happenstance: simulate a
        future walker/pin participant calling `register_to_config` mid-run
        (ADR-CDG-011 Decision 4's exact write mechanism) DURING call 1's
        first step, via a callback-adjacent hook wired through the fake
        pipeline. Call 2, built fresh, must read the ORIGINAL requested
        config — never the mutated one — because `run_diffusion` never
        retains a reference to call 1's scheduler."""
        registry: list = []

        class FakeScheduler:
            def __init__(self, *, entropy_bound, t_max, t_min, num_inference_steps):
                self._config = _RecordingFrozenConfig(
                    entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
                )
                self.num_inference_steps = num_inference_steps
                registry.append(self)

            @property
            def config(self):
                return self._config

            def register_to_config(self, **kwargs):
                merged = dict(object.__getattribute__(self._config, "_values"))
                merged.update(kwargs)
                self._config = _RecordingFrozenConfig(**merged)

        mutate_on_first_call = {"armed": True}

        class FakePipeline:
            def __init__(self, model, scheduler, processor):
                self._scheduler = scheduler
                self.eos_token_id = 999

            def __call__(self, **kwargs):
                callback = kwargs["callback_on_step_end"]
                if mutate_on_first_call["armed"]:
                    # Simulate a walker/pin participant mutating the
                    # scheduler's config mid-run (the exact mechanism
                    # ADR-CDG-011 Decision 4 names) — armed only for the
                    # first `run_diffusion` call in this test.
                    self._scheduler.register_to_config(entropy_bound=0.999)
                    mutate_on_first_call["armed"] = False
                for step_idx in range(2):
                    callback_kwargs = {
                        "scheduler_output": FakeSchedulerOutput([[True]]),
                        "canvas": torch.tensor([[step_idx]]),
                    }
                    callback(self, step_idx, step_idx, callback_kwargs)
                return FakePipelineOutput(sequences=[torch.tensor([2], dtype=torch.long)])

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)

        _, _, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )
        _, _, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )

        # Call 1's scheduler instance really was mutated (sanity check the
        # scenario is live, not a no-op).
        assert registry[0].config.entropy_bound == pytest.approx(0.999)
        # Call 2's fresh scheduler was never touched by call 1's mutation —
        # it reads the requested 0.1, not the leaked 0.999.
        assert registry[1].config.entropy_bound == pytest.approx(0.1)
        # And the trace `run_diffusion` returns for call 2 reports the
        # ORIGINAL requested/effective config `run_diffusion` was called
        # with, not call 1's mutated in-scheduler value — the field that
        # would lie if the engine ever leaked scheduler state across calls.
        assert trace2.scheduler_config["entropy_bound"] == pytest.approx(0.1)
        assert trace1.scheduler_config["entropy_bound"] == pytest.approx(0.1)


class TestPinStatePerRun:
    """ADR-CDG-010 Decision 7 (issue #64 Phase 2, plan §5
    `TestPinStatePerRun`): two identical `run_diffusion(constraints=...)`
    calls on one loaded (fake) model yield identical `pinned_mask`
    telemetry — no pin-derived state observable before the payload that
    created it, and nothing carried over from a previous call's
    `Constraints`. No pin PARTICIPANT exists yet (Phase 3); this proves the
    Phase 2 `pinned_mask` derivation itself is per-run stateless."""

    def test_two_identical_calls_with_same_constraints_yield_identical_pinned_mask(self, monkeypatch):
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry, num_steps=2)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        _, _, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8,
            num_inference_steps=2, constraints=constraints,
        )
        _, _, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8,
            num_inference_steps=2, constraints=constraints,
        )

        masks1 = [f.pinned_mask.tolist() for f in trace1.frames]
        masks2 = [f.pinned_mask.tolist() for f in trace2.frames]
        assert masks1 == masks2
        assert all(mask == [True] for mask in masks1)

    def test_pinned_mask_does_not_survive_into_a_call_with_no_constraints(self, monkeypatch):
        """The exact cross-call-leak shape the statelessness surface exists
        to catch: a run WITH `constraints=` followed by a run with NO
        `constraints=` must not somehow carry the first run's pin positions
        forward — `run_diffusion` builds a fresh `_FrameCollector` every
        call, so there is no shared object for the mask to ride on."""
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry, num_steps=2)
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))

        _, _, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8,
            num_inference_steps=2, constraints=constraints,
        )
        _, _, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2,
        )

        assert all(f.pinned_mask is not None for f in trace1.frames)
        assert all(f.pinned_mask is None for f in trace2.frames)


class TestEffectiveKnobTelemetryStatelessness:
    """ADR-CDG-011 clause 8 (issue #64 Phase 2): two identical `run_diffusion`
    calls yield identical `effective_entropy_bound`/`effective_t_min`/
    `effective_t_max` telemetry, and — shared surface with
    `TestSameInSameOutTelemetry` above — a mid-call `register_to_config`
    mutation never survives into the next call's effective-knob readings.
    No walker exists yet (Phase 4); this proves the Phase 2 read-path itself
    (scheduler.config, read fresh per callback) is per-run stateless."""

    def test_two_identical_calls_yield_identical_effective_knob_telemetry(self, monkeypatch):
        registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=registry, num_steps=2)

        _, _, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )
        _, _, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )

        telemetry1 = [(f.effective_entropy_bound, f.effective_t_min, f.effective_t_max) for f in trace1.frames]
        telemetry2 = [(f.effective_entropy_bound, f.effective_t_min, f.effective_t_max) for f in trace2.frames]
        assert telemetry1 == telemetry2
        assert all(t == (pytest.approx(0.1), pytest.approx(0.4), pytest.approx(0.8)) for t in telemetry1)

    def test_mid_call_mutation_of_effective_knobs_does_not_survive_into_next_call(self, monkeypatch):
        """Same mutation mechanism as `TestSameInSameOutTelemetry`'s sibling
        test above (`register_to_config` mid-run, the exact write path a
        future walker uses), but asserting on the NEW `effective_*` frame
        fields directly rather than only `trace.scheduler_config`."""
        registry: list = []

        class FakeScheduler:
            def __init__(self, *, entropy_bound, t_max, t_min, num_inference_steps):
                self._config = _RecordingFrozenConfig(
                    entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
                )
                self.num_inference_steps = num_inference_steps
                registry.append(self)

            @property
            def config(self):
                return self._config

            def register_to_config(self, **kwargs):
                merged = dict(object.__getattribute__(self._config, "_values"))
                merged.update(kwargs)
                self._config = _RecordingFrozenConfig(**merged)

        mutate_on_first_call = {"armed": True}

        class FakePipeline:
            def __init__(self, model, scheduler, processor):
                self._scheduler = scheduler
                self.eos_token_id = 999

            def __call__(self, **kwargs):
                callback = kwargs["callback_on_step_end"]
                if mutate_on_first_call["armed"]:
                    self._scheduler.register_to_config(entropy_bound=0.999, t_min=0.11, t_max=0.22)
                    mutate_on_first_call["armed"] = False
                for step_idx in range(2):
                    callback_kwargs = {
                        "scheduler_output": FakeSchedulerOutput([[True]]),
                        "canvas": torch.tensor([[step_idx]]),
                    }
                    callback(self, step_idx, step_idx, callback_kwargs)
                return FakePipelineOutput(sequences=[torch.tensor([2], dtype=torch.long)])

        monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
        monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)

        _, _, trace1 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )
        _, _, trace2 = run_diffusion(
            _fake_model(), "hi", entropy_bound=0.1, t_min=0.4, t_max=0.8, num_inference_steps=2
        )

        # Call 1's frames show the mutated values (the mutation happened
        # before the pipeline's per-step loop fires any callback in this
        # fixture, so every frame in call 1 reflects it).
        assert all(f.effective_entropy_bound == pytest.approx(0.999) for f in trace1.frames)
        assert all(f.effective_t_min == pytest.approx(0.11) for f in trace1.frames)
        assert all(f.effective_t_max == pytest.approx(0.22) for f in trace1.frames)

        # Call 2's fresh scheduler was never touched by call 1's mutation.
        assert all(f.effective_entropy_bound == pytest.approx(0.1) for f in trace2.frames)
        assert all(f.effective_t_min == pytest.approx(0.4) for f in trace2.frames)
        assert all(f.effective_t_max == pytest.approx(0.8) for f in trace2.frames)
