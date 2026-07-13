"""Self-tests for #35 R4's shared fake-pipeline/scheduler fixture
(`tests/conftest.py`: `FakeFrozenConfig`, `FakeEntropyBoundScheduler`,
`HookRecordingModel`, `FakeDiffusionGemmaPipeline`, `fake_pipeline_factory`).

This IS the enforcement test for the R4 row in `ARCHITECTURE.md`'s
enforcement-surface table ("Shared fake-pipeline/scheduler fixture (N
steps, mutable `config`, hook-recording model, `{"canvas":…}` application)
| `tests/conftest.py` fixture (gates R1/R5 testing)"): each clause of that
row gets its own test class below, pinning the fixture's behavior against
the real `diffusers==0.39.0` seam it was built to mirror (grounded by
reading `pipeline_diffusion_gemma.py`/`scheduling_entropy_bound.py`/
`configuration_utils.py` directly — not re-derived from memory) so a
future change to the fixture that silently drifts from that seam fails
here, not three test files downstream in R1/R5.

Deliberately exercises the fixture classes ONLY through the
`fake_pipeline_factory`/`fake_pipeline` pytest fixtures (never a direct
`from conftest import FakeEntropyBoundScheduler` module import): the suite
runs under `--import-mode=importlib` with no `tests/__init__.py`
(`pyproject.toml`'s documented reason — the repo-root `__init__.py` breaks
"prepend" mode's dotted-package resolution), under which `conftest.py`'s
own module is not reliably importable by name from a sibling test file —
only pytest's fixture-injection machinery reaches it. Going through the
fixtures is also the honest test of the actual public surface: every real
R1/R5 test will reach these classes the same way, never via a bare import.
"""
from __future__ import annotations

import pytest
import torch


class TestConfigurableStepCount:
    """R4 clause: "N steps" — the fixture must drive exactly the requested
    `num_inference_steps`, not a hardcoded count (the gap #35 F7 names in
    the pre-R4 per-file fakes: "fires callback once")."""

    def test_pipeline_drives_exactly_num_inference_steps_callbacks(self, fake_pipeline_factory):
        built = fake_pipeline_factory(num_inference_steps=7)
        seen_steps: list[int] = []
        built.pipeline(
            num_inference_steps=7,
            callback_on_step_end=lambda pipe, gstep, sidx, kwargs: (seen_steps.append(sidx), {})[1],
            callback_on_step_end_tensor_inputs=["canvas"],
        )
        assert seen_steps == list(range(7))

    def test_zero_callback_calls_when_num_inference_steps_differs_from_default(self, fake_pipeline_factory):
        """A test can drive a different step count than the fixture was
        built with — the pipeline call's own `num_inference_steps` kwarg
        governs the loop, exactly like the real pipeline (which reads its
        own `num_inference_steps` param, not anything cached on the
        scheduler at construction time)."""
        built = fake_pipeline_factory(num_inference_steps=4)
        seen_steps: list[int] = []
        built.pipeline(
            num_inference_steps=2,
            callback_on_step_end=lambda pipe, gstep, sidx, kwargs: (seen_steps.append(sidx), {})[1],
        )
        assert seen_steps == [0, 1]

    def test_no_callback_still_runs_all_steps(self, fake_pipeline_factory):
        """Omitting `callback_on_step_end` must not short-circuit the
        drive loop — mirrors the real pipeline's `if callback_on_step_end
        is not None` guard (`pipeline_diffusion_gemma.py:402`), which wraps
        only the callback invocation, not the step loop itself."""
        built = fake_pipeline_factory(num_inference_steps=3)
        result = built.pipeline(num_inference_steps=3)
        assert len(built.scheduler.step_calls) == 3
        assert result.sequences.shape == (1, 4)


class TestMutableSchedulerConfig:
    """R4 clause: "mutable `scheduler.config`" — resolved against the real
    `diffusers.EntropyBoundScheduler`/`ConfigMixin` class (pulled and read
    directly, see `tests/conftest.py`'s module docstring) as "mutable
    *through* `register_to_config`, frozen against direct attribute
    assignment" — the real `FrozenDict.__setattr__` raises
    (`configuration_utils.py:77-80`), so a fixture that allowed
    `scheduler.config.entropy_bound = x` to silently succeed would validate
    a shape the real class does not offer (the "lying sigma" class of bug,
    ADR-CDG-001)."""

    def test_config_reads_ctor_values(self, fake_pipeline_factory):
        built = fake_pipeline_factory(entropy_bound=0.25, t_min=0.1, t_max=0.5)
        assert built.scheduler.config.entropy_bound == 0.25
        assert built.scheduler.config.t_min == 0.1
        assert built.scheduler.config.t_max == 0.5

    def test_direct_config_attribute_assignment_raises(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        with pytest.raises(AttributeError):
            built.scheduler.config.entropy_bound = 0.9

    def test_register_to_config_rebuilds_config_wholesale(self, fake_pipeline_factory):
        built = fake_pipeline_factory(entropy_bound=0.1, t_min=0.4, t_max=0.8)
        built.scheduler.register_to_config(entropy_bound=0.5)
        # The mutated key changed...
        assert built.scheduler.config.entropy_bound == 0.5
        # ...and the untouched keys survive the rebuild (merged, not reset).
        assert built.scheduler.config.t_min == 0.4
        assert built.scheduler.config.t_max == 0.8

    def test_num_inference_steps_is_plain_attribute_not_config_mediated(self, fake_pipeline_factory):
        """Issue #20's own grounding (mirrored in `dgemma/loop.py`'s
        `_FrameCollector` docstring): the real scheduler's anneal formula
        divides by `self.num_inference_steps`, a plain instance attribute
        set directly in `__init__`/`set_timesteps` — never
        `self.config.num_inference_steps`. A fixture that only exposed this
        through `.config` would let a collector-denominator test pass
        against the wrong attribute."""
        built = fake_pipeline_factory(num_inference_steps=10)
        assert built.scheduler.num_inference_steps == 10
        built.scheduler.set_timesteps(6)
        assert built.scheduler.num_inference_steps == 6
        # config's OWN snapshot of the ctor-time value is untouched by
        # set_timesteps — exactly like the real class, where set_timesteps
        # reassigns the plain attribute only, never touching `.config`.
        assert built.scheduler.config.num_inference_steps == 10

    def test_set_timesteps_rejects_non_positive(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        with pytest.raises(ValueError, match="must be > 0"):
            built.scheduler.set_timesteps(0)


class TestHookRecordingModel:
    """R4 clause: "hook-recording model" — the surface #35 R5's "no hook
    survives a `run_diffusion` call" lifecycle test needs (`ARCHITECTURE.md`
    "Zero hooks after run" row). Built on a real `torch.nn.Module` so the
    `register_forward_hook`/`RemovableHandle` behavior under test is genuine
    PyTorch semantics, not a hand-rolled approximation of it."""

    def test_register_forward_hook_returns_real_removable_handle(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        handle = built.model.register_forward_hook(lambda mod, inp, out: None)
        assert hasattr(handle, "remove")
        assert built.model.live_hook_count == 1

    def test_removal_is_recorded_and_live_count_drops(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        handle = built.model.register_forward_hook(lambda mod, inp, out: None)
        handle.remove()
        assert built.model.live_hook_count == 0
        assert built.model.install_log == built.model.removal_log == [0]

    def test_multiple_hooks_tracked_independently(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        h1 = built.model.register_forward_hook(lambda mod, inp, out: None)
        h2 = built.model.register_forward_hook(lambda mod, inp, out: None)
        assert built.model.live_hook_count == 2
        h1.remove()
        assert built.model.live_hook_count == 1
        h2.remove()
        assert built.model.live_hook_count == 0

    def test_unremoved_hook_leaves_nonzero_live_count(self, fake_pipeline_factory):
        """The exact shape R5's lifecycle test asserts against: a hook
        installed in one "run" and never torn down (F4's leakage finding)
        must be OBSERVABLE as a nonzero `live_hook_count`, not silently
        invisible."""
        built = fake_pipeline_factory()
        built.model.register_forward_hook(lambda mod, inp, out: None)
        assert built.model.live_hook_count == 1

    def test_hook_actually_fires_on_forward(self, fake_pipeline_factory):
        """Not just bookkeeping — the installed hook is a REAL forward
        hook, invoked by a real forward pass."""
        built = fake_pipeline_factory()
        calls = []
        built.model.register_forward_hook(lambda mod, inp, out: calls.append(out.logits.shape))
        built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))
        assert calls == [(1, 4, built.model.vocab_size)]


class TestCanvasApplication:
    """R4 clause: `{"canvas": ...}` application — faithful to the real
    pipeline's `canvas = callback_outputs.pop("canvas", canvas)`
    (`pipeline_diffusion_gemma.py:407`), including the un-overridden
    default path."""

    def test_callback_returning_canvas_overwrites_scheduler_output(self, fake_pipeline_factory):
        built = fake_pipeline_factory(num_inference_steps=1, canvas_shape=(1, 4))
        override = torch.full((1, 4), 99, dtype=torch.long)

        result = built.pipeline(
            num_inference_steps=1,
            callback_on_step_end=lambda pipe, gstep, sidx, kwargs: {"canvas": override},
        )

        assert torch.equal(result.sequences, override)

    def test_no_canvas_key_keeps_scheduler_output(self, fake_pipeline_factory):
        """`{}.pop("canvas", canvas)` falls back to the scheduler's own
        `prev_sample` when the callback returns nothing — pure capture
        (mirrors `dgemma.loop._FrameCollector.on_step_end`, which always
        returns `{}`)."""
        built = fake_pipeline_factory(
            num_inference_steps=1, canvas_shape=(1, 4), accepted=torch.ones((1, 4), dtype=torch.bool)
        )

        result = built.pipeline(num_inference_steps=1, callback_on_step_end=lambda pipe, gstep, sidx, kwargs: {})

        # accepted=all-True -> prev_sample == sample (the zero-initialized
        # canvas the fixture starts from), never the override sentinel.
        assert torch.equal(result.sequences, torch.zeros((1, 4), dtype=torch.long))

    def test_overwritten_canvas_feeds_forward_into_next_step(self, fake_pipeline_factory):
        """A composition-ordering concern (R1): a canvas-write participant's
        overwrite must be visible to the NEXT step's model call, not just
        reflected in the final return — otherwise the composite's writers
        can't actually condition each other."""
        built = fake_pipeline_factory(num_inference_steps=2, canvas_shape=(1, 4))
        seen_inputs: list[torch.Tensor] = []

        original_forward = built.model.forward

        def spying_forward(decoder_input_ids, **kwargs):
            seen_inputs.append(decoder_input_ids.clone())
            return original_forward(decoder_input_ids, **kwargs)

        built.model.forward = spying_forward
        override = torch.full((1, 4), 7, dtype=torch.long)

        def cb(pipe, gstep, sidx, kwargs):
            return {"canvas": override} if sidx == 0 else {}

        built.pipeline(num_inference_steps=2, callback_on_step_end=cb)

        # Step 0 sees the initial zero canvas; step 1 must see the step-0
        # override, not the scheduler's own (un-overridden) prev_sample.
        assert torch.equal(seen_inputs[0], torch.zeros((1, 4), dtype=torch.long))
        assert torch.equal(seen_inputs[1], override)


class TestCallbackTensorInputsAllowlist:
    """Mirrors `check_inputs`'s allowlist validation
    (`pipeline_diffusion_gemma.py:155-161`): requesting a tensor input
    outside `_callback_tensor_inputs` must raise, not silently ignore."""

    def test_unknown_tensor_input_key_raises(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        with pytest.raises(ValueError, match="callback_on_step_end_tensor_inputs"):
            built.pipeline(
                num_inference_steps=1,
                callback_on_step_end=lambda *a: {},
                callback_on_step_end_tensor_inputs=["not_a_real_key"],
            )

    def test_scheduler_output_key_available_matching_dgemma_pipeline_widening(self, fake_pipeline_factory):
        """`dgemma.loop.DGemmaPipeline` widens `_callback_tensor_inputs` to
        add `"scheduler_output"` (`dgemma/loop.py:81`) so the callback can
        read `accepted_index` — the base fixture pipeline already allows
        it, matching that widened contract rather than the narrower
        upstream base-class default."""
        built = fake_pipeline_factory(num_inference_steps=1)
        seen_kwargs: dict = {}

        def cb(pipe, gstep, sidx, kwargs):
            seen_kwargs.update(kwargs)
            return {}

        built.pipeline(
            num_inference_steps=1, callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["scheduler_output"]
        )
        assert "scheduler_output" in seen_kwargs
        assert hasattr(seen_kwargs["scheduler_output"], "accepted_index")


class TestScriptedAcceptancePattern:
    """The `accepted` knob: composition-ordering tests (R1) want to script
    per-step acceptance ("nothing accepted until step N") rather than
    always-accept — this is the seam that makes that possible without
    reimplementing the entropy-bound math."""

    def test_callable_accepted_source_scripted_per_step(self, fake_pipeline_factory):
        def accepted_fn(step_idx: int) -> torch.Tensor:
            return torch.ones((1, 4), dtype=torch.bool) if step_idx >= 2 else torch.zeros((1, 4), dtype=torch.bool)

        built = fake_pipeline_factory(num_inference_steps=3, canvas_shape=(1, 4), accepted=accepted_fn)
        fractions: list[float] = []

        def cb(pipe, gstep, sidx, kwargs):
            fractions.append(kwargs["scheduler_output"].accepted_index.float().mean().item())
            return {}

        built.pipeline(num_inference_steps=3, callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["scheduler_output"])
        assert fractions == [0.0, 0.0, 1.0]

    def test_static_bool_tensor_accepted_every_step(self, fake_pipeline_factory):
        accepted = torch.tensor([[True, False, True, False]])
        built = fake_pipeline_factory(num_inference_steps=2, canvas_shape=(1, 4), accepted=accepted)

        result = built.pipeline(num_inference_steps=2)

        assert torch.equal(result.sequences, torch.tensor([[0, 0, 0, 0]]))
        # Every recorded step call used the same static pattern.
        assert len(built.scheduler.step_calls) == 2

    def test_bare_bool_accepted_source_broadcasts_to_shape(self, fake_pipeline_factory):
        """`accepted` can also be scripted as a plain bool return from a
        callable (not just a tensor) — `_accepted_index_for` broadcasts it
        to the sample's shape via `torch.full`, the non-tensor branch."""
        built = fake_pipeline_factory(
            num_inference_steps=1, canvas_shape=(1, 4), accepted=lambda step_idx: False
        )

        result = built.pipeline(num_inference_steps=1)

        assert torch.equal(result.sequences, torch.zeros((1, 4), dtype=torch.long))
        assert built.scheduler.step_calls[0]["timestep"] == 0


class TestPipelineTextDecoding:
    """The optional `texts` derivation: only populated when the processor
    exposes a real `.decode` — mirrors `DiffusionGemmaPipelineOutput.texts`
    being `None` unless `output_type == "text"` AND a processor is present
    (`pipeline_diffusion_gemma.py:451-453`)."""

    def test_processor_with_tokenizer_decode_populates_texts(self, fake_pipeline_factory):
        class _Tokenizer:
            def decode(self, ids, skip_special_tokens=True):
                return "DECODED:" + ",".join(str(i) for i in ids)

        class _Processor:
            tokenizer = _Tokenizer()

        built = fake_pipeline_factory(num_inference_steps=1, canvas_shape=(1, 4), processor=_Processor())

        result = built.pipeline(num_inference_steps=1)

        assert result.texts == ["DECODED:0,0,0,0"]

    def test_no_processor_leaves_texts_none(self, fake_pipeline_factory):
        built = fake_pipeline_factory(num_inference_steps=1, processor=None)

        result = built.pipeline(num_inference_steps=1)

        assert result.texts is None


class TestFakePipelineFactoryIsolation:
    """Each `fake_pipeline_factory()` call must build an independent triple
    — no shared mutable state leaking scheduler config, hook logs, or step
    history across builds (the exact statelessness R5/F5's same-in/same-out
    test needs to trust in the fixture itself before it can trust it in
    `run_diffusion`)."""

    def test_two_builds_do_not_share_scheduler_state(self, fake_pipeline_factory):
        first = fake_pipeline_factory(num_inference_steps=3)
        first.pipeline(num_inference_steps=3)
        second = fake_pipeline_factory(num_inference_steps=5)

        assert len(first.scheduler.step_calls) == 3
        assert len(second.scheduler.step_calls) == 0

    def test_two_builds_do_not_share_hook_logs(self, fake_pipeline_factory):
        first = fake_pipeline_factory()
        first.model.register_forward_hook(lambda mod, inp, out: None)
        second = fake_pipeline_factory()

        assert first.model.live_hook_count == 1
        assert second.model.live_hook_count == 0


class TestFakePipelineConvenienceFixture:
    """The bare `fake_pipeline` fixture (no factory call) — the common
    case for a test that doesn't need to vary the fixture's knobs."""

    def test_fake_pipeline_fixture_is_default_configured(self, fake_pipeline):
        assert fake_pipeline.scheduler.num_inference_steps == 4
        assert fake_pipeline.scheduler.config.entropy_bound == 0.1

    def test_fake_pipeline_fixture_is_fresh_per_test(self, fake_pipeline):
        """Regression guard against accidental fixture-scope widening: if
        `fake_pipeline` were ever changed to `scope="module"` or broader,
        step history would leak across tests silently."""
        assert fake_pipeline.scheduler.step_calls == []


class TestFrozenConfigViaScheduler:
    """Unit coverage of `FakeFrozenConfig`'s behavior, reached the same way
    every real test reaches it — off `built.scheduler.config` — since it's
    the load-bearing piece of the "mutable `scheduler.config`" resolution.
    (Not imported as a bare class: see this module's docstring on
    `--import-mode=importlib` — the fixture surface IS the public
    contract.)"""

    def test_getattr_returns_registered_value(self, fake_pipeline_factory):
        built = fake_pipeline_factory(entropy_bound=0.2, t_min=0.3, t_max=0.7)
        assert built.scheduler.config.entropy_bound == 0.2
        assert built.scheduler.config.t_min == 0.3
        assert built.scheduler.config.t_max == 0.7

    def test_getattr_missing_key_raises_attribute_error(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        with pytest.raises(AttributeError):
            _ = built.scheduler.config.not_a_registered_key

    def test_setattr_always_raises(self, fake_pipeline_factory):
        built = fake_pipeline_factory()
        with pytest.raises(AttributeError):
            built.scheduler.config.entropy_bound = 2.0

    def test_getitem_and_get(self, fake_pipeline_factory):
        built = fake_pipeline_factory(entropy_bound=0.3)
        assert built.scheduler.config["entropy_bound"] == 0.3
        assert built.scheduler.config.get("entropy_bound") == 0.3
        assert built.scheduler.config.get("missing", "default") == "default"

    def test_repr_surfaces_registered_values(self, fake_pipeline_factory):
        """Debugging aid: a failed composition-ordering assertion (R1) that
        prints `scheduler.config` in its failure message should show the
        actual registered values, not a bare object address."""
        built = fake_pipeline_factory(entropy_bound=0.3)
        assert "entropy_bound" in repr(built.scheduler.config)
        assert "0.3" in repr(built.scheduler.config)


class TestPipelineRegistersModulesAsAttributes:
    """Mirrors `register_modules` (`pipeline_diffusion_gemma.py:85`): the
    real pipeline exposes `self.model`/`self.scheduler`/`self.processor` as
    plain attributes — the exact reachability path #35's required ADR
    clauses name as explicitly NOT a sanctioned hook-installation door
    (`pipe.model` reachability), so a hook-lifecycle test (R5) needs a real
    attribute here to test that door is actually closed against."""

    def test_model_scheduler_processor_are_attributes(self, fake_pipeline_factory):
        processor = object()
        built = fake_pipeline_factory(processor=processor)

        assert built.pipeline.model is built.model
        assert built.pipeline.scheduler is built.scheduler
        assert built.pipeline.processor is processor
