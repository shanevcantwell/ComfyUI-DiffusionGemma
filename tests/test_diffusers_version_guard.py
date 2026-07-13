"""`dgemma/loop.py`'s diffusers version guard + structural probe (issue #35
R3, ARCHITECTURE.md "No diffusers version guard (the transformers guard's
missing twin)" row).

Twins `tests/test_model_load.py`'s `TestTransformersVersionGuard` for the
diffusers side, adapted for a lower-bounded range (`>=0.39.0`) instead of an
exact-pin series: `TestDiffusersVersionGuard` covers the version-floor check
(`_check_diffusers_version`), `TestDiffusersStructuralProbe` covers the
structural probe (`_check_diffusers_structure`) that guards the names/shapes
a version-floor check alone cannot (a newer-than-floor diffusers is
*accepted* by the version check but not thereby guaranteed to keep the
scheduler ctor kwargs, `accepted_index`, or the base pipeline's
`_callback_tensor_inputs` allowlist unchanged), and `TestAnnealFormulaPin`
covers the one drift neither of those can see (PR #48 gate finding F-1): a
rewrite of the anneal formula's *body* that keeps every probed name in
place. The pin drives the REAL installed scheduler's `step()` and asserts
`anneal_temperature`'s re-derivation matches the temperature it actually
applied.

The structural-probe tests monkeypatch the REAL installed diffusers classes
(`EntropyBoundScheduler`, `EntropyBoundSchedulerOutput`,
`DiffusionGemmaPipeline`) to simulate a future version that moved one of the
three probed structures — not a fake/stand-in, so a passing test here means
the probe actually looks at the real objects `dgemma.loop` depends on, not a
shape this test suite invented independently of them.
"""
from __future__ import annotations

import pytest

from dgemma.loop import (
    REQUIRED_DIFFUSERS_MINIMUM,
    _check_diffusers_structure,
    _check_diffusers_version,
    _tuple_version,
    anneal_temperature,
)


class TestDiffusersVersionGuard:
    """issue #35 R3 front-door guard, twin of `test_model_load.py`'s
    `TestTransformersVersionGuard`. Range semantics (`>=0.39.0`), not an
    exact-pin series: anything at or above the floor is accepted; anything
    below is rejected with an actionable message."""

    @pytest.mark.parametrize("version", ["0.39.0", "0.39.1", "0.40.0", "1.0.0"])
    def test_at_or_above_floor_is_accepted(self, version):
        _check_diffusers_version(version)  # must not raise

    def test_prerelease_of_the_floor_itself_is_rejected(self):
        """`0.39.0.dev0` sorts BELOW `0.39.0` per PEP 440 (pre-releases order
        before their final release) — the version-floor check must honor
        that ordering rather than string-prefix-matching "0.39.0" and
        accepting it. Documents the guard's real semantics rather than
        asserting a same-series patch-tolerance the transformers guard has
        but this range-based guard deliberately does not."""
        with pytest.raises(RuntimeError):
            _check_diffusers_version("0.39.0.dev0")

    @pytest.mark.parametrize("version", ["0.38.9", "0.38.0", "0.30.0", "0.9.0"])
    def test_below_floor_raises_actionable_runtime_error(self, version):
        with pytest.raises(RuntimeError) as excinfo:
            _check_diffusers_version(version)

        message = str(excinfo.value)
        assert REQUIRED_DIFFUSERS_MINIMUM in message  # names the required floor
        assert version in message  # names what's actually installed
        assert "pip install" in message  # concrete fix
        assert "diffusers>=" in message
        assert "#35" in message

    def test_message_explains_manager_downgrade_skip_behavior(self):
        """Same Manager-downgrade-skip explanation as the transformers guard
        (issue #25's reasoning applies identically here — see
        `dgemma.model._version_mismatch_message`)."""
        with pytest.raises(RuntimeError) as excinfo:
            _check_diffusers_version("0.30.0")

        assert "downgrade" in str(excinfo.value).lower()

    def test_installed_none_reads_the_real_diffusers_version(self):
        """Default (no `installed` arg) path: reads the real, currently
        importable `diffusers.__version__` — exercised here as a no-op
        because the dev/test environment satisfies the declared floor."""
        _check_diffusers_version()  # must not raise in this repo's own env

    def test_tuple_version_fallback_orders_correctly(self):
        """`_tuple_version` (the no-`packaging` fallback path) must compare
        as an honest numeric tuple, including across differing lengths and a
        non-numeric suffix — the exact shape a `packaging`-free environment
        would fall back to."""
        assert _tuple_version("0.39.0") == (0, 39, 0)
        assert _tuple_version("0.39.0") < _tuple_version("0.40.0")
        assert _tuple_version("0.38.9") < _tuple_version("0.39.0")
        # Named limitation (see _tuple_version's docstring): the fallback
        # drops a non-numeric suffix entirely, so a pre-release of the floor
        # compares EQUAL to the floor here — the opposite of packaging's
        # correct PEP 440 ordering (dev < final). Documented, not silently
        # papered over: this fallback only runs when packaging itself is
        # missing, an already-degraded environment.
        assert _tuple_version("0.39.0.dev0") == (0, 39, 0)
        assert _tuple_version("garbage") == (0,)

    def test_packaging_unavailable_fallback_path_still_gates_correctly(self, monkeypatch):
        """Force the `packaging` import to fail so `_check_diffusers_version`
        exercises its string/tuple fallback branch, not just the `packaging`
        path every other test here implicitly takes."""
        import builtins

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "packaging.version" or name == "packaging":
                raise ImportError("simulated: packaging not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        with pytest.raises(RuntimeError):
            _check_diffusers_version("0.30.0")
        _check_diffusers_version("0.39.0")  # must not raise


class TestDiffusersStructuralProbe:
    """issue #35 R3 structural probe: asserts the vendored surface
    `anneal_temperature`/`_FrameCollector`/`DGemmaPipeline` depend on still
    has the shape this module was written against, independent of whether
    the installed version satisfies the floor check above.

    Baseline: the probe must pass clean against the real installed
    diffusers 0.39.0 (no monkeypatching) — this is the "does not
    false-positive on the actual dependency" half of the gate.
    """

    def test_passes_against_the_real_installed_diffusers(self):
        _check_diffusers_structure()  # must not raise

    def test_missing_scheduler_ctor_kwarg_fails_loud(self, monkeypatch):
        """Simulate a future EntropyBoundScheduler whose __init__ dropped
        one of the kwargs `run_diffusion` constructs it with."""
        import inspect

        from diffusers import EntropyBoundScheduler

        def _mutated_init(self, entropy_bound=0.1, t_max=0.8, num_inference_steps=32):
            # `t_min` dropped — mirrors a real signature mutation, not a
            # hand-rolled fake object standing in for one.
            self.num_inference_steps = num_inference_steps

        monkeypatch.setattr(EntropyBoundScheduler, "__init__", _mutated_init)
        assert "t_min" not in inspect.signature(EntropyBoundScheduler.__init__).parameters

        with pytest.raises(RuntimeError) as excinfo:
            _check_diffusers_structure()

        message = str(excinfo.value)
        assert "t_min" in message
        assert "EntropyBoundScheduler" in message
        assert "#35" in message

    def test_missing_accepted_index_field_fails_loud(self, monkeypatch):
        """Simulate a future EntropyBoundSchedulerOutput that renamed/dropped
        `accepted_index` — the exact field `_FrameCollector.on_step_end`
        reads every step."""
        from diffusers.schedulers.scheduling_entropy_bound import EntropyBoundSchedulerOutput

        mutated_fields = {
            k: v for k, v in EntropyBoundSchedulerOutput.__dataclass_fields__.items() if k != "accepted_index"
        }
        monkeypatch.setattr(EntropyBoundSchedulerOutput, "__dataclass_fields__", mutated_fields)

        with pytest.raises(RuntimeError) as excinfo:
            _check_diffusers_structure()

        message = str(excinfo.value)
        assert "accepted_index" in message
        assert "EntropyBoundSchedulerOutput" in message
        assert "#35" in message

    def test_narrowed_base_callback_tensor_inputs_fails_loud(self, monkeypatch):
        """Simulate a future base DiffusionGemmaPipeline that renamed
        "logits" out of its own _callback_tensor_inputs allowlist —
        DGemmaPipeline widens this exact base list by appending
        "scheduler_output", so a base rename could silently ship a widened
        list built on a stale assumption about what the base already has."""
        from diffusers import DiffusionGemmaPipeline

        monkeypatch.setattr(DiffusionGemmaPipeline, "_callback_tensor_inputs", ["canvas"])

        with pytest.raises(RuntimeError) as excinfo:
            _check_diffusers_structure()

        message = str(excinfo.value)
        assert "logits" in message
        assert "_callback_tensor_inputs" in message
        assert "#35" in message

    def test_extra_base_callback_inputs_do_not_trip_the_probe(self, monkeypatch):
        """A future base pipeline that ADDS to its own allowlist (superset of
        what this module expects) is not a break — the probe checks
        `expected_base_inputs.issubset(...)`, not set equality, since
        DGemmaPipeline only ever widens, never narrows, the base list."""
        from diffusers import DiffusionGemmaPipeline

        monkeypatch.setattr(
            DiffusionGemmaPipeline, "_callback_tensor_inputs", ["canvas", "logits", "some_new_field"]
        )

        _check_diffusers_structure()  # must not raise


class TestAnnealFormulaPin:
    """issue #35 R3 / PR #48 gate finding F-1: the enforcement surface for
    the one drift the structural probe cannot see — a rewrite of the anneal
    formula's *body* (`scheduling_entropy_bound.py:153-154`) that keeps
    every probed name in place.

    `dgemma.loop.anneal_temperature` re-derives that inlined formula because
    the scheduler doesn't expose it. This pin recovers the temperature the
    REAL installed scheduler's `step()` actually applied — `step()` scales
    the raw logits by the annealed temperature exactly once and returns the
    scaled tensor as `pred_logits` (`scheduling_entropy_bound.py:155,181`),
    so `input_logits / pred_logits` is elementwise-constant at exactly that
    temperature — and asserts the re-derivation matches, across first/mid/
    last steps and multiple `(t_min, t_max, num_inference_steps)` configs.

    Non-vacuity, by construction: the expected value comes out of the real
    `EntropyBoundScheduler.step()` call, never from constants this suite
    copied out of the formula (that would just re-assert dgemma against
    itself — the exact vacuity the PR #48 gate reviewed against). A future
    accepted diffusers that changes the anneal math while keeping the names
    fails here loudly instead of `dgemma`'s telemetry silently lying. No
    probed temperature equals 1.0, so a semantics change in `pred_logits`
    (e.g. returning UNscaled logits, making the ratio uniformly 1.0) also
    fails rather than aliasing a pass. Pure CPU tensor math — no weights, no
    CUDA.
    """

    # (num_inference_steps, step_idx, t_min, t_max) spanning the schedule:
    # first step (t=1.0 -> temperature == t_max), exact midpoint, last step,
    # plus off-default configs so the pin constrains the full affine formula
    # (both endpoints and the fraction), not one lucky point. At most ONE
    # point sits at the t=0.5 midpoint: mutation-checking this pin (PR #48
    # F-1 close-out) showed a reflection mutation (`t_max - (t_max-t_min)*t`)
    # coincides with the true formula exactly at t=0.5, so midpoint-heavy
    # cases would alias that whole mutation class to a pass.
    CASES = [
        (48, 0, 0.4, 0.8),
        (48, 24, 0.4, 0.8),
        (48, 47, 0.4, 0.8),
        (32, 11, 0.2, 0.9),
        (10, 9, 0.5, 0.7),
    ]

    @pytest.mark.parametrize("num_inference_steps, step_idx, t_min, t_max", CASES)
    def test_rederivation_matches_real_scheduler_applied_temperature(
        self, num_inference_steps, step_idx, t_min, t_max
    ):
        import torch
        from diffusers import EntropyBoundScheduler

        scheduler = EntropyBoundScheduler(
            entropy_bound=0.1, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
        )

        # Deterministic, strictly positive logits (no zeros: the recovery
        # divides by the returned tensor), small shapes — (batch=1, block=8,
        # vocab=16) is enough to pin a scalar temperature.
        logits = torch.linspace(0.5, 3.5, steps=1 * 8 * 16).reshape(1, 8, 16)
        sample = torch.zeros(1, 8, dtype=torch.long)

        output = scheduler.step(
            logits,
            timestep=step_idx,
            sample=sample,
            generator=torch.Generator().manual_seed(0),
        )

        recovered = logits / output.pred_logits
        # The scheduler applies ONE scalar temperature per step, so the
        # ratio must be elementwise-constant — this also pins `pred_logits`
        # still meaning "temperature-scaled logits" at all.
        assert torch.allclose(recovered, torch.full_like(recovered, recovered.flatten()[0].item()), rtol=1e-5)
        applied_temperature = recovered.mean().item()

        expected_t, expected_temperature = anneal_temperature(
            step_idx, num_inference_steps, t_min, t_max
        )
        assert applied_temperature == pytest.approx(expected_temperature, rel=1e-5), (
            f"anneal_temperature's re-derived temperature {expected_temperature} diverged from the "
            f"temperature the real installed EntropyBoundScheduler.step() actually applied "
            f"({applied_temperature}) at step {step_idx}/{num_inference_steps}, "
            f"t_min={t_min}, t_max={t_max} — the vendored anneal formula has drifted "
            f"(issue #35 R3 residual; update dgemma/loop.py:anneal_temperature and re-verify "
            f"every consumer of DiffusionFrame.t/.temperature)."
        )
        # And the recovered fraction pins `t` itself (temperature is affine
        # in t with distinct endpoints, so this is determined, not extra):
        recovered_t = (applied_temperature - t_min) / (t_max - t_min)
        assert recovered_t == pytest.approx(expected_t, rel=1e-4)
