"""Unit-level tests for `dgemma.participants.WalkerParticipant` (ADR-CDG-011,
issue #64 Phase 4) — the `StepEndComposite`'s `walker` slot in isolation,
without driving a full `run_diffusion` call.

`dgemma.participants.PinParticipant` (issue #64 Phase 3) has no dedicated
unit-only test module of its own — its behavior is covered end-to-end via
`tests/test_constraints.py` and cross-call statelessness via
`tests/test_run_diffusion_statelessness.py::TestPinStatePerRun`. This module
gives `WalkerParticipant` the equivalent isolated-unit coverage that
`tests/test_control_signals.py` (end-to-end) and
`tests/test_run_diffusion_statelessness.py::TestWalkerStatePerRun`
(cross-call) don't reach on their own: the participant's `__call__` contract
directly, against a minimal hand-rolled scheduler double (this module's own
`_RecordingScheduler`, not `tests/conftest.py`'s `FakeEntropyBoundScheduler`
— consistent with `tests/test_run_diffusion_statelessness.py`'s own
`_RecordingFrozenConfig`/inline-fake convention for a unit-only module that
doesn't need the full R4 pipeline fixture).
"""
from __future__ import annotations

import pytest

from dgemma.participants import WalkerParticipant
from dgemma.payloads import Binding, ControlSignals


class _RecordingFrozenConfig:
    """Minimal `FrozenDict`-alike: direct attribute-set raises, the only
    mutation path is `register_to_config`, which rebuilds the dict wholesale
    — mirrors `tests/conftest.py:FakeFrozenConfig` /
    `tests/test_run_diffusion_statelessness.py:_RecordingFrozenConfig`."""

    def __init__(self, **kwargs):
        object.__setattr__(self, "_values", dict(kwargs))

    def __getattr__(self, name):
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(name)

    def __setattr__(self, name, value):
        raise AttributeError(f"frozen — use register_to_config, not direct set of {name!r}")


class _RecordingScheduler:
    """Records every `register_to_config` call (as a dict of kwargs, in call
    order) AND maintains a genuine frozen-config mutation path, so a test can
    assert both "what was written" and "how many separate writes happened"
    (the single-fold-in-one-call invariant, ADR-CDG-011 Decision 4)."""

    def __init__(self, **kwargs):
        self._config = _RecordingFrozenConfig(**kwargs)
        self.register_calls: list[dict] = []

    @property
    def config(self):
        return self._config

    def register_to_config(self, **kwargs):
        self.register_calls.append(dict(kwargs))
        merged = dict(object.__getattribute__(self._config, "_values"))
        merged.update(kwargs)
        self._config = _RecordingFrozenConfig(**merged)


def _walker(bindings: tuple[Binding, ...], scheduler: _RecordingScheduler) -> WalkerParticipant:
    return WalkerParticipant(control_signals=ControlSignals(bindings=bindings), scheduler=scheduler)


class TestReturnValueAlwaysNone:
    """The walker is a config-mutator, not a canvas-writer — its `__call__`
    must always return `None`, regardless of whether it wrote anything."""

    def test_returns_none_when_it_writes(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker((Binding(target="entropy_bound", signal=(0.0, 1.0), low=0.0, high=1.0),), scheduler)

        result = walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})

        assert result is None
        assert scheduler.register_calls == [{"entropy_bound": 1.0}]

    def test_returns_none_at_the_final_step_no_op(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker((Binding(target="entropy_bound", signal=(0.0, 1.0), low=0.0, high=1.0),), scheduler)

        result = walker(pipe=None, global_step=1, step_idx=1, callback_kwargs={"canvas": None})

        assert result is None
        assert scheduler.register_calls == []  # no write — signal[2] is out of bounds


class TestStepIndexing:
    """The walker writes `signal[step_idx + 1]` — "prepares the next step"
    (ADR-CDG-011 clause 6 / gate ruling O1)."""

    def test_step_0_callback_writes_signal_index_1(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker(
            (Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0), low=0.0, high=1.0),), scheduler
        )

        walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})

        assert scheduler.register_calls == [{"entropy_bound": 0.5}]

    def test_step_1_callback_writes_signal_index_2(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker(
            (Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0), low=0.0, high=1.0),), scheduler
        )

        walker(pipe=None, global_step=1, step_idx=1, callback_kwargs={"canvas": None})

        assert scheduler.register_calls == [{"entropy_bound": 1.0}]

    def test_last_valid_step_writes_nothing_out_of_bounds_signal_index(self):
        """`signal` has length 3 (indices 0,1,2); at `step_idx=2` the walker
        would need `signal[3]`, which doesn't exist — must be silently
        skipped for THAT binding, not raise `IndexError`."""
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker(
            (Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0), low=0.0, high=1.0),), scheduler
        )

        result = walker(pipe=None, global_step=2, step_idx=2, callback_kwargs={"canvas": None})

        assert result is None
        assert scheduler.register_calls == []


class TestRangeMapping:
    """ADR-CDG-011 Decision 4: `value = low + (high - low) * signal[k]`."""

    @pytest.mark.parametrize(
        "raw, low, high, expected",
        [
            (0.0, 0.02, 0.3, 0.02),
            (1.0, 0.02, 0.3, 0.3),
            (0.5, 0.0, 1.0, 0.5),
            (0.25, 0.02, 0.3, 0.02 + 0.28 * 0.25),
        ],
    )
    def test_mapping_formula(self, raw, low, high, expected):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker(
            (Binding(target="entropy_bound", signal=(0.0, raw), low=low, high=high),), scheduler
        )

        walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})

        assert scheduler.register_calls[0]["entropy_bound"] == pytest.approx(expected)


class TestMultipleBindingsFoldIntoOneCall:
    """Decision 4's fold-in: bindings governing the SAME step land in ONE
    `register_to_config` call, never two separate calls that could clobber
    each other's whole-dict rebuild."""

    def test_two_bindings_for_the_same_step_produce_exactly_one_register_call(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1, t_min=0.4, t_max=0.8)
        walker = _walker(
            (
                Binding(target="entropy_bound", signal=(0.0, 0.9), low=0.0, high=1.0),
                Binding(target="t_min", signal=(0.0, 0.2), low=0.0, high=1.0),
            ),
            scheduler,
        )

        walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})

        assert len(scheduler.register_calls) == 1
        assert scheduler.register_calls[0] == {"entropy_bound": 0.9, "t_min": 0.2}


class TestEmptyBindingsIsANoOp:
    """`ControlSignals(bindings=())` must never call `register_to_config` —
    "empty == no-op" (`dgemma/payloads.py`)."""

    def test_no_bindings_never_writes(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker((), scheduler)

        result = walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})

        assert result is None
        assert scheduler.register_calls == []


class TestStatelessConstruction:
    """ADR-CDG-011 clause 8 / F5: the participant holds only the immutable
    `control_signals` payload and the `scheduler` reference from its own
    construction — no mutable internal state (e.g. an internal step counter)
    that could carry over between calls if a caller reused one instance
    (which `run_diffusion` never does — a fresh instance every call, see
    `dgemma/loop.py` and `tests/test_run_diffusion_statelessness.py::
    TestWalkerStatePerRun`)."""

    def test_repeated_calls_at_the_same_step_idx_are_idempotent(self):
        """No internal counter — calling the SAME step_idx twice writes the
        SAME value both times (proves indexing is purely a function of the
        passed `step_idx`, not an incrementing internal cursor)."""
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker(
            (Binding(target="entropy_bound", signal=(0.0, 0.7), low=0.0, high=1.0),), scheduler
        )

        walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})
        walker(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": None})

        assert scheduler.register_calls == [{"entropy_bound": 0.7}, {"entropy_bound": 0.7}]


class TestParticipantNameField:
    def test_name_is_walker(self):
        scheduler = _RecordingScheduler(entropy_bound=0.1)
        walker = _walker((), scheduler)
        assert walker.name == "walker"
