"""Unit-level tests for `dgemma.participants.BetaRebuildParticipant`
(ADR-CDG-010, issue #64 Phase 5) — the `StepEndComposite`'s `beta_rebuild`
slot in isolation, without driving a full `run_diffusion` call.

Mirrors `tests/test_participants.py`'s isolated-unit convention for
`WalkerParticipant`: this class has no `run_diffusion`-level end-to-end test
module of its own, because — unlike `PinParticipant`/`WalkerParticipant` —
`run_diffusion` builds no `BetaRebuildParticipant` this phase (no ingress
payload exists to build one from; ADR-CDG-010 Open Question 2 is unresolved
and the 2026-07-13 gate ruling O3 defers the beta-viscosity math and its wire
shape rather than guessing one — see `dgemma/participants.py`'s module
docstring). Composite-level ordering (beta-rebuild before pin, using this
REAL participant rather than the generic `_RecordingParticipant` test double)
is covered separately in `tests/test_step_end_composite.py::
TestBetaRebuildBeforePinRealParticipants`.
"""
from __future__ import annotations

import torch

from dgemma.participants import BetaRebuildParticipant, RebuildWrite


def _canvas(*values: int) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.long)


class TestCanvasWrite:
    """The core mechanism: every `RebuildWrite` rewrites its `token_id` at
    its `position` in the canvas, returned as `{"canvas": ...}` — the same
    canvas-writer contract `PinParticipant` implements for its own slot."""

    def test_writes_token_id_at_position(self):
        participant = BetaRebuildParticipant(writes=(RebuildWrite(position=1, token_id=99),))
        callback_kwargs = {"canvas": _canvas(5, 5, 5, 5)}

        result = participant(pipe=None, global_step=0, step_idx=0, callback_kwargs=callback_kwargs)

        assert result is not None
        assert torch.equal(result["canvas"], _canvas(5, 99, 5, 5))

    def test_multiple_writes_land_at_their_own_positions(self):
        participant = BetaRebuildParticipant(
            writes=(RebuildWrite(position=0, token_id=1), RebuildWrite(position=3, token_id=2))
        )
        callback_kwargs = {"canvas": _canvas(0, 0, 0, 0)}

        result = participant(pipe=None, global_step=0, step_idx=0, callback_kwargs=callback_kwargs)

        assert torch.equal(result["canvas"], _canvas(1, 0, 0, 2))

    def test_does_not_mutate_the_input_canvas_tensor(self):
        """The participant clones before writing (same discipline as
        `PinParticipant.__call__`) — the caller's `callback_kwargs["canvas"]`
        tensor is left untouched, only the returned tensor is rewritten."""
        original = _canvas(7, 7, 7)
        participant = BetaRebuildParticipant(writes=(RebuildWrite(position=0, token_id=1),))

        result = participant(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": original})

        assert torch.equal(original, _canvas(7, 7, 7))
        assert torch.equal(result["canvas"], _canvas(1, 7, 7))

    def test_batched_canvas_shape_writes_across_the_batch_dim(self):
        """Same ellipsis-index shape assumption `PinParticipant` documents:
        `canvas[..., position] = token_id` covers both `[canvas_len]` and
        `[batch, canvas_len]` with no batch-dim branching."""
        participant = BetaRebuildParticipant(writes=(RebuildWrite(position=2, token_id=42),))
        canvas = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)

        result = participant(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": canvas})

        assert torch.equal(result["canvas"], torch.tensor([[0, 0, 42], [1, 1, 42]], dtype=torch.long))


class TestEmptyWritesIsANoOp:
    """`writes=()` (the dataclass default) is a legal, inert no-op — the
    canvas passes through unchanged, still returned as `{"canvas": ...}`
    (a canvas-writer always returns its slot's contract, even when it wrote
    nothing this step)."""

    def test_default_construction_writes_nothing(self):
        participant = BetaRebuildParticipant()
        callback_kwargs = {"canvas": _canvas(3, 3, 3)}

        result = participant(pipe=None, global_step=0, step_idx=0, callback_kwargs=callback_kwargs)

        assert torch.equal(result["canvas"], _canvas(3, 3, 3))


class TestStatelessConstruction:
    """ADR-CDG-010 Decision 7: the participant holds only the immutable
    `writes` tuple from its own construction — no cross-call state. Calling
    the same instance at different `step_idx` values writes identically,
    proving the write is purely a function of `writes`, never an internal
    cursor keyed on step."""

    def test_repeated_calls_at_different_step_idx_write_identically(self):
        participant = BetaRebuildParticipant(writes=(RebuildWrite(position=0, token_id=9),))

        first = participant(pipe=None, global_step=0, step_idx=0, callback_kwargs={"canvas": _canvas(0, 0)})
        second = participant(pipe=None, global_step=5, step_idx=5, callback_kwargs={"canvas": _canvas(0, 0)})

        assert torch.equal(first["canvas"], second["canvas"])


class TestParticipantNameField:
    def test_name_is_beta_rebuild(self):
        assert BetaRebuildParticipant().name == "beta_rebuild"


class TestRebuildWriteIsFrozen:
    """`RebuildWrite` is a frozen dataclass (matches `Pin`'s immutability
    discipline, `dgemma/payloads.py`) — a caller cannot mutate a write after
    construction."""

    def test_position_and_token_id_are_not_settable(self):
        write = RebuildWrite(position=0, token_id=1)
        try:
            write.position = 2
        except Exception:
            pass
        else:
            raise AssertionError("RebuildWrite must be frozen")
        assert write.position == 0
