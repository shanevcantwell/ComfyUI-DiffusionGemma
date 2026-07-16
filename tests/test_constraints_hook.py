"""Unit tests for `dgemma.constraints_hook.build_logit_mask_hook` (ADR-CDG-010
Decision 1(a)/Decision 5, issue #64 §5 `tests/test_constraints_hook.py`).

Two classes: `TestBuildLogitMaskHook` (the mask math, against
`HookRecordingModel`/`install_logit_shaping_hook` — the same fixtures
`tests/test_hook_lifecycle.py` uses) and `TestSoleInstallPath` (the R5 grep
discipline, extended per issue #64 §4: this module must never call
`register_forward_hook` itself — it only builds a closure that
`run_diffusion` installs through the one sanctioned path).
"""
from __future__ import annotations

import inspect

import torch

from dgemma.constraints_hook import build_logit_mask_hook
from dgemma.hooks import install_logit_shaping_hook
from dgemma.payloads import Pin


class TestBuildLogitMaskHook:
    """The mask math (issue #64 §4): each pin's position is masked to
    `-inf` everywhere except `token_id`, which reads `0`; non-pinned
    positions are left untouched."""

    def test_pinned_position_masked_to_single_token_id(self, fake_pipeline_factory):
        built = fake_pipeline_factory(vocab_size=8)
        hook_fn = build_logit_mask_hook((Pin(position=1, token_id=3),), vocab_size=8)

        with install_logit_shaping_hook(built.model, hook_fn):
            output = built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))

        pinned_row = output.logits[0, 1, :]
        assert pinned_row[3] == 0.0
        assert torch.all(pinned_row[:3] == float("-inf"))
        assert torch.all(pinned_row[4:] == float("-inf"))

    def test_non_pinned_positions_untouched(self, fake_pipeline_factory):
        built = fake_pipeline_factory(vocab_size=8)
        hook_fn = build_logit_mask_hook((Pin(position=1, token_id=3),), vocab_size=8)

        with install_logit_shaping_hook(built.model, hook_fn):
            output = built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))

        # HookRecordingModel.forward always returns all-zero logits
        # (tests/conftest.py) — positions 0, 2, 3 must stay exactly zero,
        # never touched by the mask meant only for position 1.
        for position in (0, 2, 3):
            assert torch.all(output.logits[0, position, :] == 0.0)

    def test_multiple_pins_each_masked_independently(self, fake_pipeline_factory):
        built = fake_pipeline_factory(vocab_size=8)
        pins = (Pin(position=0, token_id=1), Pin(position=3, token_id=6))
        hook_fn = build_logit_mask_hook(pins, vocab_size=8)

        with install_logit_shaping_hook(built.model, hook_fn):
            output = built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))

        assert output.logits[0, 0, 1] == 0.0
        assert torch.all(output.logits[0, 0, torch.tensor([0, 2, 3, 4, 5, 6, 7])] == float("-inf"))
        assert output.logits[0, 3, 6] == 0.0
        assert torch.all(output.logits[0, 3, torch.tensor([0, 1, 2, 3, 4, 5, 7])] == float("-inf"))
        # Position 1/2 (no pin) stay untouched.
        assert torch.all(output.logits[0, 1, :] == 0.0)
        assert torch.all(output.logits[0, 2, :] == 0.0)

    def test_empty_pins_is_a_no_op(self, fake_pipeline_factory):
        """`build_logit_mask_hook(())` mirrors `Constraints()`'s "empty ==
        no-op" contract — a direct unit-test path only; `run_diffusion`
        never builds this hook at all for an empty/`None` `constraints=`
        (see `dgemma/loop.py`)."""
        built = fake_pipeline_factory(vocab_size=8)
        hook_fn = build_logit_mask_hook((), vocab_size=8)

        with install_logit_shaping_hook(built.model, hook_fn):
            output = built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))

        assert torch.all(output.logits == 0.0)

    def test_vocab_size_none_skips_the_bounds_check(self, fake_pipeline_factory):
        """Mirrors `resolve_vocab_size`'s own degradation: a caller with no
        resolvable vocab size still gets a working hook, just without the
        defensive out-of-vocab re-check."""
        built = fake_pipeline_factory(vocab_size=8)
        hook_fn = build_logit_mask_hook((Pin(position=0, token_id=3),), vocab_size=None)

        with install_logit_shaping_hook(built.model, hook_fn):
            output = built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))

        assert output.logits[0, 0, 3] == 0.0

    def test_out_of_vocab_token_id_raises_when_vocab_size_given(self, fake_pipeline_factory):
        built = fake_pipeline_factory(vocab_size=8)
        hook_fn = build_logit_mask_hook((Pin(position=0, token_id=99),), vocab_size=8)

        try:
            with install_logit_shaping_hook(built.model, hook_fn):
                built.model(decoder_input_ids=torch.zeros((1, 4), dtype=torch.long))
        except ValueError as exc:
            assert "out of vocabulary" in str(exc)
        else:
            raise AssertionError("expected ValueError for out-of-vocab token_id")


class TestSoleInstallPath:
    """Extends the R5 grep discipline (`dgemma/hooks.py`'s own docstring,
    ADR-CDG-010 Decision 5): `dgemma/constraints_hook.py` must never call
    `register_forward_hook` — it only builds a closure that `run_diffusion`
    installs through `dgemma.hooks.install_logit_shaping_hook`, the engine's
    one sanctioned installation site. A second install site anywhere would
    duplicate the exact failure mode (#28) Decision 5 closes."""

    def test_module_source_contains_no_register_forward_hook_call(self):
        """Checks for an actual CALL (`.register_forward_hook(` / bare
        `register_forward_hook(`), not the bare identifier — the module's
        own docstring legitimately discusses the invariant by name (as
        `dgemma/hooks.py`'s docstring does for itself), which a naive
        substring check on the identifier alone would misfire on."""
        import dgemma.constraints_hook as constraints_hook_module

        source = inspect.getsource(constraints_hook_module)
        assert "register_forward_hook(" not in source
