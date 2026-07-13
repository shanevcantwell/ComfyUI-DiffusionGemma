"""End-to-end `constraints=` tests (ADR-CDG-010, issue #64 Phase 3) — the
two-mechanism givens LIVE through `run_diffusion`, driven against the R4
fake-pipeline fixture (`tests/conftest.py`) so a participant's `{"canvas":
...}` write is actually threaded back into the applied canvas, unlike the
lighter hand-rolled fakes `tests/test_run_diffusion_ingress.py`/
`tests/test_hook_lifecycle.py` use for ingress/lifecycle-only assertions.

This IS the enforcement surface for issue #64 §5's `tests/test_constraints.py`
spec: `TestPinReassertion`, `TestBothMechanisms` (+ the independent-absence
tests ADR-CDG-010 negative-consequence 2 demands), and `TestPinnedMask`.

Per the R4 self-test module's own import-mode caveat
(`tests/test_conftest_fake_pipeline.py`): this module never does a bare
`from tests.conftest import FakeEntropyBoundScheduler` — under
`--import-mode=importlib` with no `tests/__init__.py`, `conftest.py` is not
reliably importable by name from a sibling file. Instead, `_wire_fake_pipeline`
below monkeypatches `dgemma.loop.EntropyBoundScheduler`/`dgemma.loop.
DGemmaPipeline` to the CLASSES of a `fake_pipeline_factory()`-built instance
(`type(built.scheduler)`, `type(built.pipeline)`) — reached only through the
fixture, matching the caveat, while still getting the real R4 canvas-
threading fixture (not a hand-rolled fake that silently drops a participant's
canvas write, the gap `tests/test_run_diffusion_ingress.py`'s own
`_install_fakes` has — see that module's `TestValidPayloadsAreIgnoredBehaviorally`
docstring).
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import run_diffusion
from dgemma.payloads import Constraints, Pin
from dgemma.types import DGemmaModel


class FakeTokenizer:
    """Matches `tests/test_hook_lifecycle.py`'s `FakeTokenizer` shape (the
    minimum `run_diffusion` needs: `decode`, `convert_tokens_to_ids`,
    `eos_token_id`, `unk_token_id`), plus `vocab_size` so
    `dgemma.loop.resolve_vocab_size` resolves a real size instead of
    degrading to `None` — this module's whole point is exercising the C3/
    hook-bounds-check path against a real vocab, not the stub-fallback."""

    eos_token_id = 999
    unk_token_id = 0
    vocab_size = 8  # matches HookRecordingModel's default vocab_size

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)


class FakeProcessor:
    tokenizer = FakeTokenizer()


def _fake_model_with(model) -> DGemmaModel:
    return DGemmaModel(
        model=model, processor=FakeProcessor(), device="cpu", dtype="bfloat16", repo_id="fake/repo", quant="none"
    )


def _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, *, accepted=None, **factory_kwargs):
    """Builds one R4 fixture triple (to get at `HookRecordingModel` and the
    real scheduler/pipeline CLASSES) and monkeypatches `dgemma.loop`'s two
    construction sites so a real `run_diffusion` call drives the real
    canvas-threading fixture end to end.

    **`accepted=` wiring (the part a naive "just patch the class" approach
    gets wrong):** `run_diffusion` constructs its OWN fresh scheduler
    instance every call (`TestSchedulerFreshPerCall`'s own invariant) — so
    patching `dgemma.loop.EntropyBoundScheduler` to `type(built.scheduler)`
    alone would build a scheduler whose `_accepted_source` is the class
    default (`torch.tensor(True)`, all-True), silently ignoring whatever
    `accepted=` a test requested on the throwaway `built.scheduler` instance
    `fake_pipeline_factory` returns (that instance is never the one
    `run_diffusion` actually uses). The factory closure below re-applies
    `accepted` to every scheduler `run_diffusion` constructs, not just the
    fixture's own discarded one.

    Returns the built triple so a test can also inspect `built.model`
    (hook install/removal logs) afterward.
    """
    built = fake_pipeline_factory(**factory_kwargs)
    scheduler_cls = type(built.scheduler)
    pipeline_cls = type(built.pipeline)

    def _scheduler_factory(**kwargs):
        scheduler = scheduler_cls(**kwargs)
        if accepted is not None:
            scheduler._accepted_source = accepted
        return scheduler

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", _scheduler_factory)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", pipeline_cls)
    return built


class TestPinReassertion:
    """ADR-CDG-010 Decision 1(b)/Decision 3: a pinned position survives to
    the final canvas regardless of what the scheduler itself accepted —
    the canvas re-assertion mechanism, proven through `run_diffusion`."""

    def test_pinned_position_holds_token_id_in_final_canvas(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(
            monkeypatch,
            fake_pipeline_factory,
            num_inference_steps=3,
            canvas_shape=(1, 4),
            # Nothing is ever accepted by the scheduler — every step's
            # `prev_sample` would revert to zeros were the pin participant
            # not the LAST writer re-asserting token_id regardless.
            accepted=torch.zeros((1, 4), dtype=torch.bool),
        )
        constraints = Constraints(pins=(Pin(position=2, token_id=5),))

        text, state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            gen_length=4,
            constraints=constraints,
        )

        assert state.canvas_ids[2] == 5
        # Captured frames show the PRE-pin canvas (capture runs before pin in
        # the fixed order, ADR-CDG-010 Decision 3) — with acceptance always
        # False, the scheduler zeroes position 2 every step BEFORE pin
        # re-asserts it, so every captured frame reads 0 there, never 5.
        # This is the same ordering `test_step_end_composite.py`'s
        # `test_capture_sees_pre_writer_canvas_never_post_pin_state` pins;
        # confirming it here (rather than asserting the frame already shows
        # 5) is itself part of proving the two are consistent under a real
        # `run_diffusion` call, not just the composite in isolation.
        assert all(int(f.canvas[..., 2]) == 0 for f in trace.frames)

    def test_pin_survives_when_scheduler_accepts_everything_too(self, monkeypatch, fake_pipeline_factory):
        """The given holds even when the scheduler's own acceptance would
        have agreed anyway — pin is unconditional, not a fallback only
        exercised on rejection."""
        built = _wire_fake_pipeline(
            monkeypatch, fake_pipeline_factory, num_inference_steps=2, canvas_shape=(1, 4)
        )
        constraints = Constraints(pins=(Pin(position=0, token_id=3),))

        _, state, _ = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            gen_length=4,
            constraints=constraints,
        )
        assert state.canvas_ids[0] == 3


class TestBothMechanisms:
    """ADR-CDG-010 clause-1 enforcement row: one constraint installs BOTH a
    logit mask AND a canvas re-assertion. Plus the independent-absence
    tests negative-consequence 2 demands: mask-only (no re-assertion) lets
    the cell drift; re-assertion-only (no mask) writes but leaves logits
    untouched — each exercised so a regression in either half is caught."""

    def test_constraints_installs_both_mask_and_reassertion(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(monkeypatch, fake_pipeline_factory, num_inference_steps=2, vocab_size=8)
        constraints = Constraints(pins=(Pin(position=1, token_id=4),))

        _, state, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=2,
            gen_length=4,
            constraints=constraints,
        )

        # (a) the canvas re-assertion mechanism actually wrote the pin.
        assert state.canvas_ids[1] == 4
        # (b) the hook actually masked the pinned position's logits on at
        # least one forward pass — HookRecordingModel's forward is called
        # from inside FakeDiffusionGemmaPipeline.__call__, so the model's
        # live_hook_count during the run (asserted indirectly: the hook was
        # installed then torn down) plus the logits mutation are both
        # observable via a direct call through the model post-hoc using the
        # SAME hook-building path — checked directly in
        # tests/test_constraints_hook.py:TestBuildLogitMaskHook. Here we
        # confirm the hook fired at all (zero hooks after run, non-zero
        # install log) as the run_diffusion-level half of the "both
        # mechanisms installed" claim.
        assert built.model.install_log == [0]
        assert built.model.live_hook_count == 0

    def test_mask_only_no_reassertion_lets_cell_drift(self, monkeypatch, fake_pipeline_factory):
        """Independent-absence test (negative-consequence 2): install ONLY
        the logit-mask hook directly (bypassing the pin participant) and
        confirm the canvas is free to diverge from the masked id when the
        scheduler doesn't happen to sample it — proving the mask alone does
        not guarantee the final canvas without the pin participant's
        re-assertion."""
        from dgemma.constraints_hook import build_logit_mask_hook
        from dgemma.hooks import install_logit_shaping_hook

        built = fake_pipeline_factory(num_inference_steps=1, vocab_size=8, canvas_shape=(1, 4))
        pins = (Pin(position=0, token_id=5),)
        hook_fn = build_logit_mask_hook(pins, vocab_size=8)

        with install_logit_shaping_hook(built.model, hook_fn):
            result = built.pipeline(
                num_inference_steps=1,
                callback_on_step_end=lambda *a: {},  # no pin participant wired
                callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
            )
        # The fixture's FakeEntropyBoundScheduler.step() ignores model_output
        # entirely (it scripts acceptance via `accepted`, not real sampling)
        # — so masking the logits alone has NO observable effect on the
        # canvas the scheduler produces without a pin participant to also
        # re-assert it. The canvas is whatever the (all-True-by-default)
        # acceptance produced from the ORIGINAL zero-initialized canvas, not
        # necessarily `token_id`.
        assert int(result.sequences[..., 0]) != 5

    def test_reassertion_only_no_mask_writes_canvas_but_leaves_logits_untouched(
        self, monkeypatch, fake_pipeline_factory
    ):
        """Independent-absence test (negative-consequence 2), the other
        half: install ONLY the pin participant (no logit-mask hook) and
        confirm the canvas write still happens, but the model's raw logits
        output is never mutated — proving the pin participant alone gives
        no commit-ordering advantage, only the re-assertion write."""
        from dgemma.participants import PinParticipant

        built = fake_pipeline_factory(num_inference_steps=1, vocab_size=8, canvas_shape=(1, 4))
        constraints = Constraints(pins=(Pin(position=0, token_id=5),))
        pin = PinParticipant(constraints=constraints)

        seen_logits: list = []
        real_forward = built.model.forward

        def _spying_forward(*args, **kwargs):
            output = real_forward(*args, **kwargs)
            seen_logits.append(output.logits.clone())
            return output

        built.model.forward = _spying_forward

        result = built.pipeline(
            num_inference_steps=1,
            callback_on_step_end=lambda pipe, gs, si, kw: pin(pipe, gs, si, kw),
            callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
        )

        assert int(result.sequences[..., 0]) == 5
        # Untouched: HookRecordingModel.forward always returns all-zero
        # logits (tests/conftest.py) — with no hook installed, nothing masks
        # position 0 down to -inf/token_id, so it stays exactly zero.
        assert torch.all(seen_logits[0][..., 0, :] == 0.0)


class TestPinnedMask:
    """ADR-CDG-010 Decision 4 trace-honesty test: a pinned cell's
    `frame.pinned_mask[position]` is True on every frame regardless of the
    scheduler's own commit reading — now backed by `PinParticipant`
    actually (re-)writing that position every step (Phase 3), not merely
    the Phase 2 static derivation."""

    def test_pinned_mask_true_every_frame_regardless_of_acceptance(self, monkeypatch, fake_pipeline_factory):
        built = _wire_fake_pipeline(
            monkeypatch,
            fake_pipeline_factory,
            num_inference_steps=3,
            canvas_shape=(1, 4),
            accepted=torch.zeros((1, 4), dtype=torch.bool),
        )
        constraints = Constraints(pins=(Pin(position=1, token_id=2), Pin(position=3, token_id=6)))

        _, _, trace = run_diffusion(
            _fake_model_with(built.model),
            "hi",
            entropy_bound=0.1,
            t_min=0.4,
            t_max=0.8,
            num_inference_steps=3,
            gen_length=4,
            constraints=constraints,
        )

        for frame in trace.frames:
            mask = frame.pinned_mask.tolist()
            assert mask == [False, True, False, True]
