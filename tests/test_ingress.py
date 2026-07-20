"""Ingress validator tests (ADR-CDG-010/011/014, issue #64 §1.4/§1.5/§5;
issue #61 P-B for `capture.top_k`).

One test class per validator. Every reject path asserts BOTH the precondition
AND the remedy token from the locked error register (issue #64 §1.5) via
`pytest.raises(..., match=...)` — a message that states the precondition but
drops the fix (or vice versa) is a partial register entry, so both halves are
independently checked line-by-line, not just "it raised."

Pure unit tests: no fake pipeline, no `run_diffusion` call — these exercise
`dgemma.ingress`'s functions directly against `dgemma.payloads` dataclasses,
with ONE deliberate exception: `TestExactTemperatureRunLevelIngress` below
(issue #110's pre-registration) imports `dgemma.loop.run_diffusion` because
that is where the `t_min`/`t_max` run-level guard this issue names actually
lives today, not in `dgemma.ingress` itself — no fixture/fake-pipeline setup
is needed since the guard fires before any pipeline construction (see that
test's own docstring).
"""
from __future__ import annotations

import pytest

from dgemma.ingress import (
    reject_conflicting_hook_sources,
    validate_capture,
    validate_constraints,
    validate_control_signals,
    validate_ingress,
)
from dgemma.payloads import Binding, CaptureSpec, Constraints, ControlSignals, MUTABLE_TARGETS, Pin


class TestConstraintsIngress:
    """C1-C4 (issue #64 §1.4)."""

    def test_none_is_a_no_op(self):
        validate_constraints(None, gen_length=10, vocab_size=100)  # must not raise

    def test_empty_pins_is_a_no_op(self):
        validate_constraints(Constraints(pins=()), gen_length=10, vocab_size=100)  # must not raise

    def test_valid_pins_pass(self):
        constraints = Constraints(pins=(Pin(position=0, token_id=5), Pin(position=3, token_id=9)))
        validate_constraints(constraints, gen_length=10, vocab_size=100)  # must not raise

    def test_c1_callable_rejected(self):
        def not_a_payload():
            return None

        with pytest.raises(ValueError, match="must be a Constraints payload, not a callable"):
            validate_constraints(not_a_payload, gen_length=10, vocab_size=100)
        with pytest.raises(ValueError, match=r"Fix: pass Constraints\(pins="):
            validate_constraints(not_a_payload, gen_length=10, vocab_size=100)

    def test_c2_position_out_of_range_too_high(self):
        constraints = Constraints(pins=(Pin(position=10, token_id=5),))
        with pytest.raises(ValueError, match=r"Pin position 10 out of range for gen_length=10"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)
        with pytest.raises(ValueError, match="Fix: pin a position inside the generated canvas"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)

    def test_c2_position_out_of_range_negative(self):
        constraints = Constraints(pins=(Pin(position=-1, token_id=5),))
        with pytest.raises(ValueError, match="out of range for gen_length=10"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)

    def test_c3_token_out_of_vocab(self):
        constraints = Constraints(pins=(Pin(position=0, token_id=200),))
        with pytest.raises(ValueError, match=r"Pin token_id 200 is out of vocabulary \(vocab_size=100"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)
        with pytest.raises(ValueError, match="Fix: pin an in-vocab token id"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)

    def test_c3_skipped_with_named_degradation_when_vocab_size_unavailable(self):
        """A bare test stub with no resolvable vocab (`vocab_size=None`,
        issue #64 §3.4's named degradation) must not raise C3 — position/
        duplicate checks still run."""
        constraints = Constraints(pins=(Pin(position=0, token_id=99_999_999),))
        validate_constraints(constraints, gen_length=10, vocab_size=None)  # must not raise

    def test_c4_duplicate_position_rejected(self):
        constraints = Constraints(pins=(Pin(position=2, token_id=1), Pin(position=2, token_id=2)))
        with pytest.raises(ValueError, match="Duplicate Pin for position 2"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)
        with pytest.raises(ValueError, match="Fix: supply one Pin per position"):
            validate_constraints(constraints, gen_length=10, vocab_size=100)


class TestControlSignalsIngress:
    """V1-V6 (issue #64 §1.4)."""

    def test_none_is_a_no_op(self):
        validate_control_signals(None, num_inference_steps=4)  # must not raise

    def test_empty_bindings_is_a_no_op(self):
        validate_control_signals(ControlSignals(bindings=()), num_inference_steps=4)  # must not raise

    def test_valid_binding_passes(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0, 0.5), low=0.02, high=0.3),)
        )
        validate_control_signals(control_signals, num_inference_steps=4)  # must not raise

    def test_v1_callable_rejected(self):
        def not_a_payload():
            return None

        with pytest.raises(ValueError, match="must be a ControlSignals payload, not a callable"):
            validate_control_signals(not_a_payload, num_inference_steps=4)
        with pytest.raises(ValueError, match=r"Fix: pass ControlSignals\(bindings="):
            validate_control_signals(not_a_payload, num_inference_steps=4)

    def test_v2_signal_shorter_than_steps_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5), low=0.02, high=0.3),)
        )
        with pytest.raises(ValueError, match=r"signal length 2, but num_inference_steps=4"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="never truncated or padded"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="Fix: generate a signal of exactly 4 samples"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v2_signal_longer_than_steps_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.2, 0.4, 0.6, 0.8), low=0.02, high=0.3),)
        )
        with pytest.raises(ValueError, match=r"signal length 5, but num_inference_steps=4"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v3_unknown_target_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="not_a_real_knob", signal=(0.0, 0.5, 1.0, 0.5), low=0.02, high=0.3),)
        )
        with pytest.raises(ValueError, match="is not a mutable scheduler knob"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="Fix: bind one of the fresh-read config knobs"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v4_num_inference_steps_target_rejected_with_20_anchored_message(self):
        """The #20 regression anchor: a distinct, named reject — not just
        the generic V3 unknown-target message — even though V3 would also
        exclude this target (issue #64 §1.4)."""
        control_signals = ControlSignals(
            bindings=(Binding(target="num_inference_steps", signal=(0.0, 0.5, 1.0, 0.5), low=1.0, high=100.0),)
        )
        with pytest.raises(ValueError, match="'num_inference_steps' is rejected"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="non-mutable mid-run by design"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="issue #20's mechanism"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="Fix: to change effective step count, set num_inference_steps="):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v5_degenerate_range_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0, 0.5), low=0.1, high=0.1),)
        )
        with pytest.raises(ValueError, match=r"degenerate range low=0\.1 == high=0\.1"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="bind t_min and t_max to the same signal instead"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v5_non_finite_range_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0, 0.5), low=float("nan"), high=0.3),)
        )
        with pytest.raises(ValueError, match="degenerate range"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v6_signal_value_above_one_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 1.5, 1.0, 0.5), low=0.02, high=0.3),)
        )
        with pytest.raises(ValueError, match=r"signal value 1\.5 at step 1 outside the unitless range"):
            validate_control_signals(control_signals, num_inference_steps=4)
        with pytest.raises(ValueError, match="Fix: normalize the generator's output to \\[0,1\\]"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v6_signal_value_below_zero_rejected(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(-0.1, 0.5, 1.0, 0.5), low=0.02, high=0.3),)
        )
        with pytest.raises(ValueError, match=r"signal value -0\.1 at step 0 outside the unitless range"):
            validate_control_signals(control_signals, num_inference_steps=4)

    def test_v6_boundary_values_zero_and_one_are_accepted(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="t_min", signal=(0.0, 1.0, 0.0, 1.0), low=0.1, high=0.9),)
        )
        validate_control_signals(control_signals, num_inference_steps=4)  # must not raise

    def test_mutable_targets_registry_matches_documented_set(self):
        """Pins MUTABLE_TARGETS itself against the ADR-CDG-011 clause-4
        vocabulary, so a silent addition/removal in payloads.py is caught
        here too, not just by V3's behavior."""
        assert MUTABLE_TARGETS == frozenset({"entropy_bound", "t_min", "t_max"})
        assert "num_inference_steps" not in MUTABLE_TARGETS


class TestCaptureIngress:
    """P1 `keep_frames` (issue #64 §1.4) + P-B `top_k` (ADR-CDG-014 Decision
    3 Tier 1, issue #61) + P-C `capture_full_distribution`/
    `max_full_distribution_steps` (ADR-CDG-014 Decision 3/5 Tier 2, issue
    #61 P-C)."""

    class _FakeCaptureSpec:
        def __init__(
            self,
            keep_frames="all",
            top_k=0,
            capture_full_distribution=False,
            max_full_distribution_steps=None,
        ):
            self.keep_frames = keep_frames
            self.top_k = top_k
            self.capture_full_distribution = capture_full_distribution
            self.max_full_distribution_steps = max_full_distribution_steps

    def test_none_is_a_no_op(self):
        validate_capture(None)  # must not raise

    def test_valid_keep_frames_last_passes(self):
        validate_capture(self._FakeCaptureSpec("last"))  # must not raise

    def test_valid_keep_frames_all_passes(self):
        validate_capture(self._FakeCaptureSpec("all"))  # must not raise

    def test_p1_invalid_keep_frames_rejected(self):
        capture = self._FakeCaptureSpec("everything")
        with pytest.raises(ValueError, match=r"keep_frames must be 'last' or 'all', got 'everything'"):
            validate_capture(capture)
        with pytest.raises(ValueError, match="Fix: use one of the two retention policies"):
            validate_capture(capture)

    def test_real_capture_spec_default_is_a_no_op(self):
        """The minted `CaptureSpec` dataclass itself (ADR-CDG-014 Decision
        7) — not just a duck-typed stand-in — passes with its defaults."""
        validate_capture(CaptureSpec())  # must not raise

    def test_top_k_zero_is_a_no_op(self):
        validate_capture(self._FakeCaptureSpec(top_k=0))  # must not raise

    def test_top_k_absent_defaults_to_zero_no_op(self):
        """Duck-typed `getattr(capture, "top_k", 0)` — an object with no
        `top_k` attribute at all (e.g. a pre-P-B stand-in) is a no-op, not a
        crash."""

        class NoTopKAttr:
            keep_frames = "all"

        validate_capture(NoTopKAttr())  # must not raise

    def test_positive_top_k_within_vocab_passes(self):
        validate_capture(self._FakeCaptureSpec(top_k=16), vocab_size=100)  # must not raise

    def test_positive_top_k_with_vocab_size_none_skips_ceiling_check(self):
        """Named degradation (mirrors validate_constraints' C3 skip): no
        vocab_size available means the ceiling check is skipped, not
        defaulted to a reject."""
        validate_capture(self._FakeCaptureSpec(top_k=999999), vocab_size=None)  # must not raise

    def test_negative_top_k_rejected(self):
        capture = self._FakeCaptureSpec(top_k=-1)
        with pytest.raises(ValueError, match=r"top_k must be >= 0, got -1"):
            validate_capture(capture)
        with pytest.raises(ValueError, match="Fix: use 0 to disable Tier 1"):
            validate_capture(capture)

    def test_non_int_top_k_rejected(self):
        capture = self._FakeCaptureSpec(top_k=2.5)
        with pytest.raises(ValueError, match=r"top_k must be an int, got 2\.5"):
            validate_capture(capture)

    def test_bool_top_k_rejected(self):
        """`bool` is a subclass of `int` in Python — `True`/`False` must not
        silently pass as `1`/`0`; a caller passing a bool almost certainly
        meant something else (e.g. confusing this knob with a toggle)."""
        capture = self._FakeCaptureSpec(top_k=True)
        with pytest.raises(ValueError, match="top_k must be an int"):
            validate_capture(capture)

    def test_top_k_exceeding_vocab_size_rejected(self):
        capture = self._FakeCaptureSpec(top_k=200)
        with pytest.raises(ValueError, match=r"top_k=200 exceeds vocab_size=100"):
            validate_capture(capture, vocab_size=100)
        with pytest.raises(ValueError, match=r"Fix: use top_k <= 100"):
            validate_capture(capture, vocab_size=100)

    def test_top_k_equal_to_vocab_size_passes(self):
        """Boundary: exactly vocab_size is allowed (only strictly-greater is
        rejected — `topk(vocab_size)` over a `vocab_size`-wide row is a
        legitimate, if degenerate, "capture the whole vocabulary" request)."""
        validate_capture(self._FakeCaptureSpec(top_k=100), vocab_size=100)  # must not raise

    def test_tier2_off_by_default_is_a_no_op(self):
        validate_capture(self._FakeCaptureSpec())  # must not raise

    def test_tier2_absent_attributes_default_to_off_no_op(self):
        """Duck-typed: an object exposing neither `capture_full_distribution`
        nor `max_full_distribution_steps` (e.g. a pre-P-C stand-in) is a
        no-op, not a crash."""

        class NoTier2Attrs:
            keep_frames = "all"
            top_k = 0

        validate_capture(NoTier2Attrs())  # must not raise

    def test_tier2_with_budget_passes(self):
        validate_capture(
            self._FakeCaptureSpec(capture_full_distribution=True, max_full_distribution_steps=4)
        )  # must not raise

    def test_tier2_unbounded_request_rejected(self):
        """The load-bearing clause (ADR-CDG-014 Decision 3 Tier 2):
        `capture_full_distribution=True` with no budget is rejected at
        ingress, never silently honored."""
        capture = self._FakeCaptureSpec(capture_full_distribution=True, max_full_distribution_steps=None)
        with pytest.raises(ValueError, match=r"capture_full_distribution=True requires max_full_distribution_steps"):
            validate_capture(capture)
        with pytest.raises(ValueError, match=r"Fix: pass max_full_distribution_steps=<positive int>"):
            validate_capture(capture)

    def test_tier2_budget_without_flag_is_inert_no_op(self):
        """A budget given without `capture_full_distribution=True` does not
        enable Tier 2 and does not raise — the boolean is the knob that
        actually turns capture on."""
        validate_capture(
            self._FakeCaptureSpec(capture_full_distribution=False, max_full_distribution_steps=4)
        )  # must not raise

    def test_tier2_zero_budget_rejected(self):
        capture = self._FakeCaptureSpec(capture_full_distribution=True, max_full_distribution_steps=0)
        with pytest.raises(ValueError, match=r"max_full_distribution_steps must be > 0, got 0"):
            validate_capture(capture)

    def test_tier2_negative_budget_rejected(self):
        capture = self._FakeCaptureSpec(capture_full_distribution=True, max_full_distribution_steps=-3)
        with pytest.raises(ValueError, match=r"max_full_distribution_steps must be > 0, got -3"):
            validate_capture(capture)

    def test_tier2_non_int_budget_rejected(self):
        capture = self._FakeCaptureSpec(capture_full_distribution=True, max_full_distribution_steps=2.5)
        with pytest.raises(ValueError, match=r"max_full_distribution_steps must be an int, got 2\.5"):
            validate_capture(capture)

    def test_tier2_bool_budget_rejected(self):
        """`bool` is a subclass of `int` — `True`/`False` must not silently
        pass as `1`/`0` (same guard as `top_k`'s bool reject)."""
        capture = self._FakeCaptureSpec(capture_full_distribution=True, max_full_distribution_steps=True)
        with pytest.raises(ValueError, match=r"max_full_distribution_steps must be an int"):
            validate_capture(capture)

    def test_tier2_non_bool_flag_rejected(self):
        capture = self._FakeCaptureSpec(capture_full_distribution="yes", max_full_distribution_steps=4)
        with pytest.raises(ValueError, match=r"capture_full_distribution must be a bool"):
            validate_capture(capture)

    def test_real_capture_spec_tier2_budget_passes(self):
        """The minted `CaptureSpec` dataclass itself, not just a duck-typed
        stand-in — Tier 2 on with a real budget passes."""
        validate_capture(
            CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=8)
        )  # must not raise

    def test_real_capture_spec_tier2_unbounded_rejected(self):
        with pytest.raises(ValueError, match=r"requires max_full_distribution_steps"):
            validate_capture(CaptureSpec(capture_full_distribution=True))


class TestHookSourceConflict:
    """H1 (issue #64 §1.4)."""

    def test_neither_given_passes(self):
        reject_conflicting_hook_sources(None, None)  # must not raise

    def test_only_constraints_given_passes(self):
        reject_conflicting_hook_sources(Constraints(pins=(Pin(position=0, token_id=1),)), None)  # must not raise

    def test_only_logit_hook_given_passes(self):
        def a_hook(module, args, output):
            return output

        reject_conflicting_hook_sources(None, a_hook)  # must not raise

    def test_h1_both_given_rejected(self):
        def a_hook(module, args, output):
            return output

        constraints = Constraints(pins=(Pin(position=0, token_id=1),))
        with pytest.raises(ValueError, match="cannot both be given: two logit-mask sources on one door"):
            reject_conflicting_hook_sources(constraints, a_hook)
        with pytest.raises(ValueError, match="Fix: pass only constraints="):
            reject_conflicting_hook_sources(constraints, a_hook)


class TestValidateIngressComposition:
    """`validate_ingress` runs all four checks at one call site — a smoke
    test that composition doesn't short-circuit or reorder past a valid
    payload's peers, plus one reject-path-per-surface round trip."""

    def test_all_none_passes(self):
        validate_ingress(
            None, None, None, None, gen_length=10, num_inference_steps=4, vocab_size=100
        )  # must not raise

    def test_all_valid_payloads_pass_together(self):
        constraints = Constraints(pins=(Pin(position=0, token_id=1),))
        control_signals = ControlSignals(
            bindings=(Binding(target="entropy_bound", signal=(0.0, 0.5, 1.0, 0.5), low=0.02, high=0.3),)
        )
        validate_ingress(
            constraints,
            control_signals,
            None,
            None,
            gen_length=10,
            num_inference_steps=4,
            vocab_size=100,
        )  # must not raise

    def test_constraints_reject_surfaces_through_validate_ingress(self):
        constraints = Constraints(pins=(Pin(position=99, token_id=1),))
        with pytest.raises(ValueError, match="out of range"):
            validate_ingress(
                constraints, None, None, None, gen_length=10, num_inference_steps=4, vocab_size=100
            )

    def test_control_signals_reject_surfaces_through_validate_ingress(self):
        control_signals = ControlSignals(
            bindings=(Binding(target="bogus", signal=(0.0, 0.5, 1.0, 0.5), low=0.02, high=0.3),)
        )
        with pytest.raises(ValueError, match="not a mutable scheduler knob"):
            validate_ingress(
                None, control_signals, None, None, gen_length=10, num_inference_steps=4, vocab_size=100
            )

    def test_hook_conflict_reject_surfaces_through_validate_ingress(self):
        def a_hook(module, args, output):
            return output

        constraints = Constraints(pins=(Pin(position=0, token_id=1),))
        with pytest.raises(ValueError, match="cannot both be given"):
            validate_ingress(
                constraints, None, None, a_hook, gen_length=10, num_inference_steps=4, vocab_size=100
            )

    def test_valid_capture_top_k_passes_through_validate_ingress(self):
        validate_ingress(
            None, None, CaptureSpec(top_k=16), None, gen_length=10, num_inference_steps=4, vocab_size=100
        )  # must not raise

    def test_capture_top_k_reject_surfaces_through_validate_ingress(self):
        """Proves `validate_ingress` threads its own `vocab_size=` kwarg
        through to `validate_capture` (not just to `validate_constraints`) —
        a top_k that exceeds THIS call's vocab_size is rejected here, not
        silently passed."""
        with pytest.raises(ValueError, match=r"top_k=200 exceeds vocab_size=100"):
            validate_ingress(
                None, None, CaptureSpec(top_k=200), None,
                gen_length=10, num_inference_steps=4, vocab_size=100,
            )


    def test_valid_capture_tier2_budget_passes_through_validate_ingress(self):
        validate_ingress(
            None, None, CaptureSpec(capture_full_distribution=True, max_full_distribution_steps=4), None,
            gen_length=10, num_inference_steps=4, vocab_size=100,
        )  # must not raise

    def test_capture_tier2_unbounded_reject_surfaces_through_validate_ingress(self):
        with pytest.raises(ValueError, match=r"requires max_full_distribution_steps"):
            validate_ingress(
                None, None, CaptureSpec(capture_full_distribution=True), None,
                gen_length=10, num_inference_steps=4, vocab_size=100,
            )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "issue #110: ADR-CDG-011 clause 5 names t_min=t_max=v as THE "
        "exact-per-step-temperature mechanism, but dgemma.loop.run_diffusion's "
        "own t_min/t_max ingress guard currently rejects flat equality "
        "(`if t_min >= t_max: raise ValueError(...)`) — found live "
        "2026-07-19, functionality demo had to fudge t_max=0.550001 to work "
        "around it. Pre-registered per the battery S3/#9 convention (issue "
        "#59 phase E3): expected RED until #110 resolves one of two ways — "
        "(a) the guard relaxes to `t_min > t_max` (strict-greater-than "
        "only), and this test flips to PASS, forcing this marker's removal "
        "as part of that fix's own PR; or (b) ADR-CDG-011 clause 5 is "
        "amended to retire the equality mechanism, and this test is removed "
        "with the amendment rather than left green-by-accident."
    ),
)
class TestExactTemperatureRunLevelIngress:
    """Issue #110 pre-registration — deliberately NOT a `dgemma.ingress`-only
    test (see module docstring): the guard this issue names lives in
    `dgemma.loop.run_diffusion` itself, fired as the FIRST statement in the
    function body, before any pipeline/scheduler construction — so this call
    needs no fake-pipeline monkeypatching to reach the assertion; a real
    `DGemmaModel`-shaped stub (never touched beyond attribute presence) is
    sufficient.
    """

    def test_t_min_equals_t_max_is_accepted(self):
        from dgemma.loop import run_diffusion
        from dgemma.types import DGemmaModel

        class _StubProcessor:
            pass

        model = DGemmaModel(
            model=object(),
            processor=_StubProcessor(),
            device="cpu",
            dtype="bfloat16",
            repo_id="fake/repo",
            quant="none",
        )

        # Today this raises ValueError before ever reaching the pipeline —
        # the whole point of the pre-registration is that this call must NOT
        # raise once #110 is resolved via the validator-relaxed path.
        run_diffusion(model, "hi", t_min=0.5, t_max=0.5)
