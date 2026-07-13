"""dgemma/ingress.py — payload ingress validators (ADR-CDG-010/011, issue #64
§1.4/§1.5).

`EMIT-CANONICAL / PARSE-AT-THE-DOOR` applied to the `constraints=`/
`control_signals=`/`capture=` payloads (ARCHITECTURE.md rule 5): every reject
here is a `ValueError` naming the violated precondition AND a remedy — never
a silent clamp or trust-and-degrade. `run_diffusion` (`dgemma/loop.py`) calls
`validate_ingress(...)` before constructing the scheduler.

**Fail-on-unknown is structural for payload KEYS:** because `Constraints`/
`ControlSignals`/`Pin`/`Binding` are frozen dataclasses, an unrecognized
field cannot be passed at all — the dataclass constructor itself raises
`TypeError`, which IS the enforcement surface for "reject unknown payload
keys" ("name the enforcement surface", CLAUDE.md). The validators below
cover unknown *values* only (an out-of-vocab id, an unknown target knob,
an out-of-range signal sample) — the class of error a frozen dataclass
cannot catch on its own.

**Phase 1 scope (issue #64 §6):** every validator lands now, with its full
reject register. No participant reads a validated payload yet — Phase 1
validates-then-ignores; Phases 3/4 wire the accepted payloads into
`PinParticipant`/`WalkerParticipant`.
"""
from __future__ import annotations

import math
from typing import Any

from .payloads import Binding, Constraints, ControlSignals, MUTABLE_TARGETS, Pin


def validate_constraints(constraints: "Constraints | None", *, gen_length: int, vocab_size: int | None) -> None:
    """Validate a `constraints=` payload (ADR-CDG-010).

    `vocab_size=None` skips C3 (pin-token-in-vocab) with a named degradation
    — mirrors `resolve_thought_channel_ids`'s stub fallback (`dgemma/loop.py`)
    for a bare test double / a processor that doesn't expose a usable
    tokenizer, per issue #64 §3.4. `None`/no pins is always a no-op.
    """
    if constraints is None:
        return
    if callable(constraints) and not isinstance(constraints, Constraints):
        raise ValueError(
            f"constraints= must be a Constraints payload, not a callable (got {type(constraints)}). "
            "run_diffusion widens by declarative payloads only (ADR-CDG-010 D2, rule 7); the engine "
            "builds the logit-mask hook from your pins. Fix: pass Constraints(pins=(Pin(position=..., "
            "token_id=...),))."
        )

    seen_positions: set[int] = set()
    for pin in constraints.pins:
        if not (0 <= pin.position < gen_length):
            raise ValueError(
                f"Pin position {pin.position} out of range for gen_length={gen_length} "
                f"(valid: 0..{gen_length - 1}). Fix: pin a position inside the generated canvas."
            )
        if vocab_size is not None and not (0 <= pin.token_id < vocab_size):
            raise ValueError(
                f"Pin token_id {pin.token_id} is out of vocabulary (vocab_size={vocab_size}, "
                f"valid: 0..{vocab_size - 1}) — a constraint id must be in-vocab (ADR-CDG-010 D2). "
                "Fix: pin an in-vocab token id (tokenizer.convert_tokens_to_ids for a string)."
            )
        if pin.position in seen_positions:
            raise ValueError(
                f"Duplicate Pin for position {pin.position}: two givens for one canvas cell is "
                "ambiguous. Fix: supply one Pin per position."
            )
        seen_positions.add(pin.position)


def validate_control_signals(control_signals: "ControlSignals | None", *, num_inference_steps: int) -> None:
    """Validate a `control_signals=` payload (ADR-CDG-011)."""
    if control_signals is None:
        return
    if callable(control_signals) and not isinstance(control_signals, ControlSignals):
        raise ValueError(
            f"control_signals= must be a ControlSignals payload, not a callable (got {type(control_signals)}). "
            "Declarative payloads only (ADR-CDG-011 clause 1). Fix: pass ControlSignals(bindings=(Binding("
            "target=..., signal=..., low=..., high=...),))."
        )

    for binding in control_signals.bindings:
        len_signal = len(binding.signal)
        if len_signal != num_inference_steps:
            raise ValueError(
                f"Binding for target {binding.target!r} has signal length {len_signal}, but "
                f"num_inference_steps={num_inference_steps}. A control signal is rejected when its "
                "length != step count, never truncated or padded (ADR-CDG-011 clause 1). Fix: generate "
                f"a signal of exactly {num_inference_steps} samples."
            )

        # V4 is checked before V3 so a num_inference_steps binding gets the
        # #20-specific remedy, never the generic "unknown target" message
        # (issue #64 §1.4: "a caller who binds num_inference_steps gets the
        # #20-specific remedy, not a generic unknown-target one").
        if binding.target == "num_inference_steps":
            raise ValueError(
                "Binding target 'num_inference_steps' is rejected: it is non-mutable mid-run by design "
                "(ADR-CDG-011 clause 4). Mutating it would desync the scheduler's cached "
                "predictor_steps/_num_timesteps from the anneal denominator — issue #20's mechanism, "
                "foreclosed here by construction. Fix: to change effective step count, set "
                "num_inference_steps= on the run itself, not as a bound target."
            )
        if binding.target not in MUTABLE_TARGETS:
            raise ValueError(
                f"Binding target {binding.target!r} is not a mutable scheduler knob "
                f"(valid: {sorted(MUTABLE_TARGETS)}). Fix: bind one of the fresh-read config knobs."
            )

        low, high = binding.low, binding.high
        if (
            not isinstance(low, (int, float))
            or not isinstance(high, (int, float))
            or not math.isfinite(low)
            or not math.isfinite(high)
            or low == high
        ):
            raise ValueError(
                f"Binding for target {binding.target!r} has a degenerate range low={low} == high={high}. "
                "Fix: use low != high; for an exact per-step temperature bind t_min and t_max to the "
                "same signal instead (ADR-CDG-011 clause 5)."
            )

        for i, value in enumerate(binding.signal):
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Binding for target {binding.target!r} has signal value {value} at step {i} "
                    "outside the unitless range [0.0, 1.0]. A control signal is unitless; units are "
                    "applied by the binding's [low, high] (ADR-CDG-011 clause 3). Fix: normalize the "
                    "generator's output to [0,1]."
                )


def validate_capture(capture: Any) -> None:
    """Validate a `capture=` payload's `keep_frames` field (P1 per issue #64
    §1.4).

    Deliberately duck-typed (`getattr`, not `isinstance` against a dataclass
    minted here): per ADR-CDG-014 Decision 7, the `capture=` payload's
    dataclass is owned by the capture cluster (issue #61), not this module
    (see `dgemma/payloads.py`'s module docstring). This validator only checks
    the one field issue #64 contributes (`keep_frames`); it does not assume
    or import a `CaptureSpec` type. `None` is always a no-op.
    """
    if capture is None:
        return
    keep_frames = getattr(capture, "keep_frames", "all")
    if keep_frames not in ("last", "all"):
        raise ValueError(
            f"CaptureSpec.keep_frames must be 'last' or 'all', got {keep_frames!r}. "
            "Fix: use one of the two retention policies."
        )


def reject_conflicting_hook_sources(constraints: "Constraints | None", logit_hook: Any) -> None:
    """H1: `constraints=` and a raw `logit_hook=` cannot both be given — two
    logit-mask sources on one door (ADR-CDG-010 D5, the R5 single-
    installation-path clause)."""
    if constraints is not None and logit_hook is not None:
        raise ValueError(
            "constraints= and logit_hook= cannot both be given: two logit-mask sources on one door "
            "(ADR-CDG-010 D5, the R5 single-installation-path clause). Fix: pass only constraints=; "
            "the engine builds and installs the mask hook."
        )


def validate_ingress(
    constraints: "Constraints | None",
    control_signals: "ControlSignals | None",
    capture: Any,
    logit_hook: Any,
    *,
    gen_length: int,
    num_inference_steps: int,
    vocab_size: int | None,
) -> None:
    """Run every ingress validator, in the order issue #64 §3.4 specifies.

    Single call site for `run_diffusion` — validates all four ingress
    surfaces (`constraints`, `control_signals`, `capture`, the
    `constraints`/`logit_hook` hook-source conflict) before any participant
    or scheduler is constructed.
    """
    validate_constraints(constraints, gen_length=gen_length, vocab_size=vocab_size)
    validate_control_signals(control_signals, num_inference_steps=num_inference_steps)
    validate_capture(capture)
    reject_conflicting_hook_sources(constraints, logit_hook)
