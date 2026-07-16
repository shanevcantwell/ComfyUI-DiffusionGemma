"""dgemma/constraints_hook.py — the logit-mask hook body built from a
validated `Constraints` payload (ADR-CDG-010 Decision 1(a), Decision 5;
issue #64 §4, Phase 3).

`build_logit_mask_hook` is the sibling mechanism to
`dgemma.participants.PinParticipant`: where the pin participant guarantees
*what conditions* the next forward pass (a canvas re-assertion, run every
step regardless of what the model would have chosen), this module builds a
forward hook that restricts *what commits* — masking a pinned position's
logits down to its single given `token_id` so that cell reads ~zero
predictive entropy and the entropy-bound scheduler accepts it first
(ADR-CDG-010's most-constrained-first framing; the failure this prevents is
Option B's rejection in the ADR: an unmasked pinned cell pays full
entropy-budget weight like any other cell and has no commit-ordering
advantage, so it is only "eventually correct" rather than "commits first,
costs nothing").

**Sole installation path (ADR-CDG-010 Decision 5):** this module ONLY
builds a `hook_fn` closure — it never calls `register_forward_hook` itself.
`run_diffusion` (`dgemma/loop.py`) passes the built hook through the
existing `logit_hook=` parameter to `dgemma.hooks.install_logit_shaping_hook`,
the engine's one sanctioned installation site (R5, F4). A second install
site anywhere — including here — would duplicate the exact failure mode
(#28) ADR-CDG-010 D5 closes; `tests/test_constraints_hook.py`'s
`TestSoleInstallPath` greps this module's source for `register_forward_hook`
and asserts it is absent, extending the R5 grep discipline
`dgemma/hooks.py`'s own docstring names.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .hooks import ForwardHookFn

if TYPE_CHECKING:
    from .payloads import Pin


def build_logit_mask_hook(pins: "tuple[Pin, ...]", *, vocab_size: int | None) -> ForwardHookFn:
    """Build a `register_forward_hook`-shaped closure that masks each pin's
    position to its single `token_id` (ADR-CDG-010 Decision 1(a)).

    `pins`: the validated `Constraints.pins` tuple (ingress already confirmed
    every `position`/`token_id` in range — `dgemma.ingress.validate_constraints`
    C2/C3 — so this function does no bounds-checking of its own; it is a pure
    logit-mutation body, not a second validation site).

    `vocab_size`: the model's vocabulary size — the same value
    `dgemma.ingress.validate_constraints` resolved at ingress
    (`dgemma.loop.resolve_vocab_size`), passed through so this module needs no
    tokenizer/processor access of its own (rule 1: this module imports zero
    ComfyUI and stays a pure tensor-mutation body). Kept in the signature per
    issue #64 §4's locked shape even though the mask write itself
    (`logits[..., position, :] = -inf`) needs no declared size — `logits`'s
    own last-dim shape IS the vocabulary at runtime. `None` mirrors
    `resolve_vocab_size`'s own named degradation (a bare test stub whose
    processor exposes no usable tokenizer, `dgemma/loop.py`): the defensive
    bounds re-check below is skipped in that case, exactly like C3 is
    skipped at ingress for the same reason — this function never invents a
    size the caller couldn't resolve.

    **Mask math (locked, issue #64 §4):** for each pin, set
    `logits[..., position, :] = -inf` then `logits[..., position, token_id] =
    0` — an additive RESTRICTION of one position's vocabulary to a single id,
    never a rewrite of the whole logit field; every non-pinned position is
    left untouched. The returned closure mutates `output.logits` in place and
    returns `output`, matching the standard `register_forward_hook`
    replacement contract (`(module, args, output) -> Any | None` — returning
    a value replaces `output`; `dgemma/hooks.py`'s own `ForwardHookFn` alias).

    Returns a no-op-equivalent closure (mutates nothing, returns `output`
    unchanged) when `pins` is empty — mirrors `Constraints()`'s "empty ==
    no-op" contract (`dgemma/payloads.py`); `run_diffusion` does not build
    this hook at all for an empty/`None` `constraints=` (see
    `dgemma/loop.py`), so this branch exists for direct unit-test callers of
    `build_logit_mask_hook` itself, not a real `run_diffusion` code path.
    """

    def hook_fn(module: Any, args: tuple, output: Any) -> Any:
        if not pins:
            return output
        logits = output.logits
        for pin in pins:
            if vocab_size is not None and not (0 <= pin.token_id < vocab_size):
                raise ValueError(
                    f"Pin token_id {pin.token_id} is out of vocabulary (vocab_size={vocab_size}, valid: "
                    f"0..{vocab_size - 1}) at logit-mask time. This should have been caught at ingress "
                    "(dgemma.ingress.validate_constraints C3) — reaching this check means "
                    "build_logit_mask_hook was called directly, bypassing run_diffusion's ingress. Fix: "
                    "validate the Constraints payload via dgemma.ingress.validate_constraints before "
                    "building the hook, or pass an in-vocab token_id."
                )
            logits[..., pin.position, :] = float("-inf")
            logits[..., pin.position, pin.token_id] = 0.0
        return output

    return hook_fn
