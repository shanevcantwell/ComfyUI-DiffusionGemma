"""dgemma/hooks.py — the forward-hook lifecycle context manager (#35 R5,
F4; ADR-CDG-010 Decision 5, ADR-CDG-011 clause 8's sibling invariant).

`register_forward_hook` on `pipe.model` is the only logit-shaping door
(issue #28: a `callback_on_step_end` returning `{"logits": ...}` is silently
discarded by the installed `diffusers` pipeline — it isn't in
`_callback_tensor_inputs`'s writable set, only `"canvas"` is applied via
`.pop`). This module is the ENGINE's sole sanctioned installation path for
that door (ADR-CDG-010 Decision 5: "R5's context manager is the sole
installation path... A hook installed any other way... would duplicate F4's
failure mode inside the very feature meant to close it").

**The invariant this closes (ARCHITECTURE.md rule 6, F4):** no hook survives
a `run_diffusion` call. An un-torn-down hook from run A would shape run B's
logits — silent cross-run contamination of a ~53 GB resident model shared
across every subsequent call. `no_hook_survives_a_run` is not a performance
nicety; it is `STATELESS-CORE` applied to the one piece of mutable state a
forward hook represents on an otherwise-immutable loaded module.

**Shape.** `install_logit_shaping_hook(model, hook_fn)` is a context manager
(`@contextlib.contextmanager`) wrapping exactly one `model.register_forward_hook`
call. Teardown (`handle.remove()`) runs in a `finally` block, so it fires on
all three paths a `run_diffusion` call can take:

- clean return (the pipeline call finishes normally);
- `DiffusionCancelled` (`dgemma.composite`'s partial-return path,
  ADR-CDG-010's cancellation amendment) — the exception propagates through
  the `with` block same as any other, and `finally` still runs before it
  reaches `run_diffusion`'s own `except DiffusionCancelled` handler;
- any other exception raised mid-run (a participant defect, a real model
  error) — `finally` runs before the exception continues propagating to the
  caller, exactly like `_FrameCollector`'s own "callback exceptions
  propagate, never swallowed" contract (`dgemma/loop.py`) applied to hook
  teardown instead of frame capture.

`hook_fn` is a plain `torch.nn.Module` forward-hook callable
(`(module, args, output) -> Any | None` — the standard `register_forward_hook`
contract; `None`/no return leaves `output` untouched, a real replacement
value overrides it exactly as `register_forward_hook` already documents).
This module does not interpret or validate `hook_fn`'s body — logit-mask
construction (ADR-CDG-010's `constraints=` mechanism) is a future R-item's
concern; this module only owns the install/teardown lifecycle, the one
piece of the door ADR-CDG-010 Decision 5 requires to be engine-owned and
singular.

`run_diffusion` (`dgemma/loop.py`) wraps its one pipeline call in
`install_logit_shaping_hook(dgemma_model.model, hook_fn)` whenever a hook
function is supplied (`logit_hook=` — today only reachable internally, since
no `constraints=` ingress exists yet per ADR-CDG-010/011's `NOT-YET-IMPLEMENTED`
status; a future constraints implementation builds its hook function and
passes it through this same parameter, never installing its own hook
directly). When no hook function is given (`logit_hook=None`, today's only
real call shape), the context manager is a no-op pass-through — no
`register_forward_hook` call happens at all, so a run with no logit-shaping
need pays zero hook-lifecycle cost and leaves zero installed hooks, which is
also the state the "zero hooks after run" invariant demands trivially.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

# A forward hook, per `torch.nn.Module.register_forward_hook`'s own contract:
# `(module, args, output) -> Any | None`. Returning `None` leaves `output`
# unchanged; returning a value replaces it.
ForwardHookFn = Callable[[Any, tuple, Any], Any]


@contextlib.contextmanager
def install_logit_shaping_hook(model: Any, hook_fn: ForwardHookFn | None) -> Iterator[None]:
    """Install `hook_fn` as a forward hook on `model` for exactly the
    duration of this `with` block; guaranteed removal on every exit path
    (clean return, `DiffusionCancelled`, or any other exception) via
    `try/finally`.

    `hook_fn=None` is a no-op: no `register_forward_hook` call is made and
    nothing is torn down, since nothing was installed. This is the default,
    unwired shape every `run_diffusion` call takes today (no
    `constraints=`-driven logit mask exists yet, ADR-CDG-010/011
    `NOT-YET-IMPLEMENTED`), and it keeps a hookless run's cost at exactly
    zero rather than installing-then-immediately-removing a pass-through
    hook every call.

    This is the ONLY function in `dgemma/` that calls
    `model.register_forward_hook` — ADR-CDG-010 Decision 5's "sole
    installation path" clause, checkable by grep as much as by test: no
    other module in this package may call `register_forward_hook` directly.
    """
    if hook_fn is None:
        yield
        return

    handle = model.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()
