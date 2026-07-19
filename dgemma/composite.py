"""dgemma/composite.py — the engine-internal ordered step-end composite
(ADR-CDG-010 Decision 3, replacing the single hardcoded callback binding at
`dgemma/loop.py:582` this module's docstring cites as F1/ARCHITECTURE.md's
"Single hardcoded callback binding" row).

`run_diffusion` builds exactly one `StepEndComposite` per call and passes its
`__call__` as `callback_on_step_end` — same `(pipe, global_step, step_idx,
callback_kwargs) -> dict` shape the diffusers pipeline already calls
(`pipeline_diffusion_gemma.py:404-407`), so this is a drop-in replacement at
the pipeline boundary: existing direct-call tests
(`tests/test_run_diffusion_knobs.py`'s `callback(self, 0, 0, callback_kwargs)`
pattern) keep working against whatever object occupies that slot.

**Fixed ordering (ADR-CDG-010 Decision 3 + its cancellation amendment
2026-07-13, ADR-CDG-011 clause 6), engine-owned, never caller-configurable:**

    capture -> cancellation check -> beta-rebuild -> pin -> walker

- **Capture runs first — before the cancellation check and before any
  canvas-writer** (ADR-CDG-010 Decision 3 + amendment, PR #45). This
  callback fires at `callback_on_step_end`, AFTER the scheduler's `step()`
  has already committed this step's canvas
  (`pipeline_diffusion_gemma.py:365-371` -> `:404-407`) — so at the moment
  cancellation could trip, the step's canvas is a *committed* frame, not an
  in-flight partial. Capturing before the cancellation check means a
  cancelled run's trace retains its exact truncation point: the committed
  frame of the very step the caller cancelled on. An instrumentation-first
  pack pays one capture to keep that evidence (#38's "a cancelled
  experiment run is still data"); the cancel-first alternative (rejected —
  see the ADR amendment) silently discarded it.
- **Cancellation runs second — after capture, before every canvas-writer**
  (issue #38's fold-in): the read-only cancel check still gates all writer
  work, so no beta-rebuild/pin pass runs for a step whose result will never
  be used. Raising `DiffusionCancelled` here is caught by `run_diffusion`
  (`dgemma/loop.py`), which returns the partial `CanvasTrace` built from
  the frames captured so far — INCLUDING this step's truncation-point
  frame (#38's "return what exists" clause).
- **Beta-rebuild runs before pin.** Renoise/rebuild participants must finish
  writing before pin re-asserts, or a pin's re-assertion could be immediately
  overwritten by a renoise pass that doesn't know the cell was just pinned.
- **Pin is the last writer.** Every other participant has had its turn to
  write the canvas; pin's re-assertion is what actually reaches the next
  forward pass unclobbered.

**Walker (ADR-CDG-011, issue #64 Phase 4):** the `walker` field is a
config-mutator, not a canvas-writer — it never occupies a slot in the
capture/beta-rebuild/pin canvas-write order above. It runs LAST, after every
canvas-writer, per the design-gate ratification on issue #64 (2026-07-13):
ADR-CDG-011 clause 6 fixes only a lower bound (walker must run after capture
reads the current step's effective knobs, or capture would report the next
step's config as the current step's) and is silent on the upper bound
relative to pin; because the walker provably touches only
`scheduler.config` and never the canvas, its position relative to the
canvas-writers is behaviorally inert on canvas output, so walker-last keeps
the fixed writer-order loop below untouched by its addition. Skipped on a
cancelled step exactly like `beta_rebuild`/`pin` — the cancellation check
(step 2 below) raises before the writer loop or the walker ever run.

This module holds ONLY engine-built participants (ARCHITECTURE.md rule 7):
`run_diffusion` widens by declarative payloads (`constraints=`,
`control_signals=`, `capture=`), validated at ingress and turned into
participant instances constructed here — never a surface-supplied closure or
hook (#35 delta Correction 3). `on_frame` (`dgemma/loop.py:477`) is
deliberately NOT a composite participant (#35 delta Correction 2): it stays
on the existing read-only observer seam, invoked by `_FrameCollector` itself
after this composite's capture participant has built the frame, structurally
outside the ordered list below.

**Exception propagation (engine contract, unchanged from `_FrameCollector`'s
existing convention):** a participant's exception PROPAGATES out of
`StepEndComposite.__call__` — the composite does not swallow a participant's
error any more than `_FrameCollector.on_frame` swallows a caller's callback
error (`dgemma/loop.py`'s `_FrameCollector` docstring, review finding
2026-07-05). `DiffusionCancelled` is the one distinguished exception
`run_diffusion` itself catches (by design, to implement partial-return
semantics); every other participant exception is not caught anywhere in this
module or in `run_diffusion` — it propagates to the pipeline's `__call__` and
out to the caller, same as today's single hardcoded binding.

**Why this shape survived R5/R2 without reshaping (the brief's requirement):**
adding the walker (ADR-CDG-011, issue #64 Phase 4, landed) and the pin
participant (ADR-CDG-010, issue #64 Phase 3, landed) was appending a typed
field plus a constructor branch here — the composite's public shape
(`StepEndComposite`, fixed writer order, `__call__` signature) did not
change for either addition. The still-`NOT-YET-IMPLEMENTED` beta-rebuild
participant (ADR-CDG-010, Phase 5) fits the same way: one more entry in the
`beta_rebuild` tuple. A participant that is both a config-mutator and a
canvas-writer (ADR-CDG-011's Cross-references caveat) still fits: it is one
more named slot in the same fixed list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


class DiffusionCancelled(Exception):
    """Raised by the cancellation participant when the surface-supplied
    cancel predicate reports True (issue #38).

    Distinguished from a generic exception so `run_diffusion`
    (`dgemma/loop.py`) can catch it specifically and return the partial
    `(text, CanvasState, CanvasTrace)` built from whatever frames were
    already captured, rather than losing the run's evidence — #38's
    "a cancelled experiment run is still data" clause. Every other
    participant exception is NOT caught anywhere in this module; only
    cancellation gets partial-return treatment, because only cancellation is
    an expected, surface-triggered stop rather than a participant defect.
    """

    def __init__(self, step_idx: int) -> None:
        self.step_idx = step_idx
        super().__init__(f"Diffusion cancelled at step_idx={step_idx}.")


class StepEndParticipant(Protocol):
    """One named slot in the ordered composite.

    `__call__(pipe, global_step, step_idx, callback_kwargs) -> dict | None`:
    the same shape `callback_on_step_end` itself is called with
    (`pipeline_diffusion_gemma.py:404-407`), so a participant can be unit
    tested standalone with the same fake `callback_kwargs` fixture
    (`tests/conftest.py`) used for the composite as a whole. A non-writer
    participant (cancellation, capture) returns `None`/`{}`; a canvas-writer
    (beta-rebuild, pin) returns `{"canvas": <tensor>}` to override the
    scheduler's own `prev_sample` for that step, exactly the `{"canvas":
    ...}` application the fixture's `FakeDiffusionGemmaPipeline` reproduces.
    """

    name: str

    def __call__(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict | None: ...


@dataclass
class _CancellationParticipant:
    """Checks a surface-neutral cancel predicate once per step (issue #38).

    `should_cancel`: a zero-argument predicate returning `True` when the run
    should stop — surface-agnostic by construction (ARCHITECTURE.md rule 1):
    a ComfyUI surface wires this to `comfy.model_management`'s interrupt
    check, an MCP surface wires it to its own abort signal, and this module
    never imports either. `None` (the default — no `cancel=` payload given)
    means the participant is a no-op every step, at the cost of one
    attribute check; this is cheaper and simpler than conditionally omitting
    the slot from `_STEP_ORDER`, and keeps the fixed order genuinely fixed
    regardless of whether a given run wires cancellation.

    Raises `DiffusionCancelled` the FIRST step the predicate reports `True` —
    it does not re-check or debounce; a predicate that flips back to `False`
    on a later call (not expected from a real interrupt flag, but not this
    participant's business to guard against) still cancels once tripped,
    because the composite doesn't get a second chance to run this step's
    writers after raising. By the time this check runs, capture has already
    recorded the step's committed frame (the amendment's capture-first
    ordering) — cancellation truncates the run, never the evidence.
    """

    should_cancel: Callable[[], bool] | None = None
    name: str = "cancellation"

    def __call__(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict | None:
        if self.should_cancel is not None and self.should_cancel():
            raise DiffusionCancelled(step_idx)
        return None


@dataclass
class StepEndComposite:
    """The ordered composite occupying `run_diffusion`'s one
    `callback_on_step_end` slot (ADR-CDG-010 Decision 3).

    Construct with `capture` (today's `_FrameCollector.on_step_end`-shaped
    callable — kept as a plain callable, not wrapped in a
    `StepEndParticipant`, since `_FrameCollector` is the pre-existing capture
    participant and already matches the call shape exactly) and, optionally,
    `should_cancel` (issue #38's cancellation seam). `beta_rebuild`/`pin` are
    accepted now as optional participant lists so ADR-CDG-010's beta-renoise/
    pin participants slot in without changing this class's shape — no
    beta-rebuild participant exists yet (ADR-CDG-010's own R2/Phase 5 scope),
    so it defaults to `()`; `pin` is filled by issue #64 Phase 3. `walker`
    (issue #64 Phase 4, ADR-CDG-011) is a single optional participant, not a
    tuple — the walker is a config-mutator, not a canvas-writer, and this
    phase's scope names exactly one control-signal walker per run (no
    ordering-among-walkers question exists to motivate a list, unlike
    `beta_rebuild`/`pin`).

    `__call__` runs, in this fixed order: `capture`, the cancellation check,
    every `beta_rebuild` participant (in list order), every `pin` participant
    (in list order), then `walker` (if given). A canvas-writer's returned
    `{"canvas": ...}` is threaded into `callback_kwargs["canvas"]` for the
    NEXT participant in the list, so a later writer sees an earlier writer's
    output rather than the step's original pre-callback canvas — the same
    threading the diffusers pipeline itself does across callback calls
    (`callback_outputs.pop("canvas", canvas)`,
    `pipeline_diffusion_gemma.py:407`), just applied within one callback
    instead of across callbacks. `walker` runs after the canvas-writer loop
    and its return is ignored (ADR-CDG-011: a config-mutator's return is
    never threaded as a canvas). The final writer's `{"canvas": ...}` (or,
    if no writer fired, `{}`) is this composite's own return — exactly what
    `callback_on_step_end`'s contract expects; `walker`'s presence never
    changes that return value.
    """

    capture: Callable[[Any, int, int, dict], dict]
    should_cancel: Callable[[], bool] | None = None
    beta_rebuild: tuple[StepEndParticipant, ...] = field(default_factory=tuple)
    pin: tuple[StepEndParticipant, ...] = field(default_factory=tuple)
    walker: StepEndParticipant | None = None

    def __post_init__(self) -> None:
        self._cancellation = _CancellationParticipant(should_cancel=self.should_cancel)

    def __call__(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict:
        # 1. Capture — must see the pre-writer canvas (ADR-CDG-010 Decision
        #    3), and runs BEFORE the cancellation check (ADR-CDG-010
        #    amendment 2026-07-13, PR #45): the scheduler has already
        #    committed this step's canvas by callback time, so capturing
        #    first retains the truncation-point frame on a cancelled run.
        self.capture(pipe, global_step, step_idx, callback_kwargs)

        # 2. Cancellation — read-only; raises after this step's committed
        #    frame is captured but before any writer runs (or the walker
        #    mutates scheduler.config) for a step whose result will never
        #    be used.
        self._cancellation(pipe, global_step, step_idx, callback_kwargs)

        # 3. Beta-rebuild, then 4. pin — canvas-writers, in that fixed order;
        #    each sees the previous writer's output via callback_kwargs
        #    threading, mirroring the pipeline's own across-callback
        #    "canvas = callback_outputs.pop(...)" application.
        result: dict = {}
        working_kwargs = callback_kwargs
        for participant in (*self.beta_rebuild, *self.pin):
            output = participant(pipe, global_step, step_idx, working_kwargs)
            if output and "canvas" in output:
                result = output
                working_kwargs = {**working_kwargs, "canvas": output["canvas"]}

        # 5. Walker (ADR-CDG-011, issue #64 Phase 4) — LAST, after every
        #    canvas-writer (design-gate ratification, 2026-07-13): a
        #    config-mutator, never a canvas-writer, so its return (always
        #    None, dgemma.participants.WalkerParticipant) is discarded, never
        #    threaded into `result`.
        if self.walker is not None:
            self.walker(pipe, global_step, step_idx, working_kwargs)

        return result
