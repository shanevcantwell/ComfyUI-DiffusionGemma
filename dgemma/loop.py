"""dgemma/loop.py — the denoising-loop spine (ADR-CDG-004 drive seam).

Drives a preloaded `DiffusionGemmaForBlockDiffusion` (from `dgemma/model.py`)
through `diffusers.DiffusionGemmaPipeline` + `EntropyBoundScheduler`, per
ADR-CDG-004. Per-step frames are the loop's native contract from day one
(plan.md, `dgemma/loop.py` per-module notes): P1 keeps only the last frame
(`keep_frames="last"`), but the collection seam iterates every step
regardless, so P2 (knobs) and P3 (instrumentation) grow the same generator
without a reshape.

**Diffusers version guard + structural probe (issue #35 R3, ARCHITECTURE.md
"No diffusers version guard" row).** This module is diffusers' real import
site (`model.py`'s transformers guard has its own module; this is the twin,
here rather than there because `import diffusers` happens here, not in
`model.py` — verified: `dgemma/__init__.py` imports `.loop` before `.model`,
so in practice diffusers lands in `sys.modules` before transformers does on
a fresh `import dgemma`). `anneal_temperature` below re-derives
`EntropyBoundScheduler.step()`'s inlined anneal formula
(`scheduling_entropy_bound.py:153-155`, installed diffusers 0.39.0) instead
of reading it off the scheduler, because the formula isn't exposed on
`EntropyBoundSchedulerOutput` — a version bump that renames/reshapes
`accepted_index`/`.config.t_min`/`.config.t_max`/`.num_inference_steps`, or
narrows the base pipeline's `_callback_tensor_inputs` allowlist
`DGemmaPipeline` widens, would make this module silently report wrong values
with no error at all (the exact trust-and-degrade gap ADR-CDG-001 forbids,
CLAUDE.md). `_check_diffusers_version` + `_check_diffusers_structure` turn
*that* drift — the name/shape kind — into a loud, actionable `RuntimeError`
naming which structure moved.

**What the guards structurally CANNOT see (named residual, PR #48 gate
finding F-1):** a change to the anneal formula's *body* — the arithmetic at
`scheduling_entropy_bound.py:153-154` — that keeps every probed name in
place (same ctor kwargs, same config attrs, same output fields, different
math). A `hasattr`/`inspect.signature` probe has no purchase on an
expression inlined inside `step()`. That residual carries its own
enforcement surface instead: `tests/test_diffusers_version_guard.py:
TestAnnealFormulaPin` drives the REAL installed scheduler's `step()` with
known logits, recovers the temperature it actually applied (from
`pred_logits = model_output / temperature`, the scaled-logits field `step()`
returns — `scheduling_entropy_bound.py:155,181`), and asserts
`anneal_temperature`'s re-derivation matches — so an accepted diffusers
version that rewrites the formula fails that test loudly instead of this
module silently lying through its `t`/`temperature` telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch

# `pyproject.toml`/`requirements.txt` pin `diffusers>=0.39.0` — an open
# lower-bounded range, unlike transformers' exact `==5.13.0` pin
# (`dgemma/model.py`), because this pack does not vendor a diffusers fork and
# has no reason to reject a newer compatible release outright. Guard
# semantics are therefore a two-layer split rather than one exact-pin check:
#
# 1. `_check_diffusers_version`: reject anything BELOW the declared floor
#    (`>=0.39.0` needs the real check, not just "importable") — a stale
#    diffusers predating `DiffusionGemmaPipeline`/`EntropyBoundScheduler`
#    entirely fails as an ImportError with no context otherwise.
# 2. `_check_diffusers_structure`: a version bump ABOVE the floor is
#    accepted (the declared range says so) but is untested surface for the
#    names/shapes this module reads off diffusers — so instead of trusting
#    a newer version to keep every probed structure byte-identical, this
#    probe asserts each one directly, unconditional on version number, and
#    fails loud naming exactly which structure moved. This is the
#    range+structural-probe split (not exact-pin): an exact pin would be
#    both stricter than the declared dependency bound and wrong the moment
#    diffusers ships a compatible 0.40/0.41.
#
#    Honest scope: the probe protects the NAMES/SHAPES `anneal_temperature`
#    and friends read (ctor kwargs, config attrs, output fields, the base
#    callback allowlist) — it cannot see the anneal formula's *body*, which
#    `anneal_temperature` re-derives. That formula-body residual is enforced
#    by `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin`, which
#    pins the re-derivation against the temperature the REAL installed
#    scheduler's `step()` actually applies (see the module docstring above).
REQUIRED_DIFFUSERS_MINIMUM = "0.39.0"


def _check_diffusers_version(installed: str | None = None) -> None:
    """Raise an actionable `RuntimeError` (issue #35 R3) unless the installed
    diffusers is `>= REQUIRED_DIFFUSERS_MINIMUM`, matching the
    `pyproject.toml`/`requirements.txt` declared floor.

    Twin of `dgemma.model._check_transformers_version`, adapted for a
    lower-bounded range instead of an exact-pin series: transformers is
    pinned `==5.13.0` (patch-tolerant series match) because this pack reads
    undocumented internals across that exact tested release; diffusers is
    declared `>=0.39.0` because nothing here forks diffusers, so a newer
    release is intentionally accepted — `_check_diffusers_structure` (below)
    guards the names/shapes a version-floor check alone cannot, and
    `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin` guards the
    re-derived anneal formula's body, which neither check here can see.

    `installed` is normally left `None` (reads the real `diffusers.__version__`
    at call time) — the parameter exists so this thin guard is directly
    unit-testable without monkeypatching `sys.modules`. Compares with
    `packaging.version.Version` when `packaging` is importable (it normally
    is: diffusers depends on it itself); falls back to a best-effort tuple
    compare of the leading numeric dotted components when `packaging` isn't
    importable, so a missing `packaging` degrades to a slightly less precise
    check rather than an uncaught ImportError from this guard itself.
    """
    if installed is None:
        import diffusers as _diffusers

        installed = getattr(_diffusers, "__version__", "unknown")

    try:
        from packaging.version import Version

        below_floor = Version(installed) < Version(REQUIRED_DIFFUSERS_MINIMUM)
    except Exception:
        below_floor = _tuple_version(installed) < _tuple_version(REQUIRED_DIFFUSERS_MINIMUM)

    if below_floor:
        raise RuntimeError(
            f"ComfyUI-DiffusionGemma requires diffusers >= {REQUIRED_DIFFUSERS_MINIMUM} "
            f"(pyproject.toml/requirements.txt declare 'diffusers>={REQUIRED_DIFFUSERS_MINIMUM}'), "
            f"but diffusers=={installed} is installed in this Python environment. "
            "ComfyUI-Manager's dependency installer silently skips a requirements.txt pin "
            "that would downgrade an already-installed package, so this environment can "
            "hold a diffusers version older than this pack's declared floor even after a "
            "normal Manager install. Fix: run "
            f"`pip install 'diffusers>={REQUIRED_DIFFUSERS_MINIMUM}'` in ComfyUI's own Python "
            "environment. See issue #35."
        )


def _tuple_version(version: str) -> tuple[int, ...]:
    """Best-effort `(major, minor, patch, ...)` int tuple from a dotted
    version string, used only when `packaging` isn't importable. Stops at
    the first non-numeric component (e.g. a `rc1`/`dev0`/`+local` suffix) so
    a pre-release/local-build tag doesn't raise `ValueError` here; a version
    string with no leading numeric component at all degrades to `(0,)`
    rather than raising, since the caller only ever compares this against
    another `_tuple_version` result and an under-full tuple already compares
    correctly-low.

    Named limitation (honest, not silently papered over): dropping the
    non-numeric suffix entirely means a pre-release of the floor itself
    (`"0.39.0.dev0"`) compares EQUAL to, not below, the plain floor
    (`"0.39.0"`) on this fallback path — the opposite of `packaging.version.
    Version`'s PEP 440 ordering, which correctly sorts a dev release below
    its final. This fallback only runs when `packaging` is missing from an
    environment where diffusers itself is installed (diffusers depends on
    `packaging`, so this is already the degraded, off-the-happy-path case);
    trading exact pre-release ordering for a dependency-free string parse is
    the same shape of compromise `_check_transformers_version`'s own
    string-prefix fallback makes (patch-tolerant, not full-spec)."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) or (0,)


def _check_diffusers_structure() -> None:
    """Structural probe (issue #35 R3): assert the shapes this module reads
    off `diffusers` actually exist, independent of the version-floor check
    above — a version bump within (or even at) the declared `>=0.39.0` range
    is accepted by `_check_diffusers_version` but is not thereby guaranteed
    to keep any of these structures byte-identical. Each assertion below
    fails LOUD, naming exactly which structure moved and what this module
    depends on it for (EMIT-CANONICAL: fail on the unknown, never
    trust-and-degrade — CLAUDE.md).

    Probed, and why:

    - `EntropyBoundScheduler.__init__` accepts `entropy_bound`/`t_max`/
      `t_min`/`num_inference_steps` and exposes them on `.config`
      (`scheduling_entropy_bound.py:78-85`) — `run_diffusion` constructs the
      scheduler with exactly these kwargs, and `anneal_temperature` re-derives
      its formula from `.config.t_min`/`.config.t_max`.
    - The scheduler exposes `.num_inference_steps` as a plain instance
      attribute, not only via `.config` (`scheduling_entropy_bound.py:84,89`)
      — `_FrameCollector` reads `scheduler.num_inference_steps` lazily per
      callback (issue #20; see that class's own docstring) precisely because
      it is a mutable plain attribute, not the frozen config copy.
    - `EntropyBoundSchedulerOutput` (the real `scheduler.step()` return
      type) exposes an `accepted_index` field (`scheduling_entropy_bound.py
      :26-40`) — `_FrameCollector.on_step_end` reads
      `scheduler_output.accepted_index` directly.
    - The base `diffusers.DiffusionGemmaPipeline._callback_tensor_inputs`
      allowlist is `["canvas", "logits"]` (`pipeline_diffusion_gemma.py:76`)
      — `DGemmaPipeline` widens it by appending `"scheduler_output"`
      (this module, below); if the base class ever renamed or dropped either
      of its own two entries, silently keeping this module's widened list
      could paper over an upstream allowlist rename this module never
      accounted for.

    Raises `RuntimeError` naming the missing/changed structure and which
    dgemma symbol depends on it. Never imports `torch`/builds tensors here —
    a pure `hasattr`/`inspect.signature` probe, cheap enough to run at every
    module import (mirrors `_check_transformers_version`'s own
    module-import-time invocation, `dgemma/model.py:120`).
    """
    import inspect

    from diffusers import DiffusionGemmaPipeline, EntropyBoundScheduler
    from diffusers.schedulers.scheduling_entropy_bound import EntropyBoundSchedulerOutput

    ctor_params = inspect.signature(EntropyBoundScheduler.__init__).parameters
    required_ctor_kwargs = {"entropy_bound", "t_max", "t_min", "num_inference_steps"}
    missing_ctor_kwargs = required_ctor_kwargs - ctor_params.keys()
    if missing_ctor_kwargs:
        raise RuntimeError(
            "diffusers structural probe failed (issue #35 R3): "
            f"EntropyBoundScheduler.__init__ no longer accepts {sorted(missing_ctor_kwargs)} "
            "— dgemma.loop.run_diffusion constructs this scheduler with exactly these kwargs. "
            "Re-verify scheduling_entropy_bound.py's __init__ signature against the installed "
            "diffusers version and update dgemma/loop.py accordingly."
        )

    output_fields = getattr(EntropyBoundSchedulerOutput, "__dataclass_fields__", {})
    if "accepted_index" not in output_fields:
        raise RuntimeError(
            "diffusers structural probe failed (issue #35 R3): "
            "EntropyBoundSchedulerOutput no longer has an 'accepted_index' field — "
            "dgemma.loop._FrameCollector.on_step_end reads scheduler_output.accepted_index "
            "directly every step. Re-verify scheduling_entropy_bound.py's "
            "EntropyBoundSchedulerOutput dataclass against the installed diffusers version."
        )

    base_callback_inputs = set(getattr(DiffusionGemmaPipeline, "_callback_tensor_inputs", []))
    expected_base_inputs = {"canvas", "logits"}
    if not expected_base_inputs.issubset(base_callback_inputs):
        raise RuntimeError(
            "diffusers structural probe failed (issue #35 R3): "
            f"diffusers.DiffusionGemmaPipeline._callback_tensor_inputs is "
            f"{sorted(base_callback_inputs)}, missing {sorted(expected_base_inputs - base_callback_inputs)} "
            "— dgemma.loop.DGemmaPipeline widens this base allowlist by appending "
            "'scheduler_output' (pipeline_diffusion_gemma.py:76). Re-verify the base pipeline's "
            "allowlist against the installed diffusers version and update DGemmaPipeline's "
            "widened list accordingly."
        )


_check_diffusers_version()
_check_diffusers_structure()

from diffusers import DiffusionGemmaPipeline, EntropyBoundScheduler  # noqa: E402

from .composite import DiffusionCancelled, StepEndComposite  # noqa: E402
from .constraints_hook import build_logit_mask_hook  # noqa: E402
from .hooks import ForwardHookFn, install_logit_shaping_hook  # noqa: E402
from .ingress import validate_ingress  # noqa: E402
from .kv_cache import validate_kv_cache_ingress  # noqa: E402
from .participants import PinParticipant, WalkerParticipant  # noqa: E402
from .payloads import Constraints, ControlSignals  # noqa: E402
from .types import CanvasState, CanvasTrace, DGemmaModel, DiffusionFrame, KVCache, Provenance  # noqa: E402

# Grounded defaults (CLAUDE.md / plan.md — first local run, Q4_K_M).
DEFAULT_NUM_INFERENCE_STEPS = 48
DEFAULT_T_MIN = 0.4
DEFAULT_T_MAX = 0.8
DEFAULT_ENTROPY_BOUND = 0.1
DEFAULT_GEN_LENGTH = 256
DEFAULT_CONFIDENCE = 0.005

# ONE-MINT terms-and-units vocabulary (units-glossary-tooltips work item):
# every knob's units/semantics live HERE, once, and every door that surfaces
# a knob — ComfyUI widget tooltips (`surfaces/comfyui/sampler.py`), the MCP
# `generate`/`load_model` JSON-schema `description`s (`surfaces/mcp/commands/
# generate.py`, `surfaces/mcp/commands/model.py`) — imports and re-uses this
# text rather than re-typing it (rule-8 parity by construction: two doors
# describing one knob can't drift apart if there is only one string). This is
# the doctrine's `EMIT-CANONICAL` discipline applied to prose, not just
# payloads: a tooltip and a schema description are two renderings of the same
# canonical fact, not two independent claims that happen to agree today.
#
# Provenance for the units claims themselves (authoritative, transcribed from
# the operator's terms-and-units brief, not re-derived here):
#
# - `T` (sampling temperature — the WIDGET value users think of as "the
#   temperature") is the divisor in `softmax(z / T)`: a dimensionless scale
#   factor on the model-native logit calibration, `T=1` reproducing the
#   trained calibration exactly. Applied ONCE per step, upstream of BOTH
#   candidate sampling and the acceptance-entropy computation — there is no
#   second, independently-tunable temperature hiding downstream of it.
# - `t` (schedule position — NOT a temperature, despite the shared letter) is
#   `(N - step_idx) / N`: dimensionless, DECREASING from 1 down toward (but
#   never reaching) `1/N` as `step_idx` runs 0..N-1. See `anneal_temperature`
#   above for the exact formula this recomputes.
# - `t_min`/`t_max` are TEMPERATURE endpoints (config knobs), not schedule
#   positions, despite the lowercase-`t` naming: `T = t_min + (t_max -
#   t_min) * t` (the affine map `anneal_temperature` evaluates). `t_min` is a
#   virtual endpoint no real step ever actually applies — `t` bottoms out at
#   `1/N`, never 0, so the coldest realized temperature is
#   `t_min + (t_max - t_min) / N`, strictly above `t_min` itself. These field
#   names come from the upstream `EntropyBoundScheduler` checkpoint config
#   (`scheduling_entropy_bound.py`) — do NOT rename them to something more
#   self-describing; that would desync this pack's kwargs from the installed
#   diffusers scheduler's own `.config` attribute names `_FrameCollector`
#   reads live (`effective_t_min`/`effective_t_max`, `dgemma/types.py`).
# - `entropy_bound` is the per-step joint acceptance budget in NATS:
#   `torch.distributions.Categorical.entropy()` (what this pack's capture
#   path and the scheduler both use) is natural-log entropy, not bits.
#   Default `0.1` nats. For scale: the 18-bits-per-position uniform-vocabulary
#   melt VISION.md opens with is `18 * ln(2) ≈ 12.48` nats — i.e. roughly two
#   orders of magnitude hotter than the default per-step acceptance budget,
#   not directly comparable to it (one is the corruption entropy of the
#   INITIAL canvas draw; the other is a per-step ACCEPTANCE threshold), but
#   sharing the same nats unit is what makes that comparison meaningful at
#   all rather than a bits-vs-nats category error.
# - `confidence` is the pipeline's early-stop threshold: a dimensionless
#   probability (not a unit-bearing quantity at all).
KNOB_DOCS: dict[str, str] = {
    "t_min": (
        "Cold end of the temperature anneal (dimensionless, applied as the "
        "divisor T in softmax(z/T)). Despite the lowercase-t name this is a "
        "TEMPERATURE, not a schedule position — t bottoms out at 1/"
        "num_inference_steps, so t_min itself is a virtual endpoint no step "
        "actually reaches. T = t_min + (t_max - t_min) * t, t decreasing 1 -> "
        "1/num_inference_steps across the run."
    ),
    "t_max": (
        "Hot end of the temperature anneal (dimensionless, same softmax(z/T) "
        "divisor as t_min) — the temperature applied at the very first step, "
        "where the schedule position t == 1."
    ),
    "entropy_bound": (
        "Per-step joint acceptance budget, in NATS (natural-log entropy, "
        "matching torch.distributions.Categorical.entropy() — not bits). "
        "A position commits this step only once its acceptance entropy "
        "clears this bound. Default 0.1 nats; for scale, the uniform-vocab "
        "noise draw's 18 bits/position is ~12.48 nats."
    ),
    "confidence": (
        "Early-stop threshold: a dimensionless probability the pipeline's "
        "adaptive-stop check compares a candidate's confidence against."
    ),
    "num_inference_steps": (
        "Requested denoising step budget N (a plain count, not a physical "
        "unit) — the schedule-position denominator: t = (N - step_idx)/N."
    ),
    "gen_length": (
        "Canvas length in tokens (a token count) — how many positions the "
        "denoising loop allocates for the generated turn."
    ),
    "seed": (
        "RNG seed for the generator driving canvas initialization/renoise "
        "(a plain integer, not unit-bearing). Omit/leave unset for a "
        "nondeterministic run."
    ),
    "thinking": (
        "EXPERIMENTAL boolean toggle: injects the <|think|> control token "
        "via a system turn. Structurally one token short of native "
        "enable_thinking=True (the chat template's `| trim` eats the "
        "newline after <|think|>) — see run_diffusion's own docstring for "
        "the honest gap. Behavioral impact unverified pending an E2E "
        "thinking-mode run on real weights."
    ),
}

# ONE-MINT provenance (issue #8 / model-card "thinking" toggle): these
# literal strings are the DiffusionGemma tokenizer's control tokens, sourced
# from `google/diffusiongemma-26B-A4B-it`'s `tokenizer_config.json`
# (`model_specific_special_tokens`: `think_token="<|think|>"`,
# `soc_token="<|channel>"`, `eoc_token="<channel|>"`), cross-checked against
# the cached `tokenizer.json` `added_tokens` table (2026-07-05): id 98
# (`<|think|>`), id 100 (`<|channel>`), id 101 (`<channel|>`). The chat
# template's `<|channel>thought\n...content...\n<channel|>` framing (see
# `chat_template.jinja`) is what issue #8 excises. `THOUGHT_CHANNEL_START_ID`/
# `THOUGHT_CHANNEL_END_ID` are the fallback only — `resolve_thought_channel_ids`
# prefers reading them off the loaded processor's own tokenizer vocab, so a
# checkpoint swap that renumbers ids can't silently desync from a hardcoded
# pair.
THINK_TOKEN = "<|think|>"
THOUGHT_CHANNEL_START_TOKEN = "<|channel>"
THOUGHT_CHANNEL_END_TOKEN = "<channel|>"
THOUGHT_CHANNEL_START_ID = 100
THOUGHT_CHANNEL_END_ID = 101

# Provenance: `chat_template.jinja` always renders the channel as
# `'<|channel>thought\n' + thinking_text + '\n<channel|>'` — "thought" is a
# fixed channel-NAME label the template emits before any real content, not
# part of the reasoning text itself. Verified against the installed
# tokenizer (`AutoTokenizer.from_pretrained`, cached weights, 2026-07-05):
# decoding ids `[45518, 107]` (the label's own ids) with
# `skip_special_tokens=True` yields exactly `"thought\n"` — confirming the
# canonical *empty* channel (issue #8's `[100, 45518, 107, 101, ...]`) is the
# label with nothing after it. String-level label strip (not a special
# token — ordinary vocab), applied only to the already id-isolated
# between-delimiter span, never to the full decoded payload.
THOUGHT_CHANNEL_LABEL = "thought"


class DGemmaPipeline(DiffusionGemmaPipeline):
    """`DiffusionGemmaPipeline` subclass widening the per-step callback allowlist.

    The ONLY change from the base pipeline: `_callback_tensor_inputs` gains
    `"scheduler_output"`. The base class allowlist is `["canvas", "logits"]`
    (`pipeline_diffusion_gemma.py:76`); `check_inputs` validates
    `callback_on_step_end_tensor_inputs` against `self._callback_tensor_inputs`
    (`:155-161`), and the callback-kwargs extraction is generic —
    `callback_kwargs[k] = locals()[k]` (`:404-405`) — not a hardcoded
    two-key dispatch. Widening the allowlist here is therefore enough to hand
    the callback the full scheduler `.step()` output object (`accepted_index`,
    `sampled_probs`, `pred_logits`, ...) with no method override needed
    (ADR-CDG-004, resolved open question (a)).

    Caveat carried from that resolution: `"accepted_index"` alone is NOT a
    valid key — it is not a bound local in `__call__`'s scope. Only the
    `scheduler_output` container is.
    """

    _callback_tensor_inputs = ["canvas", "logits", "scheduler_output"]


def anneal_temperature(
    step_idx: int, num_inference_steps: int, t_min: float, t_max: float
) -> tuple[float, float]:
    """Replicate `EntropyBoundScheduler.step()`'s inlined anneal formula.

    Source: `scheduling_entropy_bound.py:153-155` (installed diffusers
    0.39.0) — the formula is inlined directly in `step()`, not exposed on
    `EntropyBoundSchedulerOutput`, so this dgemma layer recomputes it from the
    same inputs rather than reading it off the scheduler.

    Enforcement surface for this replication (issue #35 R3 / PR #48 gate
    finding F-1): `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin`
    — the structural probe above cannot see a formula-*body* change, so that
    test recovers the temperature the real installed scheduler's `step()`
    actually applied and asserts this function matches it. If you edit this
    formula, that test is the contract you are editing against.

    Returns `(t, temperature)` where `t` is the normalized schedule fraction
    (1.0 at the hottest/first step, decreasing toward but not reaching 0) and
    `temperature = t_min + (t_max - t_min) * t`.
    """
    t = (num_inference_steps - step_idx) / num_inference_steps
    temperature = t_min + (t_max - t_min) * t
    return t, temperature


def _build_pinned_mask(constraints: "Constraints | None", canvas: Any) -> Any | None:
    """Derive `DiffusionFrame.pinned_mask` from a validated `Constraints`
    payload (ADR-CDG-010 Decision 4, issue #64 Phase 2, gate correction A1).

    Static-from-`Constraints.pins` by construction: a boolean tensor shaped
    like one example's canvas (`canvas.shape[-1]`), `True` at every
    `pin.position`. Valid only because and only while pins are
    position-static (see `DiffusionFrame.pinned_mask`'s docstring for the
    full scope-guard reasoning) — no pin participant exists yet (Phase 3), so
    this reflects "which cells WOULD be pinned", not an observed write.

    `None` when `constraints` is `None` or carries no pins (`Constraints()`
    default, or an explicit `Constraints(pins=())`) — additive-optional
    discipline (ADR-CDG-014 Decision 1): absence, never an all-`False` mask
    standing in for "no pins".
    """
    if constraints is None or not constraints.pins:
        return None
    canvas_len = canvas.shape[-1]
    mask = torch.zeros(canvas_len, dtype=torch.bool)
    for pin in constraints.pins:
        mask[pin.position] = True
    return mask


@dataclass
class _FrameCollector:
    """Per-step frame collector driving `callback_on_step_end`.

    Pure with respect to the diffusers pipeline: reads only the callback's
    own contract (`pipe, global_step, step_idx, callback_kwargs`) plus the
    scheduler config values needed for `anneal_temperature`, so it is
    unit-testable with a fake `scheduler_output` (and, for the denominator,
    a fake scheduler exposing a `num_inference_steps` attribute) and no real
    pipeline (`tests/test_frames.py`).

    `num_inference_steps` (issue #20): NOT the user-requested value — a
    scheduler-like object read *lazily*, once per callback, via
    `.num_inference_steps`. Grounded against the installed diffusers 0.39.0
    pipeline (`pipeline_diffusion_gemma.py:280-297`): `set_timesteps(
    predictor_steps, ...)` runs at pipeline entry, before the per-step loop
    (`:356`) that fires `callback_on_step_end`, and `EntropyBoundScheduler.
    set_timesteps` (`scheduling_entropy_bound.py:87-91`) — and
    `BlockRefinementScheduler.set_timesteps`, `scheduling_block_refinement.py
    :83-100` — both reassign `self.num_inference_steps = num_inference_steps`
    there, the exact attribute `step()`'s inlined anneal formula divides by
    (`scheduling_entropy_bound.py:153`). So by the first callback the
    scheduler's own attribute already holds the *effective* denominator
    (`predictor_steps`, which differs from the user's `num_inference_steps`
    whenever a corrector scheduler folds `corrector_steps` sweeps into the
    same budget — `pipeline_diffusion_gemma.py:284-290`). Reading it lazily
    (not caching the value at collector-construction time, before the
    pipeline has called `set_timesteps`) is required: the collector is built
    by `run_diffusion` before `pipeline(...)` runs (this module, below), so a
    constructor-time snapshot would still be the stale user-requested count.
    Plain `EntropyBoundScheduler` (no `corrector_steps`) leaves
    `predictor_steps == num_inference_steps`, so this path is unchanged for
    today's only scheduler — the bug is latent, not yet observable, exactly
    per ADR-CDG-001's greenfield-exception framing (CLAUDE.md).

    `keep_frames="last"` (P1 default) retains only the most recent frame —
    memory policy, not a change in what gets computed per step; `"all"`
    retains every frame (the seam P3's `CanvasTrace` grows into). `steps_used`
    counts every step regardless of retention policy.

    `on_frame`, when given, is invoked once per captured step with the
    freshly built `DiffusionFrame` — regardless of `keep_frames` (a caller
    watching every step live still wants a callback even under `"last"`
    retention, which only governs what's kept afterward). Pure w.r.t.
    ComfyUI (ADR-CDG-003): this collector never imports or touches
    `PromptServer` itself — that's `nodes/sampler.py`'s closure, built and
    passed in from the node layer. `on_frame` runs after the retention
    policy is applied, so a callback exception never loses the frame itself.

    Engine contract on `on_frame` exceptions (deliberate, review finding
    2026-07-05): they PROPAGATE. The engine does not swallow a caller's
    callback error — a user's analysis callback silently eaten here would
    be its own dishonesty. A callback whose failure must not kill the run
    (e.g. a display-only push) guards itself at its own layer; that is what
    `nodes/sampler.py`'s live-push closure does.

    `canvas_idx` tracking: the pipeline's `step_idx` resets to 0 for each
    canvas/block (inner denoising loop nested in the outer canvas loop,
    `pipeline_diffusion_gemma.py:318,356`), and the callback contract carries
    no block coordinate of its own — so the collector infers it: a
    non-increasing `step_idx` between consecutive callbacks means a new block
    began. Detection is `step_idx <= previous`, not `step_idx == 0`, so a
    future mid-schedule start (variation runs, `loose-ends.md`) whose first
    step_idx is nonzero still registers as a new block.

    **Effective-knob telemetry (ADR-CDG-011 clause 7, issue #64 Phase 2):**
    `entropy_bound`/`t_min`/`t_max` are read fresh off `self.scheduler.config`
    on every callback — the same "never cached, always effective" discipline
    issue #20 already established for `num_inference_steps` above, extended
    to the three walker-mutable knobs (ADR-CDG-011's `MUTABLE_TARGETS`). No
    walker exists yet to write through them (Phase 4, `NOT-YET-IMPLEMENTED`),
    but reading live now — rather than only once a walker lands — is what
    makes a future walker bug that silently fails to write through visible
    in the trace the day it ships, instead of requiring a second migration of
    this read site. The ctor `t_min`/`t_max` fields remain: they are the
    values `anneal_temperature` falls back to when `self.scheduler` exposes
    no `.config` at all (a bare unit-test double lighter than the real
    scheduler/R4 fixture) — a named degradation, not a raise, mirroring
    `resolve_vocab_size`'s stub fallback. Every real `EntropyBoundScheduler`
    and the R4 `FakeEntropyBoundScheduler` fixture expose `.config`, so this
    fallback is exercised only by pre-R4-style bare test doubles.

    **`pinned_mask` (ADR-CDG-010 Decision 4, issue #64 Phase 2, gate
    correction A1):** derived once at construction from `constraints.pins`
    when a `Constraints` payload is supplied — `None` otherwise. No pin
    participant exists yet (Phase 3), so this is the validated-then-ignored
    payload's positions read directly, not an observed per-step write. Valid
    **only because and only while** pins are position-static (the D6
    hard-pin invariant: a hard pin re-asserts the same positions every step,
    so the pinned-position set is provably constant for the whole run) — see
    `DiffusionFrame.pinned_mask`'s docstring for the full A1 scope-guard
    reasoning and the labeled door for a future dynamic/re-pinning constraint
    type.
    """

    scheduler: Any
    """Object exposing a `.num_inference_steps` attribute (the real
    `EntropyBoundScheduler`/`BlockRefinementScheduler`, or a fake in tests) —
    read fresh on every callback, never cached, so the collector always
    reflects the scheduler's *effective* post-`set_timesteps` value (issue
    #20; see this class's docstring)."""

    t_min: float
    """Fallback anneal `t_min` used only when `self.scheduler` exposes no
    `.config.t_min` (see the class docstring's effective-knob-telemetry
    section) — otherwise superseded every callback by the live config read."""

    t_max: float
    """Fallback anneal `t_max`, same fallback-only role as `t_min` above."""

    keep_frames: Literal["last", "all"] = "last"
    on_frame: Callable[[DiffusionFrame], None] | None = None
    constraints: "Constraints | None" = None
    """ADR-CDG-010 Decision 4 / issue #64 Phase 2: the validated `Constraints`
    payload (or `None`), used only to derive each frame's static `pinned_mask`
    at construction time — see the class docstring's `pinned_mask` section.
    Not otherwise read; no participant consumes this yet (Phase 3)."""

    top_k: int = 0
    """ADR-CDG-014 Decision 3 Tier 1 (issue #61 P-B): the validated
    `CaptureSpec.top_k` value (or `0`), read fresh in `on_step_end` to derive
    `DiffusionFrame.top_k_ids`/`top_k_weights` from the same pre-pin `logits`
    Tier 0's `entropy` derives from. `0` (default) leaves both fields `None`
    (additive-optional absence, Decision 1/2) — byte-identical to every run
    before this phase."""

    capture_full_distribution: bool = False
    """ADR-CDG-014 Decision 3 Tier 2 (issue #61 P-C): the validated
    `CaptureSpec.capture_full_distribution` value (or `False`). `False`
    (default) leaves `DiffusionFrame.distribution` `None` on every frame
    (additive-optional absence, Decision 1/2) — byte-identical to every run
    before this phase. `True` derives `distribution = softmax(logits)` from
    the same pre-pin `logits` Tier 0/1 already read, subject to
    `max_full_distribution_steps`'s retention budget (Decision 5) below."""

    max_full_distribution_steps: int | None = None
    """ADR-CDG-014 Decision 3/5 Tier 2's budget (issue #61 P-C): the
    validated `CaptureSpec.max_full_distribution_steps` value. Caps the
    number of CAPTURED steps (in step order, counted by `self.steps_used`
    at the moment each callback fires — i.e. the first
    `max_full_distribution_steps` calls to `on_step_end`) whose frame
    retains a populated `distribution`; every step beyond the budget gets
    `distribution=None` on the RETAINED frame, regardless of `keep_frames`
    (Decision 5 — the budget caps retention, not the live stream).
    `on_frame`, when given, still receives every frame's `distribution` live
    while Tier 2 is on and the step is within budget — Decision 5's "a
    streaming consumer that does not retain gets the full stream" clause
    applies to whichever steps actually computed a distribution; a
    consumer's `on_frame` never sees a *different* value than what the
    matching retained frame holds for that same step. `None` (default)
    means "no budget declared"; ingress (`dgemma.ingress.validate_capture`)
    already rejects `capture_full_distribution=True` with no budget, so a
    live `_FrameCollector` never actually reaches `capture_full_distribution
    =True, max_full_distribution_steps=None` through `run_diffusion` — this
    field still defaults to `None` for callers driving the collector
    directly in tests, where `capture_full_distribution=False` makes the
    budget irrelevant."""

    frames: list[DiffusionFrame] = field(default_factory=list)
    steps_used: int = 0
    _canvas_idx: int = -1
    _prev_step_idx: int | None = None
    _pinned_mask: Any | None = field(default=None, init=False, repr=False)
    _pinned_mask_built: bool = field(default=False, init=False, repr=False)

    def on_step_end(self, pipe: Any, global_step: int, step_idx: int, callback_kwargs: dict) -> dict:
        """`callback_on_step_end(pipe, global_step, step_idx, callback_kwargs)`.

        Pure capture (P1): never overwrites the canvas, so it always returns
        `{}` — `callback_outputs.pop("canvas", canvas)` at the call site
        (`pipeline_diffusion_gemma.py:407`) then leaves the canvas unchanged.
        Mid-loop constraint injection (P5) is a different callback that
        returns `{"canvas": ...}`.

        Raises `ValueError` on a zero-length block: `accepted_index` with
        block dim 0 would make the per-example mean NaN, and a NaN
        committed_fraction would silently read as not-converged downstream —
        degenerate input is surfaced, not laundered into a validity field.

        **Tier 0 entropy capture (ADR-CDG-014 Decision 3/4, issue #14):**
        this method IS the composite's `capture` participant, which runs
        FIRST in the fixed order (`capture -> cancel -> beta-rebuild -> pin`,
        ADR-CDG-010) — so `callback_kwargs["logits"]`, when present, is the
        model's pre-pin predictive distribution for this step, never a
        post-pin/post-constraint artifact. `DiffusionFrame.entropy` is
        always populated when `logits` is reachable (the always-on Tier 0
        default); `None` only when a caller drives this collector directly
        without requesting `logits` in `callback_on_step_end_tensor_inputs`
        (additive-optional discipline — absence, never a zero-valued
        stand-in, ADR-CDG-014 Decision 1/2).

        **Effective-knob telemetry (ADR-CDG-011 clause 7, issue #64 Phase
        2):** `entropy_bound`/`t_min`/`t_max` are read off `self.scheduler.
        config` fresh THIS callback — the values `step()` actually consumed
        producing this frame — falling back to the ctor `self.t_min`/
        `self.t_max` (and `None` for `entropy_bound`, which has no ctor
        fallback) only when `self.scheduler` exposes no `.config` at all.
        `t`/`temperature` are recomputed from the live `t_min`/`t_max`, so a
        walker-mutated anneal range (Phase 4) is reflected consistently
        across `t`/`temperature` and the `effective_*` fields together.

        **`pinned_mask` (ADR-CDG-010 Decision 4, issue #64 Phase 2):** built
        once, lazily, from `self.constraints.pins` on the first callback and
        reused for every subsequent frame this run — see the class
        docstring's `pinned_mask` section for the A1 scope-guard reasoning.
        `None` when no `Constraints` payload was supplied.

        **Tier 1 top-k capture (ADR-CDG-014 Decision 3, issue #61 P-B):**
        when `self.top_k > 0` and `logits` is reachable, `DiffusionFrame.
        top_k_ids`/`top_k_weights` are derived from the SAME pre-pin
        `logits` `entropy` reads (`logits.topk(k)` for ids/raw scores,
        `softmax` over just those k logits for weights — a per-position
        renormalization over the top-k slice, not the full-vocab softmax,
        since Tier 1 never materializes the full distribution) — so Tier 1
        inherits Tier 0's capture-pre-pin ordering guarantee for free, not a
        second derivation that could drift from it. `top_k=0` (default)
        leaves both fields `None` (additive-optional absence, Decision 1/2),
        matching every run before this phase byte-for-byte.

        **Tier 2 full-distribution capture (ADR-CDG-014 Decision 3/5, issue
        #61 P-C):** when `self.capture_full_distribution` is `True`,
        `logits` is reachable, AND this callback's step is still within
        `self.max_full_distribution_steps`'s budget (`self.steps_used`,
        read BEFORE incrementing — i.e. this is the Nth call, budget counts
        calls 0..budget-1), `DiffusionFrame.distribution` is
        `softmax(logits, dim=-1)` over the SAME pre-pin per-position logits
        entropy/top-k already read — one derivation, not a third drifting
        copy. Once the budget is exhausted, `distribution` stays `None` on
        every subsequent frame for the rest of the run (Decision 5: the
        budget caps *retained* frames regardless of `keep_frames`) — Tier 0/
        Tier 1 fields are completely unaffected by the Tier-2 budget running
        out (they have their own independent policies). `capture_full_
        distribution=False` (default) leaves `distribution` `None`
        unconditionally, byte-identical to every run before this phase.
        """
        scheduler_output = callback_kwargs["scheduler_output"]
        canvas = callback_kwargs["canvas"]

        accepted_index = scheduler_output.accepted_index
        if accepted_index.shape[-1] == 0:
            raise ValueError(
                "Degenerate scheduler_output: accepted_index has block length 0 "
                f"(shape {tuple(accepted_index.shape)}); committed_fraction would be NaN."
            )

        if self._prev_step_idx is None or step_idx <= self._prev_step_idx:
            self._canvas_idx += 1
        self._prev_step_idx = step_idx

        config = getattr(self.scheduler, "config", None)
        effective_t_min = getattr(config, "t_min", self.t_min) if config is not None else self.t_min
        effective_t_max = getattr(config, "t_max", self.t_max) if config is not None else self.t_max
        effective_entropy_bound = getattr(config, "entropy_bound", None) if config is not None else None

        t, temperature = anneal_temperature(
            step_idx, self.scheduler.num_inference_steps, effective_t_min, effective_t_max
        )
        # Mean over the block dim ONLY — one fraction per example, never a
        # batch-blended scalar (review finding, 2026-07-05).
        committed_per_example = tuple(accepted_index.float().mean(dim=-1).tolist())

        entropy = None
        top_k_ids = None
        top_k_weights = None
        distribution = None
        logits = callback_kwargs.get("logits")
        if logits is not None:
            entropy = torch.distributions.Categorical(logits=logits).entropy()
            if entropy.dim() == 2:
                # `logits` may be `[batch, canvas_len, vocab]` (real
                # pipeline) or already `[canvas_len, vocab]` (some fake
                # fixtures) — single-example scope (ADR-CDG-014 Open
                # Questions: batched capture deliberately deferred to a
                # P4+ design pass), so batch index 0 is what every existing
                # single-example consumer expects.
                entropy = entropy[0]

            # Same batch-squeeze as entropy above, applied to logits itself
            # once — shared by Tier 1 (top-k) and Tier 2 (full distribution)
            # so both derive from the identical per-position row entropy
            # just read (one normalization, not drifting copies, ADR-CDG-014
            # Decision 4).
            per_position_logits = logits[0] if logits.dim() == 3 else logits

            if self.top_k > 0:
                top_k_values, top_k_ids = per_position_logits.topk(self.top_k, dim=-1)
                # Renormalize over just the top-k slice (a per-position
                # softmax restricted to the k candidates already selected) —
                # Tier 1 never materializes the full-vocab softmax (that is
                # Tier 2's `distribution` field below, budget-gated). This is
                # the top-k conditional distribution, not an approximation of
                # the full one; a consumer reading `top_k_weights` as
                # anything other than "renormalized over these k ids" would
                # be reading past what Tier 1 actually captured.
                top_k_weights = torch.softmax(top_k_values, dim=-1)

            if self.capture_full_distribution:
                # Budget check (ADR-CDG-014 Decision 3/5, issue #61 P-C):
                # `self.steps_used` is this callback's 0-indexed ordinal
                # (read BEFORE the increment below), so the budget retains
                # the FIRST `max_full_distribution_steps` captured steps —
                # `max_full_distribution_steps=None` (no budget declared,
                # only reachable when a caller drives the collector
                # directly rather than through `run_diffusion`'s ingress
                # gate) is treated as "no cap", matching
                # `capture_full_distribution`'s own unconditional meaning in
                # that direct-use case.
                budget = self.max_full_distribution_steps
                if budget is None or self.steps_used < budget:
                    # Full per-position softmax — Tier 2's ~134 MB/step
                    # payload (ADR-CDG-014 Decision 3's Tier-2 row).
                    distribution = torch.softmax(per_position_logits, dim=-1)

        if not self._pinned_mask_built:
            self._pinned_mask = _build_pinned_mask(self.constraints, canvas)
            self._pinned_mask_built = True

        frame = DiffusionFrame(
            canvas_idx=self._canvas_idx,
            step_idx=step_idx,
            t=t,
            temperature=temperature,
            committed_fraction_per_example=committed_per_example,
            canvas=canvas,
            entropy=entropy,
            top_k_ids=top_k_ids,
            top_k_weights=top_k_weights,
            distribution=distribution,
            pinned_mask=self._pinned_mask,
            effective_entropy_bound=effective_entropy_bound,
            effective_t_min=effective_t_min,
            effective_t_max=effective_t_max,
        )
        self.steps_used += 1
        if self.keep_frames == "last":
            self.frames[:] = [frame]
        else:
            self.frames.append(frame)
        if self.on_frame is not None:
            self.on_frame(frame)
        return {}


def derive_canvas_state(
    *,
    text: str,
    canvas_ids: Any,
    frames: list[DiffusionFrame],
    steps_used: int,
    thought: str | None = None,
    stray_thought_delimiter: bool = False,
    eos_token_id: int | None = None,
) -> CanvasState:
    """Derive `CanvasState`'s validity fields from the captured frames.

    See `CanvasState.converged`'s docstring for what "converged" honestly
    does and does not claim. `thought` and `stray_thought_delimiter`
    (issue #8) are passed through unmodified — the excised thought-channel
    content (or `None`) and the stray-delimiter anomaly flag from
    `excise_thought_channel`.

    `turn_closed`/`answer_tokens` (issue #9, severable rider): reuses the
    excision/decode machinery already in `run_diffusion` rather than
    capturing anything new. `turn_closed` is `True` iff `eos_token_id` is
    given and appears somewhere in `canvas_ids`: EOS was actually committed
    inside the generated region, as opposed to the canvas simply running out
    (`gen_length` reached with no EOS ever emitted) — the exact honesty gap
    issue #9 named, independent of `converged` (a run can converge on
    non-EOS filler once the canvas is full).

    `answer_tokens` counts the (thought-excised) ids **before the first
    EOS**, mirroring `_decode_ids`'s own trim: `canvas_ids` is not
    eos-trimmed, and a converged run pads the rest of the canvas with a
    trailing EOS/renoise fill run (observed live, ~30 tokens), so a bare
    `len(canvas_ids)` would inflate the count by that padding — defeating
    the honesty purpose the field exists for (review finding, 2026-07-05).
    The EOS token itself is deliberately NOT counted: it is the stop signal,
    not answer content. When no EOS is present the full (thought-excised)
    length is the honest count — every id is content the budget-truncated
    canvas actually holds. `0` when `canvas_ids` is `None` (the existing
    unit-test call shape, which never asserts on this field).
    """
    if not frames:
        raise RuntimeError("No frames captured — the denoising callback never fired.")
    last = frames[-1]
    if canvas_ids is not None:
        ids = [int(x) for x in canvas_ids]
        turn_closed = eos_token_id is not None and eos_token_id in ids
        answer_tokens = ids.index(eos_token_id) if turn_closed else len(ids)
    else:
        turn_closed = False
        answer_tokens = 0
    return CanvasState(
        text=text,
        canvas_ids=canvas_ids,
        converged=last.committed_fraction >= 1.0,
        committed_fraction=last.committed_fraction,
        steps_used=steps_used,
        thought=thought,
        stray_thought_delimiter=stray_thought_delimiter,
        turn_closed=bool(turn_closed),
        answer_tokens=answer_tokens,
    )


def resolve_vocab_size(processor: Any) -> int | None:
    """Resolve a vocab size for `dgemma.ingress.validate_constraints`'s C3
    check (issue #64 §3.4), off `processor`'s tokenizer.

    Same tokenizer-unwrap path `resolve_thought_channel_ids` uses
    (`getattr(processor, "tokenizer", processor)`). Tries `len(tokenizer)`
    first (the usual `PreTrainedTokenizerBase.__len__`), then
    `tokenizer.vocab_size`. Returns `None` — a named degradation, not a
    raise — when neither is available (e.g. a bare stub in a unit test that
    exposes no vocab at all), mirroring `resolve_thought_channel_ids`'s own
    stub fallback: C3 is skipped rather than this resolver inventing a size.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    try:
        return len(tokenizer)
    except TypeError:
        pass
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int):
        return vocab_size
    return None


def resolve_thought_channel_ids(processor: Any) -> tuple[int, int]:
    """Resolve the (start, end) thought-channel delimiter ids from `processor`.

    Prefers reading them off the tokenizer's own vocab
    (`convert_tokens_to_ids`) so a checkpoint swap that renumbers special
    tokens can't silently desync from a hardcoded pair; falls back to the
    module-level `THOUGHT_CHANNEL_START_ID`/`THOUGHT_CHANNEL_END_ID`
    constants (provenance: `tokenizer_config.json`, see the comment above
    their definition) when `processor` doesn't expose a usable tokenizer —
    e.g. a bare stub in a unit test, or an `unk_token` fallback signaling the
    strings aren't in this vocab at all.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return THOUGHT_CHANNEL_START_ID, THOUGHT_CHANNEL_END_ID

    start_id = convert(THOUGHT_CHANNEL_START_TOKEN)
    end_id = convert(THOUGHT_CHANNEL_END_TOKEN)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        start_id is None
        or end_id is None
        or (unk_id is not None and (start_id == unk_id or end_id == unk_id))
    ):
        return THOUGHT_CHANNEL_START_ID, THOUGHT_CHANNEL_END_ID
    return start_id, end_id


@dataclass
class ThoughtChannelExcision:
    """Result of `excise_thought_channel` — named fields instead of a
    positional tuple, because the excision reports three independent things
    (the cleaned ids, zero-or-more excised spans, a stray-delimiter anomaly
    flag) and a growing anonymous tuple is how call sites silently misread
    which position means what."""

    remaining_ids: list[int]
    """Canvas ids with every well-formed thought-channel span (and, when
    applicable, the truncated turn-start frame) removed. Feeds the answer
    `STRING`."""

    thought_spans: list[list[int]]
    """Delimiter-exclusive content ids of each excised span, in canvas
    order. Empty list when no channel was present; an individual span may
    itself be `[]` (a zero-content frame)."""

    stray_start_delimiter: bool = False
    """`True` iff an unmatched `start_id` was found PAST the head of the
    generated region and therefore left in place rather than excised
    (excising it would silently destroy answer text — see
    `excise_thought_channel`). Surfaced so `CanvasState` can report the
    anomaly instead of the payload absorbing it invisibly."""


def excise_thought_channel(
    canvas_ids: Any,
    start_id: int = THOUGHT_CHANNEL_START_ID,
    end_id: int = THOUGHT_CHANNEL_END_ID,
) -> ThoughtChannelExcision:
    """Excise every thought-channel span from a canvas id sequence (issue #8).

    Pure id-level operation (ADR-CDG-001 payload-contamination discipline:
    id-span excision over decoded-string regex). The model emits
    `<|channel>thought\\n<channel|>` (empty channel — expected even with
    thinking off, per the model card) or `<|channel>...content...<channel|>`
    (non-empty, thinking on) at turn start; upstream
    `batch_decode(..., skip_special_tokens=True)` strips only the id-100/
    id-101 delimiters themselves, leaving `thought`/`\\n`/content — ordinary
    vocab tokens — to survive into the decoded string.

    Accepts a `torch.LongTensor`, a `list[int]`, or any 1-D iterable of ids;
    `remaining_ids`/`thought_spans` hold plain Python ints (never torch
    scalars), so downstream `tokenizer.decode` calls get plain id lists.

    Behavior, by case:
    - No `start_id` anywhere -> nothing excised, `thought_spans == []` —
      the false-strip guard: content that merely *mentions* "thought" as
      ordinary vocab is left untouched.
    - Each well-formed `start_id ... end_id` pair -> both delimiters and
      everything between them are removed from `remaining_ids`; the
      delimiter-exclusive content (possibly `[]`) is appended to
      `thought_spans`. ALL well-formed spans are excised, not just the
      first — a second leaked frame is the same ADR-CDG-001 breach as the
      first (review finding, 2026-07-05).
    - Unmatched `start_id` (no `end_id` anywhere after it) **at the head of
      the generated region** (index 0 — the documented turn-start frame
      position) -> treated as a truncated frame: excised through the end of
      the sequence, the tail going to `thought_spans`. No answer text can
      precede index 0, so nothing is lost but the broken frame.
    - Unmatched `start_id` **past the head** -> left in place untouched,
      along with everything after it — never silently truncate answer text.
      The raw delimiter stays in `remaining_ids` (where a
      `skip_special_tokens=True` decode drops the delimiter itself but keeps
      all surrounding answer text), and `stray_start_delimiter=True` is set
      so the anomaly surfaces on the `CanvasState` validity side rather
      than vanishing.
    """
    ids = [int(x) for x in canvas_ids]
    remaining: list[int] = []
    thought_spans: list[list[int]] = []
    stray_start_delimiter = False

    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            remaining.append(ids[i])
            i += 1
            continue
        try:
            end = ids.index(end_id, i + 1)
        except ValueError:
            if i == 0:
                # Truncated turn-start frame: excise-to-end loses nothing
                # but the broken frame.
                thought_spans.append(ids[1:])
            else:
                # Stray mid-canvas start delimiter: keep it and everything
                # after it — answer text is never silently dropped.
                stray_start_delimiter = True
                remaining.extend(ids[i:])
            break
        thought_spans.append(ids[i + 1 : end])
        i = end + 1

    return ThoughtChannelExcision(
        remaining_ids=remaining,
        thought_spans=thought_spans,
        stray_start_delimiter=stray_start_delimiter,
    )


def _decode_ids(processor: Any, ids: list[int], eos_token_id: int | None) -> str:
    """Decode `ids` the way the pipeline decodes `texts[0]`
    (`pipeline_diffusion_gemma.py:437-453`): trim at the first `eos_token_id`
    (inclusive) so post-EOS canvas-fill/renoise-garbage tokens don't leak in,
    then `skip_special_tokens=True`.

    Duplicated here rather than trusting the pipeline's own `output.texts[0]`
    because that value was decoded from the un-excised ids and still carries
    the thought-channel leak `excise_thought_channel` exists to remove; this
    re-derives the visible text from the corrected ids instead.
    """
    if eos_token_id is not None and eos_token_id in ids:
        ids = ids[: ids.index(eos_token_id) + 1]
    tokenizer = getattr(processor, "tokenizer", processor)
    return tokenizer.decode(ids, skip_special_tokens=True)


def _extract_thought_text(decoded_channel: str) -> str | None:
    """Strip the chat template's fixed `"thought\\n"` channel-name label
    (provenance: see `THOUGHT_CHANNEL_LABEL` above) from a decoded
    between-delimiter span, returning `None` when nothing real remains.

    The canonical empty channel decodes to exactly `"thought\\n"` — label,
    no content — which must surface as "no thought", not as a `CanvasState`
    field containing the literal word "thought".
    """
    stripped = decoded_channel
    if stripped.startswith(THOUGHT_CHANNEL_LABEL):
        stripped = stripped[len(THOUGHT_CHANNEL_LABEL) :]
    stripped = stripped.strip()
    return stripped or None


def decode_frames(processor: Any, frames: list[DiffusionFrame]) -> list[str]:
    """Decode each captured `DiffusionFrame.canvas` to a string, in frame
    order — the "flipbook" series (noise -> coherent text), the raw per-step
    view `tools/flipbook/flipbook.py` renders from the GGUF CLI, exposed here
    for the transformers backend (plan.md P3, node-level `frames` output).

    Deliberately RAW, unlike `_decode_ids`: `skip_special_tokens=True`, but
    NO eos-trim and NO thought-channel excision. Early frames are mostly
    noise and transient thought-channel delimiters — that IS the intended
    view; trimming or excising here would hide the evolution the flipbook
    exists to show. (Contrast `_decode_ids`, which trims at EOS and is fed
    post-excision ids — that's the *answer* text, a different concern.)

    `canvas` may be a 1-D `[canvas_len]` tensor or a 2-D `[batch, canvas_len]`
    tensor (`run_diffusion` is single-example/batch-1 today) — example 0 is
    decoded for a 2-D tensor. Ids are moved off-device and converted to a
    plain `list[int]` (`.tolist()`) before `tokenizer.decode`, so this works
    identically for a CPU/GPU tensor or a plain list/tuple already in test
    fixtures.

    `[]` when `frames` is empty.
    """
    tokenizer = getattr(processor, "tokenizer", processor)
    texts: list[str] = []
    for frame in frames:
        canvas = frame.canvas
        if hasattr(canvas, "dim") and canvas.dim() == 2:
            canvas = canvas[0]
        ids = canvas.tolist() if hasattr(canvas, "tolist") else list(canvas)
        texts.append(tokenizer.decode(ids, skip_special_tokens=True))
    return texts


def run_diffusion(
    dgemma_model: DGemmaModel,
    prompt: str,
    *,
    seed: int | None = None,
    gen_length: int = DEFAULT_GEN_LENGTH,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    entropy_bound: float = DEFAULT_ENTROPY_BOUND,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    confidence: float = DEFAULT_CONFIDENCE,
    thinking: bool = False,
    keep_frames: Literal["last", "all"] = "all",
    on_frame: Callable[[DiffusionFrame], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    logit_hook: ForwardHookFn | None = None,
    constraints: "Constraints | None" = None,
    control_signals: "ControlSignals | None" = None,
    capture: Any = None,
    kv_cache: "KVCache | None" = None,
) -> tuple[str, CanvasState, CanvasTrace]:
    """Drive one prompt through the block-diffusion denoising loop.

    Constructs `EntropyBoundScheduler` directly with the entropy/temperature
    config (`entropy_bound`, `t_min`, `t_max`, `num_inference_steps`) — these
    live on the scheduler config, NOT on the pipeline's `__call__` (ADR-CDG-004:
    the pipeline only forwards `generator`/`mask_token_id`/`temperature` to
    `scheduler.step()`, filtered by that scheduler's own signature, and
    `EntropyBoundScheduler.step()` doesn't accept `mask_token_id` or
    `temperature` at all — it anneals its own). Wraps the loaded model in
    `DGemmaPipeline` (direct-constructor idiom, not `.from_pretrained`, since
    the model is already loaded).

    `confidence` promotes the pipeline's `confidence_threshold` to a real
    parameter (P2). `stability_threshold`/`eos_early_stop` stay at the
    pipeline's own defaults (1 / True — already the grounded defaults,
    CLAUDE.md); P2 only promoted the knobs plan.md names for Phase 2.

    `thinking` (P2, model-card documented mechanism): when `True`, the
    `<|think|>` control token is injected at the start of the (otherwise
    empty) system turn by passing an explicit `messages=[{"role": "system",
    "content": THINK_TOKEN}, {"role": "user", "content": prompt}]`. This is
    the ONLY viable path here: the pipeline's `_prepare_inputs` never
    forwards `enable_thinking` (or any extra kwargs) to
    `apply_chat_template`, so the template's native toggle is unreachable
    through `pipeline.__call__`. **Honest delta, pinned by
    `tests/test_chat_template_thinking.py` against the real tokenizer
    (2026-07-05):** the injected path is NOT token-identical to the native
    `enable_thinking=True` render — the template emits system content
    through `| trim`, which eats the newline the native path places after
    `<|think|>`, so the injected render is exactly one token short (id 107,
    `"\\n"`, between `<|think|>` and `<turn|>`). Token parity is
    structurally unreachable via message content (any trailing whitespace is
    trimmed). Behavioral impact of the missing newline is unverified pending
    an E2E thinking-mode run; the `<|think|>` token itself (id 98) lands in
    the documented position either way. When `False` (default), `prompt` is
    passed bare — unchanged from P1, no system turn is added.

    Regardless of `thinking`, the thought channel the model emits at turn
    start (issue #8 — empty when off, per the model card's "an empty
    thinking channel might still be emitted"; possibly non-empty when on) is
    excised from the canvas ids via `excise_thought_channel` before `text`
    is derived, so it never leaks onto the `STRING` payload in either mode.

    `keep_frames` defaults to `"all"` (P3): per-step state here is small
    (ADR-CDG-005's own domain framing — a `gen_length`-length int64 canvas
    plus a per-example float per step), so retaining every step for the
    returned `CanvasTrace` isn't worth gating behind a toggle. `on_frame`,
    when given, is invoked once per captured step regardless of
    `keep_frames` — the seam that lets `nodes/sampler.py` push a live view
    without this module ever importing ComfyUI (ADR-CDG-003): the callback
    body that touches `PromptServer` lives in the node layer, not here.
    `on_frame` exceptions propagate (engine contract — see
    `_FrameCollector`'s docstring): a callback that must never kill the run
    guards itself, as the node layer's display-only closure does.

    `should_cancel` (issue #38, folded into R1's composer spec per the #35
    handoff): a zero-argument, surface-neutral predicate checked once per
    step by `dgemma.composite.StepEndComposite`, AFTER that step's capture
    (ADR-CDG-010 cancellation amendment 2026-07-13, PR #45) — surface-
    agnostic by construction (ARCHITECTURE.md rule 1): a ComfyUI surface
    wires this to `comfy.model_management`'s interrupt check, an MCP surface
    wires it to its own abort signal, and this module never imports either.
    When the predicate reports `True`, the composite raises
    `DiffusionCancelled`, caught here to return the PARTIAL
    `(text, CanvasState, CanvasTrace)` built from every frame captured so
    far — INCLUDING the cancelled step's own committed frame, the run's
    exact truncation point (the scheduler has already committed that step
    by `callback_on_step_end` time; see `dgemma/composite.py`'s module
    docstring) — evidence is returned, not raised away (#38's "a cancelled
    experiment run is still data" clause). `None` (default) means no
    cancellation wiring; the run always completes or raises a real error,
    exactly today's behavior.

    The single `callback_on_step_end` slot passed to the pipeline is a
    `dgemma.composite.StepEndComposite` (ADR-CDG-010 Decision 3 + its
    cancellation amendment), not the collector directly — the composite's
    fixed order is `capture -> cancellation check -> beta-rebuild -> pin ->
    walker`. `capture` and the cancellation seam are always wired; `pin` is
    wired (issue #64 Phase 3) with a fresh `PinParticipant` whenever
    `constraints=` carries at least one pin, `()` otherwise; `walker` is
    wired (issue #64 Phase 4) with a fresh `WalkerParticipant` whenever
    `control_signals=` carries at least one binding, `None` otherwise — so a
    run with no constraints/control_signals still builds an empty `pin=`
    tuple and a `None` `walker=`, and the composite's behavior is identical
    to invoking the collector alone, exactly as before either phase. The
    beta-rebuild participant (ADR-CDG-010) remains `NOT-YET-IMPLEMENTED` —
    Phase 5 lands that body; this phase only fills the `walker` slot the
    scaffold already exposed.

    `logit_hook` (#35 R5, F4; ADR-CDG-010 Decision 5): an optional forward
    hook installed on `dgemma_model.model` for exactly the duration of the
    one pipeline call below, via `dgemma.hooks.install_logit_shaping_hook` —
    the ONLY sanctioned installation path for a hook on this door (the only
    logit-shaping door per issue #28: a callback-returned `{"logits": ...}`
    is silently discarded by the installed pipeline). `None` when
    `constraints=` is also `None` installs nothing and leaves zero hooks
    registered, trivially satisfying `STATELESS-CORE`'s "no hook survives a
    `run_diffusion` call" (rule 6): the context manager's `try/finally`
    guarantees teardown on the pipeline call's clean return, on
    `DiffusionCancelled` (caught below), and on any other exception raised
    mid-run — the hook is torn down before this function's own exception
    handling (or return) is reached in every case. Passing BOTH
    `constraints=` and `logit_hook=` is rejected at ingress (H1, below) —
    two logit-mask sources on one door (ADR-CDG-010 D5).

    `constraints=`/`control_signals=`/`capture=` (ADR-CDG-010/011/014, issue
    #64/#61): declarative payloads, validated at ingress (`dgemma.ingress.
    validate_ingress`). `capture=`'s Tier 1 knob (`top_k`, ADR-CDG-014
    Decision 3, issue #61 P-B) is LIVE: when `capture.top_k > 0`, the
    `_FrameCollector` derives `DiffusionFrame.top_k_ids`/`top_k_weights` from
    the same pre-pin `logits` Tier 0's `entropy` reads — see `_FrameCollector.
    on_step_end`'s docstring. `capture=None`/`capture.top_k` absent/`0`
    (default) leaves both fields `None`, byte-identical to every run before
    that phase. `capture=`'s Tier 2 knobs (`capture_full_distribution`/
    `max_full_distribution_steps`, ADR-CDG-014 Decision 3/5, issue #61 P-C)
    are also LIVE: when `capture.capture_full_distribution=True`, the
    `_FrameCollector` derives `DiffusionFrame.distribution` (the full
    per-position `softmax(logits)`) from the same pre-pin `logits`, retained
    only for the first `capture.max_full_distribution_steps` captured steps
    — ingress rejects `capture_full_distribution=True` with no budget, so
    this call site never sees an unbounded request. `capture=None`/
    `capture.capture_full_distribution` absent/`False` (default) leaves
    `distribution` `None` on every frame, byte-identical to every run before
    P-C. `capture.keep_frames` remains validated-then-ignored (issue
    #64 P1, unchanged — see `dgemma/payloads.py:CaptureSpec`).
    `constraints=` is LIVE end-to-end (issue #64 Phase
    3, ADR-CDG-010's two-mechanism givens): when it carries at least one pin,
    `run_diffusion` (a) builds `dgemma.constraints_hook.build_logit_mask_hook`
    from the pins and installs it via the existing `logit_hook=`/
    `install_logit_shaping_hook` path — masking each pinned position's
    logits to its `token_id` so that cell reads ~zero entropy and commits
    first (Decision 1(a)); and (b) constructs a
    `dgemma.participants.PinParticipant` and wires it into the composite's
    `pin=` slot (Decision 3's LAST writer), re-asserting every pin's
    `token_id` at its `position` on every step regardless of what the
    scheduler accepted (Decision 1(b)) — the mechanism that guarantees *what
    conditions* the next forward pass, since a real scheduler step renoises
    every rejected position over the full vocabulary (no absorbing mask,
    ADR-CDG-001) and a given re-checked only at ingress would drift the
    first time its cell isn't accepted. `Constraints(pins=())`/`None`
    installs neither the hook nor the participant (empty == no-op,
    `dgemma/payloads.py`) — byte-identical to today's no-`constraints=`
    behavior. `control_signals=` is now LIVE (issue #64 Phase 4, ADR-CDG-011):
    when it carries at least one binding, `run_diffusion` constructs a
    `dgemma.participants.WalkerParticipant` bound to THIS call's `scheduler`
    and wires it into the composite's `walker=` slot (LAST, after every
    canvas-writer) — at the callback for `step_idx = k` the walker maps
    `signal[k + 1]` into the binding's declared `[low, high]` range and
    writes it via `scheduler.register_to_config(...)`, preparing step
    `k + 1`'s config (clause 6); `signal[0]` is never applied (the gate
    ruling on issue #64, O1) and the final step is a no-op (no step `k + 1`
    left to prepare) — see `dgemma.participants.WalkerParticipant`'s
    docstring for the full mechanism. `ControlSignals(bindings=())`/`None`
    builds no walker (empty == no-op) — byte-identical to today's
    no-`control_signals=` behavior. An invalid payload of any of the three
    still raises at ingress regardless of phase; `constraints=` +
    `logit_hook=` together still raise at ingress (H1) even now that
    `constraints=` builds its own hook internally — the two-source-on-one-door
    reject is unconditional.

    Returns `(text, CanvasState, CanvasTrace)` — never a bare string
    (ADR-CDG-001 Addendum). `CanvasTrace` carries `collector.frames` plus
    the scheduler's class name and the entropy/temperature config passed to
    it, per ADR-CDG-001's addendum on scheduler-relative commit semantics
    (a trace without the scheduler identity that minted its commit readings
    is a lying payload). It also carries `raw_canvas_ids` (ADR-CDG-014
    Decision 6, issue #11): the pre-excision final canvas ids, captured in
    `_build_result` before `excise_thought_channel` runs — the raw view
    `CanvasState.canvas_ids` (post-excision) does not carry. Each captured
    `DiffusionFrame` also carries `entropy` (ADR-CDG-014 Decision 3/4, issue
    #14): per-position predictive entropy derived from that step's pre-pin
    `logits`, always populated (Tier 0's always-on default);
    `top_k_ids`/`top_k_weights` (ADR-CDG-014 Decision 3, issue #61 P-B):
    per-position top-k candidate ids and their top-k-renormalized weights
    from the same pre-pin `logits`, populated only when `capture.top_k > 0`
    (`None`/`None` otherwise — Tier 1's on-request default);
    `distribution` (ADR-CDG-014 Decision 3/5, issue #61 P-C): the full
    per-position distribution (`softmax(logits)`) from the same pre-pin
    `logits`, populated only when `capture.capture_full_distribution=True`
    AND the step is still within `capture.max_full_distribution_steps`'s
    retention budget — `None` otherwise (Tier 2's explicit-opt-in-with-budget
    default; `None` also once the budget is exhausted mid-run, Decision 5);
    `pinned_mask`
    (ADR-CDG-010 D4, issue #64 Phase 2/3): `True` at every supplied
    `Constraints` pin position — now the positions `PinParticipant` actually
    (re-)writes every step (Phase 3), consistent with the Phase 2
    static-from-`Constraints.pins` derivation because a hard pin's position
    set never changes step to step (see `DiffusionFrame.pinned_mask`'s
    docstring for the scope guard) — `None` when no constraints were given;
    and
    `effective_entropy_bound`/`effective_t_min`/`effective_t_max`
    (ADR-CDG-011 clause 7, issue #64 Phase 2): the `entropy_bound`/`t_min`/
    `t_max` values `scheduler.config` actually held at that callback — the
    honest-telemetry fields the control-signal walker (issue #64 Phase 4)
    writes through via `register_to_config`, visible in the NEXT captured
    frame after the walker's write (clause 6: walker prepares the next step,
    capture records the finished step).

    `kv_cache=` (ADR-CDG-012 IN-2, issue #62 Phase 2 — types + ingress door,
    no live drive body yet): an optional injected `KVCache` payload (§62's
    `dgemma/types.py` dataclass). `None` (default) is today's EXACT behavior,
    byte-for-byte unchanged — the run mints its own cache internally via the
    pipeline's own first encode, and rule-6 `STATELESS-CORE` is trivially
    satisfied (no injected state crosses). When non-`None`,
    `dgemma.kv_cache.validate_kv_cache_ingress(kv_cache, dgemma_model)` fires
    BEFORE the scheduler/pipeline are constructed (fail-on-mismatch, rule 5
    `EMIT-CANONICAL / PARSE-AT-THE-DOOR` — a bad cache is rejected before any
    resource tied to this call is built) and, on pass,
    `CanvasTrace.injected_cache_provenance` is stamped with the payload's
    `Provenance` record (OUT-3) so a downstream analysis can always tell a
    conditioned run from an unconditioned one. This phase does NOT yet drive
    the decoder off the injected cache's tensors — that live drive body is
    GATED on the ADR's real-weights de-risk smoke test (Open Question #1,
    issue #62 Phase 4); `run_diffusion` validates and stamps provenance
    (the skeleton), and otherwise proceeds exactly as it does when
    `kv_cache=None`. The input `kv_cache` payload itself is never mutated by
    this function (§3 advance-returns-new-payload discipline — this phase
    reads it, it does not write through it).

    Raises `ValueError` if `t_min >= t_max` (parse-at-the-door validation —
    an inverted or degenerate anneal range would silently hand
    `EntropyBoundScheduler` a nonsensical temperature trajectory), if
    ingress validation of `constraints`/`control_signals`/`capture`/the
    `constraints`+`logit_hook` combination fails (see
    `dgemma.ingress.validate_ingress`'s error register), or if `kv_cache` is
    given and fails `validate_kv_cache_ingress`'s V1-V6 checks (see
    `dgemma.kv_cache.validate_kv_cache_ingress`'s error register).
    """
    if t_min >= t_max:
        raise ValueError(f"t_min must be < t_max, got t_min={t_min!r} t_max={t_max!r}.")

    # vocab_size resolution (issue #64 §3.4): same tokenizer path
    # `resolve_thought_channel_ids` uses. `None` when unavailable (e.g. a
    # bare test stub) — validate_constraints degrades by skipping C3 rather
    # than this call site inventing a size.
    vocab_size = resolve_vocab_size(dgemma_model.processor)
    validate_ingress(
        constraints,
        control_signals,
        capture,
        logit_hook,
        gen_length=gen_length,
        num_inference_steps=num_inference_steps,
        vocab_size=vocab_size,
    )

    # ADR-CDG-012 IN-2 (issue #62 Phase 2): fire the KV_CACHE door's own
    # ingress validator BEFORE any scheduler/pipeline construction below —
    # a bad injected cache is rejected before this call ties up a scheduler
    # or pipeline object (rule 5, EMIT-CANONICAL / PARSE-AT-THE-DOOR). `None`
    # (the default) skips this entirely — zero behavior change from before
    # this parameter existed.
    if kv_cache is not None:
        validate_kv_cache_ingress(kv_cache, dgemma_model)

    # Constraints -> the two-mechanism givens (ADR-CDG-010 Decision 1, issue
    # #64 Phase 3). Both mechanisms are built from the SAME validated
    # `constraints.pins` and both are no-ops when `constraints` is `None` or
    # carries no pins (`Constraints()`/`Constraints(pins=())`) — "empty ==
    # no-op" (`dgemma/payloads.py`), so a run with an empty/`None`
    # `constraints=` builds neither the hook nor the pin participant and is
    # byte-identical to today's no-`constraints=` behavior.
    #
    # H1 (validated above) already forecloses `constraints=` AND
    # `logit_hook=` both being given, so building the hook here and passing
    # it through the same `logit_hook` name below can never collide with a
    # caller-supplied one.
    pin_participants: tuple = ()
    if constraints is not None and constraints.pins:
        logit_hook = build_logit_mask_hook(constraints.pins, vocab_size=vocab_size)
        pin_participants = (PinParticipant(constraints=constraints),)

    scheduler = EntropyBoundScheduler(
        entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
    )
    pipeline = DGemmaPipeline(model=dgemma_model.model, scheduler=scheduler, processor=dgemma_model.processor)

    # Control signals -> the walker (ADR-CDG-011, issue #64 Phase 4). Built
    # from THIS call's validated `control_signals` and THIS call's freshly
    # constructed `scheduler` — no cross-call state, no shared scheduler
    # reference (rule 6 STATELESS-CORE; ADR-CDG-011 clause 8/F5). Empty/`None`
    # `control_signals=` builds no walker at all — "empty == no-op"
    # (`dgemma/payloads.py`), byte-identical to today's no-`control_signals=`
    # behavior.
    walker_participant: WalkerParticipant | None = None
    if control_signals is not None and control_signals.bindings:
        walker_participant = WalkerParticipant(control_signals=control_signals, scheduler=scheduler)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=dgemma_model.device).manual_seed(seed)

    # `scheduler` (not `num_inference_steps`) — the collector reads
    # `scheduler.num_inference_steps` lazily per-callback, so it always sees
    # the effective post-`set_timesteps` value the pipeline mutates this same
    # object with at call entry, not the user-requested count snapshotted
    # here before that call runs (issue #20; see `_FrameCollector`'s
    # docstring for the full grounding).
    # `capture.top_k` (ADR-CDG-014 Decision 3 Tier 1, issue #61 P-B): the
    # validated `CaptureSpec.top_k` value, duck-typed the same way
    # `validate_capture` reads `keep_frames` (ADR-CDG-014 Decision 7 — the
    # `capture=` dataclass is owned by this cluster, but a caller-supplied
    # stand-in with the same attribute shape is accepted, not required to
    # be `isinstance CaptureSpec`). `0` (default) when `capture` is `None`
    # or exposes no `top_k` at all — Tier 1 stays off, byte-identical to
    # every pre-P-B run.
    capture_top_k = getattr(capture, "top_k", 0) if capture is not None else 0
    # `capture.capture_full_distribution`/`capture.max_full_distribution_steps`
    # (ADR-CDG-014 Decision 3 Tier 2, issue #61 P-C): same duck-typed read as
    # Tier 1's `top_k` above. `False`/`None` (defaults) when `capture` is
    # `None` or exposes neither attribute — Tier 2 stays off, byte-identical
    # to every pre-P-C run. `validate_ingress` above already rejected
    # `capture_full_distribution=True` with no budget, so by the time this
    # line runs a `True` value is always paired with a positive budget.
    capture_full_distribution = getattr(capture, "capture_full_distribution", False) if capture is not None else False
    capture_max_full_distribution_steps = (
        getattr(capture, "max_full_distribution_steps", None) if capture is not None else None
    )

    collector = _FrameCollector(
        scheduler=scheduler,
        t_min=t_min,
        t_max=t_max,
        keep_frames=keep_frames,
        on_frame=on_frame,
        constraints=constraints,
        top_k=capture_top_k,
        capture_full_distribution=capture_full_distribution,
        max_full_distribution_steps=capture_max_full_distribution_steps,
    )
    step_end = StepEndComposite(
        capture=collector.on_step_end,
        should_cancel=should_cancel,
        pin=pin_participants,
        walker=walker_participant,
    )

    if thinking:
        prompt_kwargs: dict = {
            "messages": [
                {"role": "system", "content": THINK_TOKEN},
                {"role": "user", "content": prompt},
            ]
        }
    else:
        prompt_kwargs = {"prompt": prompt}

    try:
        # `install_logit_shaping_hook` (#35 R5, F4): the ONE place `dgemma/`
        # installs a forward hook on the loaded model, torn down by its own
        # `finally` on every exit from this `with` block — clean return,
        # `DiffusionCancelled` below, or any other exception propagating out
        # of `pipeline(...)`. No hook survives past this block under any of
        # the three paths (ADR-CDG-010 Decision 5, ARCHITECTURE.md rule 6).
        with install_logit_shaping_hook(dgemma_model.model, logit_hook):
            output = pipeline(
                **prompt_kwargs,
                gen_length=gen_length,
                num_inference_steps=num_inference_steps,
                confidence_threshold=confidence,
                generator=generator,
                callback_on_step_end=step_end,
                # "logits" (ADR-CDG-014 Decision 4, issue #14): the Tier 0
                # entropy capture's source — already a base-pipeline
                # `_callback_tensor_inputs` allowlist entry
                # (`pipeline_diffusion_gemma.py:76`), so widening this list
                # is all `run_diffusion` needs to do; `_FrameCollector.
                # on_step_end` derives `DiffusionFrame.entropy` from it.
                callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
            )
    except DiffusionCancelled:
        # #38 partial-return semantics: return the evidence already
        # captured rather than raising it away. Under the capture-first
        # amendment the last captured frame IS the cancelled step's own
        # committed frame — the run's exact truncation point — and its
        # canvas stands in for the pipeline's (never-produced)
        # `output.sequences` — same excision/decode path as the completed
        # case, so a cancelled run's `CanvasState`/`CanvasTrace` are built
        # the identical way a completed run's are, not a special-cased
        # shape.
        #
        # No-frames guard: unreachable through the composite's own flow
        # (capture precedes the cancellation check, and the collector
        # always appends a frame before returning), kept as defensive
        # honesty against a `DiffusionCancelled` raised from anywhere else
        # in the pipeline call — with zero evidence, re-raising is honest
        # and `_build_result` would otherwise mint a fabricated-empty
        # `CanvasState` (or die in `derive_canvas_state` with a less
        # truthful error).
        if not collector.frames:
            raise
        sequences = collector.frames[-1].canvas
        # `DiffusionFrame.canvas` may be 1-D `[canvas_len]` or 2-D
        # `[batch, canvas_len]` (same shape ambiguity `decode_frames`
        # resolves, `dgemma/loop.py`'s `decode_frames` docstring) — the
        # completed path always hands `_build_result` a 1-D sequence
        # (`output.sequences[0]`), so the cancelled path normalizes the
        # same way rather than introducing a second shape contract.
        if hasattr(sequences, "dim") and sequences.dim() == 2:
            sequences = sequences[0]
        return _build_result(
            dgemma_model=dgemma_model,
            pipeline=pipeline,
            scheduler=scheduler,
            sequences=sequences,
            collector=collector,
            entropy_bound=entropy_bound,
            t_min=t_min,
            t_max=t_max,
            num_inference_steps=num_inference_steps,
            injected_cache_provenance=kv_cache.provenance if kv_cache is not None else None,
        )

    return _build_result(
        dgemma_model=dgemma_model,
        pipeline=pipeline,
        scheduler=scheduler,
        sequences=output.sequences[0],
        collector=collector,
        entropy_bound=entropy_bound,
        t_min=t_min,
        t_max=t_max,
        num_inference_steps=num_inference_steps,
        injected_cache_provenance=kv_cache.provenance if kv_cache is not None else None,
    )


def _build_result(
    *,
    dgemma_model: DGemmaModel,
    pipeline: Any,
    scheduler: Any,
    sequences: Any,
    collector: "_FrameCollector",
    entropy_bound: float,
    t_min: float,
    t_max: float,
    num_inference_steps: int,
    injected_cache_provenance: "Provenance | None" = None,
) -> tuple[str, CanvasState, CanvasTrace]:
    """Shared tail of `run_diffusion`'s completed and cancelled paths:
    thought-channel excision, decode, `CanvasState`/`CanvasTrace`
    construction — identical for both so a cancelled run's returned shape is
    not a special case a caller has to branch on (#38: "return what exists"
    means the same contract, populated with less).

    `injected_cache_provenance` (ADR-CDG-012 OUT-3, issue #62 Phase 2):
    `kv_cache.provenance` when `run_diffusion` received a non-`None`
    `kv_cache=`, `None` otherwise — passed straight onto
    `CanvasTrace.injected_cache_provenance` below. Identity only, never the
    cache tensors (those already have their own OUT-1/OUT-2 node-output home,
    Phase 3)."""
    # ADR-CDG-014 Decision 6 (issue #11): capture the pre-excision `sequences`
    # onto `raw_canvas_ids` BEFORE `excise_thought_channel` runs below — this
    # is the only point the final raw (un-excised) canvas ids are ever
    # reachable; `CanvasState.canvas_ids` stays post-excision (the #8
    # contract, unchanged). Plain `list[int]`, mirroring `excise_thought_
    # channel`'s own id-level normalization, so a consumer never has to
    # branch on tensor-vs-list.
    raw_canvas_ids = [int(x) for x in sequences]

    start_id, end_id = resolve_thought_channel_ids(dgemma_model.processor)
    excision = excise_thought_channel(sequences, start_id, end_id)

    text = _decode_ids(dgemma_model.processor, excision.remaining_ids, pipeline.eos_token_id)
    # Decode and label-strip each excised span independently (the "thought\n"
    # channel-name label heads each frame, not just the first), keeping only
    # spans with real content; multiple non-empty spans are joined visibly
    # rather than jammed into one undelimited string.
    thought_parts = [
        part
        for span in excision.thought_spans
        if span
        for part in [_extract_thought_text(_decode_ids(dgemma_model.processor, span, pipeline.eos_token_id))]
        if part
    ]
    thought = "\n\n".join(thought_parts) if thought_parts else None
    canvas_ids = torch.tensor(excision.remaining_ids, dtype=torch.long)

    canvas_state = derive_canvas_state(
        text=text,
        canvas_ids=canvas_ids,
        frames=collector.frames,
        steps_used=collector.steps_used,
        thought=thought,
        stray_thought_delimiter=excision.stray_start_delimiter,
        eos_token_id=pipeline.eos_token_id,
    )
    canvas_trace = CanvasTrace(
        frames=collector.frames,
        scheduler_name=type(scheduler).__name__,
        raw_canvas_ids=raw_canvas_ids,
        scheduler_config={
            "entropy_bound": entropy_bound,
            "t_min": t_min,
            "t_max": t_max,
            # Issue #20: record BOTH, distinctly named, rather than picking
            # one and silently dropping the other. `requested` is what the
            # caller asked for; `effective` is `scheduler.num_inference_steps`
            # AFTER the pipeline's `set_timesteps` call — the actual anneal
            # denominator every frame's `t`/`temperature` was computed
            # against (same value `_FrameCollector` now reads lazily; see its
            # docstring). They are equal for today's only scheduler
            # (`EntropyBoundScheduler`, no `corrector_steps`) and diverge
            # only for a future corrector scheduler — a trace that kept only
            # `requested` would then silently misreport the schedule that
            # actually produced its own frames (ADR-CDG-001 addendum).
            "num_inference_steps_requested": num_inference_steps,
            "num_inference_steps_effective": scheduler.num_inference_steps,
        },
        injected_cache_provenance=injected_cache_provenance,
    )
    return text, canvas_state, canvas_trace
