"""dgemma/compat.py — diffusers version guard + structural probe (issue #35
R3, ARCHITECTURE.md "No diffusers version guard" row; ADR-CDG-018 Stage 2,
issue #129's `dgemma/loop.py` decomposition).

Pure import-time validation, no loop state. Extracted verbatim from
`dgemma/loop.py` (no behavior change — see `tests/test_loop_golden_trace.py`)
except for one mechanical addition: `assert_diffusers_compatible()`, a single
public entry point wrapping the two checks that used to fire as bare
module-top calls in `loop.py` itself.

**Import-time-side-effect preservation (the one non-mechanical bit of this
extraction).** Before this module existed, `import dgemma.loop` ran
`_check_diffusers_version()` + `_check_diffusers_structure()` at its own
module top, before `loop.py`'s `from diffusers import ...` line — the ordering
`dgemma/__init__.py` depends on (imports `.loop` before `.model`, so diffusers
lands in `sys.modules` before transformers does on a fresh `import dgemma`).
That ordering is preserved, not just the checks themselves: `loop.py` calls
`assert_diffusers_compatible()` (defined here) as its own first statement
after this module's import, still strictly before `loop.py`'s own
`from diffusers import DiffusionGemmaPipeline, EntropyBoundScheduler` line.
The guard therefore still fires exactly once per process, at `import dgemma`
time, in the same relative position — `compat.py`'s own module top no longer
runs the checks itself (a bare top-level call here would fire on
`import dgemma.compat` alone, which is not the invariant this module needs;
`loop.py` remains the single call site, matching today's one-guard-per-import
behavior).

Enforcement surface: `tests/test_diffusers_version_guard.py` — imports
`REQUIRED_DIFFUSERS_MINIMUM`, `_check_diffusers_version`, `_tuple_version`,
`_check_diffusers_structure` from `dgemma.loop` (facade re-export, ratified
plan's facade ruling — issue #129); it does not import `assert_diffusers_
compatible` itself, so that wrapper carries no direct test of its own beyond
the fact that `import dgemma` (exercised by every other test in this suite)
would fail loudly if it stopped firing the two checks it wraps.
"""
from __future__ import annotations

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
#    names/shapes `dgemma/capture.py` (`anneal_temperature`, `_FrameCollector`)
#    and `dgemma/loop.py` (`DGemmaPipeline`) read off diffusers — so instead
#    of trusting a newer version to keep every probed structure
#    byte-identical, this probe asserts each one directly, unconditional on
#    version number, and fails loud naming exactly which structure moved.
#    This is the range+structural-probe split (not exact-pin): an exact pin
#    would be both stricter than the declared dependency bound and wrong the
#    moment diffusers ships a compatible 0.40/0.41.
#
#    Honest scope: the probe protects the NAMES/SHAPES `anneal_temperature`
#    and friends read (ctor kwargs, config attrs, output fields, the base
#    callback allowlist) — it cannot see the anneal formula's *body*, which
#    `anneal_temperature` re-derives. That formula-body residual is enforced
#    by `tests/test_diffusers_version_guard.py:TestAnnealFormulaPin`, which
#    pins the re-derivation against the temperature the REAL installed
#    scheduler's `step()` actually applies.
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
      scheduler with exactly these kwargs, and `anneal_temperature`
      (`dgemma/capture.py`) re-derives its formula from `.config.t_min`/
      `.config.t_max`.
    - The scheduler exposes `.num_inference_steps` as a plain instance
      attribute, not only via `.config` (`scheduling_entropy_bound.py:84,89`)
      — `_FrameCollector` (`dgemma/capture.py`) reads
      `scheduler.num_inference_steps` lazily per callback (issue #20; see
      that class's own docstring) precisely because it is a mutable plain
      attribute, not the frozen config copy.
    - `EntropyBoundSchedulerOutput` (the real `scheduler.step()` return
      type) exposes an `accepted_index` field (`scheduling_entropy_bound.py
      :26-40`) — `_FrameCollector.on_step_end` reads
      `scheduler_output.accepted_index` directly.
    - The base `diffusers.DiffusionGemmaPipeline._callback_tensor_inputs`
      allowlist is `["canvas", "logits"]` (`pipeline_diffusion_gemma.py:76`)
      — `DGemmaPipeline` (`dgemma/loop.py`) widens it by appending
      `"scheduler_output"`; if the base class ever renamed or dropped either
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


def assert_diffusers_compatible() -> None:
    """Run both diffusers compatibility checks, in order — the version-floor
    check (`_check_diffusers_version`) then the structural probe
    (`_check_diffusers_structure`). One public call, so `loop.py` (this
    module's sole caller) fires the same two-check sequence its own module
    top used to run directly, before this extraction (ADR-CDG-018 Stage 2,
    issue #129). Raises `RuntimeError` from whichever check fails first;
    never swallows either."""
    _check_diffusers_version()
    _check_diffusers_structure()
