"""dgemma/model.py — load DiffusionGemma + processor (ADR-CDG-002 load seam).

ComfyUI-agnostic (ADR-CDG-003). Load seam is unchanged by ADR-CDG-004 (which
only amends the *drive* seam, see `dgemma/loop.py`):
`DiffusionGemmaForBlockDiffusion.from_pretrained()` +
`AutoProcessor.from_pretrained()`, both transformers-side.

The 26B model needs ~53.6GB in bf16 (model card); bitsandbytes quantization
was the original plan for the 48GB RTX-8000 dev box (Turing, sm_75 — no
native bf16 tensor cores) but does not fit here in practice: bnb only
quantizes `nn.Linear`/`Conv1D` modules, and DiffusionGemma's ~42.5GiB of
fused 3D MoE expert params are neither, so NF4 still needs ~46GiB on a
single card (`loose-ends.md`, 2026-07-05 bnb-MoE entry — issue #4). The
grounded default is `quant="none"` (full-precision bf16, `device_map="auto"`
CPU-spill), verified with two integration PASSes on this box (most recently
the 2026-07-30 instrumented probe, `docs/experiments/bf16-fit-mechanism/`:
42.4GiB GPU + 10.25GiB mmap-backed lazy offload, 2.2 s/step on this 48GB
card).

AutoRound INT4 (`quant="autoround"`) loads pre-quantized W4A16 checkpoints
(e.g. Intel/diffusiongemma-26B-A4B-it-int4-AutoRound) at ~30GB VRAM vs 53GB
bf16. Requires `auto-round`, a base dependency (issue #139 — was the
`[quant]` optional extra; folded in since it's one of two supported quant
modes, not a hardware-gated install). The load path patches three
transformers/auto-round issues: regex pre-compilation for MoE expert
matching, KV-cache warmup that pre-allocates bf16-sized buffers, and
tied-weight finalization on quantized modules.

`"nf4"`/`"int8"` are gone, not just de-defaulted (issue #18): bitsandbytes
can't touch the part of this architecture that dominates its size, so
selecting either was misleading on any hardware, not just this box. `quant`
is kept as a parameter (loader contract, tests) with its domain constrained
to `("none", "autoround")` — see issue #128.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import torch

from .types import DGemmaModel


class LoadInterrupted(Exception):
    """Raised by `load_model` when the surface-supplied `check_interrupted`
    predicate reports `True` at one of the phase boundaries (issue #140
    loader half).

    Mirrors `dgemma/composite.py`'s `DiffusionCancelled` in shape (a plain,
    engine-local exception so `dgemma/model.py` stays ComfyUI-agnostic,
    ADR-CDG-003 — never imports `comfy.model_management` itself) but NOT in
    partial-return semantics: a load has no partial `(text, CanvasState,
    CanvasTrace)` to salvage, so this is a hard stop, not a caught-and-return.
    The surface (`surfaces/comfyui/loader.py`) is the layer that decides what
    a `LoadInterrupted` means to ComfyUI's own executor (see that module for
    the translation into `comfy.model_management.InterruptProcessingException`).
    """


DEFAULT_REPO_ID = "google/diffusiongemma-26B-A4B-it"
# Pre-quantized AutoRound W4A16 checkpoint — ~30GB VRAM vs 53GB bf16.
# Used as the default when quant="autoround" and no explicit repo_id is given.
AUTOROUND_REPO_ID = "Intel/diffusiongemma-26B-A4B-it-int4-AutoRound"

_QUANT_CHOICES = ("none", "autoround")

# ONE-MINT: the widget default (nodes/loader.py) and this function's own
# default both source from here, so there is exactly one place that decides
# what a fresh graph starts with.
DEFAULT_QUANT = "none"

# issue #25: the ComfyUI registry archive has no build step, so
# ComfyUI-Manager installs deps from requirements.txt via plain pip — and
# pip (per Manager's own installer) silently *skips* a pin that would
# downgrade an already-installed package. An env can therefore end up
# holding a transformers other than this pack's target series, which
# DiffusionGemmaForBlockDiffusion either doesn't exist in (raw ImportError,
# no context) or behaves differently under (worse: no error at all). This
# front-door guard turns both into one actionable message.
#
# Patch-tolerant: accepts the pinned major.minor series (`5.13.x` for a
# `5.13.0` pin) and flags only a different minor or major. A working patch
# bump is a bugfix on the same API surface this pack was tested against, so
# hard-failing it would be more disruptive than the risk it guards; a
# minor/major bump is untested surface, so it stays flagged.
REQUIRED_TRANSFORMERS_VERSION = "5.13.0"


def _required_series() -> tuple[int, ...]:
    """The accepted `(major, minor)` series, DERIVED from
    `REQUIRED_TRANSFORMERS_VERSION` (never hardcoded) so the pin stays the
    single source of truth. `"5.13.0"` -> `(5, 13)`."""
    return tuple(int(part) for part in REQUIRED_TRANSFORMERS_VERSION.split(".")[:2])


def _version_mismatch_message(installed: str) -> str:
    series = ".".join(str(n) for n in _required_series())
    return (
        f"ComfyUI-DiffusionGemma requires transformers {series}.x "
        f"(this pack pins =={REQUIRED_TRANSFORMERS_VERSION}), but "
        f"transformers=={installed} is installed in this Python environment. "
        "ComfyUI-Manager's dependency installer silently skips a requirements.txt pin "
        "that would downgrade an already-installed package, so this environment can "
        "hold a transformers version other than the one this pack targets even after "
        "a normal Manager install. Fix: run "
        f"`pip install transformers=={REQUIRED_TRANSFORMERS_VERSION}` in ComfyUI's own "
        "Python environment. See issue #25."
    )


def _check_transformers_version(installed: str | None = None) -> None:
    """Raise an actionable `RuntimeError` (issue #25) unless the installed
    transformers is in `REQUIRED_TRANSFORMERS_VERSION`'s major.minor series.

    Patch-tolerant: accepts the pinned major.minor series (`5.13.x` for a
    `5.13.0` pin) and flags anything with a different minor or major
    (`5.12.*`, `5.14.*`, `6.*`, ...). A working patch bump is a bugfix on
    the same API surface this pack was tested against, so it shouldn't
    hard-fail; a minor/major bump is untested surface, so it is flagged.

    `installed` is normally left `None` (reads the real `transformers.__version__`
    at call time) — the parameter exists so this thin guard is directly
    unit-testable without monkeypatching `sys.modules`. Compares with
    `packaging.version.Version` when `packaging` is importable (it normally
    is: transformers depends on it itself), taking `.release[:2]` (major,
    minor) so a local build tag / pre-release suffix doesn't derail the
    series match; falls back to a patch-tolerant `major.minor.` string-prefix
    compare when `packaging` isn't importable. Both paths DERIVE the accepted
    series from `REQUIRED_TRANSFORMERS_VERSION` — no hardcoded `"5.13"`.
    """
    if installed is None:
        import transformers as _transformers

        installed = getattr(_transformers, "__version__", "unknown")

    required_series = _required_series()

    try:
        from packaging.version import Version

        mismatched = Version(installed).release[:2] != required_series
    except Exception:  # pragma: no cover — untriggerable: packaging is a transformers dep, always importable
        # Patch-tolerant string fallback: the installed version must start
        # with the `major.minor.` prefix. The trailing dot is load-bearing —
        # it stops `5.130.0` from matching a `5.13` series.
        prefix = ".".join(str(n) for n in required_series) + "."
        mismatched = not installed.startswith(prefix)

    if mismatched:
        raise RuntimeError(_version_mismatch_message(installed))


_check_transformers_version()

try:
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
except ImportError as exc:  # pragma: no cover — broken/partial transformers install, see issue #25
    # The version check above already raised its own actionable message for
    # a simple version mismatch — reaching here with an ImportError means
    # something else is broken about the installed transformers (partial or
    # corrupt install). Still name the required version and issue #25
    # instead of surfacing the raw traceback.
    raise RuntimeError(
        "Could not import DiffusionGemmaForBlockDiffusion from transformers "
        f"(required: transformers=={REQUIRED_TRANSFORMERS_VERSION}). See issue #25. "
        f"Original error: {exc}"
    ) from exc


# Reentrancy guard for _apply_autoround_patches() — see its docstring (H2).
_apply_autoround_patches_depth = 0


@contextlib.contextmanager
def _apply_autoround_patches():
    """Context manager: patch transformers + auto-round for INT4 checkpoint
    loading, active only across the `from_pretrained` call, restored after.

    Three patches, all verified on the 48GB RTX-8000 box with Intel's
    diffusiongemma-26B-A4B-it-int4-AutoRound (issue #128):

    1. **auto-round regex pre-compilation** — `skip_not_convert_modules`
       recompiles ~120 regex patterns for every module name in the model
       (~7K modules), pinning one CPU core at 100%. Pre-compile once.

    2. **KV-cache warmup bypass** — `caching_allocator_warmup` pre-allocates
       a bf16-sized buffer (46GB) before knowing weights are INT4, causing
       OOM on consumer GPUs. Skip it; the actual INT4 load fits in ~30GB.

    3. **Tied-weight finalization** — `mark_tied_weights_as_initialized` and
       `tie_weights` crash when lm_head.weight is tied to a quantized
       embed_tokens that has no `.weight` attribute (only `.qweight`). NOTE:
       Patch 3 only suppresses that crash — it does not materialize the tied
       tensor. `load_model`'s post-load meta-tensor assertion (issue #142)
       is what catches the resulting meta-resident `lm_head.weight`; a
       fresh re-tie there is what actually fixes it.

    Scoped, not global-and-permanent (issue #142 H2 — investigation's
    standing recommendation): these are load-time-only hooks (allocator
    warmup, weight-tying), so leaving them installed after `from_pretrained`
    returns serves no purpose and is a needless global-monkeypatch footprint
    for the rest of the process lifetime. Original attributes are restored
    on exit, success or failure.

    Reentrancy-guarded: nested/concurrent `with _apply_autoround_patches():` calls
    (e.g. a caller wrapping `load_model` while it also applies the patches)
    only patch on the outermost entry and only restore on the outermost
    exit, so an inner call never clobbers an outer call's originals.
    """
    global _apply_autoround_patches_depth
    import re as _re
    from transformers import modeling_utils as _mu

    _apply_autoround_patches_depth += 1
    if _apply_autoround_patches_depth > 1:
        # Already patched by an outer scope — no-op, defer restore to it.
        try:
            yield
        finally:
            _apply_autoround_patches_depth -= 1
        return

    orig_convert_skip = None
    orig_warmup = _mu.caching_allocator_warmup
    orig_mark = _mu.PreTrainedModel.mark_tied_weights_as_initialized
    orig_tie = _mu.PreTrainedModel.tie_weights

    try:
        # Patch 1: auto-round regex pre-compilation
        try:
            from auto_round.inference import convert_model as _ar_convert

            orig_convert_skip = _ar_convert.skip_not_convert_modules

            def _patched_skip(model, quant_config, layer_names, extra_config):
                modules_to_not_convert = []
                if extra_config:
                    for name in extra_config.keys():
                        try:
                            _re.compile(name)
                            modules_to_not_convert.append(name)
                        except _re.error:
                            pass
                compiled = [
                    _re.compile(n) if n else None for n in modules_to_not_convert
                ]
                return extra_config.copy()

            _ar_convert.skip_not_convert_modules = _patched_skip
        except ImportError:
            # auto-round not installed — patch is a no-op, will fail at load time
            pass

        # Patch 2: skip bf16 KV-cache warmup (pre-allocates wrong size for INT4)
        _mu.caching_allocator_warmup = lambda *a, **k: None

        # Patch 3: tied-weight finalization on quantized modules
        def _patched_mark(self, loading_info):
            for tied_param in self._tied_weights_keys:
                try:
                    param = self.get_parameter(tied_param)
                    if hasattr(param, "data"):
                        loading_info.missing_keys.remove(tied_param)
                except (AttributeError, KeyError):
                    pass

        _mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark

        def _patched_tie(self, *a, **kw):
            try:
                return orig_tie(self, *a, **kw)
            except (NotImplementedError, AttributeError):
                pass

        _mu.PreTrainedModel.tie_weights = _patched_tie

        yield
    finally:
        if orig_convert_skip is not None:
            try:
                from auto_round.inference import convert_model as _ar_convert

                _ar_convert.skip_not_convert_modules = orig_convert_skip
            except ImportError:
                pass
        _mu.caching_allocator_warmup = orig_warmup
        _mu.PreTrainedModel.mark_tied_weights_as_initialized = orig_mark
        _mu.PreTrainedModel.tie_weights = orig_tie
        _apply_autoround_patches_depth -= 1


def _checkpoint_quant_method(repo_id: str, local_files_only: bool) -> tuple[bool, str | None]:
    """Best-effort read of the checkpoint's declared
    `quantization_config.quant_method`, without ever taking the blocking
    `from_pretrained` path (issue #141).

    Returns `(readable, quant_method)`:
    - `(True, "auto-round")` — config was read; checkpoint declares that method.
    - `(True, None)` — config was read; checkpoint declares no `quantization_config`.
    - `(False, None)` — config could NOT be read (unreachable repo, network
      trouble, malformed JSON, ...). Callers MUST treat `readable=False` as
      "unknown" and fall through to the normal load rather than block or
      raise — this is a pre-flight hint, not a second front door with its
      own failure mode. `readable=False` is NOT the same as "confirmed
      unquantized" (`(True, None)`); conflating the two would let an
      unreadable config masquerade as a confirmed mismatch.

    Reads via `transformers.AutoConfig.from_pretrained`, which resolves both
    local paths and Hub ids uniformly and respects `local_files_only`.
    """
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(repo_id, local_files_only=local_files_only)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: best-effort only
        print(
            f"[WARN] ComfyUI-DiffusionGemma — could not pre-flight-check "
            f"quantization_config for repo_id={repo_id!r} ({exc!r}); skipping the "
            "quant/checkpoint mismatch guard and proceeding to load. If the "
            "checkpoint's quantization does not match quant=..., the load may hang "
            "or fail deeper in transformers."
        )
        return False, None

    quant_config = getattr(config, "quantization_config", None)
    if not quant_config:
        return True, None

    if isinstance(quant_config, dict):
        return True, quant_config.get("quant_method")
    return True, getattr(quant_config, "quant_method", None)


def _check_quant_checkpoint_match(
    repo_id: str, quant: str, local_files_only: bool
) -> None:
    """PRE-FLIGHT guard (issue #141): reject a `quant=...` / checkpoint
    mismatch BEFORE the blocking `from_pretrained` call.

    Root cause this prevents: an AutoRound INT4 checkpoint loaded with
    quant="none" hangs permanently — transformers deserializes INT4 data
    (`qweight`/`qzeros`) as bf16 tensors with no error surface, just a
    freeze that requires killing ComfyUI. Symmetric in the other direction
    too: quant="autoround" against an unquantized checkpoint is also a
    silent-wrong-load risk, not just the INT4-into-bf16 hang.

    Best-effort by construction: `_checkpoint_quant_method` never blocks or
    raises on its own account (an unreadable config returns
    `readable=False`, with a logged warning), so this guard only rejects a
    *confirmed* mismatch — `readable=False` always falls through, on either
    side of the direction check.

    Every raise names BOTH the violated precondition AND the actionable
    remedy in one message (house style, see `dgemma/kv_cache.py` V1-V6).
    """
    readable, declared_method = _checkpoint_quant_method(repo_id, local_files_only)
    if not readable:
        return

    if declared_method is not None and quant == "none":
        raise RuntimeError(
            f"Checkpoint {repo_id!r} declares quantization_config.quant_method="
            f"{declared_method!r} but quant='none' was passed. Loading a quantized "
            "checkpoint with quant='none' deserializes quantized weights as bf16 "
            "with no error surface — a permanent hang, not a crash. "
            "Remedy: pass quant='autoround' (if the declared method is AutoRound), "
            "or choose an unquantized checkpoint."
        )

    if declared_method is None and quant == "autoround":
        raise RuntimeError(
            f"quant='autoround' was passed but checkpoint {repo_id!r} has no "
            "quantization_config (confirmed absent, not just unread). Loading an "
            "unquantized checkpoint under the autoround path applies patches meant "
            "for INT4 W4A16 weights to bf16 weights, which is not a supported "
            "combination. "
            "Remedy: pass quant='none' for this checkpoint, or point repo_id at a "
            "pre-quantized AutoRound checkpoint (e.g. AUTOROUND_REPO_ID)."
        )


def _retie_lm_head(model) -> None:
    """Re-tie `lm_head.weight` to its true-storage sibling when the AutoRound
    load path leaves it meta-resident (issue #142).

    Root cause (probe v3, issue #142): `_apply_autoround_patches()`'s Patch 3
    suppresses `PreTrainedModel.tie_weights`'s crash on a quantized
    `embed_tokens` (no plain `.weight`, only `.qweight`) but does not
    materialize the tied tensor — `lm_head.weight` is left stranded on
    `meta`. `DiffusionGemmaForBlockDiffusion._tied_weights_keys` declares
    the tie explicitly: `{"lm_head.weight": "model.decoder.embed_tokens.weight"}`
    (`transformers/models/diffusion_gemma/modeling_diffusion_gemma.py`).
    Mirrors exactly what the library's own (unpatched) `tie_weights` does for
    this pair — `setattr(parent, name, source_param)`, i.e. point
    `lm_head.weight` at the *same* `nn.Parameter` object `embed_tokens`
    already holds real (non-meta) data for — rather than a copy, since a
    genuine weight tie shares storage.

    No-op if `lm_head.weight` is already real (e.g. the bf16 `quant="none"`
    path, or a future transformers release that fixes the underlying patch
    interaction) — this is a targeted repair for the one known-affected
    tensor, not a general meta-tensor materializer.
    """
    lm_head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    if lm_head_weight is None or lm_head_weight.device.type != "meta":
        return

    source = model.get_parameter("model.decoder.embed_tokens.weight")
    if source.device.type == "meta":
        # The source itself is meta-resident — re-tying would just point one
        # meta tensor at another. Leave it; the post-load assertion below
        # will name both and fail loud rather than silently no-op here.
        return

    model.lm_head.weight = source


def _assert_no_meta_tensors(model) -> None:
    """FAIL LOUD (issue #142's enforcement surface) if any parameter or
    buffer is still meta-resident after load + re-tie, BEFORE `.to("cuda")`.

    Without this, a stray meta tensor surfaces later as `.to("cuda")`'s
    opaque `NotImplementedError: Cannot copy out of meta tensor; no data!`
    (probe v3) — or worse, silently no-ops under a dispatch path that
    tolerates meta tensors (accelerate `device_map`, per-submodule `.to()`),
    deferring the failure to first *use* of that tensor as a much later,
    harder-to-attribute crash or hang (probe v3's stated risk). Scanning
    `named_parameters()` + `named_buffers()` here — for every quant path, not
    just autoround — turns any future instance of this class of bug into an
    immediate, named, actionable error instead of a downstream hang.
    """
    meta_names = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
    ]
    if meta_names:
        raise RuntimeError(
            "ComfyUI-DiffusionGemma: model has meta-resident tensor(s) after "
            f"load (before .to(\"cuda\")): {meta_names}. A meta tensor holds no "
            "data and cannot be moved to a real device via .to() — this would "
            "otherwise surface later as an opaque 'Cannot copy out of meta "
            "tensor' crash or a silent no-op hang, depending on the dispatch "
            "path. This is usually a tied-weight that was suppressed but "
            "never materialized during from_pretrained (see issue #142). "
            "Remedy: report this repo_id/quant combination — a new tied "
            "parameter may need its own re-tie handling alongside "
            "_retie_lm_head."
        )


def _resolve_device(model) -> str:
    """Resolve the model's *execution* device, not its first parameter's.

    Under `device_map="auto"` with CPU spill (the unquantized 26B path on the
    48GB box), accelerate may place the first parameter off-GPU while the
    execution device — where the pipeline creates the canvas and where the
    seeded `torch.Generator` must live (`run_diffusion`) — is still the
    accelerator. The first non-cpu/disk entry of `hf_device_map` is that
    device (accelerate encodes GPUs as bare ints); a fully-CPU or
    un-dispatched load falls back to the first parameter honestly.
    """
    device_map = getattr(model, "hf_device_map", None) or {}
    for dev in device_map.values():
        if isinstance(dev, int):
            return f"cuda:{dev}"
        if str(dev) not in ("cpu", "disk"):
            return str(dev)
    return str(next(model.parameters()).device)


def load_model(
    repo_id: str | None = None,
    quant: str = DEFAULT_QUANT,
    local_files_only: bool = False,
    check_interrupted: Callable[[], bool] | None = None,
) -> DGemmaModel:
    """Load `DiffusionGemmaForBlockDiffusion` + its processor onto `DGemmaModel`.

    `repo_id` defaults to the quant-appropriate checkpoint:
    - `quant="none"` → `DEFAULT_REPO_ID` (Google bf16, ~53GB VRAM)
    - `quant="autoround"` → `AUTOROUND_REPO_ID` (Intel INT4 W4A16, ~30GB VRAM)
    Pass an explicit path or HF repo ID to override.

    `quant` accepts `"none"` (full-precision bf16 load, `device_map="auto"`,
    accelerate-managed GPU+CPU-mmap placement that CPU-spills the ~42.5GiB of
    MoE expert params that bitsandbytes could never quantize) or `"autoround"`
    (pre-quantized W4A16 INT4 checkpoint via auto-round, loaded onto CPU then
    moved to CUDA with a single `.to("cuda")` — no accelerate dispatch).

    `local_files_only` forwards unchanged to both `from_pretrained` calls —
    off (default) keeps the normal HF download-and-cache behavior; on,
    resolution is restricted to whatever is already in the local HF cache.

    `check_interrupted` (issue #140 loader half): an optional zero-argument
    predicate polled at four phase boundaries — before the quant/checkpoint
    pre-flight config read, before the model `from_pretrained`, before the
    processor `from_pretrained`, and before the final `.to("cuda")` device
    move. When it reports `True`, `load_model` raises `LoadInterrupted`
    immediately, without starting the next blocking call. `None` (the
    default, and every non-ComfyUI caller — tests, MCP, direct script use)
    means "never interrupt", so this parameter is additive: no caller is
    required to supply it. This narrows the hang window from "the whole
    load" to "one blocking call" — a stuck `from_pretrained` itself still
    cannot be interrupted mid-call (issue #140's plan explicitly scopes
    mid-call interruption to 0.6.x, via thread/subprocess isolation).

    Raises `RuntimeError` (not a raw transformers/HF stack trace) when
    `repo_id` cannot be resolved — a typo'd repo, no network, or
    `local_files_only=True` with nothing cached. Raises `LoadInterrupted`
    when `check_interrupted` reports `True` at a phase boundary.
    """
    if quant not in _QUANT_CHOICES:
        raise ValueError(f"quant must be one of {_QUANT_CHOICES}, got {quant!r}.")

    def _poll(phase: str) -> None:
        if check_interrupted is not None and check_interrupted():
            raise LoadInterrupted(
                f"DiffusionGemma load interrupted before phase: {phase}."
            )

    # Auto-select the checkpoint matching the quant mode when no explicit repo
    if repo_id is None:
        repo_id = AUTOROUND_REPO_ID if quant == "autoround" else DEFAULT_REPO_ID

    # PRE-FLIGHT guard (issue #141): reject a quant=/checkpoint mismatch
    # before the blocking from_pretrained call — see _check_quant_checkpoint_match.
    # Also a phase boundary in its own right (issue #140): it reads the
    # checkpoint config via AutoConfig.from_pretrained, itself a network-
    # capable call when the config isn't already cached.
    _poll("quant/checkpoint mismatch pre-flight")
    _check_quant_checkpoint_match(repo_id, quant, local_files_only)

    # Autoround INT4 path: patches transformers + auto-round for correct load.
    # quant="none" path: accelerate-managed placement (device_map="auto") —
    # issue #173. dd2767c (2026-07-24) dropped device_map="auto" for both
    # paths because accelerate's dispatch left the tied lm_head/embed_tokens
    # pair meta-resident under CPU spill, crashing the (then-unconditional)
    # .to("cuda"). d0bb93b (#142/#143, 2026-07-29) fixed that root cause
    # directly — _retie_lm_head() + _assert_no_meta_tensors() materialize and
    # verify tied weights regardless of quant mode — so device_map="auto" no
    # longer needs to be avoided on quant="none"'s account. The autoround
    # path keeps the no-device_map / low_cpu_mem_usage=False / .to("cuda")
    # contract dd2767c introduced: the pre-quantized W4A16 checkpoint (~30GB)
    # fits on this box without CPU spill, so there is nothing for accelerate
    # dispatch to buy it, and low_cpu_mem_usage=False is what lets
    # _retie_lm_head/_assert_no_meta_tensors reason about real (non-meta)
    # tensors before the explicit device move.
    if quant == "autoround":
        dtype_kwarg = "auto"  # let transformers read quantization config
        dtype_label = "int4"
        load_kwargs: dict = {
            "low_cpu_mem_usage": False,  # force real CPU tensors; meta tensors can't move to CUDA
            "dtype": dtype_kwarg,
            "local_files_only": local_files_only,
        }
    else:
        dtype_kwarg = torch.bfloat16
        dtype_label = "bfloat16"
        load_kwargs = {
            "device_map": "auto",  # accelerate-managed GPU+CPU-mmap placement
            "dtype": dtype_kwarg,
            "local_files_only": local_files_only,
        }

    print(
        f"[INFO] ComfyUI-DiffusionGemma 0.4.0 — loading from {repo_id!r} "
        f"({dtype_label}, quant={quant})"
    )

    # The autoround patches are load-time-only hooks (allocator warmup,
    # weight-tying) — scoped to just the from_pretrained call, not left
    # installed globally for the rest of the process (issue #142 H2).
    patches_cm = _apply_autoround_patches() if quant == "autoround" else contextlib.nullcontext()

    # Phase boundary (issue #140): poll before each blocking from_pretrained
    # call, not after — an interrupt reported while a call is already
    # in-flight cannot stop that call (transformers/safetensors expose no
    # cancellation hook), so the check's only useful position is the gap
    # between phases, catching the interrupt before it commits to the next
    # one.
    _poll("model from_pretrained")
    try:
        with patches_cm:
            model = DiffusionGemmaForBlockDiffusion.from_pretrained(repo_id, **load_kwargs)
        _poll("processor from_pretrained")
        processor = AutoProcessor.from_pretrained(
            repo_id,
            local_files_only=local_files_only,
        )
    except ImportError as exc:
        # auto-round not installed when quant="autoround" — surface an
        # actionable message instead of a raw transformers ImportError deep
        # in the accelerate dispatch stack (handoff 2026-07-23 open question 3)
        if quant == "autoround":
            raise RuntimeError(
                f"quant='autoround' requires the auto-round library, but it is not "
                f"installed in this Python environment. Fix: run "
                f"`pip install 'auto-round>=0.5'` in ComfyUI's own Python environment. "
                f"Original error: {exc}"
            ) from exc
        raise
    except OSError as exc:
        # transformers/huggingface_hub surface an unresolvable repo as an
        # OSError subclass (LocalEntryNotFoundError, RepositoryNotFoundError,
        # HfHubHTTPError all derive from OSError) — narrow catch, so a bug
        # elsewhere in this function (e.g. a real ValueError/TypeError)
        # still surfaces as itself instead of being relabeled.
        likely_cause = (
            "not present in the local Hugging Face cache (local_files_only=True)"
            if local_files_only
            else "a typo'd repo_id or no network access to the Hugging Face Hub"
        )
        raise RuntimeError(
            f"Could not load DiffusionGemma from repo_id={repo_id!r}: likely cause is "
            f"{likely_cause}. Original error: {exc}"
        ) from exc

    # issue #142: the autoround path's tied lm_head.weight can be left
    # meta-resident (Patch 3 suppresses the tie crash without materializing
    # the tensor) — re-tie it to its real-storage sibling before anything
    # touches .to("cuda"), which cannot move a meta tensor.
    if quant == "autoround":
        _retie_lm_head(model)

    # POST-LOAD ASSERTION (all quant paths): fail loud, naming the tensor(s),
    # if anything is still meta-resident — the enforcement surface that keeps
    # this class of bug from ever again presenting as a downstream hang or an
    # opaque .to("cuda") crash (issue #142).
    _assert_no_meta_tensors(model)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ComfyUI-DiffusionGemma requires a CUDA-capable GPU. No CUDA device "
            "found (torch.cuda.is_available() is False). There is no supported "
            "CPU-only path for this pack: DiffusionGemma's ~53GB bf16 / ~30GB "
            "INT4 footprint and the sampler's CUDA-seeded torch.Generator "
            "(dgemma/loop.py run_diffusion) both assume an accelerator."
        )

    # Phase boundary (issue #140): still polled on both paths, so
    # check_interrupted's four-boundary contract holds regardless of quant —
    # a caller polling for cancellation should not have to know which quant
    # mode skips the actual device move.
    _poll("device move (.to(\"cuda\"))")

    # quant="none": accelerate already placed every tensor per device_map
    # during from_pretrained (GPU + CPU-mmap spill) — an unconditional
    # whole-model .to("cuda") here would pull the CPU-spilled ~10GiB back
    # onto the card, defeating the spill and OOMing (issue #173). quant=
    # "autoround": no accelerate dispatch was requested (see load_kwargs
    # above), so the model is still CPU-resident and needs the explicit
    # single .to() call.
    if quant == "autoround":
        model = model.to("cuda")

    device = _resolve_device(model)

    print(f"[INFO] ComfyUI-DiffusionGemma — model loaded on {device}")

    return DGemmaModel(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype_label,
        repo_id=repo_id,
        quant=quant,
    )
