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
CPU-spill), verified with two integration PASSes on this box.

AutoRound INT4 (`quant="autoround"`) loads pre-quantized W4A16 checkpoints
(e.g. Intel/diffusiongemma-26B-A4B-it-int4-AutoRound) at ~30GB VRAM vs 53GB
bf16. Requires `auto-round` (the `[quant]` optional extra). The load path
patches three transformers/auto-round issues: regex pre-compilation for MoE
expert matching, KV-cache warmup that pre-allocates bf16-sized buffers,
and tied-weight finalization on quantized modules.

`"nf4"`/`"int8"` are gone, not just de-defaulted (issue #18): bitsandbytes
can't touch the part of this architecture that dominates its size, so
selecting either was misleading on any hardware, not just this box. `quant`
is kept as a parameter (loader contract, tests) with its domain constrained
to `("none", "autoround")` — see issue #128.
"""
from __future__ import annotations

import torch

from .types import DGemmaModel

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


def _apply_autoround_patches() -> None:
    """Patch transformers + auto-round for INT4 checkpoint loading.

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
       embed_tokens that has no `.weight` attribute (only `.qweight`).
    """
    import re as _re

    # Patch 1: auto-round regex pre-compilation
    try:
        from auto_round.inference import convert_model as _ar_convert

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
    from transformers import modeling_utils as _mu
    _mu.caching_allocator_warmup = lambda *a, **k: None

    # Patch 3: tied-weight finalization on quantized modules
    _orig_mark = _mu.PreTrainedModel.mark_tied_weights_as_initialized

    def _patched_mark(self, loading_info):
        for tied_param in self._tied_weights_keys:
            try:
                param = self.get_parameter(tied_param)
                if hasattr(param, "data"):
                    loading_info.missing_keys.remove(tied_param)
            except (AttributeError, KeyError):
                pass

    _mu.PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark

    _orig_tie = _mu.PreTrainedModel.tie_weights

    def _patched_tie(self, *a, **kw):
        try:
            return _orig_tie(self, *a, **kw)
        except (NotImplementedError, AttributeError):
            pass

    _mu.PreTrainedModel.tie_weights = _patched_tie


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
) -> DGemmaModel:
    """Load `DiffusionGemmaForBlockDiffusion` + its processor onto `DGemmaModel`.

    `repo_id` defaults to the quant-appropriate checkpoint:
    - `quant="none"` → `DEFAULT_REPO_ID` (Google bf16, ~53GB VRAM)
    - `quant="autoround"` → `AUTOROUND_REPO_ID` (Intel INT4 W4A16, ~30GB VRAM)
    Pass an explicit path or HF repo ID to override.

    `quant` accepts `"none"` (full-precision bf16 load, `device_map="auto"`,
    CPU-spills the ~42.5GiB of MoE expert params that bitsandbytes could never
    quantize) or `"autoround"` (pre-quantized W4A16 INT4 checkpoint via auto-round).

    `local_files_only` forwards unchanged to both `from_pretrained` calls —
    off (default) keeps the normal HF download-and-cache behavior; on,
    resolution is restricted to whatever is already in the local HF cache.

    Raises `RuntimeError` (not a raw transformers/HF stack trace) when
    `repo_id` cannot be resolved — a typo'd repo, no network, or
    `local_files_only=True` with nothing cached.
    """
    if quant not in _QUANT_CHOICES:
        raise ValueError(f"quant must be one of {_QUANT_CHOICES}, got {quant!r}.")

    # Auto-select the checkpoint matching the quant mode when no explicit repo
    if repo_id is None:
        repo_id = AUTOROUND_REPO_ID if quant == "autoround" else DEFAULT_REPO_ID

    # Autoround INT4 path: patches transformers + auto-round for correct load
    if quant == "autoround":
        _apply_autoround_patches()
        dtype_kwarg = "auto"  # let transformers read quantization config
        dtype_label = "int4"
    else:
        dtype_kwarg = torch.bfloat16
        dtype_label = "bfloat16"

    load_kwargs: dict = {
        "device_map": "auto",
        "dtype": dtype_kwarg,
        "local_files_only": local_files_only,
    }

    try:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(repo_id, **load_kwargs)
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
                f"`pip install 'comfyui-diffusiongemma[quant]'` (or `pip install auto-round`) "
                f"in ComfyUI's own Python environment. Original error: {exc}"
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

    device = _resolve_device(model)

    return DGemmaModel(
        model=model,
        processor=processor,
        device=device,
        dtype=dtype_label,
        repo_id=repo_id,
        quant=quant,
    )
