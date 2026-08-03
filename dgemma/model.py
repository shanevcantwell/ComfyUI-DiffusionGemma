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

**Fix #119 (tied-weight corruption under split `device_map="auto"`).**
transformers 5.13.0 ties every encoder text-stack weight to the decoder's by
regex (`modeling_diffusion_gemma.py:1480-1492`) via a shape-unchecked
`setattr` (`modeling_utils.py:2770-2771` — no shape/equality check once both
sides are off the meta device). Under a split placement, accelerate's
per-submodule dispatch hooks can independently re-materialize the encoder's
nominally-tied parameter with the wrong shape while the decoder's stays
correct, producing a cryptic mid-sample `numel==seq_len` view crash instead
of an honest load-time failure. `_assert_tie_integrity` (2026-07-22 forensic
verdict fix-candidate 1) is the backstop for whatever placement
`device_map="auto"` produces: it asserts a sample of encoder layers'
`self_attn.q_proj.weight` is correctly shaped and tied to its decoder
counterpart, raising an actionable `RuntimeError` at the end of `load_model`
naming `model.hf_device_map` rather than letting a corrupted load reach
`run_diffusion`. `_tensor_ties_match`/`_resolve_meta_weight` make that
comparison offload-aware (2026-07-22 discriminating probe, PR #121 comment
5044645329) — a cpu-offloaded tied pair's weights are meta-at-rest under
accelerate's `AlignDevicesHook`, and a naive `torch.equal` on the unresolved
meta tensors reports a false positive on every offloaded sampled layer,
which this guard resolves through `module._hf_hook.weights_map` before
comparing. Deliberately NOT included: an explicit pairwise-co-located
`device_map` placement *policy* that would override `"auto"` — issues
#173/#183 hardened `device_map="auto"` as this pack's field-verified,
release-blocking-protected default for both quant modes, and an untested
override belongs behind its own live-GPU verification, not folded into this
guard fix. See issue #119 and this repo's PR referencing it for the scope
note.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
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


def _explained_by_device_map_offload(name: str, device_map: dict) -> bool:
    """True when parameter/buffer `name` is meta *because* accelerate offloaded
    the module that owns it to `cpu`/`disk` under `device_map="auto"`.

    `hf_device_map` keys are module paths — dotted prefixes of the full
    parameter name (e.g. key `model.decoder.layers.27` owns param
    `model.decoder.layers.27.mlp.gate_up_proj.weight`). accelerate places a
    whole module at one device, so the owning entry is the *longest* device-map
    key that is a dot-boundary prefix of `name` (longest-prefix match, so a
    nested submodule with its own entry wins over an ancestor's). That module's
    device is the offload signal: a value of `"cpu"` or `"disk"` (the same
    strings `_resolve_device` skips over to find the accelerator) means the
    meta residency is a legitimate mmap-backed offload, not a stranded tensor.
    A bare-int / accelerator entry (or no matching entry) is NOT an offload —
    a meta tensor there is genuinely stranded.
    """
    best_key = None
    for key in device_map:
        if name == key or name.startswith(key + "."):
            if best_key is None or len(key) > len(best_key):
                best_key = key
    if best_key is None:
        return False
    return str(device_map[best_key]) in ("cpu", "disk")


def _assert_no_meta_tensors(model) -> None:
    """FAIL LOUD (issue #142's enforcement surface) if any parameter or
    buffer is still meta-resident after load + re-tie, before the model is
    returned to any caller (issue #183: no `.to("cuda")` follows anymore) —
    SPILL-AWARE: a tensor that is meta *because* accelerate offloaded its
    module to `cpu`/`disk` under `device_map="auto"` is legitimate and exempt.

    Two meta-residency causes must be told apart (issue #173):

    - **Legitimate offload (exempt).** Under `device_map="auto"` on a card that
      cannot hold the whole bf16 checkpoint, accelerate places the overflow
      modules at `cpu`/`disk` and their params report device **`meta`** in
      `named_parameters()` — the mmap-backed lazy-load regime the probe record
      quantified (`docs/experiments/bf16-fit-mechanism/README.md`, Trap #2:
      "Offloaded params report device `meta`, not `cpu`, in
      `named_parameters()`"; 13 modules / 5.50 B params / 10.25 GiB observed).
      This meta is expected and correct — the module IS placed, at cpu/disk per
      `hf_device_map`. Rejecting it would hard-refuse the default load the
      restore of `device_map="auto"` exists to enable.
    - **Stranded tensor (still raises).** A meta tensor NOT explained by an
      offload entry — e.g. a tied `lm_head` suppressed but never materialized
      during `from_pretrained` (issue #142's #142 class), whose owning module
      the device map places on an accelerator — holds no data and cannot move
      via `.to()`. It would otherwise surface later as an opaque 'Cannot copy
      out of meta tensor' crash or a silent no-op hang. This still raises, by
      name, with the actionable remedy.

    The offload signal is the same one `_resolve_device` already reads: an
    `hf_device_map` entry of `cpu`/`disk` for the module owning the tensor. See
    `_explained_by_device_map_offload` for the (longest-prefix) name→module
    match. Why the exemption exists is recorded in the probe:
    `docs/experiments/bf16-fit-mechanism/README.md`.
    """
    device_map = getattr(model, "hf_device_map", None) or {}
    meta_names = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.device.type == "meta"
        and not _explained_by_device_map_offload(name, device_map)
    ]
    if meta_names:
        raise RuntimeError(
            "ComfyUI-DiffusionGemma: model has meta-resident tensor(s) after "
            f"load: {meta_names}. A meta tensor holds no "
            "data and cannot be moved to a real device via .to() — this would "
            "otherwise surface later as an opaque 'Cannot copy out of meta "
            "tensor' crash or a silent no-op hang, depending on the dispatch "
            "path. These tensor(s) are NOT explained by a cpu/disk offload "
            "entry in hf_device_map, so this is not accelerate's legitimate "
            "device_map=\"auto\" spill (issue #173) — it is usually a tied-"
            "weight that was suppressed but never materialized during "
            "from_pretrained (see issue #142). "
            "Remedy: report this repo_id/quant combination — a new tied "
            "parameter may need its own re-tie handling alongside "
            "_retie_lm_head."
        )


# Whole-fit floor for the AutoRound INT4 checkpoint, in bytes. Grounded by
# the 2026-07-30 forced-split probe (docs/experiments/
# 2026-07-30-autoround-unified-path-split-check/): weights measure 28.55 GiB
# resident after load; the floor adds margin for accelerate's dispatch
# reserve so a card that passes the check genuinely places the model whole.
# (Activation peak — 30.67 GiB process-wide in the 8-step probe — is a
# run-time concern, not what the load-time dispatch crash gates on.)
AUTOROUND_MIN_FREE_VRAM_BYTES = 30 * 1024**3


@dataclass(frozen=True)
class GpuMemoryHolder:
    """One process NVML reports as holding GPU memory on the current device —
    the measured unit issue #191's guard message is built from. `is_self` is
    this process's own pid (`os.getpid()`), so a prior in-process DGemma load
    is identified distinctly from a foreign tenant, per the issue's acceptance
    criteria."""

    pid: int
    name: str
    used_mib: float
    is_self: bool


def _gpu_memory_holders(device_index: int) -> tuple[list[GpuMemoryHolder], str | None]:
    """Return `(holders, unavailable_reason)` for `device_index` via NVML.

    `holders` lists every process NVML's `nvmlDeviceGetComputeRunningProcesses`
    reports for the device (pid, process name, MiB used, self-vs-foreign),
    empty when NVML enumerates but finds no compute processes. When
    enumeration cannot be performed at all, `holders` is `[]` and
    `unavailable_reason` names why (no NVML binding installed, driver not
    loaded, or any other NVML-surfaced error) — the caller's job is to say so
    honestly rather than fall back to speculative cause-prose (issue #191).

    Uses `pynvml` (the binding module `torch.cuda.list_gpu_processes` itself
    depends on — `nvidia-ml-py`'s installed package also imports under this
    name) directly rather than adding a new pack dependency: this guard reads
    exactly what's already on the machine to drive `torch.cuda`'s own
    process-listing helper, and degrades honestly when it is absent.
    """
    try:
        import pynvml
    except ImportError as exc:
        return [], f"pynvml not installed ({exc})"

    try:
        pynvml.nvmlInit()
    except Exception as exc:  # noqa: BLE001 - any NVML init failure is unavailability
        return [], f"NVML init failed ({exc})"

    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except Exception as exc:  # noqa: BLE001 - any NVML query failure is unavailability
        return [], f"NVML process enumeration failed ({exc})"
    finally:
        with contextlib.suppress(Exception):
            pynvml.nvmlShutdown()

    self_pid = os.getpid()
    holders = []
    for proc in procs:
        try:
            proc_name = pynvml.nvmlSystemGetProcessName(proc.pid)
            if isinstance(proc_name, bytes):
                proc_name = proc_name.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - a name lookup failure still reports the pid/MiB
            proc_name = "<name unavailable>"
        used_mib = (proc.usedGpuMemory or 0) / (1024 * 1024)
        holders.append(
            GpuMemoryHolder(pid=proc.pid, name=proc_name, used_mib=used_mib, is_self=proc.pid == self_pid)
        )
    return holders, None


def _format_gpu_memory_holders(holders: list[GpuMemoryHolder], unavailable_reason: str | None) -> str:
    """Render the measured-holder report line the guard message includes —
    the enumeration itself, never a hypothesis about it."""
    if unavailable_reason is not None:
        return f"occupants unmeasurable: {unavailable_reason}"
    if not holders:
        return "no per-process GPU memory holders reported"
    lines = []
    for holder in holders:
        tag = "this process" if holder.is_self else "foreign process"
        lines.append(f"pid {holder.pid} ({holder.name}, {tag}): {holder.used_mib:.0f} MiB")
    return "; ".join(lines)


def _assert_autoround_vram_precondition() -> None:
    """FAIL LOUD, pre-load, when the card cannot hold the AutoRound INT4
    checkpoint whole (issue #183 — the split-fails probe outcome).

    Why refuse instead of split: under `device_map="auto"`, a card that
    cannot fit the INT4 checkpoint whole makes accelerate attempt a CPU/GPU
    split, and that split CANNOT LOAD — dispatch crashes inside
    `from_pretrained` (`ValueError: weight is on the meta device...`,
    forced-split probe leg, docs/experiments/
    2026-07-30-autoround-unified-path-split-check/). This is unlike the bf16
    path, whose CPU-mmap spill is field-proven. So the honest pre-load
    behavior on a small card is a refusal naming both numbers and the
    remedy, never a silent attempt at a split that is proven to crash.
    Split-capable INT4 (block-wise onload) is banked as future work.

    Skips silently when CUDA is unavailable — `load_model`'s existing
    post-load CUDA check owns that refusal with its own canonical message.

    Issue #191: the message states the ONE measured condition — required
    floor, measured free, and the measured per-process holder list (NVML,
    self vs. foreign) — with at most one remedy line derived from that
    measured state. No speculative cause-prose, no remedy menu; when NVML
    can't enumerate holders the message says so rather than guessing.
    """
    if not torch.cuda.is_available():
        return
    free_bytes, _total_bytes = torch.cuda.mem_get_info()
    if free_bytes < AUTOROUND_MIN_FREE_VRAM_BYTES:
        device_index = torch.cuda.current_device()
        holders, unavailable_reason = _gpu_memory_holders(device_index)
        holder_report = _format_gpu_memory_holders(holders, unavailable_reason)
        remedy = (
            "unload the foreign process(es) named above and retry"
            if any(not h.is_self for h in holders)
            else "free VRAM and retry"
        )
        raise RuntimeError(
            "quant='autoround' needs the whole INT4 checkpoint resident on one "
            f"GPU: {AUTOROUND_MIN_FREE_VRAM_BYTES / 1024**3:.1f} GiB free VRAM "
            f"required, {free_bytes / 1024**3:.1f} GiB free now. "
            f"Measured GPU memory holders (device {device_index}): {holder_report}. "
            f"Remedy: {remedy}."
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


def _resolve_meta_weight(module):
    """Resolve `module.weight` to its real (non-meta) backing tensor when the
    module sits under a cpu-offloaded `AlignDevicesHook(offload=True)` — the
    common `device_map="auto"` + CPU-spill regime this repo's default load
    path uses (2026-07-22 false-positive probe, issue #119, PR #121 comment
    5044645329). Under that hook, the parameter itself is meta-at-rest and
    the real data lives in `module._hf_hook.weights_map`, a `PrefixedDataset`
    keyed by the parameter's LOCAL name (`"weight"`), not its fully-qualified
    module path — so the lookup is always `weights_map["weight"]`, resolved
    independently per side (one side may be GPU-resident while the other is
    offloaded; the two sides are never assumed symmetric).

    Returns the resolved tensor, or `None` when the weight is meta and there
    is no resolvable hook/weights_map to resolve it through — a genuinely
    unverifiable tie, which the caller must treat as a failure, not a pass.
    """
    weight = module.weight
    if not weight.is_meta:
        return weight
    hook = getattr(module, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is None:
        return None
    try:
        return weights_map["weight"]
    except KeyError:
        return None


def _tensor_ties_match(enc_module, dec_module) -> bool:
    """Same tensor (fast path, the common in-memory-tie case) OR
    value-equal (the offload/meta-tensor case).

    Takes the encoder/decoder `q_proj` MODULES (not bare `.weight` tensors)
    so a meta-at-rest weight — the state accelerate's
    `AlignDevicesHook(offload=True)` leaves the live parameter in under this
    repo's default `device_map="auto"` CPU-spill load — can be resolved to
    its real backing tensor via `_resolve_meta_weight` before any comparison
    is attempted. This closes a false positive the 2026-07-22 discriminating
    probe found on issue #119 (PR #121 comment 5044645329): `torch.equal`
    itself raises `NotImplementedError` on a meta tensor ("Cannot copy out of
    meta tensor; no data!"), and a naive version of this helper caught that
    into an unconditional `False` — so every cpu-offloaded sampled layer
    would report a broken tie even when the tie was, per the probe's three
    independent checks (shared weights_map storage, hook-materialized
    `data_ptr` equality, and coherent live generation with the guard
    bypassed), fully intact.

    Meta tensors are EXCLUDED from the fast `data_ptr()` identity path on the
    UNRESOLVED weight (mirrors transformers' own `tie_weights`,
    `modeling_utils.py`'s "In case the AlignDevicesHook is on meta device,
    ignore tied weights as data_ptr() is then always zero" comment): every
    meta tensor's `data_ptr()` reads `0`, so two UNRELATED meta tensors would
    otherwise spuriously compare as tied. Resolution happens first, then the
    same identity/equality logic runs on the resolved (never-meta) tensors.
    """
    if enc_module.weight.shape != dec_module.weight.shape:
        return False

    enc_weight = _resolve_meta_weight(enc_module)
    dec_weight = _resolve_meta_weight(dec_module)
    if enc_weight is None or dec_weight is None:
        # A meta weight with no resolvable _hf_hook/weights_map — genuinely
        # unverifiable. Never silently pass an unverifiable tie.
        return False

    same_real_storage = (
        enc_weight.device.type != "meta"
        and dec_weight.device.type != "meta"
        and enc_weight.device == dec_weight.device
        and enc_weight.data_ptr() == dec_weight.data_ptr()
    )
    if same_real_storage:
        return True
    try:
        return bool(torch.equal(enc_weight.detach().to("cpu"), dec_weight.detach().to("cpu")))
    except (RuntimeError, NotImplementedError):  # pragma: no cover — backstop only: both
        # branches above resolve meta tensors before this line runs, so the healthy
        # offloaded-tie path (this fix's actual target) never reaches this except;
        # kept as a non-crashing backstop for any other unresolvable comparison —
        # never silently pass an unverifiable tie.
        return False


def _assert_tie_integrity(model) -> None:
    """Load-time tie-integrity guard (issue #119, 2026-07-22 forensic
    verdict fix-candidate 1): converts the cryptic mid-sample
    `numel==seq_len` view crash (`modeling_diffusion_gemma.py:329-330`'s
    `hidden_shape` view, downstream of a corrupted encoder `q_proj` weight)
    into an honest, actionable load-time `RuntimeError` — EMIT-CANONICAL /
    fail-at-the-door applied to a model object instead of a socket payload.

    Root cause this guards (static pass, issue #119 2026-07-22 forensic
    verdict): transformers 5.13.0 ties every encoder text-stack weight to the
    decoder's by regex (`modeling_diffusion_gemma.py:1480-1492`) via a
    shape-unchecked `setattr` (`modeling_utils.py:2770-2771`, no
    `torch.equal`/shape check on the *live* materialized tensors once both
    are off the meta device). Under a split `device_map` placement,
    accelerate's per-submodule dispatch hooks can independently
    re-materialize the encoder's "tied" parameter, collapsing it to the
    wrong shape while the decoder's stays correct — the tie holds at
    Python-object-identity time and silently breaks at dispatch time.
    `device_map="auto"` is this pack's default placement for both quant
    modes (issues #173/#183) — this guard is the backstop for whatever
    placement that produces, without changing the placement itself.

    Checks a SAMPLE of encoder layers (layer 0 and the last layer — cheap,
    and sufficient to catch a placement-driven corruption, which is a
    per-layer dispatch-hook effect, not a per-tensor content bug that would
    need every layer checked) for its `self_attn.q_proj.weight`:
    - 2-D with shape `(num_attention_heads * head_dim, hidden_size)`,
      DERIVED from `model.config.text_config` at call time (never
      hardcoded — a config for a different-sized checkpoint must not
      silently pass a stale hardcoded shape). `head_dim` is PER-LAYER, not
      global: `DiffusionGemmaEncoderTextAttention`/
      `DiffusionGemmaDecoderTextAttention`
      (`modeling_diffusion_gemma.py:296,398`) both use
      `config.global_head_dim` for full-attention layers and
      `config.head_dim` for sliding-attention layers
      (`config.layer_types[layer_idx]`) — a single global `head_dim` would
      misjudge a healthy full-attention layer as corrupt.
    - identical (or value-equal, for the offload/data_ptr-unreliable case —
      see `_tensor_ties_match`) to the decoder counterpart at the same layer
      index.

    Raises `RuntimeError` naming the defect, the observed (bad) shape, and
    `model.hf_device_map` (whatever placement produced the corruption) — the
    three facts an operator needs to diagnose this without re-deriving the
    forensic pass. Called at the end of `load_model`, before `DGemmaModel` is
    returned: a caller that gets a `DGemmaModel` back has a load this guard
    has already vetted.

    No-op (returns without checking) if the model has no `.config`, the
    config has no `text_config`, or `text_config` reports zero hidden
    layers — a real `DiffusionGemmaForBlockDiffusion` always has all three,
    so this is defensive only (untriggerable against a real checkpoint), but
    this guard must never itself be the reason an otherwise-valid model
    fails to load, and must not require every existing test double
    elsewhere in this module's test suite to grow encoder/decoder layer
    structure it has no reason to model.
    """
    config = getattr(model, "config", None)
    if config is None:
        return
    text_config = getattr(config, "text_config", None)
    if text_config is None:  # pragma: no cover — DiffusionGemmaConfig always sets this; defensive only
        return

    num_hidden_layers = getattr(text_config, "num_hidden_layers", 0)
    if num_hidden_layers <= 0:  # pragma: no cover — untriggerable against a real checkpoint config
        return

    num_attention_heads = text_config.num_attention_heads
    hidden_size = text_config.hidden_size
    layer_types = getattr(text_config, "layer_types", None) or []
    global_head_dim = getattr(text_config, "global_head_dim", None)
    sliding_head_dim = text_config.head_dim

    inner = model.model if hasattr(model, "model") else model
    encoder_layers = inner.encoder.language_model.layers
    decoder_layers = inner.decoder.layers

    sample_indices = sorted({0, num_hidden_layers - 1})

    for layer_idx in sample_indices:
        is_sliding = layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention"
        # Mirrors DiffusionGemmaEncoderTextAttention.__init__'s own derivation
        # (modeling_diffusion_gemma.py:296): global_head_dim wins on a
        # full-attention layer only when it is truthy/set.
        head_dim = sliding_head_dim if (is_sliding or not global_head_dim) else global_head_dim
        expected_shape = (num_attention_heads * head_dim, hidden_size)

        enc_q_proj = encoder_layers[layer_idx].self_attn.q_proj
        dec_q_proj = decoder_layers[layer_idx].self_attn.q_proj
        enc_weight = enc_q_proj.weight
        dec_weight = dec_q_proj.weight

        shape_ok = tuple(enc_weight.shape) == expected_shape
        tie_ok = _tensor_ties_match(enc_q_proj, dec_q_proj)

        if not shape_ok or not tie_ok:
            device_map = getattr(model, "hf_device_map", None)
            raise RuntimeError(
                "DiffusionGemma tied-weight corruption detected under split "
                "placement (issue #119): transformers 5.13.0 ties every "
                "encoder text-stack weight to the decoder's via a "
                "shape-unchecked setattr, and a split device_map placement "
                "can independently re-materialize the encoder's 'tied' "
                f"parameter with the wrong shape. layer {layer_idx}'s "
                f"encoder self_attn.q_proj.weight has shape "
                f"{tuple(enc_weight.shape)}; expected {expected_shape} "
                f"(derived from config: num_attention_heads={num_attention_heads}, "
                f"head_dim={head_dim}, hidden_size={hidden_size}) and equal to "
                f"the decoder's counterpart (shape {tuple(dec_weight.shape)}). "
                f"model.hf_device_map={device_map!r}. This model load is "
                "unsafe to use — reduce the split (fewer/no CPU-offloaded "
                "encoder/decoder layer pairs) or use a single-device load if "
                "the model fits. See issue #119."
            )


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

    `quant` accepts `"none"` (full-precision bf16) or `"autoround"`
    (pre-quantized W4A16 INT4 checkpoint via auto-round). Placement is
    UNIFORM across quant modes (issue #183): both load under
    `device_map="auto"` — accelerate-managed GPU+CPU-mmap placement that
    spills overflow modules to CPU where the card cannot hold the
    checkpoint whole (the bf16 26B path on a 48GB card CPU-spills the
    ~10GiB of overflow; the ~30GB INT4 checkpoint fits whole there). The
    only per-quant difference is `dtype` — a checkpoint-identity fact, not
    a placement decision.

    `local_files_only` forwards unchanged to both `from_pretrained` calls —
    off (default) keeps the normal HF download-and-cache behavior; on,
    resolution is restricted to whatever is already in the local HF cache.

    `check_interrupted` (issue #140 loader half): an optional zero-argument
    predicate polled at four phase boundaries — before the quant/checkpoint
    pre-flight config read, before the model `from_pretrained`, before the
    processor `from_pretrained`, and at the (historical) device-move
    boundary — kept as a poll point even though no `.to("cuda")` happens
    for any quant mode anymore (issue #183: accelerate places everything
    during `from_pretrained`). When it reports `True`, `load_model` raises
    `LoadInterrupted`
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

    # ONE load path — placement is UNIFORM across quant modes (issue #183).
    # `device_map="auto"` (accelerate-managed GPU+CPU-mmap placement) for
    # BOTH quants; the ONLY per-quant residue is `dtype`, a checkpoint-
    # identity fact (`"auto"` lets transformers read the INT4 checkpoint's
    # own quantization_config; bf16 is stated explicitly), never a placement
    # decision.
    #
    # device_map="auto" was fought for, not assumed — keep the history:
    # dd2767c (2026-07-24) dropped it for both paths because accelerate's
    # dispatch left the tied lm_head/embed_tokens pair meta-resident under
    # CPU spill, crashing the (then-unconditional) .to("cuda"). d0bb93b
    # (#142/#143, 2026-07-29) fixed that root cause directly —
    # _retie_lm_head() + _assert_no_meta_tensors() materialize and verify
    # tied weights regardless of quant mode — and PR #177 (#173) restored
    # device_map="auto" for quant="none", where it is field-proven
    # (docs/experiments/bf16-fit-mechanism/: 42.4GiB GPU + 10.25GiB
    # mmap-backed spill, 2.2s/step). The autoround branch briefly kept
    # dd2767c's no-device_map / low_cpu_mem_usage=False / .to("cuda")
    # contract on a bare-process "~30GB fits whole" observation; #183's
    # field report falsified that as placement policy, and the divergent
    # branch was deleted (docs/experiments/
    # 2026-07-30-autoround-unified-path-split-check/).
    if quant == "autoround":
        dtype_kwarg: object = "auto"  # checkpoint identity: read its quantization_config
        dtype_label = "int4"
    else:
        dtype_kwarg = torch.bfloat16
        dtype_label = "bfloat16"
    load_kwargs: dict = {
        "device_map": "auto",  # accelerate-managed GPU+CPU-mmap placement
        "dtype": dtype_kwarg,
        "local_files_only": local_files_only,
    }

    # PRE-LOAD VRAM precondition — the ONE quant-conditional besides dtype
    # (issue #183, split-fails probe outcome): an INT4 checkpoint that cannot
    # fit whole makes accelerate attempt a split that crashes inside
    # from_pretrained, so refuse loudly BEFORE the blocking call. bf16 has no
    # precondition — its spill path is field-proven.
    if quant == "autoround":
        _assert_autoround_vram_precondition()

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
    # dispatches through it (a stranded meta tensor holds no data; issue
    # #142's field failure was an opaque crash/hang downstream).
    if quant == "autoround":
        _retie_lm_head(model)

    # POST-LOAD ASSERTION (all quant paths): fail loud, naming the tensor(s),
    # if anything is still meta-resident — the enforcement surface that keeps
    # this class of bug from ever again presenting as a downstream hang or an
    # opaque .to("cuda") crash (issue #142).
    _assert_no_meta_tensors(model)

    # POST-LOAD ASSERTION (all quant paths): fail loud, naming the device
    # map, if the encoder/decoder tied text-stack weights were corrupted by
    # a split device_map dispatch — the enforcement surface for issue #119.
    # Pure backstop: does not change device_map="auto" (issues #173/#183's
    # field-proven, release-blocking-protected default) — only vets whatever
    # placement accelerate actually produced.
    _assert_tie_integrity(model)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "ComfyUI-DiffusionGemma requires a CUDA-capable GPU. No CUDA device "
            "found (torch.cuda.is_available() is False). There is no supported "
            "CPU-only path for this pack: DiffusionGemma's ~53GB bf16 / ~30GB "
            "INT4 footprint and the sampler's CUDA-seeded torch.Generator "
            "(dgemma/loop.py run_diffusion) both assume an accelerator."
        )

    # Phase boundary (issue #140): still polled on both paths, so
    # check_interrupted's four-boundary contract holds regardless of quant.
    # NO device move happens here for EITHER quant mode (issue #183):
    # accelerate already placed every tensor per device_map="auto" during
    # from_pretrained (GPU + CPU-mmap spill where needed) — a whole-model
    # .to("cuda") after that would pull any CPU-spilled weights back onto
    # the card, defeating the spill and OOMing (issue #173, the exact
    # mechanism PR #177 fixed for quant="none"). The boundary is kept as a
    # poll point so the four-boundary cancellation contract stays stable.
    _poll("device move (.to(\"cuda\"))")

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
