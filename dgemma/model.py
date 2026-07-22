"""dgemma/model.py — load DiffusionGemma + processor (ADR-CDG-002 load seam).

ComfyUI-agnostic (ADR-CDG-003). Load seam is unchanged by ADR-CDG-004 (which
only amends the *drive* seam, see `dgemma/loop.py`):
`DiffusionGemmaForBlockDiffusion.from_pretrained()` +
`AutoProcessor.from_pretrained()`, both transformers-side.

**Fix #119 (tied-weight corruption under split `device_map="auto"`).**
transformers 5.13.0 ties every encoder text-stack weight to the decoder's by
regex (`modeling_diffusion_gemma.py:1480-1492`) via a shape-unchecked
`setattr` (`modeling_utils.py:2770-2771` — no shape/equality check once both
sides are off the meta device). Under a split placement, accelerate's
per-submodule dispatch hooks can independently re-materialize the encoder's
nominally-tied parameter with the wrong shape while the decoder's stays
correct, producing a cryptic mid-sample `numel==seq_len` view crash instead
of an honest load-time failure. Two mechanisms address this (2026-07-22
forensic verdict, issue #119):

1. **Placement policy** (`_resolve_placement`) — tries a placement that
   cannot split a tied pair (single device, or an explicit
   `model.encoder.language_model.layers.N` / `model.decoder.layers.N`
   pairwise-co-located map) before falling back to memory-oblivious
   `"auto"`.
2. **Tie-integrity guard** (`_assert_tie_integrity`) — the backstop for
   whatever placement is actually used (including a caller-supplied explicit
   `device_map` or the `"auto"` fallback): asserts a sample of encoder layers'
   `self_attn.q_proj.weight` is correctly shaped and tied to its decoder
   counterpart, raising an actionable `RuntimeError` at the end of
   `load_model` rather than letting a corrupted load reach `run_diffusion`.

LIVE verification (the real ~53GB checkpoint under a genuine multi-device
split) is deferred pending GPU availability — see the PR body for #119.

The 26B model needs ~53.6GB in bf16 (model card); bitsandbytes quantization
was the original plan for the 48GB RTX-8000 dev box (Turing, sm_75 — no
native bf16 tensor cores) but does not fit here in practice: bnb only
quantizes `nn.Linear`/`Conv1D` modules, and DiffusionGemma's ~42.5GiB of
fused 3D MoE expert params are neither, so NF4 still needs ~46GiB on a
single card (`loose-ends.md`, 2026-07-05 bnb-MoE entry — issue #4). The
grounded default is `quant="none"` (full-precision bf16, `device_map="auto"`
CPU-spill), verified with two integration PASSes on this box.

`"nf4"`/`"int8"` are gone, not just de-defaulted (issue #18): bitsandbytes
can't touch the part of this architecture that dominates its size, so
selecting either was misleading on any hardware, not just this box. `quant`
is kept as a parameter (loader contract, tests) with its domain constrained
to `("none",)` — a real quantized path is future strategy work, tracked in
issue #4 (AWQ-INT4 checkpoint is the lead candidate), not a bnb config here.
"""
from __future__ import annotations

import torch

from .types import DGemmaModel

DEFAULT_REPO_ID = "google/diffusiongemma-26B-A4B-it"

_QUANT_CHOICES = ("none",)

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
    from transformers import AutoConfig, AutoProcessor, DiffusionGemmaForBlockDiffusion
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


def _pairwise_colocated_device_map(config, device) -> dict:
    """Build an explicit single-device `device_map` that pins every encoder
    text layer to the SAME device as its tied decoder counterpart (fix #119,
    ARCHITECTURE.md rule 6 spirit extended to load-time placement): every
    tied pair is co-located by construction — the degenerate (single-device)
    case of the pairwise invariant `_assert_tie_integrity` checks at the end
    of `load_model`. Everything not named by a more specific key falls under
    the catch-all `""` entry, `device`.

    `_fanout_pairwise_device_map` is this function's multi-device sibling —
    both build the same pairwise-co-located shape, differing only in whether
    every layer pair lands on one device or is spread across several."""
    num_layers = getattr(getattr(config, "text_config", None), "num_hidden_layers", 0)
    device_map = {"": device}
    for layer_idx in range(num_layers):
        device_map[f"model.encoder.language_model.layers.{layer_idx}"] = device
        device_map[f"model.decoder.layers.{layer_idx}"] = device
    return device_map


def _fanout_pairwise_device_map(config, ranked_devices: list[str]) -> dict:
    """Build an explicit multi-device `device_map` that still pins every
    `model.encoder.language_model.layers.N` / `model.decoder.layers.N` pair
    to the SAME device (fix #119's invariant — never splittable, at whatever
    granularity), while spreading pairs across `ranked_devices` (largest
    free-memory device first) to fit a model too large for any single one of
    them. Everything else (embeddings, `lm_head`, vision tower,
    self-conditioning) is pinned to the FIRST (largest) device — the
    catch-all `""` entry names that default, then per-layer-pair entries
    override it for the layers assigned elsewhere.

    Round-robin by device rank (NOT a capacity-weighted bin-packer): each
    layer pair goes to `ranked_devices[layer_idx % len(ranked_devices)]` —
    simple, deterministic, and sufficient for this fix's scope (fixes #119's
    actual hazard — a pair split across a device/hook boundary — without
    claiming to out-optimize accelerate's own balancer). A real
    capacity-weighted split across asymmetric GPUs is exactly what
    `device_map="auto"` (this function's own fallback, gated by
    `_resolve_placement`) already computes; named here as the deliberate
    scope line this function does not cross, not silently half-built."""
    num_layers = getattr(getattr(config, "text_config", None), "num_hidden_layers", 0)
    device_map = {"": ranked_devices[0]}
    for layer_idx in range(num_layers):
        device = ranked_devices[layer_idx % len(ranked_devices)]
        device_map[f"model.encoder.language_model.layers.{layer_idx}"] = device
        device_map[f"model.decoder.layers.{layer_idx}"] = device
    return device_map


def _estimate_model_bytes(config) -> int:
    """Rough bf16 byte-size estimate for a `DiffusionGemmaConfig`-shaped
    model, used only to decide *placement policy* (single-device vs. split)
    in `_resolve_placement` — never a precise accounting, and never used for
    anything load-bearing beyond that one decision (an underestimate just
    means `_resolve_placement` tries single-device and, if it genuinely
    doesn't fit, `from_pretrained` itself raises an OOM — a loud, immediate
    failure, not a silent corruption; an overestimate degrades to the
    pairwise-map-on-largest-device path, which is still tie-safe).

    Counts the dominant text-stack parameter groups from `config.text_config`
    (embeddings, attention projections, MoE expert stacks) at
    `num_hidden_layers` depth, `* 2` bytes/param (bf16). Deliberately coarse
    — a precise accounting needs `compute_module_sizes` against an
    instantiated (even meta) model, which `_resolve_placement` deliberately
    avoids building (this estimate runs BEFORE `from_pretrained`, off the
    config alone, so placement can be decided without a throwaway model
    construction)."""
    text = getattr(config, "text_config", None)
    if text is None:  # pragma: no cover — DiffusionGemmaConfig always sets this; defensive only
        return 0

    hidden = getattr(text, "hidden_size", 0)
    vocab = getattr(text, "vocab_size", 0)
    layers = getattr(text, "num_hidden_layers", 0)
    heads = getattr(text, "num_attention_heads", 0)
    kv_heads = getattr(text, "num_key_value_heads", heads)
    head_dim = getattr(text, "head_dim", 0)
    moe_intermediate = getattr(text, "moe_intermediate_size", None) or getattr(text, "intermediate_size", 0)
    num_experts = getattr(text, "num_experts", None) or 1

    embed_params = vocab * hidden
    q_out = heads * head_dim
    kv_out = kv_heads * head_dim
    attn_params_per_layer = hidden * q_out + 2 * hidden * kv_out + q_out * hidden  # q/k/v/o proj
    # gate_up_proj + down_proj, fused 3D MoE experts (dgemma/model.py module
    # docstring: ~42.5GiB of this pack's 53.6GB is exactly these params).
    moe_params_per_layer = num_experts * (2 * hidden * moe_intermediate + moe_intermediate * hidden)
    per_layer = attn_params_per_layer + moe_params_per_layer

    # Decoder + tied encoder text stack both walk this many layers logically,
    # but the encoder's are the SAME storage once tied (that is the whole
    # point of this fix) — so total resident bytes count `per_layer` once per
    # layer, not twice.
    total_params = embed_params + layers * per_layer
    return total_params * 2  # bf16


def _resolve_placement(config) -> str | dict:
    """Placement policy (fix #119, forensic verdict fix-candidate 2): prefer
    a placement that CANNOT split a tied encoder/decoder pair over the
    memory-oblivious `device_map="auto"` default, which can (and — under
    llama-server VRAM pinning on this box — did, 2026-07-21) split
    `model.encoder.language_model.layers.N` onto a different device/hook
    tier than `model.decoder.layers.N`, corrupting the encoder's tied weight
    via transformers' shape-unchecked `setattr` tie mechanism
    (`modeling_diffusion_gemma.py:1480-1492`, `modeling_utils.py:2770-2771`).

    Policy, in order:
    1. **Single device** — if the largest visible device (GPU preferred,
       `get_max_memory` free-byte ranking; else "cpu") plausibly holds the
       whole bf16 model per `_estimate_model_bytes`, pin everything there via
       `_pairwise_colocated_device_map` (degenerate single-device case: every
       tied pair trivially co-located). No split is possible — the safest
       placement this function can choose.
    2. **Multi-GPU pairwise fan-out** — reachable when the estimate says the
       model does NOT fit on the single largest GPU but MULTIPLE GPUs are
       visible: `_fanout_pairwise_device_map` spreads layer PAIRS (never a
       lone encoder or decoder half) round-robin across every visible GPU,
       ranked largest-free-memory first. This is fix #119's actual
       acceptable-shape target — a split that still can't separate a tied
       pair, at pairwise granularity, instead of `"auto"`'s
       per-submodule-hook granularity that caused the corruption.
    3. **`"auto"`** — only when neither (1) nor (2) applies: no accelerator
       and no measurable system memory visible at all (`get_max_memory`
       returns nothing usable), or exactly one device is visible and the
       model doesn't fit on it (nowhere left to fan out to). The
       tie-integrity guard (`_assert_tie_integrity`) is the backstop for
       this path, converting a silent corruption into a load-time
       `RuntimeError` naming `hf_device_map` (EMIT-CANONICAL / fail-at-the-door).

    Returns either a `device_map` dict (for `from_pretrained`) or the literal
    string `"auto"`. Never returns `None` — callers always get a decided
    kwarg value, honest-absence being `"auto"` itself, not a missing key.
    """
    from accelerate.utils import get_max_memory

    try:
        max_memory = get_max_memory()
    except Exception:  # pragma: no cover — accelerate's own device-probe failure, not this policy's concern
        max_memory = {}

    if not max_memory:
        return "auto"

    # Prefer accelerators over "cpu"/"disk" — mirrors `_resolve_device`'s own
    # device-ranking convention (a bare int is accelerate's GPU encoding;
    # "cpu"/"disk" are never the accelerator). Ranked largest-free-memory
    # first; "cpu" is the whole-model fallback only when no GPU is visible.
    gpu_candidates = sorted(
        ((dev, size) for dev, size in max_memory.items() if isinstance(dev, int)),
        key=lambda item: item[1],
        reverse=True,
    )

    if not gpu_candidates:
        if "cpu" not in max_memory:  # pragma: no cover — get_max_memory always populates "cpu" or "mps"
            return "auto"
        return _pairwise_colocated_device_map(config, "cpu")

    ranked_gpu_devices = [f"cuda:{dev}" for dev, _size in gpu_candidates]
    largest_gpu_free_bytes = gpu_candidates[0][1]
    estimated_bytes = _estimate_model_bytes(config)

    if estimated_bytes <= largest_gpu_free_bytes:
        return _pairwise_colocated_device_map(config, ranked_gpu_devices[0])

    if len(ranked_gpu_devices) > 1:
        return _fanout_pairwise_device_map(config, ranked_gpu_devices)

    # Exactly one GPU, model doesn't fit on it per the estimate, nowhere to
    # fan out to — "auto" is the only path left that can genuinely offload to
    # CPU/disk; the tie-integrity guard is the backstop for whatever it does.
    return "auto"


def _assert_tie_integrity(model) -> None:
    """Load-time tie-integrity guard (fix #119, forensic verdict fix-candidate
    1): converts the cryptic mid-sample `numel==seq_len` view crash
    (`modeling_diffusion_gemma.py:329-330`'s `hidden_shape` view, downstream
    of a corrupted encoder `q_proj` weight) into an honest, actionable
    load-time `RuntimeError` — EMIT-CANONICAL / fail-at-the-door applied to a
    model object instead of a socket payload.

    Root cause this guards (static pass, issue #119 2026-07-22 forensic
    verdict): transformers 5.13.0 ties every encoder text-stack weight to the
    decoder's by regex (`modeling_diffusion_gemma.py:1480-1492`) via a
    shape-unchecked `setattr` (`modeling_utils.py:2770-2771`, no
    `torch.equal`/shape check on the *live* materialized tensors once both
    are off the meta device). Under a split `device_map` placement, accelerate's
    per-submodule dispatch hooks can independently re-materialize the
    encoder's "tied" parameter, collapsing it to the wrong shape while the
    decoder's stays correct — the tie holds at Python-object-identity time
    and silently breaks at dispatch time.

    Checks a SAMPLE of encoder layers (layer 0 and the last layer — cheap,
    and sufficient to catch a placement-driven corruption, which is a
    per-layer dispatch-hook effect, not a per-tensor content bug that would
    need every layer checked) for its `self_attn.q_proj.weight`:
    - 2-D with shape `(num_attention_heads * head_dim, hidden_size)`,
      DERIVED from `model.config.text_config` at call time (never
      hardcoded — a config for a different-sized checkpoint must not silently
      pass a stale hardcoded shape). `head_dim` is PER-LAYER, not global:
      `DiffusionGemmaEncoderTextAttention`/`DiffusionGemmaDecoderTextAttention`
      (`modeling_diffusion_gemma.py:296,398`) both use `config.global_head_dim`
      for full-attention layers and `config.head_dim` for sliding-attention
      layers (`config.layer_types[layer_idx]`) — a single global `head_dim`
      would misjudge a healthy full-attention layer as corrupt.
    - identical (or value-equal, for the offload/data_ptr-unreliable case —
      see `_tensor_ties_match`) to the decoder counterpart at the same layer
      index.

    Raises `RuntimeError` naming the defect, the observed (bad) shape, and
    `model.hf_device_map` (whatever placement produced the corruption) —
    the three facts an operator needs to diagnose this without re-deriving
    the forensic pass. Called at the end of `load_model`, before
    `DGemmaModel` is returned: a caller that gets a `DGemmaModel` back has a
    load this guard has already vetted."""
    text_config = getattr(model.config, "text_config", None)
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

        enc_weight = encoder_layers[layer_idx].self_attn.q_proj.weight
        dec_weight = decoder_layers[layer_idx].self_attn.q_proj.weight

        shape_ok = tuple(enc_weight.shape) == expected_shape
        tie_ok = _tensor_ties_match(enc_weight, dec_weight)

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
                "unsafe to use — re-load with a placement that keeps "
                "model.encoder.language_model.layers.N co-located with "
                "model.decoder.layers.N (see dgemma/model.py's "
                "_resolve_placement), or a single-device load if the model "
                "fits. See issue #119."
            )


def _tensor_ties_match(enc_weight, dec_weight) -> bool:
    """Same tensor (fast path, the common in-memory-tie case) OR
    value-equal (the offload/meta-tensor case, where `data_ptr()`/identity
    is not a reliable tie signal per the forensic verdict's toy-reproducer
    finding that `torch.equal` itself can raise on meta tensors — this
    helper narrows that to a clean boolean the guard can act on, catching
    exactly the meta-tensor `NotImplementedError` and treating an
    unresolvable comparison as a mismatch rather than crashing the guard
    itself).

    Meta tensors are EXCLUDED from the fast `data_ptr()` identity path
    (mirrors transformers' own `tie_weights`, `modeling_utils.py`'s "In case
    the AlignDevicesHook is on meta device, ignore tied weights as
    data_ptr() is then always zero" comment): every meta tensor's
    `data_ptr()` reads `0`, so two UNRELATED meta tensors would otherwise
    spuriously compare as tied."""
    if enc_weight.shape != dec_weight.shape:
        return False
    same_real_storage = (
        enc_weight.device.type != "meta"
        and enc_weight.device == dec_weight.device
        and enc_weight.data_ptr() == dec_weight.data_ptr()
    )
    if same_real_storage:
        return True
    try:
        return bool(torch.equal(enc_weight.detach().to("cpu"), dec_weight.detach().to("cpu")))
    except (RuntimeError, NotImplementedError):
        # Meta tensors (still on the meta device at guard time) or any other
        # unresolvable comparison — never silently pass an unverifiable tie.
        return False


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
    repo_id: str = DEFAULT_REPO_ID,
    quant: str = DEFAULT_QUANT,
    local_files_only: bool = False,
    device_map: str | dict | None = None,
) -> DGemmaModel:
    """Load `DiffusionGemmaForBlockDiffusion` + its processor onto `DGemmaModel`.

    `quant` accepts only `"none"` (issue #18 — full-precision bf16 load,
    `device_map="auto"`, CPU-spills the ~42.5GiB of MoE expert params that
    bitsandbytes could never quantize anyway). Kept as a parameter/field for
    the loader contract; a real quantized path is tracked in issue #4.

    `local_files_only` forwards unchanged to both `from_pretrained` calls —
    off (default) keeps the normal HF download-and-cache behavior; on,
    resolution is restricted to whatever is already in the local HF cache.

    `device_map` (fix #119): an explicit caller-supplied placement is used
    VERBATIM, unchanged — this function never overrides a caller's stated
    intent. `None` (the default) instead calls `_resolve_placement` to pick a
    placement that cannot split a tied encoder/decoder pair (single-device
    when it plausibly fits, else an explicit pairwise-co-located map, else
    `"auto"` as the last resort — see `_resolve_placement`'s docstring for
    the full policy and why `"auto"`'s memory-obliviousness caused issue
    #119 in the first place).

    Raises `RuntimeError` (not a raw transformers/HF stack trace) when
    `repo_id` cannot be resolved — a typo'd repo, no network, or
    `local_files_only=True` with nothing cached. Also raises `RuntimeError`
    (fix #119) when the loaded model's encoder/decoder tie integrity fails
    `_assert_tie_integrity`'s check — a load-time failure instead of a
    cryptic mid-sample crash.
    """
    if quant not in _QUANT_CHOICES:
        raise ValueError(f"quant must be one of {_QUANT_CHOICES}, got {quant!r}.")

    def _wrap_unresolvable_repo(exc: OSError) -> RuntimeError:
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
        return RuntimeError(
            f"Could not load DiffusionGemma from repo_id={repo_id!r}: likely cause is "
            f"{likely_cause}. Original error: {exc}"
        )

    resolved_device_map: str | dict
    if device_map is not None:
        # Caller stated an explicit placement intent — never second-guessed,
        # and the config fetch below (needed only to DECIDE a placement) is
        # skipped entirely.
        resolved_device_map = device_map
    else:
        try:
            config = AutoConfig.from_pretrained(repo_id, local_files_only=local_files_only)
        except OSError as exc:
            raise _wrap_unresolvable_repo(exc) from exc
        resolved_device_map = _resolve_placement(config)

    load_kwargs: dict = {
        "device_map": resolved_device_map,
        "dtype": torch.bfloat16,
        "local_files_only": local_files_only,
    }

    try:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(repo_id, **load_kwargs)
        processor = AutoProcessor.from_pretrained(repo_id, local_files_only=local_files_only)
    except OSError as exc:
        raise _wrap_unresolvable_repo(exc) from exc

    _assert_tie_integrity(model)

    device = _resolve_device(model)

    return DGemmaModel(
        model=model,
        processor=processor,
        device=device,
        dtype="bfloat16",
        repo_id=repo_id,
        quant=quant,
    )
