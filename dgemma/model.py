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
grounded default is therefore `quant="none"` (full-precision bf16,
`device_map="auto"` CPU-spill), verified with two integration PASSes on this
box; `"nf4"`/`"int8"` remain available for hardware where the quantized
footprint fits.
"""
from __future__ import annotations

import torch
from transformers import AutoProcessor, BitsAndBytesConfig, DiffusionGemmaForBlockDiffusion

from .types import DGemmaModel

DEFAULT_REPO_ID = "google/diffusiongemma-26B-A4B-it"

_QUANT_CHOICES = ("nf4", "int8", "none")
_QUANT_DTYPE_LABELS = {"nf4": "float16", "int8": "int8", "none": "bfloat16"}

# ONE-MINT: the widget default (nodes/loader.py) and this function's own
# default both source from here, so there is exactly one place that decides
# what a fresh graph starts with. Flipped from "nf4" (2026-07-05, issue #4):
# "nf4" OOMs structurally on this 48GB box — bitsandbytes only quantizes
# `nn.Linear`/`Conv1D` modules, and DiffusionGemma's fused 3D MoE expert
# weights are neither, so NF4 only shrinks ~1B of the model's 26B params and
# the quantized load still needs ~46GiB on a single card (`loose-ends.md`'s
# 2026-07-05 bnb-MoE entry). "none" (bf16, `device_map="auto"` CPU-spill) has
# two verified PASSes on this box instead.
DEFAULT_QUANT = "none"


def _quantization_config(quant: str) -> BitsAndBytesConfig | None:
    """Build the bitsandbytes config for `quant`, or `None` for a full-precision load.

    Compute dtype is float16, not bfloat16, for the "nf4" path: the dev box is
    a Turing (sm_75) RTX-8000 with no native bf16 tensor-core support
    (CLAUDE.md grounded fact).
    """
    if quant == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    if quant == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if quant == "none":
        return None
    raise ValueError(f"quant must be one of {_QUANT_CHOICES}, got {quant!r}.")


def _device_map(quant: str) -> str | dict:
    """Choose the `device_map` for `from_pretrained`.

    Quantized loads (`"nf4"`/`"int8"`) with a CUDA device present pin the
    whole model to GPU 0 (`{"": 0}`) instead of `"auto"`. Observed failure
    this prevents (integration run, 2026-07-05): accelerate's `"auto"`
    placement estimate is conservative and spilled modules to CPU/disk even
    with ~46GB free — which the bnb 4-bit quantizer's `validate_environment`
    rejects outright (`quantizer_bnb_4bit.py:74`, ValueError) unless fp32 CPU
    offload is enabled. NF4 of the 26B is ~14GB, so the quantized model fits
    a single 48GB card with headroom; pinning skips the estimator entirely.

    `"auto"` is kept only where it earns its keep: the `quant="none"` path
    (a >=60GB bf16 load that genuinely may need multi-GPU sharding or CPU
    offload — full-precision has no bnb guard forbidding offload) and the
    no-CUDA fallback (let accelerate place onto whatever exists).
    """
    if quant in ("nf4", "int8") and torch.cuda.is_available():
        return {"": 0}
    return "auto"


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


def load_model(repo_id: str = DEFAULT_REPO_ID, quant: str = DEFAULT_QUANT) -> DGemmaModel:
    """Load `DiffusionGemmaForBlockDiffusion` + its processor onto `DGemmaModel`.

    `quant` accepts `"nf4"` | `"int8"` | `"none"` (default — full-precision
    bf16 load, `device_map="auto"`, CPU-spills the ~42.5GiB of unquantizable
    MoE expert params bitsandbytes can't touch). Grounded truth on this 48GB
    dev box (`loose-ends.md`, 2026-07-05 bnb-MoE entry), corrected from an
    earlier ungrounded claim that `"none"` was "not viable" here: `"none"` is
    the one that actually works, with two verified integration PASSes
    (bf16 CPU-spill); `"nf4"` is the one that OOMs structurally, because
    bitsandbytes only replaces `nn.Linear`/`Conv1D` modules and
    DiffusionGemma's fused 3D MoE expert weights
    (`DiffusionGemmaTextExperts`) are neither — NF4 only shrinks ~1B of the
    model's 26B params, so the quantized load still tries to allocate
    ~46GiB on a single card. `"nf4"`/`"int8"` are kept for hardware where the
    quantized footprint does fit.
    """
    if quant not in _QUANT_CHOICES:
        raise ValueError(f"quant must be one of {_QUANT_CHOICES}, got {quant!r}.")

    quantization_config = _quantization_config(quant)
    load_kwargs: dict = {"device_map": _device_map(quant)}
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config
    else:
        load_kwargs["dtype"] = torch.bfloat16

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(repo_id, **load_kwargs)
    processor = AutoProcessor.from_pretrained(repo_id)

    device = _resolve_device(model)

    return DGemmaModel(
        model=model,
        processor=processor,
        device=device,
        dtype=_QUANT_DTYPE_LABELS[quant],
        repo_id=repo_id,
        quant=quant,
    )
