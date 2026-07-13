"""nodes/loader.py — DGemmaLoader: thin ComfyUI adapter (ADR-CDG-003).

Unpacks widget inputs, calls one `dgemma.*` function, wraps the result in a
tuple. No logic lives here — if a `for` loop or a loading decision ever
creeps into this file, it belongs in `dgemma/model.py`, not here.

Issue #17 — folder_paths dropdown retrofit. `folder_paths` is ComfyUI-side
(the running server owns the configured model-search roots), so the
scanning/resolution glue lives HERE, in the node layer, not in
`dgemma/model.py` — `dgemma/` stays ComfyUI-agnostic (ADR-CDG-003). The seam
is: this module turns "what local model dirs exist, and which one did the
user pick" into a plain local filesystem path; `dgemma.model.load_model`
still only ever sees a path/repo_id string and a `local_files_only` bool, no
different than before.
"""
from __future__ import annotations

import os

# Dual-context import, explicit package-depth gate (same discipline as the
# root __init__.py — no blanket try/except, which masks real failures).
# ComfyUI loads the pack as a package named after its directory path
# (`/srv/dev/ComfyUI/nodes.py:2233,2241`) and never puts the pack root on
# sys.path, so this module's __package__ is "<pack>.nodes" (dotted) and only
# the relative `..dgemma` can resolve. Under pytest/standalone the repo root
# is on sys.path and this module is top-level "nodes" (no dot), so only the
# absolute form can resolve. Observed violation: graph smoke test 2026-07-05
# (`loose-ends.md`); enforcement: tests/test_comfyui_loader_context.py.
if __package__ and "." in __package__:
    from ..dgemma.model import DEFAULT_QUANT, load_model
else:
    from dgemma.model import DEFAULT_QUANT, load_model

# `folder_paths` is a ComfyUI-runtime module: real inside a live ComfyUI
# process (its repo root is on sys.path at startup — `ComfyUI/main.py`), and
# genuinely absent under pytest/standalone (no such package on PyPI). This is
# the same "one specific import may legitimately not exist here" situation
# `dgemma/model.py` already has for `transformers`, and gets the same
# treatment: a narrow `try/except ImportError` around exactly one import,
# never a blanket catch that would also swallow unrelated bugs. It is NOT
# the dual-context gate above (that branches on a deterministic `__package__`
# signal that is always one of two known values); here the signal is
# "importable or not" itself, so ImportError is the correct, narrow gate.
try:
    import folder_paths
except ImportError:
    folder_paths = None

# Both scanned, unioned (issue #17): DiffusionGemma is a DiT-peer by role
# (a denoising loader analogous to UNETLoader) but Gemma-lineage by weights,
# so users may reasonably drop a checkpoint under either ComfyUI models
# folder. Order matters only for de-duplication below (first hit wins), not
# for correctness — a name present under both folders resolves to whichever
# is scanned first.
_MODEL_FOLDER_KEYS = ("diffusion_models", "text_encoders")

# transformers/HF checkpoints are a SHARD DIRECTORY (config.json,
# tokenizer files, one-or-more model-*.safetensors), not a single file, so
# `folder_paths.get_filename_list` (file-extension search, never returns
# bare directory names — `folder_paths.py`'s own `recursive_search` walks to
# files only) cannot enumerate them directly, and `folder_paths.get_full_path`
# cannot resolve them either (it requires `os.path.isfile`, not `isdir`).
# Decision: scan each folder key's configured roots (`get_folder_paths`) one
# level deep with `os.listdir`, and treat an immediate subdirectory as a
# valid model dir iff it contains `config.json` — the one file every HF
# `transformers` checkpoint always has, and cheap to `os.path.isfile`-check
# without touching (or even listing) the multi-GB safetensors shards
# themselves. Rejected alternative: list every subdirectory unconditionally
# (no sentinel) — that would also surface unrelated junk directories (empty
# folders, partial downloads, non-model dirs a user happens to keep there),
# offering a name in the dropdown that fails at load time instead of not
# appearing at all. See PR body for this as a ratification question.
_MODEL_DIR_SENTINEL_FILE = "config.json"


def _is_model_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, _MODEL_DIR_SENTINEL_FILE))


def list_local_model_dirs() -> list[str]:
    """Names (not full paths) of local model directories found under the
    configured `diffusion_models` + `text_encoders` roots, unioned and
    sorted. Empty (never raises) when `folder_paths` is unavailable (outside
    ComfyUI) or no roots contain a valid model dir — an empty dropdown is the
    honest ComfyUI idiom for "nothing found" (matches stock loaders such as
    `CheckpointLoaderSimple`/`UNETLoader`, which pass `get_filename_list`'s
    result straight through with no placeholder sentinel).
    """
    if folder_paths is None:
        return []

    names: set[str] = set()
    for key in _MODEL_FOLDER_KEYS:
        try:
            roots = folder_paths.get_folder_paths(key)
        except KeyError:
            # A folder key ComfyUI hasn't registered in this install (e.g. an
            # older ComfyUI predating "text_encoders") — skip it, not fatal.
            continue
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in os.listdir(root):
                full = os.path.join(root, entry)
                if os.path.isdir(full) and _is_model_dir(full):
                    names.add(entry)
    return sorted(names)


def resolve_local_model_dir(name: str) -> str | None:
    """The full local path for a dropdown-selected directory `name`, or
    `None` if it cannot be resolved against either configured folder key's
    roots. First match wins (mirrors `folder_paths.get_full_path`'s own
    "first configured root that has it" semantics).

    `name` is REJECTED (not sanitized-and-continued) unless it is exactly
    one path component with no separator: ComfyUI's COMBO widget type is
    UI-only guidance, not a server-side enforced enum — a `/prompt` POST can
    send any string for `model_name`, dropdown or not. Without this check,
    `os.path.join(root, name)` for an absolute `name` silently discards
    `root` entirely (documented `os.path.join` behavior), and a `../`-laden
    `name` walks out of the configured model folder — either way handing an
    attacker-chosen directory straight to `from_pretrained(local_files_only=
    True)`. One-component-only is a stricter, more legible guard than
    replicating `folder_paths.get_full_path`'s `os.path.relpath(os.path.join(
    "/", filename), "/")` normalize-and-clamp trick, and it matches this
    function's actual contract: a dropdown entry is always a bare directory
    name, never a nested path.
    """
    if not name or folder_paths is None:
        return None
    if os.path.basename(name) != name or name in (os.curdir, os.pardir):
        return None

    for key in _MODEL_FOLDER_KEYS:
        try:
            roots = folder_paths.get_folder_paths(key)
        except KeyError:
            continue
        for root in roots:
            full = os.path.join(root, name)
            if os.path.isdir(full) and _is_model_dir(full):
                return full
    return None


class DGemmaLoader:
    """Loads a DiffusionGemma model + processor onto the `DGEMMA_MODEL` socket."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_local_model_dirs(),),
                # "none" is the only honest option (issue #18): bitsandbytes
                # cannot quantize DiffusionGemma's fused 3D MoE experts, so
                # "nf4"/"int8" were removed from the selector rather than left
                # to silently not do what they claim — see dgemma/model.py's
                # DEFAULT_QUANT provenance comment.
                "quant": (["none"], {"default": DEFAULT_QUANT}),
            }
        }

    RETURN_TYPES = ("DGEMMA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DiffusionGemma"

    def load(self, model_name: str, quant: str):
        model_path = resolve_local_model_dir(model_name)
        if model_path is None:
            raise RuntimeError(
                f"DGemmaLoader: could not resolve model_name={model_name!r} to a local "
                "model directory under the configured 'diffusion_models' or "
                "'text_encoders' folders. Place the DiffusionGemma checkpoint "
                "(a directory containing config.json) under one of those ComfyUI "
                "model folders — this loader never falls back to a network fetch."
            )
        # local_files_only=True unconditionally: the dropdown only ever
        # offers local directories already resolved to a path, so there is
        # never a repo_id left to fetch from the network (issue #17).
        return (load_model(repo_id=model_path, quant=quant, local_files_only=True),)
