"""surfaces/comfyui/loader.py — DGemmaLoader: thin ComfyUI adapter (ADR-CDG-003).

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

RATIFICATION 2026-07-13 — the folder_paths dropdown ships DISABLED by default
(`_LOCAL_FOLDERS_ENABLED = False`). Rationale (operator ratification feedback):
the pack's current load path is deliberately un-ComfyUI — weights come via
`from_pretrained()` out of the HF hub cache (`HF_HOME`), NOT from ComfyUI's
`models/diffusion_models`/`text_encoders` directories. A folder_paths scan
against directories the model never inhabits would present an empty (or worse,
misleadingly populated) selector, so the HF-identifier flow (`repo_id`) stays
the primary, visible input and the dropdown is scaffolding held behind the flag
until its ENABLE TRIGGER is met:

  * #15 — GGUF backend graduation (weights placed as `.gguf` under a ComfyUI
    model dir), and/or
  * #4  — an AWQ/quantized checkpoint placed conventionally under
    `models/diffusion_models` (or `text_encoders`).

Until then the scanning/resolution code + its path-traversal guard are shipped,
tested, and ready — but not wired into the visible UI. Flip `_LOCAL_FOLDERS_ENABLED`
to `True` (and remove this note's "until then" clause) on the day weights
actually live under ComfyUI model dirs. `local_files_only` and the traversal
guard remain active for the HF-cache flow regardless of the flag — they are
wanted independent of the dropdown (see `resolve_local_model_dir`'s docstring).
"""
from __future__ import annotations

import os

# Dual-context import, explicit package-depth gate (same discipline as the
# root __init__.py — no blanket try/except, which masks real failures).
# ComfyUI loads the pack as a package named after its directory path
# (`/srv/dev/ComfyUI/nodes.py:2233,2241`) and never puts the pack root on
# sys.path, so this module's __package__ is "<pack>.surfaces.comfyui"
# (dotted) and only the relative import can resolve. This module now lives
# two levels under the pack root (surfaces/comfyui/, was one level under
# nodes/), so the relative climb to dgemma/ is THREE dots, not two
# (ADR-CDG-008 Phase 1 / issue #52 risk R-1 — the riskiest step named in the
# plan).
#
# GATE CORRECTION vs #52's stated design (found during execution, not
# preempted by the plan): the gate can no longer be a bare `"." in
# __package__` check. Under pytest/standalone, this module's OWN absolute
# package name is "surfaces.comfyui" — which itself contains a dot, because
# the surface directory is two segments deep — so the naive dotted-check
# would wrongly take the relative branch even outside ComfyUI (that branch
# would then try to climb past the top-level package and raise
# ImportError). The `nodes/` layout never hit this because "nodes" was a
# single, undotted top-level segment. The real ComfyUI loader's
# `__package__` is "<synthetic-pack-name>.surfaces.comfyui" (>= 2 dots: one
# for "comfyui-under-surfaces", one for "surfaces-under-the-pack-name");
# bare pytest/standalone gives exactly "surfaces.comfyui" (1 dot). Gating on
# `__package__.count(".") >= 2` distinguishes the two correctly. Observed
# violation: graph smoke test 2026-07-05 (`loose-ends.md`) for the original
# nodes/ case; this depth-count correction is a new observed violation from
# this move's own execution. Enforcement: tests/test_comfyui_loader_context.py
# and tests/test_dual_context_import.py (the R-1 tripwires).
if __package__ and __package__.count(".") >= 2:
    from ...dgemma.model import (
        _QUANT_CHOICES,
        DEFAULT_QUANT,
        DEFAULT_REPO_ID,
        load_model,
    )
    from .socket_types import DGEMMA_MODEL
else:
    from dgemma.model import (
        _QUANT_CHOICES,
        DEFAULT_QUANT,
        DEFAULT_REPO_ID,
        load_model,
    )
    from surfaces.comfyui.socket_types import DGEMMA_MODEL

# Ratification 2026-07-13: the folder_paths dropdown is SCAFFOLDING held OFF
# until weights actually live under ComfyUI model dirs. See the module
# docstring for the enable trigger (#15 GGUF graduation / #4 conventional
# checkpoint placement). When False (the default, current state): the dropdown
# is omitted from `INPUT_TYPES` entirely — hidden, not merely de-defaulted —
# and the HF-identifier `repo_id` flow is the sole visible load path. Flip to
# True on the trigger day; the scan/resolve functions and their tests already
# ship, so enabling is a one-line change, not a re-implementation.
_LOCAL_FOLDERS_ENABLED = False

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
from huggingface_hub.errors import LocalEntryNotFoundError

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
        # PRIMARY, always-visible flow: the HF identifier. Weights resolve via
        # `from_pretrained()` out of the HF hub cache (`HF_HOME`) — the pack's
        # deliberate (un-ComfyUI) load path (ratification 2026-07-13).
        required = {
            # "none" = full bf16 (~53GB VRAM); "autoround" = pre-quantized
            # W4A16 INT4 checkpoint (~30GB VRAM, requires auto-round extra).
            # See issue #128.
            "quant": (list(_QUANT_CHOICES), {
                "default": DEFAULT_QUANT,
                "tooltip": "none = full bf16 (~53GB VRAM) · autoround = pre-quantized INT4 (~30GB VRAM, requires auto-round extra)",
            }),
            # Off by default: keep the HF download-and-cache behavior. On:
            # forces both from_pretrained calls to resolve only from the local
            # HF cache (no network) — useful once a checkpoint is already
            # cached (e.g. tokenizer-only test runs). Kept active regardless of
            # the folder_paths flag (ratification 2026-07-13): it applies to the
            # HF-cache flow too and is wanted independent of the dropdown.
            "local_files_only": ("BOOLEAN", {"default": False}),
        }
        spec: dict = {"required": required}

        # SCAFFOLDING, hidden by default: the folder_paths dropdown is only
        # surfaced once `_LOCAL_FOLDERS_ENABLED` is flipped (enable trigger:
        # #15 GGUF graduation / #4 conventional checkpoint placement — see the
        # module docstring). Until then it is omitted from INPUT_TYPES entirely
        # so ComfyUI renders no empty/misleading selector. When enabled it lands
        # in `optional` (advanced/local-folders path), NOT `required`, so the
        # HF-identifier flow stays the primary one a user reaches for.
        if _LOCAL_FOLDERS_ENABLED:
            spec["optional"] = {"local_model_dir": (list_local_model_dirs(),)}
        return spec

    RETURN_TYPES = (DGEMMA_MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "DiffusionGemma"

    def load(
        self,
        quant: str,
        repo_id: str = DEFAULT_REPO_ID,  # compat param — ignored, hardcoded below
        local_files_only: bool = False,
        local_model_dir: str | None = None,
    ):
        # Advanced/local-folders path (only reachable when the dropdown is
        # enabled AND a selection was made): resolve the dropdown pick through
        # the path-traversal guard (`resolve_local_model_dir`) and force
        # local_files_only — a resolved local directory is never a network
        # fetch. The guard stays active here (ratification 2026-07-13) even
        # though the dropdown is scaffolding: a `/prompt` POST can carry a
        # `local_model_dir` string regardless of what the UI renders.
        if _LOCAL_FOLDERS_ENABLED and local_model_dir:
            model_path = resolve_local_model_dir(local_model_dir)
            if model_path is None:
                raise RuntimeError(
                    f"DGemmaLoader: could not resolve local_model_dir={local_model_dir!r} "
                    "to a model directory under the configured 'diffusion_models' or "
                    "'text_encoders' folders. Place the DiffusionGemma checkpoint "
                    "(a directory containing config.json) under one of those ComfyUI "
                    "model folders — the local-folders path never falls back to a "
                    "network fetch."
                )
            return (load_model(repo_id=model_path, quant=quant, local_files_only=True),)

        # PRIMARY path: HF identifier. Try offline-first (skip all HEAD
        # requests when cached), fall back to network on cache miss.
        # The widget's local_files_only toggle is honored — if the user
        # explicitly set it True, we never retry online; if False (default)
        # or omitted, we try offline first and only hit the network on miss.
        try:
            return (load_model(repo_id=DEFAULT_REPO_ID, quant=quant, local_files_only=True),)
        except LocalEntryNotFoundError:
            if local_files_only:
                raise  # user explicitly requested offline — don't retry
            return (load_model(repo_id=DEFAULT_REPO_ID, quant=quant, local_files_only=False),)
