"""ComfyUI-DiffusionGemma — node pack entry point.

STATUS: Phase 3 (plan.md) — instrumentation. `DGemmaLoader` + `DGemmaSampler`
(P1) + `DGemmaTrace` (P3) land here; prompt in, text + validity readout +
canvas trace out, plus a live per-step view via the `web/` extension.
`DGemmaSampler` (issue #21, reworked) also emits a `frames_image` `IMAGE`
batch — the same per-step `frames` STRING series rendered watchable/
shareable, alongside the STRING itself, rather than a separate node.

ComfyUI discovers a custom node pack by importing this module and reading
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS (+ `WEB_DIRECTORY`, checked
by `nodes.py:2269-2272` and mounted into `EXTENSION_WEB_DIRS`, served at
`/extensions/<pack-dir-basename>` per `server.py:1225-1226`). Aggregated
from `nodes/` (ADR-CDG-003) — nothing else lives here.

Import gate below (`__package__`, not try/except): ComfyUI's loader gives
this file a real package context (verified manually via
`importlib.util.spec_from_file_location(..., submodule_search_locations=[...])`),
so the relative branch is what actually runs there. pytest's own `Package`
collector, however, imports any ancestor directory's `__init__.py` standalone
— no package context, regardless of `--import-mode` — purely as a side effect
of this file existing alongside `tests/`; the absolute branch exists only to
survive that. The gate is explicitly on `__package__` rather than a blanket
`except ImportError`: inside a live ComfyUI process a blanket catch would
shadow-import ComfyUI's own top-level `nodes.py` on any real failure of the
relative import, masking the actual dependency error behind a baffling
"'nodes' is not a package" (review finding, 2026-07-05, verified
empirically by the reviewer).
"""
if __package__:
    from .nodes.loader import DGemmaLoader
    from .nodes.sampler import DGemmaSampler
    from .nodes.trace import DGemmaTrace
else:
    from nodes.loader import DGemmaLoader
    from nodes.sampler import DGemmaSampler
    from nodes.trace import DGemmaTrace

NODE_CLASS_MAPPINGS: dict = {
    "DGemmaLoader": DGemmaLoader,
    "DGemmaSampler": DGemmaSampler,
    "DGemmaTrace": DGemmaTrace,
}
NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "DGemmaLoader": "DiffusionGemma Loader",
    "DGemmaSampler": "DiffusionGemma Sampler",
    "DGemmaTrace": "DiffusionGemma Trace",
}

# P3 (a): the live per-step view (`web/live_view.js`). Relative to this
# file's own directory, per `nodes.py:2269-2272`'s
# `os.path.join(module_dir, WEB_DIRECTORY)` resolution.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
