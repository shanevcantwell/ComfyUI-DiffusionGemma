"""ComfyUI-DiffusionGemma — node pack entry point.

STATUS: Phase 1 (plan.md) — thin vertical slice. `DGemmaLoader` +
`DGemmaSampler` land here; prompt in, text + validity readout out.

ComfyUI discovers a custom node pack by importing this module and reading
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS. Aggregated from `nodes/`
(ADR-CDG-003) — nothing else lives here.

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
else:
    from nodes.loader import DGemmaLoader
    from nodes.sampler import DGemmaSampler

NODE_CLASS_MAPPINGS: dict = {
    "DGemmaLoader": DGemmaLoader,
    "DGemmaSampler": DGemmaSampler,
}
NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "DGemmaLoader": "DiffusionGemma Loader",
    "DGemmaSampler": "DiffusionGemma Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
