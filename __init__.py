"""ComfyUI-DiffusionGemma — node pack entry point.

STATUS: design-only. This pack registers ZERO nodes on purpose. The design
lives in decisions/ and plan.md; the first real nodes (DGemmaLoader,
DGemmaSampler) land in Phase 1 of plan.md.

ComfyUI discovers a custom node pack by importing this module and reading
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS. They are intentionally empty
so that installing this pack today is a no-op in the node menu rather than a
crash — honest emptiness over a fake surface.

When Phase 1 nodes exist, populate these mappings (or aggregate them from a
`nodes/` package) here.
"""

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
