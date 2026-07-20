"""ComfyUI-DiffusionGemma — node pack entry point.

STATUS: Phase 3 (plan.md) — instrumentation. `DGemmaLoader` + `DGemmaSampler`
(P1) + `DGemmaTrace` (P3) land here; prompt in, text + validity readout +
canvas trace out, plus a live per-step view via the `web/` extension.
`DGemmaSampler` (issue #21, reworked) also emits a `frames_image` `IMAGE`
batch — the same per-step `frames` STRING series rendered watchable/
shareable, alongside the STRING itself, rather than a separate node.
`DGemmaTallyAudit` (issue #84) takes that same `frames` STRING list and
audits a "count the numerals" task's per-step tally claims against the
model's own restated evidence — `surfaces/comfyui/tally_audit.py` wrapping
`consumers/tally_audit.py`'s pure functions, same composition pattern as
`DGemmaTrace`/`consumers/analysis.py`. `DGemmaRunLogWriter` (issue #72)
writes a schema'd JSONL run log — a SaveImage-convention node that owns its
own file handle (`surfaces/comfyui/run_log_writer.py` wrapping
`consumers/run_log.py`'s pure builders), superseding the escaped-newline-
trap-prone `.txt` step-log convention `consumers/tally_audit.py`'s
`extract_decoded_frames_from_composite_blob` was built to reverse.
`DGemmaEncode`/`DGemmaDenoise` (ADR-CDG-012, issue #62 Phase 3) are the
`KV_CACHE` seam's node pair — mint/advance a `DGEMMA_KV_CACHE` payload from
text, and optionally consume one to condition a run (the decoder is not yet
driven off the injected cache's tensors — Phase 4, gated on the ADR's
real-weights de-risk smoke test).

ComfyUI discovers a custom node pack by importing this module and reading
NODE_CLASS_MAPPINGS / NODE_DISPLAY_NAME_MAPPINGS (+ `WEB_DIRECTORY`, checked
by `nodes.py:2269-2272` and mounted into `EXTENSION_WEB_DIRS`, served at
`/extensions/<pack-dir-basename>` per `server.py:1225-1226`). Aggregated
from `surfaces/comfyui/` (ADR-CDG-003; relocated from `nodes/` per
ADR-CDG-008 Phase 1, issue #52) — nothing else lives here.

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
    from .surfaces.comfyui.denoise import DGemmaDenoise
    from .surfaces.comfyui.encode import DGemmaEncode
    from .surfaces.comfyui.loader import DGemmaLoader
    from .surfaces.comfyui.run_log_writer import DGemmaRunLogWriter
    from .surfaces.comfyui.sampler import DGemmaSampler
    from .surfaces.comfyui.tally_audit import DGemmaTallyAudit
    from .surfaces.comfyui.trace import DGemmaTrace
else:
    from surfaces.comfyui.denoise import DGemmaDenoise
    from surfaces.comfyui.encode import DGemmaEncode
    from surfaces.comfyui.loader import DGemmaLoader
    from surfaces.comfyui.run_log_writer import DGemmaRunLogWriter
    from surfaces.comfyui.sampler import DGemmaSampler
    from surfaces.comfyui.tally_audit import DGemmaTallyAudit
    from surfaces.comfyui.trace import DGemmaTrace

NODE_CLASS_MAPPINGS: dict = {
    "DGemmaLoader": DGemmaLoader,
    "DGemmaSampler": DGemmaSampler,
    "DGemmaTrace": DGemmaTrace,
    "DGemmaTallyAudit": DGemmaTallyAudit,
    "DGemmaRunLogWriter": DGemmaRunLogWriter,
    "DGemmaEncode": DGemmaEncode,
    "DGemmaDenoise": DGemmaDenoise,
}
NODE_DISPLAY_NAME_MAPPINGS: dict = {
    "DGemmaLoader": "DiffusionGemma Loader",
    "DGemmaSampler": "DiffusionGemma Sampler",
    "DGemmaTrace": "DiffusionGemma Trace",
    "DGemmaTallyAudit": "DiffusionGemma Tally Audit",
    "DGemmaRunLogWriter": "DiffusionGemma Run Log Writer",
    "DGemmaEncode": "DiffusionGemma Encode (KV Cache)",
    "DGemmaDenoise": "DiffusionGemma Denoise (KV Cache)",
}

# P3 (a): the live per-step view (`surfaces/comfyui/web/live_view.js`).
# Relative to this file's own directory, per `nodes.py:2269-2272`'s
# `os.path.join(module_dir, WEB_DIRECTORY)` resolution.
WEB_DIRECTORY = "./surfaces/comfyui/web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
