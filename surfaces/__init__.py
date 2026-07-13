"""Surfaces — the parent package for peer surfaces over the `dgemma/` core.

Empty package marker (ADR-CDG-008 Phase 1, issue #52). Needed so
`surfaces.comfyui` (this phase) and the future `surfaces.mcp` (Phase 2)
resolve as subpackages of one `surfaces` namespace. This module intentionally
holds no logic — it is also the target `tests/test_seam.py` asserts absence
of in `dgemma`'s import graph (core imports no surface).
"""
