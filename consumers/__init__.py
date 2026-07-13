"""Consumers — the parent package for downstream analysis over the `dgemma/`
core's emitted `CanvasTrace`.

Empty package marker (ADR-CDG-008 Phase 3, issue #55 §2 recommendation) —
mirrors `surfaces/__init__.py`'s shape. Intentionally holds no logic and no
re-exports: a re-export here would recreate one tier up the exact "public
face re-exports analysis" smell the relocation exists to remove from
`dgemma/__init__.py` (issue #55 §2, risk R4). Import `consumers.analysis`
directly for the pure trace-analysis functions
(`build_commit_heatmap`, `build_avalanche_curve`, `corroborate_no_mask_token`,
`MaskTokenCorroboration`). This module is also the target
`tests/test_seam.py` asserts absence of in `dgemma`'s import graph (core
imports no analysis, CDG-008 Phase 4).
"""
