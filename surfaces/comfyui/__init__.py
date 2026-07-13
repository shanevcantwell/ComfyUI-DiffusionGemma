"""surfaces/comfyui — thin ComfyUI adapters (ADR-CDG-003). No logic; see
per-module files. Relocated from `nodes/` per ADR-CDG-008 Phase 1 (issue #52).

Loader-context note — anticipated at scaffold time, OBSERVED 2026-07-05
(graph smoke test failed at custom-node import; `loose-ends.md`): ComfyUI
loads this pack as a package named after its directory path
(`/srv/dev/ComfyUI/nodes.py:2233,2241`) and puts `custom_nodes/` — never the
pack root — on sys.path. Two consequences bind this package:

- ComfyUI's own process has a root-level `nodes.py`; nothing here may do a
  bare `import nodes`, which could resolve to that module.
- Bare `from dgemma...` imports are unresolvable under the real loader.
  Every module here therefore uses the explicit package-depth gate
  (`if __package__ and "." in __package__:` → relative `...dgemma` (three
  dots — this package now sits two levels under the pack root), else
  absolute) — see `loader.py` for the full rationale. Enforcement surface:
  `tests/test_comfyui_loader_context.py`, which replays ComfyUI's exact load
  mechanics with the repo root stripped from sys.path.
"""
