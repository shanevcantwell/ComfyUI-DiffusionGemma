"""nodes — thin ComfyUI adapters (ADR-CDG-003). No logic; see per-module files.

Naming note for the downstream ComfyUI-launch bracket (not exercised this
phase): ComfyUI's own process has a root-level `nodes.py` module. The root
`__init__.py` here reaches this package via a *relative* import
(`from .nodes.loader import ...`), which cannot collide with that module
regardless of what is already in `sys.modules`. Code inside this package must
not switch to an absolute `import nodes` — that would risk resolving to
ComfyUI's own module instead of this one once a real ComfyUI process is
involved.
"""
