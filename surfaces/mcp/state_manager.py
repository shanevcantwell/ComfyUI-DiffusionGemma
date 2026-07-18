"""surfaces/mcp/state_manager.py — the ONLY cross-call state this surface
holds: the loaded `DGemmaModel` (ADR-CDG-008 Phase 2 Correction 1,
`STATELESS-CORE`).

Transcribed from `semantic-kinematics-mcp`'s `mcp/state_manager.py` shape (a
dataclass the server constructs once and hands to every command handler),
deliberately NARROWED against that source's own documented debt: sk-mcp's
`StateManager` retains a live `_adapter` **and** a cross-call
`_embedding_cache` (`.../mcp/state_manager.py:51-52,83-86`), which its own
ADR-SKM-0009 names as the statelessness violation still on its roadmap to fix
(`ADR-SKM-0009:71`). This `StateManager` deliberately holds nothing else: no
scheduler, no canvas, no run-state, no cache keyed on prompt/knobs. The
model load is the one object this pack's own doctrine says must persist
(the ~53GB weights, `README.md` local-run defaults) — every `generate` call
still goes through `dgemma.run_diffusion`, which builds its own fresh
`EntropyBoundScheduler` / `_FrameCollector` / `StepEndComposite` internally
(`dgemma/loop.py:run_diffusion`) — this surface adds no memoization of any
of that on top.

`load_model`/`is_loaded`/`model_status` are the whole surface: there is no
`set_backend`-equivalent, no per-call cache lookup, nothing that could grow
into a second axis of cross-call state. A future addition that stores
anything ELSE here (a scheduler, a partial canvas, a last-prompt cache) is
exactly the ARCHITECTURE.md rule-6 violation this module exists to foreclose
— see `tests/test_mcp_statelessness.py`, which mutation-checks this file
directly (asserts a hypothetical cached-scheduler shape would be caught).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

if __package__ and __package__.count(".") >= 2:
    from ...dgemma.model import load_model
    from ...dgemma.types import DGemmaModel
else:
    from dgemma.model import load_model
    from dgemma.types import DGemmaModel


@dataclass
class StateManager:
    """Holds ONLY the loaded model. Constructed once by `server.py`, passed
    to every command handler — same shape as sk-mcp's `StateManager`, with
    the embedding-cache / live-adapter cross-call state (that source's own
    named debt) not transcribed at all.
    """

    _model: Optional[DGemmaModel] = field(default=None, repr=False)
    _repo_id: Optional[str] = None
    _quant: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, *, repo_id: str, quant: str, local_files_only: bool = False) -> DGemmaModel:
        """(Re)load the model, replacing whatever was previously held.

        No implicit reuse-if-same-args short-circuit: a caller that wants to
        avoid a reload checks `model_status` first (`is_loaded` +
        `repo_id`/`quant`) and skips the call itself — this method always
        does what it's told, so "did a real load just happen" never has to
        be inferred from a cache-hit side effect.
        """
        self._model = load_model(repo_id=repo_id, quant=quant, local_files_only=local_files_only)
        self._repo_id = repo_id
        self._quant = quant
        return self._model

    def require_model(self) -> DGemmaModel:
        """The model, or a loud `RuntimeError` naming the missing precondition
        — never a silent `None` handed on to `dgemma.run_diffusion` (which
        would fail with a confusing attribute error deep inside the engine
        instead of a clear message at the door)."""
        if self._model is None:
            raise RuntimeError(
                "No DiffusionGemma model is loaded. Call the 'load_model' tool "
                "first (with an explicit repo_id + quant) before 'generate'."
            )
        return self._model

    def status(self) -> dict:
        return {
            "is_loaded": self.is_loaded,
            "repo_id": self._repo_id,
            "quant": self._quant,
            "device": self._model.device if self._model is not None else None,
        }
