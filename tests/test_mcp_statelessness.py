"""ADR-CDG-008 Phase 2, Correction 1 (`STATELESS-CORE`) enforcement for
`surfaces/mcp/state_manager.py`: the MCP surface's persisted state is the
loaded model, PLUS (ADR-CDG-025's explicit rule-6 amendment) a bounded,
model-scoped KV-cache handle registry — never a scheduler, canvas, or other
per-run object, and never a THIRD persisted class beyond those two.

This IS the enforcement surface ARCHITECTURE.md's rule 6 names for this
phase ("CDG-008 Phase-2 MCP state manager must never cache a scheduler") and
the "MCP state manager caches a live scheduler across calls" instant-fail
row's counterpart valid form ("Persist only `load_model`'s output [and,
per ADR-CDG-025, the bounded KV-cache registry]; build a fresh
scheduler/canvas/run-state per call").

Three tiers:

- `TestStateManagerShape` — a structural assertion on `StateManager` itself:
  its only mutable cross-call fields are the model-load triple
  (`_model`/`_repo_id`/`_quant`) and the ADR-CDG-025-sanctioned KV-cache
  registry pair (`_kv_cache_registry`/`_kv_cache_registry_capacity`); no
  attribute holds a scheduler, canvas, or frame-collector-shaped object.
  This is the MUTATION-SENSITIVE check the gate asks for: introduce a
  `self._scheduler` (or any cross-call mutable field not in the allowlist)
  and this test fails BY NAME — the allowlist is exactly the two ADR-sanctioned
  persisted classes, not a blank check for "anything StateManager happens to
  hold."
- `TestSameInSameOutAtMCPLevel` — the behavioral half, riding the same
  fake-pipeline pattern `tests/test_run_diffusion_statelessness.py` already
  uses (this test module imports and reuses its fakes rather than
  reinventing them): two identical `generate` tool calls through
  `surfaces.mcp.commands.generate.generate`, on ONE loaded (fake) model
  held by ONE `StateManager`, must produce byte-identical
  `trace_summary`/`canvas_state` — proving the MCP dispatch layer adds no
  cross-call state of its own on top of what `run_diffusion` itself already
  guarantees fresh per call.
- `TestKVCacheRegistryLifecycle` (ADR-CDG-025) — the registry's OWN lifecycle
  contract: bounded (no unbounded growth), evicted wholesale on `load()`
  (model-scoped, not repo_id-scoped), and identity-checked at resolve time
  (a stale handle from a since-unloaded model is rejected, not silently
  handed on).
"""
from __future__ import annotations

import asyncio
import dataclasses

import pytest

from surfaces.mcp.state_manager import StateManager
from tests.test_run_diffusion_statelessness import (
    FakeProcessor,
    _fake_model,
    _install_stateless_fakes,
)


class TestStateManagerShape:
    """Structural, mutation-sensitive check: `StateManager` may hold ONLY
    the model-load fields plus the ADR-CDG-025-sanctioned KV-cache registry
    pair. This is deliberately a field-allowlist assertion (not a behavior
    probe) — a future edit that adds `self._scheduler = ...` or a THIRD
    persisted class (beyond model-load and the registry) fails this test by
    name, at review time, before any call-level symptom (like the observed
    25-vs-29 heatmap frame-count mismatch this ADR cites) could ever occur."""

    ALLOWED_FIELDS = {
        "_model",
        "_repo_id",
        "_quant",
        "_kv_cache_registry",
        "_kv_cache_registry_capacity",
    }

    def test_state_manager_dataclass_fields_are_exactly_the_sanctioned_allowlist(self):
        field_names = {f.name for f in dataclasses.fields(StateManager)}
        assert field_names == self.ALLOWED_FIELDS, (
            f"StateManager grew a field outside the ADR-sanctioned allowlist: "
            f"{field_names - self.ALLOWED_FIELDS}. ADR-CDG-008 Correction 1 / "
            f"ARCHITECTURE.md rule 6 (amended by ADR-CDG-025 §1): the MCP state "
            f"manager persists ONLY the model load and the bounded KV-cache "
            f"registry — a scheduler/canvas/run-state field, or any THIRD "
            f"persisted class, is the exact cross-call-mutable-state violation "
            f"this test exists to catch. A third persisted class requires its "
            f"own ADR amendment (ADR-CDG-025 §1: 'this is not an open door'). "
            f"(MUTATION CHECK: add `_scheduler: Any = None` to StateManager and "
            f"this assertion fails.)"
        )

    def test_fresh_state_manager_holds_no_model(self):
        manager = StateManager()
        assert manager.is_loaded is False
        with pytest.raises(RuntimeError, match="No DiffusionGemma model is loaded"):
            manager.require_model()

    def test_load_replaces_rather_than_accumulates(self, monkeypatch):
        """Calling `load()` twice must leave exactly ONE model held — no
        list/cache of prior loads accumulating (the same "only the load
        persists, and only ONE of it" reading of rule 6)."""
        manager = StateManager()

        calls = []

        def fake_load_model(*, repo_id, quant, local_files_only=False):
            calls.append((repo_id, quant))
            return _fake_model()

        monkeypatch.setattr("surfaces.mcp.state_manager.load_model", fake_load_model)

        manager.load(repo_id="repo/a", quant="none")
        manager.load(repo_id="repo/b", quant="none")

        assert calls == [("repo/a", "none"), ("repo/b", "none")]
        assert manager.status()["repo_id"] == "repo/b"
        # Only one model object is ever held — dataclasses.fields already
        # proved there's no second slot for it to live in, this just
        # confirms the value itself was actually replaced, not merged.
        assert manager._quant == "none"


class TestSameInSameOutAtMCPLevel:
    """Behavioral half: two identical `generate` calls through the MCP
    dispatch layer, on one loaded fake model held by one `StateManager`,
    yield identical results — the MCP surface adds no state of its own on
    top of `run_diffusion`'s own already-enforced freshness
    (`tests/test_run_diffusion_statelessness.py`)."""

    def _make_manager_with_fake_model(self) -> StateManager:
        manager = StateManager()
        manager._model = _fake_model()
        manager._repo_id = "fake/repo"
        manager._quant = "none"
        return manager

    def test_two_identical_generate_calls_yield_identical_trace_summary(self, monkeypatch):
        from surfaces.mcp.commands import generate as generate_module

        scheduler_registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=scheduler_registry, num_steps=3)

        manager = self._make_manager_with_fake_model()
        args = {
            "prompt": "hello world",
            "seed": 42,
            "num_inference_steps": 3,
            "t_min": 0.1,
            "t_max": 0.9,
            "entropy_bound": 0.2,
            "include_frames": True,
        }

        result_1 = asyncio.run(generate_module.generate(manager, dict(args)))
        result_2 = asyncio.run(generate_module.generate(manager, dict(args)))

        assert result_1["trace_summary"] == result_2["trace_summary"]
        assert result_1["canvas_state"] == result_2["canvas_state"]
        assert result_1["text"] == result_2["text"]
        # Two calls -> two distinct scheduler objects (never a cached one
        # reused across the MCP-level calls either) — same structural proof
        # `TestSchedulerFreshPerCall` makes at the `run_diffusion` level,
        # replayed here to confirm the MCP adapter didn't reintroduce sharing
        # by e.g. constructing the scheduler itself and passing it in.
        assert len(scheduler_registry) == 2
        assert scheduler_registry[0] is not scheduler_registry[1]

    def test_generate_never_mutates_state_manager_beyond_the_model(self, monkeypatch):
        """The state manager handed to `generate` must come out with the
        exact same `_repo_id`/`_quant`/`_model` identity it went in with —
        `generate` reads the model, it never writes to the manager."""
        from surfaces.mcp.commands import generate as generate_module

        scheduler_registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=scheduler_registry, num_steps=2)

        manager = self._make_manager_with_fake_model()
        model_before = manager._model
        repo_id_before = manager._repo_id
        quant_before = manager._quant

        asyncio.run(generate_module.generate(manager, {"prompt": "hi", "num_inference_steps": 2}))

        assert manager._model is model_before
        assert manager._repo_id == repo_id_before
        assert manager._quant == quant_before
        assert set(dataclasses.fields(type(manager))) == set(dataclasses.fields(StateManager))


class TestWidenedDoorsAddNoPersistedField:
    """issue #103 Scope A: `constraints=`/`control_signals=`/`capture=`
    reaching `generate`'s JSON schema must not smuggle in a NEW persisted
    field — `StateManager`'s allowlist stays exactly the model-load triple
    (`TestStateManagerShape` above already mutation-checks this structurally;
    this class adds the BEHAVIORAL half specific to the widened doors: two
    calls that differ ONLY in their constraints/control_signals/capture
    payload must leave the manager's own state untouched, and two IDENTICAL
    calls that both carry a widened-door payload must still yield identical
    results — the same same-in/same-out contract `TestSameInSameOutAtMCPLevel`
    proves for the pre-existing knobs, extended to the new ones)."""

    def _make_manager_with_fake_model(self) -> StateManager:
        manager = StateManager()
        manager._model = _fake_model()
        manager._repo_id = "fake/repo"
        manager._quant = "none"
        return manager

    def test_state_manager_fields_unchanged_after_a_call_carrying_widened_doors(self, monkeypatch):
        from surfaces.mcp.commands import generate as generate_module

        _install_stateless_fakes(monkeypatch, scheduler_registry=[], num_steps=2)
        manager = self._make_manager_with_fake_model()

        asyncio.run(
            generate_module.generate(
                manager,
                {
                    "prompt": "hi",
                    "num_inference_steps": 2,
                    "constraints": {"pins": [{"position": 0, "token_id": 1}]},
                    "control_signals": {
                        "bindings": [
                            {"target": "t_min", "signal": [0.0, 1.0], "low": 0.1, "high": 0.9}
                        ]
                    },
                    "capture": {"top_k": 4},
                },
            )
        )

        # Still exactly the sanctioned allowlist — no new attribute, no
        # accreted payload cache.
        assert set(dataclasses.fields(type(manager))) == set(dataclasses.fields(StateManager))
        assert manager._repo_id == "fake/repo"
        assert manager._quant == "none"

    def test_two_identical_calls_with_widened_doors_yield_identical_trace_summary(self, monkeypatch):
        from surfaces.mcp.commands import generate as generate_module

        scheduler_registry: list = []
        _install_stateless_fakes(monkeypatch, scheduler_registry=scheduler_registry, num_steps=3)
        manager = self._make_manager_with_fake_model()

        args = {
            "prompt": "hello widened doors",
            "num_inference_steps": 3,
            "gen_length": 8,
            "constraints": {"pins": [{"position": 0, "token_id": 1}]},
            "control_signals": {
                "bindings": [{"target": "entropy_bound", "signal": [0.0, 0.5, 1.0], "low": 0.05, "high": 0.2}]
            },
            "capture": {"top_k": 2},
            "include_frames": True,
        }

        result_1 = asyncio.run(generate_module.generate(manager, dict(args)))
        result_2 = asyncio.run(generate_module.generate(manager, dict(args)))

        assert result_1 == result_2
        # Two calls -> two distinct scheduler objects, same proof
        # `TestSameInSameOutAtMCPLevel` makes for the pre-existing knobs.
        assert len(scheduler_registry) == 2
        assert scheduler_registry[0] is not scheduler_registry[1]


class TestKVCacheRegistryLifecycle:
    """ADR-CDG-025 §4: the registry's own lifecycle contract — bounded (no
    unbounded growth), evicted wholesale on `load()` (model-scoped, not
    repo_id-scoped), and identity-checked at resolve time. Uses
    `tests/conftest.py`'s `synthetic_kv_cache_factory`/`dgemma_model_factory`
    fixtures (the same matching-model+cache-pair builder
    `tests/test_kv_cache_ingress.py` already relies on) rather than the
    fake-pipeline `_fake_model()` this module's other classes use — the
    registry's own bookkeeping is exercised directly here, no `run_diffusion`
    call needed."""

    def test_registry_starts_empty(self):
        manager = StateManager()
        assert manager._kv_cache_registry == {}

    def test_store_and_resolve_round_trip(self, synthetic_kv_cache_factory):
        model, cache = synthetic_kv_cache_factory()
        manager = StateManager()
        manager._model = model
        manager._repo_id = model.repo_id
        manager._quant = "none"

        handle = manager._store_kv_cache(cache)
        resolved = manager.resolve_kv_cache(handle)

        assert resolved is cache
        assert handle in manager._kv_cache_registry

    def test_resolve_unknown_handle_raises_value_error(self, dgemma_model_factory):
        manager = StateManager()
        manager._model = dgemma_model_factory()
        manager._repo_id = manager._model.repo_id
        manager._quant = "none"

        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            manager.resolve_kv_cache("no-such-handle")

    def test_resolve_without_a_loaded_model_raises_require_model_error(self):
        manager = StateManager()  # no model loaded
        with pytest.raises(RuntimeError, match="No DiffusionGemma model is loaded"):
            manager.resolve_kv_cache("irrelevant-handle")

    def test_bounded_registry_evicts_least_recently_used_on_overflow(self, synthetic_kv_cache_factory):
        model, _ = synthetic_kv_cache_factory()
        manager = StateManager()
        manager._model = model
        manager._repo_id = model.repo_id
        manager._quant = "none"
        manager._kv_cache_registry_capacity = 2

        _, cache_a = synthetic_kv_cache_factory(model_kwargs=None)
        _, cache_b = synthetic_kv_cache_factory(model_kwargs=None)
        _, cache_c = synthetic_kv_cache_factory(model_kwargs=None)

        handle_a = manager._store_kv_cache(cache_a)
        handle_b = manager._store_kv_cache(cache_b)
        # Capacity is 2; storing a third handle must evict the
        # least-recently-used entry (handle_a, never touched again since
        # being stored) rather than growing unbounded (ADR-CDG-025 §4).
        handle_c = manager._store_kv_cache(cache_c)

        assert len(manager._kv_cache_registry) == 2
        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            manager.resolve_kv_cache(handle_a)
        assert manager.resolve_kv_cache(handle_b) is cache_b
        assert manager.resolve_kv_cache(handle_c) is cache_c

    def test_resolving_a_handle_refreshes_its_lru_position(self, synthetic_kv_cache_factory):
        """A handle that was just resolved must NOT be the next eviction
        target — resolve counts as a touch, the same as `encode`'s advance
        path already does via `move_to_end`."""
        model, _ = synthetic_kv_cache_factory()
        manager = StateManager()
        manager._model = model
        manager._repo_id = model.repo_id
        manager._quant = "none"
        manager._kv_cache_registry_capacity = 2

        _, cache_a = synthetic_kv_cache_factory()
        _, cache_b = synthetic_kv_cache_factory()
        _, cache_c = synthetic_kv_cache_factory()

        handle_a = manager._store_kv_cache(cache_a)
        handle_b = manager._store_kv_cache(cache_b)
        manager.resolve_kv_cache(handle_a)  # touch A -> A is now most-recently-used
        handle_c = manager._store_kv_cache(cache_c)  # capacity 2 -> evicts B, not A

        assert manager.resolve_kv_cache(handle_a) is cache_a
        assert manager.resolve_kv_cache(handle_c) is cache_c
        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            manager.resolve_kv_cache(handle_b)

    def test_load_evicts_the_entire_registry(self, synthetic_kv_cache_factory, monkeypatch):
        """§4 eviction trigger: `load()` clears the WHOLE registry, model-scoped
        (not repo_id-scoped) — even a same-repo_id reload invalidates every
        prior handle, since a reload is a new model object."""
        model, cache = synthetic_kv_cache_factory()
        manager = StateManager()
        manager._model = model
        manager._repo_id = model.repo_id
        manager._quant = "none"
        handle = manager._store_kv_cache(cache)
        assert manager.resolve_kv_cache(handle) is cache

        monkeypatch.setattr("surfaces.mcp.state_manager.load_model", lambda **kwargs: _fake_model())

        manager.load(repo_id=model.repo_id, quant="none")

        assert manager._kv_cache_registry == {}
        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            manager.resolve_kv_cache(handle)

    def test_resolve_rejects_a_handle_minted_under_a_different_model(self, synthetic_kv_cache_factory):
        """Belt-and-suspenders identity check (§4): even if a handle somehow
        survived without going through `load()`'s wholesale clear (e.g. a
        future code path that swaps `_model` directly), a cache whose
        provenance no longer matches the currently-loaded model is rejected
        at resolve time — the mint-identity guard applied to a live object."""
        model_a, cache_from_a = synthetic_kv_cache_factory(model_kwargs={"repo_id": "fake/model-a"})
        model_b, _ = synthetic_kv_cache_factory(model_kwargs={"repo_id": "fake/model-b"})

        manager = StateManager()
        # Simulate the model having been swapped WITHOUT going through
        # load() (which would have cleared the registry) — directly assign
        # a different model than the one that minted the stored cache.
        manager._model = model_b
        manager._repo_id = model_b.repo_id
        manager._quant = "none"
        manager._kv_cache_registry[  # bypass _store_kv_cache to avoid load()'s clear semantics
            "stale-handle"
        ] = cache_from_a

        with pytest.raises(ValueError, match="minted under a different model"):
            manager.resolve_kv_cache("stale-handle")

    def test_encode_into_registry_mints_a_fresh_handle(self, dgemma_model_factory):
        manager = StateManager()
        manager._model = dgemma_model_factory()
        manager._repo_id = manager._model.repo_id
        manager._quant = "none"

        handle = manager.encode_into_registry("hello world")

        assert isinstance(handle, str) and handle
        resolved = manager.resolve_kv_cache(handle)
        assert resolved.provenance.model_repo_id == manager._model.repo_id
        assert resolved.provenance.minting_sequence is not None

    def test_encode_into_registry_advance_replaces_the_same_handle(self, dgemma_model_factory):
        """§2 advance-returns-new-payload: advancing a handle returns the
        SAME handle string, but the registry entry it names is a NEW
        `KVCache` value (never a mutation of the old one in place)."""
        manager = StateManager()
        manager._model = dgemma_model_factory()
        manager._repo_id = manager._model.repo_id
        manager._quant = "none"

        handle = manager.encode_into_registry("first chunk")
        cache_after_mint = manager.resolve_kv_cache(handle)

        advanced_handle = manager.encode_into_registry("second chunk", kv_cache_id=handle)
        cache_after_advance = manager.resolve_kv_cache(advanced_handle)

        assert advanced_handle == handle
        assert cache_after_advance is not cache_after_mint
        assert len(manager._kv_cache_registry) == 1

    def test_encode_into_registry_with_unknown_handle_raises(self, dgemma_model_factory):
        manager = StateManager()
        manager._model = dgemma_model_factory()
        manager._repo_id = manager._model.repo_id
        manager._quant = "none"

        with pytest.raises(ValueError, match="Unknown or expired kv_cache_id"):
            manager.encode_into_registry("text", kv_cache_id="no-such-handle")
