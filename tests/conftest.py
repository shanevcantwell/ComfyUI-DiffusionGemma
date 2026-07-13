"""Shared fixtures for the mocked half of the suite plus the `live` half
(see `tests/README.md` for the `pytest` vs `pytest -m live` convention).

One gate, not two: `test_integration.py` and `test_live_seams.py` both need
"real weights cached + a CUDA device, else skip gracefully" — kept here so
that check exists exactly once instead of being re-derived per file.
`require_live_weights` SKIPS (never errors) when either precondition is
missing, so `pytest -m live` on a box without the checkpoint/GPU reports
skips, not failures.

The second half of this module (`FakeFrozenConfig` down to the
`fake_pipeline_factory`/`fake_pipeline` fixtures) is #35 R4: the shared
fake-pipeline/scheduler fixture that gates R1's composition-ordering tests
and R5's hook-lifecycle/statelessness tests. Before this, each test module
that needed a fake pipeline hand-rolled its own inline `FakeScheduler`/
`FakePipeline` pair (see `tests/test_run_diffusion_knobs.py`'s
`_install_fakes`, pre-R4) — boilerplate duplicated per file and, per #35
finding F7, unable to test composition/hooks/walker behavior since it fired
the callback exactly once. R4 lands the one seam every later refactor test
is written against; existing per-file fakes are NOT rewritten onto it here
(#35's delta comment scopes R4 to landing the fixture, not to an
opportunistic suite-wide migration — that migration happens if/when R1/R5
land and actually need it).

Grounding: the real seam this mirrors is `diffusers==0.39.0`'s
`DiffusionGemmaPipeline.__call__` (`pipeline_diffusion_gemma.py`) driving
`EntropyBoundScheduler` (`scheduling_entropy_bound.py`) — both pulled and
read directly (not re-derived from memory) to ground this fixture:

- Per-step loop (`pipeline_diffusion_gemma.py:356-407`): for each
  `step_idx` in `range(predictor_steps)`, call the model, call
  `self.scheduler.step(...)`, assign `canvas = scheduler_output.prev_sample`,
  then — when a callback is given — build `callback_kwargs` from the
  requested `callback_on_step_end_tensor_inputs` (each read via
  `locals()[k]`), call `callback_on_step_end(self, global_step, step_idx,
  callback_kwargs)`, and apply the return via `canvas =
  callback_outputs.pop("canvas", canvas)` — the exact `{"canvas": ...}`
  application R4's spec names. This fixture's fake pipeline reproduces that
  same three-line contract verbatim.
- `EntropyBoundScheduler.step()` (`scheduling_entropy_bound.py:118-182`)
  reads `self.config.entropy_bound`/`self.config.t_min`/`self.config.t_max`
  (config-mediated) but `self.num_inference_steps` (a **plain instance
  attribute**, not config-mediated — issue #20's own grounding, mirrored
  already in `dgemma/loop.py`'s `_FrameCollector` docstring) directly in its
  anneal formula. Both are reproduced here.
- `ConfigMixin.config` (`configuration_utils.py:609-617`) returns
  `self._internal_dict`, a `FrozenDict` (`:56-85`): `__setattr__`/
  `__setitem__` RAISE once frozen. The only real mutation path is
  `register_to_config(**kwargs)` (`:143-158`), which rebuilds
  `_internal_dict` as a brand-new `FrozenDict` wholesale — there is no
  real-class path that mutates a single config value in place. R4's spec
  and ADR-CDG-011's ingress clause both say "mutate `scheduler.config`
  only" — this fixture resolves that against the real class as "mutable
  *through* `register_to_config`," not "attribute-settable in place," and
  `FakeFrozenConfig` below reproduces the real `FrozenDict` write-raises
  behavior rather than inventing a more permissive fake that would let a
  walker-composition test (R1) pass against a shape the real scheduler
  does not actually offer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch

from dgemma.model import DEFAULT_REPO_ID


def weights_cached(repo_id: str = DEFAULT_REPO_ID) -> bool:
    """True iff `repo_id` is present in the local Hugging Face cache."""
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return False
    try:
        cache_info = scan_cache_dir()
    except Exception:
        return False
    return any(repo.repo_id == repo_id for repo in cache_info.repos)


@pytest.fixture(scope="session")
def require_live_weights():
    """Depend on this fixture from any `@pytest.mark.live` test that needs
    the real checkpoint + a CUDA device. Session-scoped: the two checks are
    cheap and their result can't change mid-run, so there is no reason to
    repeat them per test or per module.
    """
    if not weights_cached():
        pytest.skip(
            f"{DEFAULT_REPO_ID} not present in the local HF cache (~53.6GB) — "
            "skipping live test."
        )
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available — skipping live test.")


# --- #35 R4: shared fake-pipeline/scheduler fixture -------------------------


class FakeFrozenConfig:
    """Reproduces `diffusers.configuration_utils.FrozenDict`'s write-raises
    behavior (`configuration_utils.py:56-85`) for exactly the keys
    `EntropyBoundScheduler` registers: `entropy_bound`, `t_min`, `t_max`,
    `num_inference_steps`. Attribute access reads the frozen snapshot;
    `__setattr__` raises — the real class offers no in-place single-value
    mutation, only whole-dict replacement via `register_to_config`. A more
    permissive fake here would let a composition test (R1) pass against a
    scheduler-config shape the real class does not offer — the exact
    trust-and-degrade gap ADR-CDG-001 forbids (CLAUDE.md).
    """

    def __init__(self, **kwargs: Any) -> None:
        object.__setattr__(self, "_values", dict(kwargs))

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name in values:
            return values[name]
        raise AttributeError(f"FakeFrozenConfig has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"FakeFrozenConfig is frozen — cannot set {name!r} directly; "
            "use the owning scheduler's `register_to_config(**kwargs)`."
        )

    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, "_values")[key]

    def get(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, "_values").get(key, default)

    def __repr__(self) -> str:
        return f"FakeFrozenConfig({object.__getattribute__(self, '_values')!r})"


class FakeEntropyBoundScheduler:
    """Mirrors `diffusers.EntropyBoundScheduler` (`scheduling_entropy_bound.py`,
    installed diffusers 0.39.0) at exactly the surface `dgemma/loop.py` and
    its future R1/R5 composition/statelessness tests touch:

    - Ctor kwargs `entropy_bound`, `t_max`, `t_min`, `num_inference_steps`
      (`:82-85`) — `self.num_inference_steps` set as a **plain attribute**,
      matching the real ctor (`:84`) exactly (not config-mediated — the real
      `step()` divides by this attribute directly, `:153`).
    - `.config` — a `FakeFrozenConfig` snapshot of the ctor kwargs, matching
      `@register_to_config`'s real effect (`:80`). Read-only by direct
      assignment; mutate via `register_to_config(**kwargs)`, exactly like
      the real class (`configuration_utils.py:143-158`).
    - `set_timesteps(num_inference_steps, device=None)` (`:87-91`):
      reassigns `self.num_inference_steps` — the exact seam issue #20's
      corrector-divergence case exercises (`predictor_steps !=
      num_inference_steps`), and the seam a same-in/same-out statelessness
      test (R5/F5) must confirm resets cleanly across two calls on one
      scheduler instance rather than accumulating state.
    - `step(model_output, timestep, sample, *, entropy_bound=None,
      generator=None, return_dict=True)` (`:118-182`): configurable
      acceptance per call via `accepted` (a bool tensor or a callable
      `(step_idx) -> bool tensor`, so a test can script per-step acceptance
      patterns for composition-ordering scenarios) instead of running the
      real entropy-bound math — this fixture is a controllable stand-in,
      not a numerical reimplementation. Returns a `FakeSchedulerOutput`
      exposing the same fields as `EntropyBoundSchedulerOutput`
      (`:26-48`): `prev_sample`, `accepted_index`, `sampled_tokens`,
      `sampled_probs`, `pred_logits`.
    """

    config_name = "scheduler_config.json"

    def __init__(
        self,
        *,
        entropy_bound: float = 0.1,
        t_max: float = 0.8,
        t_min: float = 0.4,
        num_inference_steps: int = 32,
    ) -> None:
        self.num_inference_steps = num_inference_steps
        self._config = FakeFrozenConfig(
            entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
        )
        self.step_calls: list[dict[str, Any]] = []
        self._accepted_source: Any = torch.tensor(True)

    @property
    def config(self) -> FakeFrozenConfig:
        return self._config

    def register_to_config(self, **kwargs: Any) -> None:
        """Mirrors the real mutation path (`configuration_utils.py:143-158`):
        rebuild `.config` wholesale from the merged kwargs — never mutate a
        single value on the existing frozen object in place."""
        merged = dict(object.__getattribute__(self._config, "_values"))
        merged.update(kwargs)
        self._config = FakeFrozenConfig(**merged)

    def set_timesteps(self, num_inference_steps: int, device: Any = None) -> None:
        if num_inference_steps <= 0:
            raise ValueError(f"`num_inference_steps` must be > 0, got {num_inference_steps}.")
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        model_output: Any,
        timestep: int,
        sample: torch.Tensor,
        *,
        entropy_bound: float | None = None,
        generator: Any = None,
        return_dict: bool = True,
        **_ignored: Any,
    ) -> "FakeSchedulerOutput":
        self.step_calls.append({"timestep": timestep, "entropy_bound": entropy_bound})
        accepted = self._accepted_index_for(timestep, sample.shape)
        prev_sample = torch.where(accepted, sample, torch.zeros_like(sample))
        output = FakeSchedulerOutput(
            prev_sample=prev_sample,
            accepted_index=accepted,
            sampled_tokens=sample.clone(),
            sampled_probs=torch.ones_like(sample, dtype=torch.float),
            pred_logits=model_output,
        )
        return output

    def _accepted_index_for(self, step_idx: int, shape: torch.Size) -> torch.Tensor:
        accepted = self._accepted_source
        if callable(accepted):
            result = accepted(step_idx)
        else:
            result = accepted
        if torch.is_tensor(result):
            return result.to(dtype=torch.bool).expand(shape)
        return torch.full(shape, bool(result), dtype=torch.bool)


@dataclass
class FakeSchedulerOutput:
    """Mirrors `EntropyBoundSchedulerOutput` (`scheduling_entropy_bound.py:26-48`)
    field-for-field — the exact object `dgemma.loop._FrameCollector.on_step_end`
    reads `.accepted_index` off of."""

    prev_sample: torch.Tensor
    accepted_index: torch.Tensor
    sampled_tokens: torch.Tensor
    sampled_probs: torch.Tensor
    pred_logits: torch.Tensor


class HookRecordingModel(torch.nn.Module):
    """A real `torch.nn.Module` (not a bare stand-in) so
    `register_forward_hook`/`RemovableHandle` behavior is genuine PyTorch
    semantics, not reinvented — grounded against the installed `torch`
    (verified interactively: `register_forward_hook` returns a
    `torch.utils.hooks.RemovableHandle`; hooks live in the module's
    `_forward_hooks` `OrderedDict`; `.remove()` deletes the entry).

    Layered on top: `install_log`/`removal_log` record every
    `register_forward_hook` call and every `.remove()` on a handle this
    model produced, keyed by an incrementing id — the surface #35 R5's
    "no hook survives a `run_diffusion` call" lifecycle test (clean +
    raising paths) needs to assert against (`ARCHITECTURE.md`'s "Zero hooks
    after run" row). A trivial `forward` (returns a zeros logits tensor
    shaped like a real `DiffusionGemmaForBlockDiffusion` call) is enough:
    this fixture is a controllable stand-in for hook lifecycle and
    composition-ordering tests, not a numerical model reimplementation.
    """

    def __init__(self, vocab_size: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.install_log: list[int] = []
        self.removal_log: list[int] = []
        self._next_hook_id = 0

    def register_forward_hook(self, hook, *, prepend=False, with_kwargs=False, always_call=False):
        hook_id = self._next_hook_id
        self._next_hook_id += 1
        handle = super().register_forward_hook(
            hook, prepend=prepend, with_kwargs=with_kwargs, always_call=always_call
        )
        self.install_log.append(hook_id)
        real_remove = handle.remove

        def _tracked_remove() -> None:
            real_remove()
            self.removal_log.append(hook_id)

        handle.remove = _tracked_remove
        return handle

    @property
    def live_hook_count(self) -> int:
        """Hooks installed but not yet removed — the "zero hooks after run"
        invariant reads `live_hook_count == 0` after a `run_diffusion` call
        (R5)."""
        return len(self.install_log) - len(self.removal_log)

    def forward(self, decoder_input_ids: torch.Tensor, **_ignored: Any) -> "FakeModelOutput":
        batch, canvas_len = decoder_input_ids.shape
        logits = torch.zeros((batch, canvas_len, self.vocab_size))
        return FakeModelOutput(logits=logits)


@dataclass
class FakeModelOutput:
    """Mirrors the `.logits` access `pipeline_diffusion_gemma.py:365-371`
    reads off the real model's forward output."""

    logits: torch.Tensor


class FakeDiffusionGemmaPipeline:
    """Mirrors `diffusers.DiffusionGemmaPipeline.__call__`'s per-step drive
    loop (`pipeline_diffusion_gemma.py:356-407`) at exactly the seam
    `dgemma/loop.py`'s `DGemmaPipeline`/`run_diffusion` touches:

    - Runs `num_inference_steps` iterations of: call `scheduler.step(...)`,
      assign `canvas = scheduler_output.prev_sample`, then — when a
      callback is given — build `callback_kwargs` from
      `callback_on_step_end_tensor_inputs` (each key resolved from a small
      per-step "available locals" mapping, not hardcoded to two keys —
      `dgemma.loop.DGemmaPipeline`'s whole reason for widening
      `_callback_tensor_inputs` to include `"scheduler_output"` depends on
      this genericity), call `callback_on_step_end(self, global_step,
      step_idx, callback_kwargs)`, and apply
      `canvas = callback_outputs.pop("canvas", canvas)` verbatim — the
      faithful `{"canvas": ...}` application R4's spec names, including the
      case where a participant does NOT return a `"canvas"` key (the `.pop`
      default keeps the scheduler's own `prev_sample`, exercised by
      `TestFakePipelineCanvasApplication::test_no_canvas_key_keeps_scheduler_output`).
    - `_callback_tensor_inputs` allowlist + `check_inputs`-style validation
      (`pipeline_diffusion_gemma.py:76,134-161`): requesting a tensor input
      not in the allowlist raises `ValueError`, mirroring the real pipeline
      rather than silently ignoring an unknown key.
    - `self.model`/`self.scheduler`/`self.processor` registered as
      attributes (`register_modules`, `:85`) — so a callback closing over
      `pipe.model` (the ONE-DOOR reachability path #35's required ADR
      clauses explicitly name as NOT a sanctioned installation path) can be
      tested against a real attribute, not an attribute this fixture is
      missing.
    - Returns a `FakePipelineOutput` mirroring `DiffusionGemmaPipelineOutput`
      (`sequences`, `texts`).
    """

    _callback_tensor_inputs = ["canvas", "logits", "scheduler_output"]

    def __init__(self, model: Any, scheduler: Any, processor: Any) -> None:
        self.model = model
        self.scheduler = scheduler
        self.processor = processor
        tokenizer = getattr(processor, "tokenizer", processor)
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None

    def __call__(
        self,
        *,
        prompt: str | None = None,
        messages: list[dict] | None = None,
        gen_length: int = 256,
        num_inference_steps: int = 48,
        confidence_threshold: float | None = None,
        generator: Any = None,
        callback_on_step_end: Callable[[Any, int, int, dict], dict] | None = None,
        callback_on_step_end_tensor_inputs: list[str] | None = None,
        **_ignored: Any,
    ) -> "FakePipelineOutput":
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            unknown = [k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {unknown}"
            )
        requested_inputs = callback_on_step_end_tensor_inputs or ["canvas"]

        canvas_length = 4
        canvas = torch.zeros((1, canvas_length), dtype=torch.long)
        global_step = 0
        for step_idx in range(num_inference_steps):
            logits = self.model(decoder_input_ids=canvas).logits
            scheduler_output = self.scheduler.step(
                model_output=logits, timestep=step_idx, sample=canvas, generator=generator, return_dict=True
            )
            canvas = scheduler_output.prev_sample

            if callback_on_step_end is not None:
                available = {"canvas": canvas, "logits": logits, "scheduler_output": scheduler_output}
                callback_kwargs = {k: available[k] for k in requested_inputs}
                callback_outputs = callback_on_step_end(self, global_step, step_idx, callback_kwargs)
                canvas = callback_outputs.pop("canvas", canvas)
            global_step += 1

        sequences = canvas
        texts = None
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        if tokenizer is not None and hasattr(tokenizer, "decode"):
            texts = [tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in sequences]
        return FakePipelineOutput(sequences=sequences, texts=texts)


@dataclass
class FakePipelineOutput:
    """Mirrors `DiffusionGemmaPipelineOutput` (`pipeline_output.py`)."""

    sequences: torch.Tensor
    texts: list[str] | None = None


@dataclass
class FakePipelineFactory:
    """Handle returned by the `fake_pipeline_factory` fixture: holds the
    just-constructed `FakeEntropyBoundScheduler`/`HookRecordingModel`/
    `FakeDiffusionGemmaPipeline` triple plus the knobs used to build them,
    so a test can both drive the pipeline and inspect/mutate the scheduler
    or model afterward (e.g. asserting `model.live_hook_count == 0`, or
    calling `scheduler.register_to_config(entropy_bound=...)` mid-test for
    a composition-ordering scenario)."""

    scheduler: FakeEntropyBoundScheduler
    model: HookRecordingModel
    pipeline: FakeDiffusionGemmaPipeline
    accepted: Any


@pytest.fixture
def fake_pipeline_factory() -> Callable[..., FakePipelineFactory]:
    """Factory fixture (#35 R4): builds a fresh
    `(scheduler, model, pipeline)` triple per call, so a test can configure
    step count, acceptance pattern, and vocab size without monkeypatching
    module-level names (contrast the pre-R4 `_install_fakes` pattern in
    `tests/test_run_diffusion_knobs.py`, which patches
    `dgemma.loop.EntropyBoundScheduler`/`dgemma.loop.DGemmaPipeline`
    directly — still valid for engine-level `run_diffusion` threading
    tests, but not reusable for a bare composition/hook test that wants to
    drive a pipeline object directly without going through `run_diffusion`
    at all).

    `accepted`: what each `scheduler.step()` call reports as
    `accepted_index` — a bool tensor (any shape broadcastable to the
    canvas), or a callable `step_idx -> bool tensor`/`bool` for scripted
    per-step acceptance patterns (composition-ordering tests want e.g.
    "nothing accepted until step 3"). Defaults to all-True (full acceptance
    every step) — the common case; self-tests below override only where it
    matters.
    """

    def _build(
        *,
        num_inference_steps: int = 4,
        entropy_bound: float = 0.1,
        t_min: float = 0.4,
        t_max: float = 0.8,
        vocab_size: int = 8,
        canvas_shape: tuple[int, int] = (1, 4),
        accepted: Any = None,
        processor: Any = None,
    ) -> FakePipelineFactory:
        if accepted is None:
            accepted = torch.ones(canvas_shape, dtype=torch.bool)
        scheduler = FakeEntropyBoundScheduler(
            entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
        )
        scheduler._accepted_source = accepted
        model = HookRecordingModel(vocab_size=vocab_size)
        pipeline = FakeDiffusionGemmaPipeline(model=model, scheduler=scheduler, processor=processor)
        return FakePipelineFactory(scheduler=scheduler, model=model, pipeline=pipeline, accepted=accepted)

    return _build


@pytest.fixture
def fake_pipeline(fake_pipeline_factory: Callable[..., FakePipelineFactory]) -> FakePipelineFactory:
    """Convenience fixture: the default-configured `fake_pipeline_factory()`
    build, for tests that don't need to vary the knobs."""
    return fake_pipeline_factory()
