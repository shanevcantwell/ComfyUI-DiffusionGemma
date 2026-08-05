"""dgemma/loop.py — the denoising-loop spine (ADR-CDG-004 drive seam).

Drives a preloaded `DiffusionGemmaForBlockDiffusion` (from `dgemma/model.py`)
through `diffusers.DiffusionGemmaPipeline` + `EntropyBoundScheduler`, per
ADR-CDG-004. Per-step frames are the loop's native contract from day one
(plan.md, `dgemma/loop.py` per-module notes): P1 keeps only the last frame
(`keep_frames="last"`), but the collection seam iterates every step
regardless, so P2 (knobs) and P3 (instrumentation) grow the same generator
without a reshape.

**Diffusers version guard + structural probe (issue #35 R3, ARCHITECTURE.md
"No diffusers version guard" row).** Enforced by `dgemma/compat.py`'s
`assert_diffusers_compatible()`, called below before this module's own
`from diffusers import ...` line — this module is still diffusers' real
import site (`model.py`'s transformers guard has its own module; this is the
twin, here rather than there because `import diffusers` happens here, not in
`model.py` — verified: `dgemma/__init__.py` imports `.loop` before `.model`,
so in practice diffusers lands in `sys.modules` before transformers does on a
fresh `import dgemma`). See `compat.py`'s module docstring for the guard's
full rationale, including the anneal-formula-body residual neither check can
see (PR #48 gate finding F-1), enforced instead by
`tests/test_diffusers_version_guard.py:TestAnnealFormulaPin` against
`anneal_temperature` (`dgemma/capture.py`).
"""
from __future__ import annotations

import inspect
from typing import Any, Callable, Literal

import torch

# `dgemma/compat.py` (ADR-CDG-018 Stage 2, issue #129): the diffusers
# version-floor check + structural probe moved there verbatim, behind one
# public `assert_diffusers_compatible()` wrapper. Called here, BEFORE this
# module's own `from diffusers import ...` line below, to preserve the exact
# import-time-guard ordering `dgemma/__init__.py` depends on (imports `.loop`
# before `.model`, so diffusers lands in `sys.modules` before transformers
# does on a fresh `import dgemma`) — see `compat.py`'s module docstring for
# the full preservation argument. `_check_diffusers_version`/
# `_check_diffusers_structure`/`_tuple_version`/`REQUIRED_DIFFUSERS_MINIMUM`
# are re-imported below (not re-defined) so every existing
# `from dgemma.loop import _check_diffusers_version` import site (the facade
# ruling, issue #129) keeps resolving unchanged.
from .compat import (  # noqa: E402
    REQUIRED_DIFFUSERS_MINIMUM,
    _check_diffusers_structure,
    _check_diffusers_version,
    _tuple_version,
    assert_diffusers_compatible,
)

assert_diffusers_compatible()

from diffusers import DiffusionGemmaPipeline, EntropyBoundScheduler  # noqa: E402

from .capture import _FrameCollector, _build_pinned_mask, anneal_temperature  # noqa: E402
from .composite import DiffusionCancelled, StepEndComposite  # noqa: E402
from .config import (  # noqa: E402
    DEFAULT_CONFIDENCE,
    DEFAULT_ENTROPY_BOUND,
    DEFAULT_GEN_LENGTH,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_T_MAX,
    DEFAULT_T_MIN,
    KNOB_DOCS,
    THINK_TOKEN,
    THOUGHT_CHANNEL_END_ID,
    THOUGHT_CHANNEL_END_TOKEN,
    THOUGHT_CHANNEL_LABEL,
    THOUGHT_CHANNEL_START_ID,
    THOUGHT_CHANNEL_START_TOKEN,
)
from .constraints_hook import build_logit_mask_hook  # noqa: E402
from .excision import (  # noqa: E402
    ThoughtChannelExcision,
    _decode_ids,
    _extract_thought_text,
    decode_frames,
    derive_canvas_state,
    excise_thought_channel,
    resolve_thought_channel_ids,
    resolve_vocab_size,
)
from .hooks import ForwardHookFn, install_logit_shaping_hook  # noqa: E402
from .ingress import validate_ingress  # noqa: E402
from .kv_cache import prefill_templated_turn, validate_kv_cache_ingress  # noqa: E402
from .participants import PinParticipant, WalkerParticipant  # noqa: E402
from .payloads import Constraints, ControlSignals  # noqa: E402
from .types import CanvasState, CanvasTrace, DGemmaModel, DiffusionFrame, KVCache, Provenance  # noqa: E402

# `dgemma/config.py` (ADR-CDG-018 Stage 1, issue #129): DEFAULT_*/KNOB_DOCS/
# THINK_TOKEN/THOUGHT_CHANNEL_* moved there verbatim. Imported (not
# redefined) above so this module stays the compatibility facade — every
# existing `from dgemma.loop import DEFAULT_GEN_LENGTH` / `KNOB_DOCS` / etc.
# import site keeps resolving unchanged (`__all__`-free re-export by
# reference, not a duplicate literal).
#
# `dgemma/capture.py` (ADR-CDG-018 Stage 3, issue #129): `_FrameCollector`/
# `_build_pinned_mask`/`anneal_temperature` moved there verbatim (same
# re-import-not-redefine facade discipline as `.config`/`.compat` above).
#
# `dgemma/excision.py` (ADR-CDG-018 Stage 4, issue #129): thought-channel
# excision + the decode/derive family (`ThoughtChannelExcision`,
# `excise_thought_channel`, `_decode_ids`, `_extract_thought_text`,
# `decode_frames`, `resolve_vocab_size`, `resolve_thought_channel_ids`,
# `derive_canvas_state`) moved there verbatim — same facade discipline.


class DGemmaPipeline(DiffusionGemmaPipeline):
    """`DiffusionGemmaPipeline` subclass widening the per-step callback allowlist.

    The ONLY change from the base pipeline: `_callback_tensor_inputs` gains
    `"scheduler_output"`. The base class allowlist is `["canvas", "logits"]`
    (`pipeline_diffusion_gemma.py:76`); `check_inputs` validates
    `callback_on_step_end_tensor_inputs` against `self._callback_tensor_inputs`
    (`:155-161`), and the callback-kwargs extraction is generic —
    `callback_kwargs[k] = locals()[k]` (`:404-405`) — not a hardcoded
    two-key dispatch. Widening the allowlist here is therefore enough to hand
    the callback the full scheduler `.step()` output object (`accepted_index`,
    `sampled_probs`, `pred_logits`, ...) with no method override needed
    (ADR-CDG-004, resolved open question (a)).

    Caveat carried from that resolution: `"accepted_index"` alone is NOT a
    valid key — it is not a bound local in `__call__`'s scope. Only the
    `scheduler_output` container is.
    """

    _callback_tensor_inputs = ["canvas", "logits", "scheduler_output"]


def _run_pipeline_with_injected_cache(
    pipeline: "DGemmaPipeline",
    *,
    kv_cache: KVCache,
    gen_length: int,
    num_inference_steps: int,
    confidence_threshold: float,
    generator: "torch.Generator | None",
    callback_on_step_end: StepEndComposite,
    prompt_kwargs: "dict | None" = None,
) -> tuple[Any, KVCache]:
    """ADR-CDG-012 Phase 4 decoder-drive body (issue #62) — productionizes the
    Q-2 smoke skeleton (`scratch/q2-skeleton-2026-08-04` @ `d67e62f`,
    `docs/experiments/2026-08-04-adr-cdg-012-q2-smoke/skeleton-loop.py.diff`)
    into the full multi-block loop.

    Mirrors `diffusers.DiffusionGemmaPipeline.__call__`'s own per-block loop
    (`pipeline_diffusion_gemma.py:301-436`) line-for-line for every knob this
    pack exposes (`confidence_threshold`/`stability_threshold`/
    `eos_early_stop`/`generator`), since the pipeline offers no injected-cache
    parameter to call through (the skeleton's own docstring) — this function
    IS that call path, not a divergent reimplementation. Deltas from the
    pipeline body, each named:

    - **IN-2 skip-first-encode.** Block 0 does NOT call
      `pipeline.model.model.encoder(...)` — `kv_cache.cache` (already a live
      `DynamicCache`, minted by a prior `DGemmaEncode`/`encode_sequence`
      call) is used directly as `past_key_values`, and `cached_len` is read
      from `kv_cache.cumulative_length` rather than
      `past_key_values.get_seq_length()` on an empty cache. Every block AFTER
      the first re-encodes normally (the committed canvas from the previous
      block), identical to the pipeline's own IN-3-shaped per-block encode —
      this function does not special-case any block beyond the first.
    - **Composed prefill (ADR-CDG-024, issue #257).** `prompt_kwargs` empty/
      `None` degrades to the original skeleton's behavior byte-for-byte: the
      injected cache stands in for what would otherwise be the first block's
      encode, `DGemmaDenoise`'s `prompt` widget is not tokenized, and block
      0's `decoder_start` derives from `kv_cache`'s own pre-call
      `cumulative_length` (unchanged). A non-empty `prompt_kwargs` (the SAME
      dict `run_diffusion`'s no-cache path builds at `:736-744` — one
      template-construction site, two consumers) is chat-templated and
      prefilled onto `kv_cache.cache` via `dgemma.kv_cache.
      prefill_templated_turn` BEFORE block 0's decode setup, in place of the
      skipped re-encode; block 0's `decoder_start` is then rebound to the
      cache's own POST-prefill `past_key_values.get_seq_length()` (never a
      hand-computed length — ADR-CDG-024 §4's named failure-mode
      prevention), mirroring how the block>0 re-encode below already derives
      its own splice from the cache's advanced state. **OPEN (ADR-CDG-024,
      not resolved by this ADR):** block>0's `block_start`/`decoder_start`
      arithmetic still derives from the ORIGINAL pre-prefill `cached_len`
      captured before this function's loop starts — the ADR's §1 layout
      commits to block 0 only and does not walk through whether a
      templated-turn prefill needs a compensating shift for block>0 offsets;
      this is carried forward as a verify-during-implementation item, not
      silently resolved here (see the implementing PR's deviations
      section).
    - **OUT-1 (advanced-cache output) stays deferred**
      (`surfaces/comfyui/denoise.py`'s named delta 2, ADR-CDG-012 §D.2): this
      function runs every block to completion/EOS exactly like the pipeline
      does — there is no `stop_at_block` parameter and no early-return
      mid-loop, because `DGemmaDenoise` ships no widget to request one.
    - **Cancellation/participant wiring reused verbatim.** `callback_on_step_end`
      is the SAME `StepEndComposite` `run_diffusion` builds for the no-cache
      path (capture, cancellation, pin, walker) — this function does not
      construct its own composite or collector, so `DiffusionCancelled`
      raised mid-block propagates to `run_diffusion`'s existing
      `except DiffusionCancelled` handler unchanged.
    - **`stability_threshold`/`eos_early_stop` hardcoded, at parity with the
      no-cache path's own hardcoding.** Neither is a `run_diffusion`
      parameter today (see that function's own docstring: "`stability_
      threshold`/`eos_early_stop` stay at the pipeline's own defaults" —
      `1`/`True`) — this function hardcodes the same two values
      (`argmax_history`'s leading dim `1`, the unconditional
      `eos_token_id is not None` check) rather than re-deriving them from a
      caller-supplied knob that doesn't exist yet. If a future PR promotes
      either to a `run_diffusion` parameter, this function's hardcoded
      values must be threaded through in the same change, or the two paths
      silently diverge — named here so that PR doesn't miss this call site.

    Returns `(sequences, advanced_cache)` where `sequences` is
    `cur_input_ids[:, 0:]` shaped like `output.sequences[0]` in the no-cache
    path (batch size is always 1 on this path — the skeleton/ADR's tier-1
    scope never batches an injected cache) and `advanced_cache` is the final
    block's `past_key_values` (not emitted anywhere yet — OUT-1 is deferred,
    per above; returned so a future OUT-1 wiring has it without a second
    pass).
    """
    # Batch size 1 only (tier-1 scope, never batches an injected cache —
    # this function's own docstring/return-value note). Asserted here
    # rather than silently truncating to `[0]` at the `sequences =
    # torch.cat(...)` return below, per ADR-CDG-001's fail-loud discipline:
    # a batch>1 cache would otherwise produce a plausible-but-wrong single
    # sequence instead of a caught precondition violation.
    if kv_cache.cache.layers and kv_cache.cache.layers[0].keys.shape[0] != 1:
        raise ValueError(
            f"_run_pipeline_with_injected_cache: kv_cache has batch size "
            f"{kv_cache.cache.layers[0].keys.shape[0]}, only batch size 1 is "
            "supported (ADR-CDG-012 tier-1 scope never batches an injected "
            "cache)."
        )

    model = pipeline.model
    scheduler = pipeline.scheduler
    device = model.device
    # No-arg `get_text_config()` (not the skeleton's `decoder=True`) —
    # matches every other call site in this codebase (`dgemma/kv_cache.py:75,124`)
    # and `FakeDGemmaModelConfig`'s fake shape; behaviorally identical here
    # since `DiffusionGemmaConfig.get_text_config()` always resolves the
    # single `text_config` sub-config regardless of the `decoder=`/`encoder=`
    # hint (`dgemma/kv_cache.py:66-73`'s docstring).
    text_config = model.config.get_text_config()
    canvas_length = model.config.canvas_length
    num_canvases = (gen_length + canvas_length - 1) // canvas_length
    eos_token_id = pipeline.eos_token_id

    past_key_values = kv_cache.cache
    cached_len = kv_cache.cumulative_length[0] if kv_cache.cumulative_length else 0

    scheduler.set_timesteps(num_inference_steps, device=device)
    step_param_names = set(inspect.signature(scheduler.step).parameters)

    # `cur_input_ids` tracks only the committed CANVAS tokens (the injected
    # cache already holds the donor context — there is no `prompt_ids` prefix
    # to concatenate onto, unlike the pipeline's own `cur_input_ids =
    # prompt_ids` seed at `pipeline_diffusion_gemma.py:301`). Each committed
    # block is appended here for `sequences`, decode, and (for block > 0)
    # re-encode.
    committed_blocks: list[torch.Tensor] = []
    finished = torch.zeros(1, dtype=torch.bool, device=device)
    global_step = 0

    for canvas_idx in range(num_canvases):
        if canvas_idx == 0:
            if prompt_kwargs:
                # ADR-CDG-024 §1 (issue #257): a non-empty `prompt` alongside
                # `kv_cache` is the current-turn text — chat-template it
                # (SAME `prompt_kwargs` dict the no-cache path builds) and
                # prefill it onto the injected cache, in place of the
                # skipped IN-2 re-encode. `decoder_start`'s base is rebound
                # below to the cache's own POST-prefill `get_seq_length()`,
                # not the pre-prefill `cached_len` — ADR §4's named
                # position-id-drift prevention.
                past_key_values = prefill_templated_turn(pipeline, past_key_values, prompt_kwargs)
                decoder_start_base = past_key_values.get_seq_length()
            else:
                # IN-2: skip the first encode entirely — `past_key_values` is
                # the injected cache as-is. `prompt_kwargs` empty/`None`
                # degrades to this exact byte-for-byte behavior
                # (ADR-CDG-024 §1's additive-optional framing).
                decoder_start_base = cached_len
        else:
            # Re-encode the previously committed block into the (already
            # cache-seeded) `past_key_values` — identical in shape to the
            # pipeline's own per-block encode
            # (`pipeline_diffusion_gemma.py:322-333`), just operating on a
            # cache whose starting length is the injected cache's, not 0.
            prev_block = committed_blocks[-1]
            block_start = cached_len + (canvas_idx - 1) * canvas_length
            torch.compiler.cudagraph_mark_step_begin()
            model.model.encoder(
                input_ids=prev_block,
                past_key_values=past_key_values,
                position_ids=torch.arange(
                    block_start, block_start + canvas_length, device=device
                ).unsqueeze(0),
            )

        # Block 0's base is `decoder_start_base` (cache's post-prefill
        # `get_seq_length()` when a prefill ran, else the pre-loop
        # `cached_len` unchanged). Block>0 still derives from the ORIGINAL
        # `cached_len` (OPEN, ADR-CDG-024 — the ADR commits to block 0's
        # splice only; a compensating shift for block>0 once a prefill has
        # run is unresolved, not silently assumed here).
        decoder_start = (
            decoder_start_base if canvas_idx == 0 else cached_len + canvas_idx * canvas_length
        )
        decoder_position_ids = torch.arange(
            decoder_start, decoder_start + canvas_length, device=device
        ).unsqueeze(0)

        seq_len_now = past_key_values.get_seq_length()
        decoder_attention_mask = torch.nn.functional.pad(
            torch.ones((1, seq_len_now), dtype=torch.bool, device=device),
            (0, canvas_length),
            value=True,
        )
        mask_mapping = model.model.decoder.create_diffusion_decoder_attention_mask(
            config=model.config,
            inputs_embeds=torch.empty((1, canvas_length, 0), device=device),
            past_key_values=past_key_values,
            decoder_attention_mask=decoder_attention_mask,
        )

        canvas = torch.randint(
            0, text_config.vocab_size, (1, canvas_length), device=device, generator=generator
        )
        self_conditioning_logits = None
        argmax_history = torch.full((1, 1, canvas_length), -1, dtype=torch.long, device=device)

        for step_idx in range(num_inference_steps):
            torch.compiler.cudagraph_mark_step_begin()
            logits = model(
                decoder_input_ids=canvas,
                past_key_values=past_key_values,
                self_conditioning_logits=self_conditioning_logits,
                decoder_attention_mask=mask_mapping,
                decoder_position_ids=decoder_position_ids,
            ).logits.clone()

            step_kwargs = {"mask_token_id": None, "temperature": 0.0, "generator": generator}
            step_kwargs = {k: v for k, v in step_kwargs.items() if k in step_param_names}
            scheduler_output = scheduler.step(
                model_output=logits, timestep=step_idx, sample=canvas, return_dict=True, **step_kwargs
            )
            canvas = scheduler_output.prev_sample
            self_conditioning_logits = scheduler_output.pred_logits

            callback_kwargs = {"canvas": canvas, "logits": logits, "scheduler_output": scheduler_output}
            callback_outputs = callback_on_step_end(pipeline, global_step, step_idx, callback_kwargs)
            canvas = callback_outputs.pop("canvas", canvas)
            global_step += 1

            if confidence_threshold is not None:
                argmax_canvas = logits.argmax(dim=-1)
                stable = (argmax_history == argmax_canvas[None]).all(dim=-1).all(dim=0)
                argmax_history = torch.roll(argmax_history, shifts=-1, dims=0)
                argmax_history[-1] = argmax_canvas
                confident = torch.distributions.Categorical(logits=logits.float()).entropy().mean(-1) < (
                    confidence_threshold
                )
                if bool((stable & confident).all()):
                    canvas = argmax_canvas
                    break

        committed_blocks.append(canvas)

        if eos_token_id is not None:
            finished = finished | (canvas == eos_token_id).any(dim=-1)
            if finished.all():
                break

    sequences = torch.cat(committed_blocks, dim=-1)[0]
    return sequences, past_key_values


def run_diffusion(
    dgemma_model: DGemmaModel,
    prompt: str,
    *,
    seed: int | None = None,
    gen_length: int = DEFAULT_GEN_LENGTH,
    num_inference_steps: int = DEFAULT_NUM_INFERENCE_STEPS,
    entropy_bound: float = DEFAULT_ENTROPY_BOUND,
    t_min: float = DEFAULT_T_MIN,
    t_max: float = DEFAULT_T_MAX,
    confidence: float = DEFAULT_CONFIDENCE,
    thinking: bool = False,
    keep_frames: Literal["last", "all"] = "all",
    on_frame: Callable[[DiffusionFrame], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    logit_hook: ForwardHookFn | None = None,
    constraints: "Constraints | None" = None,
    control_signals: "ControlSignals | None" = None,
    capture: Any = None,
    kv_cache: "KVCache | None" = None,
) -> tuple[str, CanvasState, CanvasTrace]:
    """Drive one prompt through the block-diffusion denoising loop.

    Constructs `EntropyBoundScheduler` directly with the entropy/temperature
    config (`entropy_bound`, `t_min`, `t_max`, `num_inference_steps`) — these
    live on the scheduler config, NOT on the pipeline's `__call__` (ADR-CDG-004:
    the pipeline only forwards `generator`/`mask_token_id`/`temperature` to
    `scheduler.step()`, filtered by that scheduler's own signature, and
    `EntropyBoundScheduler.step()` doesn't accept `mask_token_id` or
    `temperature` at all — it anneals its own). Wraps the loaded model in
    `DGemmaPipeline` (direct-constructor idiom, not `.from_pretrained`, since
    the model is already loaded).

    `confidence` promotes the pipeline's `confidence_threshold` to a real
    parameter (P2). `stability_threshold`/`eos_early_stop` stay at the
    pipeline's own defaults (1 / True — already the grounded defaults,
    CLAUDE.md); P2 only promoted the knobs plan.md names for Phase 2.

    `thinking` (P2, model-card documented mechanism): when `True`, the
    `<|think|>` control token is injected at the start of the (otherwise
    empty) system turn by passing an explicit `messages=[{"role": "system",
    "content": THINK_TOKEN}, {"role": "user", "content": prompt}]`. This is
    the ONLY viable path here: the pipeline's `_prepare_inputs` never
    forwards `enable_thinking` (or any extra kwargs) to
    `apply_chat_template`, so the template's native toggle is unreachable
    through `pipeline.__call__`. **Honest delta, pinned by
    `tests/test_chat_template_thinking.py` against the real tokenizer
    (2026-07-05):** the injected path is NOT token-identical to the native
    `enable_thinking=True` render — the template emits system content
    through `| trim`, which eats the newline the native path places after
    `<|think|>`, so the injected render is exactly one token short (id 107,
    `"\\n"`, between `<|think|>` and `<turn|>`). Token parity is
    structurally unreachable via message content (any trailing whitespace is
    trimmed). Behavioral impact of the missing newline is unverified pending
    an E2E thinking-mode run; the `<|think|>` token itself (id 98) lands in
    the documented position either way. When `False` (default), `prompt` is
    passed bare — unchanged from P1, no system turn is added.

    Regardless of `thinking`, the thought channel the model emits at turn
    start (issue #8 — empty when off, per the model card's "an empty
    thinking channel might still be emitted"; possibly non-empty when on) is
    excised from the canvas ids via `excise_thought_channel` before `text`
    is derived, so it never leaks onto the `STRING` payload in either mode.

    `keep_frames` defaults to `"all"` (P3): per-step state here is small
    (ADR-CDG-005's own domain framing — a `gen_length`-length int64 canvas
    plus a per-example float per step), so retaining every step for the
    returned `CanvasTrace` isn't worth gating behind a toggle. `on_frame`,
    when given, is invoked once per captured step regardless of
    `keep_frames` — the seam that lets `nodes/sampler.py` push a live view
    without this module ever importing ComfyUI (ADR-CDG-003): the callback
    body that touches `PromptServer` lives in the node layer, not here.
    `on_frame` exceptions propagate (engine contract — see
    `_FrameCollector`'s docstring): a callback that must never kill the run
    guards itself, as the node layer's display-only closure does.

    `should_cancel` (issue #38, folded into R1's composer spec per the #35
    handoff): a zero-argument, surface-neutral predicate checked once per
    step by `dgemma.composite.StepEndComposite`, AFTER that step's capture
    (ADR-CDG-010 cancellation amendment 2026-07-13, PR #45) — surface-
    agnostic by construction (ARCHITECTURE.md rule 1): a ComfyUI surface
    wires this to `comfy.model_management`'s interrupt check, an MCP surface
    wires it to its own abort signal, and this module never imports either.
    When the predicate reports `True`, the composite raises
    `DiffusionCancelled`, caught here to return the PARTIAL
    `(text, CanvasState, CanvasTrace)` built from every frame captured so
    far — INCLUDING the cancelled step's own committed frame, the run's
    exact truncation point (the scheduler has already committed that step
    by `callback_on_step_end` time; see `dgemma/composite.py`'s module
    docstring) — evidence is returned, not raised away (#38's "a cancelled
    experiment run is still data" clause). `None` (default) means no
    cancellation wiring; the run always completes or raises a real error,
    exactly today's behavior.

    The single `callback_on_step_end` slot passed to the pipeline is a
    `dgemma.composite.StepEndComposite` (ADR-CDG-010 Decision 3 + its
    cancellation amendment), not the collector directly — the composite's
    fixed order is `capture -> cancellation check -> beta-rebuild -> pin ->
    walker`. `capture` and the cancellation seam are always wired; `pin` is
    wired (issue #64 Phase 3) with a fresh `PinParticipant` whenever
    `constraints=` carries at least one pin, `()` otherwise; `walker` is
    wired (issue #64 Phase 4) with a fresh `WalkerParticipant` whenever
    `control_signals=` carries at least one binding, `None` otherwise — so a
    run with no constraints/control_signals still builds an empty `pin=`
    tuple and a `None` `walker=`, and the composite's behavior is identical
    to invoking the collector alone, exactly as before either phase. The
    beta-rebuild participant (ADR-CDG-010) remains `NOT-YET-IMPLEMENTED` —
    Phase 5 lands that body; this phase only fills the `walker` slot the
    scaffold already exposed.

    `logit_hook` (#35 R5, F4; ADR-CDG-010 Decision 5): an optional forward
    hook installed on `dgemma_model.model` for exactly the duration of the
    one pipeline call below, via `dgemma.hooks.install_logit_shaping_hook` —
    the ONLY sanctioned installation path for a hook on this door (the only
    logit-shaping door per issue #28: a callback-returned `{"logits": ...}`
    is silently discarded by the installed pipeline). `None` when
    `constraints=` is also `None` installs nothing and leaves zero hooks
    registered, trivially satisfying `STATELESS-CORE`'s "no hook survives a
    `run_diffusion` call" (rule 6): the context manager's `try/finally`
    guarantees teardown on the pipeline call's clean return, on
    `DiffusionCancelled` (caught below), and on any other exception raised
    mid-run — the hook is torn down before this function's own exception
    handling (or return) is reached in every case. Passing BOTH
    `constraints=` and `logit_hook=` is rejected at ingress (H1, below) —
    two logit-mask sources on one door (ADR-CDG-010 D5).

    `constraints=`/`control_signals=`/`capture=` (ADR-CDG-010/011/014, issue
    #64/#61): declarative payloads, validated at ingress (`dgemma.ingress.
    validate_ingress`). `capture=`'s Tier 1 knob (`top_k`, ADR-CDG-014
    Decision 3, issue #61 P-B) is LIVE: when `capture.top_k > 0`, the
    `_FrameCollector` derives `DiffusionFrame.top_k_ids`/`top_k_weights` from
    the same pre-pin `logits` Tier 0's `entropy` reads — see `_FrameCollector.
    on_step_end`'s docstring. `capture=None`/`capture.top_k` absent/`0`
    (default) leaves both fields `None`, byte-identical to every run before
    that phase. `capture=`'s Tier 2 knobs (`capture_full_distribution`/
    `max_full_distribution_steps`, ADR-CDG-014 Decision 3/5, issue #61 P-C)
    are also LIVE: when `capture.capture_full_distribution=True`, the
    `_FrameCollector` derives `DiffusionFrame.distribution` (the full
    per-position `softmax(logits)`) from the same pre-pin `logits`, retained
    only for the first `capture.max_full_distribution_steps` captured steps
    — ingress rejects `capture_full_distribution=True` with no budget, so
    this call site never sees an unbounded request. `capture=None`/
    `capture.capture_full_distribution` absent/`False` (default) leaves
    `distribution` `None` on every frame, byte-identical to every run before
    P-C. `capture.keep_frames` remains validated-then-ignored (issue
    #64 P1, unchanged — see `dgemma/payloads.py:CaptureSpec`).
    `constraints=` is LIVE end-to-end (issue #64 Phase
    3, ADR-CDG-010's two-mechanism givens): when it carries at least one pin,
    `run_diffusion` (a) builds `dgemma.constraints_hook.build_logit_mask_hook`
    from the pins and installs it via the existing `logit_hook=`/
    `install_logit_shaping_hook` path — masking each pinned position's
    logits to its `token_id` so that cell reads ~zero entropy and commits
    first (Decision 1(a)); and (b) constructs a
    `dgemma.participants.PinParticipant` and wires it into the composite's
    `pin=` slot (Decision 3's LAST writer), re-asserting every pin's
    `token_id` at its `position` on every step regardless of what the
    scheduler accepted (Decision 1(b)) — the mechanism that guarantees *what
    conditions* the next forward pass, since a real scheduler step renoises
    every rejected position over the full vocabulary (no absorbing mask,
    ADR-CDG-001) and a given re-checked only at ingress would drift the
    first time its cell isn't accepted. `Constraints(pins=())`/`None`
    installs neither the hook nor the participant (empty == no-op,
    `dgemma/payloads.py`) — byte-identical to today's no-`constraints=`
    behavior. `control_signals=` is now LIVE (issue #64 Phase 4, ADR-CDG-011):
    when it carries at least one binding, `run_diffusion` constructs a
    `dgemma.participants.WalkerParticipant` bound to THIS call's `scheduler`
    and wires it into the composite's `walker=` slot (LAST, after every
    canvas-writer) — at the callback for `step_idx = k` the walker maps
    `signal[k + 1]` into the binding's declared `[low, high]` range and
    writes it via `scheduler.register_to_config(...)`, preparing step
    `k + 1`'s config (clause 6); `signal[0]` is never applied (the gate
    ruling on issue #64, O1) and the final step is a no-op (no step `k + 1`
    left to prepare) — see `dgemma.participants.WalkerParticipant`'s
    docstring for the full mechanism. `ControlSignals(bindings=())`/`None`
    builds no walker (empty == no-op) — byte-identical to today's
    no-`control_signals=` behavior. An invalid payload of any of the three
    still raises at ingress regardless of phase; `constraints=` +
    `logit_hook=` together still raise at ingress (H1) even now that
    `constraints=` builds its own hook internally — the two-source-on-one-door
    reject is unconditional.

    Returns `(text, CanvasState, CanvasTrace)` — never a bare string
    (ADR-CDG-001 Addendum). `CanvasTrace` carries `collector.frames` plus
    the scheduler's class name and the entropy/temperature config passed to
    it, per ADR-CDG-001's addendum on scheduler-relative commit semantics
    (a trace without the scheduler identity that minted its commit readings
    is a lying payload). It also carries `raw_canvas_ids` (ADR-CDG-014
    Decision 6, issue #11): the pre-excision final canvas ids, captured in
    `_build_result` before `excise_thought_channel` runs — the raw view
    `CanvasState.canvas_ids` (post-excision) does not carry. Each captured
    `DiffusionFrame` also carries `entropy` (ADR-CDG-014 Decision 3/4, issue
    #14): per-position predictive entropy derived from that step's pre-pin
    `logits`, always populated (Tier 0's always-on default);
    `top_k_ids`/`top_k_weights` (ADR-CDG-014 Decision 3, issue #61 P-B):
    per-position top-k candidate ids and their top-k-renormalized weights
    from the same pre-pin `logits`, populated only when `capture.top_k > 0`
    (`None`/`None` otherwise — Tier 1's on-request default);
    `distribution` (ADR-CDG-014 Decision 3/5, issue #61 P-C): the full
    per-position distribution (`softmax(logits)`) from the same pre-pin
    `logits`, populated only when `capture.capture_full_distribution=True`
    AND the step is still within `capture.max_full_distribution_steps`'s
    retention budget — `None` otherwise (Tier 2's explicit-opt-in-with-budget
    default; `None` also once the budget is exhausted mid-run, Decision 5);
    `pinned_mask`
    (ADR-CDG-010 D4, issue #64 Phase 2/3): `True` at every supplied
    `Constraints` pin position — now the positions `PinParticipant` actually
    (re-)writes every step (Phase 3), consistent with the Phase 2
    static-from-`Constraints.pins` derivation because a hard pin's position
    set never changes step to step (see `DiffusionFrame.pinned_mask`'s
    docstring for the scope guard) — `None` when no constraints were given;
    and
    `effective_entropy_bound`/`effective_t_min`/`effective_t_max`
    (ADR-CDG-011 clause 7, issue #64 Phase 2): the `entropy_bound`/`t_min`/
    `t_max` values `scheduler.config` actually held at that callback — the
    honest-telemetry fields the control-signal walker (issue #64 Phase 4)
    writes through via `register_to_config`, visible in the NEXT captured
    frame after the walker's write (clause 6: walker prepares the next step,
    capture records the finished step).

    `kv_cache=` (ADR-CDG-012 IN-2, issue #62 Phase 2 — types + ingress door;
    Phase 4 decoder-drive body NOT YET BUILT, issue #207 makes that gap fail
    loud rather than silently no-op): an optional injected `KVCache` payload
    (§62's `dgemma/types.py` dataclass).

    **Composition (ADR-CDG-024, issue #257 — supersedes issue #248's
    interim exclusivity):** `kv_cache` and a non-empty `prompt` are jointly
    permitted. When both are supplied, `prompt` is treated as the current
    model-turn: chat-templated exactly as the no-cache path templates it
    (same `prompt_kwargs` construction, below) and prefilled onto
    `kv_cache.cache` before the decode loop begins, via
    `dgemma.kv_cache.prefill_templated_turn`. Empty/absent `prompt` alongside
    a cache stays the pure injection-only shape, byte-for-byte unchanged.
    `dgemma.ingress.reject_prompt_and_kv_cache`, which rejected this pair
    under issue #248's interim posture, is tombstoned (no-op) — see that
    function's docstring for the supersession.

    `None` (default) is today's EXACT
    behavior, byte-for-byte unchanged — the run mints its own cache
    internally via the pipeline's own first encode, and rule-6
    `STATELESS-CORE` is trivially satisfied (no injected state crosses).
    When non-`None`, `dgemma.kv_cache.validate_kv_cache_ingress(kv_cache,
    dgemma_model)` fires BEFORE the scheduler/pipeline are constructed
    (fail-on-mismatch, rule 5 `EMIT-CANONICAL / PARSE-AT-THE-DOOR` — a bad
    cache is rejected before any resource tied to this call is built), and
    on pass this function **still raises** — `NotImplementedError` naming
    issue #62 Phase 4 — because a well-formed injected cache is not the same
    thing as a path that can honor it: the decoder-drive body that would
    actually consume the cache's tensors does not exist yet (Open Question
    #1, gated on the ADR's real-weights de-risk smoke test), and letting a
    validated-but-ignored cache silently fall through to an uninjected run
    would itself be the `EMIT-CANONICAL / PARSE-AT-THE-DOOR` violation this
    ADR's own ingress discipline forbids (issue #207 operator ruling,
    2026-08-01). The input `kv_cache` payload is never mutated by this
    function regardless (§3 advance-returns-new-payload discipline — this
    phase only reads it before raising).

    Raises `ValueError` if `t_min >= t_max` (parse-at-the-door validation —
    an inverted or degenerate anneal range would silently hand
    `EntropyBoundScheduler` a nonsensical temperature trajectory), if
    ingress validation of `constraints`/`control_signals`/`capture`/the
    `constraints`+`logit_hook` combination fails (see
    `dgemma.ingress.validate_ingress`'s error register), or if `kv_cache` is
    given and fails `validate_kv_cache_ingress`'s V1-V6 checks (see
    `dgemma.kv_cache.validate_kv_cache_ingress`'s error register). A
    non-empty `prompt` given together with a non-`None` `kv_cache` no longer
    raises (ADR-CDG-024, issue #257 — supersedes issue #248's exclusivity);
    the pair composes instead, per above.

    **Phase 4 (issue #62, ADR-CDG-012 §D.1 IN-2) — the decoder-drive body is
    LIVE.** A well-formed `kv_cache` that passes V1-V6 drives the decoder via
    `_run_pipeline_with_injected_cache` (skip-first-encode, full multi-block
    loop to completion/EOS — OUT-1 stop-at-block stays deferred per
    `surfaces/comfyui/denoise.py`) instead of the retired fail-loud door
    (issue #207's `NotImplementedError`, removed by the implementing PR).
    """
    if t_min >= t_max:
        raise ValueError(f"t_min must be < t_max, got t_min={t_min!r} t_max={t_max!r}.")

    # vocab_size resolution (issue #64 §3.4): same tokenizer path
    # `resolve_thought_channel_ids` uses. `None` when unavailable (e.g. a
    # bare test stub) — validate_constraints degrades by skipping C3 rather
    # than this call site inventing a size.
    vocab_size = resolve_vocab_size(dgemma_model.processor)
    validate_ingress(
        constraints,
        control_signals,
        capture,
        logit_hook,
        gen_length=gen_length,
        num_inference_steps=num_inference_steps,
        vocab_size=vocab_size,
    )

    # ADR-CDG-012 IN-2 (issue #62 Phase 2): fire the KV_CACHE door's own
    # ingress validator BEFORE any scheduler/pipeline construction below —
    # a bad injected cache is rejected before this call ties up a scheduler
    # or pipeline object (rule 5, EMIT-CANONICAL / PARSE-AT-THE-DOOR). `None`
    # (the default) skips this entirely — zero behavior change from before
    # this parameter existed.
    #
    # Issue #62 Phase 4 (this function): V1-V6 above confirm the PAYLOAD is
    # well-formed; the drive body below (`_run_pipeline_with_injected_cache`)
    # is what actually honors it, replacing issue #207's fail-loud
    # `NotImplementedError` door now that the ADR's real-weights de-risk
    # smoke test has PASSed (ledger #240, run 2026-08-04b) per the ADR's own
    # resolution trigger.
    #
    # ADR-CDG-024 (issue #257): the issue #248 exclusivity door that used to
    # sit here (`reject_prompt_and_kv_cache(prompt, kv_cache)`) is removed —
    # `prompt` + `kv_cache` is now the composed/prefill path (see this
    # function's docstring and `_run_pipeline_with_injected_cache`'s
    # `prompt_kwargs` branch below). `dgemma.ingress.reject_prompt_and_kv_cache`
    # is tombstoned, not deleted, so a reader grepping issue #248 lands on
    # live code explaining the reversal.

    if kv_cache is not None:
        validate_kv_cache_ingress(kv_cache, dgemma_model)

    # Constraints -> the two-mechanism givens (ADR-CDG-010 Decision 1, issue
    # #64 Phase 3). Both mechanisms are built from the SAME validated
    # `constraints.pins` and both are no-ops when `constraints` is `None` or
    # carries no pins (`Constraints()`/`Constraints(pins=())`) — "empty ==
    # no-op" (`dgemma/payloads.py`), so a run with an empty/`None`
    # `constraints=` builds neither the hook nor the pin participant and is
    # byte-identical to today's no-`constraints=` behavior.
    #
    # H1 (validated above) already forecloses `constraints=` AND
    # `logit_hook=` both being given, so building the hook here and passing
    # it through the same `logit_hook` name below can never collide with a
    # caller-supplied one.
    pin_participants: tuple = ()
    if constraints is not None and constraints.pins:
        logit_hook = build_logit_mask_hook(constraints.pins, vocab_size=vocab_size)
        pin_participants = (PinParticipant(constraints=constraints),)

    scheduler = EntropyBoundScheduler(
        entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
    )
    pipeline = DGemmaPipeline(model=dgemma_model.model, scheduler=scheduler, processor=dgemma_model.processor)

    # Control signals -> the walker (ADR-CDG-011, issue #64 Phase 4). Built
    # from THIS call's validated `control_signals` and THIS call's freshly
    # constructed `scheduler` — no cross-call state, no shared scheduler
    # reference (rule 6 STATELESS-CORE; ADR-CDG-011 clause 8/F5). Empty/`None`
    # `control_signals=` builds no walker at all — "empty == no-op"
    # (`dgemma/payloads.py`), byte-identical to today's no-`control_signals=`
    # behavior.
    walker_participant: WalkerParticipant | None = None
    if control_signals is not None and control_signals.bindings:
        walker_participant = WalkerParticipant(control_signals=control_signals, scheduler=scheduler)

    generator = None
    if seed is not None:
        generator = torch.Generator(device=dgemma_model.device).manual_seed(seed)

    # `scheduler` (not `num_inference_steps`) — the collector reads
    # `scheduler.num_inference_steps` lazily per-callback, so it always sees
    # the effective post-`set_timesteps` value the pipeline mutates this same
    # object with at call entry, not the user-requested count snapshotted
    # here before that call runs (issue #20; see `_FrameCollector`'s
    # docstring for the full grounding).
    # `capture.top_k` (ADR-CDG-014 Decision 3 Tier 1, issue #61 P-B): the
    # validated `CaptureSpec.top_k` value, duck-typed the same way
    # `validate_capture` reads `keep_frames` (ADR-CDG-014 Decision 7 — the
    # `capture=` dataclass is owned by this cluster, but a caller-supplied
    # stand-in with the same attribute shape is accepted, not required to
    # be `isinstance CaptureSpec`). `0` (default) when `capture` is `None`
    # or exposes no `top_k` at all — Tier 1 stays off, byte-identical to
    # every pre-P-B run.
    capture_top_k = getattr(capture, "top_k", 0) if capture is not None else 0
    # `capture.capture_full_distribution`/`capture.max_full_distribution_steps`
    # (ADR-CDG-014 Decision 3 Tier 2, issue #61 P-C): same duck-typed read as
    # Tier 1's `top_k` above. `False`/`None` (defaults) when `capture` is
    # `None` or exposes neither attribute — Tier 2 stays off, byte-identical
    # to every pre-P-C run. `validate_ingress` above already rejected
    # `capture_full_distribution=True` with no budget, so by the time this
    # line runs a `True` value is always paired with a positive budget.
    capture_full_distribution = getattr(capture, "capture_full_distribution", False) if capture is not None else False
    capture_max_full_distribution_steps = (
        getattr(capture, "max_full_distribution_steps", None) if capture is not None else None
    )

    collector = _FrameCollector(
        scheduler=scheduler,
        t_min=t_min,
        t_max=t_max,
        keep_frames=keep_frames,
        on_frame=on_frame,
        constraints=constraints,
        top_k=capture_top_k,
        capture_full_distribution=capture_full_distribution,
        max_full_distribution_steps=capture_max_full_distribution_steps,
    )
    step_end = StepEndComposite(
        capture=collector.on_step_end,
        should_cancel=should_cancel,
        pin=pin_participants,
        walker=walker_participant,
    )

    if thinking:
        prompt_kwargs: dict = {
            "messages": [
                {"role": "system", "content": THINK_TOKEN},
                {"role": "user", "content": prompt},
            ]
        }
    else:
        prompt_kwargs = {"prompt": prompt}

    try:
        # `install_logit_shaping_hook` (#35 R5, F4): the ONE place `dgemma/`
        # installs a forward hook on the loaded model, torn down by its own
        # `finally` on every exit from this `with` block — clean return,
        # `DiffusionCancelled` below, or any other exception propagating out
        # of `pipeline(...)`/the injected-cache drive body. No hook survives
        # past this block under any of the three paths (ADR-CDG-010 Decision
        # 5, ARCHITECTURE.md rule 6).
        with install_logit_shaping_hook(dgemma_model.model, logit_hook):
            if kv_cache is not None:
                # ADR-CDG-012 IN-2 (issue #62 Phase 4): drive the decoder off
                # the injected cache instead of calling `pipeline(...)` —
                # diffusers offers no injected-cache parameter to call
                # through (see `_run_pipeline_with_injected_cache`'s
                # docstring). Same composite/collector, same hook context,
                # same `DiffusionCancelled` handling below.
                #
                # ADR-CDG-024 (issue #257): pass the SAME `prompt_kwargs`
                # dict the no-cache branch below uses (built once above,
                # regardless of `kv_cache`) — but only when `prompt` is
                # non-empty; `None`/empty degrades to pure injection
                # (`prompt_kwargs=None` default). Same empty-prompt
                # definition `reject_prompt_and_kv_cache` used before its
                # tombstone (`prompt is None or not prompt.strip()`).
                composed_prompt_kwargs = prompt_kwargs if (prompt is not None and prompt.strip()) else None
                sequences, _advanced_cache = _run_pipeline_with_injected_cache(
                    pipeline,
                    kv_cache=kv_cache,
                    gen_length=gen_length,
                    num_inference_steps=num_inference_steps,
                    confidence_threshold=confidence,
                    generator=generator,
                    callback_on_step_end=step_end,
                    prompt_kwargs=composed_prompt_kwargs,
                )
                output = None
            else:
                output = pipeline(
                    **prompt_kwargs,
                    gen_length=gen_length,
                    num_inference_steps=num_inference_steps,
                    confidence_threshold=confidence,
                    generator=generator,
                    callback_on_step_end=step_end,
                    # "logits" (ADR-CDG-014 Decision 4, issue #14): the Tier 0
                    # entropy capture's source — already a base-pipeline
                    # `_callback_tensor_inputs` allowlist entry
                    # (`pipeline_diffusion_gemma.py:76`), so widening this list
                    # is all `run_diffusion` needs to do; `_FrameCollector.
                    # on_step_end` derives `DiffusionFrame.entropy` from it.
                    callback_on_step_end_tensor_inputs=["canvas", "logits", "scheduler_output"],
                )
                sequences = output.sequences[0]
    except DiffusionCancelled:
        # #38 partial-return semantics: return the evidence already
        # captured rather than raising it away. Under the capture-first
        # amendment the last captured frame IS the cancelled step's own
        # committed frame — the run's exact truncation point — and its
        # canvas stands in for the pipeline's (never-produced)
        # `output.sequences` — same excision/decode path as the completed
        # case, so a cancelled run's `CanvasState`/`CanvasTrace` are built
        # the identical way a completed run's are, not a special-cased
        # shape.
        #
        # No-frames guard: unreachable through the composite's own flow
        # (capture precedes the cancellation check, and the collector
        # always appends a frame before returning), kept as defensive
        # honesty against a `DiffusionCancelled` raised from anywhere else
        # in the pipeline call — with zero evidence, re-raising is honest
        # and `_build_result` would otherwise mint a fabricated-empty
        # `CanvasState` (or die in `derive_canvas_state` with a less
        # truthful error).
        if not collector.frames:
            raise
        sequences = collector.frames[-1].canvas
        # `DiffusionFrame.canvas` may be 1-D `[canvas_len]` or 2-D
        # `[batch, canvas_len]` (same shape ambiguity `decode_frames`
        # resolves, `dgemma/loop.py`'s `decode_frames` docstring) — the
        # completed path always hands `_build_result` a 1-D sequence
        # (`output.sequences[0]`), so the cancelled path normalizes the
        # same way rather than introducing a second shape contract.
        if hasattr(sequences, "dim") and sequences.dim() == 2:
            sequences = sequences[0]
        return _build_result(
            dgemma_model=dgemma_model,
            pipeline=pipeline,
            scheduler=scheduler,
            sequences=sequences,
            collector=collector,
            entropy_bound=entropy_bound,
            t_min=t_min,
            t_max=t_max,
            num_inference_steps=num_inference_steps,
            injected_cache_provenance=kv_cache.provenance if kv_cache is not None else None,
        )

    return _build_result(
        dgemma_model=dgemma_model,
        pipeline=pipeline,
        scheduler=scheduler,
        sequences=sequences,
        collector=collector,
        entropy_bound=entropy_bound,
        t_min=t_min,
        t_max=t_max,
        num_inference_steps=num_inference_steps,
        injected_cache_provenance=kv_cache.provenance if kv_cache is not None else None,
    )


def _build_result(
    *,
    dgemma_model: DGemmaModel,
    pipeline: Any,
    scheduler: Any,
    sequences: Any,
    collector: "_FrameCollector",
    entropy_bound: float,
    t_min: float,
    t_max: float,
    num_inference_steps: int,
    injected_cache_provenance: "Provenance | None" = None,
) -> tuple[str, CanvasState, CanvasTrace]:
    """Shared tail of `run_diffusion`'s completed and cancelled paths:
    thought-channel excision, decode, `CanvasState`/`CanvasTrace`
    construction — identical for both so a cancelled run's returned shape is
    not a special case a caller has to branch on (#38: "return what exists"
    means the same contract, populated with less).

    `injected_cache_provenance` (ADR-CDG-012 OUT-3, issue #62 Phase 2):
    `kv_cache.provenance` when `run_diffusion` received a non-`None`
    `kv_cache=`, `None` otherwise — passed straight onto
    `CanvasTrace.injected_cache_provenance` below. Identity only, never the
    cache tensors (those already have their own OUT-1/OUT-2 node-output home,
    Phase 3)."""
    # ADR-CDG-014 Decision 6 (issue #11): capture the pre-excision `sequences`
    # onto `raw_canvas_ids` BEFORE `excise_thought_channel` runs below — this
    # is the only point the final raw (un-excised) canvas ids are ever
    # reachable; `CanvasState.canvas_ids` stays post-excision (the #8
    # contract, unchanged). Plain `list[int]`, mirroring `excise_thought_
    # channel`'s own id-level normalization, so a consumer never has to
    # branch on tensor-vs-list.
    raw_canvas_ids = [int(x) for x in sequences]

    start_id, end_id = resolve_thought_channel_ids(dgemma_model.processor)
    excision = excise_thought_channel(sequences, start_id, end_id)

    text = _decode_ids(dgemma_model.processor, excision.remaining_ids, pipeline.eos_token_id)
    # Decode and label-strip each excised span independently (the "thought\n"
    # channel-name label heads each frame, not just the first), keeping only
    # spans with real content; multiple non-empty spans are joined visibly
    # rather than jammed into one undelimited string.
    thought_parts = [
        part
        for span in excision.thought_spans
        if span
        for part in [_extract_thought_text(_decode_ids(dgemma_model.processor, span, pipeline.eos_token_id))]
        if part
    ]
    thought = "\n\n".join(thought_parts) if thought_parts else None
    canvas_ids = torch.tensor(excision.remaining_ids, dtype=torch.long)

    canvas_state = derive_canvas_state(
        text=text,
        canvas_ids=canvas_ids,
        frames=collector.frames,
        steps_used=collector.steps_used,
        thought=thought,
        stray_thought_delimiter=excision.stray_start_delimiter,
        eos_token_id=pipeline.eos_token_id,
    )
    canvas_trace = CanvasTrace(
        frames=collector.frames,
        scheduler_name=type(scheduler).__name__,
        raw_canvas_ids=raw_canvas_ids,
        scheduler_config={
            "entropy_bound": entropy_bound,
            "t_min": t_min,
            "t_max": t_max,
            # Issue #20: record BOTH, distinctly named, rather than picking
            # one and silently dropping the other. `requested` is what the
            # caller asked for; `effective` is `scheduler.num_inference_steps`
            # AFTER the pipeline's `set_timesteps` call — the actual anneal
            # denominator every frame's `t`/`temperature` was computed
            # against (same value `_FrameCollector` now reads lazily; see its
            # docstring). They are equal for today's only scheduler
            # (`EntropyBoundScheduler`, no `corrector_steps`) and diverge
            # only for a future corrector scheduler — a trace that kept only
            # `requested` would then silently misreport the schedule that
            # actually produced its own frames (ADR-CDG-001 addendum).
            "num_inference_steps_requested": num_inference_steps,
            "num_inference_steps_effective": scheduler.num_inference_steps,
        },
        injected_cache_provenance=injected_cache_provenance,
    )
    return text, canvas_state, canvas_trace
