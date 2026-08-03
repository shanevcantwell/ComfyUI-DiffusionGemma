"""dgemma/config.py — knob defaults + the ONE-MINT terms-and-units vocabulary
(issue #129, ADR-CDG-018 Stage 1: `dgemma/loop.py` decomposition).

Pure constants, no loop state, no diffusers/torch dependency of its own.
Extracted verbatim from `dgemma/loop.py` (no behavior change — see
`tests/test_loop_golden_trace.py`, the decomposition's no-behavior-change
oracle). `dgemma/loop.py` re-imports every name below and re-exports it, so
every existing `from dgemma.loop import DEFAULT_*` / `KNOB_DOCS` / `THINK_TOKEN`
/ `THOUGHT_CHANNEL_*` import site keeps working unchanged (the ratified plan's
facade ruling, issue #129).
"""
from __future__ import annotations

# Grounded defaults (CLAUDE.md / plan.md — first local run, Q4_K_M).
DEFAULT_NUM_INFERENCE_STEPS = 48
DEFAULT_T_MIN = 0.4
DEFAULT_T_MAX = 0.8
DEFAULT_ENTROPY_BOUND = 0.1
DEFAULT_GEN_LENGTH = 256
DEFAULT_CONFIDENCE = 0.005

# ONE-MINT terms-and-units vocabulary (units-glossary-tooltips work item):
# every knob's units/semantics live HERE, once, and every door that surfaces
# a knob — ComfyUI widget tooltips (`surfaces/comfyui/sampler.py`), the MCP
# `generate`/`load_model` JSON-schema `description`s (`surfaces/mcp/commands/
# generate.py`, `surfaces/mcp/commands/model.py`) — imports and re-uses this
# text rather than re-typing it (rule-8 parity by construction: two doors
# describing one knob can't drift apart if there is only one string). This is
# the doctrine's `EMIT-CANONICAL` discipline applied to prose, not just
# payloads: a tooltip and a schema description are two renderings of the same
# canonical fact, not two independent claims that happen to agree today.
#
# Provenance for the units claims themselves (authoritative, transcribed from
# the operator's terms-and-units brief, not re-derived here):
#
# - `T` (sampling temperature — the WIDGET value users think of as "the
#   temperature") is the divisor in `softmax(z / T)`: a dimensionless scale
#   factor on the model-native logit calibration, `T=1` reproducing the
#   trained calibration exactly. Applied ONCE per step, upstream of BOTH
#   candidate sampling and the acceptance-entropy computation — there is no
#   second, independently-tunable temperature hiding downstream of it.
# - `t` (schedule position — NOT a temperature, despite the shared letter) is
#   `(N - step_idx) / N`: dimensionless, DECREASING from 1 down to `1/N`
#   (reached exactly at the last step, `step_idx == N-1`) — toward but never
#   reaching 0 — as `step_idx` runs 0..N-1. See `anneal_temperature`
#   (`dgemma/capture.py`) for the exact formula this recomputes.
# - `t_min`/`t_max` are TEMPERATURE endpoints (config knobs), not schedule
#   positions, despite the lowercase-`t` naming: `T = t_min + (t_max -
#   t_min) * t` (the affine map `anneal_temperature` evaluates). `t_min` is a
#   virtual endpoint no real step ever actually applies — `t` bottoms out at
#   `1/N`, never 0, so the coldest realized temperature is
#   `t_min + (t_max - t_min) / N`, strictly above `t_min` itself. These field
#   names come from the upstream `EntropyBoundScheduler` checkpoint config
#   (`scheduling_entropy_bound.py`) — do NOT rename them to something more
#   self-describing; that would desync this pack's kwargs from the installed
#   diffusers scheduler's own `.config` attribute names `_FrameCollector`
#   reads live (`effective_t_min`/`effective_t_max`, `dgemma/types.py`).
# - `entropy_bound` is the per-step joint acceptance budget in NATS:
#   `torch.distributions.Categorical.entropy()` (what this pack's capture
#   path and the scheduler both use) is natural-log entropy, not bits.
#   Default `0.1` nats. For scale: the 18-bits-per-position uniform-vocabulary
#   melt VISION.md opens with is `18 * ln(2) ≈ 12.48` nats — i.e. roughly two
#   orders of magnitude hotter than the default per-step acceptance budget,
#   not directly comparable to it (one is the corruption entropy of the
#   INITIAL canvas draw; the other is a per-step ACCEPTANCE threshold), but
#   sharing the same nats unit is what makes that comparison meaningful at
#   all rather than a bits-vs-nats category error.
# - `confidence` is the pipeline's early-stop threshold: a dimensionless
#   probability (not a unit-bearing quantity at all).
KNOB_DOCS: dict[str, str] = {
    "t_min": (
        "Cold end of the temperature anneal (dimensionless, applied as the "
        "divisor T in softmax(z/T)). Despite the lowercase-t name this is a "
        "TEMPERATURE, not a schedule position — t bottoms out at 1/"
        "num_inference_steps, so t_min itself is a virtual endpoint no step "
        "actually reaches. T = t_min + (t_max - t_min) * t, t decreasing 1 -> "
        "1/num_inference_steps across the run."
    ),
    "t_max": (
        "Hot end of the temperature anneal (dimensionless, same softmax(z/T) "
        "divisor as t_min) — the temperature applied at the very first step, "
        "where the schedule position t == 1."
    ),
    "entropy_bound": (
        "Per-step joint acceptance budget, in NATS (natural-log entropy, "
        "matching torch.distributions.Categorical.entropy() — not bits). "
        "A position commits this step only once its acceptance entropy "
        "clears this bound. Default 0.1 nats; for scale, the uniform-vocab "
        "noise draw's 18 bits/position is ~12.48 nats."
    ),
    "confidence": (
        "Early-stop threshold: a dimensionless probability the pipeline's "
        "adaptive-stop check compares a candidate's confidence against."
    ),
    "num_inference_steps": (
        "Requested denoising step budget N (a plain count, not a physical "
        "unit) — the schedule-position denominator: t = (N - step_idx)/N."
    ),
    "gen_length": (
        "Canvas length in tokens (a token count) — how many positions the "
        "denoising loop allocates for the generated turn. This is split "
        "into blocks of DEFAULT_GEN_LENGTH (256) tokens each: gen_length is "
        "processed as ceil(gen_length / 256) blocks, each denoised over "
        "num_inference_steps substeps (e.g. gen_length 1024 -> 4 blocks, "
        "each denoised over num_inference_steps steps). Larger gen_length "
        "costs proportionally more block passes."
    ),
    "seed": (
        "RNG seed for the generator driving canvas initialization/renoise "
        "(a plain integer, not unit-bearing). Omit/leave unset for a "
        "nondeterministic run."
    ),
    "thinking": (
        "EXPERIMENTAL boolean toggle: injects the <|think|> control token "
        "via a system turn. Structurally one token short of native "
        "enable_thinking=True (the chat template's `| trim` eats the "
        "newline after <|think|>) — see run_diffusion's own docstring for "
        "the honest gap. Behavioral impact unverified pending an E2E "
        "thinking-mode run on real weights."
    ),
}

# ONE-MINT provenance (issue #8 / model-card "thinking" toggle): these
# literal strings are the DiffusionGemma tokenizer's control tokens, sourced
# from `google/diffusiongemma-26B-A4B-it`'s `tokenizer_config.json`
# (`model_specific_special_tokens`: `think_token="<|think|>"`,
# `soc_token="<|channel>"`, `eoc_token="<channel|>"`), cross-checked against
# the cached `tokenizer.json` `added_tokens` table (2026-07-05): id 98
# (`<|think|>`), id 100 (`<|channel>`), id 101 (`<channel|>`). The chat
# template's `<|channel>thought\n...content...\n<channel|>` framing (see
# `chat_template.jinja`) is what issue #8 excises. `THOUGHT_CHANNEL_START_ID`/
# `THOUGHT_CHANNEL_END_ID` are the fallback only — `resolve_thought_channel_ids`
# (`dgemma/excision.py`) prefers reading them off the loaded processor's own
# tokenizer vocab, so a checkpoint swap that renumbers ids can't silently
# desync from a hardcoded pair.
THINK_TOKEN = "<|think|>"
THOUGHT_CHANNEL_START_TOKEN = "<|channel>"
THOUGHT_CHANNEL_END_TOKEN = "<channel|>"
THOUGHT_CHANNEL_START_ID = 100
THOUGHT_CHANNEL_END_ID = 101

# Provenance: `chat_template.jinja` always renders the channel as
# `'<|channel>thought\n' + thinking_text + '\n<channel|>'` — "thought" is a
# fixed channel-NAME label the template emits before any real content, not
# part of the reasoning text itself. Verified against the installed
# tokenizer (`AutoTokenizer.from_pretrained`, cached weights, 2026-07-05):
# decoding ids `[45518, 107]` (the label's own ids) with
# `skip_special_tokens=True` yields exactly `"thought\n"` — confirming the
# canonical *empty* channel (issue #8's `[100, 45518, 107, 101, ...]`) is the
# label with nothing after it. String-level label strip (not a special
# token — ordinary vocab), applied only to the already id-isolated
# between-delimiter span, never to the full decoded payload.
THOUGHT_CHANNEL_LABEL = "thought"
