"""dgemma/types.py — engine-native dataclasses (ADR-CDG-001's socket payloads).

ComfyUI-agnostic (ADR-CDG-003). A custom ComfyUI socket type is just a string
matched by equality; the object riding it is the corresponding dataclass here,
passed through untouched — so these dataclasses ARE the public contract, not
an implementation detail behind one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DGemmaModel:
    """A loaded DiffusionGemma model + its processor + device/dtype info.

    Rides the `DGEMMA_MODEL` socket. Load seam per ADR-CDG-002 (unchanged by
    ADR-CDG-004): `DiffusionGemmaForBlockDiffusion.from_pretrained()` +
    `AutoProcessor.from_pretrained()`.
    """

    model: Any  # transformers.DiffusionGemmaForBlockDiffusion
    processor: Any  # transformers processor (AutoProcessor.from_pretrained(...))
    device: str
    dtype: str  # human-readable compute dtype label — always "bfloat16" since issue #18
    repo_id: str
    quant: str  # "none" (issue #18 — bnb nf4/int8 removed, misled on this architecture)


@dataclass
class DiffusionFrame:
    """One denoising step's captured state.

    Keyed on `(canvas_idx, step_idx, t, temperature)` — block identity plus
    the absolute position in the noise schedule — never on loop index alone:

    - `step_idx` resets to 0 for every canvas/block (the denoising loop is
      nested inside the outer autoregressive canvas loop,
      `pipeline_diffusion_gemma.py:318,356`), so once `gen_length` exceeds
      the model's `canvas_length`, `(step_idx, t, temperature)` alone
      collides across blocks. `canvas_idx` says *which block's* anneal a
      frame belongs to; it is also part of the block identity a resumable
      save-state needs (ADR-CDG-005's schedule-position field, extended with
      the outer-loop coordinate).
    - a run that starts mid-schedule (a variation/renoise run,
      `loose-ends.md`) would make loop-index keying silently incomparable
      across runs (plan.md P3 rationale) — hence absolute `t`, not list
      position.

    `t` and `temperature` replicate `EntropyBoundScheduler.step()`'s inlined
    anneal formula (`scheduling_entropy_bound.py:153-155`, installed diffusers
    0.39.0) — the scheduler does not expose either value on its step output,
    so `dgemma.loop.anneal_temperature` recomputes them from the same inputs:

        fraction = (num_inference_steps - step_idx) / num_inference_steps
        temperature = t_min + (t_max - t_min) * fraction

    `t` is that `fraction`: DIMENSIONLESS, the normalized position in the
    schedule — 1.0 at the hottest (first) step, DECREASING toward (but not
    reaching exactly) `1/num_inference_steps` at the last step. Despite the
    letter, `t` is a schedule POSITION, not a temperature (see `KNOB_DOCS`,
    `dgemma/loop.py`, for the full terms-and-units vocabulary this pack mints
    once). `temperature` is the resulting annealed sampling temperature: the
    divisor `T` applied once per step in `softmax(z / T)` upstream of both
    candidate sampling and the acceptance-entropy computation — also
    dimensionless, `T=1` reproducing the model-native logit calibration.
    Both ride the frame so a reader never has to reconstruct one from the
    other against the scheduler config.

    `num_inference_steps` in that formula is the scheduler's EFFECTIVE value
    (`scheduler.num_inference_steps` after `set_timesteps`), not necessarily
    the value `run_diffusion` was called with — see `_FrameCollector`'s
    docstring (`dgemma/loop.py`, issue #20) for the corrector-scheduler case
    where the two diverge.

    Commit info is **per-example**: `committed_fraction_per_example[i]` is
    the fraction of example i's canvas positions accepted this step (mean of
    `scheduler_output.accepted_index[i]` over the block dim only — never the
    batch dim, which would silently blend examples). The scalar
    `committed_fraction` property is a batch_size==1 convenience and raises
    on batched frames rather than inventing a blended number.
    """

    canvas_idx: int
    step_idx: int
    t: float
    temperature: float
    committed_fraction_per_example: tuple[float, ...]
    canvas: Any  # torch.LongTensor snapshot of the canvas after this step (scheduler_output.prev_sample)

    entropy: Any | None = None
    """Tier 0 capture (ADR-CDG-014 Decision 3, issue #14): per-position
    predictive entropy, `float32[canvas_len]`, in NATS — derived from this
    step's **pre-pin** `logits` (`Categorical(logits=...).entropy()`, and
    `torch.distributions.Categorical.entropy()` is natural-log, not bits/log2)
    in the capture participant — which runs FIRST in the composite (`capture
    -> cancel -> beta-rebuild -> pin`, ADR-CDG-010). Directly comparable to
    `effective_entropy_bound` below (same nats unit — the acceptance check
    this value is measured against) and to `KNOB_DOCS["entropy_bound"]`'s
    (`dgemma/loop.py`) 18-bits-per-position ≈ 12.48 nats scale reference.
    `entropy` is therefore the model's own predictive entropy over the
    canvas, not a post-pin/post-constraint artifact. Always populated when
    `logits` are reachable (`run_diffusion` requests `"logits"` in
    `callback_on_step_end_tensor_inputs`) — this is the "always on" Tier 0
    default (ADR-CDG-014 Decision 3): the cheapest honest slice of the
    DISTRIBUTION seam, ~1 KB/step.

    `None` means "not captured this run" (e.g. `logits` unreachable),
    **never** "every position had zero entropy" — reading a `None` field as
    a zero-valued measurement is exactly the ADR-CDG-001 lying-payload trap
    ADR-CDG-014 Decision 2 forbids. A consumer reads absence honestly
    (raises or skips)."""

    top_k_ids: Any | None = None
    """Tier 1 capture (ADR-CDG-014 Decision 3, `capture=` `top_k` knob,
    issue #14): per-position top-k candidate token ids, `int64[canvas_len,
    k]`, derived from this step's pre-pin `logits` alongside `entropy`.
    `None` when Tier 1 was not requested this run (`top_k=0`, the default)
    — absence, not an empty/degenerate capture. P-A does not populate this
    field (it ships in P-B); it exists now under the additive-optional
    discipline so the frame shape is decided once."""

    top_k_weights: Any | None = None
    """Tier 1 capture's companion: per-position top-k weights (softmax
    probabilities over `top_k_ids`), `float32[canvas_len, k]`. Same
    absence-vs-empty semantics as `top_k_ids`; the two are always populated
    or absent together. P-A does not populate this field (P-B scope)."""

    distribution: Any | None = None
    """Tier 2 capture (ADR-CDG-014 Decision 3, `capture_full_distribution`
    knob + `max_full_distribution_steps` budget, issue #61 P-C): the full
    per-position distribution, `float32[canvas_len, vocab]`
    (`softmax(logits)`) derived from this step's **pre-pin** `logits` in the
    capture participant (the same ordering guarantee `entropy`/`top_k_ids`
    already have, ADR-CDG-014 Decision 4) — ~134 MB/step, budget-gated at
    ingress (`capture_full_distribution=True` with no
    `max_full_distribution_steps` is rejected; an unbounded request is
    never silently honored, rule 5 `EMIT-CANONICAL / PARSE-AT-THE-DOOR`).
    `None` when Tier 2 was not requested (`capture_full_distribution=False`,
    the default), or when this step fell outside the retained budget — the
    budget caps *retained* frames regardless of `keep_frames`
    (ADR-CDG-014 Decision 5), so a run with `keep_frames="all"` and a
    budget of N still shows `distribution=None` on every frame past the
    Nth captured step. This completes the DISTRIBUTION socket (issue #61
    P-C) — the full-per-position capture surface H0-observe/H0-project
    (`docs/experiments/liquid-phase-decoding/concept.md`) read from."""

    pinned_mask: Any | None = None
    """ADR-CDG-010 Decision 4: boolean `[gen_length]` tensor/sequence, `True`
    at every canvas position the constraint layer re-asserts this frame —
    the field that keeps a frame's commit information honest by
    distinguishing "the model committed this" from "the constraint layer
    forced this regardless of what the model would have chosen".

    **Issue #64 Phase 2 scope (gate correction A1):** no pin participant
    exists yet (Phase 3 — `PinParticipant` is `NOT-YET-IMPLEMENTED`), so this
    field is populated directly from a supplied `Constraints` payload's pin
    positions when one is given — the validated-then-ignored payload's
    positions still describe *which cells would be pinned*, even though
    nothing yet writes them. This is a static-from-`Constraints.pins`
    computation, valid **only because and only while** pins are
    position-static (the D6 id-level hard-pin invariant: a hard pin
    re-asserts the same positions every step, so "which cells the constraint
    layer touches this frame" is provably the constant pin-position set for
    the whole run). It is **not** licensed in general — ADR-CDG-010 Open
    Question 3 explicitly parks a future soft/dynamic constraint that could
    re-pin different cells per step, at which point this static shortcut
    would silently become a lying trace. **Labeled door, not welded shut:**
    a future dynamic/re-pinning constraint type must switch capture to
    observe the pin participant's actual per-step write instead of deriving
    this field from the static pin set.

    `None` when no `Constraints` payload was supplied this run (additive-
    optional discipline, ADR-CDG-014 Decision 1) — never an all-`False`
    mask standing in for "no pins"."""

    effective_entropy_bound: float | None = None
    """ADR-CDG-011 clause 7 (honest telemetry): `scheduler.config.
    entropy_bound` read at THIS callback, i.e. the value `step()` actually
    consumed producing this frame — in NATS, same unit as `entropy` above
    (both natural-log; see `KNOB_DOCS["entropy_bound"]`, `dgemma/loop.py`,
    for the full units note) — never the `run_diffusion` ctor snapshot or a
    control-signal binding's static curve. A walker (Phase 4,
    `NOT-YET-IMPLEMENTED`) that mutates `scheduler.config` mid-run makes this
    field diverge from the run's requested `entropy_bound`; a walker bug
    that silently fails to write through is therefore visible in the trace
    rather than papered over. `None` only when the scheduler exposes no
    `.config.entropy_bound` at all (never populated this phase means a
    scheduler-shape regression, not a normal path — every real/fake
    scheduler in this suite exposes it)."""

    effective_t_min: float | None = None
    """Companion to `effective_entropy_bound`: `scheduler.config.t_min` read
    at this callback (ADR-CDG-011 clause 7) — a TEMPERATURE (dimensionless
    softmax divisor), not a schedule position, despite the lowercase-`t`
    name (see `KNOB_DOCS["t_min"]`, `dgemma/loop.py`). Feeds the same
    live-`t_min` read `anneal_temperature` now uses for `t`/`temperature`
    (see `_FrameCollector`), so a walker-mutated `t_min` is reflected
    consistently across `t`/`temperature` and this field."""

    effective_t_max: float | None = None
    """Companion to `effective_entropy_bound`/`effective_t_min`:
    `scheduler.config.t_max` read at this callback (ADR-CDG-011 clause 7) —
    same TEMPERATURE units as `effective_t_min` (see `KNOB_DOCS["t_max"]`,
    `dgemma/loop.py`)."""

    @property
    def committed_fraction(self) -> float:
        """Scalar commit fraction — defined only for single-example frames."""
        if len(self.committed_fraction_per_example) != 1:
            raise ValueError(
                "committed_fraction is a batch_size==1 convenience; this frame has "
                f"{len(self.committed_fraction_per_example)} examples — use "
                "committed_fraction_per_example."
            )
        return self.committed_fraction_per_example[0]


@dataclass
class EditOp:
    """One tier-2 perturbation op (ADR-CDG-012 §D.0, `Provenance.edit_script`
    member). The against-distribution edit-script entry: what was done to a
    `KVCache` when no single minting sequence reproduces it anymore (ADR
    §5, §1). Tier-2 surgery (`dgemma/kv_surgery.py`) is out of Phase 1 scope
    (issue #62 ratification, Q-1: tier-2 deferred) — this dataclass lands now
    because `Provenance.edit_script` needs its element type decided once,
    but no op-producing code exists yet in Phase 1.
    """

    op: str
    """One of `"splice"` | `"ablate"` | `"scale"` | `"ablate_full_attention"`
    (ADR-CDG-012 §5) — a closed, ADR-named vocabulary, not a caller-invented
    string. Phase 1 does not construct any `EditOp`; the field exists so
    tier-2 (Phase 5) has a decided shape to append to."""

    params: dict
    """Op-specific parameters (layer indices, source cache id, scale factor,
    ...) — shape depends on `op`; not further typed here (ADR-CDG-012 §5
    leaves per-op parameter shape to the tier-2 surgery implementation)."""


@dataclass
class Provenance:
    """The mint record (ADR-CDG-012 §1, §D.0): the record that keeps a
    `KVCache` non-lying (`EMIT-CANONICAL / PARSE-AT-THE-DOOR` applied to a
    live cache object instead of a tensor payload). Rides every `KV_CACHE`
    crossing (node-to-node, disk) as part of the `KVCache` payload, and —
    identity only, never the tensors — on `CanvasTrace.injected_cache_provenance`
    (OUT-3) when a run was driven with an injected cache.

    Tier 1 (the only tier Phase 1 constructs test fixtures for; tier-2
    surgery itself is Phase-5/out-of-scope per issue #62 Q-1): `minting_sequence`
    is present, `edit_script` is empty. Tier 2: `minting_sequence` is `None`,
    `edit_script` is non-empty (no single prefill reproduces a perturbed
    cache — the tensors are the only reproduction path once perturbed).

    `minting_sequence is None and edit_script == ()` is the ILLEGAL orphan
    state (§D.0) — a cache with no reproduction path at all, unreproducible
    and unauditable. `validate_kv_cache_ingress`'s V5 check
    (`dgemma/kv_cache.py`) rejects it at every `KV_CACHE` ingress door.
    """

    minting_sequence: "tuple[int, ...] | None"
    """Token ids the encoder consumed to produce this cache — present for
    tier 1, `None` once perturbed (tier 2, no single prefill reproduces it)."""

    edit_script: "tuple[EditOp, ...]"
    """`()` for tier 1; the splice/ablate/scale ops for tier 2 (non-empty
    exactly when `minting_sequence` is `None`, except the illegal orphan
    state where both are empty/`None` — V5 rejects that combination)."""

    model_repo_id: str
    """Which model minted this cache (`DGemmaModel.repo_id`) — V4 checks
    this against the loaded model at ingress."""

    tokenizer_fingerprint: str
    """Which vocab minted this cache — V4 checks alignment against the
    loaded model's tokenizer, preventing a cache minted under one tokenizer
    from conditioning a canvas on token ids that mean something else under
    another (ADR-CDG-012 §D.0, "orphan-provenance poisoning, vocab flavor")."""


@dataclass
class KVCache:
    """The `DGEMMA_KV_CACHE` socket payload (ADR-CDG-012 §D.0). A live cache
    object plus the mint metadata that keeps it honest — the same shape as
    `CanvasTrace`/`CanvasState` (a live object + mint identity).

    The socket *string* `DGEMMA_KV_CACHE` is surface-side envelope
    (`surfaces/comfyui/socket_types.py`, ARCHITECTURE.md rule 4,
    `IDENTITY⊥ENVELOPE`); this dataclass is the core-side identity that
    string carries — Phase 1 lands only this dataclass and its ingress
    validator, not the socket mint (Phase 3, per the ADR-CDG-012
    implementation plan).
    """

    cache: Any
    """The live per-layer K/V store (`transformers.DynamicCache`). Per layer
    `i`: `key_cache[i]`, `value_cache[i]` each a tensor of shape `(batch,
    num_kv_heads, seq_len, head_dim)`, dtype matching the loaded model
    (bfloat16 in production), on the model's device. Layer count must equal
    the loaded model's decoder-layer count (V1)."""

    cumulative_length: "tuple[int, ...]"
    """Per-layer running committed length — the ADR-CDG-012 grounding
    report's ranked-#1 blocker (`cache_utils.py:254`, mask offsets computed
    from it at `:270`). One entry per layer; a consumer NEVER hand-tracks
    this (V3 checks it is present, one-per-layer, non-negative)."""

    geometry: dict
    """The geometry fingerprint (ADR-CDG-012 §2): `layer_types` pattern
    (which layers are full-attention vs. sliding), `sliding_window` size,
    `batch`, `dtype`, per-layer-type RoPE params. What ingress validates
    against the loaded model's config (V2) — the mandatory check that
    forecloses silent mis-masking from a geometry mismatch."""

    provenance: Provenance
    """The mint record (see `Provenance` above) — V4 (vocab/repo alignment)
    and V5 (non-orphan) check this at ingress."""


@dataclass
class CanvasTrace:
    """The complete per-step record of one `run_diffusion` call — the
    `CANVAS_TRACE` socket payload (ADR-CDG-001), consumed post-hoc by
    `DGemmaTrace` (plan.md Phase 3 (b)).

    ADR-CDG-001's addendum on scheduler-relative commit semantics is
    non-negotiable here: a commit mask (or, as here, a per-step
    `committed_fraction` reading) means something different depending on
    which scheduler minted it — a ratchet under `BlockRefinementScheduler`,
    a stateless per-step reading under `EntropyBoundScheduler`. A
    `CanvasTrace` that carried `frames` alone, without saying which
    scheduler produced them and with what config, would be a lying payload
    the same way a disguised `SIGMAS` tensor is. `scheduler_name` +
    `scheduler_config` are therefore not optional metadata — they are the
    mint identity that gives the frames' commit readings their meaning.

    Frames are already self-keyed (`DiffusionFrame`'s own
    `(canvas_idx, step_idx, t, temperature)` identity, see that class's
    docstring) — this type carries the collection as-is, no new keying
    logic.
    """

    frames: list[DiffusionFrame]
    scheduler_name: str
    scheduler_config: dict
    """The `entropy_bound` (nats)/`t_min`/`t_max` (temperatures — dimensionless
    softmax(z/T) endpoints, NOT schedule positions despite the lowercase-`t`
    names) requested for this run, plus `num_inference_steps_requested`/
    `num_inference_steps_effective` (see `_build_result`, `dgemma/loop.py`,
    for the full per-key provenance). Same units as `KNOB_DOCS`
    (`dgemma/loop.py`) mints once for every knob-facing door — see there for
    the authoritative terms-and-units vocabulary this dict's keys instantiate."""

    raw_canvas_ids: Any | None = None
    """Pre-excision final canvas ids (ADR-CDG-014 Decision 6, issue #11):
    the un-excised `sequences` `run_diffusion` produced, captured in
    `_build_result` BEFORE `excise_thought_channel` runs — the raw view
    `CanvasState.canvas_ids` (post-excision, #8 contract) deliberately does
    not carry. Lives on the TRACE side, never on `CanvasState`
    (ADR-CDG-005's save-state/display split is load-bearing: a resumable
    save-state must not carry a thought-channel leak; a research probe —
    #9's EOS-in-thought-span probe, #3's token-identity signals — needs to
    see the pre-excision truth).

    `None` on a legacy/no-capture path (additive-optional discipline,
    ADR-CDG-014 Decision 1) — never an empty tensor standing in for "no
    ids". A consumer reads absence honestly rather than treating `None` as
    a zero-length canvas."""

    injected_cache_provenance: "Provenance | None" = None
    """OUT-3 (ADR-CDG-012 §D.2): identity-only record that this run was
    driven with an injected `KVCache` (`run_diffusion(kv_cache=...)`, Phase
    2). Carries the envelope's IDENTITY — the `Provenance` record (minting
    sequence / edit-script / model+tokenizer identity) — NEVER the cache
    tensors themselves (those already have their own OUT-1/save-node home;
    duplicating them here would violate OUT-1's one-live-cache-at-a-time
    retention policy, ADR-CDG-012 §D.2).

    `None` for every non-injected run — unchanged default (additive-optional
    per #35 R6 discipline, the same discipline `raw_canvas_ids` above
    follows). Phase 1 lands this field with no populating call site yet
    (`run_diffusion` gains no `kv_cache` parameter until Phase 2); it exists
    now so the trace shape is decided once. Without it, a fossil-wave
    ablation study's trace could not say whether the run it analyzed was
    conditioned on an injected/perturbed cache — a conclusion that looks
    grounded but isn't (ADR-CDG-012 Negative Consequences, "orphan-cache
    poisoning downstream conclusions")."""


@dataclass
class CanvasState:
    """Validity readout riding alongside the decoded `STRING` output.

    ADR-CDG-001 Addendum ("time-axis lying payload"): a bare string cannot say
    whether the canvas it was decoded from actually finished denoising — with
    too few steps or too wide an entropy bound, "finished-looking" text can
    still contain uncommitted renoise garbage. These fields are the evidence
    that keeps the `STRING` honest about its own completion state.

    `converged` is defined **honestly, not aspirationally**: `True` iff the
    *last captured frame's* `committed_fraction == 1.0` — every canvas
    position was simultaneously accepted under the entropy bound on the final
    denoising step this run captured. This is a per-step reading, not a
    ratchet: under `EntropyBoundScheduler` (this pack's P1 default scheduler)
    there is no persistent commit state to consult (ADR-CDG-001 Addendum,
    "scheduler-relative commit semantics") — `converged` says "the schedule
    bottomed out by the last step we saw," not "every position was locked in
    and has stayed locked since." It does NOT independently confirm EOS was
    emitted or that the pipeline's own `confidence_threshold` early-stop
    fired — that decision is internal to `DiffusionGemmaPipeline.__call__`'s
    loop and is not surfaced to `callback_on_step_end`. Do not read more into
    `converged` than the entropy-bound reading it actually is.

    **`converged=False` is NOT "this run failed" (issue #22 honesty
    finding)**: the pipeline's own `confidence`/`eos_early_stop` adaptive
    stopping (see `run_diffusion`'s docstring) can halt the loop once the
    answer is confidently decided while some low-stakes canvas position
    (commonly padding/trailing filler) is still short of the entropy bound
    — observed live at `committed_fraction=0.9961` (255/256) on a clean,
    correct, EOS-terminated run. `converged` reads `False` on exactly that
    screenshot. If the question being asked is "did the run finish
    honestly" rather than "did the entropy schedule bottom out on every
    single position," read `turn_closed` (EOS actually committed) and
    `answer_tokens` (how much answer that EOS closed off) instead — or the
    `finished_honestly` property below, which states that combination
    directly. `converged` keeps its narrow, literal meaning; it is not
    being redefined here.

    Scope: single-example. P1's `run_diffusion` drives one prompt (batch 1);
    deriving a `CanvasState` from a batched frame raises via
    `DiffusionFrame.committed_fraction` rather than blending examples.

    `canvas_ids`/`text` are post thought-channel-excision (P2, issue #8): the
    model emits an id-100/id-101 `<|channel>...<channel|>` frame at turn
    start (empty when `thinking=False`, possibly non-empty when
    `thinking=True`) that upstream `skip_special_tokens=True` decode does not
    fully strip — `dgemma.loop.excise_thought_channel` removes every
    well-formed span from the canvas ids before either field is derived, so
    neither payload can leak a frame (ADR-CDG-001 payload-contamination
    discipline).
    """

    text: str
    canvas_ids: Any  # torch.LongTensor, final canvas token ids (prompt stripped, thought-channel excised)
    converged: bool
    committed_fraction: float
    steps_used: int
    thought: str | None = None
    """Decoded content of the excised thought channel(s), or `None` when the
    channel was empty (the `thinking=False` common case — the model card
    notes an empty channel "might still be emitted" even with thinking off)
    or absent entirely; multiple non-empty channels are joined with a blank
    line. Not surfaced on the `STRING` payload (ADR-CDG-001: payloads mean
    what they say — the answer text is the answer text); this is the
    "natural slot" decision for issue #8's optional thought-surfacing ask,
    landing on `CanvasState` since it already carries validity/diagnostic
    readouts alongside the answer."""

    stray_thought_delimiter: bool = False
    """`True` iff the canvas held an unmatched `<|channel>` start delimiter
    past the head of the generated region — a malformed frame
    `excise_thought_channel` deliberately did NOT excise, because
    excise-to-end there would silently destroy answer text (review finding,
    2026-07-05). The answer `STRING` keeps all surrounding text (the
    delimiter itself vanishes in `skip_special_tokens=True` decode); this
    flag is the validity-side signal that an anomalous frame remnant exists
    rather than letting the condition vanish."""

    turn_closed: bool = False
    """Issue #9's honesty-readout half: `True` iff `eos_token_id` was found
    committed somewhere in the (thought-excised) answer ids — the turn ran
    to a real stop, not just "the canvas is full of plausible-looking
    tokens." `False` covers BOTH of issue #9's named specimens: an
    all-thought/empty-answer turn (nothing to find EOS in) and a
    budget-truncated answer that hit `gen_length` mid-token with
    `converged=True, committed_fraction=1.0` — the exact gap `converged`
    alone cannot see (`CanvasState.converged`'s docstring already says it
    doesn't confirm EOS). `turn_closed` is deliberately independent of
    `converged`: a run can converge (every position locked in this step)
    while still not being turn_closed (it locked in on non-EOS filler
    because the canvas ran out first)."""

    answer_tokens: int = 0
    """Issue #9's companion honesty field: the count of (thought-excised)
    answer ids **before the first EOS** — the EOS itself and any trailing
    EOS/renoise fill run (a converged run pads the canvas tail with one;
    ~30 tokens observed live) are excluded, mirroring `_decode_ids`'s own
    trim, because a bare `len(canvas_ids)` would inflate the count by that
    padding and defeat the field's honesty purpose (review finding,
    2026-07-05). When no EOS is present, the full thought-excised length
    counts (every id is content the budget-truncated canvas actually
    holds); `0` when `canvas_ids` is unavailable (e.g. a unit test
    constructing `CanvasState` directly without a real canvas). Read
    alongside `turn_closed`: a small `answer_tokens` with
    `turn_closed=False` is the all-thought/empty-answer specimen; a large
    `answer_tokens` with `turn_closed=False` is the budget-truncated
    specimen — same field pair, different failure shape."""

    @property
    def finished_honestly(self) -> bool:
        """The single reassurance field for "did this run actually finish
        well" (issue #22 honesty finding): `turn_closed` alone, restated
        under the name people reach for `converged` expecting to find it.

        `converged` intentionally does NOT answer this question (see its
        own docstring) — it can read `False` on a clean, correct,
        adaptive-stopped run (observed live: 255/256 committed, EOS emitted,
        answer complete) because the pipeline's `confidence`/
        `eos_early_stop` early-stopping can halt before every last canvas
        position individually clears the entropy bound. `turn_closed`,
        which checks for a real committed EOS in the answer ids, is
        unaffected by that gap — that IS the "finished honestly" signal.
        This property adds no new information (`turn_closed` already carries
        it); it exists so the question "did it finish?" has an honestly-named
        home instead of pulling readers toward `converged` by default."""
        return self.turn_closed
