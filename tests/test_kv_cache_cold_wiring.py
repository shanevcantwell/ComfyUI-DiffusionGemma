"""tests/test_kv_cache_cold_wiring.py — ADR-CDG-012 DV.3c (issue #62 Phase 3)
+ issue #207 (fail-loud at the inert `kv_cache` door): the "effortless"
guarantee, made executable.

Constructs the MINIMAL `DGemmaEncode -> DGemmaDenoise` graph PROGRAMMATICALLY
— calling the node bodies directly with all-default parameters, NOT loading
a shipped `examples/*.json` (that is DV.2's job, `tests/test_kv_cache_workflows.py`)
— and asserts the result matches the current contract. Independent of DV.2's
fixtures by construction: this test cannot be satisfied by a hand-tuned
example file, only by the node signatures + engine actually composing
correctly with no tribal knowledge.

Post-#207, "effortless to get valid results" (DV.3c) is served by TWO
distinct assertions rather than one: (1) the minimal graph WITH `kv_cache`
wired fails loud, naming issue #62 Phase 4, because that path cannot yet
honor an injected cache (#207's fail-loud ruling supersedes DV.3c's
pre-#207 "wire the two nodes together, get output" framing for THIS
specific wiring); (2) the minimal `DGemmaDenoise`-alone graph (`kv_cache`
omitted) remains legal-and-non-degenerate — DV.3c's guarantee holds in full
for every wiring this path CAN honor today.

Deliberately builds its own minimal decode-capable fake model + fake
scheduler/pipeline (mirrors `tests/test_kv_cache_run_diffusion.py`'s
`_kv_capable_fake_model`/`_install_fakes` shape) rather than importing that
module's private helpers — DV.3c's own text is explicit that this test must
be independent of any other fixture set built for a different clause.
"""
from __future__ import annotations

import pytest
import torch

from dgemma.loop import DEFAULT_ENTROPY_BOUND, DEFAULT_T_MAX, DEFAULT_T_MIN
from dgemma.types import DGemmaModel
from surfaces.comfyui.denoise import DGemmaDenoise
from surfaces.comfyui.encode import DGemmaEncode
from tests.conftest import FakeDGemmaModelConfig


class _FakeTokenizer:
    eos_token_id = 999
    unk_token_id = 0
    vocab_size = 32

    def convert_tokens_to_ids(self, token):
        return None

    def decode(self, ids, skip_special_tokens=True):
        return "TEXT:" + ",".join(str(i) for i in ids)

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % self.vocab_size for ch in text] or [0]


class _FakeProcessor:
    tokenizer = _FakeTokenizer()


class _FakeEncoderOutput:
    def __init__(self, past_key_values):
        self.past_key_values = past_key_values


class _FakeEncoderModel:
    """Minimal `.model.model.encoder` stand-in — same shape as
    `tests/conftest.py`'s `_FakeEncoderModel`, duplicated here (not
    imported) to keep this cold-wiring test's fixture set self-contained
    per DV.3c's independence requirement.

    `parameters()` (issue #187): `encode_sequence` reads
    `next(encoder.parameters()).device` to pin minted tensors — this fake
    must expose that surface too, or the minimal-graph call crashes on
    `AttributeError` before DV.3c's own assertions ever run."""

    def __init__(self, num_hidden_layers: int):
        self.num_hidden_layers = num_hidden_layers

    def parameters(self):
        import torch

        yield torch.zeros(1)

    def __call__(self, *, input_ids, past_key_values=None, position_ids=None):
        from tests.conftest import FakeDynamicCache

        num_new_tokens = input_ids.shape[-1]
        cache = past_key_values or FakeDynamicCache(num_layers=self.num_hidden_layers, seq_len=0)
        cache.append(num_new_tokens)
        return _FakeEncoderOutput(past_key_values=cache)


class _FakeDiffusionGemmaModel:
    def __init__(self, config: FakeDGemmaModelConfig):
        self.encoder = _FakeEncoderModel(config.get_text_config().num_hidden_layers)


class _FakeInnerModel:
    def __init__(self, config: FakeDGemmaModelConfig):
        self.config = config
        self.model = _FakeDiffusionGemmaModel(config)


def _cold_wiring_model() -> DGemmaModel:
    config = FakeDGemmaModelConfig(num_hidden_layers=6, sliding_window=16)
    return DGemmaModel(
        model=_FakeInnerModel(config),
        processor=_FakeProcessor(),
        device="cpu",
        dtype="bfloat16",
        repo_id="fake/cold-wiring-model",
        quant="none",
    )


def _install_denoise_fakes(monkeypatch, *, num_steps: int = 2):
    class FakeSchedulerOutput:
        def __init__(self, accepted):
            self.accepted_index = torch.tensor(accepted, dtype=torch.bool)

    class FakePipelineOutput:
        def __init__(self, sequences):
            self.sequences = sequences
            self.texts = ["<<unused>>"]

    class _RecordingFrozenConfig:
        def __init__(self, **kwargs):
            object.__setattr__(self, "_values", dict(kwargs))

        def __getattr__(self, name):
            values = object.__getattribute__(self, "_values")
            if name in values:
                return values[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            raise AttributeError(f"frozen — use register_to_config, not direct set of {name!r}")

    class FakeScheduler:
        def __init__(self, *, entropy_bound, t_max, t_min, num_inference_steps):
            self._config = _RecordingFrozenConfig(
                entropy_bound=entropy_bound, t_max=t_max, t_min=t_min, num_inference_steps=num_inference_steps
            )
            self.num_inference_steps = num_inference_steps

        @property
        def config(self):
            return self._config

        def register_to_config(self, **kwargs):
            merged = dict(object.__getattribute__(self._config, "_values"))
            merged.update(kwargs)
            self._config = _RecordingFrozenConfig(**merged)

    class FakePipeline:
        def __init__(self, model, scheduler, processor):
            self._scheduler = scheduler
            self.eos_token_id = getattr(getattr(processor, "tokenizer", processor), "eos_token_id", None)

        def __call__(self, **kwargs):
            callback = kwargs["callback_on_step_end"]
            # Full acceptance every step -> a converged, non-degenerate result.
            for step_idx in range(num_steps):
                callback_kwargs = {
                    "scheduler_output": FakeSchedulerOutput([[True]]),
                    "canvas": torch.tensor([[7 + step_idx]]),
                }
                callback(self, step_idx, step_idx, callback_kwargs)
            return FakePipelineOutput(sequences=[torch.tensor([999], dtype=torch.long)])

    monkeypatch.setattr("dgemma.loop.EntropyBoundScheduler", FakeScheduler)
    monkeypatch.setattr("dgemma.loop.DGemmaPipeline", FakePipeline)


def _widget_defaults(input_types: dict) -> dict:
    """Pulls each `required` widget's declared `default` out of a node's
    `INPUT_TYPES()` spec — the node-declared defaults DV.3c's minimal-graph
    guarantee is actually about (no magic values the test itself invents).
    Skips 1-tuple socket entries (e.g. `("DGEMMA_MODEL",)`), which carry no
    widget default at all."""
    return {
        name: spec[1].get("default")
        for name, spec in input_types["required"].items()
        if isinstance(spec, tuple) and len(spec) > 1
    }


class TestMinimalKVCacheGraphIsNonDegenerate:
    """The executable form of "effortless": `DGemmaEncode` -> `DGemmaDenoise`
    with every parameter at its node default, called programmatically."""

    def test_minimal_graph_with_kv_cache_wired_fails_loud_naming_phase_4(self, monkeypatch):
        """Issue #207 (operator ruling 2026-08-01) supersedes this test's
        pre-#207 shape: wiring `DGemmaEncode`'s output into `DGemmaDenoise`'s
        `kv_cache` input is precisely the inert path — the minted cache
        passes ingress, then `run_diffusion` raises `NotImplementedError`
        naming issue #62 Phase 4, rather than silently completing a run
        that never actually consumed the cache. DV.3c's "effortless to get
        valid results" guarantee is served by the OTHER minimal graph
        (`DGemmaDenoise` alone, `kv_cache` omitted — the test below): a cold
        user who wires the pair together gets an explicit, actionable error
        naming the gap, not a silently degenerate or misleading success."""
        _install_denoise_fakes(monkeypatch, num_steps=2)
        model = _cold_wiring_model()

        encode_node = DGemmaEncode()
        (kv_cache,) = encode_node.encode(model, "hello world")

        denoise_node = DGemmaDenoise()
        denoise_defaults = _widget_defaults(DGemmaDenoise.INPUT_TYPES())

        with pytest.raises(NotImplementedError, match="Phase 4"):
            denoise_node.denoise(
                model,
                prompt="hi",
                seed=denoise_defaults["seed"],
                num_inference_steps=2,
                t_min=DEFAULT_T_MIN,
                t_max=DEFAULT_T_MAX,
                entropy_bound=DEFAULT_ENTROPY_BOUND,
                confidence=denoise_defaults["confidence"],
                gen_length=denoise_defaults["gen_length"],
                thinking=denoise_defaults["thinking"],
                kv_cache=kv_cache,
            )

    def test_minimal_graph_with_kv_cache_omitted_still_non_degenerate(self, monkeypatch):
        """`kv_cache=None` (the default, IN-2's "no injection" path) keeps
        the minimal `DGemmaDenoise`-alone graph legal-and-non-degenerate —
        DV.3c's "any KV_CACHE-specific default keeps the minimal graph legal"
        clause, exercised without `DGemmaEncode` in the graph at all."""
        _install_denoise_fakes(monkeypatch, num_steps=2)
        model = _cold_wiring_model()

        denoise_node = DGemmaDenoise()
        denoise_defaults = _widget_defaults(DGemmaDenoise.INPUT_TYPES())
        text, canvas_state, canvas_trace, frames, images, run_config = denoise_node.denoise(
            model,
            prompt="hi",
            seed=denoise_defaults["seed"],
            num_inference_steps=2,
            t_min=DEFAULT_T_MIN,
            t_max=DEFAULT_T_MAX,
            entropy_bound=DEFAULT_ENTROPY_BOUND,
            confidence=denoise_defaults["confidence"],
            gen_length=denoise_defaults["gen_length"],
            thinking=denoise_defaults["thinking"],
        )

        assert text
        assert canvas_state.converged is True
        assert canvas_trace.injected_cache_provenance is None
