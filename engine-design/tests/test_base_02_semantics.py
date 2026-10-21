"""BASE-02 source subtests; the group runner separately enforces BLOCKED gate inputs."""
from __future__ import annotations

import ast
import importlib.metadata
import math
from pathlib import Path
import subprocess
import types
import unittest
from unittest import mock

import torch

from i0_testlib import (
    ENGINE_DESIGN,
    FIXTURES,
    REPO_ROOT,
    checked_distribution_bytes,
    decoder_visible_positions,
    entropy_bound_accept,
    inverse_cdf,
    load_authenticated_cdg_package,
    read_authenticated_cdg_source,
    read_evidence_artifact,
    read_json,
    retained_prefix_positions,
    sha256,
    stable_softmax,
    top_k_contract,
    verify_base02_gate,
    verify_evidence_scope,
)


def class_source(source: str, class_name: str) -> str:
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == class_name)
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError(f"no source segment for {class_name}")
    return segment


def git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return __import__("hashlib").sha1(header + data).hexdigest()


class Base02SemanticBindings(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lock = read_json(ENGINE_DESIGN / "reference.lock.json")
        cls.pack, cls.pack_loader = cls.enterClassContext(
            load_authenticated_cdg_package(cls.lock, verify_oracle_bindings=True)
        )
        cls.semantic = read_json(FIXTURES / "semantics.json")
        cls.defects = read_json(FIXTURES / "baseline-defects.json")

    def test_unequal_phase_scalar_expectation_is_bound_to_hf_classes(self):
        source = read_evidence_artifact(self.lock, "package.transformers.modeling_diffusion_gemma.py").decode()
        for name in ("DiffusionGemmaEncoderTextLayer", "DiffusionGemmaDecoderTextLayer"):
            segment = class_source(source, name)
            self.assertIn('self.register_buffer("layer_scalar", torch.ones(1))', segment)
            self.assertIn("hidden_states *= self.layer_scalar", segment)
        dg27 = read_evidence_artifact(self.lock, "dg27.model").decode()
        self.assertIn('create_tensor(tn(LLM_TENSOR_ENC_LAYER_OUT_SCALE, "weight", i)', dg27)
        self.assertIn("ggml_mul(ctx0, cur, model.layers[il].out_scale)", dg27)
        fixture = self.semantic["phase_scalars"]
        self.assertNotEqual(fixture["encoder_scalar"], fixture["decoder_scalar"])
        self.assertEqual(
            [value * fixture["encoder_scalar"] for value in fixture["input"]], fixture["encoder_expected"]
        )
        self.assertEqual(
            [value * fixture["decoder_scalar"] for value in fixture["input"]], fixture["decoder_expected"]
        )

    def test_mask_and_rollback_expectations_are_bound_to_hf_and_donor_source(self):
        hf = read_evidence_artifact(self.lock, "package.transformers.modeling_diffusion_gemma.py").decode()
        decoder = class_source(hf, "DiffusionGemmaDecoderModel")
        self.assertIn("past_key_values.get_mask_sizes(q_length, layer_idx)", decoder)
        self.assertIn("kv_length += additional_kv_length", decoder)
        self.assertIn("mask_function=bidirectional_mask_function", decoder)
        attention = class_source(hf, "DiffusionGemmaDecoderTextAttention")
        attention_tree = ast.parse(attention)
        update_calls = [
            node
            for node in ast.walk(attention_tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
        ]
        self.assertEqual(update_calls, [])
        self.assertIn("the tensors are concatenated and returned", attention)
        dg23 = read_evidence_artifact(self.lock, "dg23.model").decode()
        self.assertIn("canvas_prompt_lo = P - (int64_t) hparams.n_swa + 1", dg23)
        self.assertIn("allow = swa ? (k >= canvas_prompt_lo) : true", dg23)
        self.assertIn("allow = true;                     // bidirectional over the canvas", dg23)
        dg27_cli = read_evidence_artifact(self.lock, "dg27.cli").decode()
        self.assertIn("llama_memory_seq_rm(mem, 0, n_past, -1)", dg27_cli)
        self.assertIn("roll back the canvas K/V", dg27_cli)
        fixture = self.semantic["mask_cache"]
        for case in fixture["eviction_examples"]:
            self.assertEqual(
                retained_prefix_positions(case["prefix_length"], fixture["small_window"]),
                case["retained_prefix"],
            )
        for prefix_length in fixture["prefix_lengths"]:
            for canvas_length in fixture["canvas_lengths"]:
                visible = decoder_visible_positions(
                    prefix_length, canvas_length, fixture["small_window"], sliding=True
                )
                self.assertEqual(visible[-canvas_length:], list(range(prefix_length, prefix_length + canvas_length)))
        prefix = tuple(range(10))
        digest = hash(prefix)
        scratch = list(prefix) + [100, 101, 102]
        del scratch[len(prefix):]
        self.assertEqual(tuple(scratch), prefix)
        self.assertEqual(hash(prefix), digest)

    def test_trace_stage_order_and_top_k_semantics_are_bound_to_current_cdg(self):
        composite = read_authenticated_cdg_source(self.lock, "dgemma/composite.py").decode()
        capture = read_authenticated_cdg_source(self.lock, "dgemma/capture.py").decode()
        constraints = read_authenticated_cdg_source(self.lock, "dgemma/constraints_hook.py").decode()
        loop = read_authenticated_cdg_source(self.lock, "dgemma/loop.py").decode()
        ordered = [
            composite.index("self.capture(pipe"),
            composite.index("self._cancellation(pipe"),
            composite.index("for participant in (*self.beta_rebuild, *self.pin)"),
            composite.index("if self.walker is not None"),
        ]
        self.assertEqual(ordered, sorted(ordered))
        self.assertIn("top_k_weights = torch.softmax(top_k_values, dim=-1)", capture)
        self.assertIn("distribution = torch.softmax(per_position_logits, dim=-1)", capture)
        self.assertIn('logits[..., pin.position, :] = float("-inf")', constraints)
        self.assertIn("logits[..., pin.position, pin.token_id] = 0.0", constraints)
        self.assertIn("with install_logit_shaping_hook(dgemma_model.model, logit_hook):", loop)
        self.assertLess(loop.index("scheduler_output = scheduler.step"), loop.index('callback_kwargs = {"canvas"'))
        fixture = self.semantic["trace"]
        self.assertEqual(fixture["stage_order"][:3], ["model_output", "constraint_output", "sampler_input"])
        for constrained, sampler in zip(fixture["constraint_output"], fixture["sampler_input"]):
            self.assertEqual(
                [None if value is None else value / fixture["temperature"] for value in constrained],
                sampler,
            )
        ids, conditional, retained = top_k_contract(fixture["constraint_output"][0], fixture["top_k"])
        self.assertEqual(ids, [0, 1])
        self.assertTrue(math.isclose(sum(conditional), 1.0, abs_tol=1e-12))
        self.assertLess(retained, 1.0)
        full = stable_softmax(fixture["constraint_output"][0])
        self.assertTrue(math.isclose(retained, full[0] + full[1], abs_tol=1e-12))
        self.assertEqual(top_k_contract(fixture["constraint_output"][0], 1)[1], [1.0])
        self.assertTrue(math.isclose(top_k_contract(fixture["constraint_output"][0], 3)[2], 1.0))
        for invalid_k in (0, 4):
            with self.assertRaises(ValueError):
                top_k_contract(fixture["constraint_output"][0], invalid_k)

    def test_composite_executes_capture_cancel_writer_walker_order(self):
        composite_module = self.pack_loader.import_module("dgemma.composite")
        DiffusionCancelled = composite_module.DiffusionCancelled
        StepEndComposite = composite_module.StepEndComposite

        events = []

        class Participant:
            def __init__(self, name): self.name = name
            def __call__(self, pipe, global_step, step_idx, kwargs):
                events.append(self.name)
                return {"canvas": kwargs["canvas"]}

        cancelled = StepEndComposite(
            capture=lambda *args: events.append("capture") or {},
            should_cancel=lambda: True,
            beta_rebuild=(Participant("beta_rebuild"),),
            pin=(Participant("pin"),),
            walker=Participant("walker"),
        )
        with self.assertRaises(DiffusionCancelled):
            cancelled(None, 0, 0, {"canvas": torch.tensor([1])})
        self.assertEqual(events, ["capture"])
        events.clear()
        complete = StepEndComposite(
            capture=lambda *args: events.append("capture") or {},
            should_cancel=lambda: False,
            beta_rebuild=(Participant("beta_rebuild"),),
            pin=(Participant("pin"),),
            walker=Participant("walker"),
        )
        complete(None, 0, 0, {"canvas": torch.tensor([1])})
        self.assertEqual(events, ["capture", "beta_rebuild", "pin", "walker"])

    def test_entropy_temperature_rng_and_self_conditioning_are_source_bound(self):
        scheduler_source = read_evidence_artifact(
            self.lock, "package.diffusers.scheduling_entropy_bound.py"
        ).decode()
        for clause in (
            "fraction = (self.num_inference_steps - int(timestep)) / self.num_inference_steps",
            "model_output = model_output / temperature",
            "cumulative_entropy - sorted_token_entropy <= entropy_bound",
            "random_tokens = torch.randint(",
            "pred_logits=model_output",
        ):
            self.assertIn(clause, scheduler_source)
        modeling = read_evidence_artifact(
            self.lock, "package.transformers.modeling_diffusion_gemma.py"
        ).decode()
        self.assertIn("self_conditioning_logits.softmax(dim=-1, dtype=torch.float32)", modeling)
        entropy_fixture = self.semantic["entropy_bound"]
        for case in entropy_fixture["cases"]:
            accepted = entropy_bound_accept(case["entropies"], case["bound"])
            self.assertEqual([i for i, value in enumerate(accepted) if value], case["accepted_positions"])
        decommit = entropy_fixture["decommit_fixture"]
        self.assertEqual(entropy_bound_accept(decommit["step_0_entropies"], decommit["bound"]), [True, False])
        self.assertEqual(entropy_bound_accept(decommit["step_1_entropies"], decommit["bound"]), [False, True])
        for invalid in ([math.nan], [math.inf], [-math.inf]):
            with self.assertRaises(ValueError):
                entropy_bound_accept(invalid, 0.1)
        for schedule in entropy_fixture["temperature_schedules"]:
            steps = schedule["steps"]
            values = [0.4 + (0.8 - 0.4) * ((steps - step) / steps) for step in range(steps)]
            if "temperatures" in schedule:
                self.assertEqual(values, schedule["temperatures"])
            else:
                self.assertEqual((values[0], values[-1]), (schedule["first"], schedule["last"]))
        margin = entropy_fixture["threshold_margin"]
        bounds = [math.nextafter(margin["boundary"], -math.inf), margin["boundary"], math.nextafter(margin["boundary"], math.inf)]
        self.assertEqual(
            [entropy_bound_accept(margin["entropies"], bound)[1] for bound in bounds],
            margin["second_position_expected"],
        )

    def test_exact_pipeline_processor_and_model_classes_execute_lightweight_unbound_paths(self):
        loop_module = self.pack_loader.import_module("dgemma.loop")
        model_module = self.pack_loader.import_module("dgemma.model")
        Pipeline = loop_module.DiffusionGemmaPipeline
        AutoProcessor = model_module.AutoProcessor
        Model = model_module.DiffusionGemmaForBlockDiffusion
        self.assertIs(Pipeline, self.pack_loader.oracle_object("DiffusionGemmaPipeline"))
        self.assertIs(AutoProcessor, self.pack_loader.oracle_object("AutoProcessor"))
        self.assertIs(Model, self.pack_loader.oracle_object("DiffusionGemmaForBlockDiffusion"))

        pipeline = object.__new__(Pipeline)
        pipeline.processor = object()
        self.assertIsNone(Pipeline.__dict__["check_inputs"](
            pipeline, "prompt", None, 1, 1, "seq", None
        ))
        with self.assertRaisesRegex(ValueError, "output_type"):
            Pipeline.__dict__["check_inputs"](pipeline, "prompt", None, 1, 1, "hostile", None)

        # The exact no-network constructor is itself an executable contract: it
        # must reject direct use and direct callers to from_pretrained instead.
        processor = object.__new__(AutoProcessor)
        with self.assertRaisesRegex(OSError, "from_pretrained"):
            AutoProcessor.__dict__["__init__"](processor)

        model = object.__new__(Model)
        torch.nn.Module.__init__(model)

        class TinyBackbone(torch.nn.Module):
            def forward(self, **_kwargs):
                return types.SimpleNamespace(
                    last_hidden_state=torch.tensor([[[0.0, 1.0]]]),
                    hidden_states=("hidden",),
                    attentions=("attention",),
                    past_key_values="cache",
                    encoder_last_hidden_state="encoder",
                )

        model.model = TinyBackbone()
        model.lm_head = torch.nn.Identity()
        model.final_logit_softcapping = 2.0
        output = Model.__dict__["forward"](model, decoder_input_ids=torch.tensor([[1]]))
        expected = torch.tanh(torch.tensor([[[0.0, 0.5]]])) * 2.0
        self.assertTrue(torch.allclose(output.logits, expected))
        self.assertEqual(output.past_key_values, "cache")
        self.pack_loader.assert_oracle_bindings()

    def test_exact_scheduler_static_method_matches_hand_calculation(self):
        loop_module = self.pack_loader.import_module("dgemma.loop")
        EntropyBoundScheduler = loop_module.EntropyBoundScheduler
        self.assertIs(
            EntropyBoundScheduler,
            self.pack_loader.oracle_object("EntropyBoundScheduler"),
        )
        self.pack_loader.assert_oracle_bindings()

        scheduler = EntropyBoundScheduler(entropy_bound=0.15, t_min=0.4, t_max=0.8, num_inference_steps=2)
        logits = torch.tensor([[[4.0, 0.0], [2.0, 0.0], [0.0, 0.0]]], dtype=torch.float32)
        sample = torch.zeros((1, 3), dtype=torch.long)
        output = scheduler.step(logits, 0, sample, generator=torch.Generator().manual_seed(4))
        entropies = torch.distributions.Categorical(logits=logits / 0.8).entropy()[0].tolist()
        self.assertEqual(output.accepted_index[0].tolist(), entropy_bound_accept(entropies, 0.15))
        self.assertTrue(torch.allclose(output.pred_logits, logits / 0.8))

    def test_common_draw_fixture_is_independent_and_ordered(self):
        fixture = self.semantic["common_draws"]
        consumed = list(fixture["initial_canvas_token_draws"])
        for step in fixture["steps"]:
            candidates = [
                inverse_cdf(probabilities, draw)
                for probabilities, draw in zip(step["categorical_probabilities"], step["categorical_uniforms"])
            ]
            self.assertEqual(candidates, step["categorical_expected"])
            self.assertEqual(
                [candidate if accepted else noise for candidate, noise, accepted in zip(candidates, step["uniform_renoise_token_draws"], step["accepted_mask"])],
                step["next_canvas_expected"],
            )
            consumed.extend(step["categorical_uniforms"])
            consumed.extend(step["uniform_renoise_token_draws"])
        self.assertEqual(len(consumed), 20)

    def test_length_and_eos_expectations_are_bound_to_hf_pipeline(self):
        source = read_evidence_artifact(self.lock, "package.diffusers.pipeline_diffusion_gemma.py").decode()
        for clause in (
            "num_canvases = (gen_length + canvas_length - 1) // canvas_length",
            "sequences = cur_input_ids[:, prompt_length:]",
            "seq[: int((seq == eos_token_id).nonzero(as_tuple=True)[0][0]) + 1]",
        ):
            self.assertIn(clause, source)
        for case in self.semantic["lengths"]["eos_absent"]:
            blocks = math.ceil(case["requested"] / 256)
            self.assertEqual(blocks, case["executed_blocks"])
            self.assertEqual(blocks * 256, case["raw_returned_width"])
        self.assertEqual(
            [case["requested"] for case in self.semantic["lengths"]["eos_absent"]],
            [1, 255, 256, 257, 512],
        )
        prefix_end = 19
        first = range(prefix_end, prefix_end + 256)
        second = range(prefix_end + 256, prefix_end + 512)
        self.assertEqual(first.stop, second.start)
        self.assertEqual(second.stop - 1, 530)
        for case in self.semantic["lengths"]["eos_trim_cases"]:
            raw, eos = case["raw"], case["eos_id"]
            self.assertEqual(raw[: raw.index(eos) + 1] if eos in raw else raw, case["decode"])

    def test_distribution_budget_is_bound_to_float32_source_and_checked(self):
        modeling = read_evidence_artifact(
            self.lock, "package.transformers.modeling_diffusion_gemma.py"
        ).decode()
        capture = read_authenticated_cdg_source(self.lock, "dgemma/capture.py").decode()
        self.assertIn("logits = logits.to(torch.float32)", modeling)
        self.assertIn("distribution = torch.softmax(per_position_logits, dim=-1)", capture)
        fixture = self.semantic["distribution_budget"]
        actual = checked_distribution_bytes(fixture["shape"], fixture["element_bytes"])
        self.assertEqual(actual, 268435456)
        self.assertEqual(actual // (1024 * 1024), 256)
        with self.assertRaises(OverflowError):
            checked_distribution_bytes([1 << 62, 8], 4)



if __name__ == "__main__":
    unittest.main(verbosity=2)
