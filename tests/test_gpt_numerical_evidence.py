import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync" / "evidence.py"
TRAINING_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "training.py"
COMPARE_SCRIPT = REPO_ROOT / "tools" / "validation" / "compare_numerical_evidence.py"


def load_evidence():
    assert MODULE_PATH.exists(), "GPT numerical evidence module is missing"
    name = "megatron.grad_sync.evidence"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def descriptor(
    logical_param_id="weight",
    full_shape=(8, 4),
    local_shape=(4, 4),
    tensor_parallel=True,
    partition_dim=0,
    partition_stride=1,
    tp_degree=2,
    tp_rank=0,
):
    return SimpleNamespace(
        logical_param_id=logical_param_id,
        full_shape=full_shape,
        local_shape=local_shape,
        tensor_parallel=tensor_parallel,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        local_tp_degree=tp_degree,
        local_tp_rank=tp_rank,
    )


@dataclass(frozen=True)
class SerializableDescriptor:
    logical_param_id: str
    full_shape: tuple
    local_shape: tuple
    tensor_parallel: bool
    partition_dim: int
    partition_stride: int
    local_tp_degree: int
    local_tp_rank: int
    local_dp_degree: int = 1
    local_dp_rank: int = 0
    pipeline_id: int = 0
    stage_id: int = 0
    shared_weight_id: str = None
    shared_weight_root: bool = False


class GptNumericalEvidenceTest(unittest.TestCase):
    def test_logical_initializer_is_stable_and_parameter_specific(self):
        evidence = load_evidence()

        first = evidence.deterministic_logical_tensor(
            "gpt.layers.0.attention.query_key_value.weight",
            (12, 4),
            torch.float32,
        )
        repeated = evidence.deterministic_logical_tensor(
            "gpt.layers.0.attention.query_key_value.weight",
            (12, 4),
            torch.float32,
        )
        other = evidence.deterministic_logical_tensor(
            "gpt.layers.0.attention.dense.weight",
            (12, 4),
            torch.float32,
        )

        self.assertTrue(torch.equal(first, repeated))
        self.assertFalse(torch.equal(first, other))
        self.assertTrue(torch.isfinite(first).all())

    def test_extracts_dim_zero_and_dim_one_tensor_parallel_shards(self):
        evidence = load_evidence()
        full = torch.arange(32, dtype=torch.float32).view(8, 4)

        dim_zero = [
            evidence.local_tensor_from_logical(
                full,
                descriptor(tp_rank=rank),
            )
            for rank in range(2)
        ]
        dim_one = [
            evidence.local_tensor_from_logical(
                full,
                descriptor(
                    full_shape=(8, 4),
                    local_shape=(8, 2),
                    partition_dim=1,
                    tp_rank=rank,
                ),
            )
            for rank in range(2)
        ]

        self.assertTrue(torch.equal(torch.cat(dim_zero, dim=0), full))
        self.assertTrue(torch.equal(torch.cat(dim_one, dim=1), full))

    def test_extracts_strided_partition_and_replicated_tensor(self):
        evidence = load_evidence()
        full = torch.arange(16, dtype=torch.float32).view(8, 2)
        rank_zero = evidence.local_tensor_from_logical(
            full,
            descriptor(
                full_shape=(8, 2),
                local_shape=(4, 2),
                partition_stride=2,
                tp_rank=0,
            ),
        )
        rank_one = evidence.local_tensor_from_logical(
            full,
            descriptor(
                full_shape=(8, 2),
                local_shape=(4, 2),
                partition_stride=2,
                tp_rank=1,
            ),
        )
        replicated = evidence.local_tensor_from_logical(
            full,
            descriptor(
                full_shape=(8, 2),
                local_shape=(8, 2),
                tensor_parallel=False,
                partition_dim=-1,
                tp_degree=1,
            ),
        )

        self.assertTrue(
            torch.equal(rank_zero, torch.cat((full[0:2], full[4:6]), dim=0))
        )
        self.assertTrue(
            torch.equal(rank_one, torch.cat((full[2:4], full[6:8]), dim=0))
        )
        self.assertTrue(torch.equal(replicated, full))

    def test_partition_shape_error_names_logical_parameter(self):
        evidence = load_evidence()

        with self.assertRaisesRegex(ValueError, "gpt.bad.weight.*local shape"):
            evidence.local_tensor_from_logical(
                torch.zeros((8, 4)),
                descriptor(
                    logical_param_id="gpt.bad.weight",
                    local_shape=(3, 4),
                ),
            )

    def test_canonical_initializer_updates_runtime_parameter_binding(self):
        evidence = load_evidence()
        item = SerializableDescriptor(
            logical_param_id="gpt.layers.0.attention.dense.weight",
            full_shape=(4, 4),
            local_shape=(4, 2),
            tensor_parallel=True,
            partition_dim=1,
            partition_stride=1,
            local_tp_degree=2,
            local_tp_rank=1,
        )
        parameter = torch.nn.Parameter(torch.zeros((4, 2)))
        binding = SimpleNamespace(
            logical_param_id=item.logical_param_id,
            parameter=parameter,
            main_grad=torch.zeros_like(parameter),
            device=parameter.device,
        )

        checksums = evidence.apply_canonical_initialization(
            (item,), (binding,)
        )

        expected_full = evidence.deterministic_logical_tensor(
            item.logical_param_id, item.full_shape, parameter.dtype
        )
        self.assertTrue(
            torch.equal(
                parameter.detach(),
                evidence.local_tensor_from_logical(expected_full, item),
            )
        )
        self.assertEqual(tuple(checksums), (item.logical_param_id,))
        self.assertEqual(len(checksums[item.logical_param_id]), 64)

    def test_snapshot_round_trip_records_cpu_tensors_and_training_state(self):
        evidence = load_evidence()
        item = SerializableDescriptor(
            logical_param_id="gpt.final_layernorm.weight",
            full_shape=(4,),
            local_shape=(4,),
            tensor_parallel=False,
            partition_dim=-1,
            partition_stride=1,
            local_tp_degree=1,
            local_tp_rank=0,
        )
        parameter = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
        binding = SimpleNamespace(
            logical_param_id=item.logical_param_id,
            parameter=parameter,
            main_grad=torch.full_like(parameter, 2.0),
            device=parameter.device,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = evidence.save_rank_snapshot(
                directory,
                phase="post_step",
                iteration=2,
                global_rank=3,
                descriptors=(item,),
                runtime_bindings=(binding,),
                optimizer_state_by_logical={
                    item.logical_param_id: {
                        "exp_avg": torch.ones_like(parameter),
                        "step": torch.tensor(2.0),
                    }
                },
                optimizer_hyperparameters_by_logical={
                    item.logical_param_id: {
                        "beta1": 0.9,
                        "beta2": 0.999,
                        "epsilon": 1.0e-8,
                        "weight_decay": 0.0,
                    }
                },
                clock={
                    "learning_rate": 1.5e-4,
                    "global_data_cursor": 48,
                    "successful_optimizer_steps": 2,
                    "loss_scale": 1024.0,
                },
                metrics={
                    "grad_norm": 3.0,
                    "clip_coefficient": 1.0 / 3.0,
                    "overflow": False,
                    "update_successful": True,
                },
                run_metadata={
                    "model_impl": "gpt",
                    "config_hash": "config",
                    "topology_fingerprint": "topology",
                    "manifest_hash": "manifest",
                    "grad_sync_mode": "block",
                },
            )
            snapshot = evidence.load_rank_snapshot(path)

        self.assertEqual(snapshot["version"], 1)
        self.assertEqual(snapshot["phase"], "post_step")
        self.assertEqual(snapshot["iteration"], 2)
        self.assertEqual(snapshot["global_rank"], 3)
        self.assertEqual(snapshot["clock"]["global_data_cursor"], 48)
        self.assertEqual(snapshot["run_metadata"]["model_impl"], "gpt")
        self.assertEqual(snapshot["records"][0]["descriptor"]["stage_id"], 0)
        self.assertEqual(snapshot["records"][0]["parameter"].device.type, "cpu")
        self.assertEqual(snapshot["records"][0]["main_grad"].device.type, "cpu")
        self.assertEqual(
            snapshot["records"][0]["optimizer_state"]["step"].item(), 2.0
        )
        self.assertEqual(
            snapshot["records"][0]["optimizer_hyperparameters"]["beta1"],
            0.9,
        )

    def test_snapshot_rejects_non_finite_tensor_with_logical_context(self):
        evidence = load_evidence()
        item = SerializableDescriptor(
            logical_param_id="gpt.bad.weight",
            full_shape=(1,),
            local_shape=(1,),
            tensor_parallel=False,
            partition_dim=-1,
            partition_stride=1,
            local_tp_degree=1,
            local_tp_rank=0,
        )
        parameter = torch.nn.Parameter(torch.tensor([float("nan")]))
        binding = SimpleNamespace(
            logical_param_id=item.logical_param_id,
            parameter=parameter,
            main_grad=torch.zeros_like(parameter),
            device=parameter.device,
        )

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "gpt.bad.weight.*non-finite"):
                evidence.save_rank_snapshot(
                    directory,
                    phase="init",
                    iteration=0,
                    global_rank=0,
                    descriptors=(item,),
                    runtime_bindings=(binding,),
                )

    def test_pre_sync_snapshot_preserves_non_finite_gradient_for_overflow(self):
        evidence = load_evidence()
        item = SerializableDescriptor(
            logical_param_id="gpt.overflow.weight",
            full_shape=(1,),
            local_shape=(1,),
            tensor_parallel=False,
            partition_dim=-1,
            partition_stride=1,
            local_tp_degree=1,
            local_tp_rank=0,
        )
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        binding = SimpleNamespace(
            logical_param_id=item.logical_param_id,
            parameter=parameter,
            main_grad=torch.tensor([float("inf")]),
            device=parameter.device,
        )

        snapshot = evidence.build_rank_snapshot(
            phase="pre_sync",
            iteration=1,
            global_rank=0,
            descriptors=(item,),
            runtime_bindings=(binding,),
        )

        self.assertTrue(torch.isinf(snapshot["records"][0]["main_grad"]).all())

    def test_training_wires_evidence_hooks_in_numerical_order(self):
        training = TRAINING_PATH.read_text(encoding="utf-8")

        canonical_init = training.index("apply_canonical_initialization(")
        tied_parameter_sync = training.index(
            "synchronize_shared_weight_parameters(model)", canonical_init
        )
        pre_sync = training.index('_record_gradient_evidence("pre_sync"')
        manager_execute = training.index(
            "args.gradient_sync_manager.execute(", pre_sync
        )
        tied_gradient_sync = training.index(
            "synchronize_shared_weights_grads(model)", manager_execute
        )
        shared_root_replication = training.index(
            "synchronize_shared_root_gradients(", tied_gradient_sync
        )
        post_sync = training.index(
            '_record_gradient_evidence("post_sync"', shared_root_replication
        )
        optimizer_step = training.index("optimizer.step()", post_sync)
        cursor_update = training.index(
            "args.global_data_cursor += args.total_gbs", optimizer_step
        )
        post_step = training.index(
            '_record_gradient_evidence("post_step"', cursor_update
        )

        self.assertLess(canonical_init, tied_parameter_sync)
        self.assertLess(pre_sync, manager_execute)
        self.assertLess(tied_gradient_sync, post_sync)
        self.assertLess(tied_gradient_sync, shared_root_replication)
        self.assertLess(shared_root_replication, post_sync)
        self.assertLess(post_sync, optimizer_step)
        self.assertLess(cursor_update, post_step)

    def test_reconstructs_equal_parameters_across_mixed_tp_layouts(self):
        evidence = load_evidence()
        full = torch.arange(8, dtype=torch.float32).view(4, 2)
        records = []
        for item, value in (
            (
                SerializableDescriptor(
                    "weight", (4, 2), (4, 2), False, -1, 1, 1, 0,
                    pipeline_id=0,
                ),
                full,
            ),
            (
                SerializableDescriptor(
                    "weight", (4, 2), (2, 2), True, 0, 1, 2, 0,
                    pipeline_id=1,
                ),
                full[:2],
            ),
            (
                SerializableDescriptor(
                    "weight", (4, 2), (2, 2), True, 0, 1, 2, 1,
                    pipeline_id=1,
                ),
                full[2:],
            ),
        ):
            records.append(
                {
                    "descriptor": vars(item),
                    "parameter": value.clone(),
                    "main_grad": torch.zeros_like(value),
                    "optimizer_state": {},
                }
            )

        reconstructed = evidence.reconstruct_equal_logical_values(
            records, "parameter"
        )

        torch.testing.assert_close(reconstructed["weight"], full)

    def test_full_gradient_reference_sums_layouts_and_averages_tp_duplicates(self):
        evidence = load_evidence()
        records = []
        cases = (
            (SerializableDescriptor("sharded", (4,), (4,), False, -1, 1, 1, 0, pipeline_id=0), torch.tensor([1.0, 2.0, 3.0, 4.0])),
            (SerializableDescriptor("sharded", (4,), (2,), True, 0, 1, 2, 0, pipeline_id=1), torch.tensor([10.0, 20.0])),
            (SerializableDescriptor("sharded", (4,), (2,), True, 0, 1, 2, 1, pipeline_id=1), torch.tensor([30.0, 40.0])),
            (SerializableDescriptor("replicated", (2,), (2,), False, -1, 1, 1, 0, pipeline_id=0), torch.tensor([2.0, 4.0])),
            (SerializableDescriptor("replicated", (2,), (2,), False, -1, 1, 2, 0, pipeline_id=1), torch.tensor([6.0, 8.0])),
            (SerializableDescriptor("replicated", (2,), (2,), False, -1, 1, 2, 1, pipeline_id=1), torch.tensor([6.0, 8.0])),
            (SerializableDescriptor("embedding", (2,), (2,), False, -1, 1, 1, 0, shared_weight_id="tied", shared_weight_root=True), torch.tensor([1.0, 2.0])),
            (SerializableDescriptor("output_head", (2,), (2,), False, -1, 1, 1, 0, shared_weight_id="tied", shared_weight_root=False), torch.tensor([3.0, 4.0])),
        )
        for item, gradient in cases:
            records.append(
                {
                    "descriptor": vars(item),
                    "parameter": torch.zeros_like(gradient),
                    "main_grad": gradient,
                    "optimizer_state": {},
                }
            )

        reference = evidence.full_gradient_reference(
            records, normalization_scale=0.5
        )

        torch.testing.assert_close(
            reference["sharded"], torch.tensor([5.5, 11.0, 16.5, 22.0])
        )
        torch.testing.assert_close(
            reference["replicated"], torch.tensor([4.0, 6.0])
        )
        torch.testing.assert_close(
            reference["embedding"], torch.tensor([2.0, 3.0])
        )
        torch.testing.assert_close(
            reference["output_head"], torch.tensor([2.0, 3.0])
        )

    def test_norm_clip_and_adamw_reference_match_pytorch(self):
        evidence = load_evidence()
        parameter = torch.tensor([1.0, -2.0], dtype=torch.float64)
        gradient = torch.tensor([3.0, 4.0], dtype=torch.float64)
        gradients, norm, coefficient = evidence.clip_logical_gradients(
            {"weight": gradient}, max_norm=1.0
        )
        expected_parameter = torch.nn.Parameter(parameter.clone())
        optimizer = torch.optim.AdamW(
            [expected_parameter],
            lr=0.01,
            betas=(0.9, 0.95),
            eps=1.0e-8,
            weight_decay=0.1,
        )
        expected_parameter.grad = gradients["weight"].clone()
        optimizer.step()

        actual_parameter, exp_avg, exp_avg_sq = evidence.adamw_reference_step(
            parameter,
            gradients["weight"],
            torch.zeros_like(parameter),
            torch.zeros_like(parameter),
            step=1,
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.95,
            epsilon=1.0e-8,
            weight_decay=0.1,
        )

        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(coefficient, 1.0 / (5.0 + 1.0e-6))
        torch.testing.assert_close(actual_parameter, expected_parameter.detach())
        torch.testing.assert_close(exp_avg, gradients["weight"] * 0.1)
        torch.testing.assert_close(exp_avg_sq, gradients["weight"].square() * 0.05)

    def test_tied_weight_norm_counts_only_the_semantic_root(self):
        evidence = load_evidence()
        gradients = {
            "embedding": torch.tensor([3.0, 4.0]),
            "output_head": torch.tensor([3.0, 4.0]),
        }

        clipped, norm, coefficient = evidence.clip_logical_gradients(
            gradients, max_norm=1.0, norm_owner_ids={"embedding"}
        )

        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(coefficient, 1.0 / (5.0 + 1.0e-6))
        torch.testing.assert_close(clipped["embedding"], clipped["output_head"])

    def test_comparison_reports_failing_parameter_and_index(self):
        evidence = load_evidence()

        report = evidence.compare_logical_tensors(
            {"good": torch.zeros(2), "bad": torch.zeros(3)},
            {"good": torch.zeros(2), "bad": torch.tensor([0.0, 0.0, 1.0])},
            rtol=1.0e-5,
            atol=1.0e-6,
        )

        self.assertFalse(report["passed"])
        failure = next(item for item in report["parameters"] if item["logical_param_id"] == "bad")
        self.assertFalse(failure["passed"])
        self.assertEqual(failure["max_abs_index"], [2])
        self.assertEqual(
            failure["max_abs_region"], {"shape": [3], "coordinate": [2]}
        )
        self.assertEqual(failure["max_abs_error"], 1.0)

    def test_overflow_tensor_equality_accepts_matching_non_finite_layout(self):
        evidence = load_evidence()
        descriptor = vars(SerializableDescriptor(
            "weight", (3,), (3,), False, -1, 1, 2, 0
        ))
        records = []
        for dp_rank in (0, 1):
            replica_descriptor = dict(descriptor)
            replica_descriptor["local_dp_rank"] = dp_rank
            records.append({
                "descriptor": replica_descriptor,
                "main_grad": torch.tensor([float("nan"), float("inf"), -float("inf")]),
            })

        reconstructed = evidence.reconstruct_equal_logical_values(
            records, "main_grad"
        )
        report = evidence.compare_logical_tensors(
            reconstructed,
            {"weight": torch.tensor([float("nan"), float("inf"), -float("inf")])},
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

        self.assertTrue(report["passed"])

    def test_reconstructs_optimizer_moments_across_mixed_tp_layouts(self):
        evidence = load_evidence()
        full_moment = torch.arange(8, dtype=torch.float32).view(4, 2)
        records = []
        for item, moment in (
            (
                SerializableDescriptor(
                    "weight", (4, 2), (4, 2), False, -1, 1, 1, 0,
                    pipeline_id=0,
                ),
                full_moment,
            ),
            (
                SerializableDescriptor(
                    "weight", (4, 2), (2, 2), True, 0, 1, 2, 0,
                    pipeline_id=1,
                ),
                full_moment[:2],
            ),
            (
                SerializableDescriptor(
                    "weight", (4, 2), (2, 2), True, 0, 1, 2, 1,
                    pipeline_id=1,
                ),
                full_moment[2:],
            ),
        ):
            records.append(
                {
                    "descriptor": vars(item),
                    "parameter": torch.zeros_like(moment),
                    "main_grad": torch.zeros_like(moment),
                    "optimizer_state": {
                        "exp_avg": moment.clone(),
                        "exp_avg_sq": moment.square(),
                        "step": torch.tensor(2.0),
                    },
                }
            )

        states = evidence.reconstruct_equal_optimizer_states(records)

        torch.testing.assert_close(states["weight"]["exp_avg"], full_moment)
        torch.testing.assert_close(
            states["weight"]["exp_avg_sq"], full_moment.square()
        )
        self.assertEqual(states["weight"]["step"].item(), 2.0)

    def test_cli_compares_runs_and_full_gradient_reference(self):
        evidence = load_evidence()
        item = SerializableDescriptor(
            logical_param_id="weight",
            full_shape=(2,),
            local_shape=(2,),
            tensor_parallel=False,
            partition_dim=-1,
            partition_stride=1,
            local_tp_degree=1,
            local_tp_rank=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "actual"
            expected = root / "expected"
            report_path = root / "report.json"
            for run_dir in (actual, expected):
                parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
                binding = SimpleNamespace(
                    logical_param_id="weight",
                    parameter=parameter,
                    main_grad=torch.zeros(2),
                    device=parameter.device,
                )
                metadata = {
                    "model_impl": "gpt",
                    "config_hash": "config",
                    "topology_fingerprint": "topology",
                    "manifest_hash": "manifest",
                    "grad_sync_mode": "tensor",
                    "clip_grad": 1.0,
                    "nominal_global_loss_scale": 1.0,
                    "world_size": 1,
                    "total_gbs": 2,
                    "max_lr": 0.01,
                    "min_lr": 0.01,
                    "lr_decay_style": "constant",
                    "lr_decay_steps": 100,
                    "lr_warmup_steps": 0,
                    "precision": "fp32",
                    "loss_scale_mode": "fixed",
                    "initial_loss_scale": 1.0,
                    "min_loss_scale": 1.0,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                }
                hyperparameters = {
                    "weight": {
                        "beta1": 0.9,
                        "beta2": 0.95,
                        "epsilon": 1.0e-8,
                        "weight_decay": 0.1,
                    }
                }
                common = dict(
                    output_dir=run_dir,
                    global_rank=0,
                    descriptors=(item,),
                    runtime_bindings=(binding,),
                    optimizer_hyperparameters_by_logical=hyperparameters,
                    run_metadata=metadata,
                )
                evidence.save_rank_snapshot(
                    phase="init",
                    iteration=0,
                    clock={
                        "learning_rate": 0.0,
                        "global_data_cursor": 0,
                        "successful_optimizer_steps": 0,
                        "loss_scale": 1.0,
                    },
                    **common,
                )
                exp_avg = torch.zeros(2)
                exp_avg_sq = torch.zeros(2)
                for iteration in (1, 2):
                    binding.main_grad = torch.tensor([3.0, 4.0])
                    pre_clock = {
                        "learning_rate": 0.01,
                        "global_data_cursor": 2 * (iteration - 1),
                        "successful_optimizer_steps": iteration - 1,
                        "loss_scale": 1.0,
                    }
                    evidence.save_rank_snapshot(
                        phase="pre_sync",
                        iteration=iteration,
                        clock=pre_clock,
                        metrics={
                            "local_loss_weight": 1.0,
                            "global_loss_weight": 1.0,
                            "normalization_scale": 1.0,
                        },
                        **common,
                    )
                    evidence.save_rank_snapshot(
                        phase="post_sync",
                        iteration=iteration,
                        clock=pre_clock,
                        **common,
                    )
                    clipped, norm, coefficient = evidence.clip_logical_gradients(
                        {"weight": binding.main_grad}, 1.0
                    )
                    next_parameter, exp_avg, exp_avg_sq = (
                        evidence.adamw_reference_step(
                            parameter.detach(),
                            clipped["weight"],
                            exp_avg,
                            exp_avg_sq,
                            step=iteration,
                            learning_rate=0.01,
                            beta1=0.9,
                            beta2=0.95,
                            epsilon=1.0e-8,
                            weight_decay=0.1,
                        )
                    )
                    with torch.no_grad():
                        parameter.copy_(next_parameter)
                    evidence.save_rank_snapshot(
                        phase="post_step",
                        iteration=iteration,
                        optimizer_state_by_logical={
                            "weight": {
                                "exp_avg": exp_avg,
                                "exp_avg_sq": exp_avg_sq,
                                "step": torch.tensor(float(iteration)),
                            }
                        },
                        clock={
                            "learning_rate": 0.01,
                            "global_data_cursor": 2 * iteration,
                            "successful_optimizer_steps": iteration,
                            "loss_scale": 1.0,
                        },
                        metrics={
                            "grad_norm": norm,
                            "clip_coefficient": coefficient,
                            "overflow": False,
                            "update_successful": True,
                        },
                        **common,
                    )

            result = subprocess.run(
                [
                    sys.executable,
                    str(COMPARE_SCRIPT),
                    "--actual",
                    str(actual),
                    "--expected",
                    str(expected),
                    "--full-gradient",
                    "--output",
                    str(report_path),
                ],
                cwd=REPO_ROOT,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=20,
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertTrue(report["passed"])
        self.assertTrue(report["full_gradient"]["passed"])
        self.assertLess(report["full_gradient"]["iterations"][0]["clip_coefficient"], 1.0)
        self.assertEqual(report["actual_metadata"]["manifest_hash"], "manifest")
        self.assertEqual(report["comparisons"][0]["iteration"], 0)
        post_step = next(
            item for item in report["comparisons"] if item["phase"] == "post_step"
        )
        self.assertTrue(post_step["optimizer_state"]["passed"])

    def test_cli_rejects_incomplete_evidence(self):
        module_name = "compare_gpt_evidence_test"
        spec = importlib.util.spec_from_file_location(module_name, COMPARE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with self.assertRaisesRegex(ValueError, "at least two contiguous iterations"):
            module.validate_run({(0, "init"): [{"global_rank": 0}]})

        complete_keys = {(0, "init")}
        complete_keys.update(
            (iteration, phase)
            for iteration in (1, 2)
            for phase in ("pre_sync", "post_sync", "post_step")
        )
        incomplete_ranks = {
            key: [{
                "global_rank": 0,
                "run_metadata": {"world_size": 2},
            }]
            for key in complete_keys
        }
        with self.assertRaisesRegex(ValueError, "declared world size"):
            module.validate_run(incomplete_ranks)

    def test_overflow_oracle_requires_unchanged_state_and_cursor_only_clock(self):
        module_name = "compare_gpt_evidence_overflow_test"
        spec = importlib.util.spec_from_file_location(module_name, COMPARE_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        descriptor_value = vars(SerializableDescriptor(
            "weight", (1,), (1,), False, -1, 1, 1, 0
        ))

        def snapshot(iteration, phase, cursor, overflow=False, loss_scale=1024.0):
            return {
                "global_rank": 0,
                "run_metadata": {
                    "world_size": 1,
                    "total_gbs": 2,
                    "loss_scale_mode": "dynamic",
                    "initial_loss_scale": 1024.0,
                    "min_loss_scale": 1.0,
                    "loss_scale_window": 1000,
                    "hysteresis": 2,
                },
                "clock": {
                    "learning_rate": 0.01,
                    "global_data_cursor": cursor,
                    "successful_optimizer_steps": 0,
                    "loss_scale": loss_scale,
                },
                "metrics": {
                    "overflow": overflow,
                    "update_successful": not overflow,
                },
                "records": [{
                    "descriptor": descriptor_value,
                    "parameter": torch.tensor([1.0]),
                    "main_grad": torch.tensor([float("nan")]),
                    "optimizer_parameter": torch.tensor([1.0]),
                    "optimizer_state": {},
                    "optimizer_hyperparameters": {},
                }],
            }

        run = {(0, "init"): [snapshot(0, "init", 0)]}
        for phase in ("pre_sync", "post_sync", "post_step"):
            run[(1, phase)] = [snapshot(1, phase, 2 if phase == "post_step" else 0, phase == "post_step")]
            run[(2, phase)] = [snapshot(
                2,
                phase,
                4 if phase == "post_step" else 2,
                phase == "post_step",
                512.0 if phase == "post_step" else 1024.0,
            )]

        report = module.compare_overflow_semantics(run, 1.0e-5, 1.0e-6)

        self.assertTrue(report["passed"])
        self.assertEqual(
            [item["iteration"] for item in report["iterations"]], [1, 2]
        )
        comparisons = module.compare_runs(run, run, 1.0e-5, 1.0e-6)
        self.assertTrue(all(item["passed"] for item in comparisons))

        for snapshots in run.values():
            for snapshot_item in snapshots:
                snapshot_item["run_metadata"]["topology_fingerprint"] = "distributed"
        different_topology = copy.deepcopy(run)
        for snapshots in different_topology.values():
            for snapshot_item in snapshots:
                snapshot_item["run_metadata"]["topology_fingerprint"] = "single"
        different_topology[(1, "post_sync")][0]["records"][0]["main_grad"] = (
            torch.tensor([float("nan")])
        )
        run[(1, "post_sync")][0]["records"][0]["main_grad"] = torch.tensor(
            [float("nan")]
        )
        different_topology[(2, "post_sync")][0]["records"][0]["main_grad"] = (
            torch.tensor([3.0005])
        )
        run[(2, "post_sync")][0]["records"][0]["main_grad"] = torch.tensor([3.0])

        cross_topology = module.compare_runs(
            run, different_topology, 1.0e-5, 1.0e-6
        )
        self.assertTrue(all(item["passed"] for item in cross_topology))
        overflow_gradients = [
            item for item in cross_topology if item["phase"] == "post_sync"
        ]
        self.assertTrue(
            all(
                item["tensors"]["comparison_mode"] == "unscaled_values"
                for item in overflow_gradients
            )
        )

        material_gradient_error = copy.deepcopy(different_topology)
        material_gradient_error[(2, "post_sync")][0]["records"][0][
            "main_grad"
        ] = torch.tensor([7.0])
        material_comparisons = module.compare_runs(
            run, material_gradient_error, 1.0e-5, 1.0e-6
        )
        self.assertFalse(
            next(
                item
                for item in material_comparisons
                if item["iteration"] == 2 and item["phase"] == "post_sync"
            )["passed"]
        )

        same_topology = copy.deepcopy(different_topology)
        for snapshots in same_topology.values():
            for snapshot_item in snapshots:
                snapshot_item["run_metadata"]["topology_fingerprint"] = None
        for snapshots in run.values():
            for snapshot_item in snapshots:
                snapshot_item["run_metadata"]["topology_fingerprint"] = None
        same_topology_comparisons = module.compare_runs(
            run, same_topology, 1.0e-5, 1.0e-6
        )
        self.assertFalse(
            next(
                item
                for item in same_topology_comparisons
                if item["iteration"] == 2 and item["phase"] == "post_sync"
            )["passed"]
        )

        changed_model = copy.deepcopy(run)
        changed_model[(1, "post_step")][0]["records"][0]["parameter"].add_(1.0)
        changed_report = module.compare_overflow_semantics(
            changed_model, 1.0e-5, 1.0e-6
        )
        self.assertFalse(changed_report["passed"])


if __name__ == "__main__":
    unittest.main()
