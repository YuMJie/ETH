import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]


class _Timer:
    def start(self):
        pass

    def stop(self):
        pass


class _Model:
    def set_input_tensor(self, input_tensor):
        self.input_tensor = input_tensor


def load_schedules(num_microbatches=2, args=None):
    module_path = REPO_ROOT / "eth" / "runtime" / "megatron" / "schedules.py"
    timer = _Timer()
    megatron = types.ModuleType("megatron")
    megatron.get_args = lambda: args or types.SimpleNamespace()
    megatron.get_num_microbatches = lambda: num_microbatches
    megatron.get_timers = lambda: lambda unused_name: timer
    megatron.mpu = types.SimpleNamespace(is_pipeline_last_stage=lambda: True)
    megatron.p2p_communication = types.SimpleNamespace()
    utils = types.ModuleType("megatron.utils")
    utils.unwrap_model = lambda model, unused_types: model
    model_module = types.ModuleType("megatron.model")
    model_module.DistributedDataParallel = type("LocalDDP", (), {})
    model_module.Float16Module = type("Float16Module", (), {})
    replacements = {
        "megatron": megatron,
        "megatron.utils": utils,
        "megatron.model": model_module,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location("test_schedules", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module, timer


def load_optimizer_module():
    module_path = (
        REPO_ROOT / "eth" / "runtime" / "megatron" / "optimizer" / "optimizer.py"
    )
    apex = types.ModuleType("apex")
    apex_multi = types.ModuleType("apex.multi_tensor_apply")
    apex_multi.multi_tensor_applier = object()
    amp_c = types.ModuleType("amp_C")
    megatron = types.ModuleType("megatron")
    megatron.mpu = types.SimpleNamespace()
    megatron.print_rank_0 = lambda unused_message: None
    clip_grads = types.ModuleType("test_optimizer.clip_grads")
    clip_grads.clip_grad_norm_fp32 = lambda unused_params, unused_limit: 0.0
    clip_grads.count_zeros_fp32 = lambda unused_params: 0
    replacements = {
        "apex": apex,
        "apex.multi_tensor_apply": apex_multi,
        "amp_C": amp_c,
        "megatron": megatron,
        "test_optimizer.clip_grads": clip_grads,
    }
    previous = {name: sys.modules.get(name) for name in replacements}
    sys.modules.update(replacements)
    try:
        spec = importlib.util.spec_from_file_location(
            "test_optimizer.optimizer", module_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    return module


class TrainingLossContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pretrain = (REPO_ROOT / "eth" / "runtime" / "pretrain_gpt.py").read_text(
            encoding="utf-8"
        )
        cls.schedules = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "schedules.py"
        ).read_text(encoding="utf-8")
        cls.training = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "training.py"
        ).read_text(encoding="utf-8")
        cls.arguments = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "arguments.py"
        ).read_text(encoding="utf-8")
        cls.checkpointing = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "checkpointing.py"
        ).read_text(encoding="utf-8")
        cls.optimizer = (
            REPO_ROOT / "eth" / "runtime" / "megatron" / "optimizer" / "optimizer.py"
        ).read_text(encoding="utf-8")

    def test_gpt_and_schedule_use_numerator_weight_without_microbatch_mean(self):
        self.assertIn("gpt_loss_output(", self.pretrain)
        self.assertNotIn("/ loss_mask.sum()", self.pretrain)
        self.assertIn(
            "loss_result.local_loss_sum",
            self.schedules,
        )

    def test_schedule_supports_legacy_and_gpt_forward_contracts(self):
        schedules, unused_timer = load_schedules()
        model = _Model()
        legacy_calls = []
        gpt_calls = []

        def legacy(data_iterator, passed_model, extra_tensors):
            legacy_calls.append((data_iterator, passed_model, extra_tensors))
            return torch.tensor(1.0), None, lambda unused_output: (
                torch.tensor(8.0),
                {"legacy": torch.tensor(3.0)},
            )

        def gpt(data_iterator, passed_model, extra_tensors, current_data_idx):
            gpt_calls.append(current_data_idx)
            output = types.SimpleNamespace(
                local_loss_sum=torch.tensor(12.0),
                local_loss_weight=torch.tensor(3.0),
                nominal_global_loss_scale=6.0,
            )
            return torch.tensor(1.0), None, lambda unused_output: output

        legacy_losses = []
        legacy_backward, _ = schedules.forward_step(
            legacy, "iterator", model, None, legacy_losses, current_data_idx=1
        )
        gpt_losses = []
        gpt_backward, _ = schedules.forward_step(
            gpt, "iterator", model, None, gpt_losses, current_data_idx=1
        )

        self.assertEqual(len(legacy_calls), 1)
        self.assertEqual(legacy_backward.item(), 4.0)
        self.assertEqual(legacy_losses[0]["legacy"].item(), 3.0)
        self.assertEqual(gpt_calls, [1])
        self.assertEqual(gpt_backward.item(), 2.0)
        self.assertEqual(gpt_losses[0].local_loss_weight.item(), 3.0)

    def test_no_pipeline_schedule_passes_every_microbatch_index(self):
        schedules, timer = load_schedules(
            num_microbatches=5,
            args=types.SimpleNamespace(current_pipe=0, mbs_seq=(((2, 8),) * 3,)),
        )
        seen = []

        def gpt(unused_iterator, unused_model, unused_extra, current_data_idx):
            seen.append(current_data_idx)
            output = types.SimpleNamespace(
                local_loss_sum=torch.tensor(1.0),
                local_loss_weight=torch.tensor(1.0),
                nominal_global_loss_scale=1.0,
            )
            return torch.tensor(1.0), None, lambda unused_output: output

        losses = schedules.forward_backward_no_pipelining(
            gpt, None, [_Model()], optimizer=None, timers=lambda unused: timer,
            forward_only=True,
        )

        self.assertEqual(seen, [0, 1, 2])
        self.assertEqual(len(losses), 3)

    def test_interleaved_schedule_wraps_global_forward_index(self):
        schedules, unused_timer = load_schedules(
            num_microbatches=2,
            args=types.SimpleNamespace(
                current_pipe=0,
                mbs_seq=(((4, 8), (4, 8), (8, 8)),),
            ),
        )
        indices = [
            schedules.interleaved_data_index(i, pipeline_size=2, num_chunks=2)
            for i in range(8)
        ]
        self.assertEqual(schedules.scheduled_num_microbatches(), 3)
        with self.assertRaisesRegex(ValueError, "not divisible"):
            schedules.validate_interleaved_microbatch_count(3, 2)
        schedules.validate_interleaved_microbatch_count(4, 2)
        self.assertEqual(
            indices,
            [0, 1, 0, 1, 2, 3, 2, 3],
        )
        self.assertIn(
            "validate_interleaved_microbatch_count(",
            self.training,
        )

    def test_training_world_reduces_weight_before_normalized_bucket_execution(self):
        self.assertIn("global_loss_weight", self.training)
        self.assertIn("group=torch.distributed.group.WORLD", self.training)
        self.assertIn(
            "normalization_scale=global_normalization_scale(", self.training
        )

    def test_scheduler_and_cursor_use_total_gbs_with_success_semantics(self):
        self.assertIn("decay_steps = args.lr_decay_iters * args.total_gbs", self.training)
        self.assertIn("lr_scheduler.step(increment=args.total_gbs)", self.training)
        self.assertIn("args.global_data_cursor += args.total_gbs", self.training)
        self.assertIn("args.successful_optimizer_steps += 1", self.training)
        self.assertIn("args.global_data_cursor = 0", self.arguments)
        self.assertIn("args.successful_optimizer_steps = 0", self.arguments)
        self.assertIn("group=torch.distributed.group.WORLD", self.optimizer)
        self.assertIn("if found_inf_flag:", self.optimizer)
        self.assertNotIn("if found_inf_flag and SKIP_WHEN_INF", self.optimizer)

    def test_fp16_overflow_skips_the_underlying_optimizer_step(self):
        optimizer_module = load_optimizer_module()
        optimizer = optimizer_module.Float16OptimizerWithFloat16Params.__new__(
            optimizer_module.Float16OptimizerWithFloat16Params
        )
        base_optimizer = types.SimpleNamespace(step_calls=0)
        base_optimizer.step = lambda: setattr(
            base_optimizer, "step_calls", base_optimizer.step_calls + 1
        )
        optimizer.optimizer = base_optimizer
        optimizer.grad_scaler = types.SimpleNamespace(
            update_calls=[],
            update=lambda found_inf: optimizer.grad_scaler.update_calls.append(
                found_inf
            ),
        )
        optimizer.clip_grad = 1.0
        optimizer.log_num_zeros_in_grad = True
        optimizer._copy_model_grads_to_main_grads = lambda: None
        optimizer._unscale_main_grads_and_check_for_nan = lambda: True
        optimizer.clip_grad_norm = lambda unused_limit: self.fail(
            "overflow must skip clipping"
        )
        optimizer.count_zeros = lambda: self.fail(
            "overflow must skip zero counting"
        )
        optimizer._copy_main_params_to_model_params = lambda: self.fail(
            "overflow must skip model updates"
        )

        result = optimizer.step()

        self.assertEqual(result, (False, None, None))
        self.assertEqual(optimizer.grad_scaler.update_calls, [True])
        self.assertEqual(base_optimizer.step_calls, 0)

    def test_checkpoint_restores_global_clocks(self):
        self.assertIn("restore_training_clock", self.checkpointing)
        self.assertIn("validate_training_clock_states", self.training)
        self.assertIn("optimizer.get_loss_scale().item()", self.training)

    def test_gpt_evaluation_reduces_numerator_and_weight_once(self):
        self.assertIn("uses_loss_output_contract", self.pretrain)
        self.assertIn("evaluation_loss_sum", self.training)
        self.assertIn("evaluation_loss_weight", self.training)


if __name__ == "__main__":
    unittest.main()
