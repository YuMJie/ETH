import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from tests.test_gradient_executor import load_executor
from tests.test_gradient_fragment_planner import descriptor


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "grad_sync" / "loss.py"


def load_loss_module():
    assert MODULE_PATH.exists(), "global loss normalization module is missing"
    full_name = "megatron.grad_sync.loss"
    spec = importlib.util.spec_from_file_location(full_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


class GlobalLossAndClockTest(unittest.TestCase):
    def test_unequal_pipeline_weights_match_single_device_reference_after_executor(self):
        loss_module = load_loss_module()
        executor, bucket_plan, planner = load_executor()
        descriptor_cls = sys.modules[
            "megatron.grad_sync.metadata"
        ].SerializableGradientDescriptor
        descriptors = tuple(
            descriptor(
                descriptor_cls,
                "weight",
                rank,
                rank,
                1,
                0,
                (1,),
                (1,),
                False,
                -1,
            )
            for rank in range(2)
        )
        fragment_plan = planner.plan_canonical_fragments(descriptors, (0, 1))
        plan = bucket_plan.build_gradient_sync_plan(
            fragment_plan, "tensor", 1024, "gpt", "manifest", "topology"
        )
        parameters = [torch.tensor([1.0], requires_grad=True) for _ in range(2)]
        outputs = (
            loss_module.gpt_loss_output(
                parameters[0] * torch.tensor([1.0, 3.0]),
                torch.tensor([1.0, 1.0]),
                8.0,
            ),
            loss_module.gpt_loss_output(
                parameters[1] * torch.tensor([5.0, 9.0]),
                torch.tensor([1.0, 0.0]),
                8.0,
            ),
        )
        main_grads = []
        for parameter, output in zip(parameters, outputs):
            (output.local_loss_sum / output.nominal_global_loss_scale).backward()
            main_grads.append(parameter.grad.detach().clone())

        condition = threading.Condition()
        payloads = []

        def all_reduce_sum(payload, unused_handle):
            with condition:
                payloads.append(payload)
                if len(payloads) == 2:
                    total = payloads[0].clone() + payloads[1]
                    for item in payloads:
                        item.copy_(total)
                    condition.notify_all()
                else:
                    condition.wait_for(lambda: len(payloads) == 2)

        class Registry:
            def handle(self, unused_group_id):
                return "group"

        runners = []
        for rank in range(2):
            runners.append(
                executor.GradientBucketExecutor(
                    plan,
                    rank,
                    (descriptors[rank],),
                    (
                        SimpleNamespace(
                            logical_param_id="weight",
                            main_grad=main_grads[rank],
                            device=main_grads[rank].device,
                        ),
                    ),
                    Registry(),
                    all_reduce_sum,
                )
            )
        normalization_scale = loss_module.global_normalization_scale(
            8.0, sum(item.local_loss_weight for item in outputs)
        )
        threads = [
            threading.Thread(
                target=runner.execute,
                kwargs={"normalization_scale": normalization_scale},
            )
            for runner in runners
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)

        self.assertFalse(any(thread.is_alive() for thread in threads))
        for main_grad in main_grads:
            self.assertTrue(torch.equal(main_grad, torch.tensor([3.0])))

    def test_gpt_loss_returns_numerator_weight_and_global_mean_gradient(self):
        loss_module = load_loss_module()
        first = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        second = torch.tensor([4.0, 5.0], requires_grad=True)
        out_a = loss_module.gpt_loss_output(
            first, torch.tensor([1.0, 0.0, 1.0]), nominal_global_loss_scale=8.0
        )
        out_b = loss_module.gpt_loss_output(
            second, torch.tensor([1.0, 0.0]), nominal_global_loss_scale=8.0
        )

        self.assertEqual(out_a.local_loss_sum.item(), 4.0)
        self.assertEqual(out_a.local_loss_weight.item(), 2.0)
        self.assertEqual(out_b.local_loss_sum.item(), 4.0)
        self.assertEqual(out_b.local_loss_weight.item(), 1.0)
        ((out_a.local_loss_sum + out_b.local_loss_sum) / 8.0).backward()
        scale = loss_module.global_normalization_scale(
            8.0, out_a.local_loss_weight + out_b.local_loss_weight
        )
        self.assertTrue(
            torch.equal(first.grad * scale, torch.tensor([1 / 3, 0.0, 1 / 3]))
        )
        self.assertTrue(
            torch.equal(second.grad * scale, torch.tensor([1 / 3, 0.0]))
        )

    def test_zero_global_weight_and_invalid_nominal_scale_fail(self):
        loss_module = load_loss_module()
        with self.assertRaisesRegex(ValueError, "nominal_global_loss_scale"):
            loss_module.gpt_loss_output(torch.ones(1), torch.ones(1), 0.0)
        with self.assertRaisesRegex(ValueError, "global_loss_weight"):
            loss_module.global_normalization_scale(4.0, torch.tensor(0.0))

    def test_sample_slots_are_disjoint_and_dp_transitions_preserve_union(self):
        loss_module = load_loss_module()
        sequences = (((4, 8), (2, 8)), ((2, 8), (2, 8)))
        offsets = (0, 6)
        total_gbs = 10

        dp1 = set(
            loss_module.local_sample_slots(0, 0, 1, 0, sequences, offsets, 0)
        )
        dp2 = set()
        for dp_rank in range(2):
            dp2.update(
                loss_module.local_sample_slots(
                    0, 0, 2, dp_rank, sequences, offsets, 0
                )
            )
        self.assertEqual(dp1, dp2)

        all_first_iteration = []
        for pipeline_id, sequence in enumerate(sequences):
            for microbatch_index, (global_mbs, unused_seq) in enumerate(sequence):
                for dp_rank in range(2):
                    if global_mbs % 2 == 0:
                        all_first_iteration.extend(
                            loss_module.local_sample_slots(
                                pipeline_id,
                                microbatch_index,
                                2,
                                dp_rank,
                                sequences,
                                offsets,
                                0,
                            )
                        )
        self.assertEqual(len(all_first_iteration), len(set(all_first_iteration)))
        second_iteration = set(
            loss_module.local_sample_slots(
                0, 0, 1, 0, sequences, offsets, total_gbs
            )
        )
        self.assertTrue(set(all_first_iteration).isdisjoint(second_iteration))

        batch_a = loss_module.synthetic_gpt_batch((0, 1), 4, 100, "cpu")
        batch_b = loss_module.synthetic_gpt_batch((2, 3), 4, 100, "cpu")
        self.assertFalse(torch.equal(batch_a[0], batch_b[0]))
        self.assertFalse(torch.equal(batch_a[1], batch_b[1]))

    def test_training_clock_advances_attempts_and_only_counts_successes(self):
        loss_module = load_loss_module()
        clock = loss_module.TrainingClock(
            total_gbs=10, global_data_cursor=20, successful_optimizer_steps=3
        )
        skipped = clock.after_attempt(update_successful=False)
        successful = clock.after_attempt(update_successful=True)

        self.assertEqual(skipped, 0)
        self.assertEqual(successful, 10)
        self.assertEqual(clock.global_data_cursor, 40)
        self.assertEqual(clock.successful_optimizer_steps, 4)

    def test_checkpoint_clock_restore_and_cross_rank_validation(self):
        loss_module = load_loss_module()
        restored = loss_module.restore_training_clock(
            SimpleNamespace(
                consumed_train_samples=12,
                global_data_cursor=30,
                successful_optimizer_steps=4,
            ),
            iteration=7,
        )
        legacy = loss_module.restore_training_clock(
            SimpleNamespace(consumed_train_samples=12), iteration=7
        )

        self.assertEqual(restored, (30, 4))
        self.assertEqual(legacy, (12, 7))
        loss_module.validate_training_clock_states([(30, 4, 0.1, 1024.0)] * 2)
        with self.assertRaisesRegex(ValueError, "training clock mismatch"):
            loss_module.validate_training_clock_states(
                [(30, 4, 0.1, 1024.0), (40, 4, 0.1, 1024.0)]
            )

    def test_evaluation_aggregates_numerator_and_weight_before_dividing(self):
        loss_module = load_loss_module()
        outputs = (
            loss_module.LossOutput(torch.tensor(2.0), torch.tensor(1.0), 8.0, {}),
            loss_module.LossOutput(torch.tensor(6.0), torch.tensor(1.0), 8.0, {}),
            loss_module.LossOutput(torch.tensor(12.0), torch.tensor(2.0), 8.0, {}),
        )

        numerator, weight = loss_module.sum_loss_outputs(outputs)

        self.assertEqual(numerator.item(), 20.0)
        self.assertEqual(weight.item(), 4.0)
        self.assertEqual((numerator / weight).item(), 5.0)


if __name__ == "__main__":
    unittest.main()
