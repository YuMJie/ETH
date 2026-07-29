import ast
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DDP_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "model" / "distributed.py"
TRAINING_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "training.py"
OPTIMIZER_INIT_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "optimizer" / "__init__.py"
MPU_INIT_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "mpu" / "initialize.py"
MEGATRON_INIT_PATH = REPO_ROOT / "eth" / "runtime" / "megatron" / "initialize.py"


class LocalDDPGradSyncContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = DDP_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.training_source = TRAINING_PATH.read_text(encoding="utf-8")
        cls.optimizer_init_source = OPTIMIZER_INIT_PATH.read_text(encoding="utf-8")
        cls.mpu_init_source = MPU_INIT_PATH.read_text(encoding="utf-8")
        cls.megatron_init_source = MEGATRON_INIT_PATH.read_text(encoding="utf-8")

    def test_has_one_active_distributed_data_parallel_definition(self):
        definitions = [
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DistributedDataParallel"
        ]
        self.assertEqual(len(definitions), 1)

    def test_runtime_ddp_does_not_create_groups_or_hardcode_npu(self):
        self.assertNotIn("torch.distributed.new_group", self.source)
        self.assertNotIn(".npu()", self.source)

    def test_main_grad_binding_is_not_guarded_by_contiguous_buffer_flag(self):
        definitions = [
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef) and node.name == "DistributedDataParallel"
        ]
        self.assertEqual(len(definitions), 1)
        class_source = ast.get_source_segment(self.source, definitions[0])
        self.assertIn("self._initialize_main_grad_buffers()", class_source)
        self.assertNotIn("if self.use_contiguous_buffers", class_source)
        self.assertIn("param.main_grad", self.source)

    def test_training_initializes_and_executes_one_shared_gradient_manager(self):
        self.assertIn("initialize_gradient_sync", self.training_source)
        self.assertIn("args.gradient_sync_manager =", self.training_source)
        self.assertEqual(
            self.training_source.count("args.gradient_sync_manager.execute("), 1
        )
        self.assertNotIn(
            "for model_module in model:\n            model_module.allreduce_gradients()",
            self.training_source,
        )

    def test_local_ddp_zeroes_main_grad_even_when_legacy_flag_is_disabled(self):
        self.assertIn(
            "if args.DDP_impl == 'local':\n        for partition in model:\n            partition.zero_grad_buffer()",
            self.training_source,
        )

    def test_optimizer_preserves_unconditional_local_ddp_main_grad_views(self):
        self.assertIn(
            "persistent_main_grad = args.DDP_impl == 'local'",
            self.optimizer_init_source,
        )

    def test_mpu_runtime_group_lookup_is_frozen_after_initialization(self):
        tree = ast.parse(self.mpu_init_source)
        get_group = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "get_group"
        )
        get_group_source = ast.get_source_segment(self.mpu_init_source, get_group)
        self.assertNotIn("new_group", get_group_source)
        self.assertIn("is not initialized", get_group_source)
        self.assertIn("def initialize_process_groups", self.mpu_init_source)
        self.assertNotIn("use_local_synchronization=True", self.mpu_init_source)

    def test_profiler_precreates_groups_before_frozen_lookup(self):
        self.assertIn(
            "mpu.initialize_process_groups(profiler_groups)",
            self.megatron_init_source,
        )
        self.assertIn(
            "params_have_main_grad, persistent_main_grad",
            self.optimizer_init_source,
        )


if __name__ == "__main__":
    unittest.main()
