import unittest

from eth.searching.config_adapter import convert


def make_search_config(ops_per_layer):
    first_stage_count = 1 + ops_per_layer
    second_stage_count = 2
    return {
        "model_name": "gpt",
        "model_size": "350M",
        "num_layers": 1,
        "seq_length": 2048,
        "max_position_embeddings": 2048,
        "num_attention_heads": 16,
        "hidden_size": 1024,
        "global_batch_size": 8,
        "micro_batch_size": 1,
        "num_stages": 2,
        "num_gpus": [1, 2],
        "num_ops_in_each_stage": [first_stage_count, second_stage_count],
        "checkpoint_activations": [False, True],
        "resharding_stages": [False, False],
        "model_parallel_size_of_each_op": [
            [1] * first_stage_count,
            [2] * second_stage_count,
        ],
        "data_parallel_size_of_each_op": [
            [1] * first_stage_count,
            [1] * second_stage_count,
        ],
        "algo_of_each_op": [
            [0] * first_stage_count,
            [1] * second_stage_count,
        ],
        "recompute_ops": [
            [0] * first_stage_count,
            [1] * second_stage_count,
        ],
    }


class ConfigAdapterContractTest(unittest.TestCase):
    def test_public_and_compatible_manifests_preserve_stage_topology(self):
        for ops_per_layer in (10, 13):
            with self.subTest(ops_per_layer=ops_per_layer):
                pipe = convert(make_search_config(ops_per_layer))["pipe"][0]

                self.assertEqual(pipe["num_ops_in_each_stage"], [11, 2])
                self.assertEqual(pipe["num_gpus"], [1, 2])
                self.assertEqual(
                    pipe["model_parallel_size_of_each_op"],
                    [[1] * 11, [2] * 2],
                )
                self.assertEqual(pipe["algo_of_each_op"], [[0] * 11, [1] * 2])
                self.assertEqual(pipe["recompute_ops"], [[0] * 11, [1] * 2])

    def test_other_manifest_widths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "supported GPT manifest"):
            convert(make_search_config(12))


if __name__ == "__main__":
    unittest.main()
