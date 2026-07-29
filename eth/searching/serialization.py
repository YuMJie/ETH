# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json

from .config import config_details
from .state import HeterogeneousPlan, SearchConfig, SearchStage

gpt_configs = {
    "350M": (24, 2048, 1024, 4096, 16, 64, 51200, "fp16"),
    "1_3B": (24, 2048, 2048, 8192, 32, 64, 51200, "fp16"),
    "2_6B": (32, 2048, 2560, 10240, 32, 80, 51200, "fp16"),
    "6_7B": (32, 2048, 4096, 16384, 32, 128, 51200, "fp16"),
    "13B": (40, 2048, 5120, 20480, 40, 128, 51200, "fp16"),
    "70B": (80, 2048, 8192, 28672, 64, 128, 32000, "fp16"),
    "scale-layer": (1, 2048, 512, 2048, 8, 64, 51200, "fp16"),
}

def dump_config_to_json(config, file_name, args):
    if args.model_name == "scale-layer":
        model_name = "gpt"
        model_size = "scale-layer"
    else:
        model_name = args.model_name
        model_size = args.model_size
    num_layers = args.num_layers
    config_dict = {}
    config_dict["node_info"] = config.node_info
    config_dict["model_name"] = model_name
    config_dict["model_size"] = model_size
    if model_name == "gpt":
        _, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = gpt_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["seq_length"] = seq_len
        config_dict["max_position_embeddings"] = seq_len
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["hidden_size"] = hidden_size
    else:
        raise RuntimeError(f"{model_name} not supported.")

    config_dict["global_batch_size"] = config.global_bs
    config_dict["micro_batch_size"] = config.micro_bs
    config_dict["num_stages"] = config.num_stages

    tp_size_of_each_op = []
    dp_size_of_each_op = []
    recompute_ops = []
    algo_of_each_op = []
    num_ops_in_each_stage = []
    config_dict["num_gpus"] = []
    config_dict["checkpoint_activations"] = []
    config_dict["resharding_stages"] = []
    for i in range(config.num_stages):
        tp_size_of_each_op.append(config.stages[i].tp_size)
        dp_size_of_each_op.append(config.stages[i].dp_size)
        recompute_ops.append(config.stages[i].recompute_ops)
        algo_of_each_op.append(config.stages[i].algo)
        num_ops_in_each_stage.append(len(config.stages[i].ops))

        config_dict["num_gpus"].append(config.stages[i].num_gpus)
        if max(config.stages[i].recompute_ops) > 0:
            config_dict["checkpoint_activations"].append(True)
        else:
            config_dict["checkpoint_activations"].append(False)
        if max(config.stages[i].tp_size) != min(config.stages[i].tp_size) \
            or max(config.stages[i].dp_size) != min(config.stages[i].dp_size) \
            or max(config.stages[i].algo) != min(config.stages[i].algo):
            config_dict["resharding_stages"].append(True)
        else:
            config_dict["resharding_stages"].append(False)

    config_dict["num_ops_in_each_stage"] = num_ops_in_each_stage
    config_dict["model_parallel_size_of_each_op"] = tp_size_of_each_op
    config_dict["data_parallel_size_of_each_op"] = dp_size_of_each_op
    config_dict["recompute_ops"] = recompute_ops
    config_dict["algo_of_each_op"] = algo_of_each_op
    json.dump(config_dict, open(file_name, 'w'), indent=4)
    print(f"config has been saved to {file_name}")


def dump_hybrid_config_to_json(config: HeterogeneousPlan, file_name, args):


    if args.model_name == "scale-layer":
        model_name = "gpt"
        model_size = "scale-layer"
    else:
        model_name = args.model_name
        model_size = args.model_size
    num_layers = args.num_layers
    config_dict = {}
    config_dict["model_name"] = model_name
    config_dict["model_size"] = model_size
    if model_name == "gpt":
        _, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, _ = gpt_configs[model_size]
        config_dict["num_layers"] = num_layers
        config_dict["seq_length"] = seq_len
        config_dict["max_position_embeddings"] = seq_len
        config_dict["num_attention_heads"] = num_attention_heads
        config_dict["hidden_size"] = hidden_size
    else:
        raise RuntimeError(f"{model_name} not supported.")

    config_dict["pipe"]= []
    for pipe_idx in range(config.num_pipelines):
        config_dict_pipe = {}
        config_dict_pipe["global_batch_size"] = config.pipelines[pipe_idx].global_bs
        config_dict_pipe["micro_batch_size"] = config.pipelines[pipe_idx].micro_bs
        config_dict_pipe["num_stages"] = config.pipelines[pipe_idx].num_stages
        config_dict_pipe["node_info"] = config.pipelines[pipe_idx].node_info



        tp_size_of_each_op = []
        dp_size_of_each_op = []
        recompute_ops = []
        algo_of_each_op = []
        num_ops_in_each_stage = []
        config_dict_pipe["num_gpus"] = []
        config_dict_pipe["checkpoint_activations"] = []
        config_dict_pipe["resharding_stages"] = []

        for i in range(config.pipelines[pipe_idx].num_stages):
            tp_size_of_each_op.append(config.pipelines[pipe_idx].stages[i].tp_size)
            dp_size_of_each_op.append(config.pipelines[pipe_idx].stages[i].dp_size)
            recompute_ops.append(config.pipelines[pipe_idx].stages[i].recompute_ops)
            algo_of_each_op.append(config.pipelines[pipe_idx].stages[i].algo)
            num_ops_in_each_stage.append(len(config.pipelines[pipe_idx].stages[i].ops))

            config_dict_pipe["num_gpus"].append(config.pipelines[pipe_idx].stages[i].num_gpus)
            if max(config.pipelines[pipe_idx].stages[i].recompute_ops) > 0:
                config_dict_pipe["checkpoint_activations"].append(True)
            else:
                config_dict_pipe["checkpoint_activations"].append(False)
            if max(config.pipelines[pipe_idx].stages[i].tp_size) != min(config.pipelines[pipe_idx].stages[i].tp_size) \
                or max(config.pipelines[pipe_idx].stages[i].dp_size) != min(config.pipelines[pipe_idx].stages[i].dp_size) \
                or max(config.pipelines[pipe_idx].stages[i].algo) != min(config.pipelines[pipe_idx].stages[i].algo):
                config_dict_pipe["resharding_stages"].append(True)
            else:
                config_dict_pipe["resharding_stages"].append(False)

        config_dict_pipe["num_ops_in_each_stage"] = num_ops_in_each_stage
        config_dict_pipe["model_parallel_size_of_each_op"] = tp_size_of_each_op
        config_dict_pipe["data_parallel_size_of_each_op"] = dp_size_of_each_op
        config_dict_pipe["recompute_ops"] = recompute_ops
        config_dict_pipe["algo_of_each_op"] = algo_of_each_op
        config_dict["pipe"].append(config_dict_pipe)

    json.dump(config_dict, open(file_name, 'w'), indent=4)
    print(f"config has been saved to {file_name}")

def save_config_info_to_csv(config, reserved_mem_list, file_name):
    info_to_csv = [["stage-index", "time", "memory(total)", "memory(normal)", "memory(reserved)"]]
    for i in range(config.num_stages):
        info_to_csv.append([f"stage-{i}", f"{config.time_list[i]:.2f}", f"{config.memory_list[i]:.2f}", f"{(config.memory_list[i]-reserved_mem_list[i]):.2f}", f"{reserved_mem_list[i]:.2f}"])

    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)

def save_distribution_to_csv(num_targets_list, num_hops_list, file_name):
    info_to_csv = [["num_targets"] + num_targets_list, ["num_hops"] + num_hops_list]

    with open(file_name, mode="w", newline="") as file:
        writer = csv.writer(file)
        for row in info_to_csv:
            writer.writerow(row)
