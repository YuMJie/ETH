# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import datetime
import json
import os
import sys

gpt_configs = {
    "350M": (24, 2048, 1024, 4096, 16, 64, 51200, "fp16"),
    "1_3B": (24, 2048, 2048, 8192, 32, 64, 51200, "fp16"),
    "2_6B": (32, 2048, 2560, 10240, 32, 80, 51200, "fp16"),
    "6_7B": (32, 2048, 4096, 16384, 32, 128, 51200, "fp16"),
    "13B": (40, 2048, 5120, 20480, 40, 128, 51200, "fp16"),
    "70B": (80, 2048, 8192, 28672, 64, 128, 32000, "fp16"),
    "scale-layer": (1, 2048, 512, 2048, 8, 64, 51200, "fp16"),
}
memory_ratio = {"gpt": {"main_params": 2, "optimizer": 4}}

def add_model_args(parser):
    group = parser.add_argument_group(title='model information')
    group.add_argument('--model-name', type=str, choices=['gpt'], default='gpt', help='Configure this search option.')
    group.add_argument('--model-size', type=str, default=None, help='Configure this search option.')
    group.add_argument('--num-layers', type=int, default=None, help='Configure this search option.')
    group.add_argument('--global-batch-size', type=int, default=None, help='Configure this search option.')
    group.add_argument('--micro-batch-size', nargs='+', type=int, default=None, help='Configure this search option.')
    group.add_argument('--max-tp', type=int, default=None, help='Configure this search option.')
    group.add_argument('--num-algos', type=int, default=None, help='Configure this search option.')

    return parser

def add_hardware_args(parser):
    group = parser.add_argument_group(title='hardware information')
    group.add_argument('--num-nodes', type=int, default=None, help='Configure this search option.')

    return parser

def add_path_args(parser):
    group = parser.add_argument_group(title='path information')
    group.add_argument('--log-path', type=str, default=None, help='Configure this search option.')
    group.add_argument('--profiled-time-path', type=str, default=None, help='Configure this search option.')
    group.add_argument('--config-save-path', type=str, default=None, help='Configure this search option.')
    group.add_argument('--config-suffix', type=str, default=None, help='Configure this search option.')
    group.add_argument('--save-prefix', type=str, default="", help='Configure this search option.')

    return parser

def add_budget_args(parser):
    group = parser.add_argument_group(title='budgets for the search algorithm')
    group.add_argument('--max-num-hops', type=int, default=None, help='Configure this search option.')
    group.add_argument('--max-num-trials', type=int, default=100, help='max number of search time')
    group.add_argument('--time-budget-per-trial', type=int, default=None, help='Configure this search option.')
    group.add_argument('--time-budget-total', type=int, default=10, help='Total time budget for the search')
    group.add_argument('--start-num-stages', type=int, default=None, help='Configure this search option.')
    group.add_argument('--end-num-stages', type=int, default=None, help='Configure this search option.')
    group.add_argument('--op-grain', type=str ,default="operator",choices=["operator", "block","layer","other"], help='Configure this search option.')
    group.add_argument('--bisection', action='store_true', help='bisect the dec_op operator')

    return parser

def add_heuristic_args(parser):
    group = parser.add_argument_group(title='heuristics in the search algorithm')
    group.add_argument('--op-group-size', type=int, default=1, help='Configure this search option.')
    group.add_argument('--max-op-move-steps', type=int, default=5, help='Configure this search option.')
    group.add_argument('--memory-pred-type', type=str, default='MAX', help='Configure this search option.')
    group.add_argument('--initial-point', type=str, default="balance", help='Configure this search option.')

    return parser

def add_debug_args(parser):
    group = parser.add_argument_group(title='debug arguments')
    group.add_argument('--print-debug-info', action='store_true', help='Configure this search option.')
    group.add_argument('--print-move-op-details', action='store_true', help='Configure this search option.')

    return parser

def add_test_args(parser):
    group = parser.add_argument_group(title='arguments under development')
    group.add_argument('--do-not-consider-shared-tensor-space', action='store_false', help='Configure this search option.', dest='consider_shared_space')
    group.add_argument('--do-not-consider-reserved-space', action='store_false', help='Configure this search option.', dest='consider_reserved_space')
    group.add_argument('--predict-delta-time', action='store_true', help='Configure this search option.')
    group.add_argument('--do-not-use-flex-recompute', action='store_false', help='Configure this search option.', dest='flex_recompute')
    group.add_argument('--add-action-tp-dp-exchange', action='store_true', help='Configure this search option.')
    group.add_argument('--peak-mem-in-backward', type=int, default=0, help='Configure this search option.')
    group.add_argument('--finetune-after-trial', type=int, default=0, help='Configure this search option.')
    group.add_argument('--no-multi-process', action='store_false', help='Configure this search option.', dest='multi_process')
    group.add_argument('--random-order-actions', action='store_true', help='Configure this search option.')
    group.add_argument('--support-comm-predict', action='store_true', help='Configure this search option.')
    group.add_argument('--sort-metric', type=str, default='max_stage_time', help='Configure this search option.')
    group.add_argument('--print-gpu-mig-details', action='store_true', help='Configure this search option.')
    group.add_argument('--finetune-tp-dp-after-trial', action='store_true', help='Configure this search option.')
    group.add_argument('--init-dim', type=str, default="tp", help='Configure this search option.')
    group.add_argument('--num-partners-in-op-mig', type=int, default=1, help='Configure this search option.')
    group.add_argument('--do-not-continue-when-fail', action='store_false', help='Configure this search option.', dest='continue_when_fail')
    group.add_argument('--adaptive-hyper-parameters', type=int, default=5, help='Configure this search option.')
    group.add_argument('--num-of-saved-configs', type=int, default=1, help='Configure this search option.')
    group.add_argument('--simple-prim-mbs', action='store_true', help='Configure this search option.')
    group.add_argument('--simple-prim-mig', action='store_true', help='Configure this search option.')
    group.add_argument('--only-top-1-target', action='store_true', help='Configure this search option.')
    group.add_argument('--consider-collective-memory', action='store_true', help='Configure this search option.')

    return parser

def add_hetero_search_args(parser):
    group = parser.add_argument_group(title='hetero search arguments')
    # group.add_argument('--num-gpus-per-node-list', nargs='+', type=int, default=None, help='Configure this search option.')
    # group.add_argument('--gpu-type-list', nargs='+', type=str, default=None, help='Configure this search option.')
    group.add_argument('--node-json-path', type=str, default=None, help='Actual node information, like gpu_type and gpu_num')
    group.add_argument('--node-info', type=json.loads, default=None, help='Actual node information, like gpu_type and gpu_num')
    group.add_argument('--device-json-path', type=str, default=None, help='GPU information, like memory')
    group.add_argument('--comm-revised', action='store_true', help='enable revised communication model')
    group.add_argument('--config-node-order-idx', type = int, default=None, help='node order index')
    group.add_argument('--end-dg-idx', type=int, default=sys.maxsize, help='exclusive device-group end index')
    group.add_argument('--start-dg-idx', type=int, default=-1, help='exclusive device-group start index')
    group.add_argument('--num-multi-process', type=int, default=32, help='Configure this search option.')
    group.add_argument('--num-of-show-configs', type=int, default=5, help='Configure this search option.')
    group.add_argument('--no-add-action-tp-dp', action='store_false', help='Configure this search option.', dest='add_action_tp_dp')
    group.add_argument('--min-num-pipeline', type=int, default=1, help='Configure this search option.')
    group.add_argument('--max-num-pipeline', type=int, default=1, help='Configure this search option.')
    group.add_argument('--test-search-time', action='store_true', help='Measure search time without saving a result.')
    group.add_argument('--reduce-output', action='store_true', help='Reduce progress output during search.')
    group.add_argument('--easy-mode', action='store_true', help='Use the reduced device-order search space.')
    group.add_argument('--use-cache', action='store_true', help='Reuse cached search results when available.')
    group.add_argument('--optimal-prune', action='store_true', help='Enable upper-bound pruning.')
    group.add_argument('--upper-bound-rate', type=float, default=1.5, help='Configure this search option.')
    group.add_argument('--sort-pipeline', action='store_true', help='Sort pipeline splits by the imbalance score.')
    group.add_argument('--dp-find-pipeline', action='store_true', help='Use dynamic programming to enumerate pipeline splits.')
    group.add_argument('--homo-stage', action='store_true', help='Require homogeneous device counts per stage.')
    group.add_argument('--elastic-time', action='store_true', help='Allocate search time according to candidate cost.')

    return parser

def add_exp_args(parser):
    group = parser.add_argument_group(title='experiment arguments')
    group.add_argument('--no-recomp', action='store_true', help='Disable recomputation actions.')
    group.add_argument('--no-dp', action='store_true', help='Disable data-parallel actions.')
    return parser
global_args = None


def validate_stage_range(args):
    if args.start_num_stages < 1:
        raise ValueError("--start-num-stages must be at least 1")
    if args.end_num_stages < args.start_num_stages:
        raise ValueError("--end-num-stages must be greater than or equal to --start-num-stages")


def validate_device_group_range(args):
    if args.start_dg_idx < -1:
        raise ValueError("--start-dg-idx must be at least -1")
    if args.end_dg_idx <= args.start_dg_idx:
        raise ValueError("--end-dg-idx must be greater than --start-dg-idx")


def check_initial_point(args):
    with open(args.initial_point, "r") as f:
        config_dict = json.load(f)
        model_name = config_dict["model_name"]
        model_size = config_dict["model_size"]
        num_layers = config_dict["num_layers"]

        if(args.model_name != None):
            assert args.model_name == model_name, f"model name in the initial point is {model_name}, not {args.model_name}"
        else:
            args.model_name = model_name
        if(args.model_size != None):
            assert args.model_size == model_size, f"model size in the initial point is {model_size}, not {args.model_size}"
        else:
            args.model_size = model_size
        if(args.num_layers != None):
            assert args.num_layers == num_layers, f"num_layers in the initial point is {num_layers}, not {args.num_layers}"
        else:
            args.num_layers = num_layers
        if(config_dict.get("pipe")==None):
            global_bs = config_dict["global_batch_size"]
            if(args.global_batch_size != None):
                assert args.global_batch_size == global_bs, f"global batch size in the initial point is {global_bs}, not {args.global_batch_size}"
            else:
                args.global_batch_size = global_bs
            args.num_gpus = sum(config_dict["num_gpus"])

        # if(args.num_nodes != None and args.num_gpus_per_node_list != None): #TODO
        #     assert args.num_nodes == len(args.num_gpus_per_node_list), f"num_nodes in the initial point is {len(args.num_gpus_per_node_list)}, not {args.num_nodes}"


def parse_args():
    global global_args
    if global_args is not None:
        return global_args
    # else:
    #     print("global_args is None, parsing args again")
    parser = argparse.ArgumentParser()
    parser = add_model_args(parser)
    parser = add_hardware_args(parser)
    parser = add_path_args(parser)
    parser = add_budget_args(parser)
    parser = add_heuristic_args(parser)
    parser = add_debug_args(parser)
    parser = add_test_args(parser)
    parser = add_hetero_search_args(parser)
    parser = add_exp_args(parser)
    args = parser.parse_args()
    validate_device_group_range(args)

    if args.initial_point is not None and os.path.exists(args.initial_point):
        check_initial_point(args)
    else:
        pass
        # print(f"initial point {args.initial_point} does not exist.")
    # args.num_gpus = args.num_gpus_per_node * args.num_nodes


    config_dict = {}
    if os.path.exists(args.initial_point):
        with open(args.initial_point, "r") as f:
            config_dict = json.load(f)
            args.model_name = config_dict["model_name"]
            args.model_size = config_dict["model_size"]
    if(config_dict.get("pipe")!=None):
            args.gpu_type_num_dict = config_dict["pipe"][0]["node_info"]
            args.max_tp = 8
    else:
        node_id = 0
        if(config_dict.get("node_info") !=None):
            gpu_type_num_dict = config_dict["node_info"]
            args.gpu_type_num_dict = {}
            for key in gpu_type_num_dict.keys():
                for i in range (gpu_type_num_dict[key]["NODE_NUM"]):
                    args.gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict[key]["GPU"], "GPU_NUM": gpu_type_num_dict[key]["GPU_NUM"]}
                    node_id +=1
        else:
            if(args.node_info == None):
                assert args.node_json_path is not None  , "node_json_path should be provided"
                with open(args.node_json_path, "r") as f:
                    gpu_type_num_dict_ = json.load(f)
            else:
                gpu_type_num_dict_ = args.node_info
            node_id = 0
            args.gpu_type_num_dict = {}
            for key in gpu_type_num_dict_.keys():
                for i in range (gpu_type_num_dict_[key]["NODE_NUM"]):
                    args.gpu_type_num_dict[f"{node_id}"] = {"GPU": gpu_type_num_dict_[key]["GPU"], "GPU_NUM": gpu_type_num_dict_[key]["GPU_NUM"]}
                    node_id +=1
        args.num_nodes = len(args.gpu_type_num_dict)
        args.max_num_pipeline = min(args.max_num_pipeline,args.num_nodes)
        args.num_gpus = sum([args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])
        args.gpu_type_set = [args.gpu_type_num_dict[f"{i}"]["GPU"] for i in range(args.num_nodes)]
        args.gpu_type_set = list(set(args.gpu_type_set))
        if args.max_tp is None:
            # args.max_tp = args.num_gpus_per_node
            args.max_tp = max([args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)]) #TODO different nodes may have different max_tp
        # print("args.max_tp",args.max_tp)
        if args.start_num_stages is None:
            args.start_num_stages = 1
        if args.end_num_stages is None:
            args.end_num_stages = min(args.num_gpus, 16)
        validate_stage_range(args)


    if args.device_json_path is not None:
        with open(args.device_json_path, "r") as f:
            device_json = json.load(f)
            args.device_info = device_json
    # print(args.device_info)
    if args.model_name not in ["gpt", "scale-layer"]:
        raise RuntimeError(f"model {args.model_name} is not supported yet.")

    if args.num_layers is None:
        if args.model_name == "gpt":
            args.num_layers = gpt_configs[args.model_size][0]
        elif args.model_name == "scale-layer":
            raise RuntimeError(f"should provide --num-layers for scale-layer exp")
    # if args.num_layers <= 24:
    #     args.print_recompute_ops = True

    if args.micro_batch_size is None:
        args.micro_batch_size = [1, 2, 4, 8]


    if args.time_budget_per_trial is None   :
        assert args.time_budget_total is not None, "a time budget should be given, with --time-budget-total"
        args.time_budget_per_trial = args.time_budget_total

    args.min_mbs = min(args.micro_batch_size)
    args.num_algos = 2

    if args.model_name == "scale-layer":
        args.memory_main_params = memory_ratio["gpt"]["main_params"]
        args.memory_optimizer = memory_ratio["gpt"]["optimizer"]
    else:
        args.memory_main_params = memory_ratio[args.model_name]["main_params"]
        args.memory_optimizer = memory_ratio[args.model_name]["optimizer"]

    args.resharding = True


    cur_time = datetime.datetime.now()
    args.config_suffix = f"{cur_time.year}-{cur_time.month}-{cur_time.day}-{cur_time.hour}-{cur_time.minute}-{cur_time.second}"
    global_args = args
    return args


def update_args(new_args):
    global global_args
    global_args = new_args
