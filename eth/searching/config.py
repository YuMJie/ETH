# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json

from .device_groups import get_max_tp, gpu_num_list_to_each_stage_gpu_type_num
from .model_ops import get_full_op_list, get_ops_per_layer
from .state import HeterogeneousPlan, SearchConfig, SearchStage

MAX_VALUE = 2**30
MIN_VALUE = -(2**30)

def get_config(num_ops_list, tp_size_list, dp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num):
    op_start_index = 0
    num_stages = len(num_ops_list)
    stages_info_list = []
    for i in range(num_stages):
        stage_info = SearchStage(
            index = i,
            num_stages_behind = (num_stages - 1 - i),
            num_gpus = tp_size_list[op_start_index] * dp_size_list[op_start_index],
            ops = list(full_op_list[op_start_index: op_start_index + num_ops_list[i]]),
            recompute_ops = list(recompute_ops[op_start_index: op_start_index + num_ops_list[i]]),
            tp_size = list(tp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
            dp_size = list(dp_size_list[op_start_index: op_start_index + num_ops_list[i]]),
            algo = list(algo_list[op_start_index: op_start_index + num_ops_list[i]]),
            node_id_num=node_id_num[i]
            )
        stages_info_list.append(stage_info)
        op_start_index += num_ops_list[i]

    current_config = SearchConfig(global_bs=global_batch_size, micro_bs=aggregate_mbs, stages=stages_info_list, num_stages=num_stages)
    return current_config

def config_details(config, get_string=False):
    if config is None:
        return ""
    num_ops_stage = []
    tp_size_list = []
    dp_size_list = []
    recompute_ops = []
    algo_list = []
    base_batch_size = config.micro_bs
    for i in range(config.num_stages):
        num_ops_stage.append(len(config.stages[i].ops))
        tp_size_list.append(config.stages[i].tp_size)
        dp_size_list.append(config.stages[i].dp_size)
        recompute_ops.append(config.stages[i].recompute_ops)
        algo_list.append(config.stages[i].algo)
    if get_string:
        return f"{num_ops_stage}, {tp_size_list}, {dp_size_list}, {recompute_ops}, {base_batch_size}, {algo_list}"
    else:
        return num_ops_stage, tp_size_list, dp_size_list, recompute_ops, base_batch_size, algo_list

def read_config_from_json(args, return_config_dict=False):
    config_file_name = args.initial_point
    with open(config_file_name, "r") as f:
        config_dict = json.load(f)

    model_name = config_dict["model_name"]
    num_layers = config_dict["num_layers"]
    model_size = config_dict["model_size"]

    aggregate_mbs = config_dict["micro_batch_size"]
    global_batch_size = config_dict["global_batch_size"]
    num_ops_list = config_dict["num_ops_in_each_stage"]
    tp_size_list = []
    for _tp_size_list in config_dict["model_parallel_size_of_each_op"]:
        tp_size_list += _tp_size_list
    dp_size_list = []
    for _dp_size_list in config_dict["data_parallel_size_of_each_op"]:
        dp_size_list += _dp_size_list
    recompute_ops = []
    for _recompute_ops in config_dict["recompute_ops"]:
        recompute_ops += _recompute_ops
    algo_list = []
    for _algo_list in config_dict["algo_of_each_op"]:
        algo_list += _algo_list
    full_op_list = get_full_op_list(args)

    num_gpus_list = config_dict["num_gpus"]

    # args.node_order = args.gpu_type_num_dict
    args.num_gpus_per_node_list=[]
    args.gpu_type_list = []
    for key in args.node_order.keys():
        args.num_gpus_per_node_list.append(args.node_order[key]["GPU_NUM"])
        args.gpu_type_list.append(args.node_order[key]["GPU"])
    if(config_dict.get("node_info")!=None):
        args.node_order = config_dict["node_info"]

    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)

    # print(f"node_id_num: {node_id_num}")
    if return_config_dict:
        return get_config(num_ops_list, tp_size_list, dp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num), config_dict
    else:
        return get_config(num_ops_list, tp_size_list, dp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num)

def read_hybrid_config_from_json(args):
    config_file_name = args.initial_point
    with open(config_file_name, "r") as f:
        config_dict = json.load(f)


    model_name = config_dict["model_name"]
    num_layers = config_dict["num_layers"]
    model_size = config_dict["model_size"]
    num_pipelines = len (config_dict["pipe"])
    hybridconfig = HeterogeneousPlan(num_pipelines=num_pipelines)

    args.num_gpus_per_node_list_pipe=[]
    args.gpu_type_list_pipe = []
    for pipe_idx in range(num_pipelines):
        aggregate_mbs = config_dict["pipe"][pipe_idx]["micro_batch_size"]
        global_batch_size = config_dict["pipe"][pipe_idx]["global_batch_size"]
        num_ops_list = config_dict["pipe"][pipe_idx]["num_ops_in_each_stage"]
        tp_size_list = []
        for _tp_size_list in config_dict["pipe"][pipe_idx]["model_parallel_size_of_each_op"]:
            tp_size_list += _tp_size_list
        dp_size_list = []
        for _dp_size_list in config_dict["pipe"][pipe_idx]["data_parallel_size_of_each_op"]:
            dp_size_list += _dp_size_list
        recompute_ops = []
        for _recompute_ops in config_dict["pipe"][pipe_idx]["recompute_ops"]:
            recompute_ops += _recompute_ops
        algo_list = []
        for _algo_list in config_dict["pipe"][pipe_idx]["algo_of_each_op"]:
            algo_list += _algo_list
        full_op_list = get_full_op_list(args)

        num_gpus_list = config_dict["pipe"][pipe_idx]["num_gpus"]

        # args.node_order = args.gpu_type_num_dict
        args.num_gpus_per_node_list=[]
        args.gpu_type_list = []
        for key in args.node_order[pipe_idx].keys():
            args.num_gpus_per_node_list.append(args.node_order[pipe_idx][key]["GPU_NUM"])
            args.gpu_type_list.append(args.node_order[pipe_idx][key]["GPU"])
        args.num_gpus_per_node_list_pipe.append(args.num_gpus_per_node_list)
        args.gpu_type_list_pipe.append(args.gpu_type_list)
        if(config_dict.get("node_info")!=None):
            args.node_order = config_dict["node_info"]

        node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)

        hybridconfig.pipelines.append(get_config(num_ops_list, tp_size_list, dp_size_list, recompute_ops, aggregate_mbs, global_batch_size, full_op_list, algo_list,node_id_num))
        hybridconfig.global_bs_pipeline.append(global_batch_size)
        hybridconfig.micro_bs_pipeline.append(aggregate_mbs)
        hybridconfig.num_mb_pipeline.append(global_batch_size//aggregate_mbs)

    return hybridconfig

def generate_balance_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0]
    flex_recompute = args.flex_recompute

    num_gpus_list = [1 for _ in range(num_stages)]
    stop_flag = True
    while sum(num_gpus_list) < num_gpus and stop_flag:
        stop_flag = False
        for i in range(num_stages):
            # 如果当前stage的gpu数量小于最大gpu数量，且当前stage的gpu数量小于剩余gpu数量
            # 则将当前stage的gpu数量翻倍
            if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < args.max_tp and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)): #TODO use args.max_tp
            # if sum(num_gpus_list) < num_gpus  and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)):
                num_gpus_list[i] *= 2
                stop_flag = True
    if sum(num_gpus_list) != num_gpus:
        print(f"sum(num_gpus_list) != num_gpus,{sum(num_gpus_list)} != {num_gpus}")
        return None

    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)




    num_ops = len(full_op_list)
    num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        if args.init_dim == "tp":
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        elif args.init_dim == "dp":
            dp_size_list += [num_gpus_list[i]//2 for _ in range(num_ops_per_stage[i])]
            tp_size_list += [2 for _ in range(num_ops_per_stage[i])]
    if not flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list,node_id_num)

    return initial_config


def generate_balance_config_v1(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0]
    flex_recompute = args.flex_recompute

    num_gpus_list = [1 for _ in range(num_stages)]
    stop_flag = True
    while sum(num_gpus_list) < num_gpus and stop_flag:
        stop_flag = False
        for i in range(num_stages):
            # 如果当前stage的gpu数量小于最大gpu数量，且当前stage的gpu数量小于剩余gpu数量
            # 则将当前stage的gpu数量翻倍
            # if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < args.max_tp and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)): #TODO use args.max_tp
            if sum(num_gpus_list) < num_gpus  and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)):
                num_gpus_list[i] *= 2
                stop_flag = True
    if sum(num_gpus_list) != num_gpus:
        # print(f"sum(num_gpus_list) != num_gpus,{sum(num_gpus_list)} != {num_gpus}")
        min_stage = num_gpus_list.index(min(num_gpus_list))
        num_gpus_list[min_stage] += num_gpus - sum(num_gpus_list)
        # return

    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)




    num_ops = len(full_op_list)
    num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        if args.init_dim == "tp":
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        elif args.init_dim == "dp":
            dp_size_list += [num_gpus_list[i]//2 for _ in range(num_ops_per_stage[i])]
            tp_size_list += [2 for _ in range(num_ops_per_stage[i])]
    if not flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list,node_id_num)

    return initial_config


def generate_test_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0]

    num_gpus_list = [1 for _ in range(num_stages)]
    num_gpus_left = num_gpus - sum(num_gpus_list)
    while num_gpus_left > 0:
        initial_num_gpus_left = num_gpus_left
        for i in range(num_stages):
            if num_gpus_list[num_stages - 1 - i] <= num_gpus_left and num_gpus_list[num_stages - 1 - i] == min(num_gpus_list) and num_gpus_list[num_stages - 1 - i] <= 4:
                num_gpus_left -= num_gpus_list[num_stages - 1 - i]
                num_gpus_list[num_stages - 1 - i] *= 2
                break
        if num_gpus_left == initial_num_gpus_left:
            break
    if sum(num_gpus_list) != num_gpus:
        return generate_initial_config(full_op_list, num_stages, num_gpus)
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    num_ops = len(full_op_list)
    num_ops_per_stage = []
    for i in range(num_stages):
        num_ops_per_stage.append(int(num_ops * (num_gpus_list[i]/num_gpus)))
    num_ops_per_stage[-1] += num_ops - sum(num_ops_per_stage)

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
        dp_size_list += [1 for _ in range(num_ops_per_stage[i])]

    recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(node_id_num,num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)

    return initial_config

def generate_imbalance_gpu_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0]
    num_ops = len(full_op_list)
    recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]
    ## op distribution
    num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)
    ## gpu distribution
    num_gpus_list = [1 for _ in range(num_stages)]
    num_gpus_remained = num_gpus - sum(num_gpus_list)
    print(f"{num_gpus_list}")
    micro_bs_index = 0
    while num_gpus_remained > 0:
        found = False
        for i in range(num_stages):
            # if num_gpus_list[num_stages - 1 - i] <= num_gpus_remained and \
            #     (num_gpus_list[num_stages - 1 - i] * 2 // args.max_tp == 0 or \
            #     micro_bs // (num_gpus_list[num_stages - 1 - i] * 2 // args.max_tp) in args.micro_batch_size):
            node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
            max_tp_group = get_max_tp(args,None,node_id_num[i])
            if num_gpus_list[num_stages - 1 - i] <= num_gpus_remained and \
                (num_gpus_list[num_stages - 1 - i] * 2 // max_tp_group == 0 or \
                micro_bs // (num_gpus_list[num_stages - 1 - i] * 2 // max_tp_group) in args.micro_batch_size):
                num_gpus_remained -=  num_gpus_list[num_stages - 1 - i]
                num_gpus_list[num_stages - 1 - i] *= 2
                found = True
                print(f"update: {num_gpus_list} (inc gpus in stage {num_stages - 1 - i})")
                break
            else:
                print(f"fail to update on stage {num_stages - 1 - i}")
        if not found:
            micro_bs_index += 1
            assert micro_bs_index < len(args.micro_batch_size)
            micro_bs =  args.micro_batch_size[micro_bs_index]
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
        # if num_gpus_list[i] <= args.max_tp:
        if num_gpus_list[i] <= get_max_tp(args,None,node_id_num[i]):
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        else:
            # tp_size_list += [args.max_tp for _ in range(num_ops_per_stage[i])]
            tp_size_list += [get_max_tp(args,None,node_id_num[i]) for _ in range(num_ops_per_stage[i])]
            # dp_size_list += [num_gpus_list[i] // args.max_tp for _ in range(num_ops_per_stage[i])]
            dp_size_list += [num_gpus_list[i] // get_max_tp(args,None,node_id_num[i]) for _ in range(num_ops_per_stage[i])]

    initial_config = get_config(node_id_num,num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)

    return initial_config

def generate_imbalance_op_config(full_op_list, num_stages, args):
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0]
    flex_recompute = args.flex_recompute

    num_gpus_list = [1 for _ in range(num_stages)]
    stop_flag = True
    while sum(num_gpus_list) < num_gpus and stop_flag:
        stop_flag = False
        for i in range(num_stages):
            # if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < args.max_tp and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)):
            node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
            max_tp_group = get_max_tp(args,None,node_id_num[i])
            if sum(num_gpus_list) < num_gpus and num_gpus_list[i] < max_tp_group and num_gpus_list[i] <= (num_gpus - sum(num_gpus_list)):
                num_gpus_list[i] *= 2
                stop_flag = True
    if sum(num_gpus_list) != num_gpus:
        return None
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    if args.model_name in ["gpt", "scale-layer"]:
        num_ops = len(full_op_list)
        ops_per_layer = get_ops_per_layer(args)
        num_layers_list = [1 for _ in range(num_stages)]
        num_layers_list[-1] += args.num_layers - sum(num_layers_list)

        num_ops_per_stage = [
            num_layers_list[i] * ops_per_layer for i in range(num_stages)
        ]
        num_ops_per_stage[0] += 1
        num_ops_per_stage[-1] += 2

    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
        dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
    if flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    initial_config = get_config(node_id_num,num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list)

    return initial_config

#deprecated
def generate_initial_config(num_stages, node_order_idx,args):
    full_op_list = get_full_op_list(args)
    if args.initial_point == "balance":
        return generate_balance_config(full_op_list, num_stages, args)
    elif args.initial_point == "imbalance_gpu":
        return generate_imbalance_gpu_config(full_op_list, num_stages, args)
    elif args.initial_point == "imbalance_op":
        return generate_imbalance_op_config(full_op_list, num_stages, args)
    elif args.initial_point == "test":
        return generate_test_config(full_op_list, num_stages, args)
    elif args.initial_point == "balance_v1":
        return generate_balance_config_v1(full_op_list, num_stages, args)
    else:
        with open(args.initial_point, "r") as f:
            config_dict = json.load(f)
        if(config_dict["num_stages"] != num_stages):
            args.initial_point = "balance_v1"
            return generate_initial_config(num_stages,node_order_idx, args)
        elif(args.config_node_order_idx == node_order_idx):
            print(f"read initial point from {args.initial_point}")
            # print(f"read_config_from_json {read_config_from_json(args)}")
            return read_config_from_json(args)
        else:
            args.initial_point = "balance_v1"
            return generate_initial_config(num_stages,node_order_idx, args)


def generate_device_group_initial_config(num_stages, node_order_idx, device_group, args):
    full_op_list = get_full_op_list(args)
    num_gpus = args.num_gpus
    global_bs = args.global_batch_size
    micro_bs = args.micro_batch_size[0]
    flex_recompute = args.flex_recompute
    num_gpus_list = device_group
    node_id_num = gpu_num_list_to_each_stage_gpu_type_num(args, num_gpus_list)
    num_ops = len(full_op_list)
    # num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
    # num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)
    # print("args.model_name ",args.model_name )
    # print("args.op_grain ",args.op_grain )
    if(args.model_name == "gpt" and args.op_grain == "layer"):
        ops_per_layer = get_ops_per_layer(args)
        num_ops_per_stage = [0] * num_stages
        for i in range(len(num_ops_per_stage)):
            num_ops_per_stage[i] = (
                num_ops // num_stages // ops_per_layer * ops_per_layer
            )
        num_ops_per_stage[0]+=1
        # num_ops_per_stage[-1] +=num_ops - sum(num_ops_per_stage)
        # print("num_ops_per_stage",num_ops_per_stage)
        num_ops_per_stage[-1] += 2
        num_ops_temp = num_ops - sum(num_ops_per_stage)
        # print("num_ops_temp",num_ops_temp)
        # num_ops_per_stage[-1] += num_ops_temp

        while(num_ops_temp>0):
            for i in range(len(num_ops_per_stage)-1,-1,-1):
                # print("num_ops_temp",num_ops_temp)
                if(num_ops_temp>0):
                    num_ops_per_stage[i] += ops_per_layer
                    num_ops_temp -= ops_per_layer
                else:
                    break

        print("num_ops_per_stage",num_ops_per_stage)
        print("num_ops",num_ops)
        # print("sum(num_ops_per_stage)",sum(num_ops_per_stage))
    else:
        num_ops_per_stage = [num_ops // num_stages for _ in range(num_stages)]
        num_ops_per_stage[-1] += num_ops - num_stages * (num_ops//num_stages)
    tp_size_list = []
    dp_size_list = []
    for i in range(num_stages):
        if args.init_dim == "tp":
            tp_size_list += [num_gpus_list[i] for _ in range(num_ops_per_stage[i])]
            dp_size_list += [1 for _ in range(num_ops_per_stage[i])]
        elif args.init_dim == "dp":
            dp_size_list += [num_gpus_list[i]//2 for _ in range(num_ops_per_stage[i])]
            tp_size_list += [2 for _ in range(num_ops_per_stage[i])]
    micro_bs =  dp_size_list [0]
    if not flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
    else:
        recompute_ops = [0 for _ in range(num_ops)]
    algo_list = [0 for _ in range(num_ops)]

    # print("num_ops_per_stage",num_ops_per_stage)
    # print("dp_size_list",dp_size_list)
    # print("tp_size_list",tp_size_list)
    # print("num_ops",num_ops)
    initial_config = get_config(num_ops_per_stage, tp_size_list, dp_size_list, recompute_ops, micro_bs, global_bs, full_op_list, algo_list,node_id_num)
    return initial_config


def sort_configs(config_list, sort_metric):
    '''
    按照sort_metric对config_list进行从小到大排序
    使用的sort算法是插入排序
    '''
    new_list = []

    if sort_metric == "max_stage_time":
        for i in range(len(config_list)):
            max_stage_time = max(config_list[i].time_list)
            if len(new_list) == 0:
                new_list.append(config_list[i])
            else:
                j = 0
                inserted = False
                for j in range(len(new_list)):
                    if max_stage_time < max(new_list[j].time_list):
                        new_list.insert(j, config_list[i])
                        inserted = True
                        break
                if not inserted:
                    new_list.append(config_list[i])
    elif sort_metric == "total_gpu_time":
        for i in range(len(config_list)):
            gpu_time = config_list[i].total_gpu_time
            if len(new_list) == 0:
                new_list.append(config_list[i])
            else:
                j = 0
                inserted = False
                for j in range(len(new_list)):
                    if gpu_time < new_list[j].total_gpu_time:
                        new_list.insert(j, config_list[i])
                        inserted = True
                        break
                if not inserted:
                    new_list.append(config_list[i])
    else:
        raise RuntimeError(f"sort_metric {sort_metric} not supported.")
    return new_list

def get_boundary_list(args):
    num_gpus_at_boundary_list = []
    for i in range(0,args.num_nodes):
        if(i):
            num_gpus_at_boundary_list.append(args.num_gpus_per_node_list[i] + num_gpus_at_boundary_list[i-1])
        else:
            num_gpus_at_boundary_list.append(args.num_gpus_per_node_list[0])
    return num_gpus_at_boundary_list





def check_legality(config, args):
    num_gpus_from_start = []
    num_gpus = 0
    for i in range(config.num_stages):
        if config.stages[i].num_gpus not in [1, 2, 4, 8]:
            return False
        num_gpus += config.stages[i].num_gpus
        num_gpus_from_start.append(num_gpus)
    if num_gpus != args.num_gpus:
        return False
    # num_gpus_at_boundary_list = [args.num_gpus_per_node * i for i in range(1, args.num_nodes + 1)]

    num_gpus_at_boundary_list = get_boundary_list(args)

    for num_gpus_at_boundary in num_gpus_at_boundary_list:
        if num_gpus_at_boundary not in num_gpus_from_start:
            return False
    return True

def format_size_list(size_list):
    output_string = "["
    for i in range(len(size_list)):
        max_val = max(size_list[i])
        min_val = min(size_list[i])
        if min_val == max_val:
            output_string += f"{max_val}, "
        else:
            output_string += f"{min_val}~{max_val}, "
    output_string += "]"
    return output_string

def is_visited(visited_set, hash_str, target=""):
    if hash_str not in visited_set:
        return False
    else:
        if target in visited_set[hash_str]:
            return True
        else:
            return False

def mark_visited(visited_set, hash_str, target=""):
    if hash_str not in visited_set:
        visited_set[hash_str] = [target]
    else:
        visited_set[hash_str].append(target)

def num_visited(visited_set, hash_str):
    if hash_str not in visited_set:
        return  0
    else:
        return len(visited_set[hash_str])
