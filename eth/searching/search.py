# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .cost_model import predict_time_breakdown, update_recompute, get_reserved_memory_list
from multiprocessing import Process, Queue
import multiprocessing
import itertools
from .state import HeterogeneousPlan
from .config import (
    MAX_VALUE,
    config_details,
    generate_initial_config,
    generate_device_group_initial_config,
    is_visited,
    mark_visited,
    sort_configs,
)
from .arguments import parse_args
from .device_groups import (
    check_memory_legacy,
    enumerate_homogeneous_stage_sizes,
    enumerate_node_aligned_stage_sizes,
    generate_valid_permutations,
    get_results_for_ks,
    get_upper_bound,
    precompute_all_k,
    sort_pipeline_split_list,
    split_devices_to_pipeline_large,
    split_to_cache_result,
)
from .reporting import (
    debug_info,
    print_args,
    print_search_details,
    print_simple_config_info,
    save_and_print_top_configs_dp_hete,
    save_and_print_top_hybrid_config,
    timers,
)
from .serialization import dump_config_to_json
from .actions import action_resource_table, finetune_dim_stage_level, finetune, get_explored_cases, initialize_primitives, reset_explored_cases,get_mean_op_success_bisection_count,get_mean_op_success_linear_count
from .policy import (
    get_actions,
    get_hit_resources,
    get_target_stage,
    reset_hit_resources,
    reset_visited_partners,
)
import copy
import time
import os
import csv
from .actions import reset_move_count, get_move_count
from . import state
# read_profiled_time(args.model_name, args.model_size, args.profiled_time_path)
# get_sort_op_by_size_comp(args)
config_visited = {}
MIN_TIME_GET_BEST_CONFIG = 0
def initialize_search(num_stages,node_order_idx,args):
    global current_min_time, unexplored_configs, explored_configs
    print(f"working on num_stages = {num_stages}")
    config = generate_initial_config(num_stages,node_order_idx, args)  # 根据args生成一个config
    if config is not None:
        update_recompute(config) # 根据内存大小更新config的recompute
        predict_time_breakdown(config,args=args,print_memory=True)    # 预测config的时间
        print_simple_config_info(config, info="start", print_debug_info=args.print_debug_info, add_history=True)
        # if max(config.memory_list) < args.memory_limit:
        if check_memory_legacy(args,config):
            current_min_time = max(config.time_list)
        else:
            current_min_time = MAX_VALUE

        reset_explored_cases()
        reset_hit_resources()
        unexplored_configs = []
        explored_configs = []
    return config

def initialize_search_manual(num_stages,node_order_idx,device_group,args):
    global current_min_time, unexplored_configs, explored_configs
    print(f"working on num_stages = {num_stages}")
    config = generate_device_group_initial_config(
        num_stages, node_order_idx, device_group=device_group, args=args
    )
    if config is not None:
        update_recompute(config) # 根据内存大小更新config的recompute
        predict_time_breakdown(config,args=args,print_memory=False)    # 预测config的时间
        print_simple_config_info(config, info="start", print_debug_info=args.print_debug_info, add_history=True)
        # print("init time list", config.time_list)
        # print("init time list", config.each_stage_time_breakdown)
        # if max(config.memory_list) < args.memory_limit:
        if check_memory_legacy(args,config):
            current_min_time = max(config.time_list)
        else:
            current_min_time = MAX_VALUE

        reset_explored_cases()
        reset_hit_resources()
        unexplored_configs = []
        explored_configs = []
    return config


def take_action(config, target_stage, prim,args=None):
    timers("prim_" + prim.name).start()
    new_config = prim.apply_to(config, target_stage,args)# 对config进行prim操作
    unvisited_configs = []
    if new_config is not None:
        # debug_info(f">>>> {prim.name} success.", args.print_debug_info)
        if not is_visited(config_visited, hash_str=config_details(new_config, get_string=True)):
            timers("prim_predict_time_breakdown2").start()
            predict_time_breakdown(new_config,args=args) # 预测新的config的时间
            timers("prim_predict_time_breakdown2").stop()
            unvisited_configs.append(new_config)
            # print_simple_config_info(new_config, info=f"config after action {prim.name} is not visited.", print_debug_info=args.print_debug_info)
        else:
            # if(max(new_config.time_list)-5920.0<1):
            #     #save to json
            #     print("save to json")
            #     dump_config_to_json(new_config, f'{args.config_save_path}{args.model_name}_{args.model_size}_{new_config.num_stages}stages_{args.config_suffix}_test.json', args)
            print_simple_config_info(new_config, info=f"config after action {prim.name} is visited.", print_debug_info= args.print_debug_info)
    # else:
    #     debug_info(f">>>> {prim.name} failed.", args.print_debug_info)
    timers("prim_" + prim.name).stop()
    return unvisited_configs

unexplored_configs = []
explored_configs = []
def get_candidate_config(candidate_set):
    if len(candidate_set) == 0:
        return None

    min_time = MAX_VALUE
    best_config = None
    for config in candidate_set:
        # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit:
        if max(config.time_list) < min_time and (check_memory_legacy(state.args,config)==True):
            min_time = max(config.time_list)
            best_config = config
    if best_config is None:
        min_memory = MAX_VALUE
        for config in candidate_set:
            if max(config.memory_list) < min_memory:
                min_memory = max(config.memory_list)
                best_config = config
    candidate_set.remove(best_config)
    return best_config

def get_adaptive_config(candidate_set):
    if len(candidate_set) == 0:
        return None

    min_time = MAX_VALUE
    best_config = None
    for config in candidate_set:
        # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit and config.adaptive_times < args.adaptive_hyper_parameters:
        if max(config.time_list) < min_time and (check_memory_legacy(state.args,config)==True) and config.adaptive_times < state.args.adaptive_hyper_parameters:
            min_time = max(config.time_list)
            best_config = config
    if best_config is None:
        min_memory = MAX_VALUE
        for config in candidate_set:
            if max(config.memory_list) < min_memory and config.adaptive_times < state.args.adaptive_hyper_parameters:
                min_memory = max(config.memory_list)
                best_config = config
    if best_config is not None:
        best_config.adaptive_times += 1
    return best_config

def multi_hop_search(config, hop_index, initial_time, adaptive_flag,args=None):
    global current_min_time
    ## reach hop limit
    if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial or \
        timers("total-time").elapsed_since_first_invoke() >= args.time_budget_total or \
        hop_index == args.max_num_hops:
        return None, None
    mark_visited(config_visited, hash_str=config_details(config, get_string=True))
    if not adaptive_flag:
        if args.adaptive_hyper_parameters > 0:
            explored_configs.append(config)
        if args.continue_when_fail and config in unexplored_configs:
            unexplored_configs.remove(config)
    ## get bottleneck 获得当前config的bottleneck的stage
    bottleneck = get_target_stage(config, adaptive_flag=adaptive_flag)
    if bottleneck is None:
        return None, None
    # 根据bottleneck获得可以解决bottleneck的actions

    '''
    action_resource_table = [
    SearchAction(name = "dec_op",  time = "-", memory = "-", num_devices = "0", workloads = "-", efficiency = "0", comm = "0", func = prim_mig_op if not args.simple_prim_mig else prim_mig_op_simple),
    SearchAction(name = "inc_dp",  time = "-", memory = "-", num_devices = "+", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp),
    SearchAction(name = "dec_dp",  time = "+", memory = "+", num_devices = "-", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp),
    SearchAction(name = "inc_tp",  time = "-", memory = "-", num_devices = "+", workloads = "0", efficiency = "-", comm = "+", func = prim_tp_dp),
    SearchAction(name = "dec_tp",  time = "+", memory = "+", num_devices = "-", workloads = "0", efficiency = "+", comm = "-", func = prim_tp_dp),
    SearchAction(name = "inc_mbs", time = "-", memory = "+", num_devices = "0", workloads = "0", efficiency = "+", comm = "0", func = prim_mbs),
    SearchAction(name = "dec_mbs", time = "+", memory = "-", num_devices = "0", workloads = "0", efficiency = "-", comm = "0", func = prim_mbs)
]'''
    actions_all, turn_back_actions, stage_type = get_actions(config, bottleneck, action_resource_table, adaptive_flag=adaptive_flag)
    print_simple_config_info(config, info=f">>", print_debug_info=args.print_debug_info)
    debug_info(f">> hop_index {hop_index}, target_stage ({bottleneck}) [{stage_type}] actions: {[[prim.name for prim in actions_] for actions_ in actions_all]}. turn back actions: {turn_back_actions}", args.print_debug_info)

    config_list  = [config]
    for actions in actions_all:
        new_configs_all = []
        for action in actions:
            if timers("trial-time").elapsed_since_first_invoke() >= args.time_budget_per_trial or timers("total-time").elapsed_since_first_invoke() >= args.time_budget_total:
                break
            action_succeed_configs = take_action(config, bottleneck, action,args)
            # print("len(action_succeed_configs)", len(action_succeed_configs))
            for succeed_config in action_succeed_configs: # only one config or no config
                new_configs_all.append(succeed_config)
                if(args.print_debug_info):
                    print_simple_config_info(succeed_config, info=f">>>> [CONFIG: target({bottleneck}), prim({action.name})]\n", print_debug_info=args.print_debug_info, add_history=True)

        min_time = initial_time
        min_time_config = None
        for _config in new_configs_all:
            _config_time = max(_config.time_list)
            _config_memory = max(_config.memory_list)
            if args.continue_when_fail and _config_time < current_min_time * 1.2: # 如果新的config的时间小于当前最小时间的1.2倍 则加入unexplored_configs
                unexplored_configs.append(_config)
            # if _config_time < min_time and _config_memory <= args.memory_limit: # 如果新的config的时间小于最小时间并且内存小于内存限制 则更新最小时间和最小时间的config
            if _config_time < min_time and check_memory_legacy(args,_config)==True:
                global MIN_TIME_GET_BEST_CONFIG
                # print("config_time",_config_time, "MIN_TIME_GET_BEST_CONFIG",MIN_TIME_GET_BEST_CONFIG)
                MIN_TIME_GET_BEST_CONFIG = timers("total-time").elapsed_since_first_invoke()
                min_time = _config_time
                min_time_config = _config

        if min_time_config is not None: # 当有一个action能够找到一个更优解时
            return min_time_config, hop_index + 1
        else: # 当没有一个action能够找到一个更优解时 则继续搜索
            _new_configs_all = sort_configs(new_configs_all, args.sort_metric)# 使用插入排序对新的config按照max_stage_time进行排序
            for _config in _new_configs_all:
                config_list.append(_config)
                next_config, next_config_hop_index = multi_hop_search(_config, hop_index + 1, initial_time, adaptive_flag=False,args=args) # 递归搜索
                if next_config is not None:
                    # if max(next_config.time_list) < initial_time and max(next_config.memory_list) <= args.memory_limit:
                    if max(next_config.time_list) < initial_time and check_memory_legacy(args,next_config)==True:
                        return next_config, next_config_hop_index
                    else:
                        config_list.append(next_config)

    if args.continue_when_fail:
        min_time = MAX_VALUE
        best_config = None
        for config in config_list:
            # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit:
            if max(config.time_list) < min_time and check_memory_legacy(args,config)==True:
                min_time = max(config.time_list)
                best_config = config
        if best_config is None:
            min_memory = MAX_VALUE
            for config in config_list:
                if max(config.memory_list) < min_memory:
                    min_memory = max(config.memory_list)
                    best_config = config
        return best_config, args.max_num_hops
    else:
        return None, None

def trial(config, num_trial, adaptive_flag, initial_time=None,args=None):
    global timers
    timers("trial-time").start()
    reset_visited_partners()

    if initial_time is None:
        # if max(config.memory_list) > args.memory_limit: # 如果config的内存大于内存限制,即没有合适的config
        if check_memory_legacy(args,config)==False:
            initial_time = MAX_VALUE # 初始时间为最大值
        else:
            initial_time = max(config.time_list) # 否则初始时间为config的时间

    if args.only_top_1_target:
        max_num_targets = 1
    else:
        max_num_targets = config.num_stages

    num_targets = 0
    config_list = [config]
    while num_targets < max_num_targets:
        print_simple_config_info(config, info=f">>[target# {num_targets}/{max_num_targets}]", print_debug_info=args.print_debug_info)
        num_targets += 1
        new_config, hop_index = multi_hop_search(config, 0, initial_time, adaptive_flag,args=args) # 多跳搜索 hop_index表示搜索到该config的深度

        if new_config is not None:
            config_list.append(new_config)

            new_time = max(new_config.time_list)
            new_memory = max(new_config.memory_list)

            dec_time_gap = initial_time - new_time
            # if dec_time_gap > 0 and new_memory <= args.memory_limit: # 如果新的时间小于初始时间并且内存小于内存限制 则返回新的config
            if dec_time_gap > 0 and check_memory_legacy(args,new_config)==True:

                timers("trial-time").reset()
                return new_config, num_targets, hop_index

    timers("trial-time").reset()

    min_time = MAX_VALUE
    best_config = None
    for config in config_list:
        # if max(config.time_list) < min_time and max(config.memory_list) <= args.memory_limit: # 如果config的时间小于最小时间并且内存小于内存限制 则返回config
        if max(config.time_list) < min_time and check_memory_legacy(args,config)==True:
            min_time = max(config.time_list)
            best_config = config
    if best_config is None: #如果没有找到合适的config 则返回内存最小的config

        min_memory = MAX_VALUE
        for config in config_list:
            if max(config.memory_list) < min_memory:
                min_memory = max(config.memory_list)
                best_config = config

    return best_config, max_num_targets, args.max_num_hops + 1

global profile_time_global
profile_time_global = None

def run_search(num_stages, node_order_idx, device_group_idx, profile_time ,shared_dict=None,args_=None):
    # global args
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band
    (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band )= profile_time
    global current_min_time
    state.args = args_
    state.set_profiled_time(profile_time)
    args =  state.args
    # print("global_args.gpu_type_set", global_args.gpu_type_set)
    # print("args.gpu_type_set", args.gpu_type_set)
    # print("args_.gpu_type_set", args_.gpu_type_set)
    print("args.gpu_type_set1", args.gpu_type_set)

    state.get_sort_op_by_size_comp(args)

    print("args.gpu_type_set2", args.gpu_type_set)

    global profile_time_global
    profile_time_global = profile_time
    args.node_order = args.node_order_list[node_order_idx]
    print("args.node_order ", args.node_order)


    device_group = args.total_device_group_list[num_stages][device_group_idx]
    print("device_group ", device_group)

    args.num_gpus_per_node_list=[]
    args.gpu_type_list = []
    for key in args.node_order.keys():
        args.num_gpus_per_node_list.append(args.node_order[key]["GPU_NUM"])
        args.gpu_type_list.append(args.node_order[key]["GPU"])
    config = initialize_search_manual(num_stages,node_order_idx,device_group,args)
    if config is None:
        debug_info(f"No feasible solution for # stage {num_stages}", args.print_debug_info)
        return None
    timers("total-time").start()
    print(f"args.time_budget_total {args.time_budget_total}")
    current_memory = max(config.memory_list)
    temp_config = config
    best_config = None
    num_trial = 0
    search_time_list = [0]
    config_time_list = [current_min_time]
    num_targets_list = []
    num_hops_list = []

    # if max(config.memory_list) < args.memory_limit:
    if check_memory_legacy(args,config):
        best_config = config

    adaptive_flag = False
    # 当num_trial小于最大尝试次数并且搜索时间小于总时间预算时
    while num_trial < args.max_num_trials and sum(search_time_list) < args.time_budget_total:
        trial_start_time = time.time()
        if(args.print_debug_info):
            print_simple_config_info(config, info=f"\n[ Trial {num_trial} ] ", print_debug_info=args.print_debug_info, add_history=True)
        # 进行一次trial
        new_config, num_targets, num_hops = trial(config, num_trial, adaptive_flag, current_min_time,args=args)
        if(new_config == None):
            return
        memory_list = new_config.memory_list
        each_stage_memory_breakdown = new_config.each_stage_memory_breakdown
        assert memory_list[0] == each_stage_memory_breakdown[0][0], f"memory_list[0] != each_stage_memory_breakdown[0][0] {memory_list[0]} != {each_stage_memory_breakdown[0][0]}"
        trial_end_time = time.time()
        search_time_list.append(trial_end_time - trial_start_time)
        if args.finetune_after_trial > 0:
            for _ in range(args.finetune_after_trial):
                # 对新的config进行finetune
                new_config = finetune(new_config,args=args)
                if(args.print_debug_info):
                    print_simple_config_info(new_config, info=f">>>> [TMP CONFIG : (after finetune):\n", print_debug_info=args.print_debug_info, add_history=True)
        if args.finetune_tp_dp_after_trial:
            new_config = finetune_dim_stage_level(new_config)
            if(args.print_debug_info):
                print_simple_config_info(new_config, info=f">>>> [TMP CONFIG : (after tune tp/dp) :\n", print_debug_info=args.print_debug_info, add_history=True)
        memory_list = new_config.memory_list
        each_stage_memory_breakdown = new_config.each_stage_memory_breakdown
        assert memory_list[0] == each_stage_memory_breakdown[0][0], f"1memory_list[0] != each_stage_memory_breakdown[0][0] {memory_list[0]} != {each_stage_memory_breakdown[0][0]}"
        new_time = max(new_config.time_list)
        new_memory = max(new_config.memory_list)
        config_time_list.append(new_time)

        adaptive_flag = False # 适应性标志 用于判断是否进入适应性模式 只有在没有找到更优解时才会进入适应性模式
        # 如果新的config的时间小于当前最小时间并且内存小于内存限制 或者 当前内存大于内存限制并且新的内存小于当前内存
        # if (new_time < current_min_time and new_memory <= args.memory_limit) or \
        #     (current_memory > args.memory_limit and new_memory < current_memory):


        if (new_time < current_min_time and check_memory_legacy(args,new_config)==True) or \
            (check_memory_legacy(args,temp_config)==False and new_memory < current_memory):

            if(new_time < current_min_time and check_memory_legacy(args,new_config)==True):
                best_config = copy.deepcopy(new_config)
            current_memory = new_memory
            config = new_config
            if num_hops <= args.max_num_hops:
                num_targets_list.append(num_targets)
                num_hops_list.append(num_hops)
            # if new_memory <= args.memory_limit:
            if check_memory_legacy(args,new_config):
                current_min_time = min(new_time, current_min_time)
        else:
            if args.continue_when_fail:
                # print("unexplored_configs", len(unexplored_configs))
                timers("candidate-time").start()
                config = get_candidate_config(unexplored_configs)
                timers("candidate-time").stop()
                if config is None:
                    debug_info(f"[trail {num_trial}] config is None. enter adaptive mode.", args.print_debug_info)
                    if args.adaptive_hyper_parameters > 0:
                        adaptive_flag = True # 进入适应性模式 因为没有找到更优解
                        timers("fail-time").start()
                        config = get_adaptive_config(explored_configs)
                        timers("fail-time").stop()
                        if config is None:
                            timers("fail-fail-time").start()
                            timers("fail-fail-time").stop()

                            break
                    else:
                        break
            else:
                break
        num_trial += 1
        debug_info(f"[current best time] = {current_min_time}, num_explored_cases = {get_explored_cases()}", args.print_debug_info)
    if best_config is not None and not args.test_search_time:
        best_config.node_info = args.node_order_list[node_order_idx]
        print("best_config.time_list",max(best_config.time_list))
        # memory_list = best_config.memory_list
        # each_stage_memory_breakdown = best_config.each_stage_memory_breakdown
        # assert memory_list[0] == each_stage_memory_breakdown[0][0], f"2memory_list[0] != each_stage_memory_breakdown[0][0] {memory_list[0]} != {each_stage_memory_breakdown[0][0]}"
        dump_config_to_json(best_config, f'{args.config_save_path}{args.model_name}_{args.model_size}_{best_config.num_stages}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json', args)
    print_search_details(best_config, args, num_stages, node_order_idx,device_group_idx,num_targets_list, num_hops_list, search_time_list, config_time_list, get_reserved_memory_list(best_config), get_explored_cases())
    timers("total-time").reset()

    print(f"The search time for node_order_idx {node_order_idx}   num_stages {num_stages} device_group_idx {device_group_idx}  is {timers('total-time').elapsed_since_first_invoke()} s")
    # timers.log()

    config_mem = 0
    if best_config is not None:
        config_mem = max(best_config.memory_list)
    else:
        return None #fix 多线程卡死问题
    # print("OP_SUCCESS_LINEAR_COUNT",get_mean_op_success_linear_count())
    # print("OP_SUCCESS_BISECTION_COUNT",get_mean_op_success_bisection_count())



    assert current_min_time == max(best_config.time_list), f"current_min_time != max(best_config.time_list) {current_min_time} != {max(best_config.time_list)}"
    if(not args.reduce_output and not args.test_search_time):
        print(f"num_ops_in_each_stage {[ len(stage.ops) for stage in best_config.stages]} ")
        print("\n")
        print("best_config.each_stage_time_breakdown [total (/1000), fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]")
        for i in range(best_config.num_stages):
            print([best_config.each_stage_time_breakdown[i][time_idx]/1000 if time_idx else best_config.each_stage_time_breakdown[i][time_idx]  for time_idx in range(len(best_config.each_stage_time_breakdown[i]))])
        print("\n")
        print("best_config.each_stage_memory_breakdown [total (MB) memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]")
        for i in range(best_config.num_stages):
            print(best_config.each_stage_memory_breakdown[i])

        print(f"add result to queue {node_order_idx} {num_stages} {device_group_idx}")

    print(f"best_config.time_list: {best_config.time_list}")
    print(f"MIN_TIME_GET_BEST_CONFIG: {MIN_TIME_GET_BEST_CONFIG}")
    print("====================================================================================================")

    if(not args.test_search_time):

        shared_dict[f"{node_order_idx}_{num_stages}_{device_group_idx}"] = (
            max(best_config.time_list),
            config_mem,
            get_explored_cases(),
            sum(search_time_list),
            get_hit_resources(),
            best_config.adaptive_times,
            best_config.time_list,
            best_config.memory_list,
            best_config.breakdown_eff_loss_time_per_gpu,
            best_config.breakdown_recomp_time_per_gpu,
            best_config.each_stage_time_breakdown,
            best_config.each_stage_memory_breakdown,
            MIN_TIME_GET_BEST_CONFIG,
            best_config
        )

        return None
    # return current_min_time, config_mem, get_explored_cases(), sum(search_time_list), get_hit_resources(),best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu,best_config.each_stage_time_breakdown,best_config.each_stage_memory_breakdown,MIN_TIME_GET_BEST_CONFIG,config


def workload_balance_pipeline(config : HeterogeneousPlan):

    global_batch_size = config.global_bs_pipeline[0]
    micro_bs_pipeline = config.micro_bs_pipeline
    num_sample_each_pipe = micro_bs_pipeline.copy()
    unallocated_num_sample = global_batch_size - sum(num_sample_each_pipe)

    print("config.time_list_pipeline",config.time_list_pipeline)

    time_each_pipe = []
    for pipe_config in config.pipelines:
        time_each_pipe.append(pipe_config.sum_stage_time /1000)
    # print("time_each_pipe1",time_each_pipe)
    # print("unallocated_num_sample",unallocated_num_sample)
    # print("global_batch_size",global_batch_size)

    while unallocated_num_sample > 0:

        min_time = MAX_VALUE
        for i in range(len(time_each_pipe)):
            if (time_each_pipe[i] < min_time) and (unallocated_num_sample >= micro_bs_pipeline[i]):
                min_time = time_each_pipe[i]
                min_time_idx = i

        num_sample_each_pipe[min_time_idx] +=  micro_bs_pipeline[min_time_idx]
        unallocated_num_sample -= micro_bs_pipeline[min_time_idx]

        time_each_pipe[min_time_idx] += max(config.pipelines[min_time_idx]._time_list)/1000
        # print(f" max(config.pipelines[min_time_idx]._time_list) {max(config.pipelines[min_time_idx]._time_list)}")
        # print(f" unallocated_num_sample {unallocated_num_sample}")
        # print(f" num_sample_each_pipe {num_sample_each_pipe}")
        # print(f" micro_bs_pipeline[min_time_idx] {micro_bs_pipeline[min_time_idx]}")
    # print("time_each_pipe2",time_each_pipe)

    config.time_list_pipeline = time_each_pipe
    config.global_bs_pipeline =  num_sample_each_pipe

    for i  in range(config.num_pipelines):
        config.pipelines[i].global_bs = num_sample_each_pipe[i]
        for j in range(config.pipelines[i].num_stages):
            config.pipelines[i].time_list[j] = config.pipelines[i].sum_stage_time + (config.pipelines[i].global_bs//config.pipelines[i].micro_bs-1 )*config.pipelines[i]._time_list[j]
            config.pipelines[i].time_list[j] /= 1000
        # print(f"config.pipelines[{i}].time_list",config.pipelines[i].time_list)
        # predict_time_breakdown(config.pipelines[i],args=args)
        # print(f"config.pipelines[{i}].time_list",config.pipelines[i].time_list)
        # print(f"config.pipelines[{i}].sum_stage_time",config.pipelines[i].sum_stage_time)
        # print(f"config.pipelines[{i}]._time_list",config.pipelines[i]._time_list)
        # print(f"config.pipelines[{i}].time_list",config.pipelines[i].time_list)
        # print(f"config.pipelines[{i}].global_bs",config.pipelines[i].global_bs)
        # print(f"config.pipelines[{i}].micro_bs",config.pipelines[i].micro_bs)
        # print(f"time_each_pipe[{i}]",time_each_pipe[i])
    # print("config.time_list_pipeline",config.time_list_pipeline)
    # exit()
    return config




def workload_balance_pipeline_v1(config : HeterogeneousPlan):

    global_batch_size = config.global_bs_pipeline[0]
    micro_bs_pipeline = config.micro_bs_pipeline
    num_sample_each_pipe = micro_bs_pipeline.copy()
    unallocated_num_sample = global_batch_size - sum(num_sample_each_pipe)

    print("config.time_list_pipeline",config.time_list_pipeline)

    time_each_pipe = []
    for pipe_config in config.pipelines:
        time_each_pipe.append(pipe_config.sum_stage_time /1000)
    # print("time_each_pipe1",time_each_pipe)
    # print("unallocated_num_sample",unallocated_num_sample)
    # print("global_batch_size",global_batch_size)

    while unallocated_num_sample > 0:

        time_each_pipe_temp = time_each_pipe.copy()
        for i in range(len(time_each_pipe)):
            time_each_pipe_temp[i] += max(config.pipelines[i]._time_list)/1000

        min_time_idx = time_each_pipe_temp.index(min(time_each_pipe_temp))

        num_sample_each_pipe[min_time_idx] +=  micro_bs_pipeline[min_time_idx]
        unallocated_num_sample -= micro_bs_pipeline[min_time_idx]

        time_each_pipe[min_time_idx] += max(config.pipelines[min_time_idx]._time_list)/1000


    min_mbs = min(micro_bs_pipeline)
    min_mbs_idx = micro_bs_pipeline.index(min_mbs)
    # if(unallocated_num_sample < 0 ):
    #     if( ((-unallocated_num_sample) % min_mbs) == 0):
    #         time_each_pipe[min_mbs_idx]-= max(config.pipelines[min_mbs_idx]._time_list)/1000 * ((-unallocated_num_sample) // min_mbs)
    #         num_sample_each_pipe[min_mbs_idx]  +=  unallocated_num_sample
    #     else:
    #         #TODO
    #         pass



    config.time_list_pipeline = time_each_pipe
    config.global_bs_pipeline =  num_sample_each_pipe

    for i  in range(config.num_pipelines):
        config.pipelines[i].global_bs = num_sample_each_pipe[i]
        for j in range(config.pipelines[i].num_stages):
            config.pipelines[i].time_list[j] = config.pipelines[i].sum_stage_time + (config.pipelines[i].global_bs//config.pipelines[i].micro_bs-1 )*config.pipelines[i]._time_list[j]
            config.pipelines[i].time_list[j] /= 1000
        # print(f"config.pipelines[{i}].time_list",config.pipelines[i].time_list)
        # predict_time_breakdown(config.pipelines[i],args=args)
        # print(f"config.pipelines[{i}].time_list",config.pipelines[i].time_list)
        # print(f"config.pipelines[{i}].sum_stage_time",config.pipelines[i].sum_stage_time)
        # print(f"config.pipelines[{i}]._time_list",config.pipelines[i]._time_list)
        # print(f"config.pipelines[{i}].time_list",config.pipelines[i].time_list)
        # print(f"config.pipelines[{i}].global_bs",config.pipelines[i].global_bs)
        # print(f"config.pipelines[{i}].micro_bs",config.pipelines[i].micro_bs)
        # print(f"time_each_pipe[{i}]",time_each_pipe[i])
    # print("config.time_list_pipeline",config.time_list_pipeline)
    # exit()



    return config


def hash_list_of_dicts(dict ):
    # 将每个字典转换为排序后的元组（按键排序）
    list = []

    for key in dict.keys():
        list.append(dict[key])

    processed = [tuple(sorted(d.items())) for d in list]
    # 对处理后的元组列表进行排序，确保整体顺序不影响哈希结果
    processed_sorted = sorted(processed)
    # 转换为元组并生成哈希值
    return hash(tuple(processed_sorted))

def allocate_gpu(node_order, device_group,args):
    output = {}
    # 按节点编号升序排序
    nodes = sorted(node_order.keys(), key=lambda x: int(x))

    device_list = []
    for node_key in nodes:
        node_info = node_order[node_key]
        gpu_type = node_info['GPU']
        gpu_num = node_info['GPU_NUM']
        for i in range(gpu_num):
            device_list.append(gpu_type)
    stage_device_list = []
    device_list_idx = 0
    for stage in device_group:
        list = []
        for idx in range(stage):
            list.append(device_list[device_list_idx])
            device_list_idx+=1
        stage_device_list.append(list)
    tflops_list = []
    memory_list = []
    for stage_idx in range(len(device_group)):
        tflops = 0
        memory = 0
        for idx in range(device_group[stage_idx]):
            gpu_type= stage_device_list[stage_idx][idx]
            tflops+=args.device_info[gpu_type]["TFLOPS"]
            memory+=args.device_info[gpu_type]["memory"]
        tflops_list.append(tflops)
        memory_list.append(memory)
    # current_index = 0  # 当前处理的device_group索引
    # for node_key in nodes:
    #     node_info = node_order[node_key]
    #     gpu_type = node_info['GPU']
    #     gpu_num = node_info['GPU_NUM']
    #     partitions = []
    #     accumulated = 0

    #     # 从当前索引开始累加，直到满足GPU数量
    #     while current_index < len(device_group) and accumulated + device_group[current_index] <= gpu_num:
    #         partitions.append(device_group[current_index])
    #         accumulated += device_group[current_index]
    #         current_index += 1

    #     output[gpu_type] = partitions


    # for key in output.keys():
    #     for item in output[key]:
    #         tflops_list.append(args.device_info[key]["TFLOPS"]* item)
    #         memory_list.append(args.device_info[key]["memory"]* item)
    if(len(tflops_list)==0):
        print("node_order",node_order)
        print("device_group,",device_group)
        print("output,",output)
        return 0,0
    tflops_mean_value = sum(tflops_list) / len(tflops_list)
    tflops_deviations = [(x - tflops_mean_value)*(x - tflops_mean_value) for x in tflops_list]

    memory_mean_value = sum(memory_list) / len(memory_list)
    memory_deviations = [(x - memory_mean_value)*(x - memory_mean_value) for x in memory_list]



    return sum(tflops_deviations) / len(tflops_deviations) , sum(memory_deviations) / len(memory_deviations)


def get_target_search_time(node_order_list,total_device_group_list,start_num_stages,end_num_stages,time,args):
    target_search_time_list = {}
    for i in range(len(node_order_list)):
        target_search_time_list[i]={}
        for num_stages in range(start_num_stages, end_num_stages + 1):
            target_search_time_list[i][num_stages] = {}


    if(args.elastic_time):
        TFLOPS_each_num_stages = {}
        Memory_each_num_stages = {}
        search_time_each_num_stages={}
        for num_stages in range(start_num_stages, end_num_stages + 1): # 遍历所有的stage end_num_stages 为GPU的数量
            device_group_list = total_device_group_list[num_stages]
            TFLOPS_each_num_stages[num_stages] = []
            Memory_each_num_stages[num_stages] = []
            search_time_each_num_stages[num_stages] = []
            for device_group_idx in range(len(device_group_list)):
                for i in range(len(node_order_list)):
                    flops_gap, memory_gps = allocate_gpu(node_order_list[i], device_group_list[device_group_idx],args)
                    TFLOPS_each_num_stages[num_stages].append(flops_gap)
                    Memory_each_num_stages[num_stages].append(memory_gps)
            if(len(TFLOPS_each_num_stages[num_stages])==0):
                continue

            flops_max_value = max( TFLOPS_each_num_stages[num_stages])
            flops_min_value = min ( TFLOPS_each_num_stages[num_stages])

            memory_max_value = max( Memory_each_num_stages[num_stages])
            memory_min_value = min ( Memory_each_num_stages[num_stages])
            # print("Memory_each_num_stages[num_stages]",Memory_each_num_stages[num_stages])

            if(flops_max_value == flops_min_value):
                TFLOPS_each_num_stages[num_stages] = [ 1 for item in TFLOPS_each_num_stages[num_stages]]
            else:
                TFLOPS_each_num_stages[num_stages] = [ (flops_max_value - item)/(flops_max_value - flops_min_value)   for item in TFLOPS_each_num_stages[num_stages]]

            if(memory_max_value == memory_min_value):
                Memory_each_num_stages[num_stages] = [ 1 for item in Memory_each_num_stages[num_stages]]
            else:
                Memory_each_num_stages[num_stages] = [ (memory_max_value - item)/(memory_max_value - memory_min_value)  for item in Memory_each_num_stages[num_stages]]



        for num_stages in range(start_num_stages, end_num_stages + 1): # 遍历所有的stage end_num_stages 为GPU的数量
            device_group_list = total_device_group_list[num_stages]
            idx = 0
            for device_group_idx in range(len(device_group_list)):
                for i in range(len(node_order_list)):
                    # print("node_order ", node_order_list[i])
                    # print("device_group_list ", device_group_list[device_group_idx])
                    target_search_time_list[i][num_stages][device_group_idx] = TFLOPS_each_num_stages[num_stages][idx]*Memory_each_num_stages[num_stages][idx]  * time
                    idx +=1

        # print("target_search_time_list",target_search_time_list)
        # print("TFLOPS_each_num_stages",TFLOPS_each_num_stages)
        # print("Memory_each_num_stages",Memory_each_num_stages)
        # exit()
    else:
        for i in range(len(node_order_list)):
            for num_stages in range(start_num_stages, end_num_stages + 1): # 遍历所有的stage end_num_stages 为GPU的数量
                device_group_list = total_device_group_list[num_stages]
                for device_group_idx in range(len(device_group_list)):
                    target_search_time_list[i][num_stages][device_group_idx] = time                     # TODO

    return target_search_time_list


def selected_device_group_indices(device_group_list, args):
    start = max(args.start_dg_idx + 1, 0)
    end = min(args.end_dg_idx, len(device_group_list))
    return range(start, end)



if __name__=='__main__':

    args = parse_args()
    initialize_primitives(args)
    overall_start = time.time()
    print_args(args)
    total_gpu_type_num_dict = args.gpu_type_num_dict
    profile_time = state.read_profiled_time(args.model_name, args.model_size, args.profiled_time_path,args)
    dp_cache, split_cache = None, None
    search_time = args.time_budget_total
    cache_result = {}
    dp_cache_result = None
    # 打印结果
    config_result_each_diff_num_pipeline = []
    prefix2dict={}
    best_throughput = 0
    cached_throughput = {}
    for num_pipe in range(args.min_num_pipeline, args.max_num_pipeline+1):
        balanced_config_list = []

        if(args.dp_find_pipeline):
            if(num_pipe ==4 ):
                n = len(total_gpu_type_num_dict)
                max_k = args.max_num_pipeline

                # 预处理所有k=1到max_k
                dp_cache, split_cache = precompute_all_k(n, max_k, cache_result,total_gpu_type_num_dict,hash_list_of_dicts)
                dp_cache_result = get_results_for_ks(n, [i for i in range(4,args.max_num_pipeline+1)], dp_cache, split_cache)
            if(num_pipe >=4):
                # 输出结果
                throughput , split   = dp_cache_result[num_pipe][0] , dp_cache_result[num_pipe][1]
                print(f"num_pipe {num_pipe} split {split} throughput {throughput}")
                config_list = split_to_cache_result(split, cache_result,total_gpu_type_num_dict,hash_list_of_dicts)
                if(config_list ==[]):
                    print("config_list is empty")
                    continue
                else:
                    print(f"config_list is not empty")
                imbalanced_config =  HeterogeneousPlan(num_pipe)
                for config in config_list:
                    if(config != None):
                        imbalanced_config.pipelines.append(config)

                if(len(imbalanced_config.pipelines)!=num_pipe):
                    continue

                if( not args.test_search_time):
                    for pipe_idx in range(num_pipe):
                        imbalanced_config.global_bs_pipeline.append(args.global_batch_size)
                        imbalanced_config.micro_bs_pipeline.append(imbalanced_config.pipelines[pipe_idx].micro_bs)
                        imbalanced_config.num_mb_pipeline.append(args.global_batch_size //imbalanced_config.pipelines[pipe_idx].micro_bs)
                        imbalanced_config.time_list_pipeline.append(max(imbalanced_config.pipelines[pipe_idx].time_list))
                    balanced_config = workload_balance_pipeline_v1(imbalanced_config)
                    balanced_config_list.append(balanced_config)
                if( not args.test_search_time):
                    save_and_print_top_hybrid_config(balanced_config_list,args)
                continue
        pipeline_split_list  = split_devices_to_pipeline_large(total_gpu_type_num_dict, num_pipe) # pipeline_split_list是在num_pipe个pipeline的情况下的所有划分
        pipeline_split_list = sort_pipeline_split_list(pipeline_split_list,args)

        # print(f"pipeline_split_list {pipeline_split_list}")
        # print(f"pipeline_split_list {len(pipeline_split_list)}")
        # continue
        for pipeline_split in pipeline_split_list: # 每一个pipeline_split 是一种划分
            result_for_each_pipeline = []
            imbalanced_config =  HeterogeneousPlan(num_pipe)
            current_throughput = [0] * len(pipeline_split)
            optimal_prune_pass = False

            break_flag = False

            for pipe_idx in range(len(pipeline_split)): # 每一个pipeline_split[pipe_idx] 是一条pipeline
                gpu_type_num_dict_ = pipeline_split[pipe_idx]
                gpu_type_num_dict = {}
                idx = 0
                for item in gpu_type_num_dict_:
                    gpu_type_num_dict[f"{idx}"] = item
                    idx += 1
                args.gpu_type_num_dict = gpu_type_num_dict
                args.num_nodes = len(args.gpu_type_num_dict)
                args.num_gpus = sum([args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])
                args.gpu_type_set = [args.gpu_type_num_dict[f"{i}"]["GPU"] for i in range(args.num_nodes)]
                args.gpu_type_set = list(set(args.gpu_type_set))
                args.node_order_list = [args.gpu_type_num_dict]

                p_idx = 0
                start_num_stages = args.start_num_stages
                end_num_stages = args.end_num_stages
                total_search_status =0
                total_device_group_list = {}
                args.node_order_list_each_device_group = {}

                for num_stages in range(start_num_stages, end_num_stages + 1):
                    if(args.homo_stage):
                        device_group_list = enumerate_homogeneous_stage_sizes(num_stages,args.num_gpus, args.num_gpus//num_stages, args.num_gpus//num_stages)
                    else:
                        device_group_list = enumerate_node_aligned_stage_sizes(num_stages,args.num_gpus, args.num_gpus//end_num_stages, args.num_gpus//start_num_stages,[args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])

                    # print(f"device_group_list {device_group_list}")
                    total_device_group_list[num_stages] = device_group_list
                    if( device_group_list != []):
                        # print(device_group_list)
                        # print(args.gpu_type_num_dict)
                        args.node_order_list_each_device_group[num_stages] = generate_valid_permutations(args.gpu_type_num_dict,device_group_list,args.easy_mode)
                        # print(f"args.node_order_list_each_device_group {args.node_order_list_each_device_group[num_stages]}")
                    else:
                        args.node_order_list_each_device_group[num_stages] = [[]]
                    for device_group_idx in range(len(device_group_list)):
                        total_search_status += len(args.node_order_list_each_device_group[num_stages][device_group_idx])

                    # total_search_status += len(device_group_list)*len(args.node_order_list_each_device_group[num_stages])
                    # print("node_order_list_each_device_group",args.node_order_list_each_device_group[num_stages])
                # exit()
                if(total_search_status == 0):
                    break_flag = True
                    break
            if(break_flag):
                continue
            for pipe_idx in range(len(pipeline_split)): # 每一个pipeline_split[pipe_idx] 是一条pipeline
                gpu_type_num_dict_ = pipeline_split[pipe_idx]
                gpu_type_num_dict = {}
                idx = 0
                for item in gpu_type_num_dict_:
                    gpu_type_num_dict[f"{idx}"] = item
                    idx += 1
                args.gpu_type_num_dict = gpu_type_num_dict
                args.num_nodes = len(args.gpu_type_num_dict)
                args.num_gpus = sum([args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])
                args.gpu_type_set = [args.gpu_type_num_dict[f"{i}"]["GPU"] for i in range(args.num_nodes)]
                args.gpu_type_set = list(set(args.gpu_type_set))
                # print(f"args.gpu_type_num_dict {args.gpu_type_num_dict}")
                # if(args.enable_diff_order):
                # else:
                args.node_order_list = [args.gpu_type_num_dict]
                args.node_order_list_each_device_group = {}
                # print(f" args.node_order_list {args.node_order_list }")
                p_idx = 0
                # start_num_stages =  min(args.num_nodes,8)
                # end_num_stages = args.num_nodes * min (args.gpu_type_num_dict["0"]["GPU_NUM"],2) if args.num_nodes > 1 else args.gpu_type_num_dict["0"]["GPU_NUM"]  #TODO
                start_num_stages = args.start_num_stages
                end_num_stages = args.end_num_stages
                # end_num_stages = 6
                total_search_status =0
                total_device_group_list = {}
                for num_stages in range(start_num_stages, end_num_stages + 1):
                    # device_group_list = enumerate_node_aligned_stage_sizes(num_stages,args.num_gpus, args.num_gpus//end_num_stages, args.num_gpus//start_num_stages,[args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])
                    # device_group_list = enumerate_homogeneous_stage_sizes(num_stages,args.num_gpus, args.num_gpus//end_num_stages, args.num_gpus//start_num_stages)
                    if(args.homo_stage):
                        device_group_list = enumerate_homogeneous_stage_sizes(num_stages,args.num_gpus, args.num_gpus//num_stages, args.num_gpus//num_stages)
                    else:
                        device_group_list = enumerate_node_aligned_stage_sizes(num_stages,args.num_gpus, args.num_gpus//end_num_stages, args.num_gpus//start_num_stages,[args.gpu_type_num_dict[f"{i}"]["GPU_NUM"] for i in range(args.num_nodes)])
                    total_device_group_list[num_stages] = device_group_list
                    if( device_group_list != []):
                        # print(device_group_list)
                        args.node_order_list_each_device_group[num_stages] = generate_valid_permutations(args.gpu_type_num_dict,device_group_list,args.easy_mode)
                    else:
                        args.node_order_list_each_device_group[num_stages] = [[]]
                    for device_group_idx in range(len(device_group_list)):

                        total_search_status += len(args.node_order_list_each_device_group[num_stages][device_group_idx])
                # exit()
                print("total_search_status",total_search_status)
                # print("total_device_group_list",total_device_group_list)
                args.total_device_group_list = total_device_group_list # num_stages -> device_group_list
                cache_prefix = hash_list_of_dicts(args.gpu_type_num_dict)
                prefix2dict[cache_prefix] = args.gpu_type_num_dict

                print("cache_prefix",cache_prefix)
                best_config = None
                if args.multi_process:
                    multiprocessing.set_start_method('spawn', force=True) ##
                    process_list = []
                    # queue = Queue()
                    manager = multiprocessing.Manager()
                    shared_dict = manager.dict()
                    total_process = 0
                    process_have_done = 0
                    if(args.use_cache and cache_result.get(cache_prefix)!=None):
                        print(f"cache_prefix {cache_prefix} {args.gpu_type_num_dict} has been calculated")
                        best_config = cache_result[cache_prefix]
                        if(best_config == None):
                            break_flag = True
                            break
                    else:
                        if(args.optimal_prune):
                            current_pipeline_upper_bound ,proven_upper_bound,approximate_upper_bound= get_upper_bound(pipeline_split,cached_throughput,prefix2dict,pipe_idx,args)
                            if(sum(current_pipeline_upper_bound)+ sum (current_throughput) < best_throughput):
                                optimal_prune_pass = True
                                # print("optimal_prune work!")
                                # print("current_pipeline_upper_bound" ,current_pipeline_upper_bound )
                                # print("proven_upper_bound" ,proven_upper_bound )
                                # print("approximate_upper_bound" ,approximate_upper_bound )
                                # print("current_throughput" ,current_throughput )
                                # print("best_throughput" ,best_throughput )
                                continue
                            # else:
                                # print("current_pipeline_upper_bound" ,current_pipeline_upper_bound )
                                # print("proven_upper_bound" ,proven_upper_bound )
                                # print("approximate_upper_bound" ,approximate_upper_bound )
                                # print("current_throughput" ,current_throughput )
                                # print("best_throughput" ,best_throughput )
                        target_search_time_list = get_target_search_time(args.node_order_list,total_device_group_list,start_num_stages,end_num_stages,search_time,args)
                        # print(f"cache_prefix {cache_prefix} {args.gpu_type_num_dict} calculating")
                        # for i in range(len(args.node_order_list)):
                        for num_stages in range(start_num_stages, end_num_stages + 1): # 遍历所有的stage end_num_stages 为GPU的数量
                            device_group_list = args.total_device_group_list[num_stages]
                            for device_group_idx in selected_device_group_indices(device_group_list, args):
                                args.node_order_list = args.node_order_list_each_device_group[num_stages][device_group_idx]
                                for i in range(len(args.node_order_list)):
                                # args.time_budget_total = target_search_time_list[i][num_stages][device_group_idx]
                                    if(args.time_budget_total <=0):
                                        continue
                                    # print("run_search")
                                    p = Process(target=run_search, args=(num_stages,i,device_group_idx,profile_time,shared_dict,args))
                                    p.start()
                                    process_list.append(p)
                                    total_process += 1
                                    if((total_process-process_have_done)>= args.num_multi_process):
                                        # process_list[process_have_done].join()
                                        # process_have_done += 1
                                        for p in process_list:
                                            if not p.is_alive():
                                                p.join()
                                                process_list.remove(p)
                                                process_have_done += 1
                        # for i in range(process_have_done,total_process):
                        #     process_list[i].join()
                        for p in process_list:
                            p.join()
                            process_have_done += 1
                else:
                    shared_dict = {}
                    for i in range(len(args.node_order_list)):
                        for num_stages in range(args.start_num_stages, args.end_num_stages + 1):
                            device_group_list = args.total_device_group_list[num_stages]
                            print("node_order" , args.node_order_list[i] , "num_stages", num_stages , "device_group_list",device_group_list)
                            args.device_group_list = device_group_list
                            for device_group_idx in selected_device_group_indices(device_group_list, args):
                                run_search(
                                    num_stages,
                                    i,
                                    device_group_idx,
                                    profile_time,
                                    shared_dict,
                                    args,
                                )

                if( not args.test_search_time):
                    if(args.use_cache == False or cache_result.get(cache_prefix)==None):
                        best_config, best_config_json = save_and_print_top_configs_dp_hete(shared_dict, args,start_num_stages,end_num_stages)

                        if(args.use_cache):
                            cache_result[cache_prefix] = best_config
                        if(best_config == None):
                            break_flag = True
                            break
                    cached_throughput[cache_prefix] = best_config.global_bs/max(best_config.time_list)*1000
                    current_throughput[pipe_idx] = best_config.global_bs/max(best_config.time_list)*1000
                    result_for_each_pipeline.append(best_config)
                    imbalanced_config.pipelines.append(best_config)
                # print(f"best_config {best_config}")
                # exit()
            # 对result_for_each_pipeline进行balance
            if(break_flag):
                continue
            if( (optimal_prune_pass == False) and (not args.test_search_time)):
                throughput = 0
                for pipe_idx in range(num_pipe):
                    imbalanced_config.global_bs_pipeline.append(args.global_batch_size)
                    imbalanced_config.micro_bs_pipeline.append(imbalanced_config.pipelines[pipe_idx].micro_bs)
                    imbalanced_config.num_mb_pipeline.append(args.global_batch_size //imbalanced_config.pipelines[pipe_idx].micro_bs)
                    imbalanced_config.time_list_pipeline.append(max(imbalanced_config.pipelines[pipe_idx].time_list))
                    throughput += imbalanced_config.pipelines[pipe_idx].global_bs/max(imbalanced_config.pipelines[pipe_idx].time_list)
                best_throughput = max(throughput*1000,best_throughput)
                balanced_config = workload_balance_pipeline_v1(imbalanced_config)
                balanced_config_list.append(balanced_config)

        if( len(balanced_config_list)>0 and not args.test_search_time):
            save_and_print_top_hybrid_config(balanced_config_list,args)
   

    # file_name = f'{args.log_path}{args.model_name}_{args.model_size}_summary_{args.config_suffix}.csv'
    # info_to_csv = [["search_cost"], [f"{(overall_end - overall_start):.2f}"]]
    # with open(file_name, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     for row in info_to_csv:
    #         writer.writerow(row)
    overall_end = time.time()
    print(f"\noverall search time: {(overall_end - overall_start):.2f} s")
