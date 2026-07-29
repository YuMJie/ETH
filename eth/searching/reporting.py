# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import json
import os
import shutil
import time
from typing import List

from .config import MAX_VALUE, config_details, format_size_list
from .serialization import (
    dump_config_to_json,
    dump_hybrid_config_to_json,
    save_config_info_to_csv,
    save_distribution_to_csv,
)
from .state import HeterogeneousPlan, SearchConfig

def debug_info(info, print_debug_info):
    if print_debug_info:
        print(info)

class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name_ = name
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0
        self.start_time = time.time()
        self.elapsed_history = 0.0
        self.elapsed_list = []

    def start(self):
        """Start the timer."""
        assert not self.started_, f"{self.name_} timer has already been started"
        self.start_time = time.time()
        self.started_ = True

    def stop(self):
        """Stop the timer."""
        assert self.started_, 'timer is not started'
        elapsed_time = time.time() - self.start_time
        self.elapsed_ += (elapsed_time)
        self.elapsed_list.append(elapsed_time)
        self.started_ = False
        self.count += 1

    def reset(self):
        """Reset timer."""
        self.elapsed_ = 0.0
        self.started_ = False
        self.count = 0
        self.elapsed_history = (time.time() - self.start_time)

    def elapsed(self, reset=True):
        """Calculate the elapsed time."""
        started_ = self.started_
        # If the timing in progress, end it first.
        if self.started_:
            self.stop()
        # Get the elapsed time.
        elapsed_ = self.elapsed_
        # Reset the elapsed time
        if reset:
            self.reset()
            self.count = 0
        # If timing was in progress, set it back.
        if started_:
            self.start()
        return elapsed_

    def elapsed_since_first_invoke(self, reset=True):
        """Calculate the elapsed time."""
        return time.time() - self.start_time


class Timers:
    """Group of timers."""

    def __init__(self):
        self.timers = {}

    def __call__(self, name):
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def write(self, names, writer, iteration, normalizer=1.0, reset=False):
        """Write timers to a tensorboard writer"""
        assert normalizer > 0.0
        for name in names:
            value = self.timers[name].elapsed(reset=reset) / normalizer
            writer.add_scalar(name + '-time', value, iteration)

    def log(self, reset=True):
        print(f"===== timers ====")
        for name in sorted(self.timers):
            count = self.timers[name].count
            if count == 0:
                elapsed_time = self.timers[name].elapsed_history
            else:
                elapsed_time = self.timers[name].elapsed(reset=reset)
            print('{}: {:.2f} s [count = {}]'.format(name, elapsed_time, count),flush=True)
        print(f"===== end of timers ====\n",flush=True)
def print_args(args):
    """Print arguments."""
    print('------------------------ arguments ------------------------',
            flush=True)
    str_list = []
    for arg in vars(args):
        dots = '.' * (48 - len(arg))
        str_list.append('  {} {} {}'.format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print('-------------------- end of arguments ---------------------',
            flush=True)

def print_simple_config_info(config, info="", add_history=False, print_recompute_ops=False, print_debug_info=False):
    if config is None:
        return

    num_ops_stage, tp_per_stage, dp_per_stage, recompute_ops, base_batch_size, algo_per_stage = config_details(config)
    gpu_list = [config.stages[i].num_gpus for i in range(config.num_stages)]
    recompute_ops_sum = []
    for i in range(config.num_stages):
        recompute_ops_sum.append(sum(recompute_ops[i]))
    detailed_tp_size = format_size_list(tp_per_stage)
    detailed_dp_size = format_size_list(dp_per_stage)
    detailed_algos = format_size_list(algo_per_stage)

    history = "{}|{:.2f}|{:.2f}| op = {}| tp = {} | dp = {} | algo = {} | rc = {} | gpus = {} | micro_bs = {} | time = {} | memory = {}".format(
        info, max(config.time_list)/1000, max(config.memory_list), num_ops_stage, detailed_tp_size, detailed_dp_size, detailed_algos, recompute_ops_sum, gpu_list, base_batch_size, list(map(int, config.time_list)), list(map(int, config.memory_list)))
    debug_info(history, print_debug_info)
    if add_history:
        config.history += history + "\n"

    if print_recompute_ops:
        for i in range(config.num_stages):
            ops = config.stages[i].ops
            recompute_ops = config.stages[i].recompute_ops
            stage_recompute_index_string = f"[stage {i} recompute_ops] "
            stage_string = f"stage {i}: "
            for j in range(len(recompute_ops)):
                if recompute_ops[j] == 1:
                    stage_recompute_index_string += f"{j},"
                    stage_string += ops[j] + ", "
            print(stage_recompute_index_string)
            print(stage_string)

        for i in range(config.num_stages):
            ops = config.stages[i].ops
            algo_list = config.stages[i].algo
            stage_algo_string = f"[stage {i} algo1] "
            for j in range(len(algo_list)):
                if algo_list[j] == 1:
                    stage_algo_string += f"[{j}] {ops[j]} "
            print(stage_algo_string)

    return

def print_config_info(result_dict_result,args,stage_num, device_group_idx, node_order_idx,top_count,config_dict):
        config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown ,each_stage_memory_breakdown,min_get_best_config_time ,best_config= result_dict_result
        config_throughput = args.global_batch_size / (config_time/1000)
        data = []
        data_1_1 = []
        data_1_2 = []
        data_1 = []
        data_2 = []
        data_3 = []

        data.append([stage_num, f"{config_time/1000:.2f}", f"{config_throughput:.2f}", f"{config_mem:.0f}", f"{explored_cases}"])
        data_1_1.append([f"{args.node_order_list[node_order_idx]}" , f"{node_order_idx}" ,f"{device_group_idx}"])
        data_1_2.append([f"{args.total_device_group_list[stage_num][device_group_idx]}"])

        for time_idx  in range(len(time_list)) :
            time_list[time_idx] = time_list[time_idx]/1000


        time_list = format_list(time_list)

        # for memory_idx  in range(len(memory_list)) :
        #     memory_list[memory_idx] = memory_list[memory_idx]/1000

        memory_list = format_list(memory_list)
        eff_loss_time_per_gpu = format_list(eff_loss_time_per_gpu)
        recomp_time_per_gpu = format_list(recomp_time_per_gpu)
        data_1.append([f"{adaptive_times}", f"{time_list}", f"{memory_list}", f"{eff_loss_time_per_gpu}", f"{recomp_time_per_gpu}"])


        for stage_list in each_stage_time_breakdown:
            for time_idx in range(1,len(stage_list)):
                stage_list[time_idx] = stage_list[time_idx]/1000


        each_stage_time_breakdown = [format_list(stage_list)  for stage_list in each_stage_time_breakdown]
        data_2.append([f"{each_stage_time_breakdown}"])
        each_stage_memory_breakdown = [format_list(stage_list)  for stage_list in each_stage_memory_breakdown]
        data_3.append([f"{each_stage_memory_breakdown}"])
        print(f" ============================== TOP {top_count} ==============================")
        header = ["# of stages", "est_iteration_time(s)", "estimated_throughput(samples/s)", "est_mem(MB)", "# of explored cases" ]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["Node Order" , "node_order_idx" ,"device_group_idx"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_1_1:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["Device Group"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_1_2:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["adaptive_times", "time_list (s)", "memory_list(GB)", "eff_loss_time_per_gpu", "recomp_time_per_gpu"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_1:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header= ["each_stage_time_breakdown [ total (ms) fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_2:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header= ["each_stage_memory_breakdown [total (MB) memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        for row in data_3:
            print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

        header = ["# min_get_best_config_time"]
        column_widths = [len(str(header[i])) for i in range(len(header))]
        print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
        print(min_get_best_config_time)

def format_list(list):
    formatted_list = [float(f"{x:.3f}") for x in list]
    return formatted_list

def save_and_print_top_configs(result_dict, args):
    # print("result_dict",result_dict)
    sorted_time_list = [MAX_VALUE]
    sorted_config_stage_list = [(MAX_VALUE, MAX_VALUE, MAX_VALUE)]

    sorted_config_stage_list_each_stage = (MAX_VALUE, MAX_VALUE, MAX_VALUE)
    for i in range(args.start_num_stages, args.end_num_stages + 1):
        best_stage_time = MAX_VALUE
        for node_order_idx in range(len(args.node_order_list)):
            for device_group_idx in range(len(args.total_device_group_list[i])):
        # if node_order_idx in result_dict and i in result_dict[node_order_idx] and result_dict[node_order_idx][i] is not None:
                if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                    # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                    config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                    if config_time > 0 and best_stage_time > config_time:
                        best_stage_time = config_time
                        sorted_config_stage_list_each_stage = (i, device_group_idx, node_order_idx)

        if(best_stage_time!=MAX_VALUE):
            print(f"stage_num {i}")
            stage_num, device_group_idx, node_order_idx = sorted_config_stage_list_each_stage
            src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{i}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
            with open(src_file, "r") as f:
                config_dict = json.load(f)
            print_config_info(result_dict[f"{node_order_idx}_{stage_num}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,-1,config_dict)
        else:
            print(f"stage_num {i} is None")



    for node_order_idx in range(len(args.node_order_list)):
        for i in range(args.start_num_stages, args.end_num_stages + 1):
                for device_group_idx in range(len(args.total_device_group_list[i])):
            # if node_order_idx in result_dict and i in result_dict[node_order_idx] and result_dict[node_order_idx][i] is not None:
                    if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                    # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                        config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                        if config_time > 0:
                            for j in range(len(sorted_time_list)):
                                current_config_time = sorted_time_list[j]
                                if config_time < current_config_time:
                                    sorted_time_list.insert(j, config_time)
                                    sorted_config_stage_list.insert(j, (i, device_group_idx, node_order_idx))
                                    # print(f"insert {i}stages, device_group_idx = {device_group_idx}, node_order_idx = {node_order_idx} to {j}th")
                                    break
    sorted_config_stage_list.pop()
    sorted_time_list.pop()
    # print("sorted_config_stage_list",sorted_config_stage_list)
    ##### save configs:


    save_count = 0
    top_count = 0
    for i in range(min((args.end_num_stages - args.start_num_stages + 1)*len(args.node_order_list), args.num_of_show_configs, len(sorted_time_list))): #TODO 总数需修改加上dg

        stage_num, device_group_idx, node_order_idx = sorted_config_stage_list[i]
        # print(f"stage_num = {stage_num}, node_order_idx = {node_order_idx}")
        src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
        dst_file = f'{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{args.save_prefix}_{args.time_budget_total}_{args.config_suffix}.json'

        if(save_count < args.num_of_saved_configs):
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                save_count += 1
                print(f"best config save to {dst_file} ")
                print(f"best config save to {src_file} ")
            else:
                print(f"file {src_file} does not exist.")
        else:
            print(f"TOP {top_count} config save to {src_file}")
        with open(src_file, "r") as f:
            config_dict = json.load(f)
        print_config_info(result_dict[f"{node_order_idx}_{stage_num}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,top_count,config_dict)
        top_count += 1


def save_and_print_top_configs_dp_hete(result_dict, args ,start_num_stages ,end_num_stages):
    # print("result_dict",result_dict)
    sorted_time_list = [MAX_VALUE]
    sorted_config_list = [MAX_VALUE]
    sorted_config_stage_list = [(MAX_VALUE, MAX_VALUE, MAX_VALUE)]
    sorted_config_stage_list_each_stage = (MAX_VALUE, MAX_VALUE, MAX_VALUE)
    print("total_device_group_list",args.total_device_group_list)
    print("start_num_stages",start_num_stages)
    print("end_num_stages",end_num_stages)
    total_explored_cases = 0
    for i in range(start_num_stages, end_num_stages + 1):
        best_stage_time = MAX_VALUE
        for node_order_idx in range(len(args.node_order_list)):
            for device_group_idx in range(len(args.total_device_group_list[i])):
                if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                    # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                    config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                    total_explored_cases+=explored_cases
                    if config_time > 0 and best_stage_time > config_time:
                        best_stage_time = config_time
                        sorted_config_stage_list_each_stage = (i, device_group_idx, node_order_idx)

        if(best_stage_time!=MAX_VALUE):
            print(f"stage_num {i}")
            stage_num, device_group_idx, node_order_idx = sorted_config_stage_list_each_stage
            src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{i}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
            with open(src_file, "r") as f:
                config_dict = json.load(f)
            print_config_info(result_dict[f"{node_order_idx}_{i}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,-1,config_dict)
        else:
            print(f"stage_num {i} is None")
    print("total_explored_cases",total_explored_cases)
    for node_order_idx in range(len(args.node_order_list)):
        for i in range(start_num_stages, end_num_stages + 1):
                for device_group_idx in range(len(args.total_device_group_list[i])):
            # if node_order_idx in result_dict and i in result_dict[node_order_idx] and result_dict[node_order_idx][i] is not None:
                    if result_dict.get(f"{node_order_idx}_{i}_{device_group_idx}") is not None:
                        # best_config.adaptive_times,best_config.time_list, best_config.memory_list, best_config.breakdown_eff_loss_time_per_gpu,best_config.breakdown_recomp_time_per_gpu
                        config_time, config_mem, explored_cases, search_time, case_distribution,adaptive_times, time_list, memory_list, eff_loss_time_per_gpu, recomp_time_per_gpu,each_stage_time_breakdown,each_stage_memory_breakdown, min_get_best_config_time,best_config= result_dict[f"{node_order_idx}_{i}_{device_group_idx}"]
                        if config_time > 0:
                            for j in range(len(sorted_time_list)):
                                current_config_time = sorted_time_list[j]
                                if config_time < current_config_time:
                                    sorted_time_list.insert(j, config_time)
                                    sorted_config_stage_list.insert(j, (i, device_group_idx, node_order_idx))
                                    sorted_config_list.insert(j, best_config)
                                    # print(f"insert {i}stages, device_group_idx = {device_group_idx}, node_order_idx = {node_order_idx} to {j}th")
                                    break
    sorted_config_stage_list.pop()
    sorted_time_list.pop()
    sorted_config_list.pop()
    # print("sorted_config_stage_list",sorted_config_stage_list)
    ##### save configs:
    save_count = 0
    top_count = 0

    best_config = None
    best_config_json = None

    for i in range(min((end_num_stages - args.start_num_stages + 1)*len(args.node_order_list), args.num_of_show_configs, len(sorted_time_list))): #TODO 总数需修改加上dg

        stage_num, device_group_idx, node_order_idx = sorted_config_stage_list[i]
        # print(f"stage_num = {stage_num}, node_order_idx = {node_order_idx}")
        src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{node_order_idx}node_order_idx_{device_group_idx}device_group_idx_{args.config_suffix}.json'
        dst_file = f'{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{args.save_prefix}_{args.time_budget_total}_{args.config_suffix}.json'

        if(save_count < args.num_of_saved_configs):
            if os.path.exists(src_file):
                shutil.copy(src_file, dst_file)
                save_count += 1
                print(f"best config save to {dst_file}")
            else:
                print(f"file {src_file} does not exist.")
        else:
            print(f"TOP {top_count} config save to {src_file}")

        with open(src_file, "r") as f:
            config_dict = json.load(f)
            if(best_config_json is None):
                best_config_json  = config_dict
                best_config =  sorted_config_list[i]
        print_config_info(result_dict[f"{node_order_idx}_{stage_num}_{device_group_idx}"],args,stage_num, device_group_idx, node_order_idx,top_count,config_dict)
        top_count += 1


    return  best_config, best_config_json


# cost prediction
def save_and_print_configs(config:SearchConfig , args):

    ##### save configs:
    data = []
    data_1 = []
    data_2 = []
    data_3 = []

    stage_num = len(config.stages)
    config_time = max(config.time_list)/1000
    config_mem = max(config.memory_list)
    adaptive_times = config.adaptive_times
    time_list = config.time_list.copy()
    memory_list = config.memory_list

    for i in range(len(memory_list)):
        memory_list[i]/=1024

    for i in range(len(time_list)):
        time_list[i]/=1000



    eff_loss_time_per_gpu = config.breakdown_eff_loss_time_per_gpu
    recomp_time_per_gpu = config.breakdown_recomp_time_per_gpu
    each_stage_time_breakdown = config.each_stage_time_breakdown.copy()
    each_stage_memory_breakdown = config.each_stage_memory_breakdown

    for i in range(stage_num):
        for j in range(1,len(each_stage_time_breakdown[i])):
            each_stage_time_breakdown[i][j]/=1000

    if(0): #TODO 暂时不要保存结果
        src_file = f'{args.config_save_path}{args.model_name}_{args.model_size}_{stage_num}stages_{args.config_suffix}.json'
        dst_file = f'{args.config_save_path}top_configs/{args.model_name}_{args.model_size}_{args.save_prefix}_{args.time_budget_total}_{args.config_suffix}.json'
        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
    config_throughput = args.global_batch_size / (config_time)
    data.append([stage_num,f"{config_time}",f"{config_throughput:.2f}", f"{config_mem:.0f}" ,f"{args.global_batch_size}",f"{config.micro_bs}"] )
    data_1.append([f"{adaptive_times}", f"{time_list}", f"{memory_list}", f"{eff_loss_time_per_gpu}", f"{recomp_time_per_gpu}"])
    data_2.append([f"{each_stage_time_breakdown}"])
    data_3.append([f"{each_stage_memory_breakdown}"])

    header = ["# of stages", "est_iteration_time(s)", "estimated_throughput(samples/s)", "est_mem(MB)", "global_bs", "micro_bs"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")
    header = ["adaptive_times", "time_list", "memory_list(GB)", "eff_loss_time_per_gpu", "recomp_time_per_gpu"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data_1:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")

    header= ["each_stage_time_breakdown [total (/1000), fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for list in each_stage_time_breakdown:
        print(list)
    print("\n")

    header= ["each_stage_memory_breakdown [total , memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for list in each_stage_memory_breakdown:
        print(list)
    print("\n")
    # for row in data_3:
    #     print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

# pipeline output
def print_configs(config:SearchConfig , args):

    ##### save configs:
    data = []
    data_1 = []
    data_2 = []
    data_3 = []

    stage_num = len(config.stages)
    config_time = max(config.time_list)
    config_mem = max(config.memory_list)
    config_gbs = config.global_bs
    config_mbs = config.micro_bs
    node_info = config.node_info
    adaptive_times = config.adaptive_times
    time_list = config.time_list
    # memory_list = config.memory_list
    memory_list = config.memory_list.copy()
    each_stage_time_breakdown = config.each_stage_time_breakdown.copy()
    gpu_num = 0
    Tflops = 0
    for key in node_info.keys():
        gpu_num += node_info[key]['GPU_NUM']
        Tflops +=  args.device_info[node_info[key]['GPU']]['TFLOPS'] * gpu_num


    for i in range(len(memory_list)):
        memory_list[i]/=1024

    for time_breakdown in each_stage_time_breakdown:
        for i in range(1,len(time_breakdown)):
            time_breakdown[i]/=1000

    eff_loss_time_per_gpu = config.breakdown_eff_loss_time_per_gpu
    recomp_time_per_gpu = config.breakdown_recomp_time_per_gpu
    # each_stage_time_breakdown = config.each_stage_time_breakdown
    each_stage_memory_breakdown = config.each_stage_memory_breakdown





    print(f"node_info {node_info}")
    config_throughput = config_gbs / (config_time)
    data.append([stage_num,f"{config_time}",f"{config_throughput:.2f}", f"{config_mem:.0f}", f"{config_gbs}", f"{config_mbs}" , f"{gpu_num}" , f"{Tflops:.2f}" , f"{config_throughput/Tflops * 1000000:.2f}"])
    data_1.append([f"{adaptive_times}", f"{time_list}", f"{memory_list}", f"{eff_loss_time_per_gpu}", f"{recomp_time_per_gpu}"])
    data_2.append([f"{each_stage_time_breakdown}"])
    data_3.append([f"{each_stage_memory_breakdown}"])

    header = ["# of stages", "est_iteration_time(s)", "estimated_throughput(samples/s)", "est_mem(MB)", "global_bs", "micro_bs", "GPU_NUM", "Total Tflops", "Throughput/TFLOPS *1000"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")
    header = ["adaptive_times", "time_list", "memory_list(GB)", "eff_loss_time_per_gpu", "recomp_time_per_gpu"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data_1:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")

    header= ["each_stage_time_breakdown [total (/1000), fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))
    for row in data_2:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))
    print("\n")

    header= ["each_stage_memory_breakdown [total , memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved]"]
    column_widths = [len(str(header[i])) for i in range(len(header))]
    print('\t'.join(f"{header[i]:<{column_widths[i]}}" for i in range(len(header))))

    for row in data_3:
        print('\t'.join(f"{row[i]:<{column_widths[i]}}" for i in range(len(row))))

def save_and_print_top_hybrid_config (hybrid_config_list: List[HeterogeneousPlan],args):

    #根据hybrid_config_list 中 每一个HeterogeneousPlan 的 max(每一个HeterogeneousPlan.time_list_pipeline)按从小到大进行排序
    pairs = [(max(config.time_list_pipeline), config) for config in hybrid_config_list]
    # 按max_time从小到大排序
    sorted_pairs = sorted(pairs, key=lambda x: x[0])
    # 分离出排序后的配置和对应的时间列表
    sorted_configs = [config for max_time, config in sorted_pairs]
    sorted_max_times = [max_time for max_time, config in sorted_pairs]

    for i in range(min(5, len(sorted_configs))):
        dump_hybrid_config_to_json(sorted_configs[i], f'{args.config_save_path}{args.model_name}_{args.model_size}_NUMPIPE{sorted_configs[i].num_pipelines}_TOP{i}_{args.config_suffix}.json', args)
        print(f"=================TOP DP HETE {i}===================")
        for j in range(sorted_configs[i].num_pipelines):
            print(f"=================TOP DP HETE PIPELINE {j}===================")
            print_configs(sorted_configs[i].pipelines[j], args)






    return sorted_configs , sorted_max_times


def print_search_details(config, args, num_stages, node_order_idx,device_group_idx,num_targets_list, num_hops_list, search_time_list, config_time_list, reserved_mem_list, num_explored_cases):
    if(args.reduce_output or args.test_search_time):
        return
    print(f"\n========== Best Result (num_stages = {num_stages}) node_order_idx ={node_order_idx} device_group_idx = {device_group_idx} ==========")
    if config is not None:
        print_simple_config_info(config, print_recompute_ops=False, print_debug_info=False)
        print(config.history)
        print(f"num_targets: {num_targets_list}")
        print(f"num_hops: {num_hops_list}")
        save_config_info_to_csv(config, reserved_mem_list, f'{args.config_save_path}csv/info_{args.model_name}_{args.model_size}_{config.num_stages}stages_{args.config_suffix}.csv')
        save_distribution_to_csv(num_targets_list, num_hops_list, f'{args.log_path}trends/distribution_{args.model_name}_{args.model_size}_{config.num_stages}stages_{args.config_suffix}.csv')
    else:
        print("No feasible solution.")

    sum_time = 0
    accum_list = []
    for i in range(len(search_time_list)):
        if i == 0:
            accum_list.append(0)
        else:
            time_one_trial = search_time_list[i]
            sum_time += int(time_one_trial+1)
            accum_list.append(sum_time)

    print(f"search time = {sum(search_time_list):.2f} s.  {accum_list} \nnum_explored_cases = {num_explored_cases}")
    # save_search_trend_in_csv(accum_list, config_time_list, f"{args.log_path}trends/{args.model_name}_{args.model_size}_{num_stages}stages_init_{args.initial_point}_{args.max_num_hops}hops_{args.config_suffix}.csv")
    print(f"time trend: {list(map(int, config_time_list))}\n")


timers = Timers()
