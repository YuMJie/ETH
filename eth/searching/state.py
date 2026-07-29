from dataclasses import dataclass, field
from typing import List

from .model_ops import get_op_list


@dataclass
class SearchStage:
    index: int
    num_stages_behind: int
    num_gpus: int
    ops: List[str]
    recompute_ops: List[str]
    tp_size: List[int]
    dp_size: List[int]
    algo: List[int]
    node_id_num: dict #key: node_id, value: node_num

@dataclass
class SearchConfig:
    global_bs: int
    micro_bs: int
    stages: List[SearchStage]
    num_stages: int
    history: str = ""

    time_list: List[float] = field(default_factory=list)
    memory_list: List[float] = field(default_factory=list)
    compute_time_list: List[float] = field(default_factory=list)
    total_gpu_time: float = 0

    breakdown_ideal_time_per_gpu: List[float] = field(default_factory=list)
    breakdown_eff_loss_time_per_gpu: List[float] = field(default_factory=list)
    breakdown_recomp_time_per_gpu: List[float] = field(default_factory=list)

    ## for choosing partner stages according to efficient time
    efficient_time_list: List[float] = field(default_factory=list)
    ## used for adaptive model
    adaptive_times: int = 0

    each_stage_time_breakdown: List[List[float]] = field(default_factory=list)
    each_stage_memory_breakdown: List[List[float]] = field(default_factory=list)


@dataclass
class HeterogeneousPlan:

    num_pipelines: int
    pipelines: List[SearchConfig]  = field(default_factory=list)
    global_bs_pipeline: List[int] = field(default_factory=list)
    micro_bs_pipeline: List[int] = field(default_factory=list)
    num_mb_pipeline: List[int] = field(default_factory=list)
    time_list_pipeline : List[float] = field(default_factory=list)


import os
import csv


args = None
global sort_recom_op
global sort_recom_op_name
global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, collective_time
global reserved_fwd, reserved_bwd
global inter_band, intra_band
global global_mbs_index
global op_list
supported_tp_sizes = []


def get_tp_index(tp_size, supported_values=None):
    if supported_values is None:
        supported_values = supported_tp_sizes
    try:
        return supported_values.index(tp_size)
    except ValueError as error:
        supported = ", ".join(str(value) for value in supported_values)
        raise ValueError(
            f"unsupported tensor parallel size {tp_size}; supported values: {supported}"
        ) from error



def get_mbs_index(mbs):
    global global_mbs_index
    assert global_mbs_index is not None
    return global_mbs_index[mbs]


def set_profiled_time(profile_time):
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index,op_list ,collective_time,inter_band, intra_band
    global supported_tp_sizes

    (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band )= profile_time
    if args is not None:
        max_profiled_tp = min(args.max_tp, 4 if args.model_size == "350M" else 8)
        supported_tp_sizes = [tp for tp in (1, 2, 4, 6, 8) if tp <= max_profiled_tp]


def read_profiled_time(model_name, model_size, time_path,args):
    global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index
    global op_list

    op_list = get_op_list(args)

    mbs_list = args.micro_batch_size
    global_mbs_index = {}
    for i in range(len(mbs_list)):
        global_mbs_index[mbs_list[i]] = i
    # print(f"global_mbs_index: {global_mbs_index}")
    if model_name == "gpt" and model_size == "350M":
        max_tp_size = min(args.max_tp, 4)
    else:
        max_tp_size = min(args.max_tp, 8)

    tp_size_list = [tp for tp in (1, 2, 4, 6, 8) if tp <= max_tp_size]
    globals()["supported_tp_sizes"] = tp_size_list
    comm_num_gpus_list = tp_size_list[1:]

    algo_list = [0, 1]

    compute_fwd_time = {}
    compute_bwd_time = {}
    input_size = {}
    output_size = {}
    weights = {}
    activations = {}
    reserved_fwd = {}
    reserved_bwd = {}


    for gpu_type in args.gpu_type_set:
        compute_fwd_time[gpu_type] = {}
        compute_bwd_time[gpu_type] = {}
        # input_size = {}
        # output_size = {}
        # weights = {}
        # activations = {}
        # reserved_fwd = {}
        # reserved_bwd = {}

        for op_name in op_list:
            compute_fwd_time[gpu_type][op_name] = []
            compute_bwd_time[gpu_type][op_name] = []
            input_size[op_name] = []
            output_size[op_name] = []
            weights[op_name] = []
            activations[op_name] = []

            reserved_fwd[op_name] = []
            reserved_bwd[op_name] = []

            for i in range(len(mbs_list)):
                compute_fwd_time[gpu_type][op_name].append([])
                compute_bwd_time[gpu_type][op_name].append([])
                input_size[op_name].append([])
                output_size[op_name].append([])
                weights[op_name].append([])
                activations[op_name].append([])

                reserved_fwd[op_name].append([])
                reserved_bwd[op_name].append([])

                for j in range(len(tp_size_list)):
                    compute_fwd_time[gpu_type][op_name][i].append([])
                    compute_bwd_time[gpu_type][op_name][i].append([])
                    input_size[op_name][i].append([])
                    output_size[op_name][i].append([])
                    weights[op_name][i].append([])
                    activations[op_name][i].append([])

                    reserved_fwd[op_name][i].append([])
                    reserved_bwd[op_name][i].append([])

                    for k in range(len(algo_list)):
                        compute_fwd_time[gpu_type][op_name][i][j].append(1000000)
                        compute_bwd_time[gpu_type][op_name][i][j].append(1000000)
                        input_size[op_name][i][j].append(1000000)
                        output_size[op_name][i][j].append(1000000)
                        weights[op_name][i][j].append(1000000)
                        activations[op_name][i][j].append(1000000)

                        reserved_fwd[op_name][i][j].append(1000000)
                        reserved_bwd[op_name][i][j].append(1000000)

    for gpu_type in args.gpu_type_set:
        for mbs in mbs_list:
            for tp in tp_size_list:
                mbs_index = get_mbs_index(mbs)
                tp_index = get_tp_index(tp, tp_size_list)
                for algo_index in algo_list:
                    if model_name == "scale-layer":
                        src_data_file = time_path + f"gpt_scale-layer_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
                    else:
                        # src_data_file = time_path + model_name + f"_{model_size}_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
                        src_data_file = time_path + f"{gpu_type}/"+f"{model_name}_{model_size}/"+ model_name + f"_{model_size}_mbs{mbs}_tp{tp}_algo{algo_index}.csv"
                    if not os.path.exists(src_data_file):
                        continue
                    with open(src_data_file) as f:
                        src_data = csv.reader(f)
                        line_index = 0
                        for row in src_data:
                            line_index += 1
                            if line_index > 1:
                                op_name = row[0]
                                compute_fwd_time[gpu_type][op_name][mbs_index][tp_index][algo_index] = float(row[1])
                                compute_bwd_time[gpu_type][op_name][mbs_index][tp_index][algo_index] = float(row[2])
                                input_size[op_name][mbs_index][tp_index][algo_index] = float(row[3])
                                output_size[op_name][mbs_index][tp_index][algo_index] = float(row[4])
                                weights[op_name][mbs_index][tp_index][algo_index] = float(row[5])
                                activations[op_name][mbs_index][tp_index][algo_index] = float(row[6])

                                if args.consider_reserved_space:
                                    reserved_fwd[op_name][mbs_index][tp_index][algo_index] = float(row[7])
                                    reserved_bwd[op_name][mbs_index][tp_index][algo_index] = float(row[8])
    global collective_time
    collective_time = {}
    prim_list = ["all_gather", "all_reduce", "reduce_scatter", "all_to_all"]

    for gpu_type in args.gpu_type_set:
        collective_time[gpu_type] = {}
        for prim in prim_list:
            collective_time[gpu_type][prim] = {}
            for num_gpus in comm_num_gpus_list:
                collective_time[gpu_type][prim][num_gpus] = {}
                if model_name == "scale-layer":
                    src_data_file = time_path + f"prim_gpt_scale-layer_{prim}_{num_gpus}gpus.csv"
                else:
                    src_data_file = time_path + gpu_type+ f"/{args.model_name}_{model_size}/"+ f"prim_{model_name}_{model_size}_{prim}_{num_gpus}gpus.csv"
                    # print(f"src_data_file: {src_data_file}")
                if(os.path.exists(src_data_file)): #TODO Q：为什么读取A100的信息后，结果没有变化
                    with open(src_data_file) as f:
                        src_data = csv.reader(f)
                        line_index = 0
                        for row in src_data:
                            line_index += 1
                            if line_index > 1:
                                data_size = row[0]
                                # print(f"gpu_type: {gpu_type}, prim: {prim}, num_gpus: {num_gpus}, data_size: {data_size}")
                                collective_time[gpu_type][prim][num_gpus][data_size] = float(row[1])
                else:
                    print(f"file ({src_data_file}) not exist.")
    global inter_band, intra_band
    intra_band = {}
    inter_band_file = time_path +"p2p_inter_node.csv"
    for gpu_type in args.gpu_type_set:
        intra_band_file = time_path + gpu_type + "/p2p_intra_node.csv"
        try:
            with open(intra_band_file) as f:
                src_data = csv.reader(f)
                for idx, row in enumerate(src_data):
                    if idx == 1:
                        intra_band[gpu_type] = [float(row[i]) for i in range(len(row))]
            # print(f"intra-node bandwidth {gpu_type} = {intra_band[gpu_type]}")
        except FileNotFoundError:
            print(f"intra-node bandwidth file:{intra_band_file} is not found.")

    try:
        with open(inter_band_file) as f:
            src_data = csv.reader(f)
            for idx, row in enumerate(src_data):
                if idx == 1:
                    inter_band = [float(row[i]) for i in range(len(row))]
    except FileNotFoundError:
        print(f"inter-node bandwidth file is not found, using intra-node bandwidth instead.")
        if not intra_band:
            raise RuntimeError("no P2P bandwidth profile is available")
        inter_band = next(iter(intra_band.values()))

    return (compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, reserved_fwd, reserved_bwd, global_mbs_index ,op_list ,collective_time,inter_band, intra_band)





def get_sort_op_by_size_comp(args):
    global sort_recom_op
    global sort_recom_op_name
    sort_recom_op_name ={}
    sort_recom_op ={}
    mbs_list = args.micro_batch_size
    global_mbs_index = {}
    for i in range(len(mbs_list)):
        global_mbs_index[mbs_list[i]] = i
    # print(f"global_mbs_index: {global_mbs_index}")
    if args.model_name == "gpt" and args.model_size == "350M":
        max_tp_size = min(args.max_tp, 4)
    else:
        max_tp_size = min(args.max_tp, 8)
    algo_list = [0, 1]

    tp_size_list = [tp for tp in (1, 2, 4, 6, 8) if tp <= max_tp_size]
    # print("max_tp_size: ", max_tp_size)
    # print("tp_size_list: ", tp_size_list)
    for gpu_type in args.gpu_type_set:
        sort_recom_op[gpu_type] = {}
        sort_recom_op_name[gpu_type] = {}
        for i in range(len(mbs_list)):
            sort_recom_op[gpu_type][i] = {}
            sort_recom_op_name[gpu_type][i] = {}
            for j in range(len(tp_size_list)):
                sort_recom_op[gpu_type][i][j] = {}
                sort_recom_op_name[gpu_type][i][j] = {}
                for k in range(len(algo_list)):
                    sort_recom_op[gpu_type][i][j][k] = {}
                    sort_recom_op_name[gpu_type][i][j][k] = {}
                    for op_name in op_list:
                        sort_recom_op[gpu_type][i][j][k][op_name]= activations[op_name][i][j][k] / compute_fwd_time[gpu_type][op_name][i][j][k]
    for gpu_type in args.gpu_type_set:
        for i in range(len(mbs_list)):
            for j in range(len(tp_size_list)):
                for k in range(len(algo_list)):
                    # 根据saved activations/comp排序名字
                    sort_recom_op_name[gpu_type][i][j][k] = sorted(sort_recom_op[gpu_type][i][j][k], key=sort_recom_op[gpu_type][i][j][k].get, reverse=True)

def get_recom_op_name_sort(gpu_type, mbs_index, tp_index, algo_index):
    global sort_recom_op_name
    tp_profiles = sort_recom_op_name[gpu_type][mbs_index]
    if tp_index not in tp_profiles:
        raise ValueError(f"tensor parallel profile index {tp_index} is unavailable")
    return tp_profiles[tp_index][algo_index]
