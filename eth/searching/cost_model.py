# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
import math
import os
from .model_ops import get_op_spec, get_op_list, get_no_recompute_op_list
from .config import (
    MAX_VALUE,
    get_boundary_list,
)
from .device_groups import get_memory_limit
from .reporting import timers
import sys
from . import state

# op_list = get_op_list(state.args)
# ops_not_recomputed = get_no_recompute_op_list(args)

# global_mbs_index = None

# global compute_fwd_time, compute_bwd_time, input_size, output_size, weights, activations, collective_time
# global reserved_fwd, reserved_bwd
# global inter_band, intra_band
# global global_mbs_index




def get_mbs_index(mbs):
    assert state.global_mbs_index is not None
    return state.global_mbs_index[mbs]




def identical_spec(input_spec, required_spec):
    identical = True
    if input_spec is None or required_spec is None:
        return identical

    if input_spec["R"] != required_spec["R"]:
        identical = False
    if input_spec["V"] != required_spec["V"]:
        identical = False
    for dim_index in range(len(input_spec["dims"])):
        if input_spec["dims"][dim_index] != required_spec["dims"][dim_index]:
            identical = False

    return identical

def get_reshard_primitives(input_spec, required_spec):
    if identical_spec(input_spec, required_spec):
        return None, None, 0

    if input_spec["R"] > required_spec["R"]:
        ## R -> Dim, split
        for dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][dim_index] < required_spec["dims"][dim_index]:
                assert input_spec["R"] % required_spec["R"] == 0
                num_devices = input_spec["R"] // required_spec["R"]

                return "split", "all_gather", num_devices

    elif input_spec["V"] > required_spec["V"]:
        ## V -> R, all-reduce
        if input_spec["R"] < required_spec["R"]:
            assert input_spec["V"] % required_spec["V"] == 0
            num_devices = input_spec["V"] // required_spec["V"]

            return "all_reduce", None, num_devices

        ## V-> D, reduce-scatter
        for dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][dim_index] < required_spec["dims"][dim_index]:
                assert input_spec["V"] % required_spec["V"] == 0
                num_devices = input_spec["V"] // required_spec["V"]

                return "reduce_scatter", "all_gather", num_devices

    else:
        for src_dim_index in range(len(input_spec["dims"])):
            if input_spec["dims"][src_dim_index] > required_spec["dims"][src_dim_index]:
                ## D -> R, all-gather
                if input_spec["R"] < required_spec["R"]:
                    assert input_spec["dims"][src_dim_index] % required_spec["dims"][src_dim_index] == 0
                    num_devices = input_spec["dims"][src_dim_index] // required_spec["dims"][src_dim_index]

                    return "all_gather", "split", num_devices

                for dst_dim_index in range(len(input_spec["dims"])):
                    ## D -> D, all-to-all
                    if dst_dim_index != src_dim_index and input_spec["dims"][dst_dim_index] < required_spec["dims"][dst_dim_index]:
                        assert input_spec["dims"][src_dim_index] % required_spec["dims"][src_dim_index] == 0
                        num_devices = input_spec["dims"][src_dim_index] // required_spec["dims"][src_dim_index]

                        return "all_to_all", "all_to_all", num_devices

def get_reshard_time(prim, num_devices, data_size,node_id_num):
    assert num_devices > 1
    max_time = -1
    # assert len(node_id_num.keys())==1 , f"only support one gpu type ,but len(node_id_num.keys()) ={len(node_id_num.keys())}"

    for gpu_type in node_id_num.keys():
        if prim in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
            collective_time_ = get_collective_time(gpu_type,prim, num_devices)
            _data_size = '{:.0f}'.format(float(data_size))
            if _data_size in collective_time_:
                max_time = max(max_time, collective_time_[_data_size])
            elif '{:.0f}'.format(float(data_size) - 1) in collective_time_:
                max_time = max(max_time, collective_time_['{:.0f}'.format(float(data_size) - 1)])
            else:
                return 100000
        elif prim in ["split"]:
            return 0
        else:
            return 100000
    return max_time
    # if prim in ["all_reduce", "all_gather", "reduce_scatter", "all_to_all"]:
    #     _data_size = '{:.0f}'.format(float(data_size))
    #     if _data_size in collective_time[gpu_type][prim][num_devices]:
    #         return collective_time[gpu_type][prim][num_devices][_data_size]
    #     elif '{:.0f}'.format(float(data_size) - 1) in collective_time[gpu_type][prim][num_devices]:
    #         return collective_time[gpu_type][prim][num_devices]['{:.0f}'.format(float(data_size) - 1)]
    #     else:
    #         return 100000
    # elif prim in ["split"]:
    #     return 0
    # else:
    #     return 100000

def get_reshard_memory(prim, num_devices, data_size):
    assert num_devices > 1
    if prim == "all_reduce":
        return data_size
    elif prim == "all_gather":
        return data_size * num_devices
    elif prim == "reduce_scatter":
        return data_size
    elif prim == "split":
        return data_size
    elif prim == "all_to_all":
        return data_size * num_devices

def intra_node_band(data_size,node_id_num):
    intra_band = state.intra_band
    # assert len(node_id_num.keys())==1 , f"only support one gpu type,but len(node_id_num.keys()) ={len(node_id_num.keys())}" #TODO
    gpu_type = list(node_id_num.keys())[0]
    if data_size > 0:
        index =  int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(intra_band): #TODO the drawback of ETH
            return intra_band[gpu_type][-1] * 0.001
        else:
            return intra_band[gpu_type][index] * 0.001
    else:
        return 1

def inter_node_band(data_size):
    inter_band = state.inter_band
    if data_size > 0:
        index =  int(math.log(data_size, 2))
        if index >= 1:
            index -= 1
        if index >= len(inter_band):
            return inter_band[-1] * 0.001
        else:
            return inter_band[index] * 0.001
    else:
        return 1

def get_comm_time(node_id_num,in_cross_node,out_cross_node,input_comm_size,output_comm_size): #TODO 没考虑下一个stage的GPU分布情况
    max_gpu_num = 0
    for gpu_type in node_id_num.keys():
        max_gpu_num = max(max_gpu_num,node_id_num[gpu_type])
    input_comm_size*=min(max_gpu_num ,8)
    output_comm_size*=min(max_gpu_num ,8)
    if in_cross_node:
        in_comm = input_comm_size/inter_node_band(input_comm_size)
    else:
        in_comm = input_comm_size/intra_node_band(input_comm_size,node_id_num)
    if out_cross_node:
        out_comm = output_comm_size/inter_node_band(output_comm_size)
    else:
        out_comm = output_comm_size/intra_node_band(output_comm_size,node_id_num)

    return in_comm, out_comm



def get_time_v3(node_id_num,ops, mbs, tp, algo, dp, in_cross_node, out_cross_node,first_stage = False ,last_stage = False):
    args = state.args

    if len(ops) == 0:
        return 0, 0, 0, 0, 0
    compute_fwd_time, compute_bwd_time, input_size, output_size = state.compute_fwd_time , state.compute_bwd_time, state.input_size, state.output_size
    fwd_comp, bwd_comp, in_comm, out_comm, tp_comm = 0, 0, 0, 0, 0

    # 输出gpu_type的key
    # print(node_id_num.keys())

    for i in range(len(ops)):
        op_name = ops[i]
        # print(f"mbs[{i}] {mbs[i]}")
        mbs_index = get_mbs_index(mbs[i])
        tp_index = state.get_tp_index(tp[i])
        algo_index = algo[i]
        max_fwd_comp = 0
        max_bwd_comp = 0
        for node_id in node_id_num.keys():
            # max_fwd_comp = max(max_fwd_comp, compute_fwd_time[node_id][op_name][mbs_index][tp_index][algo_index])
            # max_bwd_comp = max(max_bwd_comp, compute_bwd_time[node_id][op_name][mbs_index][tp_index][algo_index])
            max_fwd_comp = max(max_fwd_comp, get_profiled_data(compute_fwd_time,op_name,mbs_index,tp_index,algo_index,node_id))
            max_bwd_comp = max(max_bwd_comp, get_profiled_data(compute_bwd_time,op_name,mbs_index,tp_index,algo_index,node_id))
        fwd_comp += max_fwd_comp
        bwd_comp += max_bwd_comp
        # fwd_comp += compute_fwd_time[list(node_id_num.keys())[0]][op_name][mbs_index][tp_index][algo_index]
        # bwd_comp += compute_bwd_time[list(node_id_num.keys())[0]][op_name][mbs_index][tp_index][algo_index]
        if args.support_comm_predict and tp[i]>1:
            for op_name_suffix in ["qkv", "dense", "GEMM", "conv", "downsample"]:
                if op_name_suffix in op_name:
                    # tp_comm += get_reshard_time("all_reduce", tp[i], output_size[op_name][mbs_index][tp_index][algo_index],node_id_num) * 1000
                    tp_comm += get_reshard_time("all_reduce", tp[i], get_profiled_data(output_size,op_name,mbs_index,tp_index,algo_index),node_id_num) * 1000

    in_mbs_index = get_mbs_index(mbs[0])
    in_tp_index = state.get_tp_index(tp[0])#TODO
    out_mbs_index = get_mbs_index(mbs[-1])
    out_tp_index = state.get_tp_index(tp[0])
    in_algo_index = algo[0]
    out_algo_index = algo[-1]
    # input_comm_size = input_size[ops[0]][in_mbs_index][in_tp_index][in_algo_index]
    # output_comm_size = output_size[ops[-1]][out_mbs_index][out_tp_index][out_algo_index]
    input_comm_size = get_profiled_data(input_size,ops[0],in_mbs_index,in_tp_index,in_algo_index)
    output_comm_size = get_profiled_data(output_size,ops[-1],out_mbs_index,out_tp_index,out_algo_index)

    if(args.comm_revised):
        in_comm, out_comm = get_comm_time(node_id_num,in_cross_node,out_cross_node,input_comm_size,output_comm_size)
    else:
        if in_cross_node:
            in_comm = input_comm_size/inter_node_band(input_comm_size)
        else:
            in_comm = input_comm_size/intra_node_band(input_comm_size,node_id_num)

        if out_cross_node:
            out_comm = output_comm_size/inter_node_band(output_comm_size)
        else:
            out_comm = output_comm_size/intra_node_band(output_comm_size,node_id_num)
    if first_stage:
        in_comm =0
    if last_stage:
        out_comm =0


    fwd_reshard = 0
    bwd_reshard = 0
    if args.resharding:
        for i in range(1, len(ops)):
            prev_spec = get_op_spec(ops[i-1], tp[i-1], dp[i-1], algo[i-1], input_spec=False)
            current_spec = get_op_spec(ops[i], tp[i], dp[i], algo[i], input_spec=True)
            fwd_prim, bwd_prim, num_devices = get_reshard_primitives(prev_spec, current_spec)
            mbs_index = get_mbs_index(mbs[i])
            tp_index = state.get_tp_index(tp[i])
            if fwd_prim is not None:
                # fwd_reshard += get_reshard_time(fwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][algo[i]],node_id_num)
                fwd_reshard += get_reshard_time(fwd_prim, num_devices, get_profiled_data(input_size,ops[i],mbs_index,tp_index,algo[i]),node_id_num)
            if bwd_prim is not None:
                # bwd_reshard += get_reshard_time(bwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][algo[i]],node_id_num)
                bwd_reshard += get_reshard_time(bwd_prim, num_devices, get_profiled_data(input_size,ops[i],mbs_index,tp_index,algo[i]),node_id_num)
    # print(f"fwd_res:{fwd_reshard}, bwd_res:{bwd_reshard}")

    in_comm += fwd_reshard * 1000
    out_comm += bwd_reshard * 1000

    tp_comm += fwd_reshard * 1000 + bwd_reshard * 1000
    return fwd_comp, bwd_comp, in_comm, out_comm, tp_comm

def get_recompute_time_v3(gpu_type, ops, recompute_ops, mbs, tp, algo):
    if len(ops) == 0 or sum(recompute_ops) == 0:
        return 0
    compute_fwd_time = state.compute_fwd_time
    fwd_comp = 0

    debug_string = ""
    for i in range(len(ops)):
        if recompute_ops[i] == 1:
            debug_string += ops[i] + ", "
            mbs_index = get_mbs_index(mbs[i])
            tp_index = state.get_tp_index(tp[i])
            algo_index = algo[i]
            max_fwd_comp = 0
            for node_id in gpu_type.keys():  #TODO 能否支持异构的DP
                # max_fwd_comp = max(max_fwd_comp, compute_fwd_time[node_id][ops[i]][mbs_index][tp_index][algo_index])
                max_fwd_comp = max(max_fwd_comp, get_profiled_data(compute_fwd_time,ops[i],mbs_index,tp_index,algo_index,node_id))
            fwd_comp += max_fwd_comp
            # fwd_comp += compute_fwd_time[list(gpu_type.keys())[0]][ops[i]][mbs_index][tp_index][algo_index]

    return fwd_comp


def get_profiled_data(dict,ops, in_mbs_index, in_tp_index, algo,gpu_type=None):

    input_size , output_size , weights , activations , reserved_fwd , reserved_bwd = state.input_size, state.output_size, state.weights, state.activations, state.reserved_fwd, state.reserved_bwd

    if (gpu_type is None):
        # max_in_mbs_index = len(dict[ops]) - 1
        max_in_tp_index = len(dict[ops][in_mbs_index]) - 1
        # power_mbs  = 1
        power_tp = 1
        # if(in_mbs_index > max_in_mbs_index):
        #     power_mbs = 2**(in_mbs_index - max_in_mbs_index)
        #     in_mbs_index = max_in_mbs_index
        if(in_tp_index > max_in_tp_index):
            for i in range(in_tp_index - max_in_tp_index ):
                dict[ops][in_mbs_index].append({})
            # dict[ops][in_mbs_index][in_tp_index][algo] = dict[ops][in_mbs_index][max_in_tp_index][algo]/power_tp*1.5 #TODO *1.5
            if(dict == input_size or dict == output_size or dict == weights or dict == activations or dict == reserved_fwd or dict == reserved_bwd):
                dict[ops][in_mbs_index][in_tp_index] = dict[ops][in_mbs_index][max_in_tp_index]
            else:
                dict[ops][in_mbs_index][in_tp_index]= dict[ops][in_mbs_index][max_in_tp_index]
            # in_tp_index = max_in_tp_index

        # if(power_mbs != 1 or power_tp != 1):
        #     print(f"Warning: mbs {in_mbs_index} or tp {in_tp_index} is out of the range max_mbs {len(dict[ops]) - 1} and max_tp {len(dict[ops][in_mbs_index]) - 1} , using the max value instead.")
        try:
            return dict[ops][in_mbs_index][in_tp_index][algo]
        except Exception as e:

            print(e)
            print(sys.exc_info())
            print(f"ops:{ops}, in_mbs_index:{in_mbs_index}, in_tp_index:{in_tp_index}, algo:{algo} ")
            print(f"dict[ops][in_mbs_index][in_tp_index][algo] {dict[ops][in_mbs_index][in_tp_index][algo]} ")
            # print(f"gpu_type: {gpu_type}, ops: {ops}, in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, max_in_tp_index {max_in_tp_index}, algo: {algo}")
            # print("dict[ops]",dict[ops],in_mbs_index,in_tp_index)
            # return dict[ops][in_mbs_index][in_tp_index][algo]
            return MAX_VALUE
            # exit()
    else:
        # max_in_mbs_index = len(dict[gpu_type][ops]) - 1
        max_in_tp_index = len(dict[gpu_type][ops][in_mbs_index]) - 1
        # power_mbs  = 1
        power_tp = 1
        # if(in_mbs_index > max_in_mbs_index):
        #     power_mbs = 2**(in_mbs_index - max_in_mbs_index)
        #     in_mbs_index = max_in_mbs_index
        if(in_tp_index > max_in_tp_index):
            for i in range(in_tp_index - max_in_tp_index ):
                dict[gpu_type][ops][in_mbs_index].append({})
            power_tp = 2**(in_tp_index - max_in_tp_index)
            # dict[gpu_type][ops][in_mbs_index][in_tp_index][algo] = dict[gpu_type][ops][in_mbs_index][max_in_tp_index][algo]/power_tp*1.5 #TODO *1.5
            if(dict == input_size or dict == output_size or dict == weights or dict == activations or dict == reserved_fwd or dict == reserved_bwd):
                dict[gpu_type][ops][in_mbs_index][in_tp_index]= dict[gpu_type][ops][in_mbs_index][max_in_tp_index] #TODO
            else:
                dict[gpu_type][ops][in_mbs_index][in_tp_index] = dict[gpu_type][ops][in_mbs_index][max_in_tp_index]

        # if(power_mbs != 1 or power_tp != 1):
        #     print(f"Warning: mbs {in_mbs_index} or tp {in_tp_index} is out of the range max_mbs {max_in_mbs_index} and max_tp {max_in_tp_index} , using the max value instead.")
        try:
            return dict[gpu_type][ops][in_mbs_index][in_tp_index][algo]
        except Exception as e:
            print(e)
            print(sys.exc_info())
            print(f"gpu_type: {gpu_type}, ops: {ops}, in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, max_in_tp_index {max_in_tp_index}, algo: {algo}")
            # print(f" dict[gpu_type][ops][in_mbs_index][in_tp_index] :{dict[gpu_type][ops][in_mbs_index][in_tp_index]}")
            # print(f" dict[gpu_type][ops] :{dict[gpu_type][ops]}")
            return MAX_VALUE
            return dict[gpu_type][ops][in_mbs_index][in_tp_index][algo]
def get_collective_time(gpu_type,prim, num_devices): #TODO 先用最大的num_devices 替代
    # print(f"gpu_type: {gpu_type}, prim: {prim}, num_devices: {num_devices}")
    collective_time = state.collective_time
    # print(f"max_num_devices: {max_num_devices}")
    max_num_devices = 0
    # print(f"collective_time[{gpu_type}][{prim}]: {collective_time[gpu_type][prim]}")
    # print(f"gpu_type: {gpu_type}, prim: {prim}, num_devices: {num_devices}")
    for key in collective_time[gpu_type][prim].keys():
        max_num_devices = max(max_num_devices, int(key))
        if(int(key) == num_devices):
            return collective_time[gpu_type][prim][key]
    # print(f"Warning: num_devices {num_devices} is out of the range max_num_devices {max_num_devices} , using the max value instead.")
    collective_time[gpu_type][prim][num_devices] = collective_time[gpu_type][prim][max_num_devices] #TODO
    # print(f"gpu_type: {gpu_type}, prim: {prim}, num_devices: {num_devices}, max_num_devices: {max_num_devices}")
    return collective_time[gpu_type][prim][num_devices]



def get_memory_v3(ops, mbs, tp, algo,debug=False):
    args = state.args

    input_size , output_size , weights , activations , reserved_fwd , reserved_bwd = state.input_size, state.output_size, state.weights, state.activations, state.reserved_fwd, state.reserved_bwd
    # print(f"mbs[0] {mbs[0]} ")
    # print(f"tp[0] {tp[0]} ")
    in_mbs_index = get_mbs_index(mbs[0])
    # print(f"in_mbs_index {mbs[0]}")
    # print(f"global_mbs_index {global_mbs_index}")
    # if(tp[0] not in [1,2,4,8]):

    #     return MAX_VALUE ,MAX_VALUE ,MAX_VALUE
    in_tp_index = state.get_tp_index(tp[0])
    # try:
    # inputs = input_size[ops[0]][in_mbs_index][in_tp_index][algo[0]]
    inputs = get_profiled_data(input_size,ops[0],in_mbs_index,in_tp_index,algo[0])
    # except:
    #     # 输出是哪个out of range
    #     print(f"len(input_size): {len(input_size)}, ")
    #     print(f" len(input_size[ops[0]]): {input_size[ops[0]]}, ")
    #     print(f" len(input_size[ops[0]][in_mbs_index]): {len(input_size[ops[0]][in_mbs_index])}, ")
    #     print(f" len(input_size[ops[0]][in_mbs_index][in_tp_index]): {len(input_size[ops[0]][in_mbs_index][in_tp_index])}")
    #     print(f"in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, algo[0]: {algo[0]}")

    #     # len(nput_size[ops[0]][in_mbs_index]): {len(input_size[ops[0]][in_mbs_index])}, len(input_size[ops[0]][in_mbs_index][in_tp_index]): {len(input_size[ops[0]][in_mbs_index][in_tp_index])}")
    #     print(f"in_mbs_index: {in_mbs_index}, in_tp_index: {in_tp_index}, algo[0]: {algo[0]}")
    _activations = 0
    _weights = 0
    for i in range(len(ops)):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = state.get_tp_index(tp[i])
        algo_index = algo[i]
        if args.consider_shared_space and ops[i] == "enc-attention-dropout":
            # _activations += activations[ops[i]][mbs_index][tp_index][algo_index] * 1.5
            _activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,algo_index) * 1.5
            if(debug):
                  print(f"get_memopry_v3 ops{i}: {ops[i]},  activations: {activations[ops[i]][mbs_index][tp_index][algo_index]*1.5}")
        elif args.consider_shared_space and (ops[i] in ["enc-attention-softmax", "bn1"] or "-bn3" in ops[i] or ("-downsample" in ops[i] and "0-0" not in ops[i])):
            _activations += 0
        else:
            # try:
            _activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,algo_index)
            # _activations += activations[ops[i]][mbs_index][tp_index][algo_index]
            if(debug):
                print(f"get_memopry_v3 ops{i}: {ops[i]}, activations: {activations[ops[i]][mbs_index][tp_index][algo_index]}")
            # except:
            #     print(f"len(activations): {len(activations)}, len(ops): {len(ops)}, len(mbs): {len(mbs)}, len(tp): {len(tp)}, len(algo): {len(algo)}")
            #     print(f"ops[i]: {ops[i]}, mbs_index: {mbs_index}, tp_index: {tp_index}, algo_index: {algo_index}")
        # _weights += weights[ops[i]][mbs_index][tp_index][algo_index]
        _weights += get_profiled_data(weights,ops[i],mbs_index,tp_index,algo_index)
    if(debug):
        print(f"get_memopry_v3 activations: {_activations}")
    return _weights, inputs, _activations

def get_activations_v3(ops, recompute_ops, mbs, tp, algo,debug=False):
    args = state.args

    if len(ops) <= 1 or sum(recompute_ops) == 0:
        return 0
    if(debug):
        print("recompute_ops: ",recompute_ops)
    activations = state.activations
    saved_activations = 0
    for i in range(len(ops) - 1):
        if recompute_ops[i] == 1 and  recompute_ops[i+1] == 1:
            mbs_index = get_mbs_index(mbs[i])
            tp_index = state.get_tp_index(tp[i])
            algo_index = algo[i]
            if args.consider_shared_space and ops[i] == "enc-attention-dropout":
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index] * 1.5 #q:为什么是1.5
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,algo_index) * 1.5
                if(debug):
                    print(f"get_activations_v3 ops{i}: {ops[i]}, activations: {activations[ops[i]][mbs_index][tp_index][algo_index]*1.5}")
            elif args.consider_shared_space and (ops[i] in ["enc-attention-softmax", "bn1"]  or "-bn3" in ops[i] or ("-downsample" in ops[i] and "0-0" not in ops[i])):
                saved_activations += 0
            elif ops[i+1] in ["enc-1st-layernorm"] or "-conv1" in ops[i+1]:
                saved_activations += 0
            else:
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index]
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,algo_index)
                if(debug):
                    print(f"get_activations_v3 ops{i}: {ops[i]}, activations: {activations[ops[i]][mbs_index][tp_index][algo_index]}")
        else:
            if(debug):
                print(f"get_activations_v3 ops{i}: {ops[i]}, recompute_ops[i]: {recompute_ops[i]}, recompute_ops[i+1]: {recompute_ops[i+1]}")
    if(debug):
        print(f"get_activations_v3 saved_activations: {saved_activations}")
    return saved_activations

def get_peak_activations(ops, recompute_ops, mbs, tp, algo):
    args = state.args

    if len(ops) <= 1 or sum(recompute_ops) == 0:
        return 0

    activations = state.activations
    saved_activations = 0
    saved_activations_list = [0]
    # 对 recompute_ops 进行遍历，如果两个相邻的 op 都需要重计算，那么计算这两个 op 之间的 saved_activations
    # 如果两个相邻的 op 之间有一个不需要重计算，那么将 saved_activations 加入 saved_activations_list 中

    for i in range(len(ops) - 1):
        if recompute_ops[i] == 1 and  recompute_ops[i+1] == 1:
            mbs_index = get_mbs_index(mbs[i])
            tp_index = state.get_tp_index(tp[i])
            algo_index = algo[i]
            if args.consider_shared_space and (ops[i] in ["enc-attention-softmax", "bn1"] or "-bn3" in ops[i] or ("-downsample" in ops[i] and "0-0" not in ops[i])):
                saved_activations += 0
            elif args.consider_shared_space and ops[i] == "enc-attention-dropout":
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index] * 1.5
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,algo_index) * 1.5
            else:
                # saved_activations += activations[ops[i]][mbs_index][tp_index][algo_index]
                saved_activations += get_profiled_data(activations,ops[i],mbs_index,tp_index,algo_index)
                if ops[i+1] in ["enc-1st-layernorm"] or "-conv1" in ops[i+1] or i + 1 == len(ops) - 1:
                    saved_activations_list.append(saved_activations)
                    saved_activations = 0
        else:
            if saved_activations > 0:
                saved_activations_list.append(saved_activations)
                saved_activations = 0

    return max(saved_activations_list)

def get_reserved_memory(ops, mbs, tp, dp, algo, memory_weights):
    args = state.args
    reserved_fwd, reserved_bwd = state.reserved_fwd, state.reserved_bwd
    input_size , output_size , weights , activations = state.input_size, state.output_size, state.weights, state.activations
    current_reserved_fwd = 0
    current_reserved_bwd = 0
    for i in range(len(ops) - 1):
        mbs_index = get_mbs_index(mbs[i])
        tp_index = state.get_tp_index(tp[i])
        algo_index = algo[i]
        # if reserved_fwd[ops[i]][mbs_index][tp_index][algo_index] > current_reserved_fwd:
        if get_profiled_data(reserved_fwd,ops[i],mbs_index,tp_index,algo_index) > current_reserved_fwd:
            # current_reserved_fwd = reserved_fwd[ops[i]][mbs_index][tp_index][algo_index]
            current_reserved_fwd = get_profiled_data(reserved_fwd,ops[i],mbs_index,tp_index,algo_index)
        # if reserved_bwd[ops[i]][mbs_index][tp_index][algo_index] > current_reserved_bwd:
        if get_profiled_data(reserved_bwd,ops[i],mbs_index,tp_index,algo_index) > current_reserved_bwd:
            # current_reserved_bwd = reserved_bwd[ops[i]][mbs_index][tp_index][algo_index]
            current_reserved_bwd = get_profiled_data(reserved_bwd,ops[i],mbs_index,tp_index,algo_index)

    max_collective = 0
    if args.consider_collective_memory:
        if args.resharding:
            for i in range(1, len(ops)):
                prev_spec = get_op_spec(ops[i-1], tp[i-1], dp[i-1], algo[i-1], input_spec=False)
                current_spec = get_op_spec(ops[i], tp[i], dp[i], algo[i], input_spec=True)
                fwd_prim, bwd_prim, num_devices = get_reshard_primitives(prev_spec, current_spec)
                mbs_index = get_mbs_index(mbs[i])
                tp_index = state.get_tp_index(tp[i])
                if fwd_prim is not None:
                    fwd_collective = get_reshard_memory(fwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][algo[i]])
                    if fwd_collective > max_collective:
                        max_collective = fwd_collective
                if bwd_prim is not None:
                    bwd_collective = get_reshard_memory(bwd_prim, num_devices, input_size[ops[i]][mbs_index][tp_index][algo[i]])
                    if bwd_collective > max_collective:
                        max_collective = bwd_collective

    if args.memory_pred_type == "MAX":
        return max(current_reserved_fwd + current_reserved_bwd, memory_weights) + max_collective
    elif args.memory_pred_type == "MIN":
        return max(current_reserved_fwd, current_reserved_bwd, memory_weights, max_collective)
    else:
        raise RuntimeError(f"unknown args.memory_pred_type {args.memory_pred_type}")

def get_activation_size(op_name, mbs, tp, algo_index=0):
    args = state.args

    activations = state.activations
    mbs_index = get_mbs_index(mbs)
    if(tp>args.max_tp):
        return MAX_VALUE

    tp_index = state.get_tp_index(tp)

    max_tp_index = len(activations[op_name][mbs_index]) - 1
    if tp_index > max_tp_index:
        power_tp = 2**(tp_index - max_tp_index)
        tp_index = max_tp_index
        return activations[op_name][mbs_index][tp_index][algo_index] / power_tp


    return activations[op_name][mbs_index][tp_index][algo_index]

def predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, algo_list, delta=False, on_the_right=False, decrease=True ,first_stage=False,last_stage=False):
    in_cross_node = False
    out_cross_node = False
    mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]
    #TODO 为什么 in_cross_node 和 out_cross_node 都是 False
    fwd_comp, bwd_comp, in_comm, out_comm, _ = get_time_v3(node_id_num,ops, mbs_list, tp_size, algo_list, dp_size, in_cross_node, out_cross_node,first_stage ,last_stage)
    recomp_time = get_recompute_time_v3(node_id_num,ops, recompute_ops, mbs_list, tp_size, algo_list)
    if not delta:
        sum_time = fwd_comp + bwd_comp + in_comm + out_comm + recomp_time
    else:
        if on_the_right and decrease:
            sum_time = fwd_comp + bwd_comp - in_comm + out_comm + recomp_time
        elif not on_the_right and decrease:
            sum_time = fwd_comp + bwd_comp + in_comm - out_comm + recomp_time
        elif on_the_right and not decrease:
            sum_time = fwd_comp + bwd_comp + in_comm - out_comm + recomp_time
        elif not on_the_right and not decrease:
            sum_time = fwd_comp + bwd_comp - in_comm + out_comm + recomp_time
    # print(f"fwd_comp: {fwd_comp}, bwd_comp: {bwd_comp}, in_comm: {in_comm}, out_comm: {out_comm}, recomp_time: {recomp_time}")
    return sum_time/1000

def predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list, breakdown=False,debug =False):
    args = state.args

    mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]
    #q:activations 和 saved_activations 有什么区别
    memory_weights, inputs, activations = get_memory_v3(ops, mbs_list, tp_size, algo_list,debug)
    memory_gradients = memory_weights
    memory_main_params = memory_weights * args.memory_main_params
    memory_optimizer = memory_weights * args.memory_optimizer

    saved_activations = get_activations_v3(ops, recompute_ops, mbs_list, tp_size, algo_list,debug)    # saved_activations 指通过重计算来节省的激活内存
    peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)

    if args.consider_reserved_space:
        memory_reserved = get_reserved_memory(ops, mbs_list, tp_size, dp_size, algo_list, memory_weights)
    else:
        memory_reserved = 0

    memory_activations = (inputs + activations - saved_activations) * (num_stages_behind) #q:为什么不是num_stages_behind+1
    # if(memory_activations<=0):
        # print(f"inputs: {inputs}, activations: {activations}, saved_activations: {saved_activations}, num_stages_behind: {num_stages_behind},len(ops): {len(ops)}")
    memory_peak = inputs + activations - saved_activations + peak_activations

    memory_weights += memory_main_params #q:为什么要加上memory_main_params , memory_main_params是什么  a:memory_main_params是单精度参数所占的内存
    memory_sum = memory_weights + memory_gradients + memory_optimizer + memory_activations + memory_peak + memory_reserved
    if(debug):
        print(f"inputs: {inputs}, activations: {activations}, saved_activations: {saved_activations},(num_stages_behind): {num_stages_behind}")
        print(f"memory_weights: {memory_weights}, memory_gradients: {memory_gradients}, memory_optimizer: {memory_optimizer}, memory_activations: {memory_activations}, memory_peak: {memory_peak}, memory_reserved: {memory_reserved} memory_sum {memory_sum}")
    if breakdown:
        return memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved
    else:
        return memory_sum

def get_cross_node_list(config,inter_comm_gpu):
    cross_node_list = []
    exist_diff_node_gpu_list = []
    current_gpus = 0
    for i in range(config.num_stages):
        stage = config.stages[i]
        num_gpus = stage.num_gpus
        for inter_gpu in inter_comm_gpu:
            if(current_gpus < inter_gpu and current_gpus + num_gpus > inter_gpu):
                # print(f"current_gpus: {current_gpus}, num_gpus: {num_gpus}, inter_gpu: {inter_gpu}")
                exist_diff_node_gpu_list.append(True)
                break
        if(len(exist_diff_node_gpu_list) < i+1):
            exist_diff_node_gpu_list.append(False)
        current_gpus += num_gpus
    # print(exist_diff_node_gpu_list)

    current_gpus = 0
    for i in range(config.num_stages):
        current_gpus += config.stages[i].num_gpus
        if((exist_diff_node_gpu_list[i] or (i+1 < config.num_stages and exist_diff_node_gpu_list[i+1]))or (current_gpus in inter_comm_gpu)):
            cross_node_list.append(True)
        else:
            cross_node_list.append(False)
    return cross_node_list


def predict_time_breakdown(config,print_time=False,args=None, print_memory=False):
    base_batch_size = config.micro_bs
    global_batch_size = config.global_bs
    num_batches = global_batch_size // base_batch_size

    _time_list = []
    memory_list = []
    compute_time_list = []
    efficiency_list = []
    gpu_time_list = []
    breakdown_ideal_time_per_gpu_list = []

    breakdown_pure_comp_time_list = []
    breakdown_pure_eff_loss_time_list = []
    breakdown_pure_recomp_time_list = []

    memory_result_strings = []
    time_result_strings = []

    num_gpus_till_now = 0


    out_cross_node = False

    if(args==None):
        print("args is None")
    inter_comm_gpu = []
    for i in range(len(args.num_gpus_per_node_list)):
        if(i):
            inter_comm_gpu.append(args.num_gpus_per_node_list[i] + inter_comm_gpu[i-1])
        else:
            inter_comm_gpu.append(args.num_gpus_per_node_list[i])

    each_stage_time_breakdown = []
    each_stage_memory_breakdown = []
    last_out_comm = 0

    cross_node_list = get_cross_node_list(config,inter_comm_gpu)


    for i in range(config.num_stages):
        # print(f"stage {i}")
        stage = config.stages[i]
        ops = stage.ops
        num_gpus = stage.num_gpus
        tp_size = stage.tp_size
        dp_size = stage.dp_size
        algo_list = stage.algo
        recompute_ops = stage.recompute_ops
        num_stages_behind = stage.num_stages_behind
        mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]

        # in_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0 and num_gpus_till_now > 0

        #用于判断是否需要跨节点通信
        in_cross_node = out_cross_node

        num_gpus_till_now += num_gpus
        # out_cross_node = (num_gpus_till_now % args.num_gpus_per_node) == 0
        # out_cross_node = (num_gpus_till_now in inter_comm_gpu) and ( i == config.num_stages - 1 or num_gpus_till_now+ config.stages[i+1].num_gpus in inter_comm_gpu)
        out_cross_node = cross_node_list[i]

        # gpu_type = {} key:gpu_type, value:gpu_num
        # gpu_type = {}
        # temp_num_gpu_till_now = num_gpus_till_now - num_gpus
        # temp_num_gpus = num_gpus
        # print("stage {} num_gpus_till_now = {}, num_gpus = {}".format(i, num_gpus_till_now, num_gpus))
        # boundary_list = get_boundary_list(args)
        # print(f"boundary_list: {boundary_list}")
        # print(f"args.gpu_type_list: {args.gpu_type_list}")
        # for boundary in range(len(boundary_list)):
        #     if(temp_num_gpu_till_now <= boundary_list[boundary]):
        #         while(temp_num_gpus > 0):
        #             gpu_type_ = args.gpu_type_list[boundary]
        #             if(gpu_type.get("gpu_type_")==None):
        #                 gpu_type[gpu_type_] = min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
        #             else:
        #                 gpu_type[gpu_type_] += min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
        #             temp = min(temp_num_gpus, boundary_list[boundary]-temp_num_gpu_till_now)
        #             temp_num_gpu_till_now += temp
        #             temp_num_gpus -= temp
        #             boundary+=1
        #             if(gpu_type[gpu_type_] == 0):
        #                 #delete key gpu_type_
        #                 del gpu_type[gpu_type_]
        #         break



        # print(f"stage {i}: gpu_type: {gpu_type}")

        ## compute actual time of each stage
        fwd_comp, bwd_comp, in_comm, out_comm, tp_comm = get_time_v3(stage.node_id_num,ops, mbs_list, tp_size, algo_list, dp_size, in_cross_node, out_cross_node,first_stage = i==0,last_stage=i==config.num_stages-1)# Q：不同stage之间的通信是否会重复计算
        # print(f"{i} last_out_comm {last_out_comm} in_comm {in_comm}")
        if(args.comm_revised):
            if(i):
                if(last_out_comm > in_comm):
                    in_comm = last_out_comm
                else:
                    _time_list[i-1]+= (in_comm-last_out_comm)/1000
                    gpu_time_list[i-1] += (in_comm-last_out_comm) * num_gpus
                    each_stage_time_breakdown[i-1][0] += (in_comm-last_out_comm)/1000
                    each_stage_time_breakdown[i-1][5] += in_comm-last_out_comm
            last_out_comm = out_comm



        recomp_time = get_recompute_time_v3(stage.node_id_num,ops, recompute_ops, mbs_list, tp_size, algo_list)
        sum_time = (fwd_comp + bwd_comp + in_comm + out_comm + recomp_time) / 1000 #TODO 是否可以将communication和compute的时间进行overlap
        _time_list.append(sum_time)
        compute_time_list.append((fwd_comp + bwd_comp + recomp_time) / 1000)
        gpu_time_list.append(sum_time * num_gpus)
        each_stage_time_breakdown.append([sum_time,fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm])
        # print(f"{i} {sum_time,fwd_comp, bwd_comp, recomp_time, in_comm, out_comm, tp_comm}")
        if print_time:
            time_result_strings.append("[stage {}], {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, ".format(i, fwd_comp/1000*num_batches, (bwd_comp + recomp_time)/1000*num_batches, recomp_time/1000*num_batches, in_comm/1000*num_batches, out_comm/1000*num_batches, tp_comm/1000*num_batches))

        ## compute ideal time of each stage
        # print(f"base_batch_size {base_batch_size}")
        _mbs_list = [base_batch_size for _ in range(len(ops))]
        _tp_size = [1 for _ in range(len(ops))]
        _dp_size = [1 for _ in range(len(ops))]
        _fwd_comp, _bwd_comp, _in_comm, _out_comm, _tp_comm = get_time_v3(stage.node_id_num,ops, mbs_list, _tp_size, algo_list, _dp_size, in_cross_node, out_cross_node,first_stage = i==0,last_stage=i==config.num_stages-1) #该stage不考虑并行的时间
        ideal_time = (_fwd_comp + _bwd_comp + _in_comm + _out_comm) / 1000

        ## calculate time breakdown at sum of GPUs. eff_loss_time指的是并行计算的时间损失
        eff_loss_time = (fwd_comp + bwd_comp) - (_fwd_comp + _bwd_comp) / num_gpus #并行计算的时间损失, 该stage的并行计算时间 - 该stage不考虑并行的时间*gpu数， 后置为理想时间

        ## calculate time breakdown per GPU
        breakdown_ideal_time_per_gpu_list.append(((_fwd_comp + _bwd_comp)/num_gpus)/ 1000)
        breakdown_pure_eff_loss_time_list.append(eff_loss_time / 1000)
        breakdown_pure_recomp_time_list.append(recomp_time / 1000)

        ## compute memory
        memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved = \
            predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list, breakdown=True)
        memory_sum = memory_weights + memory_gradients + memory_optimizer + memory_activations + memory_peak + memory_reserved
        memory_list.append(memory_sum)
        if(abs(memory_sum -144004.124) < 0.1):
            print(f"stage {i} memory_sum = {memory_sum}")
        each_stage_memory_breakdown.append([ memory_sum,memory_weights, memory_gradients, memory_optimizer, memory_activations, memory_peak, memory_reserved])
        if print_memory:
            memory_result_strings.append(f"[stage {i}] memory = {memory_sum:.2f} MB. weights = {memory_weights:.0f}, gradients = {memory_gradients:.0f}, optimizer = {memory_optimizer:.0f}, activations = {memory_activations:.0f}, peak += {memory_peak:.0f}, memory_reserved = {memory_reserved:.0f}")

        efficiency_list.append(ideal_time / (sum_time * num_gpus))
    sum_stage_time = sum(_time_list)
    time_list = []
    max_time = 0
    bottleneck = 0
    for i in range(config.num_stages):
        time_stage = (_time_list[i] * (num_batches - 1) + sum_stage_time)
        time_list.append(time_stage)
        if print_time:
            time_result_strings[i] += f"{time_stage:.2f}"
            if time_stage > max_time:
                max_time = time_stage
                bottleneck = i
    if print_time:
        time_result_strings[bottleneck] = " * " + time_result_strings[bottleneck]
        print("overall time = {:.2f} ms".format(max_time))
        print("stage, fwd_comp, bwd_comp+recomp, recomp, in_comm(+reshard), out_comm(+reshard), reshard, sum(us)")
        for i in range(config.num_stages):
            print(time_result_strings[i])

    config.time_list = time_list
    config.memory_list = memory_list
    config.compute_time_list = compute_time_list
    config.total_gpu_time = sum(gpu_time_list) * (num_batches - 1)
    config.breakdown_ideal_time_per_gpu = breakdown_ideal_time_per_gpu_list
    config.breakdown_eff_loss_time_per_gpu = breakdown_pure_eff_loss_time_list
    config.breakdown_recomp_time_per_gpu = breakdown_pure_recomp_time_list
    config.each_stage_memory_breakdown = each_stage_memory_breakdown
    config.each_stage_time_breakdown = each_stage_time_breakdown
    config._time_list = _time_list
    config.sum_stage_time = sum_stage_time
    max_time = max(time_list)
    config.throughput = config.global_bs / (max_time / 1000 )

    # max_mem = args.memory_limit
    efficient_time_list = []
    for i in range(config.num_stages):
        used_time = time_list[i]
        used_memory = memory_list[i]
        device_memory = get_memory_limit(args,config.stages[i].node_id_num)
        idle_time = 0
        if sum(config.stages[i].recompute_ops) > 0:
            idle_time = (max_time - used_time) / 2
        else:
            idle_time_under_max_time = max_time - used_time
            # idle_time_under_max_memory = ((max_mem - used_memory) / used_memory) * used_time
            idle_time_under_max_memory = ((device_memory - used_memory) / used_memory) * used_time
            if idle_time_under_max_memory > idle_time_under_max_time:
                idle_time = idle_time_under_max_time
            else:
                idle_time = idle_time_under_max_memory + (idle_time_under_max_time - idle_time_under_max_memory)/2
        efficient_time_list.append(idle_time * config.stages[i].num_gpus * efficiency_list[i])
    config.efficient_time_list = efficient_time_list

    if print_memory:
        max_memory = 0
        bottleneck = 0
        for i in range(config.num_stages):
            if (memory_list[i]) > max_memory:
                max_memory = memory_list[i]
                bottleneck = i
        memory_result_strings[bottleneck] = " * " + memory_result_strings[bottleneck]
        print("\nmax allocated memory = {:.2f} MB".format(max_memory))
        for i in range(config.num_stages):
            print(memory_result_strings[i])
        print(" ")

    return



def get_reserved_memory_list(config):
    reserved_mem_list = []
    if config is not None:
        base_batch_size = config.micro_bs
        for i in range(config.num_stages):
            stage = config.stages[i]
            ops = stage.ops
            num_gpus = stage.num_gpus
            tp_size = stage.tp_size
            dp_size = stage.dp_size
            algo_list = stage.algo
            recompute_ops = stage.recompute_ops
            num_stages_behind = stage.num_stages_behind
            mbs_list = [base_batch_size//dp_size[j] for j in range(len(ops))]

            _, _, _, _, _, reserved_mem = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list, breakdown=True)
            reserved_mem_list.append(reserved_mem)
    return reserved_mem_list

def predict_value_after_move(config, bottleneck, partner, num_ops_moved, metric, inc_gpus=False, dec_gpus=False, dim=None,debug=False):
    base_batch_size = config.micro_bs
    ops = list(config.stages[bottleneck].ops)
    tp_size = list(config.stages[bottleneck].tp_size)
    dp_size = list(config.stages[bottleneck].dp_size)
    algo_list = list(config.stages[bottleneck].algo)
    recompute_ops = list(config.stages[bottleneck].recompute_ops)
    num_stages_behind = config.stages[bottleneck].num_stages_behind
    num_gpus = config.stages[bottleneck].num_gpus
    node_id_num = config.stages[bottleneck].node_id_num
    if num_ops_moved > 0:
        if bottleneck < partner:
            ops = ops[:-num_ops_moved]
            tp_size = tp_size[:-num_ops_moved]
            dp_size = dp_size[:-num_ops_moved]
            algo_list = algo_list[:-num_ops_moved]
        else:
            ops = ops[num_ops_moved:]
            tp_size = tp_size[num_ops_moved:]
            dp_size = dp_size[num_ops_moved:]
            algo_list = algo_list[num_ops_moved:]

    if inc_gpus:
        if dim == "tp":
            for i in range(len(tp_size)):
                tp_size[i] *= 2
        elif dim == "dp":
            for i in range(len(dp_size)):
                dp_size[i] *= 2
    if dec_gpus:
        if dim == "tp":
            for i in range(len(tp_size)):
                tp_size[i] //= 2
        elif dim == "dp":
            for i in range(len(dp_size)):
                dp_size[i] //= 2

    if num_ops_moved > 0 or inc_gpus or dec_gpus:
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_check_recompute").start()
        recompute_ops = check_recompute(node_id_num,ops, base_batch_size, tp_size, dp_size, num_stages_behind, algo_list,debug=debug)
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_check_recompute").stop()
    if metric in ["time", "time_with_efficiency"]:
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_time").start()
        pred_value = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, algo_list,first_stage=bottleneck==0 ,last_stage=bottleneck==config.num_stages-1)
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_time").stop()
    elif metric == "memory":
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_memory").start()
        pred_value = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list,debug=debug)
        if(num_ops_moved>0):
            timers("prim_dec_op_predict_value_after_move_predict_stage_memory").stop()

    else:
        raise RuntimeError(f"metric {metric} not implemented.")
    return  pred_value, recompute_ops

######## recomputation-related functions #########

stage_memory_set = {}
stage_memory_visit = 0
stage_memory_hit = 0

def predict_stage_memory_helper(config, stage_index, ops=None, recompute_ops=None, tp_size=None, dp_size=None, base_batch_size=None, num_stages_behind=None, algo_list=None):
    global stage_memory_visit, stage_memory_hit, stage_memory_set
    if ops is None:
        ops = config.stages[stage_index].ops
        recompute_ops = config.stages[stage_index].recompute_ops
        tp_size = config.stages[stage_index].tp_size
        dp_size = config.stages[stage_index].dp_size
        base_batch_size = config.stages[stage_index].base_bs
        algo_list = config.stages[stage_index].algo
        num_stages_behind = config.stages[stage_index].num_stages_behind

    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}dp{dp_size}rc{recompute_ops}algo{algo_list}bs{base_batch_size}stage{num_stages_behind}"
    stage_memory_visit += 1
    if stage_memory_set.get(config_str) is not None:
        stage_memory_hit += 1
        return stage_memory_set[config_str]

    pred_memory = predict_stage_memory(ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list)
    config
    stage_memory_set[config_str] = pred_memory

    return pred_memory

stage_time_set = {}
stage_time_visit = 0
stage_time_hit = 0

def predict_stage_time_helper(config, stage_index):
    global stage_time_visit, stage_time_hit, stage_time_set
    ops = config.stages[stage_index].ops
    recompute_ops = config.stages[stage_index].recompute_ops
    tp_size = config.stages[stage_index].tp_size
    dp_size = config.stages[stage_index].dp_size
    base_batch_size = config.micro_bs
    algo_list = config.stages[stage_index].algo
    node_id_num = config.stages[stage_index].node_id_num
    config_str = f"ops{ops[0]}{len(ops)}tp{tp_size}dp{dp_size}rc{recompute_ops}algo{algo_list}bs{base_batch_size}"
    stage_time_visit += 1
    if stage_time_set.get(config_str) is not None:
        stage_time_hit += 1
        return stage_time_set[config_str]

    pred_time = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, algo_list,first_stage=stage_index==0 ,last_stage=stage_index==config.num_stages-1)
    stage_time_set[config_str] = pred_time

    return pred_time

## op_groups: {"op_name": {"index": [], "activation_size":[], "recomputed": False, "sum_size": 0}}
def get_next_recompute_op_group(ops, recompute_ops, base_batch_size, tp_size, dp_size, num_stages_behind, algo_list, op_groups, exceed_memory,debug=False): #TODO ?why TODO
    '''
    从最大的saved activations开始，逐渐减少saved activations，直到内存占用小于阈值
    '''
    max_saved_size = 0
    max_saved_op_name = None
    for op_name in op_groups:
        if not op_groups[op_name]["recomputed"] and op_groups[op_name]["sum_size"] > max_saved_size and op_name not in ["enc-1st-layernorm"] and "-conv1" not in op_name and "-relu" not in op_name and op_name not in get_no_recompute_op_list(state.args):
            max_saved_size = op_groups[op_name]["sum_size"]
            max_saved_op_name = op_name

    if max_saved_size == 0:
        return None, None
    else:
        if(debug):
            print(f"max_saved_op_name: {max_saved_op_name}, max_saved_size: {max_saved_size}")
        op_groups[max_saved_op_name]["recomputed"] = True
        next_op_index = op_groups[max_saved_op_name]["index"][0] + 1
        if next_op_index >= len(ops):
            if(debug):
                print(f"next_op_index: {next_op_index}, len(ops): {len(ops)} , max_saved_op_name: {max_saved_op_name}")
            return None, None
        next_op_name = ops[next_op_index]
        op_groups[next_op_name]["recomputed"] = True

        if max_saved_size < exceed_memory:
            return op_groups[max_saved_op_name]["index"], op_groups[max_saved_op_name]["activation_size"]
        else:
            saved_size = 0
            index = 0
            while saved_size <= exceed_memory and index < len(op_groups[max_saved_op_name]["activation_size"]): #TODO 由于只有当前stage和下个stage的recom 为1才会使用重计算
                saved_size += op_groups[max_saved_op_name]["activation_size"][index]
                index += 1
            return list(op_groups[max_saved_op_name]["index"][:index]), list(op_groups[max_saved_op_name]["activation_size"][:index])

def check_recompute(node_id_num,ops, base_batch_size, tp_size, dp_size, num_stages_behind, algo_list,debug=False):
    args = state.args

    if(args.no_recomp):
        return [0 for _ in range(len(ops))]
    num_ops = len(ops)
    if not args.flex_recompute:
        recompute_ops = [1 for _ in range(num_ops)]
        return recompute_ops
    recompute_ops = [0 for _ in range(num_ops)]
    # 预测该stage的内存占用
    stage_memory = predict_stage_memory_helper(None, None, ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list)
    stage_memory += args.peak_mem_in_backward # args.peak_mem_in_backward = 0

    # 如果内存占用小于阈值，直接全部recompute
    # if stage_memory - args.memory_limit <= 0:
    #     return recompute_ops
    min_memory = get_memory_limit(args,node_id_num)
    # if stage_memory - min_memory <= 0: #TODO 内存够的时候不要重计算
    #     return recompute_ops

    ## generate op groups
    op_groups = {}

    for index in range(len(ops)):
        if ops[index] not in op_groups:
            op_groups[ops[index]] = {"index": [], "activation_size":[], "recomputed": False, "sum_size": 0}
        op_groups[ops[index]]["index"].append(index)
        tmp_activation_size = get_activation_size(ops[index], base_batch_size//dp_size[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1)
        op_groups[ops[index]]["activation_size"].append(tmp_activation_size)
        op_groups[ops[index]]["sum_size"] += tmp_activation_size
    if(debug):
        print(f"op_groups: {op_groups}")
    mbs_list = [base_batch_size // dp_size[i] for i in range(len(dp_size))]

    initial_saved_activations = 0
    initial_peak_activations = 0

    #获得当前的saved activations和peak activations
    current_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)
    # 从最大的saved activations开始，逐渐减少saved activations，直到内存占用小于阈值

    if(debug):
        print(f"exceed_memory: {stage_memory - min_memory}")
    if(stage_memory - min_memory > 0):
        while stage_memory - min_memory > 0:  #TODO 应该考虑计算时间和内存
            saved_index_group, saved_size_list = get_next_recompute_op_group(ops, recompute_ops, base_batch_size, tp_size, dp_size, num_stages_behind, algo_list, op_groups, stage_memory - min_memory,debug=debug)
            if saved_index_group is not None:
                for i in range(len(saved_index_group)):
                    saved_index = saved_index_group[i]
                    if saved_index >= 0:
                        recompute_ops[saved_index] = 1
                        if saved_index < len(ops) - 1:
                            recompute_ops[saved_index + 1] = 1
                            if(debug):
                                print(f"saved_index: {saved_index}")
                    else:
                        break
            else:
                # if(debug):
                break
            new_saved_activations = get_activations_v3(ops, recompute_ops, mbs_list, tp_size, algo_list)
            stage_memory -= (new_saved_activations - initial_saved_activations) * (num_stages_behind + 1)
            initial_saved_activations = new_saved_activations
            new_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)
            stage_memory += (new_peak_activations - current_peak_activations)
            current_peak_activations = new_peak_activations
    else:
        recompute_ops = get_unsaved_ops(ops, recompute_ops, mbs_list, tp_size, algo_list, num_stages_behind,min_memory-stage_memory)

    if(debug):
        print(f"new_saved_activations: {new_saved_activations}, stage_memory: {stage_memory} , min_memory {min_memory}")
    return recompute_ops




# def check_recompute(node_id_num,ops, base_batch_size, tp_size, dp_size, num_stages_behind, algo_list,debug=False):
#     num_ops = len(ops)
#     if not args.flex_recompute:
#         recompute_ops = [1 for _ in range(num_ops)]
#         return recompute_ops
#     recompute_ops = [0 for _ in range(num_ops)]
#     # 预测该stage的内存占用
#     stage_memory = predict_stage_memory_helper(None, None, ops, recompute_ops, tp_size, dp_size, base_batch_size, num_stages_behind, algo_list)
#     stage_memory += args.peak_mem_in_backward # args.peak_mem_in_backward = 0
#     min_memory = get_memory_limit(args,node_id_num)

#     op_groups = {}

#     for index in range(len(ops)):
#         if ops[index] not in op_groups:
#             op_groups[ops[index]] = {"index": [], "activation_size":[], "recomputed": False, "sum_size": 0}
#         op_groups[ops[index]]["index"].append(index)
#         tmp_activation_size = get_activation_size(ops[index], base_batch_size//dp_size[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1)
#         op_groups[ops[index]]["activation_size"].append(tmp_activation_size)
#         op_groups[ops[index]]["sum_size"] += tmp_activation_size
#     if(debug):
#         print(f"op_groups: {op_groups}")
#     mbs_list = [base_batch_size // dp_size[i] for i in range(len(dp_size))]

#     initial_saved_activations = 0
#     initial_peak_activations = 0

#     #获得当前的saved activations和peak activations
#     current_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)
#     # 从最大的saved activations开始，逐渐减少saved activations，直到内存占用小于阈值

#     if(debug):
#         print(f"exceed_memory: {stage_memory - min_memory}")
#     if(stage_memory - min_memory > 0):
#         while stage_memory - min_memory > 0:  #TODO 应该考虑计算时间和内存
#             recompute_ops , end= get_saved_ops(ops, recompute_ops, mbs_list, tp_size, algo_list, num_stages_behind,stage_memory - min_memory)
#             new_saved_activations = get_activations_v3(ops, recompute_ops, mbs_list, tp_size, algo_list)
#             stage_memory -= (new_saved_activations - initial_saved_activations) * (num_stages_behind + 1)
#             initial_saved_activations = new_saved_activations
#             new_peak_activations = get_peak_activations(ops, recompute_ops, mbs_list, tp_size, algo_list)
#             stage_memory += (new_peak_activations - current_peak_activations)
#             current_peak_activations = new_peak_activations
#             if(end):
#                 break
#     else:
#         recompute_ops = get_unsaved_ops(ops, recompute_ops, mbs_list, tp_size, algo_list, num_stages_behind,min_memory-stage_memory)
#     return recompute_ops



def update_recompute(config, stage_idx=None):
    '''
    更新stage的recompute_ops
    '''
    args = state.args

    if(args.no_recomp):
        # print(args)
        return
    if stage_idx is not None:
        updated_stages = [stage_idx]
    else:
        updated_stages = [i for i in range(config.num_stages)]

    for index in updated_stages:
        stage = config.stages[index]
        stage.recompute_ops = check_recompute(stage.node_id_num, stage.ops, config.micro_bs,  #返回在内存限制下的recompute_ops
            stage.tp_size, stage.dp_size, stage.num_stages_behind, stage.algo)

def wrap_predict_delta_time(config, longest_stage, shortest_stage, num_ops_moved, decrease=True):
    base_batch_size = config.micro_bs
    ops = config.stages[longest_stage].ops
    recompute_ops = config.stages[longest_stage].recompute_ops
    tp_size = config.stages[longest_stage].tp_size
    dp_size = config.stages[longest_stage].dp_size
    algo_list = config.stages[longest_stage].algo
    node_id_num = config.stages[longest_stage].node_id_num
    num_ops = len(ops)
    if longest_stage < shortest_stage and decrease:
        ops = ops[num_ops - num_ops_moved:]
        recompute_ops = list(recompute_ops[num_ops - num_ops_moved:])
        tp_size = tp_size[num_ops - num_ops_moved:]
        dp_size = dp_size[num_ops - num_ops_moved:]
        algo_list = algo_list[num_ops - num_ops_moved:]
    elif longest_stage > shortest_stage and decrease:
        ops = ops[0:num_ops_moved]
        recompute_ops = list(recompute_ops[0:num_ops_moved])
        tp_size = tp_size[0:num_ops_moved]
        dp_size = dp_size[0:num_ops_moved]
        algo_list = algo_list[0:num_ops_moved]
    elif longest_stage < shortest_stage and not decrease:
        ops = ops[num_ops - num_ops_moved:]
        recompute_ops = [0 for _ in range(num_ops_moved)]
        tp_size = [config.stages[longest_stage+1].tp_size[0] for _ in range(num_ops_moved) ]
        dp_size = [config.stages[longest_stage+1].dp_size[0] for _ in range(num_ops_moved) ]
        algo_list = algo_list[num_ops - num_ops_moved:]
    elif longest_stage > shortest_stage and not decrease:
        ops = ops[0:num_ops_moved]
        recompute_ops = [0 for _ in range(num_ops_moved)]
        tp_size = [config.stages[longest_stage-1].tp_size[-1] for _ in range(num_ops_moved) ]
        dp_size = [config.stages[longest_stage-1].dp_size[-1] for _ in range(num_ops_moved) ]
        algo_list = algo_list[0:num_ops_moved]
    else:
        raise RuntimeError("")

    pred_delta_time = predict_stage_time(node_id_num,ops, recompute_ops, tp_size, dp_size, base_batch_size, algo_list, delta=True, on_the_right= longest_stage < shortest_stage, decrease=decrease,first_stage=longest_stage==0 ,last_stage=longest_stage==config.num_stages-1)
    return pred_delta_time



def get_unsaved_ops(ops, recompute_ops,mbs,tp_size, algo_list, num_stages_behind, unsave_memory ,debug=False):
    args = state.args

    sort_recom_op_name = state.sort_recom_op_name
    current_unsaved_memory = 0
    mbs_index = get_mbs_index(mbs[0])
    tp_index = state.get_tp_index(tp_size[0])
    recom_op_name_sort = state.get_recom_op_name_sort(args.gpu_type_set[0],mbs_index,tp_index,algo_list[0])   #TODO 不能都是0

    for recom_op_name in reversed(recom_op_name_sort):
        for index in range(len(ops)-1):
            if ops[index] == recom_op_name:
                if recompute_ops[index] == 1 and recompute_ops[index+1] == 1:
                    if args.consider_shared_space and ops[index] == "enc-attention-dropout":
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1) * 1.5
                    elif args.consider_shared_space and (ops[index] in ["enc-attention-softmax", "bn1"]  or "-bn3" in ops[index] or ("-downsample" in ops[index] and "0-0" not in ops[index])):
                        activations_size = 0
                    elif ops[index+1] in ["enc-1st-layernorm"] or "-conv1" in ops[index+1]:
                        activations_size = 0
                    else:
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1)
                    if current_unsaved_memory + activations_size <= unsave_memory:
                        current_unsaved_memory += activations_size
                        recompute_ops[index] = 0
                        if(index+2>len(ops)-1 or recompute_ops[index+2]==0):
                            recompute_ops[index+1]=0

    return recompute_ops


def get_saved_ops(ops, recompute_ops,mbs,tp_size, algo_list, num_stages_behind, save_memory ,debug=False):
    args = state.args

    sort_recom_op_name = state.sort_recom_op_name
    current_saved_memory = 0
    mbs_index = get_mbs_index(mbs[0])
    tp_index = state.get_tp_index(tp_size[0])
    recom_op_name_sort = state.get_recom_op_name_sort(args.gpu_type_set[0],mbs_index,tp_index,algo_list[0])   #TODO 不能都是0

    for recom_op_name in recom_op_name_sort:
        for index in range(len(ops)-1):
            if ops[index] == recom_op_name:
                if recompute_ops[index] == 0 and index != len(ops)-1:
                    if args.consider_shared_space and ops[index] == "enc-attention-dropout":
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1) * 1.5
                    elif args.consider_shared_space and (ops[index] in ["enc-attention-softmax", "bn1"]  or "-bn3" in ops[index] or ("-downsample" in ops[index] and "0-0" not in ops[index])):
                        activations_size = 0
                    elif ops[index+1] in ["enc-1st-layernorm"] or "-conv1" in ops[index+1]:
                        activations_size = 0
                    else:
                        activations_size = get_activation_size(ops[index], mbs[index], tp_size[index], algo_list[index]) * (num_stages_behind + 1)
                    recompute_ops[index] = 1
                    recompute_ops[index+1] = 1
                    current_saved_memory += activations_size

                    if current_saved_memory  >= save_memory:
                        return recompute_ops, False

    return recompute_ops,True
