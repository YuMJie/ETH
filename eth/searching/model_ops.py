# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import csv
op_list = None
full_op_list = None
tunable_op_list = None
no_recompute_op_list = None

PUBLIC_GPT_PROFILE_OPS = (
    "encoder-embedding",
    "enc-1st-layernorm",
    "enc-attention-qkv",
    "enc-flash-attention",
    "enc-attention-dense",
    "enc-post-attention-dropout",
    "enc-2nd-layernorm",
    "enc-MLP-GEMM-1",
    "enc-MLP-gelu",
    "enc-MLP-GEMM-2",
    "enc-post-MLP-dropout",
    "final-layernorm",
    "gpt-post-process",
)

LEGACY_GPT_PROFILE_OPS = (
    "encoder-embedding",
    "enc-1st-layernorm",
    "enc-attention-qkv",
    "enc-attention-score",
    "enc-attention-softmax",
    "enc-attention-dropout",
    "enc-attention-context",
    "enc-attention-dense",
    "enc-post-attention-dropout",
    "enc-2nd-layernorm",
    "enc-MLP-GEMM-1",
    "enc-MLP-gelu",
    "enc-MLP-GEMM-2",
    "enc-post-MLP-dropout",
    "final-layernorm",
    "gpt-post-process",
)

def get_op_list(args):
    global op_list
    gpu_name = args.gpu_type_num_dict["0"]["GPU"]
    if op_list is None:
        op_list = []
        if args.model_name == "scale-layer":
            src_data_file = args.profiled_time_path + f"gpt_scale-layer_mbs{args.micro_batch_size[0]}_tp1_algo0.csv"
        else:
            model_size = args.model_size
            src_data_file = args.profiled_time_path + gpu_name+ f"/{args.model_name}_{model_size}/"+ args.model_name + f"_{model_size}_mbs{args.micro_batch_size[0]}_tp1_algo0.csv"
        with open(src_data_file) as f:
            src_data = csv.reader(f)
            line_index = 0
            for row in src_data:
                line_index += 1
                if line_index > 1:
                    op_list.append(row[0])
    return op_list

def get_ops_per_layer(args):
    base_ops = tuple(get_op_list(args))
    if base_ops == PUBLIC_GPT_PROFILE_OPS:
        return 10
    if base_ops == LEGACY_GPT_PROFILE_OPS:
        return 13
    raise ValueError(
        "profile does not match a supported GPT operator manifest "
        "(expected the public 10-op or compatible 13-op layer schema)"
    )

def get_full_op_list(args):
    global op_list, full_op_list
    if op_list is None:
        op_list = get_op_list(args)
    if full_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            ops_per_layer = get_ops_per_layer(args)
            head_ops = [op_list[0]]
            decoder_layer = op_list[1:1 + ops_per_layer]
            tail_ops = op_list[1 + ops_per_layer:]
            full_op_list = head_ops + decoder_layer * args.num_layers + tail_ops
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return full_op_list

def get_tunable_op_list(args):
    """
    Tunable operators are the oprators which support tensor parallelims, we can tune the tensor-parallelism size.
    """
    global tunable_op_list, op_list
    if tunable_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            tunable_op_list = ["encoder-embedding", "enc-attention-qkv", "enc-attention-dense", "enc-MLP-GEMM-1", "enc-MLP-GEMM-2"]
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return tunable_op_list

def get_no_recompute_op_list(args):
    """
    Operators that should not be recomputed.
    """
    global no_recompute_op_list
    if no_recompute_op_list is None:
        if args.model_name in ["gpt", "scale-layer"]:
            no_recompute_op_list = ["encoder-embedding", "gpt-post-process"]
        else:
            raise RuntimeError(f"model {args.model_name} not supported yet.")
    return no_recompute_op_list

def get_op_spec(op_name, tp_size, dp_size, algo_index, input_spec):
    if op_name in ["encoder-embedding"]:
        if input_spec:
            return None
        else:
            return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["enc-1st-layernorm", "enc-2nd-layernorm"]:
        return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["final-layernorm"]:
        if input_spec:
            return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
        else:
            return {"R": tp_size, "V": 1, "dims": [dp_size, 1, 1]}
    elif op_name in ["enc-attention-qkv", "enc-MLP-GEMM-1"]:
        if algo_index == 0: # column
            if input_spec:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
            else:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
        elif algo_index == 1: # row
            if input_spec:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
            else:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["enc-attention-score"]:
        if input_spec:
            return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
        else:
            return {"R": 1, "V": 1, "dims": [dp_size, tp_size, 1]}
    elif op_name in ["enc-attention-softmax", "enc-attention-dropout"]:
        return {"R": 1, "V": 1, "dims": [dp_size, tp_size, 1]}
    elif op_name in ["enc-attention-context"]:
        if input_spec:
            return {"R": 1, "V": 1, "dims": [dp_size, tp_size, 1]}
        else:
            return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
    elif op_name in ["enc-attention-dense", "enc-MLP-GEMM-2"]:
        if algo_index == 0: # row
            if input_spec:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
            else:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
        elif algo_index == 1: # column
            if input_spec:
                return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
            else:
                return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
    elif op_name in ["enc-post-attention-dropout", "enc-post-MLP-dropout"]:
        return {"R": tp_size, "V": 1, "dims": [1, dp_size, 1]}
    elif op_name in ["enc-MLP-gelu"]:
        return {"R": 1, "V": 1, "dims": [1, dp_size, tp_size]}
    elif op_name in ["gpt-post-process"]:
        if input_spec:
            return {"R": tp_size, "V": 1, "dims": [dp_size, 1, 1]}
        else:
            return None



    
