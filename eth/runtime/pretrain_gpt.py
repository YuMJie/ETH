# coding=utf-8
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# The file has been adapted from the following Megatron-LM file:
# https://github.com/NVIDIA/Megatron-LM/blob/v2.4/pretrain_gpt.py
# Git commit hash: 42c1cf4279acea5a554500dcb552211f44cbec45
# We retain the following copyright from the original files:

# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain GPT"""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.data.gpt_dataset import build_train_valid_test_datasets
from megatron.model import FlexGPTModel
from megatron.training import pretrain
from megatron.grad_sync.loss import gpt_loss_output
from megatron.grad_sync.loss import local_sample_slots
from megatron.grad_sync.loss import synthetic_gpt_batch

def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = FlexGPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def get_batch(data_iterator,current_data_idx):
    """Generate a batch"""

    ## Currently using synthetic data
    args = get_args()
    rank_facts = args.canonical_topology.expected_rank_facts(args.rank)
    args.current_batch_sample_ids = local_sample_slots(
        args.current_pipe,
        current_data_idx,
        rank_facts["dp_degree"],
        rank_facts["dp_rank"],
        args.mbs_seq,
        args.pipeline_sample_offsets,
        args.global_data_cursor,
    )
    return synthetic_gpt_batch(
        args.current_batch_sample_ids,
        args.mbs_seq[args.current_pipe][current_data_idx][1],
        50257,
        torch.device("cuda", torch.cuda.current_device()),
    )

# def get_batch(data_iterator):
#     """Generate a batch"""
#     rank = torch.distributed.get_rank()
#     args = get_args()
#     micro_batch_size = args.micro_batch_size
#     if(rank == 0):
#         print("set batch = 2")
#         micro_batch_size = 4
#     else:
#         print("set batch = 2")
#         micro_batch_size = 2
#     ## Currently using synthetic data
#     vocab_size = 50257
#     tokens = torch.rand((micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
#     # labels = torch.rand((args.micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * vocab_size
#     loss_mask =  (torch.rand((micro_batch_size//mpu.get_op_dp_size(-1), args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5).float()
#     attention_mask = (torch.rand((micro_batch_size, 1, args.seq_length, args.seq_length), requires_grad=False, device=torch.cuda.current_device()) < 0.5)
#     position_ids = torch.rand((micro_batch_size//mpu.get_op_dp_size(0), args.seq_length), requires_grad=False, device=torch.cuda.current_device()).long() * args.seq_length
#     return tokens, loss_mask, position_ids, attention_mask

def loss_func(loss_mask, output_tensor):
    args = get_args()
    losses = output_tensor["output"].float()
    return gpt_loss_output(
        losses,
        loss_mask,
        args.nominal_global_loss_scale,
    )

def forward_step(data_iterator, model, extra_tensors_,current_data_idx):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch. 
    timers('batch-generator').start()
    tokens, loss_mask, position_ids, attention_mask = get_batch(data_iterator,current_data_idx)
    input_tensors = {}
    input_tensors["enc_input_ids"] = tokens
    input_tensors["enc_position_ids"] = position_ids
    extra_tensors = {}
    extra_tensors["enc_attention_mask"] = attention_mask
    # extra_tensors["labels"] = labels
    if extra_tensors_ is not None:
        for key in extra_tensors_:
            extra_tensors[key] = extra_tensors_[key]
            
    timers('batch-generator').stop()

    if mpu.is_pipeline_last_stage():
        output_tensor = model(input_tensors, extra_tensors)
        ouput_extra_tensors = None
    else:
        output_tensor, ouput_extra_tensors = model(input_tensors, extra_tensors)

    return output_tensor, ouput_extra_tensors, partial(loss_func, loss_mask)


forward_step.uses_loss_output_contract = True


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""

    ## Currently using synthetic data
    return None, None, None


train_valid_test_datasets_provider.uses_scheduled_sample_slots = True

    # args = get_args()
    # print_rank_0('> building train, validation, and test datasets '
    #             'for GPT ...')
    # train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
    #     data_prefix=args.data_path,
    #     data_impl=args.data_impl,
    #     splits_string=args.split,
    #     train_valid_test_num_samples=train_val_test_num_samples,
    #     seq_length=args.seq_length,
    #     seed=args.seed,
    #     skip_warmup=(not args.mmap_warmup))
    # print_rank_0("> finished creating GPT datasets ...")

    # return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    forward_step_func = forward_step

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step_func,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})
