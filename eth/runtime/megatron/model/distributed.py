# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

"""Local DDP wrappers with optimizer-visible contiguous main gradients."""

from abc import ABC
from abc import abstractmethod

import torch
import torch.distributed
from torch._utils import _flatten_dense_tensors
from torch._utils import _unflatten_dense_tensors

from megatron import mpu
from .module import MegatronModule


class MemoryBuffer:
    def __init__(self, numel, dtype, device):
        self.numel = numel
        self.dtype = dtype
        self.device = torch.device(device)
        self.data = torch.zeros(
            numel,
            dtype=dtype,
            device=self.device,
            requires_grad=False,
        )

    def zero(self):
        self.data.zero_()

    def get(self, shape, start_index):
        end_index = start_index + shape.numel()
        if start_index < 0 or end_index > self.numel:
            raise ValueError(
                f"gradient buffer slice [{start_index}, {end_index}) exceeds {self.numel}"
            )
        return self.data[start_index:end_index].view(shape)


class DistributedDataParallelBase(MegatronModule, ABC):
    def __init__(self, module):
        super().__init__()
        self.module = module

    @abstractmethod
    def allreduce_gradients(self):
        pass

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.module.state_dict(destination, prefix, keep_vars)

    def state_dict_for_save_checkpoint(
        self, destination=None, prefix="", keep_vars=False
    ):
        return self.module.state_dict_for_save_checkpoint(
            destination, prefix, keep_vars
        )

    def load_state_dict(self, state_dict, strict=True):
        self.module.load_state_dict(state_dict, strict=strict)


class _MainGradBufferMixin:
    def _initialize_main_grad_buffers(self):
        dtype_numel = {}
        dtype_device = {}
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            dtype = (
                torch.float32
                if self.accumulate_allreduce_grads_in_fp32
                else param.dtype
            )
            device = torch.device(param.device)
            previous_device = dtype_device.setdefault(dtype, device)
            if previous_device != device:
                raise ValueError(
                    f"parameters with gradient dtype {dtype} span devices "
                    f"{previous_device} and {device}"
                )
            dtype_numel[dtype] = dtype_numel.get(dtype, 0) + param.numel()

        self._grad_buffers = {
            dtype: MemoryBuffer(numel, dtype, dtype_device[dtype])
            for dtype, numel in dtype_numel.items()
        }
        remaining = dict(dtype_numel)
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            dtype = (
                torch.float32
                if self.accumulate_allreduce_grads_in_fp32
                else param.dtype
            )
            remaining[dtype] -= param.numel()
            param.main_grad = self._grad_buffers[dtype].get(
                param.shape, remaining[dtype]
            )

        self.grad_accs = []
        for param in self.module.parameters():
            if not param.requires_grad:
                continue
            expanded = param.expand_as(param)
            grad_acc = expanded.grad_fn.next_functions[0][0]
            grad_acc.register_hook(self._make_param_hook(param))
            self.grad_accs.append(grad_acc)

    @staticmethod
    def _make_param_hook(param):
        def param_hook(*unused):
            if param.grad is not None:
                param.main_grad.add_(param.grad.data)
                param.grad = None

        return param_hook

    def zero_grad_buffer(self):
        for buffer in self._grad_buffers.values():
            buffer.zero()

    def _fallback_allreduce(self):
        data_parallel_size = mpu.get_data_parallel_world_size()
        data_parallel_group = mpu.get_data_parallel_group()
        for buffer in self._grad_buffers.values():
            buffer.data.div_(data_parallel_size)
            torch.distributed.all_reduce(buffer.data, group=data_parallel_group)


class DistributedDataParallel_Profile(
    _MainGradBufferMixin, DistributedDataParallelBase
):
    """Compatibility wrapper used by the existing profiler entrypoints."""

    def __init__(
        self,
        module,
        accumulate_allreduce_grads_in_fp32,
        use_contiguous_buffers,
    ):
        super().__init__(module)
        self.accumulate_allreduce_grads_in_fp32 = (
            accumulate_allreduce_grads_in_fp32
        )
        self.use_contiguous_buffers = use_contiguous_buffers
        self._grad_buffers = {}
        if use_contiguous_buffers:
            self._initialize_main_grad_buffers()

    def allreduce_gradients(self):
        if self._grad_buffers:
            self._fallback_allreduce()
            return

        buckets = {}
        for param in self.module.parameters():
            if param.requires_grad and param.grad is not None:
                buckets.setdefault((param.dtype, param.device), []).append(param)
                param.main_grad = param.grad
        for params in buckets.values():
            grads = [param.grad.data for param in params]
            coalesced = _flatten_dense_tensors(grads)
            coalesced.div_(mpu.get_data_parallel_world_size())
            torch.distributed.all_reduce(
                coalesced, group=mpu.get_data_parallel_group()
            )
            for grad, synced in zip(
                grads, _unflatten_dense_tensors(coalesced, grads)
            ):
                grad.copy_(synced)


class DistributedDataParallel(_MainGradBufferMixin, DistributedDataParallelBase):
    """Local DDP that always exposes stable contiguous ``main_grad`` views."""

    def __init__(
        self,
        module,
        accumulate_allreduce_grads_in_fp32,
        use_contiguous_buffers,
    ):
        super().__init__(module)
        self.accumulate_allreduce_grads_in_fp32 = (
            accumulate_allreduce_grads_in_fp32
        )
        self.use_contiguous_buffers = use_contiguous_buffers
        self._gradient_sync_executor = None
        self._initialize_main_grad_buffers()

    def set_gradient_sync_executor(self, executor):
        self._gradient_sync_executor = executor

    def allreduce_gradients(self):
        if self._gradient_sync_executor is not None:
            self._gradient_sync_executor.execute()
        else:
            self._fallback_allreduce()
