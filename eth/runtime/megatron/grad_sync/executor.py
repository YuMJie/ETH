"""Device-agnostic execution of an immutable gradient synchronization plan."""

import math

import torch

from .runtime_evidence import GradientRuntimeEvidenceRecorder


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


def _normalize_axis(axis, shape, logical_param_id):
    if axis < 0:
        axis += len(shape)
    if not 0 <= axis < len(shape):
        raise ValueError(
            f"logical_param_id {logical_param_id} has invalid partition axis {axis}"
        )
    return axis


class GradientBucketExecutor:
    def __init__(
        self,
        plan,
        global_rank,
        local_descriptors,
        local_runtime_bindings,
        process_group_registry,
        all_reduce_sum,
        runtime_evidence_dir=None,
    ):
        self.plan = plan
        self.global_rank = global_rank
        self._registry = process_group_registry
        self._all_reduce_sum = all_reduce_sum
        self._fragments = {
            item.fragment_id: item for item in plan.canonical_fragments
        }
        self._groups = {
            item.process_group_id: item for item in plan.process_groups
        }
        self._descriptors = self._unique_by_logical_id(
            local_descriptors, "descriptor"
        )
        self._runtime_bindings = self._unique_by_logical_id(
            local_runtime_bindings, "runtime binding"
        )
        self._binding_specs = {
            item.fragment_id: item
            for rank, bindings in plan.local_fragment_bindings_by_rank
            if rank == global_rank
            for item in bindings
        }
        self._validate_local_contract()
        self._scratch = self._allocate_scratch()
        self._runtime_evidence = (
            GradientRuntimeEvidenceRecorder(
                plan,
                global_rank,
                process_group_registry,
                self._scratch,
                runtime_evidence_dir,
            )
            if runtime_evidence_dir
            else None
        )

    @staticmethod
    def _unique_by_logical_id(items, kind):
        result = {}
        for item in items:
            logical_id = item.logical_param_id
            if logical_id in result:
                raise ValueError(f"duplicate {kind} for logical_param_id {logical_id}")
            result[logical_id] = item
        return result

    def _rank_buckets(self):
        for bucket in self.plan.global_bucket_order:
            group = self._groups[bucket.process_group_id]
            if self.global_rank in group.ranks:
                yield bucket

    def _bucket_error(self, bucket, message, binding=None):
        context = f"bucket {bucket.bucket_id} rank {self.global_rank}"
        if binding is not None:
            context += (
                f" logical_param_id {binding.logical_param_id}"
                f" fragment {binding.fragment_id}"
            )
        return ValueError(f"{context}: {message}")

    def _validate_local_contract(self):
        devices = set()
        for bucket in self._rank_buckets():
            for placement in bucket.placements:
                binding = self._binding_specs.get(placement.fragment_id)
                if binding is None:
                    fragment = self._fragments[placement.fragment_id]
                    placeholder = type(
                        "MissingBinding",
                        (),
                        {
                            "logical_param_id": fragment.logical_param_id,
                            "fragment_id": fragment.fragment_id,
                        },
                    )()
                    raise self._bucket_error(bucket, "missing local binding", placeholder)
                runtime = self._runtime_bindings.get(binding.logical_param_id)
                descriptor = self._descriptors.get(binding.logical_param_id)
                if runtime is None or descriptor is None:
                    raise self._bucket_error(
                        bucket, "missing descriptor or main_grad binding", binding
                    )
                if tuple(runtime.main_grad.shape) != tuple(
                    binding.source_segments[0].source_shape
                ):
                    raise self._bucket_error(bucket, "main_grad shape mismatch", binding)
                if _dtype_name(runtime.main_grad.dtype) != bucket.grad_dtype.removeprefix(
                    "torch."
                ):
                    raise self._bucket_error(bucket, "main_grad dtype mismatch", binding)
                runtime_device = torch.device(runtime.main_grad.device)
                binding_device = torch.device(runtime.device)
                if runtime_device != binding_device:
                    raise self._bucket_error(bucket, "main_grad device mismatch", binding)
                devices.add(runtime_device)
        if len(devices) > 1:
            raise ValueError(
                f"rank {self.global_rank} has multiple gradient execution devices: "
                f"{sorted(map(str, devices))}"
            )
        self._device = next(iter(devices), None)

    def _allocate_scratch(self):
        scratch = {}
        requested = {
            dtype: numel
            for (rank, dtype), numel in self.plan.max_scratch_numel_by_rank_and_dtype
            if rank == self.global_rank
        }
        if not requested:
            return scratch
        if self._device is None:
            raise ValueError(
                f"rank {self.global_rank} participates in gradient buckets without a device"
            )
        dtype_by_name = {}
        for runtime in self._runtime_bindings.values():
            dtype_by_name[_dtype_name(runtime.main_grad.dtype)] = runtime.main_grad.dtype
        for dtype_name, numel in requested.items():
            canonical_name = dtype_name.removeprefix("torch.")
            dtype = dtype_by_name.get(canonical_name)
            if dtype is None:
                raise ValueError(
                    f"rank {self.global_rank} has no main_grad with dtype {dtype_name}"
                )
            scratch[dtype_name] = torch.empty(
                numel, dtype=dtype, device=self._device, requires_grad=False
            )
        return scratch

    @property
    def scratch_dtypes(self):
        return tuple(sorted(self._scratch))

    def scratch(self, dtype_name):
        return self._scratch[dtype_name]

    def _segment_view(self, binding, segment):
        runtime = self._runtime_bindings[binding.logical_param_id]
        descriptor = self._descriptors[binding.logical_param_id]
        tensor_parallel = getattr(
            descriptor, "tensor_parallel", descriptor.partition_dim >= 0
        )
        if not tensor_parallel:
            if segment.flattened_numel != runtime.main_grad.numel():
                raise ValueError(
                    f"logical_param_id {binding.logical_param_id} "
                    "replicated source numel mismatch"
                )
            return runtime.main_grad
        axis = _normalize_axis(
            descriptor.partition_dim,
            tuple(runtime.main_grad.shape),
            binding.logical_param_id,
        )
        if math.prod(segment.source_shape) != runtime.main_grad.numel():
            raise ValueError(
                f"logical_param_id {binding.logical_param_id} source shape mismatch"
            )
        return runtime.main_grad.narrow(
            axis, segment.source_dim_offset, segment.source_dim_length
        )

    def _pack(self, bucket, payload):
        for placement in bucket.placements:
            fragment = self._fragments[placement.fragment_id]
            binding = self._binding_specs[placement.fragment_id]
            destination = placement.destination_offset
            copied = 0
            for segment in binding.source_segments:
                source = self._segment_view(binding, segment).reshape(-1)
                if source.numel() != segment.flattened_numel:
                    raise self._bucket_error(
                        bucket, "source segment numel mismatch", binding
                    )
                end = destination + source.numel()
                if end > bucket.exact_numel:
                    raise self._bucket_error(
                        bucket, "placement exceeds exact_numel", binding
                    )
                payload[destination:end].copy_(source)
                if binding.contribution_scale != 1.0:
                    payload[destination:end].mul_(binding.contribution_scale)
                destination = end
                copied += source.numel()
            if copied != fragment.numel:
                raise self._bucket_error(
                    bucket,
                    f"packed {copied} elements, expected {fragment.numel}",
                    binding,
                )

    def _unpack(self, bucket, payload):
        for placement in bucket.placements:
            fragment = self._fragments[placement.fragment_id]
            binding = self._binding_specs[placement.fragment_id]
            source_offset = placement.destination_offset
            copied = 0
            for segment in binding.source_segments:
                target = self._segment_view(binding, segment)
                end = source_offset + target.numel()
                target.copy_(payload[source_offset:end].view(target.shape))
                source_offset = end
                copied += target.numel()
            if copied != fragment.numel:
                raise self._bucket_error(
                    bucket,
                    f"unpacked {copied} elements, expected {fragment.numel}",
                    binding,
                )

    def execute(self, normalization_scale=1.0):
        if self._runtime_evidence is not None:
            self._runtime_evidence.begin_execution(normalization_scale)
        for bucket in self._rank_buckets():
            payload = self._scratch[bucket.grad_dtype][: bucket.exact_numel]
            self._pack(bucket, payload)
            if self._runtime_evidence is not None:
                self._runtime_evidence.observe_packed_bucket(bucket, payload)
            handle = self._registry.handle(bucket.process_group_id)
            self._all_reduce_sum(payload, handle)
            if normalization_scale != 1.0:
                payload.mul_(normalization_scale)
            self._unpack(bucket, payload)
        if self._runtime_evidence is not None:
            self._runtime_evidence.finish_execution()
