"""Canonical parameter consistency, norm ownership, and gradient clipping."""

import math

import torch


def copy_shared_tensor_(target, source, logical_param_id):
    if tuple(target.shape) != tuple(source.shape):
        raise ValueError(
            f"shared tensor shape mismatch for {logical_param_id}: "
            f"expected {tuple(target.shape)}, got {tuple(source.shape)}"
        )
    if target.dtype != source.dtype:
        raise ValueError(
            f"shared tensor dtype mismatch for {logical_param_id}: "
            f"expected {target.dtype}, got {source.dtype}"
        )
    if target.device != source.device:
        raise ValueError(
            f"shared tensor device mismatch for {logical_param_id}: "
            f"expected {target.device}, got {source.device}"
        )
    target.copy_(source)
    return target


def merge_shared_optimizer_state_records(gathered_records):
    sources = {}
    for rank_records in gathered_records:
        for key, payload in rank_records:
            if key in sources:
                if key[3] == "scalar" and torch.equal(sources[key], payload):
                    continue
                raise ValueError(
                    f"conflicting shared optimizer state source for {key}"
                )
            sources[key] = payload
    return sources


def assemble_shared_optimizer_state_payload(
    sources,
    optimizer_semantic_id,
    state_name,
    dtype_name,
    target_regions,
    expected_numel,
):
    matching = [
        (key, payload)
        for key, payload in sources.items()
        if key[:1] == (optimizer_semantic_id,)
        and key[2:5] == (state_name, "parameter", dtype_name)
    ]
    if not matching:
        raise ValueError(
            f"missing shared optimizer state source for "
            f"{optimizer_semantic_id}.{state_name}"
        )
    full_shape = tuple(matching[0][0][5])
    axis = int(matching[0][0][6])
    full_tensor = torch.empty(full_shape, dtype=matching[0][1].dtype)
    coverage = torch.zeros(full_shape[axis], dtype=torch.bool)
    for key, payload in matching:
        if tuple(key[5]) != full_shape or int(key[6]) != axis:
            raise ValueError(
                f"shared optimizer layout mismatch for "
                f"{optimizer_semantic_id}.{state_name}"
            )
        offset = 0
        for begin, end in key[1]:
            length = end - begin
            view_shape = list(full_shape)
            view_shape[axis] = length
            numel = math.prod(view_shape)
            if coverage[begin:end].any():
                raise ValueError(
                    f"overlapping shared optimizer state region "
                    f"{begin}:{end} for {optimizer_semantic_id}.{state_name}"
                )
            full_tensor.narrow(axis, begin, length).copy_(
                payload[offset:offset + numel].view(view_shape)
            )
            coverage[begin:end] = True
            offset += numel
        if offset != payload.numel():
            raise ValueError(
                f"shared optimizer payload mismatch for "
                f"{optimizer_semantic_id}.{state_name}"
            )
    payload_parts = []
    for begin, end in target_regions:
        if not coverage[begin:end].all():
            raise ValueError(
                f"missing shared optimizer state region {begin}:{end} for "
                f"{optimizer_semantic_id}.{state_name}"
            )
        payload_parts.append(
            full_tensor.narrow(axis, begin, end - begin).reshape(-1)
        )
    payload = torch.cat(payload_parts) if payload_parts else torch.empty(0)
    if payload.numel() != expected_numel:
        raise ValueError(
            f"shared optimizer state numel mismatch for "
            f"{optimizer_semantic_id}.{state_name}: "
            f"expected {expected_numel}, got {payload.numel()}"
        )
    return payload


def _normalize_axis(axis, shape, logical_param_id):
    if axis < 0:
        axis += len(shape)
    if not 0 <= axis < len(shape):
        raise ValueError(
            f"logical_param_id {logical_param_id} has invalid partition axis {axis}"
        )
    return axis


class CanonicalOptimizerSemantics:
    def __init__(
        self,
        plan,
        global_rank,
        local_descriptors,
        local_runtime_bindings,
        control_device,
    ):
        self.plan = plan
        self.global_rank = global_rank
        self.control_device = torch.device(control_device)
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
        self._logical_id_by_parameter = {
            id(item.parameter): item.logical_param_id
            for item in local_runtime_bindings
        }
        self._optimizer_gradients = {}
        self._binding_specs = {
            item.fragment_id: item
            for rank, bindings in plan.local_fragment_bindings_by_rank
            if rank == global_rank
            for item in bindings
        }

    def bind_optimizer_gradient(self, model_parameter, gradient):
        logical_id = self._logical_id_by_parameter.get(id(model_parameter))
        if logical_id is None:
            raise ValueError("model parameter has no canonical gradient binding")
        if tuple(gradient.shape) != tuple(model_parameter.shape):
            raise ValueError(
                f"optimizer gradient shape mismatch for {logical_id}: "
                f"expected {tuple(model_parameter.shape)}, got {tuple(gradient.shape)}"
            )
        self._optimizer_gradients[logical_id] = gradient

    def _gradient(self, logical_id):
        return self._optimizer_gradients.get(
            logical_id, self._runtime_bindings[logical_id].main_grad
        )

    @staticmethod
    def _unique_by_logical_id(items, kind):
        result = {}
        for item in items:
            if item.logical_param_id in result:
                raise ValueError(
                    f"duplicate {kind} for logical_param_id {item.logical_param_id}"
                )
            result[item.logical_param_id] = item
        return result

    def _segment_views(self, tensor, binding):
        descriptor = self._descriptors[binding.logical_param_id]
        if not descriptor.tensor_parallel:
            if len(binding.source_segments) != 1:
                raise ValueError(
                    f"logical_param_id {binding.logical_param_id} "
                    "replicated binding must have one segment"
                )
            segment = binding.source_segments[0]
            if segment.flattened_numel != tensor.numel():
                raise ValueError(
                    f"logical_param_id {binding.logical_param_id} "
                    "replicated source numel mismatch"
                )
            yield tensor
            return
        axis = _normalize_axis(
            descriptor.partition_dim,
            tuple(tensor.shape),
            binding.logical_param_id,
        )
        for segment in binding.source_segments:
            if math.prod(segment.source_shape) != tensor.numel():
                raise ValueError(
                    f"logical_param_id {binding.logical_param_id} "
                    "source shape mismatch"
                )
            yield tensor.narrow(
                axis, segment.source_dim_offset, segment.source_dim_length
            )

    def _pack_fragment(self, tensor, binding, fragment):
        payload = torch.empty(
            fragment.numel,
            dtype=tensor.dtype,
            device=tensor.device,
            requires_grad=False,
        )
        offset = 0
        for view in self._segment_views(tensor, binding):
            end = offset + view.numel()
            payload[offset:end].copy_(view.reshape(-1))
            offset = end
        if offset != fragment.numel:
            segments = tuple(
                (
                    item.source_dim_offset,
                    item.source_dim_length,
                    item.source_shape,
                    item.flattened_numel,
                )
                for item in binding.source_segments
            )
            raise ValueError(
                f"fragment {fragment.fragment_id} logical_param_id "
                f"{binding.logical_param_id} segments={segments} "
                f"packed {offset}, "
                f"expected {fragment.numel}"
            )
        return payload

    def _unpack_fragment(self, tensor, binding, fragment, payload):
        offset = 0
        for view in self._segment_views(tensor, binding):
            end = offset + view.numel()
            view.copy_(payload[offset:end].view(view.shape))
            offset = end
        if offset != fragment.numel:
            raise ValueError(
                f"fragment {fragment.fragment_id} unpacked {offset}, "
                f"expected {fragment.numel}"
            )

    def synchronize_parameters(self, broadcast):
        for fragment in self.plan.canonical_fragments:
            binding = self._binding_specs.get(fragment.fragment_id)
            if binding is None:
                continue
            runtime = self._runtime_bindings[binding.logical_param_id]
            group = self._groups[fragment.process_group_id]
            source_rank = min(group.ranks)
            payload = self._pack_fragment(
                runtime.parameter.data, binding, fragment
            )
            broadcast(
                payload,
                source_rank,
                fragment.process_group_id,
                group.ranks,
                fragment.fragment_id,
                self.global_rank,
            )
            self._unpack_fragment(
                runtime.parameter.data, binding, fragment, payload
            )

    def synchronize_shared_root_gradients(self, broadcast):
        logical_ids_by_semantic = {}
        for fragment in self.plan.canonical_fragments:
            logical_ids_by_semantic.setdefault(
                fragment.optimizer_semantic_id, set()
            ).add(fragment.logical_param_id)
        shared_semantics = {
            semantic_id
            for semantic_id, logical_ids in logical_ids_by_semantic.items()
            if len(logical_ids) > 1
        }
        for fragment in self.plan.canonical_fragments:
            if fragment.optimizer_semantic_id not in shared_semantics:
                continue
            if not fragment.include_in_optimizer_norm:
                continue
            binding = self._binding_specs.get(fragment.fragment_id)
            if binding is None:
                continue
            runtime = self._runtime_bindings[binding.logical_param_id]
            group = self._groups[fragment.process_group_id]
            source_rank = min(group.ranks)
            payload = self._pack_fragment(
                runtime.main_grad, binding, fragment
            )
            broadcast(
                payload,
                source_rank,
                fragment.process_group_id,
                group.ranks,
                fragment.fragment_id,
                self.global_rank,
            )
            self._unpack_fragment(
                runtime.main_grad, binding, fragment, payload
            )

    def synchronize_optimizer_state(self, state_for_parameter, broadcast):
        for fragment in self.plan.canonical_fragments:
            binding = self._binding_specs.get(fragment.fragment_id)
            if binding is None:
                continue
            runtime = self._runtime_bindings[binding.logical_param_id]
            state = state_for_parameter(runtime.parameter)
            group = self._groups[fragment.process_group_id]
            source_rank = min(group.ranks)
            for state_name in sorted(state):
                state_tensor = state[state_name]
                if not isinstance(state_tensor, torch.Tensor):
                    continue
                if tuple(state_tensor.shape) == tuple(runtime.parameter.shape):
                    payload = self._pack_fragment(
                        state_tensor, binding, fragment
                    )
                    unpack = True
                elif state_tensor.numel() == 1:
                    payload = state_tensor.reshape(-1).clone()
                    unpack = False
                else:
                    raise ValueError(
                        f"optimizer state shape mismatch for "
                        f"{binding.logical_param_id}.{state_name}: "
                        f"{tuple(state_tensor.shape)}"
                    )
                broadcast(
                    payload,
                    source_rank,
                    fragment.process_group_id,
                    group.ranks,
                    f"{fragment.fragment_id}:{state_name}",
                    self.global_rank,
                )
                if unpack:
                    self._unpack_fragment(
                        state_tensor, binding, fragment, payload
                    )
                else:
                    state_tensor.copy_(payload.view(state_tensor.shape))

    def optimizer_state_manifest_for_group(
        self, process_group_id, state_for_parameter
    ):
        manifest = []
        for fragment in self.plan.canonical_fragments:
            if fragment.process_group_id != process_group_id:
                continue
            binding = self._binding_specs.get(fragment.fragment_id)
            if binding is None:
                continue
            runtime = self._runtime_bindings[binding.logical_param_id]
            state = state_for_parameter(runtime.parameter)
            state_manifest = []
            for state_name in sorted(state):
                state_tensor = state[state_name]
                if not isinstance(state_tensor, torch.Tensor):
                    continue
                if tuple(state_tensor.shape) == tuple(runtime.parameter.shape):
                    kind = "parameter"
                elif state_tensor.numel() == 1:
                    kind = "scalar"
                else:
                    raise ValueError(
                        f"optimizer state shape mismatch for "
                        f"{binding.logical_param_id}.{state_name}: "
                        f"{tuple(state_tensor.shape)}"
                    )
                state_manifest.append(
                    (state_name, kind, str(state_tensor.dtype))
                )
            manifest.append((fragment.fragment_id, tuple(state_manifest)))
        return tuple(manifest)

    def synchronize_shared_optimizer_state(
        self, state_for_parameter, all_gather_object, world_size
    ):
        logical_ids_by_semantic = {}
        for fragment in self.plan.canonical_fragments:
            logical_ids_by_semantic.setdefault(
                fragment.optimizer_semantic_id, set()
            ).add(fragment.logical_param_id)
        shared_semantics = {
            semantic_id
            for semantic_id, logical_ids in logical_ids_by_semantic.items()
            if len(logical_ids) > 1
        }

        local_records = []
        for fragment in self.plan.canonical_fragments:
            if fragment.optimizer_semantic_id not in shared_semantics:
                continue
            if not fragment.include_in_optimizer_norm:
                continue
            if fragment.norm_owner_rank != self.global_rank:
                continue
            binding = self._binding_specs[fragment.fragment_id]
            runtime = self._runtime_bindings[binding.logical_param_id]
            state = state_for_parameter(runtime.parameter)
            for state_name in sorted(state):
                state_tensor = state[state_name]
                if not isinstance(state_tensor, torch.Tensor):
                    continue
                if tuple(state_tensor.shape) == tuple(runtime.parameter.shape):
                    kind = "parameter"
                    payload = self._pack_fragment(
                        state_tensor, binding, fragment
                    )
                elif state_tensor.numel() == 1:
                    kind = "scalar"
                    payload = state_tensor.reshape(-1)
                else:
                    raise ValueError(
                        f"optimizer state shape mismatch for "
                        f"{binding.logical_param_id}.{state_name}: "
                        f"{tuple(state_tensor.shape)}"
                    )
                key = (
                    fragment.optimizer_semantic_id,
                    None if kind == "scalar" else fragment.canonical_shard_id,
                    state_name,
                    kind,
                    str(state_tensor.dtype),
                    None if kind == "scalar" else tuple(
                        self._descriptors[binding.logical_param_id].full_shape
                    ),
                    None if kind == "scalar" else self._descriptors[
                        binding.logical_param_id
                    ].partition_dim,
                )
                if kind == "parameter":
                    key = (
                        fragment.optimizer_semantic_id,
                        fragment.global_regions,
                        state_name,
                        kind,
                        str(state_tensor.dtype),
                        tuple(
                            self._descriptors[
                                binding.logical_param_id
                            ].full_shape
                        ),
                        _normalize_axis(
                            self._descriptors[
                                binding.logical_param_id
                            ].partition_dim,
                            self._descriptors[
                                binding.logical_param_id
                            ].full_shape,
                            binding.logical_param_id,
                        ),
                    )
                local_records.append((key, payload.detach().cpu().clone()))

        gathered_records = [None] * world_size
        all_gather_object(gathered_records, tuple(local_records))
        sources = merge_shared_optimizer_state_records(gathered_records)

        for fragment in self.plan.canonical_fragments:
            if fragment.optimizer_semantic_id not in shared_semantics:
                continue
            binding = self._binding_specs.get(fragment.fragment_id)
            if binding is None:
                continue
            runtime = self._runtime_bindings[binding.logical_param_id]
            state = state_for_parameter(runtime.parameter)
            for state_name in sorted(state):
                state_tensor = state[state_name]
                if not isinstance(state_tensor, torch.Tensor):
                    continue
                if tuple(state_tensor.shape) == tuple(runtime.parameter.shape):
                    kind = "parameter"
                elif state_tensor.numel() == 1:
                    kind = "scalar"
                else:
                    raise ValueError(
                        f"optimizer state shape mismatch for "
                        f"{binding.logical_param_id}.{state_name}: "
                        f"{tuple(state_tensor.shape)}"
                    )
                key = (
                    fragment.optimizer_semantic_id,
                    None if kind == "scalar" else fragment.canonical_shard_id,
                    state_name,
                    kind,
                    str(state_tensor.dtype),
                    None if kind == "scalar" else tuple(
                        self._descriptors[binding.logical_param_id].full_shape
                    ),
                    None if kind == "scalar" else self._descriptors[
                        binding.logical_param_id
                    ].partition_dim,
                )
                if kind == "parameter":
                    payload = assemble_shared_optimizer_state_payload(
                        sources,
                        fragment.optimizer_semantic_id,
                        state_name,
                        str(state_tensor.dtype),
                        fragment.global_regions,
                        fragment.numel,
                    )
                else:
                    if key not in sources:
                        raise ValueError(
                            f"missing shared optimizer state source for {key}"
                        )
                    payload = sources[key]
                payload = payload.to(
                    device=state_tensor.device, dtype=state_tensor.dtype
                )
                if kind == "parameter":
                    self._unpack_fragment(
                        state_tensor, binding, fragment, payload
                    )
                else:
                    state_tensor.copy_(payload.view(state_tensor.shape))

    def _owned_gradient_views(self):
        for fragment in self.plan.canonical_fragments:
            if not fragment.include_in_optimizer_norm:
                continue
            if fragment.norm_owner_rank != self.global_rank:
                continue
            binding = self._binding_specs.get(fragment.fragment_id)
            if binding is None:
                raise ValueError(
                    f"norm owner rank {self.global_rank} is missing binding "
                    f"for fragment {fragment.fragment_id}"
                )
            runtime = self._runtime_bindings[binding.logical_param_id]
            yield from self._segment_views(
                self._gradient(binding.logical_param_id), binding
            )

    def _all_local_gradients(self):
        seen = set()
        for logical_id in sorted(self._runtime_bindings):
            gradient = self._gradient(logical_id)
            if id(gradient) in seen:
                continue
            seen.add(id(gradient))
            yield gradient

    def clip_grad_norm(
        self,
        max_norm,
        norm_type,
        all_reduce_sum,
        all_reduce_max,
    ):
        max_norm = float(max_norm)
        norm_type = float(norm_type)
        statistic = torch.zeros(
            (), dtype=torch.float32, device=self.control_device
        )
        if norm_type == float("inf"):
            for gradient in self._owned_gradient_views():
                if gradient.numel():
                    statistic.copy_(
                        torch.maximum(statistic, gradient.detach().abs().max())
                    )
            all_reduce_max(statistic, "inf_norm")
            total_norm = float(statistic.item())
        elif norm_type == 2.0:
            for gradient in self._owned_gradient_views():
                statistic.add_(gradient.detach().float().pow(2).sum())
            all_reduce_sum(statistic, "l2_norm_sq")
            total_norm = math.sqrt(float(statistic.item()))
        else:
            raise ValueError(f"unsupported canonical norm type {norm_type}")

        clip_coefficient = max_norm / (total_norm + 1.0e-6)
        if clip_coefficient < 1.0:
            for gradient in self._all_local_gradients():
                gradient.mul_(clip_coefficient)
        return total_norm

    def count_zeros(self, all_reduce_sum):
        total = torch.zeros(
            (), dtype=torch.float32, device=self.control_device
        )
        for gradient in self._owned_gradient_views():
            total.add_(gradient.numel() - torch.count_nonzero(gradient))
        all_reduce_sum(total, "zero_count")
        return float(total.item())
