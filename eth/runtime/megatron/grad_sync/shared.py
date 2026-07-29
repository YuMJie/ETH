import torch


def merge_shared_tensor_groups(
    tensor_groups, split_dim, *, average_replicas=False
):
    if not tensor_groups:
        raise ValueError("shared tensor merge requires at least one group")
    if split_dim == -1:
        values = [group[0] for group in tensor_groups]
    else:
        values = [torch.cat(group, dim=split_dim) for group in tensor_groups]
    if len(values) == 1:
        return values[0]
    merged = torch.stack(values).sum(dim=0)
    if average_replicas:
        merged.div_(len(values))
    return merged
