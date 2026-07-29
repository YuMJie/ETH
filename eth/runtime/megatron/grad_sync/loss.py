"""Global loss normalization, sample ownership, and training clocks."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LossOutput:
    local_loss_sum: object
    local_loss_weight: object
    nominal_global_loss_scale: float
    metrics: dict


def gpt_loss_output(losses, loss_mask, nominal_global_loss_scale, metrics=None):
    if nominal_global_loss_scale <= 0:
        raise ValueError("nominal_global_loss_scale must be positive")
    flat_losses = losses.float().reshape(-1)
    flat_mask = loss_mask.float().reshape(-1)
    if flat_losses.numel() != flat_mask.numel():
        raise ValueError("GPT loss and loss_mask must have the same numel")
    local_loss_sum = torch.sum(flat_losses * flat_mask)
    local_loss_weight = torch.sum(flat_mask)
    return LossOutput(
        local_loss_sum=local_loss_sum,
        local_loss_weight=local_loss_weight,
        nominal_global_loss_scale=float(nominal_global_loss_scale),
        metrics={} if metrics is None else metrics,
    )


def global_normalization_scale(nominal_global_loss_scale, global_loss_weight):
    if nominal_global_loss_scale <= 0:
        raise ValueError("nominal_global_loss_scale must be positive")
    weight_value = float(global_loss_weight.detach().item())
    if weight_value <= 0:
        raise ValueError("global_loss_weight must be positive")
    return nominal_global_loss_scale / global_loss_weight


def sum_loss_outputs(loss_outputs):
    loss_outputs = tuple(loss_outputs)
    if not loss_outputs:
        raise ValueError("loss_outputs must not be empty")
    local_loss_sum = sum(item.local_loss_sum for item in loss_outputs)
    local_loss_weight = sum(item.local_loss_weight for item in loss_outputs)
    return local_loss_sum, local_loss_weight


def restore_training_clock(checkpoint_args, iteration):
    consumed_train_samples = getattr(
        checkpoint_args, "consumed_train_samples", 0
    )
    return (
        getattr(checkpoint_args, "global_data_cursor", consumed_train_samples),
        getattr(checkpoint_args, "successful_optimizer_steps", iteration),
    )


def validate_training_clock_states(clock_states):
    clock_states = tuple(clock_states)
    if not clock_states:
        raise ValueError("training clock states must not be empty")
    expected = clock_states[0]
    if any(state != expected for state in clock_states[1:]):
        raise ValueError(f"training clock mismatch across ranks: {clock_states}")


def local_sample_slots(
    pipeline_id,
    microbatch_index,
    dp_degree,
    dp_rank,
    mbs_sequences,
    pipeline_sample_offsets,
    global_data_cursor,
):
    if not 0 <= pipeline_id < len(mbs_sequences):
        raise ValueError(f"unknown pipeline_id {pipeline_id}")
    sequence = mbs_sequences[pipeline_id]
    if not 0 <= microbatch_index < len(sequence):
        raise ValueError(f"unknown microbatch_index {microbatch_index}")
    if dp_degree <= 0 or not 0 <= dp_rank < dp_degree:
        raise ValueError("invalid DP coordinate")
    global_mbs = sequence[microbatch_index][0]
    if global_mbs % dp_degree:
        raise ValueError("global microbatch size must be divisible by DP degree")
    local_mbs = global_mbs // dp_degree
    microbatch_prefix = sum(item[0] for item in sequence[:microbatch_index])
    begin = (
        global_data_cursor
        + pipeline_sample_offsets[pipeline_id]
        + microbatch_prefix
        + dp_rank * local_mbs
    )
    return tuple(range(begin, begin + local_mbs))


def synthetic_gpt_batch(sample_ids, sequence_length, vocab_size, device):
    sample_ids = torch.as_tensor(sample_ids, dtype=torch.long, device=device)
    positions = torch.arange(sequence_length, dtype=torch.long, device=device)
    tokens = (
        sample_ids[:, None] * 1103515245 + positions[None, :] * 12345
    ).remainder(vocab_size)
    loss_mask = ((sample_ids[:, None] + positions[None, :]) % 4 != 0).float()
    position_ids = positions[None, :].expand(sample_ids.numel(), -1).clone()
    attention_mask = torch.triu(
        torch.ones(
            sample_ids.numel(),
            1,
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )
    return tokens, loss_mask, position_ids, attention_mask


@dataclass
class TrainingClock:
    total_gbs: int
    global_data_cursor: int = 0
    successful_optimizer_steps: int = 0

    def __post_init__(self):
        if self.total_gbs <= 0:
            raise ValueError("total_gbs must be positive")

    def after_attempt(self, update_successful):
        self.global_data_cursor += self.total_gbs
        if not update_successful:
            return 0
        self.successful_optimizer_steps += 1
        return self.total_gbs
