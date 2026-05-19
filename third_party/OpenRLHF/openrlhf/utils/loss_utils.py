from typing import Optional

import torch
import torch.distributed as dist


def _get_dp_group(strategy):
    if not dist.is_available() or not dist.is_initialized():
        return None

    device_mesh = getattr(strategy, "ds_device_mesh", None)
    return device_mesh["dp"].get_group() if device_mesh is not None else None


def get_loss_batch_info(
    strategy,
    loss_mask: torch.Tensor,
    *,
    replay_buffer=None,
    step: Optional[int] = None,
    dynamic_batch: bool = False,
    batch_num_tokens: Optional[float] = None,
    global_batch_size: Optional[float] = None,
):
    dp_group = _get_dp_group(strategy)
    dp_size = dist.get_world_size(group=dp_group) if dist.is_available() and dist.is_initialized() else 1

    if dynamic_batch and replay_buffer is not None and step is not None:
        batch_num_tokens = replay_buffer.dynamic_batch_num_tokens[step]
        global_batch_size = replay_buffer.dynamic_global_batch_size[step]

    if batch_num_tokens is None:
        batch_num_tokens = loss_mask.sum().to(loss_mask.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_num_tokens, op=dist.ReduceOp.SUM, group=dp_group)
    else:
        batch_num_tokens = torch.as_tensor(batch_num_tokens, device=loss_mask.device, dtype=torch.float32)

    if global_batch_size is None:
        sample_mask = loss_mask.reshape(loss_mask.shape[0], -1).sum(dim=-1) > 0
        global_batch_size = sample_mask.sum().to(loss_mask.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(global_batch_size, op=dist.ReduceOp.SUM, group=dp_group)
    else:
        global_batch_size = torch.as_tensor(global_batch_size, device=loss_mask.device, dtype=torch.float32)

    return {
        "dp_size": dp_size,
        "batch_num_tokens": batch_num_tokens,
        "global_batch_size": global_batch_size,
    }
