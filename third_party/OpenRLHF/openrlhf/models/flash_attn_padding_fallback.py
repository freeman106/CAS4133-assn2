"""
Pure-PyTorch substitutes for flash_attn.bert_padding and flash_attn.utils.distributed.all_gather.

Used when the optional `flash_attn` CUDA extension is not installed (e.g. removed due to a
PyTorch ABI mismatch while using vLLM).  Logic is adapted from Dao-AILab/flash-attention
(https://github.com/Dao-AILab/flash-attention) under BSD license.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

# Older PyTorch compatibility (same as flash_attn.utils.distributed)
if "all_gather_into_tensor" not in dir(dist):
    if hasattr(dist, "_all_gather_base"):
        dist.all_gather_into_tensor = dist._all_gather_base  # type: ignore[attr-defined]
if "reduce_scatter_tensor" not in dir(dist):
    if hasattr(dist, "_reduce_scatter_base"):
        dist.reduce_scatter_tensor = dist._reduce_scatter_base  # type: ignore[attr-defined]


def all_gather_raw(input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False):
    world_size = dist.get_world_size(process_group)
    output = torch.empty(
        world_size * input_.shape[0],
        *input_.shape[1:],
        dtype=input_.dtype,
        device=input_.device,
    )
    handle = dist.all_gather_into_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


def reduce_scatter_raw(input_: Tensor, process_group: dist.ProcessGroup, async_op: bool = False):
    world_size = dist.get_world_size(process_group)
    assert input_.shape[0] % world_size == 0
    output = torch.empty(
        input_.shape[0] // world_size,
        *input_.shape[1:],
        dtype=input_.dtype,
        device=input_.device,
    )
    handle = dist.reduce_scatter_tensor(
        output, input_.contiguous(), group=process_group, async_op=async_op
    )
    return output, handle


class IndexFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, indices):
        ctx.save_for_backward(indices)
        assert input.ndim >= 2
        ctx.first_axis_dim, other_shape = input.shape[0], input.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            rearrange(input, "b ... -> b (...)"), 0, repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


index_first_axis = IndexFirstAxis.apply


class IndexPutFirstAxis(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(
            first_axis_dim,
            *values.shape[1:],
            device=values.device,
            dtype=values.dtype,
        )
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


index_put_first_axis = IndexPutFirstAxis.apply


def unpad_input(hidden_states, attention_mask, unused_mask=None):
    all_masks = (attention_mask + unused_mask) if unused_mask is not None else attention_mask
    seqlens_in_batch = all_masks.sum(dim=-1, dtype=torch.int32)
    used_seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(all_masks.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        index_first_axis(rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
        used_seqlens_in_batch,
    )


def pad_input(hidden_states, indices, batch, seqlen):
    output = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return rearrange(output, "(b s) ... -> b s ...", b=batch)


class AllGatherFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: Tensor, process_group: dist.ProcessGroup) -> Tensor:
        ctx.process_group = process_group
        output, _ = all_gather_raw(input_, process_group)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        grad_input, _ = reduce_scatter_raw(grad_output, ctx.process_group)
        return grad_input, None


all_gather = AllGatherFunc.apply
