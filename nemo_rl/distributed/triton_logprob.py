# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Fused Triton kernels for next-token log-probabilities on unsharded logits.

Forward reads each logits row once: an online logsumexp in float32 plus the
target logit. Only the per-token logsumexp is saved. Backward reads each row
once more and writes grad = g * (onehot(target) - softmax) in the logits dtype.
The PyTorch path makes about eight float32 passes per chunk instead.
"""

import torch
import triton
import triton.language as tl

_BLOCK = 4096


@triton.jit
def _logprob_fwd_kernel(
    logits_ptr, stride_row, target_ptr, out_ptr, lse_ptr, V, BLOCK: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    cols = tl.arange(0, BLOCK)
    x = tl.load(base + cols).to(tl.float32)
    m = x
    s = tl.full([BLOCK], 1.0, tl.float32)
    for off in range(BLOCK, V, BLOCK):
        x = tl.load(base + off + cols).to(tl.float32)
        m_new = tl.maximum(m, x)
        s = s * tl.exp(m - m_new) + tl.exp(x - m_new)
        m = m_new
    m_row = tl.max(m, axis=0)
    lse = m_row + tl.log(tl.sum(s * tl.exp(m - m_row), axis=0))
    target = tl.load(target_ptr + row)
    x_t = tl.load(base + target).to(tl.float32)
    tl.store(lse_ptr + row, lse)
    tl.store(out_ptr + row, x_t - lse)


@triton.jit
def _logprob_bwd_kernel(
    logits_ptr,
    stride_row,
    grad_in_ptr,
    stride_grad_row,
    target_ptr,
    lse_ptr,
    grad_out_ptr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    lse = tl.load(lse_ptr + row)
    g = tl.load(grad_out_ptr + row)
    target = tl.load(target_ptr + row)
    x = tl.load(logits_ptr + row * stride_row + cols).to(tl.float32)
    grad = g * ((cols == target).to(tl.float32) - tl.exp(x - lse))
    tl.store(
        grad_in_ptr + row * stride_grad_row + cols,
        grad.to(grad_in_ptr.dtype.element_ty),
    )


def fused_logprob_supported(logits: torch.Tensor) -> bool:
    """Whether FusedLogprob can handle these logits (CUDA, contiguous rows, V % 4096 == 0)."""
    return (
        logits.is_cuda
        and logits.stride(-1) == 1
        and logits.shape[-1] % _BLOCK == 0
        and logits.dtype in (torch.bfloat16, torch.float16, torch.float32)
    )


class FusedLogprob(torch.autograd.Function):
    """log_softmax(logits)[..., target] for unsharded logits, computed in float32.

    Args:
        logits: [..., V] logits in any float dtype; the last dim must be contiguous.
        target: [...] int64 target ids with the same leading shape.

    Returns:
        [...] float32 log-probabilities.
    """

    @staticmethod
    def forward(ctx, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # pyrefly: ignore[bad-override]
        V = logits.shape[-1]
        rows = logits.reshape(-1, V)
        target_flat = target.reshape(-1).to(torch.int64).contiguous()
        n = rows.shape[0]
        out = torch.empty(n, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n, dtype=torch.float32, device=logits.device)
        if n > 0:
            _logprob_fwd_kernel[(n,)](
                rows, rows.stride(0), target_flat, out, lse, V, BLOCK=_BLOCK,
                num_warps=8,
            )
        ctx.save_for_backward(logits, target_flat, lse)
        return out.view(target.shape)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        logits, target_flat, lse = ctx.saved_tensors
        V = logits.shape[-1]
        rows = logits.reshape(-1, V)
        n = rows.shape[0]
        grad_in = torch.empty_like(rows)
        g = grad_output.reshape(-1).to(torch.float32).contiguous()
        if n > 0:
            _logprob_bwd_kernel[(n, V // _BLOCK)](
                rows, rows.stride(0), grad_in, grad_in.stride(0), target_flat, lse, g,
                BLOCK=_BLOCK, num_warps=8,
            )
        return grad_in.view(logits.shape), None
