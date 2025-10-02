# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common positional encoding functions."""
import math

import torch


@torch.jit.script
def sinusoidal_positional_encoding(data: torch.Tensor) -> torch.Tensor:
    """Computes and applies sinusoidal positional encoding.

    Args:
        data (torch.Tensor): Input tensor of shape ``(*, seq_len, dim)``.

    Returns:
        torch.Tensor: Tensor with sinusoidal positional encoding applied.
    """
    assert data.size(-1) % 2 == 0, "The last dimension of data must be even."

    pos = torch.arange(data.size(-2), dtype=data.dtype, device=data.device)
    denominator = torch.exp(
        torch.arange(0, data.size(-1), 2, dtype=data.dtype, device=data.device)
        * math.log(10000)
        / data.size(-1)
    )

    pe = torch.zeros_like(data)
    pe[..., 0::2] = torch.sin(pos[:, None] / denominator)
    pe[..., 1::2] = torch.cos(pos[:, None] / denominator)

    shape = data.shape
    out = data.reshape(-1, shape[-2], shape[-1]) + pe
    out = out.reshape(shape)

    return out
