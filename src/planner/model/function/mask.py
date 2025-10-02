# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common functions for masking operations."""
from typing import Tuple

import torch


@torch.jit.script
def extract_target(
    data: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extracting the target data based on a given mask.

    Args:
        data (torch.Tensor): The input data of shape ``(*, N, D)``.
        mask (torch.Tensor): The boolean mask of shape ``(*, N)``.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The extracted target data and valid
            mask of shape ``(*, M, D)`` and ``(*, M)``, respectively.
    """
    # print("MASK SHAPE:", mask.shape)
    # flatten the batch dimensions
    # print("DATA SHAPE:", data.shape)
    batch_shape = mask.shape[:-1]
    seq_length = mask.shape[-1]
    dim = data.shape[-1]

    flat_batch_size = int(torch.prod(torch.tensor(batch_shape)).item())
    flat_data = data.view(flat_batch_size, seq_length, dim)
    flat_mask = mask.view(flat_batch_size, seq_length)

    # extract the number of target data for each batch
    counts = flat_mask.sum(dim=1)
    max_counts = int(counts.max().item())

    # create the containers
    output = torch.zeros(
        size=[flat_batch_size, max_counts, dim],
        device=data.device,
        dtype=data.dtype,
    )
    valid = torch.zeros(
        size=[flat_batch_size, max_counts],
        device=data.device,
        dtype=torch.bool,
    )

    # traverse each batch and extract the target data
    batch_indices = torch.arange(flat_batch_size, device=data.device)
    for i in range(max_counts):
        valid_batch = counts > i
        if valid_batch.any():
            cumsum = torch.cumsum(flat_mask[valid_batch], dim=1)
            idx = torch.argmax((cumsum == i + 1).int(), dim=1)
            valid_indices = batch_indices[valid_batch]
            output[valid_indices, i] = flat_data[valid_indices, idx]
            valid[valid_indices, i] = True

    return output, valid
