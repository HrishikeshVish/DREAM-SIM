# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common functions for geometric operations."""
import torch


@torch.jit.script
def get_topk_neighbors(
    target_xy: torch.Tensor,
    neighbor_xy: torch.Tensor,
    neighbor_valid: torch.Tensor,
    num_neighbors: int,
) -> torch.Tensor:
    """Get the top-k neighbors for each target point.

    Args:
        target_xy (torch.Tensor): Target coordinates of shape `(*, N, 2)`.
        neighbor_xy (torch.Tensor): Neighbor coordinates of shape `(*, M, 2)`.
        neighbor_valid (torch.Tensor): The mask for valid neighbors.
        num_neighbors (int): The number of neighbors to consider.

    Returns:
        torch.Tensor: The indices of top-k neighbors of shape `(*, N, K)`.
    """
    # compute the squared distance between target and neighbor points
    num_neighbors = min(num_neighbors, neighbor_xy.size(-2))
    dist = torch.where(
        neighbor_valid.unsqueeze(-2),
        torch.cdist(target_xy, neighbor_xy, p=2.0),
        torch.tensor(1e8, device=target_xy.device),
    )
    indices = torch.topk(dist, k=num_neighbors, dim=-1, largest=False).indices
    return indices


@torch.jit.script
def wrap_angles(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles in radians to the range [-pi, pi].

    Args:
        angles (torch.Tensor): The input angles in radians.

    Returns:
        torch.Tensor: The wrapped angles.
    """
    return (angles + torch.pi) % (2 * torch.pi) - torch.pi
