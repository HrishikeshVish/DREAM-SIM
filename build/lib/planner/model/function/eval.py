# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common evaluation metrics for motion prediction."""
import torch
from torchmetrics import Metric

__all__ = ["MinADE", "MinFDE"]


# =============================================================================
# Functions
# =============================================================================
# @torch.jit.script
def displacement_error(
    input_xy: torch.Tensor,
    target_xy: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Computes the displacement error between input and target.

    Args:
        input_xy (torch.Tensor): Input coordinates of shape `(*, T, 2)`.
        target_xy (torch.Tensor): Target coordinates of shape `(*, T, 2)`.
        valid (torch.Tensor): The mask for valid time steps of shape `(*, T)`.

    Returns:
        torch.Tensor: The displacement error of shape `(*, T)`.
    """
    return torch.linalg.norm(input_xy - target_xy, ord=2, dim=-1) * valid


# =============================================================================
# Main API
# =============================================================================
class MinADE(Metric):
    """Minimum Average Displacement Error (MinADE) metric."""

    total: torch.Tensor
    """The total displacement error."""
    count: torch.Tensor
    """The total number of evaluated records."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        input_xy: torch.Tensor,
        target_xy: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        ade = displacement_error(input_xy, target_xy, valid).sum(dim=-1)
        steps = valid.sum(dim=-1)
        ade = torch.where(steps > 0, ade / steps, torch.zeros_like(ade))
        self.total += ade.min(dim=-1, keepdim=False).values.sum()
        self.count += ade[..., 0].ravel().size(0)

    def compute(self) -> torch.Tensor:
        return self.total / self.count


class MinFDE(Metric):
    """Minimum Final Displacement Error (MinFDE) metric."""

    total: torch.Tensor
    """The total displacement error."""
    count: torch.Tensor
    """The total number of evaluated records."""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state(
            "total", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        input_xy: torch.Tensor,
        target_xy: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        fde = displacement_error(
            input_xy[..., -1:, :], target_xy[..., -1:, :], valid[..., -1:]
        ).squeeze(-1)
        self.total += fde.min(dim=-1).values.sum()
        self.count += fde[..., 0].ravel().size(0)

    def compute(self) -> torch.Tensor:
        return self.total / self.count
