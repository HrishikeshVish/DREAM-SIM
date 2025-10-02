# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Relative Space-Time Encoding module."""
import torch
import torch.nn as nn

from planner.asserts import assert_shape
from planner.model.function import wrap_angles
from planner.model.module import layer_norm, linear
from planner.type import OptArray


@torch.jit.script
def get_rst_features(
    input_xy: torch.Tensor,
    input_yaw: torch.Tensor,
    other_xy: torch.Tensor,
    other_yaw: torch.Tensor,
    input_t: torch.Tensor,
    other_t: torch.Tensor,
) -> torch.Tensor:
    """Computes the four-dimensional relative space-time features.

    Args:
        input_xy (torch.Tensor): The target object's position (x, y).
        input_yaw (torch.Tensor): The target object's orientation.
        other_xy (torch.Tensor): The source object's position (x, y).
        other_yaw (torch.Tensor): The source object's orientation.
        input_t (torch.Tensor): The target object's timestamp.
        other_t (torch.Tensor): The source object's timestamp.

    Returns:
        torch.Tensor: The relative space-time features from source to target,
            ``(distance, angle, heading difference, time difference)``.
    """
    displacements = other_xy - input_xy
    euc_dist = torch.linalg.norm(displacements, dim=-1, keepdim=True)
    angle = torch.arccos(
        torch.clip(
            torch.true_divide(
                displacements[..., 0:1] * torch.cos(input_yaw)
                + displacements[..., 1:2] * torch.sin(input_yaw),
                euc_dist + 1e-6,
            ),
            min=-1.0,
            max=1.0,
        )
    )
    heading_difference = wrap_angles(other_yaw - input_yaw)
    time_difference = other_t - input_t
    output = torch.cat(
        [euc_dist, angle, heading_difference, time_difference], dim=-1
    )

    return output


class RSTEncoder(nn.Module):
    """Relative Space-Time encoder."""

    def __init__(self, hidden_size: int, stddev: float = 1.0) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.stddev = stddev

        self.fourier_matrix = nn.Parameter(
            data=stddev * torch.randn(4, hidden_size)
        )
        self.mlp = nn.Sequential(
            linear(2 * hidden_size, hidden_size),
            layer_norm(hidden_size),
            nn.SiLU(inplace=True),
        )

    def forward(
        self,
        input_xy: torch.Tensor,
        input_yaw: torch.Tensor,
        other_xy: torch.Tensor,
        other_yaw: torch.Tensor,
        input_t: OptArray = None,
        other_t: OptArray = None,
    ) -> torch.Tensor:
        self._check(input_xy, input_yaw, other_xy, other_yaw, input_t, other_t)
        if input_t is None:
            input_t = torch.zeros_like(input_xy[..., 0:1])
        if other_t is None:
            other_t = torch.zeros_like(other_xy[..., 0:1])

        out: torch.Tensor = get_rst_features(
            input_xy=input_xy,
            input_yaw=input_yaw,
            other_xy=other_xy,
            other_yaw=other_yaw,
            input_t=input_t,
            other_t=other_t,
        )
        assert out.size(-1) == 4
        base = 2 * torch.pi * torch.matmul(out, self.fourier_matrix)
        out = torch.cat([base.cos(), base.sin()], dim=-1)
        out = self.mlp.forward(out)

        return out

    def _check(
        self,
        input_xy: torch.Tensor,
        input_yaw: torch.Tensor,
        other_xy: torch.Tensor,
        other_yaw: torch.Tensor,
        input_t: OptArray,
        other_t: OptArray,
    ) -> None:
        # check the input tensor shapes
        base_shape = input_xy.shape[:-1]
        assert_shape(
            [input_xy, input_yaw],
            [base_shape + (2,), base_shape + (1,)],
        )
        if input_t is not None:
            assert_shape(input_t, base_shape + (1,))
        base_shape = other_xy.shape[:-1]
        assert_shape(
            [other_xy, other_yaw],
            [base_shape + (2,), base_shape + (1,)],
        )
        if other_t is not None:
            assert_shape(other_t, base_shape + (1,))
