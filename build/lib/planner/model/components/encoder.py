# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common upstream history encoder module."""
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from planner.data.dataclass import MapPoint, ObjectProperty, Trajectory
from planner.model.components.tokenizer import (
    MapPointTokenizer,
    TrajectoryTokenizer,
)
from planner.model.module.init import variance_scaling
from planner.model.module.layers import layer_norm, linear
from planner.model.module.rst import RSTEncoder


class HistoryEncoderBlock(nn.Module):
    """Repeated self-attention block for the history encoder."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        init_scale: float = 0.2,
    ) -> None:
        super().__init__()

        # save the arguments
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.init_scale = init_scale
        self.num_heads = num_heads
        if self.num_heads is None:
            self.num_heads = hidden_size // 64

        # build the attention and feed-forward layers
        self.attn_norm = layer_norm(normalized_shape=hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            dropout=self.dropout,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(p=self.dropout)
        self.ffn_norm = layer_norm(normalized_shape=self.hidden_size)
        self.ffn = nn.Sequential(
            linear(
                in_features=self.hidden_size,
                out_features=4 * self.hidden_size,
                init_scale=init_scale,
            ),
            nn.SiLU(inplace=True),
            nn.Dropout(p=self.dropout),
            linear(
                in_features=4 * self.hidden_size,
                out_features=self.hidden_size,
                init_scale=init_scale,
            ),
            nn.Dropout(p=self.dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the history encoder block.

        Args:
            x (torch.Tensor): The input feature tensor of shape `(*, T, E)`.
            attn_mask (Optional[torch.Tensor], optional): Optional mask for
                attention weights. Defaults to ``None``.
            key_padding_mask (Optional[torch.Tensor], optional): Optional
                valid mask for keys. Defaults to ``None``.

        Returns:
            torch.Tensor: The output feature tensor of shape `(*, T, E)`.
        """
        out = self.attn_norm.forward(x)
        out = out + self._sa_forward(
            x=out, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        out = out + self._ffn_forward(x=out)

        return out

    def reset_parameters(self) -> None:
        """Reset the parameters of the attention block."""
        if self.attn.in_proj_weight is not None:
            variance_scaling(self.attn.in_proj_weight, scale=self.init_scale)
        else:
            variance_scaling(self.attn.q_proj_weight, scale=self.init_scale)
            variance_scaling(self.attn.k_proj_weight, scale=self.init_scale)
            variance_scaling(self.attn.v_proj_weight, scale=self.init_scale)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        variance_scaling(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        if self.attn.bias_k is not None:
            nn.init.zeros_(self.attn.bias_k)
        if self.attn.bias_v is not None:
            nn.init.zeros_(self.attn.bias_v)

    def _sa_forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.attn.forward(
            query=x,
            key=x,
            value=x,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
        )
        return self.attn_dropout.forward(out)

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn.forward(self.ffn_norm.forward(x))


class HistoryEncoder(nn.Module):
    """A module to encode representations of history trajecotires and map."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_blocks: int = 1,
        num_heads: Optional[int] = None,
        init_scale: float = 0.2,
    ) -> None:
        """Instantiate a new :class:`HistoryEncoder` module.

        Args:
            hidden_size (int): The hidden size of the layer.
            dropout (float, optional): Dropout rate. Defaults to :math:`0.1`.
            num_heads (int, optional): The number of attention heads.
                If ``None``, use ``hidden_size // 64``. Defaults to ``None``.
            init_scale (float, optional): Standard deviation in initialization.
                Defaults to :math:`0.2`.
        """
        super().__init__()
        # save the parameters
        self.hidden_size = hidden_size

        # initialize input tokenizer
        self.map_point_tokenizer = MapPointTokenizer(
            hidden_size=hidden_size, init_scale=init_scale
        )
        self.trajectory_tokenizer = TrajectoryTokenizer(
            hidden_size=hidden_size, init_scale=init_scale
        )

        # initialize the temporal attention module for trajectory encoding
        self.rst_encoder = RSTEncoder(hidden_size=hidden_size)

        # build the attention and feed-forward layers
        self.blocks = nn.ModuleDict(
            {
                f"block_{i}": HistoryEncoderBlock(
                    hidden_size=hidden_size,
                    dropout=dropout,
                    num_heads=num_heads,
                    init_scale=init_scale,
                )
                for i in range(num_blocks)
            }
        )


    def forward(
        self,
        map_point: MapPoint,
        trajectory: Trajectory,
        properties: ObjectProperty,
        current_time: Optional[Union[int, list[int]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the history encoder.

        Args:
            map_point (MapPoint): The map points.
            trajectory (Trajectory): The history trajectory.
            properties (ObjectProperty): The object properties.
            current_time (Optional[int], optional): The current time step.
                Defaults to ``None``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The encoded map and trajectory.
        """
        #print("CURRENT TIME PASSED TO HISTORY ENCODER: ", current_time)
        # tokenize the input
        if current_time is None:
            current_time = trajectory.shape[-1]
        if isinstance(current_time, list):
            assert len(current_time) == 2, (
                "If current_time is a list, it must have exactly two elements."
            )
            assert current_time[0] >= 0 and current_time[1] < trajectory.shape[-1], (
                f"Invalid time steps {current_time} for trajectory with shape {trajectory.shape}."
            )
        elif isinstance(current_time, int):
            assert current_time >= 0 and current_time < trajectory.shape[-1], (
                f"Invalid time step {current_time} "
                f"for trajectory with shape {trajectory.shape}."
            )
        x_map = self.map_point_tokenizer.forward(map_point=map_point)
        #print("CURRENT TIME: ", current_time)
        # print("TRAJECTORY SHAPE: ", trajectory.shape)
        # print("CURRENT TIME: ", current_time)
        # print("MAP POINT: ", map_point.shape)
        # print("Properties: ", properties.shape)
        
        if isinstance(current_time, int):
            x_traj = self.trajectory_tokenizer.forward(
                properties=properties,
                trajectories=trajectory[..., : current_time + 1],
            )
        elif isinstance(current_time, list):
            x_traj = self.trajectory_tokenizer.forward(
                properties=properties,
                trajectories=trajectory[..., current_time],
            )
        #print("TOKENIZED TRAJECTORY SHAPE: ", x_traj.shape)


        if( isinstance(current_time, int)):
        # encode the trajectory
            pos_enc = self.rst_encoder.forward(
                input_xy=trajectory.xy[..., current_time : current_time + 1, :],
                input_yaw=torch.unsqueeze(
                    trajectory.yaw[..., current_time : current_time + 1],
                    dim=-1,
                ),
                other_xy=trajectory.xy[..., : current_time + 1, :],
                other_yaw=torch.unsqueeze(
                    trajectory.yaw[..., : current_time + 1], dim=-1
                ),
                input_t=torch.unsqueeze(
                    trajectory.timestamp_s[..., current_time : current_time + 1],
                    dim=-1,
                ),
                other_t=torch.unsqueeze(
                    trajectory.timestamp_s[..., : current_time + 1], dim=-1
                ),
            )
        elif( isinstance(current_time, list)):
            pos_enc = self.rst_encoder.forward(
                input_xy=trajectory.xy[..., current_time[0]:current_time[0]+1, :],
                input_yaw=torch.unsqueeze(
                    trajectory.yaw[..., current_time[0]:current_time[0]+1],
                    dim=-1,
                ),
                other_xy=trajectory.xy[...,  current_time, :],
                other_yaw=torch.unsqueeze(
                    trajectory.yaw[...,  current_time], dim=-1
                ),
                input_t=torch.unsqueeze(
                    trajectory.timestamp_s[..., current_time[0]:current_time[0]+1],
                    dim=-1,
                ),
                other_t=torch.unsqueeze(
                    trajectory.timestamp_s[..., current_time], dim=-1
                ),
            )

        # forward pass the attention and feed-forward layers
        shape = x_traj.shape
        x_traj = x_traj + pos_enc

        # create key padding mask from the original valid mask
        if isinstance(current_time, int):
            valid = trajectory.valid[..., : current_time + 1]
        elif isinstance(current_time, list):
            valid = trajectory.valid[..., current_time]
        key_padding_mask = torch.zeros_like(valid.float())
        key_padding_mask = ~valid

        # create the lower triangular attention mask where the upper-triangle
        # elements are masked out.
        attn_mask = torch.triu(
            torch.full(
                size=(shape[-2], shape[-2]),
                fill_value=-1e6,
                device=x_traj.device,
            ),
            diagonal=1,
        )

        # flatten the batch dimensions and forward pass the blocks
        x_traj = x_traj.reshape(-1, *shape[-2:])
        key_padding_mask = key_padding_mask.reshape(-1, shape[-2])
        for _, blocks in self.blocks.items():
            x_traj = blocks.forward(
                x=x_traj,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
            )

        # only return the current time step
        if isinstance(current_time, int):
            x_traj = x_traj[..., current_time, :].reshape(*shape[:-2], shape[-1])
        elif isinstance(current_time, list):
            x_traj = x_traj[..., current_time[0], :].reshape(*shape[:-2], shape[-1])

       
        return x_map, x_traj
