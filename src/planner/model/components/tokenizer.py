# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Input tokenization modules."""
import torch
import torch.nn as nn

from planner.data.dataclass import (
    MapPoint,
    MapPolylineType,
    ObjectProperty,
    ObjectType,
    Trajectory,
)
from planner.model.function import wrap_angles
from planner.model.module import embedding, layer_norm, linear

__all__ = ["MapPointTokenizer", "TrajectoryTokenizer"]


class MapPointTokenizer(nn.Module):
    """Map point tokenization module."""

    def __init__(self, hidden_size: int, init_scale: float = 0.2) -> None:
        super(MapPointTokenizer, self).__init__()

        self.polyline_type_embedding = embedding(
            num_embeddings=len(MapPolylineType),
            embedding_dim=len(MapPolylineType) // 2,
            init_scale=init_scale,
        )
        self.mlp = nn.Sequential(
            linear(3 + len(MapPolylineType) // 2, hidden_size),
            layer_norm(hidden_size),
            nn.SiLU(inplace=True),
            linear(hidden_size, hidden_size),
            layer_norm(hidden_size),
            nn.SiLU(inplace=True),
        )

    def forward(self, map_point: MapPoint) -> torch.Tensor:
        """Tokenizes the map point into embeddings.

        Args:
            map_point (MapPoint): The input map points.

        Returns:
            torch.Tensor: The tokenized map points.
        """
        pnt_type = torch.where(
            map_point.valid,
            map_point.types,
            torch.full_like(map_point.types, MapPolylineType.UNDEFINED),
        )
        pnt_type = self.polyline_type_embedding.forward(pnt_type)
        x = torch.cat([map_point.dir_xyz, pnt_type], dim=-1)
        out = self.mlp.forward(x)

        return out


class TrajectoryTokenizer(nn.Module):
    """Trajectory tokenization module."""

    def __init__(self, hidden_size: int, init_scale: float = 0.2) -> None:
        super(TrajectoryTokenizer, self).__init__()

        self.object_type_embedding = embedding(
            num_embeddings=len(ObjectType),
            embedding_dim=len(ObjectType) // 2,
            init_scale=init_scale,
        )
        self.mlp = nn.Sequential(
            linear(6 + len(ObjectType) // 2, hidden_size),
            layer_norm(hidden_size),
            nn.SiLU(inplace=True),
            linear(hidden_size, hidden_size),
            layer_norm(hidden_size),
            nn.SiLU(inplace=True),
        )

    def forward(
        self, trajectories: Trajectory, properties: ObjectProperty
    ) -> torch.Tensor:
        """Tokenizes the trajectory into embeddings.

        Args:
            trajectories (Trajectory): The input trajectory.
            properties (ObjectProperty): The object properties.

        Returns:
            torch.Tensor: The tokenized trajectory.
        """
        yaw: torch.Tensor = wrap_angles(trajectories.yaw)
        obj_type = torch.where(
            properties.valid,
            properties.object_types,
            torch.full_like(properties.object_types, ObjectType.UNDEFINED),
        )
        obj_type = self.object_type_embedding.forward(obj_type)
        obj_type = torch.broadcast_to(
            obj_type.unsqueeze(-2),
            size=trajectories.shape + (obj_type.shape[-1],),
        )
        x = torch.cat(
            [
                yaw.unsqueeze(-1),
                trajectories.velocity,
                trajectories.length.unsqueeze(-1),
                trajectories.width.unsqueeze(-1),
                trajectories.height.unsqueeze(-1),
                obj_type,
            ],
            dim=-1,
        )
        out = self.mlp.forward(x)

        return out
