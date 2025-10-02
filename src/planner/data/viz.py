# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Visualization utilities for the scenario."""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Polygon

from planner.data.dataclass import (
    MapPoint,
    MapPolylineType,
    ObjectProperty,
    Scenario,
    Trajectory,
)
from planner.utils.logging import get_logger

# Constants
MAP_POLYLINE_PLOT_KWARGS = {
    MapPolylineType.UNDEFINED.value: dict(
        alpha=0.0, color="#000000", zorder=1
    ),
    # Lane Centers
    MapPolylineType.LANE_CENTER_VEHICLE.value: dict(
        alpha=0.5, color="#666666", linewidth=0.75, zorder=1
    ),
    MapPolylineType.LANE_CENTER_BIKE.value: dict(
        alpha=0.5, color="#666666", linewidth=0.5, zorder=1
    ),
    MapPolylineType.LANE_CENTER_BUS.value: dict(
        alpha=0.5, color="#666666", linewidth=0.75, zorder=1
    ),
    # Lane Markings
    MapPolylineType.LANE_MARKING_DASH_SOLID_YELLOW.value: dict(
        alpha=1.0, color="#f7b500", linewidth=0.75, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_DASH_SOLID_WHITE.value: dict(
        alpha=1.0, color="#ffffff", linewidth=0.75, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_DASHED_WHITE.value: dict(
        alpha=1.0, color="#ffffff", linewidth=0.75, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_DASHED_YELLOW.value: dict(
        alpha=1.0, color="#f7b500", linewidth=0.75, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_DOUBLE_SOLID_YELLOW.value: dict(
        alpha=1.0, color="#f7b500", linewidth=1.5, zorder=1
    ),
    MapPolylineType.LANE_MARKING_DOUBLE_SOLID_WHITE.value: dict(
        alpha=1.0, color="#ffffff", linewidth=1.5, zorder=1
    ),
    MapPolylineType.LANE_MARKING_DOUBLE_DASH_YELLOW.value: dict(
        alpha=1.0, color="#f7b500", linewidth=1.5, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_DOUBLE_DASH_WHITE.value: dict(
        alpha=1.0, color="#ffffff", linewidth=1.5, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_SOLID_YELLOW.value: dict(
        alpha=1.0, color="#f7b500", linewidth=0.75, zorder=1
    ),
    MapPolylineType.LANE_MARKING_SOLID_WHITE.value: dict(
        alpha=1.0, color="#ffffff", linewidth=0.75, zorder=1
    ),
    MapPolylineType.LANE_MARKING_SOLID_DASH_WHITE.value: dict(
        alpha=1.0, color="#ffffff", linewidth=0.75, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_SOLID_DASH_YELLOW.value: dict(
        alpha=1.0, color="#f7b500", linewidth=0.75, linestyle="--", zorder=1
    ),
    MapPolylineType.LANE_MARKING_SOLID_BLUE.value: dict(
        alpha=1.0, color="#0000ff", linewidth=0.75, zorder=1
    ),
    # lane boundaries
    MapPolylineType.LANE_BOUNDARY.value: dict(
        alpha=1.0, color="#ffffff", linewidth=1.5, zorder=1
    ),
    # crosswalk
    MapPolylineType.CROSSWALK.value: dict(
        alpha=1.0, color="#387f39", linewidth=0.75, linestyle="--", zorder=1
    ),
}
LOGGER = get_logger(__name__)


def plot_scenario(
    scenario: Scenario, ax: Optional[plt.Axes], crop_to_bounds: bool = True
) -> plt.Axes:
    """Visualize the scenario.

    Args:
        scenario (Scenario): The scenario to visualize.
        ax (Optional[plt.Axes]): The axis to plot on. Defaults to `None`.
        crop_to_bounds (bool): Whether to crop the plot to the bounds of the
            scenario. Defaults to ``True``.

    Returns:
        plt.Axes: The axis that was plotted on.
    """
    assert scenario.batch_dims == (), "Can only visualize a single scenario."
    if ax is None:
        _, ax = plt.subplots(1, 1)
    print("Type of ax:", type(ax))
    ax = _plot_map(points=scenario.map_point, ax=ax)
    print("Type of ax after plot map:", type(ax))
    ax, bounds = _plot_agents(
        trajectories=scenario.log_trajectory,
        properties=scenario.object_property,
        current_time_step=scenario.current_time_step,
        ax=ax,
    )
    print("Type of ax after _plot_agents:", type(ax))

    if crop_to_bounds:
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal")
    ax.set_axis_off()

    return ax


def _plot_agents(
    trajectories: Trajectory,
    properties: ObjectProperty,
    current_time_step: int,
    ax: plt.Axes,
) -> Tuple[plt.Axes, Tuple[int, ...]]:
    # plot history trajectories
    for i, traj in enumerate(trajectories):
        mask = torch.logical_and(traj.observed, traj.valid)
        xy = traj.xy[traj.valid * traj.observed]
        is_sdc = properties.is_sdc[i].item()
        is_target = properties.is_target[i].item()
        if is_sdc:
            bounds = (
                xy[current_time_step, 0] - 20,
                xy[current_time_step, 0] + 40,
                xy[current_time_step, 1] - 20,
                xy[current_time_step, 1] + 20,
            )
            color = "#d63f2e"
        elif is_target:
            color = "#4a90e2"
        else:
            color = "#a7c6ed"
        ax.plot(
            xy[: current_time_step + 1, 0],
            xy[: current_time_step + 1, 1],
            color=color,
            linewidth=1.5,
            zorder=2,
        )

        # plot bounding boxes
        if mask[current_time_step].item():
            # NOTE: only plot the bounding box at the current time step
            yaw = traj.yaw[current_time_step]
            dim = torch.stack([traj.length, traj.width, traj.height], dim=-1)
            dim = dim[current_time_step, :]
            vertices_local = torch.tensor(
                [
                    [-dim[0] / 2.0, -dim[1] / 2.0],  # rear-left
                    [dim[0] / 2.0, -dim[1] / 2.0],  # rear-right
                    [dim[0] / 2.0, dim[1] / 2.0],  # front-right
                    [-dim[0] / 2.0, dim[1] / 2.0],  # front-left
                ]
            )
            rotation_matrix = torch.tensor(
                [
                    [torch.cos(yaw), -torch.sin(yaw)],
                    [torch.sin(yaw), torch.cos(yaw)],
                ]
            )
            vertices_rotated = torch.mm(vertices_local, rotation_matrix.T)
            vertices_global = torch.add(
                traj.xy[current_time_step, 0:2], vertices_rotated
            )

            bbox = Polygon(
                xy=vertices_global.cpu().numpy(),
                edgecolor=color,
                facecolor="#eeeeee",
                linewidth=1.5,
                zorder=2,
            )
            ax.add_patch(bbox)

    return ax, bounds


def _plot_map(points: MapPoint, ax: plt.Axes) -> plt.Axes:
    for idx in points.ids[points.valid].unique():
        mask = points.valid * (points.ids == idx)
        xy = points.xy[mask]
        type_int = points.types[mask][0].item()
        if type_int in MAP_POLYLINE_PLOT_KWARGS:
            if type_int == MapPolylineType.CROSSWALK.value:
                for _xy in xy.chunk(2, dim=0):
                    ax.plot(
                        _xy[:, 0],
                        _xy[:, 1],
                        **MAP_POLYLINE_PLOT_KWARGS[type_int],
                    )
            else:
                ax.plot(
                    xy[:, 0],
                    xy[:, 1],
                    **MAP_POLYLINE_PLOT_KWARGS[type_int],
                )

    return ax
