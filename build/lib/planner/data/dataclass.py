# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Data classes for representing scenario states."""
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Generator, Optional, Sequence, Tuple, Union

import torch

from planner.type import size_t_any

# Constants
INVALID_INT_VALUE = -1
INVALID_FLOAT_VALUE = -1.0


# =============================================================================
@unique
class MapPolylineType(IntEnum):
    """Enumeration to represent different map polyline types in integers."""

    UNDEFINED = 0
    # lane centerlines
    LANE_CENTER_VEHICLE = 1
    LANE_CENTER_BIKE = 2
    LANE_CENTER_BUS = 3
    # lane markings
    LANE_MARKING_DASH_SOLID_YELLOW = 4
    LANE_MARKING_DASH_SOLID_WHITE = 5
    LANE_MARKING_DASHED_WHITE = 6
    LANE_MARKING_DASHED_YELLOW = 7
    LANE_MARKING_DOUBLE_SOLID_YELLOW = 8
    LANE_MARKING_DOUBLE_SOLID_WHITE = 9
    LANE_MARKING_DOUBLE_DASH_YELLOW = 10
    LANE_MARKING_DOUBLE_DASH_WHITE = 11
    LANE_MARKING_SOLID_YELLOW = 12
    LANE_MARKING_SOLID_WHITE = 13
    LANE_MARKING_SOLID_DASH_WHITE = 14
    LANE_MARKING_SOLID_DASH_YELLOW = 15
    LANE_MARKING_SOLID_BLUE = 16
    # lane boundaries
    LANE_BOUNDARY = 17
    # crosswalks
    CROSSWALK = 18


@dataclass
class MapPoint:
    """Data structure representing a single point on the map.

    Attributes:
        x (torch.Tensor): The x-coordinate of the point in meters.
        y (torch.Tensor): The y-coordinate of the point in meters.
        z (torch.Tensor): The z-coordinate of the point in meters.
        dir_x (torch.Tensor): The x-component of the direction vector.
        dir_y (torch.Tensor): The y-component of the direction vector.
        dir_z (torch.Tensor): The z-component of the direction vector.
        types (torch.Tensor): The integer type of the polyline.
        ids (torch.Tensor): The unique identifier of the polyline.
        valid (torch.Tensor): The validity of the observation.
    """

    x: torch.Tensor
    """torch.Tensor: The x-coordinate of the point in meters."""
    y: torch.Tensor
    """torch.Tensor: The y-coordinate of the point in meters."""
    z: torch.Tensor
    """torch.Tensor: The z-coordinate of the point in meters."""
    dir_x: torch.Tensor
    """torch.Tensor: The x-component of the direction vector."""
    dir_y: torch.Tensor
    """torch.Tensor: The y-component of the direction vector."""
    dir_z: torch.Tensor
    """torch.Tensor: The z-component of the direction vector."""
    types: torch.Tensor
    """torch.Tensor: The integer type of the polyline."""
    ids: torch.Tensor
    """torch.Tensor: The unique identifier of the polyline."""
    valid: torch.Tensor
    """torch.Tensor: The validity of the observation."""

    def __getitem__(self, index: Union[int, slice, Tuple]) -> "MapPoint":
        try:
            x = self.x[index]
            y = self.y[index]
            z = self.z[index]
            dir_x = self.dir_x[index]
            dir_y = self.dir_y[index]
            dir_z = self.dir_z[index]
            types = self.types[index]
            ids = self.ids[index]
            valid = self.valid[index]
        except Exception as ex:
            msg = "Failed to index the map points."
            raise RuntimeError(msg) from ex

        return MapPoint(
            x=x,
            y=y,
            z=z,
            dir_x=dir_x,
            dir_y=dir_y,
            dir_z=dir_z,
            types=types,
            ids=ids,
            valid=valid,
        )

    def __iter__(self) -> Generator["MapPoint", None, None]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def shape(self) -> size_t_any:
        """Tuple[int, ...]: The shape of the map points."""
        return self.valid.shape

    @property
    def xy(self) -> torch.Tensor:
        """torch.Tensor: The x-y coordinates."""
        return torch.stack([self.x, self.y], dim=-1)

    @property
    def xyz(self) -> torch.Tensor:
        """torch.Tensor: The x-y-z coordinates."""
        return torch.stack([self.x, self.y, self.z], dim=-1)

    @property
    def dir_xy(self) -> torch.Tensor:
        """torch.Tensor: The x-y components of the direction vector."""
        return torch.stack([self.dir_x, self.dir_y], dim=-1)

    @property
    def dir_xyz(self) -> torch.Tensor:
        """torch.Tensor: The x-y-z components of the direction vector."""
        return torch.stack([self.dir_x, self.dir_y, self.dir_z], dim=-1)

    @property
    def orientation(self) -> torch.Tensor:
        """torch.Tensor: The orientation of each map point."""
        return torch.where(
            condition=self.valid,
            input=torch.atan2(self.dir_y, self.dir_x),
            other=INVALID_FLOAT_VALUE,
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "MapPoint":
        """Perform tensor dtype and/or device conversion.

        Args:
            device (Optional[torch.device], optional): The device to move the
                tensors to. Defaults to `None`.
            dtype (torch.dtype): The dtype to convert the tensors to.
                Defaults to `None`.

        Returns:
            MapPoint: The converted map points.
        """
        self.x = self.x.to(device=device, dtype=dtype)
        self.y = self.y.to(device=device, dtype=dtype)
        self.z = self.z.to(device=device, dtype=dtype)
        self.dir_x = self.dir_x.to(device=device, dtype=dtype)
        self.dir_y = self.dir_y.to(device=device, dtype=dtype)
        self.dir_z = self.dir_z.to(device=device, dtype=dtype)
        self.types = self.types.to(device=device, dtype=torch.int64)
        self.ids = self.ids.to(device=device, dtype=torch.int64)
        self.valid = self.valid.to(device=device, dtype=torch.bool)

        return self

    def validate(self) -> None:
        assert self.x.shape == self.shape and self.x.dtype == torch.float32
        assert self.y.shape == self.shape and self.y.dtype == torch.float32
        assert self.z.shape == self.shape and self.z.dtype == torch.float32
        assert (
            self.dir_x.shape == self.shape
            and self.dir_x.dtype == torch.float32
        )
        assert (
            self.dir_y.shape == self.shape
            and self.dir_y.dtype == torch.float32
        )
        assert (
            self.dir_z.shape == self.shape
            and self.dir_z.dtype == torch.float32
        )
        assert (
            self.types.shape == self.shape and self.types.dtype == torch.int64
        )
        assert self.ids.shape == self.shape and self.ids.dtype == torch.int64


# =============================================================================
# Object states and properties
@unique
class ObjectType(IntEnum):
    """Enumeration to represent different dynamic object types in integers."""

    UNDEFINED = 0
    """int: Undefined object type."""
    VEHICLE = 1
    """int: Vehicle object type."""
    PEDESTRIAN = 2
    """int: Pedestrian object type."""
    MOTORCYCLIST = 3
    """int: Motorcyclist object type."""
    CYCLIST = 4
    """int: Cyclist object type."""
    BUS = 5
    """int: Bus object type."""


@dataclass
class ObjectProperty:
    """Time-invariant properties for objects in a scenario.

    Attributes:
        ids (torch.Tensor): The object IDs.
        object_types (torch.Tensor): The object types as integers.
        valid (torch.Tensor): The validity of the observation.
        is_sdc (torch.Tensor): The self-driving car indicator.
        is_target (torch.Tensor): The object to predict indicator.
    """

    ids: torch.Tensor
    """torch.Tensor: The object IDs."""
    object_types: torch.Tensor
    """torch.Tensor: The object types as integers."""
    valid: torch.Tensor
    """torch.Tensor: The validity of the observation."""
    is_sdc: torch.Tensor
    """torch.Tensor: The self-driving car indicator."""
    is_target: torch.Tensor
    """torch.Tensor: The object to predict indicator."""

    def __getitem__(self, index: Union[int, slice, Tuple]) -> "ObjectProperty":
        try:
            ids = self.ids[index]
            object_types = self.object_types[index]
            valid = self.valid[index]
            is_sdc = self.is_sdc[index]
            is_target = self.is_target[index]
        except Exception as ex:
            msg = "Failed to index the object properties."
            raise RuntimeError(msg) from ex

        return ObjectProperty(
            ids=ids,
            object_types=object_types,
            valid=valid,
            is_sdc=is_sdc,
            is_target=is_target,
        )

    def __iter__(self) -> Generator["ObjectProperty", None, None]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def shape(self) -> size_t_any:
        """Tuple[int, ...]: The shape of the object properties."""
        return self.valid.shape

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "ObjectProperty":
        """Perform tensor dtype and/or device conversion.

        Args:
            device (Optional[torch.device], optional): The device to move the
                tensors to. Defaults to `None`.
            dtype (torch.dtype): The dtype to convert the tensors to.
                Defaults to `None`.

        Returns:
            ObjectProperty: The converted object properties.
        """
        self.ids = self.ids.to(device=device, dtype=dtype)
        self.object_types = self.object_types.to(device=device, dtype=dtype)
        self.valid = self.valid.to(device=device, dtype=torch.bool)
        self.is_sdc = self.is_sdc.to(device=device, dtype=torch.bool)
        self.is_target = self.is_target.to(device=device, dtype=torch.bool)

        return self

    def validate(self) -> None:
        assert self.ids.shape == self.shape
        assert self.object_types.shape == self.shape
        assert self.valid.shape == self.shape
        assert self.is_sdc.shape == self.shape
        assert self.is_target.shape == self.shape
        assert self.ids.dtype == torch.int64
        assert self.object_types.dtype == torch.int64
        assert self.valid.dtype == torch.bool
        assert self.is_sdc.dtype == torch.bool
        assert self.is_target.dtype == torch.bool


@dataclass
class Trajectory:
    """Time-series trajectory data for objects in a scenario.

    Attributes:
        x (torch.Tensor): The x-coordinates of shape `[B, N, T]`.
        y (torch.Tensor): The y-coordinate of shape `[B, N, T]`.
        z (torch.Tensor): The z-coordinate of shape `[B, N, T]`.
        yaw (torch.Tensor): The yaw angle of shape `[B, N, T]`.
        velocity_x (torch.Tensor): Velocity along x-axis of shape `[B, N, T]`.
        velocity_y (torch.Tensor): Velocity along y-axis of shape `[B, N, T]`.
        timestamp_ms (torch.Tensor): The timestamp in milliseconds.
        length (torch.Tensor): Object length of shape `[B, N, T]`.
        width (torch.Tensor): Object width of shape `[B, N, T]`.
        height (torch.Tensor): Object height of shape `[B, N, T]`.
        observed (torch.Tensor): If current time step is observed.
        valid (torch.Tensor): The validity of the observation.
    """

    x: torch.Tensor
    """torch.Tensor: The x-coordinates of shape `[B, N, T]`."""
    y: torch.Tensor
    """torch.Tensor: The y-coordinate of shape `[B, N, T]`."""
    z: torch.Tensor
    """torch.Tensor: The z-coordinate of shape `[B, N, T]`."""
    yaw: torch.Tensor
    """torch.Tensor: The yaw angle of shape `[B, N, T]`."""
    velocity_x: torch.Tensor
    """torch.Tensor: Velocity along x-axis of shape `[B, N, T]`."""
    velocity_y: torch.Tensor
    """torch.Tensor: Velocity along y-axis of shape `[B, N, T]`."""
    timestamp_ms: torch.Tensor
    """torch.Tensor: The timestamp in milliseconds."""
    length: torch.Tensor
    """torch.Tensor: Object length of shape `[B, N, T]`."""
    width: torch.Tensor
    """torch.Tensor: Object width of shape `[B, N, T]`."""
    height: torch.Tensor
    """torch.Tensor: Object height of shape `[B, N, T]`."""
    observed: torch.Tensor
    """torch.Tensor: If current time step is observed."""
    valid: torch.Tensor
    """torch.Tensor: The validity of the observation."""

    def __getitem__(self, index: Union[int, slice, Tuple]) -> "Trajectory":
        try:
            x = self.x[index]
            y = self.y[index]
            z = self.z[index]
            yaw = self.yaw[index]
            velocity_x = self.velocity_x[index]
            velocity_y = self.velocity_y[index]
            timestamp_ms = self.timestamp_ms[index]
            length = self.length[index]
            width = self.width[index]
            height = self.height[index]
            observed = self.observed[index]
            valid = self.valid[index]
        except Exception as ex:
            msg = "Failed to index the trajectory data."
            raise RuntimeError(msg) from ex

        return Trajectory(
            x=x,
            y=y,
            z=z,
            yaw=yaw,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            timestamp_ms=timestamp_ms,
            length=length,
            width=width,
            height=height,
            observed=observed,
            valid=valid,
        )

    def __iter__(self) -> Generator["Trajectory", None, None]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        return self.shape[0]

    @property
    def xy(self) -> torch.Tensor:
        """torch.Tensor: The x-y coordinates."""
        return torch.stack([self.x, self.y], dim=-1)

    @property
    def xyz(self) -> torch.Tensor:
        """torch.Tensor: The x-y-z coordinates."""
        return torch.stack([self.x, self.y, self.z], dim=-1)

    @property
    def velocity(self) -> torch.Tensor:
        """torch.Tensor: The velocity."""
        return torch.stack([self.velocity_x, self.velocity_y], dim=-1)

    @property
    def shape(self) -> size_t_any:
        """Tuple[int, ...]: The shape of the trajectory."""
        return self.valid.shape

    @property
    def speed(self) -> torch.Tensor:
        """torch.Tensor: The speed."""
        return torch.linalg.norm(self.velocity, ord=2, dim=-1)

    @property
    def timestamp_s(self) -> torch.Tensor:
        """torch.Tensor: The timestamp in seconds."""
        return self.timestamp_ms.float() / 1e3

    @property
    def timestamp_micros(self) -> torch.Tensor:
        """torch.Tensor: The timestamp in microseconds."""
        return self.timestamp_ms * 1e3

    @property
    def timestamp_nanosecs(self) -> torch.Tensor:
        """torch.Tensor: The timestamp in nanoseconds."""
        return self.timestamp_ms * 1e6

    @property
    def velocity_yaw(self) -> torch.Tensor:
        """torch.Tensor: The yaw angle of the velocity."""
        return torch.where(
            condition=self.valid,
            input=torch.atan2(self.velocity_y, self.velocity_x),
            other=INVALID_FLOAT_VALUE,
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Trajectory":
        """Perform tensor dtype and/or device conversion.

        Args:
            device (Optional[torch.device], optional): The device to move the
                tensors to. Defaults to `None`.
            dtype (torch.dtype): The dtype to convert the tensors to.
                Defaults to `None`.

        Returns:
            Trajectory: The converted trajectory.
        """
        self.x = self.x.to(device=device, dtype=dtype)
        self.y = self.y.to(device=device, dtype=dtype)
        self.z = self.z.to(device=device, dtype=dtype)
        self.yaw = self.yaw.to(device=device, dtype=dtype)
        self.velocity_x = self.velocity_x.to(device=device, dtype=dtype)
        self.velocity_y = self.velocity_y.to(device=device, dtype=dtype)
        self.timestamp_ms = self.timestamp_ms.to(
            device=device, dtype=torch.int64
        )
        self.length = self.length.to(device=device, dtype=dtype)
        self.width = self.width.to(device=device, dtype=dtype)
        self.height = self.height.to(device=device, dtype=dtype)
        self.observed = self.observed.to(device=device, dtype=torch.bool)
        self.valid = self.valid.to(device=device, dtype=torch.bool)

        return self

    def validate(self) -> None:
        """Sanity check for trajectory data."""
        assert self.x.shape == self.shape and self.x.dtype == torch.float32
        assert self.y.shape == self.shape and self.y.dtype == torch.float32
        assert self.z.shape == self.shape and self.z.dtype == torch.float32
        assert self.yaw.shape == self.shape and self.yaw.dtype == torch.float32
        assert (
            self.velocity_x.shape == self.shape
            and self.velocity_x.dtype == torch.float32
        )
        assert (
            self.velocity_y.shape == self.shape
            and self.velocity_y.dtype == torch.float32
        )
        assert (
            self.valid.shape == self.shape and self.valid.dtype == torch.bool
        )
        assert (
            self.timestamp_ms.shape == self.shape
            and self.timestamp_ms.dtype == torch.int64
        )
        assert (
            self.length.shape == self.shape
            and self.length.dtype == torch.float32
        )
        assert (
            self.width.shape == self.shape
            and self.width.dtype == torch.float32
        )
        assert (
            self.height.shape == self.shape
            and self.height.dtype == torch.float32
        )
        assert (
            self.observed.shape == self.shape
            and self.observed.dtype == torch.bool
        )
        assert self.valid.dtype == torch.bool


# =============================================================================
# Top-level data class for scenario objects
@dataclass
class Scenario:
    """A data structure holding states for representing a scenario.

    .. note::

        All coordinates in the data structure are in the global frame.

    Attributes:
        log_trajectories (Trajectory): The logged trajectory data.
        object_property (ObjectProperty): The object property.
        map_point (MapPoint): The map points.
    """

    scenario_id: Union[bytes, Sequence[bytes]]
    """Union[bytes, Sequence[bytes]]: The scenario ID."""
    log_trajectory: Trajectory
    """Trajectory: The logged trajectory data."""
    object_property: ObjectProperty
    """ObjectProperty: The object property."""
    map_point: MapPoint
    """MapPoint: The map points."""
    current_time_step: int
    """int: The current time step."""

    def __getitem__(self, index: Union[int, slice, Tuple]) -> "Scenario":
        try:
            log_trajectory = self.log_trajectory[index]
            object_property = self.object_property[index]
            map_point = self.map_point[index]
        except Exception as ex:
            msg = "Failed to index the scenario data."
            raise RuntimeError(msg) from ex

        return Scenario(
            scenario_id=self.scenario_id[index],
            log_trajectory=log_trajectory,
            object_property=object_property,
            map_point=map_point,
            current_time_step=self.current_time_step,
        )

    def __iter__(self) -> Generator["Scenario", None, None]:
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        if len(self.shape) > 0:
            return self.shape[0]
        return 1

    @property
    def batch_dims(self) -> size_t_any:
        """Tuple[int, ...]: The batch dimensions."""
        return self.shape

    @property
    def num_map_points(self) -> int:
        """int: The number of map points."""
        return self.map_point.shape[-1]

    @property
    def num_objects(self) -> int:
        """int: The number of objects in the scenario."""
        return self.object_property.shape[-1]

    @property
    def shape(self) -> size_t_any:
        """Tuple[int, ...]: The shape of the scenario state."""
        return self.object_property.shape[:-1]

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "Scenario":
        """Perform tensor dtype and/or device conversion.

        Args:
            device (Optional[torch.device], optional): The device to move the
                tensors to. Defaults to `None`.
            dtype (torch.dtype): The dtype to convert the tensors to.
                Defaults to `None`.

        Returns:
            Scenario: The converted scenario state.
        """
        self.log_trajectory = self.log_trajectory.to(
            device=device, dtype=dtype
        )
        self.object_property = self.object_property.to(
            device=device, dtype=dtype
        )
        self.map_point = self.map_point.to(device=device, dtype=dtype)

        return self

    def validate(self) -> None:
        prefix_len = len(self.shape)
        assert self.log_trajectory.shape[:prefix_len] == self.shape
        assert self.object_property.shape[:prefix_len] == self.shape
        assert self.map_point.shape[:prefix_len] == self.shape
        assert isinstance(self.current_time_step, int) or isinstance(self.current_time_step, list)
        if(isinstance(self.current_time_step, list)):
            assert len(self.current_time_step) == 2
            assert self.current_time_step[0] >= 0
            assert self.current_time_step[1] < self.log_trajectory.shape[-1]
        elif isinstance(self.current_time_step, int):
            assert self.current_time_step >= 0
            assert self.current_time_step < self.log_trajectory.shape[-1]
        assert isinstance(self.scenario_id, bytes) or all(
            isinstance(i, bytes) for i in self.scenario_id
        )

    def __str__(self) -> str:
        return (
            f"Scenario(batch_dims={self.shape}, "
            f"num_map_points={self.num_map_points}, "
            f"num_objects={self.num_objects})"
        )

    def __repr__(self) -> str:
        return f"<{str(self)} at {hex(id(self))}>"
