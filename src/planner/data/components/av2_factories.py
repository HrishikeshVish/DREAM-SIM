# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Factory functions for creating scenario states from the Argoverse API."""
from typing import Optional, Tuple

import torch
from av2.datasets.motion_forecasting import data_schema
from av2.map.lane_segment import LaneMarkType, LaneType
from av2.map.map_api import ArgoverseStaticMap

from planner.data.dataclass import (
    INVALID_FLOAT_VALUE,
    INVALID_INT_VALUE,
    MapPoint,
    MapPolylineType,
    ObjectProperty,
    ObjectType,
    Scenario,
    Trajectory,
)
from planner.utils.logging import get_logger

# Constants
CURRENT_TIME_INDEX = 25
LANE_CENTER_TYPE_MAPPING = {
    LaneType.VEHICLE: MapPolylineType.LANE_CENTER_VEHICLE,
    LaneType.BIKE: MapPolylineType.LANE_CENTER_BIKE,
    LaneType.BUS: MapPolylineType.LANE_CENTER_BUS,
}
LANE_MARK_TYPE_MAPPING = {
    LaneMarkType.DASH_SOLID_YELLOW: MapPolylineType.LANE_MARKING_DASH_SOLID_YELLOW,
    LaneMarkType.DASH_SOLID_WHITE: MapPolylineType.LANE_MARKING_DASH_SOLID_WHITE,
    LaneMarkType.DASHED_WHITE: MapPolylineType.LANE_MARKING_DASHED_WHITE,
    LaneMarkType.DASHED_YELLOW: MapPolylineType.LANE_MARKING_DASHED_YELLOW,
    LaneMarkType.DOUBLE_SOLID_YELLOW: MapPolylineType.LANE_MARKING_DOUBLE_SOLID_YELLOW,
    LaneMarkType.DOUBLE_SOLID_WHITE: MapPolylineType.LANE_MARKING_DOUBLE_SOLID_WHITE,
    LaneMarkType.DOUBLE_DASH_YELLOW: MapPolylineType.LANE_MARKING_DOUBLE_DASH_YELLOW,
    LaneMarkType.DOUBLE_DASH_WHITE: MapPolylineType.LANE_MARKING_DOUBLE_DASH_WHITE,
    LaneMarkType.SOLID_YELLOW: MapPolylineType.LANE_MARKING_SOLID_YELLOW,
    LaneMarkType.SOLID_WHITE: MapPolylineType.LANE_MARKING_SOLID_WHITE,
    LaneMarkType.SOLID_DASH_WHITE: MapPolylineType.LANE_MARKING_SOLID_DASH_WHITE,
    LaneMarkType.SOLID_DASH_YELLOW: MapPolylineType.LANE_MARKING_SOLID_DASH_YELLOW,
    LaneMarkType.SOLID_BLUE: MapPolylineType.LANE_MARKING_SOLID_BLUE,
    LaneMarkType.NONE: MapPolylineType.UNDEFINED,
    LaneMarkType.UNKNOWN: MapPolylineType.UNDEFINED,
}
OBJECT_TYPE_MAPPING = {
    data_schema.ObjectType.UNKNOWN: ObjectType.UNDEFINED,
    data_schema.ObjectType.VEHICLE: ObjectType.VEHICLE,
    data_schema.ObjectType.MOTORCYCLIST: ObjectType.MOTORCYCLIST,
    data_schema.ObjectType.CYCLIST: ObjectType.CYCLIST,
    data_schema.ObjectType.PEDESTRIAN: ObjectType.PEDESTRIAN,
    data_schema.ObjectType.BUS: ObjectType.BUS,
}
LOGGER = get_logger(__name__)


def create_sceanrio_state_from_api(
    scenario_id: str,
    map_api: ArgoverseStaticMap,
    scenario_api: data_schema.ArgoverseScenario,
    radius: Optional[float] = None,
) -> Scenario:
    object_property, log_trajectory = _parse_objects(
        scenario_api, radius=radius
    )
    focal_xy = log_trajectory.xy[object_property.is_sdc]
    if( type(CURRENT_TIME_INDEX) is list):
        focal_xy = focal_xy[:, CURRENT_TIME_INDEX, :]
        map_point = _parse_map_point(map_api, focal_xy=focal_xy, radius=radius)
        scenario = Scenario(
            scenario_id=str(scenario_id).encode("utf-8"),
            log_trajectory=log_trajectory,
            object_property=object_property,
            map_point=map_point,
            current_time_step=CURRENT_TIME_INDEX,
        )
    else:
        focal_xy = focal_xy[:, : CURRENT_TIME_INDEX + 1]
        map_point = _parse_map_point(map_api, focal_xy=focal_xy, radius=radius)
        scenario = Scenario(
            scenario_id=str(scenario_id).encode("utf-8"),
            log_trajectory=log_trajectory,
            object_property=object_property,
            map_point=map_point,
            current_time_step=CURRENT_TIME_INDEX,
        )
    

    scenario.validate()

    return scenario


@torch.jit.script
def _get_direction(xyz: torch.Tensor) -> torch.Tensor:
    dir_xyz = torch.zeros_like(xyz)
    dir_xyz[..., 0:-1, :] = torch.diff(xyz, n=1, dim=-2)
    return dir_xyz


def _parse_map_point(
    map_api: ArgoverseStaticMap,
    num_map_point: int = 10000,
    focal_xy: Optional[torch.Tensor] = None,
    radius: Optional[float] = None,
) -> MapPoint:
    # initialize the pointers
    _pnt_ptr: int = 0
    _obj_ptr: int = 0

    # initialize the containers
    xyz = torch.full(
        [num_map_point, 3], INVALID_FLOAT_VALUE, dtype=torch.float32
    )
    dir_xyz = torch.full(
        [num_map_point, 3], INVALID_FLOAT_VALUE, dtype=torch.float32
    )
    types = torch.full([num_map_point], INVALID_INT_VALUE, dtype=torch.int64)
    ids = torch.full([num_map_point], INVALID_INT_VALUE, dtype=torch.int64)
    valid = torch.zeros([num_map_point], dtype=torch.bool)

    LOGGER.debug(
        "Number of lane segments: %d",
        len(map_api.vector_lane_segments),
    )
    for lane_segment_id, lane_segment in map_api.vector_lane_segments.items():
        lane_center = map_api.get_lane_segment_centerline(
            lane_segment_id=lane_segment_id
        )
        xyz[_pnt_ptr : _pnt_ptr + lane_center.shape[0], :] = torch.from_numpy(
            lane_center
        )
        dir_xyz[
            _pnt_ptr : _pnt_ptr + lane_center.shape[0], :
        ] = _get_direction(torch.from_numpy(lane_center))
        types[_pnt_ptr : _pnt_ptr + lane_center.shape[0]] = torch.tensor(
            [LANE_CENTER_TYPE_MAPPING[lane_segment.lane_type].value]
            * lane_center.shape[0]
        )
        ids[_pnt_ptr : _pnt_ptr + lane_center.shape[0]] = torch.tensor(
            [_obj_ptr] * lane_center.shape[0]
        )
        valid[_pnt_ptr : _pnt_ptr + lane_center.shape[0]] = True
        _pnt_ptr += lane_center.shape[0]
        _obj_ptr += 1

        if lane_segment.left_mark_type in (
            LaneMarkType.NONE,
            LaneMarkType.UNKNOWN,
        ):
            # if left lane marking is not specified, store the lane boundary
            left_boundary = lane_segment.left_lane_boundary.xyz
            xyz[
                _pnt_ptr : _pnt_ptr + left_boundary.shape[0], :
            ] = torch.from_numpy(left_boundary)
            dir_xyz[
                _pnt_ptr : _pnt_ptr + left_boundary.shape[0], :
            ] = _get_direction(torch.from_numpy(left_boundary))
            types[_pnt_ptr : _pnt_ptr + left_boundary.shape[0]] = torch.tensor(
                [MapPolylineType.LANE_BOUNDARY.value] * left_boundary.shape[0]
            )
            ids[_pnt_ptr : _pnt_ptr + left_boundary.shape[0]] = torch.tensor(
                [_obj_ptr] * left_boundary.shape[0]
            )
            valid[_pnt_ptr : _pnt_ptr + left_boundary.shape[0]] = True
            _pnt_ptr += left_boundary.shape[0]
        else:
            # otherwise, store the lane marking
            left_marking = lane_segment.left_lane_marking.polyline
            xyz[
                _pnt_ptr : _pnt_ptr + left_marking.shape[0], :
            ] = torch.from_numpy(left_marking)
            dir_xyz[
                _pnt_ptr : _pnt_ptr + left_marking.shape[0], :
            ] = _get_direction(torch.from_numpy(left_marking))
            types[_pnt_ptr : _pnt_ptr + left_marking.shape[0]] = torch.tensor(
                [LANE_MARK_TYPE_MAPPING[lane_segment.left_mark_type].value]
                * left_marking.shape[0]
            )
            ids[_pnt_ptr : _pnt_ptr + left_marking.shape[0]] = torch.tensor(
                [_obj_ptr] * left_marking.shape[0]
            )
            valid[_pnt_ptr : _pnt_ptr + left_marking.shape[0]] = True
            _pnt_ptr += left_marking.shape[0]
        _obj_ptr += 1

        if lane_segment.right_mark_type in (
            LaneMarkType.NONE,
            LaneMarkType.UNKNOWN,
        ):
            # if right lane marking is not specified, store the lane boundary
            right_boundary = lane_segment.right_lane_boundary.xyz
            xyz[
                _pnt_ptr : _pnt_ptr + right_boundary.shape[0], :
            ] = torch.from_numpy(right_boundary)
            dir_xyz[
                _pnt_ptr : _pnt_ptr + right_boundary.shape[0], :
            ] = _get_direction(torch.from_numpy(right_boundary))
            types[
                _pnt_ptr : _pnt_ptr + right_boundary.shape[0]
            ] = torch.tensor(
                [MapPolylineType.LANE_BOUNDARY.value] * right_boundary.shape[0]
            )
            ids[_pnt_ptr : _pnt_ptr + right_boundary.shape[0]] = torch.tensor(
                [_obj_ptr] * right_boundary.shape[0]
            )
            valid[_pnt_ptr : _pnt_ptr + right_boundary.shape[0]] = True
            _pnt_ptr += right_boundary.shape[0]
        else:
            # otherwise, store the lane marking
            right_marking = lane_segment.right_lane_marking.polyline
            xyz[
                _pnt_ptr : _pnt_ptr + right_marking.shape[0], :
            ] = torch.from_numpy(right_marking)
            dir_xyz[
                _pnt_ptr : _pnt_ptr + right_marking.shape[0], :
            ] = _get_direction(torch.from_numpy(right_marking))
            types[_pnt_ptr : _pnt_ptr + right_marking.shape[0]] = torch.tensor(
                [LANE_MARK_TYPE_MAPPING[lane_segment.right_mark_type].value]
                * right_marking.shape[0]
            )
            ids[_pnt_ptr : _pnt_ptr + right_marking.shape[0]] = torch.tensor(
                [_obj_ptr] * right_marking.shape[0]
            )
            valid[_pnt_ptr : _pnt_ptr + right_marking.shape[0]] = True
            _pnt_ptr += right_marking.shape[0]
        _obj_ptr += 1

    LOGGER.debug(
        "Number of pedestrian crossings: %d",
        len(map_api.vector_pedestrian_crossings),
    )
    for _, xing in map_api.vector_pedestrian_crossings.items():
        for edge in (xing.edge1.xyz, xing.edge2.xyz):
            xyz[_pnt_ptr : _pnt_ptr + edge.shape[0], :] = torch.from_numpy(
                edge
            )
            dir_xyz[_pnt_ptr : _pnt_ptr + edge.shape[0], :] = _get_direction(
                torch.from_numpy(edge)
            )
            types[_pnt_ptr : _pnt_ptr + edge.shape[0]] = torch.tensor(
                [MapPolylineType.CROSSWALK.value] * edge.shape[0]
            )
            ids[_pnt_ptr : _pnt_ptr + edge.shape[0]] = torch.tensor(
                [_obj_ptr] * edge.shape[0]
            )
            valid[_pnt_ptr : _pnt_ptr + edge.shape[0]] = True
            _pnt_ptr += edge.shape[0]
        _obj_ptr += 1

    # filter points that are beyond the observation radius
    if isinstance(radius, (int, float)):
        radius = float(radius)
        cdist = torch.cdist(
            x1=xyz[:, :2],
            x2=focal_xy.reshape(-1, 2),
            p=2,
        )
        cdist = torch.where(
            valid.reshape(-1)[:, None],
            cdist,
            torch.full_like(
                cdist,
                float("inf"),
                dtype=torch.float32,
            ),
        )

        # NOTE: keep the map points within the radius of the focal object at
        # any of the history time step.
        valid = torch.logical_and(
            valid,
            (cdist.min(dim=1)[0] <= radius).reshape(-1),
        )
        xyz = torch.where(
            valid[:, None],
            xyz,
            torch.full_like(
                xyz,
                INVALID_FLOAT_VALUE,
                dtype=torch.float32,
            ),
        )
        dir_xyz = torch.where(
            valid[:, None],
            dir_xyz,
            torch.full_like(
                dir_xyz,
                INVALID_FLOAT_VALUE,
                dtype=torch.float32,
            ),
        )
        types = torch.where(
            valid,
            types,
            torch.full_like(
                types,
                INVALID_INT_VALUE,
                dtype=torch.int64,
            ),
        )

    # create container
    output = MapPoint(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        dir_x=dir_xyz[:, 0],
        dir_y=dir_xyz[:, 1],
        dir_z=dir_xyz[:, 2],
        types=types,
        ids=ids,
        valid=valid,
    )
    output.validate()

    return output


def _parse_objects(
    scenario_api: data_schema.ArgoverseScenario,
    num_objects: int = 288,
    radius: Optional[float] = None,
) -> Tuple[ObjectProperty, Trajectory, Tuple[float, float]]:
    # initialize containers for object metadata
    ids = torch.full([num_objects], INVALID_INT_VALUE, dtype=torch.int64)
    object_types = torch.full(
        [num_objects], INVALID_INT_VALUE, dtype=torch.int64
    )
    is_sdc = torch.zeros([num_objects], dtype=torch.bool)
    is_target = torch.zeros([num_objects], dtype=torch.bool)

    # initialize containers for object trajectories
    xyz = torch.full(
        [num_objects, 110, 3], INVALID_FLOAT_VALUE, dtype=torch.float32
    )
    yaw = torch.full(
        [num_objects, 110], INVALID_FLOAT_VALUE, dtype=torch.float32
    )
    velocity = torch.full(
        [num_objects, 110, 2], INVALID_FLOAT_VALUE, dtype=torch.float32
    )
    timestamp_ms = torch.full(
        [num_objects, 110], INVALID_INT_VALUE, dtype=torch.int64
    )
    dimensions = torch.full(
        [num_objects, 110, 3], INVALID_FLOAT_VALUE, dtype=torch.float32
    )
    observed = torch.zeros([num_objects, 110], dtype=torch.bool)
    valid = torch.zeros([num_objects, 110], dtype=torch.bool)

    i: int = 0
    LOGGER.debug(
        "Number of tracks: %d",
        len(scenario_api.tracks),
    )
    for track in scenario_api.tracks:
        if track.object_type not in OBJECT_TYPE_MAPPING:
            # skip static or unknown objects
            LOGGER.debug(
                "Skipping track with type: %s",
                str(track.object_type.name),
            )
            continue

        ids[i] = i + 1
        object_types[i] = OBJECT_TYPE_MAPPING[track.object_type].value
        is_sdc[i] = track.category == data_schema.TrackCategory.FOCAL_TRACK
        is_target[i] = track.category in (
            data_schema.TrackCategory.FOCAL_TRACK,
            data_schema.TrackCategory.SCORED_TRACK,
        )

        for j, state in enumerate(track.object_states):
            xyz[i, j, :] = torch.tensor(
                [state.position[0], state.position[1], 0.0]
            )
            yaw[i, j] = state.heading
            velocity[i, j, :] = torch.tensor(state.velocity)
            timestamp_ms[i, j] = state.timestep * 100
            if track.object_type == data_schema.ObjectType.VEHICLE:
                dimensions[i, j, :] = torch.tensor([4.0, 2.0, 1.5])
            elif track.object_type in (
                data_schema.ObjectType.CYCLIST,
                data_schema.ObjectType.MOTORCYCLIST,
            ):
                dimensions[i, j, :] = torch.tensor([2.0, 0.7, 1.0])
            elif track.object_type == data_schema.ObjectType.PEDESTRIAN:
                dimensions[i, j, :] = torch.tensor([0.5, 0.5, 1.8])
            elif track.object_type == data_schema.ObjectType.BUS:
                dimensions[i, j, :] = torch.tensor([12.0, 2.5, 3.5])
            else:
                dimensions[i, j, :] = torch.tensor([0.0, 0.0, 0.0])
            observed[i, j] = state.observed
            valid[i, j] = True
        i += 1

    # filter points that are beyond the observation radius
    if(type(CURRENT_TIME_INDEX) is list):
        last_idx = CURRENT_TIME_INDEX[0] + 1
    else:
        last_idx = CURRENT_TIME_INDEX + 1
    if isinstance(radius, (int, float)):
        radius = float(radius)
        focal_xy = xyz[is_sdc][:, :last_idx, :2]
        cdist = torch.cdist(
            x1=xyz[:, :last_idx, 0:2].reshape(-1, 2),
            x2=focal_xy.reshape(-1, 2),
            p=2,
        )  # shape: (num_objects * last_idx, last_idx)
        cdist = torch.where(
            valid[:, :last_idx].reshape(-1)[:, None],
            cdist,
            torch.full_like(
                cdist,
                float("inf"),
                dtype=torch.float32,
            ),
        )

        # NOTE: keep the history observations within the radius of the focal
        # object at the any history time step.
        observed[:, :last_idx] = torch.logical_and(
            observed[:, :last_idx],
            (cdist.min(dim=1)[0] <= radius).reshape(-1, last_idx),
        )
        valid[:, :last_idx] = torch.logical_and(
            valid[:, :last_idx],
            (cdist.min(dim=1)[0] <= radius).reshape(-1, last_idx),
        )
        xyz[:, :last_idx] = torch.where(
            torch.logical_and(
                observed[:, :last_idx, None],
                valid[:, :last_idx, None],
            ),
            xyz[:, :last_idx],
            torch.full_like(
                xyz[:, :last_idx],
                INVALID_FLOAT_VALUE,
                dtype=torch.float32,
            ),
        )
        yaw[:, :last_idx] = torch.where(
            observed[:, :last_idx] * valid[:, :last_idx],
            yaw[:, :last_idx],
            torch.full_like(
                yaw[:, :last_idx],
                INVALID_FLOAT_VALUE,
                dtype=torch.float32,
            ),
        )
        velocity[:, :last_idx] = torch.where(
            torch.logical_and(
                observed[:, :last_idx, None],
                valid[:, :last_idx, None],
            ),
            velocity[:, :last_idx],
            torch.full_like(
                velocity[:, :last_idx],
                INVALID_FLOAT_VALUE,
                dtype=torch.float32,
            ),
        )
        timestamp_ms[:, :last_idx] = torch.where(
            observed[:, :last_idx] * valid[:, :last_idx],
            timestamp_ms[:, :last_idx],
            torch.full_like(
                timestamp_ms[:, :last_idx],
                INVALID_INT_VALUE,
                dtype=torch.int64,
            ),
        )
        dimensions[:, :last_idx] = torch.where(
            torch.logical_and(
                observed[:, :last_idx, None],
                valid[:, :last_idx, None],
            ),
            dimensions[:, :last_idx],
            torch.full_like(
                dimensions[:, :last_idx],
                INVALID_FLOAT_VALUE,
                dtype=torch.float32,
            ),
        )

    # create containers
    object_property = ObjectProperty(
        ids=ids,
        object_types=object_types,
        valid=torch.any(valid, dim=-1),
        is_sdc=torch.logical_and(is_sdc, torch.all(valid, dim=-1)),
        is_target=torch.logical_and(is_target, torch.all(valid, dim=-1)),
    )
    object_property.validate()

    trajectory = Trajectory(
        x=xyz[:, :, 0],
        y=xyz[:, :, 1],
        z=xyz[:, :, 2],
        yaw=yaw,
        velocity_x=velocity[:, :, 0],
        velocity_y=velocity[:, :, 1],
        timestamp_ms=timestamp_ms,
        length=dimensions[:, :, 0],
        width=dimensions[:, :, 1],
        height=dimensions[:, :, 2],
        observed=observed,
        valid=valid,
    )
    trajectory.validate()

    return object_property, trajectory
