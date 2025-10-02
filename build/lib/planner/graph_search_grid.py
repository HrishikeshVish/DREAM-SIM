from av2.map.map_api import ArgoverseStaticMap
from typing import Final, Sequence
from av2.utils.typing import NDArrayFloat
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from shapely.geometry import Polygon, Point
import math
from planner.data.datamodule import AV2DataModule
from planner.data.viz import plot_scenario
from copy import copy
from planner.load_model import load_way_points
import os.path as osp


_DRIVABLE_AREA_COLOR: Final[str] = "#7A7A7A"
_LANE_SEGMENT_COLOR: Final[str] = "#000000"

# Tree class -- stores all nodes and used to construct the graph
class Tree():
    def __init__(self,name):
        self.name = name
        self.root = 0
        self.end = 0
        self.g = {}

    def add_node(self, node, start = False, end = False):
        self.g[node.name] = node
        if(start):
            self.root = node.name
        elif(end):
            self.end = node.name

    def set_as_root(self,node):
        self.root = True
        self.end = False

    def set_as_end(self,node):
        self.root = False
        self.end = True


class Node():
    def __init__(self, name):
        self.name = name  # Name of the node (e.g., "(x,y)")
        self.children = []  # List of connected nodes
        self.weight = []  # List of edge weights

    def __repr__(self):
        return self.name

    def add_children(self, node, w=None):
        if w is None:
            w = [1] * len(node)
        self.children.extend(node)
        self.weight.extend(w)

def _plot_polygons(
    polygons: Sequence[NDArrayFloat], *, alpha: float = 1.0, color: str = "r"
) -> None:
    """Plot a group of filled polygons with the specified config.

    Args:
        polygons: Collection of polygons specified by (N,2) arrays of vertices.
        alpha: Desired alpha for the polygon fill.
        color: Desired color for the polygon.
    """
    for polygon in polygons:
        plt.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=alpha)

def _plot_polylines(
    polylines: Sequence[NDArrayFloat],
    *,
    style: str = "-",
    line_width: float = 1.0,
    alpha: float = 1.0,
    color: str = "r",
) -> None:
    """Plot a group of polylines with the specified config.

    Args:
        polylines: Collection of (N, 2) polylines to plot.
        style: Style of the line to plot (e.g. `-` for solid, `--` for dashed)
        line_width: Desired width for the plotted lines.
        alpha: Desired alpha for the plotted lines.
        color: Desired color for the plotted lines.
    """
    for polyline in polylines:
        plt.plot(
            polyline[:, 0],
            polyline[:, 1],
            style,
            linewidth=line_width,
            color=color,
            alpha=alpha,
        )

def _plot_static_map_elements(
    static_map: ArgoverseStaticMap, show_ped_xings: bool = False
) -> None:
    """Plot all static map elements associated with an Argoverse scenario.

    Args:
        static_map: Static map containing elements to be plotted.
        show_ped_xings: Configures whether pedestrian crossings should be plotted.
    """
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        _plot_polygons([drivable_area.xyz], alpha=0.5, color=_DRIVABLE_AREA_COLOR)

    # Plot lane segments
    
    for lane_segment in static_map.vector_lane_segments.values():
        _plot_polylines(
            [
                lane_segment.left_lane_boundary.xyz,
                lane_segment.right_lane_boundary.xyz,
            ],
            line_width=0.5,
            color=_LANE_SEGMENT_COLOR,
        )
        

    # # Plot pedestrian crossings
    if show_ped_xings:
        for ped_xing in static_map.vector_pedestrian_crossings.values():
            _plot_polylines(
                [ped_xing.edge1.xyz, ped_xing.edge2.xyz],
                alpha=1.0,
                color=_LANE_SEGMENT_COLOR,
            )

def distance(x,y,x1,y1):
    return math.sqrt(((x-x1)**2)+((y-y1)**2))

def create_grid(bounds, map_graph,resolution=0.2):
    """
    Creates a uniform grid over the map area within the given bounds.

    Args:
        bounds (tuple): (min_x, max_x, min_y, max_y)
        resolution (float): cell size in meters (e.g., 0.1 for 0.1x0.1 meter grid)

    Returns:
        grid_points: (N, 2) array of grid point coordinates [(x1, y1), (x2, y2), ...]
        grid_shape: (rows, cols) shape of the grid
    """
    min_x, max_x, min_y, max_y = [b.item() if hasattr(b, "item") else b for b in bounds]

    x_coords = np.arange(min_x, max_x + resolution, resolution)
    y_coords = np.arange(min_y, max_y + resolution, resolution)

    X, Y = np.meshgrid(x_coords, y_coords)
    grid_points = np.column_stack((X.flatten(), Y.flatten()))

    grid_shape = (len(y_coords), len(x_coords))  # rows x cols
    return grid_points, grid_shape

def overlay_grid_on_map_with_log_probabilities(ax, grid_points, grid_shape, map_graph, occupancy_grid, resolution=0.2, cmap='hot_r', alpha=0.6):
    """
    Overlays grid outlines on the map with edge colors representing log-probabilities.

    Args:
        ax (matplotlib.axes.Axes): The Axes to overlay the grid on.
        grid_points (np.ndarray): List of grid points (x, y).
        grid_shape (tuple): Shape of the grid (rows, cols).
        map_graph: Graph containing node keys as "i,j" strings.
        occupancy_grid (np.ndarray): Log-probability values.
        resolution (float): Grid cell size.
        cmap (str): Matplotlib colormap.
        alpha (float): Transparency for edge coloring.
    """
    rows, cols = grid_shape

    # Replace -inf with a very low number for coloring purposes (not normalization)
    safe_grid = np.where(np.isneginf(occupancy_grid), -10.0, occupancy_grid)

    vmin = np.min(safe_grid[np.isfinite(safe_grid)])
    vmax = np.max(safe_grid[np.isfinite(safe_grid)])

    colormap = plt.cm.get_cmap(cmap)

    for node in map_graph.g:
        i, j = map(int, node.split(","))
        x, y = grid_points[i * cols + j]
        log_prob = safe_grid[i, j]

        # Get color from colormap directly using the actual value (no normalization)
        color_val = (log_prob - vmin) / (vmax - vmin)
        color = colormap(np.clip(color_val, 0, 1))

        # Draw only the edges with color (no fill)
        rect = plt.Rectangle((x, y), resolution, resolution,
                             linewidth=0.5, edgecolor=color, facecolor='none', alpha=alpha)
        ax.add_patch(rect)

    # Colorbar with real log-prob range
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Log-Probability', rotation=270, labelpad=15)
    cbar.ax.tick_params(labelsize=8)





def grid_sampling(polygon, map_graph, lane_boundaries, paths,bounds, step=0.5):
    """
    Samples points inside a drivable area in a grid pattern.
    Takes each drivable area in a polygon format and samples 
    points within them based on the step size

    """
    
    min_x, max_x, min_y, max_y = bounds

    min_x, max_x, min_y, max_y = min_x.item(), max_x.item(), min_y.item(), max_y.item()

    points = []
    x = min_x
    too_close = 0
    lane_points = sample_lane_points(lane_boundaries)
    while x <= max_x:
        y = min_y
        too_close = 0
        while y <= max_y:
            too_close = 0
            too_close_to_other_traj = 0
            p = Point(x, y)
            if polygon.contains(p):
                for i in lane_points:
                    if distance(i[0],i[1],x,y) <= 0.1:
                        too_close = 0
                        break
                for i in paths:
                    if distance(i[0],i[1],x,y) <= 0.3:
                        too_close_to_other_traj = 1
                        break


                if too_close == 0 and too_close_to_other_traj == 0:
                    node_name = f"{x},{y}"
                    node = Node(node_name)
                    map_graph.add_node(node)
                    points.append((x, y))
            y += step
        x += step
    return map_graph,points


def sample_lane_points(lane_segments, sample_interval=0.5):
    """
    Samples points along the lane segments
    
    :param lane_segments: List of LaneSegment objects.
    :param sample_interval: Distance interval for sampling.
    :return: List of sampled lane points as (x, y).

    This function is used in grid sampling to make sure that nodes are not sampled on the lanes
    """
    sampled_points = []
    
    for lane in lane_segments:
        # Use both left and right boundaries for sampling
        for boundary in [lane.left_lane_boundary, lane.right_lane_boundary]:
            waypoints = boundary.waypoints
            for i in range(len(waypoints) - 1):
                p1, p2 = waypoints[i], waypoints[i + 1]
                dist = np.hypot(p2.x - p1.x, p2.y - p1.y)
                num_samples = max(int(dist / sample_interval), 1)
                for j in range(num_samples + 1):
                    x = p1.x + j / num_samples * (p2.x - p1.x)
                    y = p1.y + j / num_samples * (p2.y - p1.y)
                    sampled_points.append((x, y))
                    
    return sampled_points

def update_occupancy_grid(grid_points, points, grid_shape, probs, prob_points, prob_grid_shape, occupancy_grid, resolution=0.2, radius=0.2, threshold = 1.0):
    """
    Updates the occupancy grid, marking cells as occupied (1) if any point is within a radius from the grid cell.

    Args:
        grid_points (np.ndarray): List of grid points (x, y).
        points (list of tuple): List of points [(x, y), (x1, y1), ...].
        resolution (float): Resolution of the grid.
        grid_shape (tuple): Shape of the grid (rows, cols).
        radius (float): Radius around each point to mark as occupied.

    Returns:
        occupancy_grid (np.ndarray): Updated occupancy grid with 1 for occupied cells.
    """
    # occupancy_grid = np.zeros(grid_shape)
    print("Occupancy Grid Stats:")
    print("Min:", np.min(probs))
    print("Max:", np.max(probs))
    print("Unique values (sample):", np.unique(probs)[:10])

    origin_x, origin_y = grid_points[0]
    radius_cells = int(np.ceil(radius / resolution))
    prob_rows, prob_cols = prob_grid_shape

    for row in range(prob_rows):
        for col in range(prob_cols):
            if probs[row,col] < threshold:
            # print("yes")
                prob_point_x, prob_point_y = prob_points[row*prob_cols + col]
                center_col = int((prob_point_x - origin_x) // resolution)
                center_row = int((prob_point_y - origin_y) // resolution)
                for drow in range(-radius_cells, radius_cells + 1):
                    for dcol in range(-radius_cells, radius_cells + 1):
                        nrow = center_row + drow
                        ncol = center_col + dcol

                        if 0 <= nrow < grid_shape[0] and 0 <= ncol < grid_shape[1]:
                            # Compute center of this grid cell
                            cell_x = origin_x + ncol * resolution
                            cell_y = origin_y + nrow * resolution

                            dist = np.sqrt((cell_x - prob_point_x)**2 + (cell_y - prob_point_y)**2)
                            if dist <= radius:
                                # occupancy_grid[row, col] = 1
                        # if 0 <= center_row < grid_shape[0] and 0 <= center_col < grid_shape[1]:
                                if occupancy_grid[nrow, ncol] == -np.inf:
                                    occupancy_grid[nrow, ncol] = probs[row,col]
                                else:
                                    max_val = np.max([occupancy_grid[nrow, ncol], probs[row,col]])
                                    new_log = max_val + np.log(np.exp(occupancy_grid[nrow, ncol] - max_val) + np.exp(probs[row,col] - max_val))

                                    occupancy_grid[nrow, ncol] = new_log

    return occupancy_grid



def connect_nodes(map_graph, lane_segments, origin, grid_points, grid_shape, step=0.2, resolution = 0.2):
    """Connect nodes while respecting lane constraints. 
    Connects nodes from same lanes, predecessor lanes, 
    successor lanes and neighbouring lanes in intersections"""
    rows, cols = grid_shape
    node_lane_mapping = assign_nodes_to_lanes(map_graph,lane_segments,origin,grid_points, grid_shape)
    nodes = list(map_graph.g.values())
    # print(node_lane_mapping)
    lane_connectivity = {} # stores data on how lanes are connected. Each lane id is mapped with its predecessior and successor lanes and neighbouring lanes in intersections
    for lane in lane_segments:
        valid_neighbors = set(lane.predecessors + lane.successors + [lane.id])
        # Include lane changes if allowed
        if lane.is_intersection:
            if lane.right_mark_type.value == "DASHED_WHITE" and lane.right_neighbor_id:
                valid_neighbors.add(lane.right_neighbor_id)
            if lane.left_mark_type.value == "DASHED_WHITE"  and lane.left_neighbor_id:
                valid_neighbors.add(lane.left_neighbor_id)
        lane_connectivity[lane.id] = valid_neighbors
    
    

    step_grid = int(np.ceil(step / resolution))

    neighbor_offsets = [
        (-step_grid, -step_grid), (0, -step_grid), (step_grid, -step_grid),
        (-step_grid, 0),               (step_grid, 0),
        (-step_grid, step_grid),  (0, step_grid),  (step_grid, step_grid)
    ]

    for node in nodes:
        node_name = node.name
        no_lanes = 0
        if node_name in node_lane_mapping.keys():
            node_lane = node_lane_mapping[node_name]
        else:
            no_lanes = 1
        for dx, dy in neighbor_offsets:
            neigh_lane_exist = 1
            neighbor = (int(node.name.split(",")[0]) + dx, int(node.name.split(",")[1]) + dy)
            neighbor_name = f"{neighbor[0]},{neighbor[1]}"

            if neighbor_name in map_graph.g.keys():
                if neighbor_name in node_lane_mapping.keys():
                    neighbor_lane = node_lane_mapping[neighbor_name]
                else:
                    neigh_lane_exist = 0
                
                # Only connect if both nodes belong to valid connected lanes
                if no_lanes and not(neigh_lane_exist):
                    dist = distance(int(node.name.split(",")[0]), int(node.name.split(",")[1]), neighbor[0], neighbor[1])
                    map_graph.g[node_name].add_children([map_graph.g[neighbor_name]], [dist])
                elif no_lanes==1 and neigh_lane_exist==1:
                    ll = []
                    rl = []
                    for lane in lane_segments:
                        if lane.id in neighbor_lane:
                            ll.append(lane.left_lane_boundary.xyz)
                            rl.append(lane.right_lane_boundary.xyz)
                    all_dist = []
                    # node_x = ((node.name.split(",")[0]) * step) + origin[0]
                    # node_y = ((node.name.split(",")[1]) * step) + origin[1]
                    node_x, node_y = grid_points[int(node.name.split(",")[0]) * cols + int(node.name.split(",")[1])]
                    for lane in ll:
                        all_dist.append(distance(node_x,node_y,lane[0][0],lane[0][1]))
                        all_dist.append(distance(node_x,node_y,lane[-1][0],lane[-1][1]))
                    for lane in rl:
                        all_dist.append(distance(node_x,node_y,lane[0][0],lane[0][1]))
                        all_dist.append(distance(node_x,node_y,lane[-1][0],lane[-1][1]))

                    min_dist = min(all_dist)
                    if min_dist < 1.0:
                        dist = distance(int(node.name.split(",")[0]), int(node.name.split(",")[1]), neighbor[0], neighbor[1])
                        map_graph.g[node_name].add_children([map_graph.g[neighbor_name]], [dist])
                elif no_lanes == 0 and neigh_lane_exist == 0:
                    ll = []
                    rl = []
                    for lane in lane_segments:
                        if lane.id in node_lane:
                            ll.append(lane.left_lane_boundary.xyz)
                            rl.append(lane.right_lane_boundary.xyz)
                    all_dist = []
                    # node_x = ((neighbor[0]) * step) + origin[0]
                    # node_y = ((neighbor[1]) * step) + origin[1]
                    node_x, node_y = grid_points[neighbor[0] * cols + neighbor[1]]
                    for lane in ll:
                        all_dist.append(distance(node_x,node_y,lane[0][0],lane[0][1]))
                        all_dist.append(distance(node_x,node_y,lane[-1][0],lane[-1][1]))
                    for lane in rl:
                        all_dist.append(distance(node_x,node_y,lane[0][0],lane[0][1]))
                        all_dist.append(distance(node_x,node_y,lane[-1][0],lane[-1][1]))

                    min_dist = min(all_dist)
                    if min_dist < 1.0:
                        dist = distance(float(node.name.split(",")[0]), float(node.name.split(",")[1]), neighbor[0], neighbor[1])
                        map_graph.g[node_name].add_children([map_graph.g[neighbor_name]], [dist])
                else: 
                    for node_l in node_lane:
                       found_c = 0
                       for neighbor_l in neighbor_lane:
                        if neighbor_l in lane_connectivity.get(node_l, set()):
                            dist = distance(float(node.name.split(",")[0]), float(node.name.split(",")[1]), neighbor[0], neighbor[1])
                            map_graph.g[node_name].add_children([map_graph.g[neighbor_name]], [dist])
                            found_c = 1
                            break
                       if found_c == 1:
                           break
    return map_graph


def assign_nodes_to_lanes(map_graph, lane_segments, origin, grid_points, grid_shape, resolution = 0.5):
    """Assigns each node to the nearest lane segment based on boundaries. Takes graph as input and outputs a dinctionary of {'node':[lane_id]} format """
    lane_polygons = {}

    rows,cols = grid_shape

    for lane in lane_segments:
        boundary = lane.left_lane_boundary.waypoints + lane.right_lane_boundary.waypoints[::-1]

        lane_polygons[lane.id] = Polygon([(p.x, p.y) for p in boundary])
    
    node_lane_mapping = {}
    for node,_ in map_graph.g.items():
        point_x, point_y = grid_points[int(node.split(",")[0])*cols + int(node.split(",")[1])] 
        point = Point(point_x, point_y)
        for lane_id, polygon in lane_polygons.items():
            if polygon.contains(point):
                if node in node_lane_mapping.keys(): 
                    node_lane_mapping[node].append(lane_id)
                else:
                    node_lane_mapping[node] = [lane_id]
                
    
    return node_lane_mapping

# Queue class for astar
class Queue():
    def __init__(self, init_queue = []):
        self.queue = copy(init_queue)
        self.start = 0
        self.end = len(self.queue)-1

    def __len__(self):
        numel = len(self.queue)
        return numel

    def __repr__(self):
        q = self.queue
        tmpstr = ""
        for i in range(len(self.queue)):
            flag = False
            if(i == self.start):
                tmpstr += "<"
                flag = True
            if(i == self.end):
                tmpstr += ">"
                flag = True

            if(flag):
                tmpstr += '| ' + str(q[i]) + '|\n'
            else:
                tmpstr += ' | ' + str(q[i]) + '|\n'

        return tmpstr

    def __call__(self):
        return self.queue

    def initialize_queue(self,init_queue = []):
        self.queue = copy(init_queue)

    def sort(self,key=str.lower):
        self.queue = sorted(self.queue,key=key)

    def push(self,data):
        self.queue.append(data)
        self.end += 1

    def pop(self):
        p = self.queue.pop(self.start)
        self.end = len(self.queue)-1
        return p
        
class AStar():
    # Init function defines the heuristic as distance from goal
    def __init__(self,in_tree,occupancy_grid):
        self.in_tree = in_tree
        self.q = Queue()
        self.dist = {name:np.Inf for name,node in in_tree.g.items()}
        self.h = {name:0 for name,node in in_tree.g.items()}
        finite_log_probs = occupancy_grid[np.isfinite(occupancy_grid)]
        min_logp = np.percentile(finite_log_probs, 1)  # 1st percentile to avoid outliers
        max_logp = np.percentile(finite_log_probs, 99)  # 99th percentile

        # Normalize for risk (0 = low risk, 1 = high risk)
        norm_risk_grid = (occupancy_grid - min_logp) / (max_logp - min_logp)
        norm_risk_grid = np.clip(norm_risk_grid, 0, 1)
        risk_weight = 300

        for name,node in in_tree.g.items():
            start = tuple(map(int, name.split(',')))
            end = tuple(map(int, self.in_tree.end.split(',')))
            self.h[name] = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
            self.h[name] += risk_weight * norm_risk_grid[start[0],start[1]]  

        self.via = {name:0 for name,node in in_tree.g.items()}
        for __,node in in_tree.g.items():
            self.q.push(node)

    # function for calculating cost as node weight + node heurestics
    def __get_f_score(self,node):
      return self.dist[node.name] + self.h[node.name]

    # calculates optimal path and updates self.via with the right connections
    def solve(self, sn, en):
        self.dist[sn.name] = 0
        while len(self.q) > 0:
          self.q.sort(key=self.__get_f_score)
          current_node = self.q.queue[self.q.start]
          for i in current_node.children:
            if self.dist[i.name] > self.dist[current_node.name] + current_node.weight[current_node.children.index(i)]:
              self.dist[i.name] = self.dist[current_node.name] + current_node.weight[current_node.children.index(i)]
              self.via[i.name] = current_node.name
          popped_node = self.q.pop()
          if popped_node.name == en.name:
            break

    # returns a list of way points points in string format
    def reconstruct_path(self,sn,en):
        path = []
        dist = self.dist[en.name]
        path.append(en.name)
        current_node = en.name
        while True:
        #   print(path)
          path.append(self.via[current_node])
          current_node = self.via[current_node]
          if current_node == sn.name:
              path.append(sn.name)
              break
        path.reverse()
        # Place code here
        return path,dist


# Finds the nearest node to a given coordinate 
def find_nearest_node(x,y,map_graph, grid_points, grid_shape):
    rows, cols = grid_shape
    nodes = list(map_graph.g.values())
    min = np.inf
    min_node = nodes[0]
    for node in nodes:
        grid_conv_x, grid_conv_y = grid_points[int(node.name.split(",")[0]) * cols + int(node.name.split(",")[1])]
        if min > distance(grid_conv_x,grid_conv_y,x,y):
            min = distance(grid_conv_x,grid_conv_y,x,y)
            min_node = node
    
    return min_node.name



def plot_tree(tree, ax, origin, grid_points, grid_shape, resolution = 0.2):
    # Draw edges between nodes and their children
    rows, cols = grid_shape
    for node_name, node in tree.g.items():
        x1, y1 = map(int, node_name.split(','))  # Convert node name to (x, y)
        x1, y1 = grid_points[x1 * cols + y1] 
        
        for child in node.children:
            x2, y2 = map(int, child.name.split(','))
            x2, y2 = grid_points[x2 * cols + y2]
            ax.plot([x1, x2], [y1, y2], 'b-')  # Draw line from parent to child

def compute_minADE(path,target_path):
    errors = []
    path = np.array(path)
    target_path = np.array(target_path)
    for tgt_pt in target_path:
    # Compute distance to all A* path points
        dists = np.linalg.norm(path - tgt_pt, axis=1)
        min_dist = np.min(dists)
        errors.append(min_dist)

    ade = np.mean(errors)

    return ade

def compute_jerk(path, dt=0.1):
    path = np.array(path)  # shape (N, 2)

    # Compute velocity (first derivative)
    velocity = np.gradient(path, dt, axis=0)

    # Compute acceleration (second derivative)
    acceleration = np.gradient(velocity, dt, axis=0)

    # Compute jerk (third derivative)
    jerk = np.gradient(acceleration, dt, axis=0)

    # Compute jerk magnitude at each time step
    jerk_magnitude = np.linalg.norm(jerk, axis=1)

    # Optionally return average jerk or full vector
    average_jerk = np.mean(jerk_magnitude)
    return average_jerk, jerk_magnitude

def main():
    av2_data = AV2DataModule(
        batch_size=1,
        root=osp.abspath("/media/prashanth/Backup/av2"),
        num_workers=2,
        pin_memory=True,
        radius=50.0,
    )
    dl = iter(av2_data.val_dataloader())
    ds = av2_data.val_dataset
    data = next(dl)
    fig, ax = plt.subplots(1,1)
    id = data[0].scenario_id

    ax,bounds = plot_scenario(scenario=data[0], ax=ax)
    scenario_static_map = av2_data.val_dataset.get_map_api(id.decode('utf-8'))
    # # Plotting of lanes and drivable area. This function was given in the argoverse 2 github
    
    # # Initializing Tree
    map_graph = Tree('graph')
    grid_points, grid_shape = create_grid(bounds, map_graph,resolution=0.2)
    occupancy_grid = np.zeros(grid_shape, dtype=np.uint8)
    all_agents_uncertainty, paths, end_point , target_path = load_way_points(ax,data,ds)
    occupancy_grid = np.full(grid_shape, -np.inf, dtype=float)
    for i in all_agents_uncertainty:
        occupancy_grid = update_occupancy_grid(grid_points, paths, grid_shape,i[0], i[1], i[2], occupancy_grid)
    rows, cols = occupancy_grid.shape

    for row in range(rows):
        for col in range(cols):
                node_name = f"{row},{col}"
                node = Node(node_name)
                map_graph.add_node(node)
    origin_x = grid_points[0][0]
    origin_y = grid_points[0][1]
    

    # # # Graph construction
    map_graph = connect_nodes(map_graph,scenario_static_map.vector_lane_segments.values(),[origin_x,origin_y], grid_points, grid_shape)
    overlay_grid_on_map_with_log_probabilities(ax, grid_points, grid_shape , map_graph, occupancy_grid)
    # # plot_tree(map_graph,ax,[origin_x,origin_y], grid_points, grid_shape)

    start_x = bounds[0].item() + 50
    start_y = bounds[2].item() + 40
    end_x = end_point[0]
    end_y  = end_point[1]


    # # Defining Start and End Points
    map_graph.root = find_nearest_node(start_x,start_y,map_graph, grid_points, grid_shape)
    map_graph.end = find_nearest_node(end_x,end_y,map_graph, grid_points, grid_shape)
    
    # # Astar
    astar = AStar(map_graph, occupancy_grid)
    astar.solve(map_graph.g[map_graph.root],map_graph.g[map_graph.end])

    print(map_graph.root,map_graph.end)

    path,_ = astar.reconstruct_path(map_graph.g[map_graph.root],map_graph.g[map_graph.end])
    print(path)
    x1, y1 = int(path[0].split(",")[0]), int(path[0].split(",")[1])
    x1,y1 = grid_points[x1 * cols + y1]

    path_in_coords = []
    path_in_coords.append((x1,y1))

    # # # plotting the path
    for i in range(1,len(path)):
        x2, y2 = int(path[i].split(",")[0]), int(path[i].split(",")[1])
        x2,y2 = grid_points[x2 * cols + y2] 
        path_in_coords.append((x2,y2))
        ax.plot([x1, x2], [y1, y2], 'b-')
        x1 = x2
        y1 = y2
    minADE = compute_minADE(path_in_coords, target_path)
    averge_jerk, _ = compute_jerk(path_in_coords)
    print("Min ADE :", minADE)
    print("Average Jerk :", averge_jerk)
    plt.show()  

if __name__=='__main__':
    main()