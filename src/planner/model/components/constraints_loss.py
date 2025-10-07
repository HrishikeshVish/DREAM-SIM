import torch
import torch.nn as nn
import numpy as np

# Assume your dataclasses are importable
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from planner.data.dataclass import MapPoint, ObjectProperty, Trajectory, MapPolylineType, Scenario

class GuidanceLoss(nn.Module):
    def forward(
        self, 
        Y: torch.Tensor, 
        scenario: Scenario, 
        c_config: Dict,
        agt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        raise NotImplementedError
    

class TargetSpeedLoss(GuidanceLoss):
    """
    Computes a loss encouraging agents to follow a specific target speed.

    This version is refactored to be stateless and work directly with the
    Scenario dataclass and a dynamic configuration dictionary.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Y: torch.Tensor,
        scenario: Scenario,
        c_config: Dict,
        agt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the mean absolute error between predicted speed and target speed.

        Args:
            Y: The predicted future trajectory. 
               Shape: [B, N, T, D], where D must contain vx, vy.
            scenario: The input Scenario object.
            c_config: A dictionary that must contain the key 'target_speed'
                      with a tensor of shape [B, N, T].
            agt_mask: An optional boolean tensor to select specific agents.

        Returns:
            A scalar tensor representing the mean loss for this constraint.
        """
        # 1. Check if this constraint is active in the current config
        if "target_speed" not in c_config:
            return torch.tensor(0.0, device=Y.device)
        
        target_speed = c_config["target_speed"]
        
        # 2. Calculate predicted speed from the trajectory's velocity vectors
        #    Assuming state format is [x, y, vx, vy, yaw], so indices 2 and 3 are velocities.
        predicted_velocities = Y[..., 2:4]
        predicted_speed = torch.linalg.norm(predicted_velocities, dim=-1) # Shape: [B, N, T]
        
        # 3. Get the validity mask for the future horizon
        #    We only want to compute loss for timesteps where the agent is valid.
        ct = scenario.current_time_step
        horizon = Y.shape[2]
        # Slicing the valid mask from the log_trajectory to match the future horizon
        valid_mask = scenario.log_trajectory.valid[:, :, ct + 1 : ct + 1 + horizon]

        # 4. Apply agent mask if provided
        if agt_mask is not None:
            # Ensure agt_mask is broadcastable
            agt_mask_expanded = agt_mask.unsqueeze(-1) # Shape: [B, N, 1]
            predicted_speed = predicted_speed[agt_mask_expanded].reshape_as(predicted_speed)
            target_speed = target_speed[agt_mask_expanded].reshape_as(target_speed)
            valid_mask = valid_mask[agt_mask_expanded].reshape_as(valid_mask)

        # 5. Calculate the deviation (loss)
        speed_deviation = torch.abs(predicted_speed - target_speed)
        
        # 6. Apply the validity mask to ignore loss from invalid timesteps
        speed_deviation = torch.where(
            valid_mask, 
            speed_deviation, 
            torch.zeros_like(speed_deviation)
        )
        
        # 7. Compute the final mean loss
        #    We sum the deviations and divide by the number of valid steps to get a true mean
        num_valid_steps = torch.sum(valid_mask.float(), dim=-1).clamp(min=1.0)
        loss_per_agent = torch.sum(speed_deviation, dim=-1) / num_valid_steps
        
        # Return the mean loss over the batch and agents
        return torch.mean(loss_per_agent)
    

class AgentCollisionLoss(GuidanceLoss):
    """
    Computes a differentiable loss for inter-agent collisions.

    This version is refactored to work directly with the Scenario dataclass,
    assuming all inputs are in a global coordinate frame.
    """
    def __init__(self, num_disks: int = 5, buffer_dist: float = 0.2, decay_rate: float = 0.9):
        super().__init__()
        self.num_disks = num_disks
        self.buffer_dist = buffer_dist
        self.decay_rate = decay_rate

    def _get_agent_disks(self, scenario: Scenario):
        """
        Helper to compute initial disk centroids and radii for all agents in the batch.
        """
        ct = scenario.current_time_step
        # Get agent extents [length, width] at the current time
        extents = torch.stack([
            scenario.log_trajectory.length[:, :, ct],
            scenario.log_trajectory.width[:, :, ct]
        ], dim=-1) # Shape: [B, N, 2]

        B, N, _ = extents.shape
        
        # Agent radius is half the width
        agt_rad = extents[..., 1] / 2.0  # Shape: [B, N]
        
        # Calculate min/max for disk centroids along the vehicle's x-axis
        cent_min = -(extents[..., 0] / 2.) + agt_rad
        cent_max = (extents[..., 0] / 2.) - agt_rad
        
        # Generate disk centroids in agent-local frames
        # linspace doesn't support batching, so we loop over the batch
        all_cent_x = []
        for i in range(B):
            cent_x_b = torch.stack([
                torch.linspace(cent_min[i, j].item(), cent_max[i, j].item(), self.num_disks, device=extents.device)
                for j in range(N)
            ], dim=0)
            all_cent_x.append(cent_x_b)
        
        cent_x = torch.stack(all_cent_x) # Shape: [B, N, num_disks]
        
        # Centroids are along the x-axis in the local frame
        centroids = torch.stack([cent_x, torch.zeros_like(cent_x)], dim=-1) # Shape: [B, N, num_disks, 2]
        
        return centroids, agt_rad

    def forward(
        self,
        Y: torch.Tensor,
        scenario: Scenario,
        c_config: Dict, # c_config is available for future use (e.g., excluding pairs)
        agt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the differentiable collision loss.

        Args:
            Y: The predicted future trajectory. Shape: [B, N, T, D].
            scenario: The input Scenario object.
            c_config: Dictionary of constraint parameters.
            agt_mask: Optional boolean tensor to select agents.
        """
        # --- 1. Get Data & Predicted States ---
        pos_pred = Y[..., :2]   # Predicted XY positions. Shape: [B, N, T, 2]
        yaw_pred = Y[..., 4:5]  # Predicted yaw. Shape: [B, N, T, 1]

        B, N, T, _ = pos_pred.shape

        # --- 2. Compute Agent Geometries ---
        local_centroids, agt_rad = self._get_agent_disks(scenario)
        
        # Minimum distance threshold for collision = sum of radii + buffer
        penalty_dists = agt_rad.unsqueeze(2) + agt_rad.unsqueeze(1) + self.buffer_dist # Shape: [B, N, N]
        
        # --- 3. Transform Disks to Global Frame for Each Timestep ---
        local_centroids = local_centroids.unsqueeze(2).expand(-1, -1, T, -1, -1) # Shape: [B, N, T, num_disks, 2]
        
        # Create rotation matrices from predicted yaws
        s = torch.sin(yaw_pred).unsqueeze(-2)
        c = torch.cos(yaw_pred).unsqueeze(-2)
        rotM = torch.cat([c, -s, s, c], dim=-1).view(B, N, T, 1, 2, 2)
        
        # Rotate local centroids and add predicted global position
        # Using torch.matmul for batched matrix-vector multiplication
        world_centroids = (local_centroids.unsqueeze(-2) @ rotM).squeeze(-2) + pos_pred.unsqueeze(-2)
        # Final shape of world_centroids: [B, N, T, num_disks, 2]

        # --- 4. Compute Pairwise Distances Efficiently ---
        # Flatten time and disk dimensions to prepare for batched distance calculation
        world_centroids_flat = world_centroids.view(B, N, T * self.num_disks, 2)
        
        # torch.cdist computes pairwise distances for each item in the batch
        # Input: [B, N, P, D], Output: [B, N, N] where each element is min distance
        dists = torch.cdist(world_centroids_flat, world_centroids_flat) # Shape: [B, N, N, T*num_disks, T*num_disks]
        
        # Find the minimum distance between any two disks of two agents at each timestep
        min_dists_over_time = []
        for t in range(T):
            start, end = t * self.num_disks, (t + 1) * self.num_disks
            time_slice = dists[..., start:end, start:end]
            # Find min distance between disk sets of agent i and agent j
            min_dist_t = time_slice.flatten(-2).min(-1).values
            min_dists_over_time.append(min_dist_t)
            
        pair_dists = torch.stack(min_dists_over_time, dim=-1) # Final shape: [B, N, N, T]

        # --- 5. Calculate Collision Penalty ---
        # Create a mask to exclude checking an agent against itself
        no_self_collision_mask = ~torch.eye(N, device=Y.device, dtype=torch.bool).view(1, N, N, 1)
        
        # A collision occurs if the minimum distance is less than the penalty distance
        is_colliding = (pair_dists <= penalty_dists.unsqueeze(-1)) & no_self_collision_mask
        
        # Penalty is 1 at contact, and decreases to 0 at the buffer distance
        penalties = 1.0 - (pair_dists / penalty_dists.unsqueeze(-1))
        penalties = torch.where(is_colliding, torch.relu(penalties), torch.zeros_like(penalties))

        # Sum penalties for each agent (sum over all other agents it collides with)
        agent_penalties = torch.sum(penalties, dim=2) # Shape: [B, N, T]

        # --- 6. Final Loss Calculation ---
        # Apply temporal decay weight (penalize earlier collisions more)
        exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=Y.device)
        exp_weights /= exp_weights.sum()
        
        # Apply weights and take the mean over batch, agents, and time
        final_loss = torch.mean(agent_penalties * exp_weights.view(1, 1, T))
        
        return final_loss
    

class TargetPosAtTimeLoss(GuidanceLoss):
    """
    Computes a loss for hitting a specific waypoint at a specific time step.
    
    This version is refactored to be stateless and work directly with the
    Scenario dataclass and a dynamic configuration dictionary.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Y: torch.Tensor,
        scenario: Scenario,
        c_config: Dict,
        agt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the Euclidean distance to a target position at a target time.

        Args:
            Y: The predicted future trajectory. Shape: [B, N, T, D].
            scenario: The input Scenario object.
            c_config: A dictionary that must contain:
                      - 'target_positions': Tensor of shape [B, N, 2]
                      - 'target_times': Long tensor of shape [B, N] of time indices.
            agt_mask: Optional boolean tensor to select agents.

        Returns:
            A scalar tensor representing the mean loss for this constraint.
        """
        # 1. Check if this constraint is active in the current config
        if "target_positions" not in c_config or "target_times" not in c_config:
            return torch.tensor(0.0, device=Y.device)
        
        target_pos = c_config["target_positions"]
        target_time = c_config["target_times"] # Shape: [B, N]

        # 2. Extract predicted positions from the future trajectory Y
        pred_pos = Y[..., :2] # Shape: [B, N, T, 2]
        B, N, T, _ = pred_pos.shape

        # 3. Use torch.gather for efficient, batched indexing
        # This selects the predicted position at the specified target_time for each agent.
        
        # Reshape target_time to be used as an index
        # Index must have the same number of dimensions as the tensor being indexed
        time_idx = target_time.view(B, N, 1, 1).expand(-1, -1, -1, 2) # Shape: [B, N, 1, 2]
        
        # Gather the positions at the target timesteps
        pos_at_target_time = torch.gather(pred_pos, 2, time_idx).squeeze(2) # Shape: [B, N, 2]
        
        # 4. Calculate the loss (Euclidean distance)
        # L2 norm between the predicted position and the target position
        loss_per_agent = torch.linalg.norm(pos_at_target_time - target_pos, dim=-1) # Shape: [B, N]

        # 5. Apply validity and agent masks
        # We only care about the loss for agents that are valid
        ct = scenario.current_time_step
        valid_mask = scenario.object_property.valid # Shape: [B, N]

        if agt_mask is not None:
            valid_mask = valid_mask & agt_mask

        # Zero out the loss for invalid agents
        loss_per_agent = torch.where(
            valid_mask,
            loss_per_agent,
            torch.zeros_like(loss_per_agent)
        )
        
        # 6. Return the mean loss over all valid agents in the batch
        # We divide by the number of valid agents to get a true mean
        num_valid_agents = torch.sum(valid_mask.float()).clamp(min=1.0)
        
        return torch.sum(loss_per_agent) / num_valid_agents
    

class TargetPosLoss(GuidanceLoss):
    """
    Computes a loss for passing near a specific waypoint at some point
    in the future horizon.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        Y: torch.Tensor,
        scenario: Scenario,
        c_config: Dict,
        agt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates a soft-min weighted distance to a target position.

        Args:
            Y: The predicted future trajectory. Shape: [B, N, T, D].
            scenario: The input Scenario object.
            c_config: A dictionary that must contain:
                      - 'target_pos': Tensor of shape [B, N, 2]
                      - Optional 'min_target_time': Float (0.0 to 1.0) for the
                        start of the evaluation window. Defaults to 0.0.
            agt_mask: Optional boolean tensor to select agents.

        Returns:
            A scalar tensor representing the mean loss for this constraint.
        """
        # 1. Check if this constraint is active in the current config
        if "target_pos" not in c_config:
            return torch.tensor(0.0, device=Y.device)

        target_pos = c_config["target_pos"]
        min_target_time_ratio = c_config.get("min_target_time", 0.0)

        # 2. Extract predicted positions and apply time window
        pred_pos = Y[..., :2] # Shape: [B, N, T, 2]
        T = pred_pos.shape[2]
        
        # Calculate the first timestep to consider for the loss
        min_t = int(min_target_time_ratio * T)
        sliced_pred_pos = pred_pos[:, :, min_t:, :]

        # 3. Calculate distance from every future point to the target
        #    Use unsqueeze to make target_pos broadcastable with the trajectory
        #    sliced_pred_pos: [B, N, T_slice, 2]
        #    target_pos.unsqueeze(2): [B, N, 1, 2]
        dist = torch.linalg.norm(sliced_pred_pos - target_pos.unsqueeze(2), dim=-1)
        # dist shape: [B, N, T_slice]

        # 4. Core Logic: Use softmin to weight the closest timesteps most heavily
        #    softmin(dist) is equivalent to softmax(-dist), giving higher weight
        #    to smaller distances.
        loss_weighting = F.softmin(dist, dim=-1)

        # The loss is the weighted sum of the squared distances
        loss_per_agent = torch.sum(loss_weighting * (dist**2), dim=-1) # Shape: [B, N]

        # 5. Apply validity and agent masks
        valid_mask = scenario.object_property.valid # Shape: [B, N]

        if agt_mask is not None:
            valid_mask = valid_mask & agt_mask

        loss_per_agent = torch.where(
            valid_mask,
            loss_per_agent,
            torch.zeros_like(loss_per_agent)
        )
        
        # 6. Return the mean loss over all valid agents in the batch
        num_valid_agents = torch.sum(valid_mask.float()).clamp(min=1.0)
        
        return torch.sum(loss_per_agent) / num_valid_agents
    
class VectorMapOffroadLoss(GuidanceLoss):
    """
    Computes a differentiable off-road loss using vectorized map polylines.
    This replaces the original raster-based MapCollisionLoss.
    """
    def __init__(self, lane_half_width: float = 2.0, decay_rate: float = 0.9):
        super().__init__()
        self.lane_half_width = lane_half_width
        self.decay_rate = decay_rate

    def _point_to_segment_distance(self, p: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Calculates the minimum distance from a batch of points `p` to a batch of line segments `ab`.
        All inputs are assumed to be broadcastable.

        Args:
            p: Points, shape [..., 2]
            a: Segment start points, shape [..., 2]
            b: Segment end points, shape [..., 2]

        Returns:
            Minimum distance from each point to its corresponding segment.
        """
        # Vector from A to B, and from A to P
        ab = b - a
        ap = p - a
        
        # Project AP onto AB, calculating the parameter t
        # t = dot(AP, AB) / dot(AB, AB)
        proj = torch.sum(ap * ab, dim=-1)
        len_sq = torch.sum(ab * ab, dim=-1).clamp(min=1e-6) # Avoid division by zero
        t = (proj / len_sq).clamp(0.0, 1.0) # Clamp t to be on the segment [A, B]
        
        # The closest point on the line segment is A + t * (B - A)
        closest_point = a + t.unsqueeze(-1) * ab
        
        # Return the Euclidean distance from P to the closest point
        return torch.linalg.norm(p - closest_point, dim=-1)

    def forward(
        self,
        Y: torch.Tensor,
        scenario: Scenario,
        c_config: Dict,
        agt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculates the differentiable off-road loss.

        Args:
            Y: The predicted future trajectory. Shape: [B, N, T, D].
            scenario: The input Scenario object containing map data.
            c_config: Dictionary of constraint parameters.
            agt_mask: Optional boolean tensor to select agents.
        """
        agent_positions = Y[..., :2] # Shape: [B, N, T, 2]
        map_points = scenario.map_point
        B, N, T, _ = agent_positions.shape
        
        total_loss = torch.tensor(0.0, device=Y.device)

        # Process each scene in the batch individually due to varying map structures
        for i in range(B):
            # --- 1. Get Lane Centerline Segments for the current scene ---
            scene_map = map_points[i]
            lane_mask = (scene_map.types == MapPolylineType.LANE_CENTER_VEHICLE) & scene_map.valid
            
            if not torch.any(lane_mask):
                continue # No lanes in this scene, so no off-road loss

            lane_points = scene_map.xy[lane_mask]
            lane_ids = scene_map.ids[lane_mask]

            # Group points into polylines and create line segments
            # This is a critical data processing step
            segments_a, segments_b = [], []
            unique_ids = torch.unique(lane_ids)
            for polyline_id in unique_ids:
                polyline_points = lane_points[lane_ids == polyline_id]
                if len(polyline_points) > 1:
                    segments_a.append(polyline_points[:-1])
                    segments_b.append(polyline_points[1:])
            
            if not segments_a:
                continue

            segments_a = torch.cat(segments_a, dim=0) # Shape: [NumSegments, 2]
            segments_b = torch.cat(segments_b, dim=0) # Shape: [NumSegments, 2]

            # --- 2. Calculate Minimum Distance from Agents to Lane Segments ---
            # Prepare tensors for broadcasting
            # Agent pos: [N, T, 1, 2]
            # Segments:  [1, 1, NumSegments, 2]
            scene_agent_pos = agent_positions[i].unsqueeze(2)
            segments_a = segments_a.view(1, 1, -1, 2)
            segments_b = segments_b.view(1, 1, -1, 2)
            
            # This computes the distance from each agent at each time to EVERY lane segment
            all_distances = self._point_to_segment_distance(scene_agent_pos, segments_a, segments_b)
            # Shape of all_distances: [N, T, NumSegments]
            
            # Find the minimum distance to any lane segment for each agent at each time
            min_dist_to_lane = torch.min(all_distances, dim=-1).values # Shape: [N, T]

            # --- 3. Compute Off-road Penalty ---
            # The loss is the amount by which an agent's distance exceeds the lane half-width
            offroad_error = torch.relu(min_dist_to_lane - self.lane_half_width)
            
            # We only care about valid agents in the current scene
            valid_mask = scenario.object_property.valid[i] # Shape: [N]
            
            # Apply temporal decay
            exp_weights = torch.tensor([self.decay_rate ** t for t in range(T)], device=Y.device)
            exp_weights /= exp_weights.sum()

            # Weight the error by time and sum over the time dimension
            loss_per_agent = torch.sum((offroad_error**2) * exp_weights, dim=1) # Shape: [N]
            
            # Zero out loss for invalid agents and add to batch total
            total_loss += torch.mean(loss_per_agent[valid_mask])

        # Return the mean loss over the batch
        return total_loss / B
    

class DifferentiableConstraintLosses(nn.Module):
    """
    Orchestrates all guidance losses and manages their learnable weights.
    This is the final, comprehensive module.
    """
    def __init__(self, initial_weights: Dict[str, float] = None):
        super().__init__()

        # --- Instantiate all available loss modules ---
        self.agent_collision = AgentCollisionLoss()
        self.vector_map_offroad = VectorMapOffroadLoss()
        self.target_speed = TargetSpeedLoss()
        self.target_pos_at_time = TargetPosAtTimeLoss()
        self.target_pos = TargetPosLoss()
        
        # --- Define Learnable Weights for each Loss ---
        # We store the raw parameters and apply a softplus to ensure they are non-negative.
        # Initializing with log(exp(w)-1) makes the initial weight approx `w`.
        if initial_weights is None:
            initial_weights = {
                "collision": 1.0, "offroad": 1.0, "speed": 0.1,
                "pos_at_time": 0.5, "pos": 0.5
            }

        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.exp(torch.tensor(weight)) - 1.0))
            for name, weight in initial_weights.items()
        })

    def forward(
        self,
        Y: torch.Tensor,
        c_config: Dict,
        scenario: Scenario
    ) -> torch.Tensor:
        """
        Computes the total, weighted guidance loss for a predicted FUTURE trajectory `Y`.
        """
        total_loss = torch.tensor(0.0, device=Y.device)

        # --- Dispatch to the appropriate loss functions ---

        # Collision is a general constraint, usually always active
        weight_collision = F.softplus(self.log_weights["collision"])
        total_loss += weight_collision * self.agent_collision(Y, scenario, c_config)

        # Off-road is a general constraint, usually always active
        weight_offroad = F.softplus(self.log_weights["offroad"])
        total_loss += weight_offroad * self.vector_map_offroad(Y, scenario, c_config)

        # Target-based constraints are only active if specified in c_config
        if "target_speed" in c_config:
            weight_speed = F.softplus(self.log_weights["speed"])
            total_loss += weight_speed * self.target_speed(Y, scenario, c_config)

        if "target_positions" in c_config and "target_times" in c_config:
            weight_pos_at_time = F.softplus(self.log_weights["pos_at_time"])
            total_loss += weight_pos_at_time * self.target_pos_at_time(Y, scenario, c_config)

        if "target_pos" in c_config:
            weight_pos = F.softplus(self.log_weights["pos"])
            total_loss += weight_pos * self.target_pos(Y, scenario, c_config)
            
        return total_loss