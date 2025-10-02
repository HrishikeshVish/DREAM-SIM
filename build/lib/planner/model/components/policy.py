# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Policy network modules."""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from planner.model.function import get_topk_neighbors
from planner.model.module.init import variance_scaling
from planner.model.module.layers import embedding, layer_norm, linear, lstm
from planner.model.module.positional import sinusoidal_positional_encoding
from planner.model.module.rst import RSTEncoder
from planner.data.dataclass import Scenario, MapPolylineType, ObjectType, MapPoint
import torchsde
from torch.cuda.amp import autocast
import math
import time
from enum import IntEnum
# Type Aliases
tensor_2_t = Tuple[torch.Tensor, torch.Tensor]
tensor_3_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

TIKHONOV_REGULARIZATION = 0.5 / math.pi


from torch.nn.utils import weight_norm

class GaussianFourierProjection(nn.Module):
    """Gaussian Fourier features for encoding time."""
    def __init__(self, embedding_size=256, scale=1.0):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_size) * scale, requires_grad=False)

    def forward(self, t):
        t_proj = t.view(-1, 1) * self.W.view(1, -1) * 2 * math.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class ResnetBlockDDPM1d(nn.Module):
    """ResNet block adapted for 1D sequences."""
    def __init__(self, in_ch, out_ch=None, temb_dim=None, dropout=0.1):
        super().__init__()
        out_ch = out_ch or in_ch
        self.norm1 = nn.GroupNorm(16, in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        
        self.temb_proj = None
        if temb_dim is not None:
            self.temb_proj = nn.Linear(temb_dim, out_ch)

        self.norm2 = nn.GroupNorm(16, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)
        
        self.nin_shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        if self.temb_proj is not None:
            h += self.temb_proj(self.act(temb))[:, :, None]

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.nin_shortcut(x)

class AttnBlock1d(nn.Module):
    """Self-attention block for 1D sequences."""
    def __init__(self, in_ch):
        super().__init__()
        self.norm = nn.GroupNorm(16, in_ch)
        self.q = nn.Conv1d(in_ch, in_ch, 1)
        self.k = nn.Conv1d(in_ch, in_ch, 1)
        self.v = nn.Conv1d(in_ch, in_ch, 1)
        self.proj_out = nn.Conv1d(in_ch, in_ch, 1)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, l = q.shape
        q = q.permute(0, 2, 1) # B, L, C
        k = k.view(b, c, l) # B, C, L
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=-1)

        w_ = w_.permute(0, 2, 1) # B, L, L
        h_ = torch.bmm(v, w_)
        h_ = h_.view(b, c, l)

        return x + self.proj_out(h_)

class Downsample1d(nn.Module):
    def __init__(self, in_ch, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        if self.with_conv:
            return self.conv(x)
        else:
            return F.avg_pool1d(x, kernel_size=2, stride=2)

class Upsample1d(nn.Module):
    def __init__(self, in_ch, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv1d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, target_size):
        # Use `size` argument instead of `scale_factor` to ensure exact match
        x = F.interpolate(x, size=target_size, mode='linear', align_corners=False)
        if self.with_conv:
            x = self.conv(x)
        return x


# =====================================================================================
# == The Main NCSNpp Class (Adapted for 1D Trajectories)
# =====================================================================================

class NCSNpp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = nn.SiLU()
        
        self.nf = config['nf']
        ch_mult = config['ch_mult']
        self.num_res_blocks = config['num_res_blocks']
        self.attn_resolutions = config['attn_resolutions']
        dropout = config['dropout']
        resamp_with_conv = config['resamp_with_conv']
        self.num_resolutions = len(ch_mult)
        
        self.time_embedding = GaussianFourierProjection(embedding_size=self.nf)
        self.temb_dense1 = nn.Linear(self.nf * 2, self.nf * 4)
        self.temb_dense2 = nn.Linear(self.nf * 4, self.nf * 4)

        # Downsampling path
        self.conv_in = nn.Conv1d(config['latent_dim'], self.nf, kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList()
        hs_channels = [self.nf]
        in_ch = self.nf
        
        for i_level in range(self.num_resolutions):
            out_ch = self.nf * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                self.down_blocks.append(ResnetBlockDDPM1d(in_ch, out_ch=out_ch, temb_dim=self.nf * 4, dropout=dropout))
                in_ch = out_ch
                hs_channels.append(in_ch)
            if i_level != self.num_resolutions - 1:
                self.down_blocks.append(Downsample1d(in_ch, resamp_with_conv))
                hs_channels.append(in_ch)

        # Bottleneck
        self.mid_block1 = ResnetBlockDDPM1d(in_ch, temb_dim=self.nf * 4, dropout=dropout)
        self.mid_attn = AttnBlock1d(in_ch)
        self.mid_block2 = ResnetBlockDDPM1d(in_ch, temb_dim=self.nf * 4, dropout=dropout)

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            out_ch = self.nf * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                self.up_blocks.append(ResnetBlockDDPM1d(in_ch + hs_channels.pop(), out_ch=out_ch, temb_dim=self.nf * 4, dropout=dropout))
                in_ch = out_ch
            if i_level != 0:
                self.up_blocks.append(Upsample1d(in_ch, resamp_with_conv))
        
        self.norm_out = nn.GroupNorm(16, in_ch)
        self.conv_out = nn.Conv1d(in_ch, config['latent_dim'], kernel_size=3, stride=1, padding=1)

    def forward(self, z_t, t, context_token=None, key_padding_mask=None):
        x = z_t.permute(0, 2, 1)

        temb = self.time_embedding(t)
        temb = self.temb_dense1(temb)
        temb = self.act(temb)
        temb = self.temb_dense2(temb)

        # Downsampling
        h = self.conv_in(x)
        hs = [h]
        for module in self.down_blocks:
            h = module(h, temb) if isinstance(module, ResnetBlockDDPM1d) else module(h)
            hs.append(h)
        
        # Bottleneck
        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)
        
        # Upsampling
        for module in self.up_blocks:
            if isinstance(module, ResnetBlockDDPM1d):
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = module(h, temb)
            else: # Upsample1d
                target_size = hs[-1].shape[-1]
                h = module(h, target_size)
                
        # Final
        h = self.norm_out(h)
        h = self.act(h)
        h = self.conv_out(h)

        return h.permute(0, 2, 1)

class GatingNetwork(nn.Module):
    def __init__(self, context_dim: int, num_energy_terms: int = 3):
        super().__init__()
        self.num_energy_terms = num_energy_terms
        self.net = nn.Sequential(
            nn.Linear(context_dim*10, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_energy_terms)
        )

    def forward(self, context: torch.Tensor, num_scenes: int, num_agents: int, num_intentions: int) -> torch.Tensor:
        
        context = context.view(-1, context.shape[-1]*context.shape[-2])
        logits_flat = self.net(context)

        #print(num_scenes, num_agents, num_intentions)

        logits = logits_flat.view(num_scenes, num_agents, num_intentions, self.num_energy_terms)
        #print("Logits Shape: ", logits.shape)
        weights = F.softmax(logits, dim=-1)
        return weights
    

class SceneGuidance(nn.Module):
    def __init__(self, grid_resolution: int, domain_extents: tuple, k: float, solver_iterations: int):
        super().__init__()
        self.grid_resolution = grid_resolution
        self.domain_extents = domain_extents
        self.k_squared = k**2
        self.solver_iterations = solver_iterations
        self.dx = (domain_extents[1] - domain_extents[0]) / (grid_resolution - 1)
        self.laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=torch.float32)

    def _world_to_grid(self, coords: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(coords[..., 0], self.domain_extents[0], self.domain_extents[1])
        y = torch.clamp(coords[..., 1], self.domain_extents[2], self.domain_extents[3])
        return torch.stack([(x - self.domain_extents[0]) / self.dx, (y - self.domain_extents[2]) / self.dx], dim=-1).long()

    def _solve_poisson(self, boundaries: torch.Tensor) -> torch.Tensor:
        u = boundaries.clone()
        is_boundary = (boundaries != 0.5)
        denominator = 4.0 + self.k_squared * (self.dx**2)
        kernel = self.laplacian_kernel.to(u.device)
        for _ in range(self.solver_iterations):
            u_laplacian = F.conv2d(u.unsqueeze(0).unsqueeze(0), kernel, padding=1).squeeze()
            u_new = u_laplacian / denominator
            u = torch.where(is_boundary, u, u_new)
        return u

    def forward(self, trajectory, goal, obstacles, map_points):
        boundaries = torch.full((self.grid_resolution, self.grid_resolution), 0.5, device=trajectory.device)
        map_coords = map_points.xy[map_points.valid]
        is_solid = (map_points.types == MapPolylineType.LANE_BOUNDARY) & map_points.valid
        if is_solid.any():
            start = time.time()
            solid_coords = self._world_to_grid(map_coords[is_solid[map_points.valid]])
            boundaries[solid_coords[:, 1], solid_coords[:, 0]] = 0.0
            end = time.time()
            #print("Solid Boundary Time: ", end - start)
        if obstacles.numel() > 0:
            start = time.time()
            obs_grid = self._world_to_grid(obstacles.view(-1, 2))
            boundaries[obs_grid[:, 1], obs_grid[:, 0]] = 0.0
            end = time.time()
            #print("Obstacle Boundary Time: ", end - start)
        start = time.time()
        goal_grid = self._world_to_grid(goal)
        end = time.time()
        #print("Goal Grid Time: ", end - start)
        boundaries[goal_grid[1], goal_grid[0]] = 1.0
        start = time.time()
        potential_field = self._solve_poisson(boundaries)
        end = time.time()
        ##print("Poisson Solve Time: ", end - start)
        traj_norm_x = (trajectory[..., 0] - self.domain_extents[0]) / (self.domain_extents[1] - self.domain_extents[0]) * 2 - 1
        traj_norm_y = (trajectory[..., 1] - self.domain_extents[2]) / (self.domain_extents[3] - self.domain_extents[2]) * 2 - 1
        traj_normalized = torch.stack([traj_norm_x, traj_norm_y], dim=-1)
        sampled_potentials = F.grid_sample(potential_field.unsqueeze(0).unsqueeze(0), traj_normalized.unsqueeze(0).unsqueeze(0), mode='bilinear', align_corners=True, padding_mode="border").squeeze()
        return -torch.sum(sampled_potentials)
    


class DirectionalGuidance(nn.Module):
    def __init__(self, penalty_weight: float):
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, trajectories, initial_headings):
        headings_norm = F.normalize(initial_headings, p=2, dim=-1)
        displacement = trajectories[:, -1, :] - trajectories[:, 0, :]
        left_vector = torch.stack([-headings_norm[:, 1], headings_norm[:, 0]], dim=-1)
        left_drift = torch.einsum('ni,ni->n', displacement, left_vector)
        energy_left = -left_drift + self.penalty_weight * F.relu(-left_drift)
        energy_right = left_drift + self.penalty_weight * F.relu(left_drift)
        return energy_left, energy_right

class GuidanceController(nn.Module):
    def __init__(self, grid_resolution: int, domain_extents: tuple, k: float, solver_iterations: int, drift_penalty_weight: float, receding_horizon_distance: float = 50.0):
        super().__init__()
        # Note: StraightLineGuidance is now used as a container for its helper methods and parameters
        self.straight_guider_params = nn.Module()
        self.straight_guider_params.grid_resolution = grid_resolution
        self.straight_guider_params.domain_extents = domain_extents
        self.straight_guider_params.k_squared = k**2
        self.straight_guider_params.solver_iterations = solver_iterations
        self.straight_guider_params.dx = (domain_extents[1] - domain_extents[0]) / (grid_resolution - 1)
        self.straight_guider_params.laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, 0, 1], [0, 1, 0]]]], dtype=torch.float32)

        self.turn_guider = DirectionalGuidance(drift_penalty_weight)
        self.receding_horizon_distance = receding_horizon_distance

    def _world_to_grid_batched(self, coords: torch.Tensor) -> torch.Tensor:
        # coords shape: [N, ..., 2]
        x = torch.clamp(coords[..., 0], self.straight_guider_params.domain_extents[0], self.straight_guider_params.domain_extents[1])
        y = torch.clamp(coords[..., 1], self.straight_guider_params.domain_extents[2], self.straight_guider_params.domain_extents[3])
        grid_x = (x - self.straight_guider_params.domain_extents[0]) / self.straight_guider_params.dx
        grid_y = (y - self.straight_guider_params.domain_extents[2]) / self.straight_guider_params.dx
        return torch.stack([grid_x, grid_y], dim=-1).long()

    def _solve_poisson_batched(self, boundaries_batch: torch.Tensor) -> torch.Tensor:
        u = boundaries_batch.clone()
        is_boundary = (boundaries_batch != 0.5)
        denominator = 4.0 + self.straight_guider_params.k_squared * (self.straight_guider_params.dx**2)
        kernel = self.straight_guider_params.laplacian_kernel.to(u.device)
        for _ in range(self.straight_guider_params.solver_iterations):
            u_laplacian = F.conv2d(u.unsqueeze(1), kernel, padding=1).squeeze(1)
            u_new = u_laplacian / denominator
            u = torch.where(is_boundary, u, u_new)
        return u

    def forward(self, agent_positions_flat, guidance_weights, initial_headings_flat, map_points, num_scenes, num_agents, num_intentions):
        N = num_scenes * num_agents * num_intentions
        
        guidance_weights_flat = guidance_weights.view(N, 3)
        w_lane, w_left, w_right = guidance_weights_flat.unbind(dim=-1)

        energy_left, energy_right = self.turn_guider(agent_positions_flat, initial_headings_flat)
        
        energy_lane = torch.zeros(N, device=agent_positions_flat.device)
        active_lane_indices = torch.where(w_lane > 1e-4)[0]

        if active_lane_indices.numel() > 0:
            # --- Step 1: Prepare Inputs for Active Trajectories ---
            active_trajs = agent_positions_flat[active_lane_indices]
            active_headings = initial_headings_flat[active_lane_indices]
            
            receding_goals = active_trajs[:, 0, :] + F.normalize(active_headings, p=2, dim=-1) * self.receding_horizon_distance
            
            # --- Step 2: Construct Boundary Grids in Parallel ---
            num_active = active_lane_indices.shape[0]
            boundaries_batch = torch.full((num_active, self.straight_guider_params.grid_resolution, self.straight_guider_params.grid_resolution), 0.5, device=agent_positions_flat.device)
            
            goal_coords_grid = self._world_to_grid_batched(receding_goals)
            batch_indices = torch.arange(num_active, device=agent_positions_flat.device)
            boundaries_batch[batch_indices, goal_coords_grid[:, 1], goal_coords_grid[:, 0]] = 1.0
            
            # --- Step 3: Solve all PDEs in Parallel ---
            potential_fields = self._solve_poisson_batched(boundaries_batch)
            
            # --- Step 4: Sample Potentials in Parallel ---
            traj_norm_x = (active_trajs[..., 0] - self.straight_guider_params.domain_extents[0]) / (self.straight_guider_params.domain_extents[1] - self.straight_guider_params.domain_extents[0]) * 2 - 1
            traj_norm_y = (active_trajs[..., 1] - self.straight_guider_params.domain_extents[2]) / (self.straight_guider_params.domain_extents[3] - self.straight_guider_params.domain_extents[2]) * 2 - 1
            traj_normalized = torch.stack([traj_norm_x, traj_norm_y], dim=-1) # Shape: [num_active, T, 2]
            
            sampled_potentials = F.grid_sample(potential_fields.unsqueeze(1), traj_normalized.unsqueeze(1), mode='bilinear', align_corners=True, padding_mode="border").squeeze()
            energy_lane.scatter_(0, active_lane_indices, -torch.sum(sampled_potentials, dim=-1))

        weighted_energy = w_lane * energy_lane + w_left * energy_left + w_right * energy_right
        total_energy = torch.sum(weighted_energy)
        
        return total_energy


def sample_with_edm(
    denoiser_network,
    initial_noise,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80,
    rho=7,
):
    """
    Generates trajectories using the pure, unguided EDM ODE sampler (Euler method).

    Args:
        denoiser_network: The trained network that takes a noisy trajectory `z_t`
                          and a noise level `sigma_t` and outputs a denoised
                          trajectory `z_0_predicted`.
        initial_noise: A tensor of pure random noise with the same shape as the
                       trajectories to be generated.
        num_steps: The number of integration steps.
        sigma_min: The minimum noise level.
        sigma_max: The maximum noise level.
        rho: The power for the time step discretization.
    """
    
    # 1. Time step discretization.
    # Create the noise schedule (sigmas) from high to low.
    step_indices = torch.arange(num_steps, device=initial_noise.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # 2. Initialize the state with pure noise scaled by the highest sigma.
    z_next = initial_noise * t_steps[0]

    # --- Main Sampling Loop ---
    for i, (sigma_cur, sigma_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        z_cur = z_next

        # 3. Get the denoised prediction from your network.
        # The network's output is its best guess of the final clean trajectory.
        denoised = denoiser_network(z_cur, sigma_cur)
        
        # 4. Calculate the derivative for the ODE step (the "drift").
        d_cur = (z_cur - denoised) / sigma_cur
        
        # 5. Perform the Euler step to get the state at the next noise level.
        # dt is (sigma_next - sigma_cur), which is negative.
        z_next = z_cur + (sigma_next - sigma_cur) * d_cur

    return z_next

def refine_with_edm_guidance(
    policy_to_refine,
    sigma_start,
    denoiser_network,
    lstm_prior_context,
    mask,
    guidance_scale=1.0,
    num_steps=10,
    sigma_min=0.002,
    rho=7,
):
    """
    Refines an existing trajectory using the guided EDM ODE sampler.
    """
    
    # 1. Add a controlled amount of noise to the initial policy
    #noise = torch.randn_like(policy_to_refine)
    z_start = policy_to_refine.detach()

    # 2. Define the time step discretization from sigma_start down to 0
    step_indices = torch.arange(num_steps, device=policy_to_refine.device)
    t_steps = (sigma_start ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_start ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
    
    # 3. Initialize the trajectory with the noised-up policy
    z_next = z_start
    denoiser_network.eval()
    # --- Main Refinement Loop ---
    for i, (sigma_cur, sigma_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        z_cur = z_next

        # Get the realism-based denoised prediction
        with torch.no_grad():
            denoised_uncond = denoiser_network(z_cur, sigma_cur, context_token=lstm_prior_context, key_padding_mask=~mask)

        # Apply guidance to the denoised prediction

        # Apply the gradient to steer the denoised output
        #denoised_final = denoised_uncond - guidance_grad * guidance_scale
        
        # Perform the ODE solver step
        d_cur = (z_cur - denoised_uncond) / sigma_cur
        print(" AT t = ", i, " Guidance score: ", d_cur.norm().item())
        z_next = z_cur + (sigma_next - sigma_cur) * d_cur

    return z_next


class VpSde(nn.Module):
    noise_type = 'diagonal'
    sde_type = 'ito'

    def __init__(self, latent_dim: int, num_agents: int, beta_min=0.1, beta_max=5.0):
        super().__init__()
        # Store shape information for reshaping
        self.latent_dim = latent_dim
        self.num_agents = num_agents
        
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_integration_steps = 1000
        self.num_energy_terms = 3
        self.guidance_module = GuidanceController(grid_resolution=10, domain_extents=(-10, 10, -10, 10), k=5, solver_iterations=10, drift_penalty_weight=1.0, receding_horizon_distance=5.0)

    def beta(self, t):
        return self.beta_min + t * (self.beta_max - self.beta_min) #Sigmoid or Linear
    
    def beta_sigmoid(self, t):
        """
        Implements a continuous-time sigmoid beta schedule.
        """
        # Define the range for the input to the sigmoid function.
        # A range like [-6, 6] gives a nice steep curve.
        start = -5.0
        end = 5.0
        T_max = 1.0
        
        # 1. Rescale t from [0, T_max] to [start, end]
        t_rebalanced = (t / T_max) * (end - start) + start
        
        # 2. Apply the sigmoid function
        sigmoid_output = torch.sigmoid(t_rebalanced)
        
        # 3. Scale the sigmoid output [0, 1] to the desired [beta_min, beta_max] range
        beta_t = self.beta_min + sigmoid_output * (self.beta_max - self.beta_min)
        
        return beta_t

    # def marginal_prob(self, z0, t):
    #     # This function is correct and remains unchanged
    #     log_mean_coeff = -0.25 * t ** 2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
    #     mean = torch.exp(log_mean_coeff.view(-1, 1, 1)) * z0
    #     std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff.view(-1, 1, 1)))
    #     return mean, std

    def marginal_prob(self, z0, t):
        """
        Calculates the mean and std of the marginal distribution p(z_t|z_0)
        by numerically integrating the beta schedule in a vectorized manner.
        """
        # Ensure t is a column vector for broadcasting
        """
        Calculates the mean and std of the marginal distribution p(z_t|z_0)
        by numerically integrating the beta schedule in a vectorized manner.
        """
        # Ensure t is a column vector for broadcasting
        t_col = t.view(-1, 1) # Shape: [B, 1]

        log_mean_coeff = -0.25 *t_col **2 * (20 - 0.1) - 0.5 * t_col * 0.1
        log_mean_coeff = log_mean_coeff.view(-1, *([1] * (z0.dim() - 1)))
        mean = torch.exp(log_mean_coeff) * z0
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        
        return mean, std
        
        #return mean, std
    
    def f_forward(self, t, z_flat):
        """ The drift for the FORWARD SDE (data to noise). Signature is f(t, y). """
        B = z_flat.shape[0]
        z = z_flat.view(B, self.num_agents, self.latent_dim)
        beta_t = self.beta(t)
        drift = -0.5 * beta_t.view(-1, 1, 1) * z
        return drift.view(B, -1) # Flatten back

    def f_reverse(self, t, z_flat, score_network, context):
        """ The drift for the REVERSE SDE (noise to data). """
        B = z_flat.shape[0]
        z = z_flat.view(B, self.num_agents, self.latent_dim)
        
        beta_t = self.beta(t)
        f_forward_drift = -0.5 * beta_t.view(-1, 1, 1) * z
        score = score_network(z, t, context)
        drift = f_forward_drift - beta_t.view(-1, 1, 1) * score
        
        return drift.view(B, -1)

    def f_reverse_guided(self, t, noise_latent, score_network, context, cur_num_agents, gating_network, 
                         emission_head, assigned_types, map_point, w_collision, w_layout, sde_t, lstm_prior, batch_dims):
        """ The drift for the REVERSE SDE (noise to data) with guidance. """

        noise_latent = noise_latent.unsqueeze(0) if len(noise_latent.shape) == 2 else noise_latent
        B, N, I = batch_dims

        try:
            z = noise_latent.view(-1, 10, self.latent_dim)
            score_context = context.view(-1, 10, self.latent_dim)[:, 0, :]
        except:
            print("Error in reshaping noise_latent")
            print("B:", B, "num_agents:", cur_num_agents, "latent_dim:", self.latent_dim, "shape:", noise_latent.shape)
            exit()
        z.requires_grad_(True)
        vec_t = sde_t

        #print("Z Shape: ", z.shape)

        vec_t = torch.ones(z.shape[0], device=z.device) * t

        with torch.no_grad():
            s_realism = score_network(z, vec_t, context_token=score_context, key_padding_mask=None)

        with torch.enable_grad():
            maneuver_probs = gating_network(lstm_prior, num_scenes=B, num_agents=N, num_intentions=I) # Shape: [B, I, 3]
            probs_flat = maneuver_probs.view(-1, 3)
        
            # Sample one maneuver for each of the B*A*I trajectories
            sampled_indices = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
        
            # --- 3. Create One-Hot Weights from the Samples ---
            # These are the "hard" weights for generating distinct futures
            guidance_weights_one_hot = F.one_hot(
                sampled_indices, num_classes=self.num_energy_terms
            ).float()
        
            # Reshape back to the required [B, A, I, 3] format
            guidance_weights = guidance_weights_one_hot.view(
                B, N, I, self.num_energy_terms
            )

            physical_trajectories_flat = emission_head(z)
            agent_positions_flat = physical_trajectories_flat[..., 0:2]
            num_agents = N
        with torch.enable_grad():
            z.requires_grad_(True)
            #agent_positions_flat.requires_grad_(True)
            #guidance_weights.requires_grad_(True)
            
            #s_realism.requires_grad_(True)
            initial_headings = agent_positions_flat[:, 1, :] - agent_positions_flat[:, 0, :]
            #initial_headings.requires_grad_(True)
            E_guidance = self.guidance_module(
                agent_positions_flat=agent_positions_flat,
                guidance_weights=guidance_weights,
                initial_headings_flat=initial_headings,
                map_points=map_point,
                num_agents=num_agents,
                num_intentions=I,
                num_scenes=B
            )
            #E_guidance.requires_grad_(True)
            s_guidance = -torch.autograd.grad(E_guidance, z, retain_graph=True)[0]
        #s_realism_norm = s_realism.flatten(start_dim=1).view_as(s_realism)
        #s_guidance_norm = s_guidance.flatten(start_dim=1).view_as(s_guidance)
        s_realism_raw_norm = torch.linalg.norm(s_realism.flatten(1), dim=-1).mean()
        s_guidance_raw_norm = torch.linalg.norm(s_guidance.flatten(1), dim=-1).mean()

        # Log these values (using your logger)
        #print(f"DEBUG: Raw realism score norm: {s_realism_raw_norm.item()}")
        #print(f"DEBUG: Raw guidance score norm: {s_guidance_raw_norm.item()}")

        # Then, normalize as before
        s_realism_norm = F.normalize(s_realism.flatten(start_dim=1)).view_as(s_realism)
        s_guidance_norm = F.normalize(s_guidance.flatten(start_dim=1)).view_as(s_guidance)
        
        s_final = s_realism #+ 0.1 * s_guidance_norm

        #s_final = s_realism_norm
        #s_final = s_guidance_norm
        print(" AT t = ", t, " Guidance score: ", s_final.norm().item())
        #sim = F.cosine_similarity(s_realism_norm.flatten(), s_guidance_norm.flatten(), dim=0)
        #print(f" AT t = {t.item():.4f}  Cosine Similarity: {sim.item():.4f}")



        ## Add the main guidance calls here
        ## This includes - learning the guidance weights. this function is called inside torch.no_grad() which means, we need to figure out how to actually learn the weights of the gating network
        ## Context is actually the mean of map encoding, so it might have limited information. We need to figure out truly what is the input to Gating network, is it LSTM mean? (Which is what I'm leaning towards as prior data)
        ## Once we have guidance weights and agent_positions_flat, we need to actually compute PDE based energy function and return the final drift
        #s_final = s_realism * w_collision
        #print('s_final norm: ', s_final.norm().item())
        beta_t = self.beta(t)
        f_forward_drift = -0.5 * beta_t.view(-1, 1, 1) * z 
        drift = f_forward_drift - beta_t.view(-1, 1, 1) * s_final
        drift = drift.view(-1, drift.shape[-1]) * -1  # Flip the sign for reverse SDE
        return drift
        
        

    def g(self, t, z_flat):
        """ The diffusion for both forward and reverse SDEs. Signature is g(t, y). """
        beta_t = self.beta(t)
        # Diffusion is a scalar multiplied by a tensor of the correct shape
        diffusion_val = torch.sqrt(beta_t)
        diffusion_val_reshaped = diffusion_val.view(-1, 1)
        return diffusion_val_reshaped.expand_as(z_flat)
    
class SDEWrapper(nn.Module):
    def __init__(self, sde_model, context, score_network, gating_network, cur_num_agents, sde_t, lstm_prior,
                         emission_head, assigned_types, map_point, w_collision, w_layout,is_forward=False,is_guided=False, batch_dims=None):
        super().__init__()
        self.sde_model = sde_model
        self.context = context
        self.score_network = score_network
        self.gating_network = gating_network
        self.is_forward = is_forward
        self.emission_head = emission_head
        self.assigned_types = assigned_types
        self.map_point = map_point
        self.w_collision = w_collision
        self.w_layout = w_layout
        self.cur_num_agents = cur_num_agents
        self.is_guided = is_guided
        self.sde_t = sde_t
        # Copy required attributes for the torchsde solver
        self.noise_type = sde_model.noise_type
        self.sde_type = sde_model.sde_type
        self.lstm_prior = lstm_prior
        self.T_max = 1.0
        self.batch_dims = batch_dims

    def f(self, t, y):
        if self.is_forward:
            return self.sde_model.f_forward(t, y)
        else:
            if not self.is_guided:
                return self.sde_model.f_reverse(t, y, self.score_network, self.context)
            else:
                t = self.T_max - t
                #print("Using guided reverse SDE, t=", t)
                return self.sde_model.f_reverse_guided(t, y, self.score_network, self.context,self.cur_num_agents, self.gating_network,
                            self.emission_head, self.assigned_types, self.map_point, self.w_collision, self.w_layout, sde_t=self.sde_t, lstm_prior=self.lstm_prior.detach(), batch_dims=self.batch_dims)

    def g(self, t, y):
        # The g function in our VpSde does not need context, so we don't pass it.
        return self.sde_model.g(t, y)*0.0 # No noise for DDIM-like sampling


class PolicyRefinementModule(nn.Module):
    def __init__(self, 
                 hidden_size: int,
                 num_agents: int = 288,
                 beta_min: float = 0.1,
                 beta_max: float = 20.0,
                 sde_T_max: float = 1,
                 sde_eps: float = 1e-2,
                 ):
        super().__init__()
        self.sde = VpSde(
            latent_dim=hidden_size,
            num_agents=num_agents, # The max number of agents from your padded data
            beta_min=beta_min,
            beta_max=beta_max # Use the recommended stable values
        )
        tcn_channels = [hidden_size, 512, 512, hidden_size]
        config = {
            'latent_dim': 128,      # Should match your model's hidden_size
            'seq_len': 10,           # The length of your trajectory sequences (e.g., your 'fixed T')
            'nf': 128,               # Base number of features
            'ch_mult': (1, 2, 2, 2), # Channel multiplier for each resolution
            'num_res_blocks': 3,     # Number of res-blocks per resolution
            'attn_resolutions': (16,), # Sequence lengths at which to apply attention (e.g., 10//(2**2) = 2)
            'dropout': 0.1,
            'resamp_with_conv': True
        }
        self.score_network = NCSNpp(config)
        #self.score_network = TransformerScoreNetwork(latent_dim=hidden_size, context_dim=hidden_size)
        self.gating_network = GatingNetwork(context_dim=hidden_size)
        # --- 3. DECODERS & GUIDANCE MODELS (INSTANTIATED) ---
        self.decoder_log_std = nn.Parameter(torch.zeros(2)) # For (x, y) or (vx, vy)
        self.sde_T_max = sde_T_max
        self.sde_eps = sde_eps
    
    # In your SeNeVAMLightningModule class
    def sample_with_guidance(self, map_point, x_map, map_context, emission_head, num_agents_to_gen: int, assigned_types: torch.Tensor, noise_latent:torch.Tensor=None,
                            w_collision: float = 0.001, w_layout: float = 0.001, steps: int = 500, t: torch.Tensor = None, lstm_prior: torch.Tensor = None, batch_dims=None):
        """
        Generates a new scene from random noise, guided by physical constraints.
        """
        B = map_point.x.shape[0]
        
        # --- 1. SETUP ---
        # Encode the map to get the conditioning context
        # Note: A helper function might be needed if your encoder expects a full trajectory object
        context = torch.mean(x_map, dim=1)
        
        # Initialize the state with pure random noise
        if noise_latent is not None:
            z_t = noise_latent
        else:
            z_t = torch.randn(B, num_agents_to_gen, self.hparams.hidden_size, device=self.device)

        #print("NOISE SHAPE: ", z_t.shape)
        
        # Set up the reverse timesteps for the solver

        # --- 2. THE SAMPLING LOOP (The "Manager") ---
        assigned_types = assigned_types.unsqueeze(-1).expand(-1, z_t.shape[1]).unsqueeze(-1)
        map_context = map_context.unsqueeze(1).expand(-1, z_t.shape[1], -1)
        

        z_t = torch.cat((assigned_types, map_context, z_t), dim=-1)
        z_t = z_t.reshape(z_t.shape[0]*z_t.shape[1], -1)

        assigned_types = z_t[:, 0]
        map_context = z_t[:, 1:129]
        z_t = z_t[:, 129:]
        z_t.requires_grad_(True)
        #t = t[0].item()
        timesteps = torch.linspace(0, t[0].item(), steps, device=z_t.device)
        emission_head.eval()
        self.score_network.eval()
        sde_rev_wrapper = SDEWrapper(sde_model=self.sde, context=map_context, cur_num_agents=num_agents_to_gen, score_network=self.score_network, emission_head = emission_head, sde_t = t,
                            assigned_types=assigned_types, map_point=map_point, w_collision=w_collision, w_layout=w_layout, gating_network=self.gating_network, lstm_prior=lstm_prior,
                            is_forward=False,is_guided=True, batch_dims=batch_dims)
        x_traj_gen = torchsde.sdeint(sde_rev_wrapper, z_t, timesteps, dt=0.01, method='euler')[-1]
        emission_head.train()
        self.score_network.train()
        #print("AFTER SDE",  x_traj_gen.shape)
        #exit()
        return x_traj_gen


class PolicyAttentionBlock(nn.Module):
    """Mutual attention and cross-attention block for policy network."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        init_scale: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.init_scale = init_scale
        self.num_heads = num_heads
        if self.num_heads is None:
            self.num_heads = hidden_size // 64

        # build the attention and feed-forward layers
        self.sa_norm = layer_norm(normalized_shape=hidden_size)
        self.sa_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            dropout=dropout,
            num_heads=self.num_heads,
            batch_first=True, 
        )
        self.sa_dropout = nn.Dropout(p=self.dropout)

        self.ca_q_norm = layer_norm(normalized_shape=hidden_size)
        self.ca_kv_norm = layer_norm(normalized_shape=hidden_size)
        self.ca_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            dropout=dropout,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.ca_dropout = nn.Dropout(p=self.dropout)

        self.ffn_norm = layer_norm(normalized_shape=hidden_size)
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

        self.reset_parameters()

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        sa_attn_mask: Optional[torch.Tensor] = None,
        sa_key_padding_mask: Optional[torch.Tensor] = None,
        ca_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the attention block.

        Args:
            query (torch.Tensor): Query feature tensor of shape `(*, E)`.
            key_value (torch.Tensor): Key-value tensor of shape `(*, K, E)`.
            sa_attn_mask (Optional[torch.Tensor], optional): Optional mask for
                self-attention weights. Defaults to ``None``.
            sa_key_padding_mask (Optional[torch.Tensor], optional): Optional
                valid mask for self-attention keys. Defaults to ``None``.
            ca_key_padding_mask (Optional[torch.Tensor], optional): Optional
                valid mask for cross-attention keys. Defaults to ``None``.

        Returns:
            torch.Tensor: The output feature tensor of shape
        """

        out = query + self._sa_forward(
            x=query,
            attn_mask=sa_attn_mask,
            key_padding_mask=sa_key_padding_mask,
        )  # shape: ``(*, E)``
        #assert torch.isnan(sa_attn_mask).sum() == 0
        assert torch.isnan(sa_key_padding_mask).sum() == 0
        assert torch.isnan(out).sum() == 0
        
        out = out.unsqueeze(-2)  # shape: ``(*, 1, E)``
        out = out + self._ca_forward(
            query=out,
            key_value=key_value,
            key_padding_mask=ca_key_padding_mask,
        )
        assert torch.isnan(ca_key_padding_mask).sum() == 0
        assert torch.isnan(key_value).sum() == 0
        assert torch.isnan(out).sum() == 0
        out = out.squeeze(-2)  # shape: ``(*, E)``
        out = out + self._ffn_forward(x=out)
        assert torch.isnan(out).sum() == 0

        return out

    def reset_parameters(self) -> None:
        """Reset the parameters of the attention block."""
        for module in (self.sa_attn, self.ca_attn):
            if module.in_proj_weight is not None:
                variance_scaling(module.in_proj_weight, scale=self.init_scale)
            else:
                variance_scaling(module.q_proj_weight, scale=self.init_scale)
                variance_scaling(module.k_proj_weight, scale=self.init_scale)
                variance_scaling(module.v_proj_weight, scale=self.init_scale)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            variance_scaling(module.out_proj.weight)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)
            if module.bias_k is not None:
                nn.init.zeros_(module.bias_k)
            if module.bias_v is not None:
                nn.init.zeros_(module.bias_v)

    def _sa_forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the self-attention block."""
        out = self.sa_norm.forward(x)
        out, _ = self.sa_attn.forward(
            query=out,
            key=out,
            value=out,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        return self.sa_dropout.forward(out)

    def _ca_forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the cross-attention block."""
        query = self.ca_q_norm(query)
        assert torch.isnan(query).sum() == 0
        key_value = self.ca_kv_norm(key_value)
        assert torch.isnan(key_value).sum() == 0
        out, _ = self.ca_attn.forward(
            query=query,
            key=key_value,
            value=key_value,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        assert torch.isnan(key_padding_mask).sum() == 0
        # print("Query", query.shape, key_value.shape, out.shape)  # --- IGNORE ---
        # print("Query max/min:", query.max(), query.min())  # --- IGNORE ---
        # print("Key Value max/min:", key_value.max(), key_value.min())  # --- IGNORE ---
        # print("Key Padding Mask sum:", key_padding_mask.sum())  # --- IGNORE ---
        assert torch.isnan(out).sum() == 0
        return self.ca_dropout.forward(out)

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn.forward(self.ffn_norm.forward(x))





class PolicyNetwork(nn.Module):
    """Policy network for the agents."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_blocks: int = 1,
        num_heads: Optional[int] = None,
        num_intentions: int = 6,
        num_neighbors: int = 768,
        init_scale: float = 0.2,
        output_size: int = 5,
        
    ) -> None:
        super().__init__()
        # save the parameters
        self.hidden_size = hidden_size
        self.num_intentions = num_intentions
        self.num_neighbors = num_neighbors
        self.output_size = output_size
        self.policy_refinement_module = PolicyRefinementModule(hidden_size = hidden_size)
        self.sigma_data = 0.5
        

        # create the intention embedding layer
        self.intention = embedding(
            num_embeddings=num_intentions,
            embedding_dim=hidden_size,
            init_scale=1.0,
        )

        # create the relative space-time positional encoding
        self.sa_rst_encoder = RSTEncoder(hidden_size=hidden_size)
        self.ca_rst_encoder = RSTEncoder(hidden_size=hidden_size)

        # create the attention blocks
        self.blocks = nn.ModuleDict(
            {
                f"block_{i}": PolicyAttentionBlock(
                    hidden_size=hidden_size,
                    dropout=dropout,
                    num_heads=num_heads,
                    init_scale=init_scale,
                )
                for i in range(num_blocks)
            }
        )

        # create the latent distribution heads
        self.generative_lstm = lstm(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.generative_state_mean = linear(
            in_features=hidden_size,
            out_features=hidden_size,
            init_scale=0.01**2,
        )
        self.generative_state_var = linear(
            in_features=hidden_size,
            out_features=hidden_size,
            init_scale=0.01**2,
        )

        # create the z-proxy network
        self.z_proxy = linear(
            in_features=hidden_size,
            out_features=1,
            init_scale=0.01**2,
        )

        self.reset_parameters()

    def forward(
        self,
        inputs: torch.Tensor,
        tar_xy: torch.Tensor,
        tar_yaw: torch.Tensor,
        tar_valid: torch.Tensor,
        context: torch.Tensor,
        ctx_xy: torch.Tensor,
        ctx_yaw: torch.Tensor,
        ctx_valid: torch.Tensor,
        emission_head: nn.Module,
        num_agents: int,
        scenario, 
        assigned_types: torch.Tensor,
        global_step: int,
        horizon: int = 1,

        
    ) -> tensor_3_t:
        """Forward pass of the policy network.

        Args:
            inputs (torch.Tensor): Input features representing current states
                of the agents to predict with a shape of ``(*, L, Eq)``.
            tar_xy (torch.Tensor): Current x-y coordinates of the target agent
                with a shape of ``(*, L, 2)``.
            tar_yaw (torch.Tensor): Current heading angle of the target agent
                with a shape of ``(*, L, 1)``.
            tar_valid (torch.Tensor): Validity mask of the target agent with a
                shape of ``(*, L)``.
            context (torch.Tensor): Context features representing the states
                of the other agents with a shape of ``(*, S, Ek)``.
            ctx_xy (torch.Tensor): x-y coordinates of the other agents with a
                shape of ``(*, S, 2)``.
            ctx_yaw (torch.Tensor): Heading angles of the other agents with a
                shape of ``(*, S, 1)``.
            ctx_valid (torch.Tensor): Validity mask of the other agents with a
                shape of ``(*, S)``.
            horizon (int, optional): The prediction horizon.
                Defaults to :math:`1`.

        Returns:
            tensor_3_t: A tuple of two tensors representing the mean and
                variance of the predicted state distributions and a tensor
                representing the z-proxy logits.
        """
        k, l, eq = self.num_intentions, inputs.size(-2), inputs.size(-1)
        s, ek = context.size(-2), context.size(-1)

        # compute the mutual-attention positional encoding
        intentions = self.intention.weight  # shape: ``(K, Eq)``
        intentions = sinusoidal_positional_encoding(intentions)
        assigned_types = assigned_types.unsqueeze(-1)
        map_context = torch.mean(context, dim=1).unsqueeze(1).expand(-1, assigned_types.shape[1], -1)
        #print("MAP CONTEXT SHAPE: ", map_context.shape)
        #print("ASSIGNED SHAPES: ", assigned_types.shape)
        #print("INPUTS SHAPE: ", inputs.shape)
        try:
            input_aug = torch.cat((assigned_types, map_context, inputs), dim=-1)
        except:
            print("SHAPE ERROR IN INPUT AUG CONCAT")
            exit()
        #print("AUG INPUT SHAPE: ", input_aug.shape)
        #inputs = inputs.unsqueeze(-2).expand(*inputs.shape[:-1], k, eq)
        
        input_aug = input_aug.unsqueeze(-2).expand(*input_aug.shape[:-1], k, 2*eq+1)
        assigned_types = input_aug[:, :, :, 0]
        map_context = input_aug[:, :, :, 1:129]
        inputs = input_aug[:, :, :, 129:]
        #print("EXPANDED INPUT SHAPES: ", inputs.shape)
        inputs = torch.add(
            inputs, torch.broadcast_to(input=intentions, size=inputs.shape)
        )
        #input_aug = torch.add(input_aug, torch.broadcast_to(input=intentions, size=input_aug.shape))
        #print("ADDED INPUT SHAPES: ", inputs.shape)
        #print("Mod Input Aug shape: ", assigned_types.shape)
        #print(assigned_types)
        

        tar_xy = tar_xy.unsqueeze(-2).expand(*tar_xy.shape[:-1], k, 2)
        tar_yaw = tar_yaw.unsqueeze(-2).expand(*tar_yaw.shape[:-1], k, 1)
        tar_t = torch.arange(1, k + 1, device=tar_yaw.device).unsqueeze(-1)
        tar_t = torch.broadcast_to(input=tar_t, size=tar_yaw.shape)
        sa_key_padding_mask = torch.logical_not(tar_valid)
        sa_key_padding_mask = torch.unsqueeze(
            sa_key_padding_mask, dim=-1
        ).expand(*tar_valid.shape, k)

        # flatten the intention dimension
        inputs = torch.cat((assigned_types.unsqueeze(-1), map_context, inputs), dim=-1)
        inputs = inputs.reshape(*inputs.shape[:-3], l * k, 2*eq+1)
        assigned_types = inputs[:, :, 0]
        map_context = inputs[:, :, 1:129]
        inputs = inputs[:, :, 129:]
        tar_xy = tar_xy.reshape(*tar_xy.shape[:-3], l * k, 2)
        tar_yaw = tar_yaw.reshape(*tar_yaw.shape[:-3], l * k, 1)
        tar_t = tar_t.reshape(*tar_t.shape[:-3], l * k, 1)
        sa_key_padding_mask = torch.reshape(
            input=sa_key_padding_mask,
            shape=(*sa_key_padding_mask.shape[:-2], l * k),
        )

        # compute and apply the relative space-time positional encoding
        sa_pos_enc = self.sa_rst_encoder.forward(
            input_xy=tar_xy,
            input_yaw=tar_yaw,
            other_xy=torch.zeros_like(tar_xy),
            other_yaw=torch.zeros_like(tar_yaw),
            input_t=tar_t,
            other_t=torch.zeros_like(tar_t),
        )
        assert torch.isnan(sa_pos_enc).sum() == 0
        assert torch.isnan(inputs).sum() == 0
        query = inputs + sa_pos_enc

        # compute the top-k neighbors for each query point
        indices: torch.Tensor = get_topk_neighbors(
            target_xy=tar_xy,
            neighbor_xy=ctx_xy,
            neighbor_valid=ctx_valid,
            num_neighbors=self.num_neighbors,
        )  # shape: ``(*, K * L, N)``
        ca_pos_enc = self.ca_rst_encoder.forward(
            input_xy=torch.broadcast_to(
                input=tar_xy.unsqueeze(-2), size=indices.shape + (2,)
            ),
            input_yaw=torch.broadcast_to(
                input=tar_yaw.unsqueeze(-2), size=indices.shape + (1,)
            ),
            other_xy=torch.gather(
                input=ctx_xy.unsqueeze(-3).expand(*indices.shape[:-1], s, 2),
                index=indices.unsqueeze(-1).expand(*indices.shape, 2),
                dim=-2,
            ),
            other_yaw=torch.gather(
                input=ctx_yaw.unsqueeze(-3).expand(*indices.shape[:-1], s, 1),
                index=indices.unsqueeze(-1).expand(*indices.shape, 1),
                dim=-2,
            ),
        )  # shape: ``(*, L, N, E)``
        ca_kv = torch.gather(
            input=context.unsqueeze(-3).expand(*indices.shape[:-1], s, ek),
            index=indices.unsqueeze(-1).expand(*indices.shape, ek),
            dim=-2,
        )  # shape: ``(*, L, N, E)``

        # forward pass the cross-attention blocks
        ca_kv = ca_kv + ca_pos_enc
        ca_key_padding_mask = torch.gather(
            input=torch.logical_not(ctx_valid)
            .unsqueeze(-2)
            .expand(*indices.shape[:-1], s),
            index=indices,
            dim=-1,
        )

        batch_dims = query.shape[:-2] + (l, k)
        
        query = torch.cat((assigned_types.unsqueeze(-1), map_context, query), dim=-1)
        query = query.reshape(-1, query.shape[-1])
        assigned_types = query[:, 0]
        map_context = query[:, 1:129]
        query = query[:, 129:]
        sa_key_padding_mask = sa_key_padding_mask.reshape(-1)
        ca_kv = ca_kv.reshape(-1, *ca_kv.shape[-2:])
        ca_key_padding_mask = ca_key_padding_mask.reshape(
            -1, ca_key_padding_mask.shape[-1]
        )
        for _, blocks in self.blocks.items():
            assert isinstance(blocks, PolicyAttentionBlock)
            assert torch.isnan(query).sum() == 0
            query = blocks.forward(
                query=query,
                key_value=ca_kv,
                sa_key_padding_mask=sa_key_padding_mask,
                ca_key_padding_mask=ca_key_padding_mask,
            )
            assert torch.isnan(query).sum() == 0

        # forward pass the distribution heads
        assert torch.isnan(query).sum() == 0
        assert horizon > 0
        #print("QUERY RESHAPED: ", query.shape)
        s_mean, s_var, sde_loss, score_loss, sde_metrics = self._gen_forward(x=query, horizon=horizon, emission_head=emission_head, context=context, map_context=map_context,
                                          tar_valid=tar_valid, assigned_types=assigned_types, scenario=scenario, global_step = global_step, batch_dims=batch_dims)
        s_mean = s_mean.reshape(*batch_dims, horizon, self.hidden_size)
        s_var = s_var.reshape(*batch_dims, horizon, self.hidden_size)
        #print("S MEAN RESHAPED: ", s_mean.shape)


        # forward pass the z-proxy network
        z_logits = self.z_proxy.forward(query)
        z_logits = z_logits.reshape(*batch_dims)

        return s_mean, s_var, z_logits, sde_loss, score_loss, sde_metrics

    def reset_parameters(self) -> None:
        """Reset the parameters of the policy network."""
        nn.init.constant_(self.generative_state_mean.weight, val=0.0)
        if self.generative_state_mean.bias is not None:
            nn.init.constant_(self.generative_state_mean.bias, val=0.0)
        nn.init.constant_(self.generative_state_var.weight, val=0.0)
        if self.generative_state_var.bias is not None:
            nn.init.constant_(self.generative_state_var.bias, val=0.0)

    def _gen_forward(self, x: torch.Tensor, horizon: int, emission_head: nn.Module, context: torch.Tensor, map_context: torch.Tensor, tar_valid: torch.Tensor, 
                     assigned_types: torch.Tensor, scenario, global_step: int, refinement_steps: int = 20, batch_dims = None) -> tensor_2_t:
        """Forward pass the generative LSTM network.

        Args:
            x (torch.Tensor): Input feature tensor of shape ``(*, L, K, E)``.
            horizon (int): The prediction horizon.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple of two tensors the mean
                and variance of the predicted state distributions
        """
        s_mean = torch.zeros(
            [*x.shape[:-1], horizon, self.hidden_size],
            device=x.device,
            dtype=x.dtype,
        )
        s_vars = torch.zeros(
            [*x.shape[:-1], horizon, self.hidden_size],
            device=x.device,
            dtype=x.dtype,
        )

        s_mean[..., 0, :] = self.generative_state_mean.forward(x)
        assert torch.isnan(s_mean).sum() == 0
        s_vars[..., 0, :] = nn.functional.softplus(
            self.generative_state_var.forward(x)
        )
        hx = (x.tanh().unsqueeze(0), torch.zeros_like(x).unsqueeze(0))
        score_loss_aggregate = None
        sde_loss_aggregate = None
        sde_to_noisy_aggregate = None
        noisy_to_gt_aggregate = None
        mseloss = torch.nn.MSELoss()
        start = time.time()
        counter = 0
        for t in range(1, horizon):
            #print("INSIDE FOR LOOP t: ", t)
            seq = s_mean[..., 0:t, :]
            inp = torch.cat(
                (seq, x.unsqueeze(-2).broadcast_to(seq.shape)), dim=-1
            ).contiguous()

            #print("INPUT SHAPE: ", inp.shape)
            out, hx = self.generative_lstm.forward(inp, hx=hx)
            out = out[..., -1, :]
            s_mean[..., t, :] = self.generative_state_mean.forward(out)
            
            if (t % refinement_steps == 0):
                counter += 1
                # Refine predicted trajectory latents using energy based models
                policy = s_mean[..., t-10:t, :].clone()
                #print("POLICY SHAPE: ", policy.shape)
                
                val = torch.rand(1).item()
                # DURING TRAINING
                #sde_t = torch.ones(policy.shape[0], device=policy.device) * val * (self.policy_refinement_module.sde_T_max - self.policy_refinement_module.sde_eps) + self.policy_refinement_module.sde_eps ## Fix the denoising range between 0-1
                #DURING INFERENCE
                # sde_t = torch.ones(policy.shape[0], device=policy.device)
                # mean, std = self.policy_refinement_module.sde.marginal_prob(policy, sde_t)
                # noise = torch.randn_like(policy)
                # z_t = mean + std * noise

                # #print("AT SCORE NETWORK: ", z_t.shape)
                # #print("Score Network context: ", torch.mean(context, dim=1).shape)
                # predicted_score = self.policy_refinement_module.score_network(z_t, sde_t, context_token=map_context, key_padding_mask=~tar_valid)

                # score_losses = predicted_score * std + noise
                # score_losses = score_losses ** 2
                # score_losses = score_losses.reshape(score_losses.shape[0], -1).mean(dim=-1).mean()
                
                P_mean = -1.2
                P_std = 1.2
                sigma_start = torch.rand(1).to(policy.device)
                sigma = torch.exp(sigma_start * P_std + P_mean).to(policy.device)
                sigma_broadcast = sigma.view(-1, *([1] * (policy.dim() - 1)))
                noise = torch.randn_like(policy)
                z_t = policy + sigma_broadcast * noise
                predicted_policy = self.policy_refinement_module.score_network(z_t, sigma, context_token=map_context, key_padding_mask=~tar_valid)
                weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data)**2
                
                score_losses = weight * ((predicted_policy - policy) ** 2)
                
                
                if score_loss_aggregate is None:
                    score_loss_aggregate = score_losses
                    
                else:
                    score_loss_aggregate += score_losses
                # num_agents_in_scene = tar_valid.shape[1] # This is the N for this batch
                # z_sde = self.policy_refinement_module.sample_with_guidance(map_point=scenario.map_point, x_map = context, map_context=map_context, num_agents_to_gen=3, 
                #                                                               assigned_types=assigned_types, w_collision=1, w_layout=1, steps=100, noise_latent=z_t, t=sde_t,
                #                                                               emission_head = emission_head, lstm_prior = policy.detach(), batch_dims=batch_dims)
                # z_sde = z_sde.reshape(-1, 10, z_sde.shape[-1])
                z_sde = refine_with_edm_guidance(z_t, sigma_start, self.policy_refinement_module.score_network, map_context, tar_valid)
                if sde_loss_aggregate is None:
                    sde_loss_aggregate = torch.nn.functional.mse_loss(z_sde, policy).mean()
                    sde_to_noisy_aggregate = torch.nn.functional.cosine_similarity(z_sde, z_t, dim=-1).mean()
                    noisy_to_gt_aggregate = torch.nn.functional.cosine_similarity(z_t, policy, dim=-1).mean()
                else:
                    sde_loss_aggregate += torch.nn.functional.mse_loss(z_sde, policy).mean()
                    sde_to_noisy_aggregate += torch.nn.functional.cosine_similarity(z_sde, z_t, dim=-1).mean()
                    noisy_to_gt_aggregate += torch.nn.functional.cosine_similarity(z_t, policy, dim=-1).mean()

                # zero_mse = (policy**2).mean().item()
                # pol_var = policy.var().item()
                # noisy_input_mse = ((z_sde - z_t)**2).mean().item()
                # gt_score = -noise / (std + 1e-12)
                # pred = predicted_score.detach().reshape(-1)
                # corr = torch.nn.functional.cosine_similarity(pred, gt_score.reshape(-1), dim=0)
                # input_output_error = F.mse_loss(z_sde, policy)
                #print(f"MSE between SDE output and LSTM output: {input_output_error.item()}")
                #print(f"MSE Between SDE output and noisy input: {mseloss(z_sde, z_t).item()}")
                #print(f"MSE Between noisy input and LSTM output: {mseloss(z_t, policy).item()}")

                sde_metrics = {"sde_to_lstm_mse": sde_loss_aggregate.item()/counter, "sde_to_noisy_mse": sde_to_noisy_aggregate.item()/counter, "noisy_to_lstm_mse": noisy_to_gt_aggregate.item()/counter}

                #print("ZERO MSE: ", zero_mse, " Cosine Similarity: ", corr)
                #print("predicted_score mean, std:", predicted_score.mean().item(), predicted_score.std().item())
                #print("STD : ", std.mean().item(), " MEAN: ", mean.mean().item())
                #print("gt_score mean, std:", gt_score.mean().item(), gt_score.std().item())
                
                #print("POLICY SHAPE: ", policy.shape)
                #print("Z SDE SHAPE: ", z_sde.shape)
                #if(global_step > 12000):
                s_mean[..., t-10:t, :] = z_sde


            #print("Mean Shape: ", mean.shape, " STD SHAPE: ", std.shape, "Predicted Score Shape: ", predicted_score.shape, " Score Loss: ", score_losses.mean())
            #print("OUTPUT SHAPE: ", out.shape)
            
            #print("S MEAN SHAPE: ", s_mean.shape)

            assert torch.isnan(s_mean).sum() == 0
            s_vars[..., t, :] = nn.functional.softplus(
                self.generative_state_var.forward(out)
            )
        return s_mean, s_vars, sde_loss_aggregate/counter, score_loss_aggregate/counter, sde_metrics
