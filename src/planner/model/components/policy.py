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

                z_sde = refine_with_edm_guidance(z_t, sigma_start, self.policy_refinement_module.score_network, map_context, tar_valid)
                if sde_loss_aggregate is None:
                    sde_loss_aggregate = torch.nn.functional.mse_loss(z_sde, policy).mean()
                    sde_to_noisy_aggregate = torch.nn.functional.cosine_similarity(z_sde, z_t, dim=-1).mean()
                    noisy_to_gt_aggregate = torch.nn.functional.cosine_similarity(z_t, policy, dim=-1).mean()
                else:
                    sde_loss_aggregate += torch.nn.functional.mse_loss(z_sde, policy).mean()
                    sde_to_noisy_aggregate += torch.nn.functional.cosine_similarity(z_sde, z_t, dim=-1).mean()
                    noisy_to_gt_aggregate += torch.nn.functional.cosine_similarity(z_t, policy, dim=-1).mean()


                sde_metrics = {"sde_to_lstm_mse": sde_loss_aggregate.item()/counter, "sde_to_noisy_cosine": sde_to_noisy_aggregate.item()/counter, "noisy_to_lstm_cosine": noisy_to_gt_aggregate.item()/counter}

                
                s_mean[..., t-10:t, :] = z_sde # COMMENT DURING TRAINING

            assert torch.isnan(s_mean).sum() == 0
            s_vars[..., t, :] = nn.functional.softplus(
                self.generative_state_var.forward(out)
            )
        return s_mean, s_vars, sde_loss_aggregate/counter, score_loss_aggregate/counter, sde_metrics
