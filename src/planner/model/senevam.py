# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Multi-agent probabilistic trajectory forecasting."""
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union


import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import (
    LRScheduler,
    
)
import pytorch_lightning as pl
from torchmetrics import MeanMetric, MetricCollection
import torchsde
import random
from planner.model.module import embedding, layer_norm, linear
from planner.data.dataclass import Scenario, MapPolylineType, ObjectType
from planner.model.components.encoder import HistoryEncoder
from planner.model.components.policy import PolicyNetwork
from planner.model.function.eval import MinADE, MinFDE
from planner.model.function.geometry import wrap_angles
from planner.model.function.mask import extract_target
from planner.model.module.layers import linear
from planner.utils.logging import get_logger
from planner.model.components.constraints_loss import DifferentiableConstraintLosses

from planner.model.components.constraint_models import (
    ConstraintPosteriorNetwork,
    ConstraintPriorNetwork,
    PotentialFieldNetwork
)

# Assume this is the file you've copied from the CCDiff repository
# We will import the specific loss classes we want to adapt
from your_project.third_party.ccdiff import AgentCollisionLoss, TargetSpeedLoss

# Assume this is the adapter function we designed previously
from your_project.utils.adapters import create_ccdiff_databatch

import torch.nn.functional as F

# Constants
TIKHONOV_REGULARIZATION = 0.5 / math.pi
LOGGER = get_logger(__name__)



# =============================================================================
# Helper functions
# =============================================================================
def get_target_mask(scenario: Scenario, include_sdc: bool) -> torch.Tensor:
    """Get the boolean mask indicating if an object is a target.

    Args:
        scenario (Scenario): Scenario dataclass object.
        include_sdc (bool): Whether to include the self-driving car as target.

    Returns:
        torch.Tensor: Boolean mask indicating if an object is a target.
    """
    if include_sdc:
        return scenario.object_property.is_target
    else:
        return torch.logical_and(
            scenario.object_property.is_target,
            torch.logical_not(scenario.object_property.is_sdc),
        )


@dataclass
class SeNeVAMOutput:
    """Container of output tensors from `SeNeVA-M` model."""

    y_means: torch.Tensor
    """torch.Tensor: Mean values of the marginal distributions."""
    y_covars: torch.Tensor
    """torch.Tensor: Covariance matrices of the marginal distributions."""
    s_means: torch.Tensor
    """torch.Tensor: Mean values of the latent states."""
    s_vars: torch.Tensor
    """torch.Tensor: Variance values of the latent states."""
    z_logits: torch.Tensor
    """torch.Tensor: Unnormalized mixture weights of the GMM."""
    current_state: torch.Tensor
    """torch.Tensor: Current state of the predicted agents."""
    current_valid: torch.Tensor
    """torch.Tensor: Valid mask for the current state."""


@dataclass
class SeNeVAMOptimizerConfig:
    """Configuration dataclass for the optimizer used in `SeNeVA-M` model."""

    # configurations for AdamW optimizer
    lr: float = 1e-3
    """float: Learning rate for the optimizer."""
    betas: Tuple[float, float] = (0.9, 0.95)
    """Tuple[float, float]: Coefficients for averaged gradients and squares."""
    eps: float = 1e-8
    """float: Epsilon value for numerical stability."""
    maximize: bool = False
    """bool: Whether to maximize the objective function."""
    weight_decay: float = 1e-2
    """float: Weight decay for the optimizer."""

    # configurations for the learning rate scheduler
    warmup_steps: int = 1000
    """int: Number of warmup steps for the learning rate scheduler."""
    total_steps: int = 100000
    """int: Total number of steps for the learning rate scheduler."""
    min_lr: float = 1e-6
    """float: Minimum learning rate for the learning rate scheduler."""


class SeNeVAMLRScheduler(LRScheduler):
    """Learning rate scheduler for `SeNeVA-M` model.

    Attributes:
        optimizer (torch.optim.Optimizer): Optimizer to be scheduled.
        min_lr (float): Minimum learning rate.
        warmup_steps (int): Number of linear warmup steps.
        total_steps (int): Total number of learning rate scheduling steps.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(
            optimizer=optimizer,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def get_lr(self) -> List[float]:
        #_warn_get_lr_called_within_step(self)
        if self._step_count <= self.warmup_steps:
            # NOTE: linear warmup from zero to base learning rate
            lr_scale = self._step_count / self.warmup_steps
            return [base_lr * lr_scale for base_lr in self.base_lrs]
        elif self._step_count <= self.total_steps:
            # NOTE: cosine annealing from base to minimum lr
            lr_scale = 0.5 * (
                1.0
                + math.cos(
                    (self._step_count - self.warmup_steps)
                    / (self.total_steps - self.warmup_steps)
                    * math.pi
                )
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * lr_scale
                for base_lr in self.base_lrs
            ]
        else:
            return [self.min_lr for _ in self.base_lrs]


class SeNeVAMLightningModule(LightningModule):
    """Implementation of `SeNeVA-M` model as a PyTorch Lightning module."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_blocks: int = 1,
        num_intentions: int = 6,
        num_heads: Optional[int] = None,
        num_neighbors: int = 768,
        init_scale: float = 0.2,
        alpha: float = 2.5,
        gamma: float = 2.0,
        optimizer_config: Mapping[str, Any] = {},  # noqa: B006
        sde_weight: float = 0.1,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # set optimizer configuration
        self.save_hyperparameters()
        

        # create network
        self.encoder = HistoryEncoder(
            hidden_size=hidden_size,
            dropout=dropout,
            num_blocks=num_blocks,
            num_heads=num_heads,
            init_scale=init_scale,
        )
        self.emission = linear(
            in_features=hidden_size,
            out_features=5,
            init_scale=0.01**2,
        )
        self.policy = PolicyNetwork(
            hidden_size=hidden_size,
            num_intentions=num_intentions,
            num_neighbors=num_neighbors,
            init_scale=init_scale,
            output_size=5,  # NOTE: dxy, dyaw, dvel
        )
        # --- 2. Constraint Posterior Definitions ---
        self.constraint_posterior_net = ConstraintPosteriorNetwork(
            hidden_size=hidden_size,
            latent_c_dim=32,  # Must be the same as the prior network's
            num_heads=num_heads
        )
        self.constraint_prior_net = ConstraintPriorNetwork(
            hidden_size=hidden_size,
            latent_c_dim=32,  # A new hyperparameter for your model
            num_heads=num_heads
        )

        self.potential_field_net = PotentialFieldNetwork(
            hidden_size=hidden_size,
            latent_c_dim=32,         # Must match the other two networks
            traj_state_dim=5,        # The feature dimension of your predicted trajectory Y
            num_encoder_layers=3,    # Similar to `num_blocks` in your HistoryEncoder
            num_heads=num_heads
        )

        self.constraint_losses = DifferentiableConstraintLosses()
        self.beta = 1.0

        


        # self.score_network = ScoreNetwork(latent_dim=hidden_size, context_dim=hidden_size)
        # self.sde = VpSde()
        # Losses and evaluation metrics
        self.optimizer_config = SeNeVAMOptimizerConfig(**optimizer_config)
        self.train_batch_time = MeanMetric()
        self.train_losses = MetricCollection(
            {
                "loss": MeanMetric(),
                "reconstruction": MeanMetric(),
                "kl_div_z": MeanMetric(),
                "z_proxy": MeanMetric(),
                "sde_score_loss": MeanMetric(),
                "sde_recon_loss": MeanMetric(),
                "sde_to_lstm_mse": MeanMetric(),
                "sde_to_noisy_cosine": MeanMetric(),
                "noisy_to_lstm_cosine": MeanMetric()
            }
        )
        metrics = MetricCollection(
            {
                "min_ade": MinADE(),
                "min_fde": MinFDE(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")


    def forward(
        self, scenario: Scenario, horizon: int, include_sdc: bool = True
    ) -> SeNeVAMOutput:
        """Forward pass the backbone generative network.

        Args:
            scenario (Scenario): Scenario dataclass object.
            horizon (int): Prediction horizon in time steps.
            include_sdc (bool, optional): Whether to include the
                self-driving car as target. Defaults to ``True``.

        Returns:
            SeNeVAMOutput: Output container for the model.
        """
        # PART 0: PREPARATIONS
        obj_type = torch.where(
            scenario.object_property.valid,
            scenario.object_property.object_types,
            torch.full_like(scenario.object_property.object_types, 0),
        )

        unique_elements, counts = torch.unique(
            obj_type[0], return_counts=True, sorted=True
        )

        ct = scenario.current_time_step
        #print("CURRENT TIME STEP:", ct)
        #print("Horizons:", horizon)
        #print("LOG TRAJECTORY SHAPE:", scenario.log_trajectory.shape)
        # PART 1: ENCODING & TARGET EXTRACTION
        x_map, x_traj = self.encoder.forward(
            map_point=scenario.map_point,
            trajectory=scenario.log_trajectory,
            properties=scenario.object_property,
            current_time=scenario.current_time_step,
        )

        _, x_future_gt = self.encoder(
            map_point=scenario.map_point,
            trajectory=scenario.log_trajectory,
            properties=scenario.object_property,
            # Pass future timesteps to the encoder
            current_time=[scenario.current_time_step + 1, scenario.current_time_step + horizon],
        )

        q_c_dist = self.constraint_posterior_net(x_traj, x_future_gt)
        p_c_dist = self.constraint_prior_net(x_traj)
        c_sample = q_c_dist.rsample()




        assert x_map is not None and x_traj is not None
        assert torch.isnan(x_map).sum() == 0 and torch.isnan(x_traj).sum() == 0
        #Linear interpolation in noise space. Intermediate noise will still yield correct positions.
        #print("ENCODED TRAJECTORY SHAPE:", x_traj.shape)
        #print("X_TRAJ SAMPLE:", x_traj.shape)
        
        is_target = get_target_mask(scenario=scenario, include_sdc=include_sdc)
        assert is_target is not None
        assert torch.isnan(is_target).sum() == 0
        x_tar, tar_valid = extract_target(data=x_traj, mask=is_target)
        assert x_tar is not None and tar_valid is not None
        assert torch.isnan(x_tar).sum() == 0 and torch.isnan(tar_valid).sum() == 0
        
        # PART 2: SDE and DECODER LOSSES
        
        #t = torch.ones(x_tar.shape[0], device=self.device)* (self.sde_T - self.sde_eps) + self.sde_eps ## DURING INFERENCE START FROM T=1
        
        # noise = torch.randn_like(x_tar)
        # noise = torch.randn_like(x_tar)
        # z_t = mean + std * noise
        
        # # Pass the NEGATION of tar_valid as the mask
        # predicted_score = self.score_network(z_t, t, context_token=torch.mean(x_map, dim=1), key_padding_mask=~tar_valid)
        # # Calculate the per-agent score matching loss
        
        # score_losses = ((predicted_score * std + noise)**2) * tar_valid.unsqueeze(-1)
        
        assert tar_valid.sum() > 0, "No valid target agents found for SDE loss calculation."
        #loss_sde = score_losses.mean()
        #losses = self.calculate_decoder_loss(x_tar, scenario, is_target, tar_valid)
        #loss_sde = loss_sde + losses
        #print("SDE LOSS: ", loss_sde.item(), " Score Loss: ", score_losses.sum().item(), " Decoder Loss: ", losses.item())
        
        num_agents_in_scene = tar_valid.shape[1] # This is the N for this batch
        gt_types_tar, _ = extract_target(scenario.object_property.object_types.unsqueeze(-1), is_target)
        assigned_types = gt_types_tar.squeeze(-1) # Shape: [B, N]
        
        #_, z_sde = self.sample_with_guidance(map_point=scenario.map_point, num_agents_to_gen=num_agents_in_scene, assigned_types=assigned_types, w_collision=1, w_layout=1, steps=100, noise_latent=z_t, t=t)
        #assert z_sde.shape == x_tar.shape, f"Shape mismatch: z_sde {z_sde.shape}, x_tar {x_tar.shape}"
        #loss_reconstruction = F.mse_loss(z_sde, x_tar)
        
            #x_tar = z_sde
        #x_tar = z_sde

        
        

        # --- 2. SDE Loss Calculation (The "Energy Optimizer") ---

        # z_true = x_traj.detach() # Detach to prevent policy gradients from affecting SDE loss
        # context = torch.mean(x_map.detach(), dim=1) # Use pooled map as context
        # T = 1.0
        # epsilon = 1e-5
        # t = torch.rand(z_true.shape[0], device=self.device) * (T - epsilon) + epsilon
        # mean, std = self.sde.marginal_prob(z_true, t)
        # noise = torch.randn_like(z_true)
        # z_t = mean + std * noise
        # predicted_score = self.score_network(z_t, t, context)
        # losses = (predicted_score * std + noise)**2
        # loss_sde = torch.mean(losses)

        #z_sde = self.sample_arrangement(x_map.detach(), num_agents=z_true.shape[1], arrangement_dim=z_true.shape[2])
        # z_sde = self.sample_for_reconstruction(z_t.detach(), x_map.detach())
        # loss_reconstruction = F.mse_loss(z_sde, z_true)

        # prepare the relative space-time features
        with torch.no_grad():
            if(isinstance(ct, int)):
                rst_traj = torch.cat(
                    [
                        scenario.log_trajectory.xy[..., ct, :],
                        scenario.log_trajectory.yaw[..., ct, None],
                        scenario.log_trajectory.velocity[..., ct, :],
                    ],
                    dim=-1,
                )
            elif(isinstance(ct, list)):
                rst_traj = torch.cat(
                    [
                        scenario.log_trajectory.xy[..., ct[0], :],
                        scenario.log_trajectory.yaw[..., ct[0], None],
                        scenario.log_trajectory.velocity[..., ct[0], :],
                    ],
                    dim=-1,
                )
            rst_tar, _ = extract_target(
                data=rst_traj,
                mask=is_target,
            )
            rst_ctx = torch.cat(
                [
                    scenario.map_point.xy,
                    scenario.map_point.orientation[..., None],
                ],
                dim=-1,
            )
            ctx_valid = scenario.map_point.valid

        # forward pass the policy network
        assert x_tar is not None and rst_tar is not None
        assert torch.isnan(x_tar).sum() == 0 and torch.isnan(rst_tar).sum() == 0
        assert rst_ctx is not None and ctx_valid is not None
        assert torch.isnan(rst_ctx).sum() == 0 and torch.isnan(ctx_valid).sum() == 0
        assert rst_tar is not None and tar_valid is not None
        assert torch.isnan(rst_tar).sum() == 0 and torch.isnan(tar_valid).sum() == 0
        assert x_map is not None
        assert torch.isnan(x_map).sum() == 0
        assert horizon > 0

        start = time.time()
        s_mean, s_vars, z_logits = self.policy.forward(
            inputs=x_tar,
            tar_xy=rst_tar[..., 0:2],
            tar_yaw=rst_tar[..., 2:3],
            tar_valid=tar_valid,
            context=x_map,
            ctx_xy=rst_ctx[..., 0:2],
            ctx_yaw=rst_ctx[..., 2:3],
            ctx_valid=ctx_valid,
            horizon=horizon,
            emission_head = self.emission,
            num_agents = num_agents_in_scene,
            assigned_types = assigned_types,
            scenario = scenario,
            global_step = self.global_step,
            constraint_embed = c_sample
        )
        
        assert s_mean is not None and s_vars is not None and z_logits is not None
        assert torch.isnan(s_mean).sum() == 0 
        assert torch.isnan(s_vars).sum() == 0 
        assert torch.isnan(z_logits).sum() == 0
        # print("S_MEAN SHAPE:", s_mean.shape)
        # print("S_VARS SHAPE:", s_vars.shape)
        # exit()

        # forward pass the emission network
        weight = self.emission.weight
        #print("BEFORE EMISSION: ", s_mean.shape, s_vars.shape)
        y_means = self.emission.forward(s_mean)
        y_covars = torch.matmul(
            torch.matmul(weight, torch.diag_embed(s_vars)),
            weight.T,
        )
        #print("AFTER EMISSION: ", y_means.shape, y_covars.shape)
        end = time.time()

        assert y_means is not None and y_covars is not None
        assert torch.isnan(y_means).sum() == 0 and torch.isnan(y_covars).sum() == 0
        _eye = torch.eye(y_covars.size(-1), device=y_covars.device)
        assert torch.isnan(_eye).sum() == 0
        y_covars = y_covars + TIKHONOV_REGULARIZATION * _eye

        return SeNeVAMOutput(
            y_means=y_means,
            y_covars=y_covars,
            s_means=s_mean,
            s_vars=s_vars,
            z_logits=z_logits,
            current_state=rst_tar,
            current_valid=tar_valid,
            q_c_dist = q_c_dist,
            p_c_dist = p_c_dist
        )


    def inference(self, scenario, horizon:int, num_samples:int=20, include_sdc:bool=True):
        
        obj_type = torch.where(
            scenario.object_property.valid,
            scenario.object_property.object_types,
            torch.full_like(scenario.object_property.object_types, 0),
        )

        unique_elements, counts = torch.unique(
            obj_type[0], return_counts=True, sorted=True
        )

        ct = scenario.current_time_step
        #print("CURRENT TIME STEP:", ct)
        #print("Horizons:", horizon)
        #print("LOG TRAJECTORY SHAPE:", scenario.log_trajectory.shape)
        # PART 1: ENCODING & TARGET EXTRACTION
        x_map, x_traj = self.encoder.forward(
            map_point=scenario.map_point,
            trajectory=scenario.log_trajectory,
            properties=scenario.object_property,
            current_time=scenario.current_time_step,
        )

        assert x_map is not None and x_traj is not None
        assert torch.isnan(x_map).sum() == 0 and torch.isnan(x_traj).sum() == 0
        #Linear interpolation in noise space. Intermediate noise will still yield correct positions.
        p_c_dist = self.constraint_prior_net(x_traj)
        c_sample = p_c_dist.sample().permute(1,0,2) # Shape: [B, num_samples, hidden_size]

        assert x_map is not None and x_traj is not None
        assert torch.isnan(x_map).sum() == 0 and torch.isnan(x_traj).sum() == 0
        #Linear interpolation in noise space. Intermediate noise will still yield correct positions.
        #print("ENCODED TRAJECTORY SHAPE:", x_traj.shape)
        #print("X_TRAJ SAMPLE:", x_traj.shape)
        
        is_target = get_target_mask(scenario=scenario, include_sdc=include_sdc)
        assert is_target is not None
        assert torch.isnan(is_target).sum() == 0
        x_tar, tar_valid = extract_target(data=x_traj, mask=is_target)
        assert x_tar is not None and tar_valid is not None
        assert torch.isnan(x_tar).sum() == 0 and torch.isnan(tar_valid).sum() == 0

        with torch.no_grad():
            if(isinstance(ct, int)):
                rst_traj = torch.cat(
                    [
                        scenario.log_trajectory.xy[..., ct, :],
                        scenario.log_trajectory.yaw[..., ct, None],
                        scenario.log_trajectory.velocity[..., ct, :],
                    ],
                    dim=-1,
                )
            elif(isinstance(ct, list)):
                rst_traj = torch.cat(
                    [
                        scenario.log_trajectory.xy[..., ct[0], :],
                        scenario.log_trajectory.yaw[..., ct[0], None],
                        scenario.log_trajectory.velocity[..., ct[0], :],
                    ],
                    dim=-1,
                )
            rst_tar, _ = extract_target(
                data=rst_traj,
                mask=is_target,
            )
            rst_ctx = torch.cat(
                [
                    scenario.map_point.xy,
                    scenario.map_point.orientation[..., None],
                ],
                dim=-1,
            )
            ctx_valid = scenario.map_point.valid

        # forward pass the policy network
        assert x_tar is not None and rst_tar is not None
        assert torch.isnan(x_tar).sum() == 0 and torch.isnan(rst_tar).sum() == 0
        assert rst_ctx is not None and ctx_valid is not None
        assert torch.isnan(rst_ctx).sum() == 0 and torch.isnan(ctx_valid).sum() == 0
        assert rst_tar is not None and tar_valid is not None
        assert torch.isnan(rst_tar).sum() == 0 and torch.isnan(tar_valid).sum() == 0
        assert x_map is not None
        assert torch.isnan(x_map).sum() == 0
        assert horizon > 0

        start = time.time()
        s_mean, s_vars, z_logits= self.policy.forward(
            inputs=x_tar,
            tar_xy=rst_tar[..., 0:2],
            tar_yaw=rst_tar[..., 2:3],
            tar_valid=tar_valid,
            context=x_map,
            ctx_xy=rst_ctx[..., 0:2],
            ctx_yaw=rst_ctx[..., 2:3],
            ctx_valid=ctx_valid,
            horizon=horizon,
            emission_head = self.emission,
            num_agents = num_agents_in_scene,
            assigned_types = assigned_types,
            scenario = scenario,
            global_step = self.global_step,
            constraint_embed = c_sample
        )

    def _compute_pde_loss(
        self,
        model_output: SeNeVAMOutput,
        c_embed: torch.Tensor,
        scenario: Scenario
    ) -> torch.Tensor:
        """
        Calculates the PDE residual loss for the PINN framework.

        This function enforces the governing PDE: ∇²Φ = ∇²L, where Φ is the
        learned potential and L is the guidance loss from constraints.
        """
        pde_loss_total = 0.0
        num_modes = model_output.y_means.size(-3)

        # The PINN loss must apply to all plausible futures (modes) the model predicts
        for i in range(num_modes):
            # 1. Get the mean trajectory for the current mode
            # This is the "answer" from the student (main model) that the teacher (PINN) will check
            Y_i = model_output.y_means[..., i, :, :]
            Y_i.requires_grad_(True)  # Crucial for computing gradients w.r.t. the trajectory

            # 2. Calculate the LHS of the PDE: ∇²Φ
            # Evaluate the potential field's energy for this trajectory
            self.potential_field_net.train()  # Ensure the network is in training mode for autograd
            phi_val = self.potential_field_net(Y_i, c_embed)
            # Compute the Laplacian of the potential field
            laplacian_phi = self._compute_laplacian(phi_val, Y_i)

            # 3. Calculate the RHS of the PDE: f = ∇²L
            # First, get the scalar guidance loss L(Y;c)
            guidance_loss = self.constraint_losses(
                Y=Y_i,
                scenario=scenario
                # c_config can be passed here if needed, e.g., for weights
            )
            # Then, compute its Laplacian to get the source term
            source_term_f = self._compute_laplacian(guidance_loss, Y_i)

            # 4. Compute the residual
            # The residual is the difference between the two sides of the PDE.
            # A perfect solution would have a residual of zero.
            residual = laplacian_phi - source_term_f
            
            # 5. Accumulate the squared error loss
            # We penalize the mean squared residual to drive it to zero.
            pde_loss_total += torch.mean(residual**2)

        # Average the loss over all modes
        return pde_loss_total / num_modes

    def _compute_laplacian(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Laplacian of a scalar output `y` w.r.t. a tensor input `x`.
        
        Laplacian(y) = sum over all elements of x (∂²y / ∂xᵢ²)

        Args:
            y: Scalar tensor, output of a function (e.g., potential or loss). Shape: [B, 1]
            x: Tensor input to the function. Shape: [B, ...]

        Returns:
            Scalar tensor representing the Laplacian. Shape: [B]
        """
        # First, compute the gradient of y w.r.t. x (the Jacobian)
        grad_y = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True  # create_graph=True is essential for second-order derivatives
        )[0]

        # Flatten the gradient and input for easier iteration
        grad_y_flat = grad_y.view(grad_y.size(0), -1)
        x_flat = x.view(x.size(0), -1)

        # Compute the second derivatives (diagonals of the Hessian)
        # NOTE: This loop-based implementation is clear but can be slow for very high
        # dimensional inputs. For production, this can be optimized using
        # functorch or by computing Hessian-vector products.
        laplacian = 0.0
        for i in range(grad_y_flat.size(1)):
            # Compute the gradient of the i-th element of the first gradient
            # w.r.t. the full input vector x
            grad2_i = torch.autograd.grad(
                outputs=grad_y_flat[:, i],
                inputs=x_flat,
                grad_outputs=torch.ones_like(grad_y_flat[:, i]),
                create_graph=True
            )[0]
            # We only need the diagonal term ∂²y/∂xᵢ², which is the i-th element
            laplacian += grad2_i[:, i]
            
        return laplacian     



    # =========================================================================
    # LightningModule hooks
    # =========================================================================
    def configure_optimizers(
        self,
    ) -> Dict[str, Union[Optimizer, Dict[str, Any]]]:
        # NOTE: create parameter groups and only apply weight decay to
        # weights in linear or convolution layers. See:
        # https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/
        wd_params, nwd_params = set(), set()
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if "bias" in pn:
                    # all biases will not be decayed
                    nwd_params.add(fpn)
                elif "weight" in pn and isinstance(
                    m, (nn.Linear, nn.MultiheadAttention, nn.LSTM)
                ):
                    # weights of whitelist modules will be weight decayed
                    wd_params.add(fpn)
                elif "weight" in pn and isinstance(
                    m,
                    (
                        nn.LayerNorm,
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.Embedding,
                    ),
                ):
                    # weights of blacklist modules will NOT be weight decayed
                    nwd_params.add(fpn)

        # validate that we considered every parameter
        # print("--- Found Parameters ---")
        # for name, _ in self.named_parameters():
        #     print(name)
        # print("------------------------")
        # exit()
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = wd_params & nwd_params
        union_params = wd_params | nwd_params
        if not len(inter_params) == 0:
            raise RuntimeError(
                f"Parameters {str(inter_params):s}"
                "appear both weight_decay_params and other_params."
            )
        if len(param_dict.keys() - union_params) != 0:
            LOGGER.warning(
                f"Parameters {str(param_dict.keys() - union_params):s} "
                "were not in weight-decay params or other params."
                "By default, they are considered as other params."
            )
            for pn in param_dict.keys() - union_params:
                nwd_params.add(pn)

        optimizer = AdamW(
            params=[
                {
                    "params": [
                        param_dict[pn] for pn in sorted(list(wd_params))
                    ],
                    "weight_decay": self.optimizer_config.weight_decay,
                },
                {
                    "params": [
                        param_dict[pn] for pn in sorted(list(nwd_params))
                    ],
                    "weight_decay": 0.0,
                },
            ],
            betas=self.optimizer_config.betas,
            lr=self.optimizer_config.lr,
            eps=self.optimizer_config.eps,
            maximize=self.optimizer_config.maximize,
        )
        scheduler = SeNeVAMLRScheduler(
            optimizer=optimizer,
            min_lr=self.optimizer_config.min_lr,
            warmup_steps=self.optimizer_config.warmup_steps,
            total_steps= self.optimizer_config.total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        # NOTE: reset warmup and total steps for the learning rate scheduler
        dl = self.trainer.train_dataloader
        if dl is not None:
            self.optimizer_config.warmup_steps = len(dl)
            if self.trainer.max_epochs is not None:
                self.optimizer_config.total_steps = (
                    len(dl) * self.trainer.max_epochs
                )

    def training_step(self, batch: Scenario) -> torch.Tensor:
        _start_time = time.monotonic()

        # prepare the target tensor for training
        target, valid = self.get_target(scenario=batch)
        horizon = target.size(-2)

        output = self.forward(scenario=batch, horizon=horizon)
        losses = self._compute_losses(
            y_obs=target,
            valid=valid,
            y_means=output.y_means,
            y_covar=output.y_covars,
            z_logits=output.z_logits,
            # NOTE: only train proxy network after training generative model
            train_proxy=self.global_step
            >= 0.5 * self.optimizer_config.total_steps,
            q_c_dist=output.q_c_dist,
            p_c_dist=output.p_c_dist
        )
        pde_loss = self._compute_pde_loss(
            model_output=output,
            scenario=batch # Pass the full scenario for context
        )
        losses = losses + self.beta * pde_loss
        losses["pde_loss"] = pde_loss.item()
        _batch_time = time.monotonic() - _start_time

        # Log training status
        for i, param_group in enumerate(self.optimizers().param_groups):
            self.log(
                f"status/lr_param_group_{i+1}",
                param_group["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        # update trackers
        self.train_batch_time(output.y_means.size(0) / _batch_time)
        self.log(
            "status/batch_step_per_second",
            value=self.train_batch_time,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )
        for name, value in losses.items():
            if name in self.train_losses:
                self.train_losses[name](value)
                self.log(
                    name=f"train/{name}",
                    value=self.train_losses[name],
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

        # evaluate the metrics
        with torch.no_grad():
            current, _ = self.get_current(scenario=batch)
            cum_target = target.cumsum(dim=-2) + current
            cum_pred = output.y_means.cumsum(dim=-2) + current.unsqueeze(-3)
            for name, metric in self.train_metrics.items():
                metric(
                    input_xy=cum_pred[..., 0:2],
                    target_xy=cum_target.unsqueeze(-3)[..., 0:2],
                    valid=valid.unsqueeze(-2),
                )
                self.log(
                    name=str(name),
                    value=metric,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )

        return losses["loss"]

    # =========================================================================
    # properties
    # =========================================================================
    @property
    def alpha(self) -> Optional[float]:
        return self.hparams.get("alpha", None)

    @property
    def gamma(self) -> float:
        return self.hparams.get("gamma", 2.0)

    # =========================================================================
    # private methods
    # =========================================================================
    def _compute_losses(
        self,
        y_obs: torch.Tensor,
        valid: torch.Tensor,
        y_means: torch.Tensor,
        y_covar: torch.Tensor,
        z_logits: torch.Tensor,
        # MODIFIED: Add the distributions for the latent constraint `c` to the signature
        q_c_dist: D.Distribution,
        p_c_dist: D.Distribution,
        train_proxy: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        # --- Steps 1-4 are your original logic for the intention `z` ---

        # step 1: evaluate the variational distribution q(z)
        log_pdf = []
        for i in range(y_means.size(-3)):
            log_pdf.append(
                torch.sum(
                    self.mvn_log_pdf(
                        y_obs=y_obs,
                        y_means=y_means[..., i, :, :],
                        y_covar=y_covar[..., i, :, :, :],
                    )
                    * valid.float(),
                    dim=-1,
                )
            )
        log_pdf = torch.stack(log_pdf, dim=-1)
        with torch.no_grad():
            log_q_z = torch.log_softmax(
                log_pdf - math.log(log_pdf.size(-1)), dim=-1
            )

        # step 2: calculate the reconstruction loss
        loss = -torch.sum(log_q_z.exp() * log_pdf, dim=-1)

        # step 3: compute the KL divergence for z
        with torch.no_grad():
            kl_div_z = torch.sum(log_q_z.exp() * log_q_z, dim=-1).neg()

        # step 4: calculate the cross-entropy loss for z-proxy network
        targets = nn.functional.one_hot(
            log_q_z.argmax(dim=-1), num_classes=log_q_z.size(-1)
        ).float()
        cross_entropy = -targets * F.log_softmax(z_logits, dim=-1)
        pt = torch.sum(F.softmax(z_logits, dim=-1) * targets, dim=-1)
        focal_weight = torch.pow(1 - pt, exponent=self.gamma)
        if self.alpha is not None:
            cross_entropy = self.alpha * cross_entropy
        z_proxy_loss = torch.sum(
            focal_weight.unsqueeze(-1) * cross_entropy, dim=-1
        )
        
        # --- NEW: Step 5 adds the KL divergence for the constraint `c` ---

        # step 5: calculate the KL divergence for the latent constraint c
        # This term regularizes the constraint embedding space.
        kl_div_c = D.kl.kl_divergence(q_c_dist, p_c_dist)

        # --- Combine all losses ---
        
        # MODIFIED: Add the original losses and the new, weighted kl_div_c
        # Note: You will need to add `self.kl_c_weight` as a hyperparameter
        #       in your model's __init__ (e.g., self.kl_c_weight = 0.1)
        loss = loss + float(train_proxy) * z_proxy_loss + self.kl_c_weight * kl_div_c
        
        # MODIFIED: Add kl_div_c to the returned dictionary for logging
        return {
            "loss": loss[valid.any(dim=-1)].mean(),
            "reconstruction": loss[valid.any(dim=-1)].mean().detach(),
            "kl_div_z": kl_div_z[valid.any(dim=-1)].mean().detach(),
            "kl_div_c": kl_div_c[valid.any(dim=-1)].mean().detach(), # New log item
            "z_proxy": z_proxy_loss[valid.any(dim=-1)].mean().detach(),
        }

    # =========================================================================
    # static methods
    # =========================================================================
    @staticmethod
    def get_current(
        scenario: Scenario, include_sdc: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the current motion state of the target agents.

        Args:
            scenario (Scenario): Scenario dataclass object.
            include_sdc (bool, optional): Whether to include the
                self-driving car as target. Defaults to ``True``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of current
                state tensor and valid mask tensor.
        """
        is_target = get_target_mask(scenario=scenario, include_sdc=include_sdc)
        if(isinstance(scenario.current_time_step, int)):
            current = scenario.log_trajectory[
                ..., scenario.current_time_step : scenario.current_time_step + 1
            ]
        elif(isinstance(scenario.current_time_step, list)):
            current = scenario.log_trajectory[
                ..., scenario.current_time_step[0] : scenario.current_time_step[0] + 1
            ]
        state = torch.cat(
            [current.xy, current.yaw.unsqueeze(-1), current.velocity], dim=-1
        )
        valid = current.valid

        state, _ = extract_target(data=state, mask=is_target)
        valid, _ = extract_target(data=valid, mask=is_target)

        state = state.reshape(*state.shape[:-1], 1, 5)
        valid = valid.reshape(*valid.shape[:-1], 1)

        return state.contiguous(), valid.contiguous()

    @staticmethod
    def get_target(
        scenario: Scenario, include_sdc: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get target tensor for the given scenario dataclass object.

        Args:
            scenario (Scenario): Scenario dataclass object.
            include_sdc (bool, optional): Whether to include the
                self-driving car as target. Defaults to ``True``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of target
                tensor and valid mask tensor.
        """
        is_target = get_target_mask(scenario=scenario, include_sdc=include_sdc)
        if( type(scenario.current_time_step) is list):
            future = scenario.log_trajectory[..., scenario.current_time_step[0] : scenario.current_time_step[1]]
        else:
            future = scenario.log_trajectory[..., scenario.current_time_step :]
        y = torch.cat(
            [future.xy, future.yaw.unsqueeze(-1), future.velocity], dim=-1
        )
        valid = future.valid
        target = torch.diff(y, n=1, dim=-2)
        target[..., 2] = wrap_angles(target[..., 2])
        valid = torch.logical_and(valid[..., 0:-1], valid[..., 1:])

        horizon = valid.size(-1)
        target = target.reshape(*target.shape[:-2], -1)
        target, _ = extract_target(data=target, mask=is_target)
        valid, _ = extract_target(data=valid, mask=is_target)

        target = target.reshape(*target.shape[:-1], horizon, 5)
        valid = valid.reshape(*valid.shape[:-1], horizon)

        return target.contiguous(), valid.contiguous()

    @staticmethod
    @torch.jit.script
    def mvn_log_pdf(
        y_obs: torch.Tensor,
        y_means: torch.Tensor,
        y_covar: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the log density of the multivariate normal distribution.

        Args:
            y_obs (torch.Tensor): Future observation of shape `(*, T, 5)`.
            y_means (torch.Tensor): Mean values in the marginal distribution
                with a shape of `(*, T, 5)`.
            y_covar (torch.Tensor): Covariance matrices in the marginal
                distribution with a shape of `(*, T, 5, 5)`.

        Returns:
            torch.Tensor: Reconstruction loss value.
        """
        # step 1: compute the cholesky decomposition
        chol: torch.Tensor = torch.linalg.cholesky(y_covar, upper=False)

        # step 2: compute the mahalanobis distance
        diff = y_obs - y_means
        mahalanobis: torch.Tensor = torch.linalg.solve_triangular(
            chol.transpose(-1, -2), diff.unsqueeze(-1), upper=False
        )
        mahalanobis = mahalanobis.squeeze(-1).square().sum(dim=-1)

        # step 2: compute the log determinant
        logdet = 2.0 * torch.sum(
            torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)), dim=-1
        )

        # step 3: compute the log normalizer
        log_normalizer = y_means.size(-1) * math.log(2.0 * math.pi)

        return -0.5 * (mahalanobis + logdet + log_normalizer)
