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
import torch.distributions as D
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
from planner.model.components.constraints_loss import DifferentiableConstraintLosses, AgentCollisionMetrics, VectorMapOffroadMetrics

from planner.model.components.constraint_models import (
    ConstraintPosteriorNetwork,
    ConstraintPriorNetwork,
    PotentialFieldNetwork
)

import torch.nn.functional as F
from typing import Any, Literal, Tuple
# Constants
TIKHONOV_REGULARIZATION = 0.5 / math.pi
LOGGER = get_logger(__name__)

def variance_scaling(
    tensor: torch.Tensor,
    scale: float = 1.0,
    mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in",
    distribution: Literal["uniform", "normal", "truncated_normal"] = "normal",
) -> None:
    """Initialize the tensor in-place with variance scaling.

    This function implements the variance scaling initialization method
    as in the TensorFlow library :url:`https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling` as well as in the JAX library :url:`https://jax.readthedocs.io/en/latest/_autosummary/jax.nn.initializers.variance_scaling.html`.

    Args:
        tensor (torch.Tensor): The input tensor to be initialized in-place.
        scale (float, optional): The scaling factor (positive float).
            Defaults to :math:`1.0`.
        mode (Literal["fan_in", "fan_out", "fan_avg"], optional): One of the
            `"fan_in"`, `"fan_out"`, or `"fan_avg"`. Defaults to `"fan_in"`.
        distribution (Literal["uniform", "normal", "truncated_normal"],
            optional): One of `"uniform"`, `"normal"`, or
            `"truncated_normal"`. Defaults to "normal".
    """
    assert (
        isinstance(scale, float) and scale >= 0.0
    ), "The scale factor must be non-negative."
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor=tensor)
    if mode == "fan_in":
        n = fan_in
    elif mode == "fan_out":
        n = fan_out
    elif mode == "fan_avg":
        n = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")
    std = (max(1e-10, scale) / n) ** 0.5
    if distribution == "uniform":
        nn.init.uniform_(tensor, a=-std, b=std)
    elif distribution == "normal":
        nn.init.normal_(tensor, mean=0.0, std=std)
    elif distribution == "truncated_normal":
        a, b = -2.0 * std, 2.0 * std
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=a, b=b)
    else:
        raise ValueError(f"Invalid distribution: {distribution}")
    
def linear(
    in_features: int,
    out_features: int,
    init_scale: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> nn.Linear:
    """Create a linear layer with custom initialization.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        init_scale (float): The scale factor for the initialization.
        *args, **kwargs: Additional arguments for the linear layer.

    Returns:
        nn.Linear: The initialized linear layer.
    """
    layer = nn.Linear(in_features=in_features, out_features=out_features)
    if hasattr(layer, "weight"):
        variance_scaling(layer.weight, scale=init_scale)
    if hasattr(layer, "bias"):
        nn.init.constant_(layer.bias, val=0.0)

    return layer

@torch.jit.script
def linear_gaussian_reconstruction_loss(
    y: torch.Tensor,
    x_vars: torch.Tensor,
    y_mean: torch.Tensor,
    weight: torch.Tensor,
    y_vars: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the analytical reconstruction loss for a linear Gaussian model.

    Args:
        y (torch.Tensor): Ground-truth observation.
        x_vars (torch.Tensor): Variance vector of the latent variable.
        y_mean (torch.Tensor): Mean vector of the emission distribution.
        weight (torch.Tensor): Weight in the linear emission layer.
        y_vars (Optional[torch.Tensor]): Variance vector of the emission
            distribution. If `None`, ignore related terms. Defaults to `None`.

    Returns:
        torch.Tensor: The analytical reconstruction loss.
    """
    # sanity checks
    n, d = weight.shape
    if not y_mean.shape[-1] == n and x_vars.shape[-1] == d:
        raise ValueError("Shape mismatched!")

    if y_vars is None:
        # NOTE: when y_vars is not provided, treat it as a identity matrix
        # compute the mahalanobis term
        mahalanobis = torch.sum(torch.square(y - y_mean), dim=-1)
        

        # compute the trace term
        w = torch.einsum(
            "ip, pj, btjq -> btiq", weight.T, weight, torch.diag_embed(x_vars)
        )
        trace = torch.sum(torch.diagonal(w, dim1=-2, dim2=-1), dim=-1)

        return 0.5 * (mahalanobis + trace + n * math.log(2 * math.pi))
    else:
        # compute the log-normalizaer
        log_normalizer = torch.sum(torch.log(2 * torch.pi * y_vars), dim=-1)

        # compute the mahalanobis term
        mahalanobis = torch.sum(torch.square(y - y_mean).div(y_vars), dim=-1)

        # compute the trace term
        _lt = torch.matmul(weight.t(), torch.diag_embed(1 / y_vars))
        _rt = torch.matmul(weight, torch.diag_embed(x_vars))
        trace = torch.sum(torch.diagonal(_lt @ _rt, dim1=-2, dim2=-1), dim=-1)

        return 0.5 * (log_normalizer + mahalanobis + trace)

class AttributeEmissionHead(nn.Module):
    """
    Decodes a latent embedding back into the physical attribute space.
    This is used for the auxiliary reconstruction loss on the posterior network.
    """
    def __init__(self, hidden_size: int, attribute_dim: int):
        super().__init__()
        # A linear layer to predict the mean of the attributes
        self.emission_mean = linear(hidden_size, attribute_dim, init_scale=1e-10)
        
        # A learnable parameter for the variance of the attributes
        self.emission_vars = nn.Parameter(torch.zeros(attribute_dim), requires_grad=True)

    def forward(self, fused_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fused_embedding: The internal state of the posterior network.
                             Shape: [B, N, HiddenSize].

        Returns:
            A tuple of (mean, variance) for the predicted attributes.
        """
        # Predict the mean of the attributes
        y_mean_attr = self.emission_mean(fused_embedding)
        
        # Get the learnable variance, ensuring it's positive
        y_vars_attr = torch.square(self.emission_vars) + 1e-2 # Tikhonov regularization
        
        return y_mean_attr, y_vars_attr

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
    p_c_mean: torch.Tensor
    """torch.Tensor: Mean of the constraint prior distribution."""
    p_c_var: torch.Tensor
    """torch.Tensor: Variance of the constraint prior distribution."""
    q_c_mean: torch.Tensor
    """torch.Tensor: Mean of the constraint posterior distribution."""
    q_c_var: torch.Tensor
    """torch.Tensor: Variance of the constraint posterior distribution."""
    emission_c_mean: torch.Tensor
    """torch.Tensor: Mean of the attribute emission distribution."""
    emission_c_var: torch.Tensor
    """torch.Tensor: Variance of the attribute emission distribution."""
    attribute_tensor: torch.Tensor
    """torch.Tensor: The ground-truth attribute tensor."""
    emission_weight: torch.Tensor
    """torch.Tensor: The weight matrix of the attribute emission head."""


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
    _init_states: nn.Parameter

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
            num_attributes=2,  # Number of constraint attributes (e.g., collision, offroad)
        )
        self.constraint_prior_net = ConstraintPriorNetwork(
            hidden_size=hidden_size,
            latent_c_dim=32,  # A new hyperparameter for your model
            num_heads=2
        )

        self.potential_field_net = PotentialFieldNetwork(
            hidden_size=hidden_size,
            latent_c_dim=32,         # Must match the other two networks
            traj_state_dim=5,        # The feature dimension of your predicted trajectory Y
            num_encoder_layers=3,    # Similar to `num_blocks` in your HistoryEncoder
            num_heads=2
        )
        
        self.num_mixtures = 1

        self.constraint_losses = DifferentiableConstraintLosses()
        self.beta = 1.0

        self.agent_collision_metric = AgentCollisionMetrics(num_disks=5, buffer_dist=0.2)
        self.vector_map_offroad_metric = VectorMapOffroadMetrics(lane_half_width=2.0)
        self.attribute_emission = AttributeEmissionHead(32, attribute_dim=2)
        


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
                "kl_div_c": MeanMetric(),
                "emission_loss": MeanMetric(),
            }
        )
        metrics = MetricCollection(
            {
                "min_ade": MinADE(),
                "min_fde": MinFDE(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self._init_states = nn.Parameter(
            torch.randn(2 * self.num_mixtures, 2 * hidden_size), requires_grad=True
        )
        self.reset_parameters()


    def forward(
        self, scenario: Scenario, horizon: int, target: Optional[torch.Tensor] = None, include_sdc: bool = True
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

        x_map, x_traj = self.encoder.forward(
            map_point=scenario.map_point,
            trajectory=scenario.log_trajectory,
            properties=scenario.object_property,
            current_time=scenario.current_time_step,
        )

        is_target = get_target_mask(scenario=scenario, include_sdc=include_sdc)

        collision_attributes = self.agent_collision_metric(target, scenario, is_target)
        offroad_attributes = self.vector_map_offroad_metric(target, scenario, is_target)

        attribute_tensor = torch.stack(
            [collision_attributes, offroad_attributes], 
            dim=-1
        ) # Final shape: [B, N, T, 2]


        assert x_map is not None and x_traj is not None
        assert torch.isnan(x_map).sum() == 0 and torch.isnan(x_traj).sum() == 0

        
        
        assert is_target is not None
        assert torch.isnan(is_target).sum() == 0
        x_tar, tar_valid = extract_target(data=x_traj, mask=is_target)
        assert x_tar is not None and tar_valid is not None
        assert torch.isnan(x_tar).sum() == 0 and torch.isnan(tar_valid).sum() == 0
        
        q_c_mean, q_c_var = self.constraint_posterior_net(x_tar, attribute_tensor)
        p_c_mean, p_c_var = self.constraint_prior_net(x_tar, x_map)
        y_mean_attr, y_vars_attr = self.attribute_emission(q_c_mean)
        emission_weight = self.attribute_emission.emission_mean.weight
        
        #print("Q_C_MEAN SHAPE:", q_c_mean.shape)
        #print("Attribute Tensor SHAPE:", attribute_tensor.shape)
        #print("X_TAR SHAPE:", x_tar.shape)
        
        assert tar_valid.sum() > 0, "No valid target agents found for SDE loss calculation."

        
        num_agents_in_scene = tar_valid.shape[1] # This is the N for this batch
        gt_types_tar, _ = extract_target(scenario.object_property.object_types.unsqueeze(-1), is_target)
        assigned_types = gt_types_tar.squeeze(-1) # Shape: [B, N]

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
            constraint_embed = p_c_mean
        )
        
        assert s_mean is not None and s_vars is not None and z_logits is not None
        assert torch.isnan(s_mean).sum() == 0 
        assert torch.isnan(s_vars).sum() == 0 
        assert torch.isnan(z_logits).sum() == 0


        # forward pass the emission network
        weight = self.emission.weight

        y_means = self.emission.forward(s_mean)
        y_covars = torch.matmul(
            torch.matmul(weight, torch.diag_embed(s_vars)),
            weight.T,
        )

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
            p_c_mean=p_c_mean,
            p_c_var=p_c_var,
            q_c_mean=q_c_mean,
            q_c_var=q_c_var, 
            emission_c_mean = y_mean_attr,
            emission_c_var = y_vars_attr,
            attribute_tensor = attribute_tensor,
            emission_weight = emission_weight
        ), is_target


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
        model_output: Dict, 
        scenario: Scenario,
        is_target: torch.Tensor
        
    ) -> torch.Tensor:
        """
        Calculates the PDE residual loss for the PINN framework.

        This function enforces the governing PDE: -∇²Φ = L(Y), where Φ is the
        learned potential and L is the sum of physical constraint violations.
        """
        # Get the necessary outputs from the main model's forward pass
        y_means = model_output.y_means
        # We use the mean of the posterior `c` distribution for conditioning the potential field
        c_embed = model_output.q_c_mean

        pde_loss_total = 0.0
        num_modes = y_means.size(-3)
        mse_loss = nn.MSELoss()

        # The PINN loss must apply to the mean prediction of each trajectory mode
        for i in range(num_modes):
            # 1. Get the mean trajectory for the current mode
            Y_i = y_means[..., i, :, :]
            Y_i.requires_grad_(True)
            # 2. Calculate the LHS of the PDE: -∇²Φ
            # Evaluate the potential field's energy for this trajectory
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                phi_val = self.potential_field_net(Y_i, c_embed)
                # Compute the Laplacian of the potential field
                laplacian_phi = self._compute_laplacian(phi_val, Y_i)

                # 3. Calculate the RHS of the PDE: The Source Term L(Y)
                #    First, compute the physical attribute tensor A(Y)
                collision_attributes = self.agent_collision_metric(Y_i, scenario, is_target)
                offroad_attributes = self.vector_map_offroad_metric(Y_i, scenario, is_target)
                attribute_pred = torch.stack(
                    [collision_attributes, offroad_attributes], 
                    dim=-1
                ) # Final shape: [B, N, T, 2]
                
                
                #    Then, define the scalar loss L as the sum of all attribute violations.
                #    This is our measure of "physical incorrectness".
                guidance_loss_L = mse_loss(attribute_pred, model_output.attribute_tensor)

                # 4. Compute the residual for the equation: -∇²Φ = L
                #    residual = (-laplacian_phi) - guidance_loss_L
                residual = -laplacian_phi - guidance_loss_L
                
                # 5. Accumulate the squared error loss
                pde_loss_total += torch.mean(residual**2)

        # Average the loss over all trajectory modes
        return pde_loss_total / num_modes


    def _compute_laplacian(self, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the Laplacian of a scalar output `y` w.r.t. a tensor input `x`.
        Laplacian(y) = sum of the diagonal elements of the Hessian of y.
        
        Args:
            y: Scalar tensor for each batch item. Shape: [B, 1]
            x: Input tensor to the function that produced y. Shape: [B, ...]

        Returns:
            A scalar Laplacian value for each batch item. Shape: [B]
        """
        B = y.shape[0]
        
        # 1. Compute the first gradient (the Jacobian vector)
        #    `create_graph=True` is essential to build the graph for the second derivative.
        grad_y = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=torch.ones_like(y),
            create_graph=True
        )[0] # Shape is the same as x

        # 2. Iteratively compute the diagonal of the Hessian and sum to get the Laplacian
        laplacian = torch.zeros(B, device=y.device)
        
        # Flatten the first gradient for easier iteration
        flat_grad_y = grad_y.view(B, -1)

        for i in range(flat_grad_y.shape[1]):
            # Compute the gradient of the i-th component of the first gradient vector.
            # This gives us the i-th COLUMN of the Hessian matrix.
            grad2_column_i = torch.autograd.grad(
                outputs=flat_grad_y[:, i],
                # CRUCIAL FIX: Always differentiate w.r.t. the original tensor `x`
                # to preserve the computation graph.
                inputs=x,
                grad_outputs=torch.ones_like(flat_grad_y[:, i]),
                retain_graph=True # We must retain the graph for the next iteration of the loop
            )[0]
            
            # We only need the diagonal term (the i-th element of the i-th column).
            # We add it to our running sum for the Laplacian.
            laplacian += grad2_column_i.view(B, -1)[:, i]
            
        return laplacian   

    def reset_parameters(self) -> None:
        """Reset the parameters of the generative network."""
        variance_scaling(self._init_states)

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

        output, is_target = self.forward(scenario=batch, horizon=horizon, target=target)
        losses = self._compute_losses(
            y_obs=target,
            valid=valid,
            y_means=output.y_means,
            y_covar=output.y_covars,
            z_logits=output.z_logits,
            # NOTE: only train proxy network after training generative model
            train_proxy=self.global_step
            >= 0.5 * self.optimizer_config.total_steps,
            p_c_mean=output.p_c_mean,
            p_c_vars=output.p_c_var,
            q_c_mean=output.q_c_mean,
            q_c_vars=output.q_c_var,
            c_emission_mean=output.emission_c_mean,
            c_emission_var=output.emission_c_var,
            emission_weight=output.emission_weight,
            attribute_tensor=output.attribute_tensor
            
        )
        # pde_loss = self._compute_pde_loss(
        #     model_output=output,
        #     scenario=batch, # Pass the full scenario for context
        #     is_target=is_target
        # )
        losses["loss"] = losses["loss"] #+ 0.0001 * pde_loss
        #losses["pde_loss"] = pde_loss.item()
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
        p_c_mean: torch.Tensor,
        p_c_vars: torch.Tensor,
        q_c_mean: torch.Tensor,
        q_c_vars: torch.Tensor,
        c_emission_mean: torch.Tensor,
        c_emission_var: torch.Tensor,
        emission_weight: torch.Tensor,
        attribute_tensor: torch.Tensor,
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
        kl_div_c = self._compute_constraint_kl_loss(
            q_c_mean=q_c_mean,
            q_c_vars=q_c_vars,
            p_c_mean=p_c_mean,
            p_c_vars=p_c_vars
        )

        # --- Combine all losses ---
        
        # MODIFIED: Add the original losses and the new, weighted kl_div_c
        # Note: You will need to add `self.kl_c_weight` as a hyperparameter
        #       in your model's __init__ (e.g., self.kl_c_weight = 0.1)
        loss = loss + float(train_proxy) * z_proxy_loss + kl_div_c

        emission_loss = linear_gaussian_reconstruction_loss(
            y=attribute_tensor, # Use aggregated attributes as target
            x_vars=q_c_vars,
            y_mean=c_emission_mean,
            y_vars=c_emission_var,
            weight=emission_weight
        ).sum(dim=-1).mean() # Sum over attributes and average over batch/agents

        loss = loss + 0.01 * emission_loss

        
        # MODIFIED: Add kl_div_c to the returned dictionary for logging
        return {
            "loss": loss[valid.any(dim=-1)].mean(),
            "reconstruction": loss[valid.any(dim=-1)].mean().detach(),
            "kl_div_z": kl_div_z[valid.any(dim=-1)].mean().detach(),
            "kl_div_c": kl_div_c.mean().detach(), # New log item
            "z_proxy": z_proxy_loss[valid.any(dim=-1)].mean().detach(),
            "emission_loss": emission_loss.detach()

        }
    

    def _compute_constraint_kl_loss(
        self,
        q_c_mean: torch.Tensor, # Posterior mean
        q_c_vars: torch.Tensor, # Posterior variance
        p_c_mean: torch.Tensor, # Prior mean
        p_c_vars: torch.Tensor, # Prior variance
    ) -> torch.Tensor:
        """
        Computes the hierarchical KL divergence for a Mixture of Gaussians latent space.
        KL( q(z,c|x,y) || p(z,c|x) ) = E_q(z|x,y)[ KL( q(c|x,y,z) || p(c|x,z) ) ] + KL( q(z|x,y) || p(z|x) )
        """
        
        def _evaluate_z(prior_mean, prior_vars, posterior_mean, posterior_vars, num_mixtures):
            """Computes the optimal posterior over the mixture components, q(z|x,y)."""
            # Using log-sum-exp trick for stability, this is equivalent to your version
            # This calculates KL(q(c|x,y,z) || p(c|x,z)) for each component z
            kl_per_component = 0.5 * (
                torch.sum(torch.log(prior_vars) - torch.log(posterior_vars), dim=-1) +
                torch.sum(posterior_vars / prior_vars, dim=-1) +
                torch.sum((posterior_mean - prior_mean)**2 / prior_vars, dim=-1) -
                prior_mean.shape[-1]
            )
            
            # The optimal q(z) is softmax of the negative KL
            log_q_z = F.log_softmax(-kl_per_component, dim=-1)
            return torch.exp(log_q_z)

        # 1. Compute the optimal posterior over mixture components, q(z|x,y)
        with torch.no_grad():
            q_z = _evaluate_z(p_c_mean, p_c_vars, q_c_mean, q_c_vars, self.num_mixtures)

        # 2. Compute KL divergence of the categorical latent `z`
        #    Assumes a uniform prior over mixtures: p(z) = 1/K
        uniform_prior_log_prob = -math.log(self.num_mixtures)
        kl_div_z = torch.sum(q_z * (torch.log(q_z + 1e-9) - uniform_prior_log_prob), dim=-1)
        
        # 3. Compute the expected KL divergence of the continuous latent `c`
        #    Create distribution objects for the KL calculation
        p_dist = D.MultivariateNormal(loc=p_c_mean, covariance_matrix=torch.diag_embed(p_c_vars + 1e-6))
        q_dist = D.MultivariateNormal(loc=q_c_mean, covariance_matrix=torch.diag_embed(q_c_vars + 1e-6))
        
        # kl_div_s is the KL for each component: [B, N, NumMixtures]
        kl_div_s = D.kl.kl_divergence(q_dist, p_dist)
        assert torch.isnan(kl_div_s).sum() == 0
        
        # Expectation of KL_s over q(z)
        expected_kl_div_s = torch.sum(q_z * kl_div_s, dim=-1)
        
        # 4. Total KL is the sum of the two parts, averaged over batch and agents
        total_kl_div = torch.mean(kl_div_z + expected_kl_div_s)
        
        return total_kl_div

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
