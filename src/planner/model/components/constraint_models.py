import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Distribution

from planner.model.module.init import variance_scaling
from planner.model.module.layers import layer_norm, linear
from planner.model.module.rst import RSTEncoder

from typing import Optional, Tuple, Literal, Any

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

class HistoryEncoderBlock(nn.Module):
    """Repeated self-attention block for the history encoder."""

    def __init__(
        self,
        hidden_size: int,
        dropout: float = 0.1,
        num_heads: Optional[int] = None,
        init_scale: float = 0.2,
    ) -> None:
        super().__init__()

        # save the arguments
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.init_scale = init_scale
        self.num_heads = num_heads
        if self.num_heads is None:
            self.num_heads = hidden_size // 64

        # build the attention and feed-forward layers
        self.attn_norm = layer_norm(normalized_shape=hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            dropout=self.dropout,
            num_heads=self.num_heads,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(p=self.dropout)
        self.ffn_norm = layer_norm(normalized_shape=self.hidden_size)
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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of the history encoder block.

        Args:
            x (torch.Tensor): The input feature tensor of shape `(*, T, E)`.
            attn_mask (Optional[torch.Tensor], optional): Optional mask for
                attention weights. Defaults to ``None``.
            key_padding_mask (Optional[torch.Tensor], optional): Optional
                valid mask for keys. Defaults to ``None``.

        Returns:
            torch.Tensor: The output feature tensor of shape `(*, T, E)`.
        """
        out = self.attn_norm.forward(x)
        out = out + self._sa_forward(
            x=out, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        out = out + self._ffn_forward(x=out)

        return out

    def reset_parameters(self) -> None:
        """Reset the parameters of the attention block."""
        if self.attn.in_proj_weight is not None:
            variance_scaling(self.attn.in_proj_weight, scale=self.init_scale)
        else:
            variance_scaling(self.attn.q_proj_weight, scale=self.init_scale)
            variance_scaling(self.attn.k_proj_weight, scale=self.init_scale)
            variance_scaling(self.attn.v_proj_weight, scale=self.init_scale)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        variance_scaling(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)
        if self.attn.bias_k is not None:
            nn.init.zeros_(self.attn.bias_k)
        if self.attn.bias_v is not None:
            nn.init.zeros_(self.attn.bias_v)

    def _sa_forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out, _ = self.attn.forward(
            query=x,
            key=x,
            value=x,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=False,
        )
        return self.attn_dropout.forward(out)

    def _ffn_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn.forward(self.ffn_norm.forward(x))

class ConstraintPosteriorNetwork(nn.Module):
    """
    Models the posterior q(c|x,y) by encoding a PHYSICAL ATTRIBUTE tensor.

    This network acts like an executive reading a summary report. Instead of seeing
    the full, raw future trajectory, it sees a concise summary of its physical
    properties (collision scores, off-road scores, etc.) and infers the latent
    "cause" or "intent" `c` that explains them.
    """
    def __init__(self, hidden_size: int, latent_c_dim: int, num_attributes: int, num_mixtures: int = 1):
        """
        Args:
            hidden_size: The main embedding dimension of the model.
            latent_c_dim: The desired dimensionality of the latent constraint vector c.
            num_attributes: The number of features in the input attribute_tensor.
        """
        super().__init__()

        self.num_mixtures = num_mixtures
        
        # An MLP to process the attribute tensor and project it into the hidden space.
        self.attribute_encoder = nn.Sequential(
            linear(num_attributes, hidden_size // 2),
            nn.ReLU(),
            linear(hidden_size // 2, hidden_size)
        )
        
        # A layer to fuse the encoded history (`x`) with the encoded attributes (`y`).
        self.fusion_layer = nn.Sequential(
            linear(hidden_size * 2, hidden_size),
            nn.ReLU()
        )

        # Final linear heads to output the parameters of a Gaussian distribution.
        self.mean_head = linear(hidden_size, num_mixtures*latent_c_dim, init_scale=1e-10)
        self.log_var_head = nn.Sequential(linear(hidden_size, num_mixtures*latent_c_dim, init_scale=1e-10), nn.Softplus())

    def forward(self, x_hist: torch.Tensor, attribute_tensor: torch.Tensor) -> Distribution:
        """
        Args:
            x_hist: Encoded agent history. Shape: [B, N, HiddenSize].
            attribute_tensor: Tensor of physical attributes from y_gt.
                              Shape: [B, N, T, NumAttributes].
        Returns:
            A torch.distributions object representing the posterior distribution q(c|x,y).
        """
        # 1. Summarize the attributes over the time dimension.
        #    We take the mean to get a single descriptive vector per agent.
        agg_attributes = torch.mean(attribute_tensor, dim=2) # Shape: [B, N, NumAttributes]
        
        # 2. Encode the summarized attributes into the hidden space.
        encoded_attributes = self.attribute_encoder(agg_attributes) # Shape: [B, N, HiddenSize]

        # 3. Fuse the history embedding with the attribute embedding.
        combined_embedding = torch.cat([x_hist, encoded_attributes], dim=-1) # Shape: [B, N, HiddenSize * 2]
        fused_embedding = self.fusion_layer(combined_embedding) # Shape: [B, N, HiddenSize]

        # 4. Project the fused embedding to the Gaussian parameters for `c`.
        mean = self.mean_head(fused_embedding)
        log_var = self.log_var_head(fused_embedding)

        B, N, _ = mean.shape
        mean = mean.view(B, N, self.num_mixtures, -1)
        log_var = torch.add(log_var.view(B, N, self.num_mixtures, -1), torch.finfo(x_hist.dtype).eps)

        # 5. Return the final distribution object.
        return mean, log_var
    
class FiLMLayer(nn.Module):
    """A Feature-wise Linear Modulation (FiLM) layer."""
    def __init__(self, channels: int, cond_channels: int):
        """
        Args:
            channels: The number of feature channels in the main tensor `x`.
            cond_channels: The number of channels in the conditioning tensor `cond`.
        """
        super().__init__()
        # A linear layer to predict the scaling factors (gammas)
        self.gammas = nn.Linear(cond_channels, channels)
        # A linear layer to predict the shifting factors (betas)
        self.betas = nn.Linear(cond_channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Applies the FiLM conditioning.

        Args:
            x: The main feature tensor. Shape: [..., channels]
            cond: The conditioning vector. Shape: [..., cond_channels]

        Returns:
            The modulated feature tensor. Shape: [..., channels]
        """
        gammas = self.gammas(cond)
        betas = self.betas(cond)
        
        # The core FiLM operation: scale and shift
        return gammas * x + betas

class PotentialFieldNetwork(nn.Module):
    """
    Learns the potential field Φ(Y, c).
    Outputs a scalar potential (energy) for a given multi-agent trajectory Y,
    conditioned on a latent constraint embedding c.
    """
    def __init__(self, hidden_size: int, latent_c_dim: int, traj_state_dim: int, num_encoder_layers: int = 3, num_heads: int = 8):
        super().__init__()
        
        # A tokenizer to project the raw trajectory state into the hidden dimension
        self.traj_tokenizer = nn.Linear(traj_state_dim, hidden_size)
        
        # A FiLM layer to condition the trajectory tokens on the constraint `c`
        self.film = FiLMLayer(hidden_size, latent_c_dim)
        
        # A standard Transformer Encoder to process the spatio-temporal information
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        # Final MLP head to map the aggregated representation to a single scalar
        self.mlp_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, Y: torch.Tensor, c_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            Y: The full future trajectory. Shape: [B, N, T, D]
            c_embed: The sampled latent constraint. Shape: [B, N, latent_c_dim]

        Returns:
            A scalar potential value for each scene in the batch. Shape: [B, 1]
        """
        B, N, T, D = Y.shape
        
        # 1. Tokenize the trajectory: [B, N, T, D] -> [B, N, T, HiddenSize]
        traj_tokens = self.traj_tokenizer(Y)
        
        # 2. Condition tokens with FiLM
        #    We need to expand c_embed to match the time dimension
        c_embed = c_embed.squeeze(2) # Shape is now [4, 1, 32]
        c_expanded = c_embed.unsqueeze(2).expand(-1, -1, T, -1)
        conditioned_tokens = self.film(traj_tokens, c_expanded)
        
        # 3. Process with Transformer
        #    Flatten the agent and time dimensions to create a sequence for each scene
        #    Shape: [B, N*T, HiddenSize]
        sequence = conditioned_tokens.view(B, N * T, -1)
        processed_sequence = self.transformer_encoder(sequence)
        
        # 4. Aggregate the sequence information (e.g., mean pooling)
        #    Shape: [B, HiddenSize]
        agg_representation = processed_sequence.mean(dim=1)
        
        # 5. Get the final scalar potential from the MLP head
        #    Shape: [B, 1]
        potential = self.mlp_head(agg_representation)
        
        return potential
    
class ConstraintPriorNetwork(nn.Module):
    """
    A lightweight head that models the prior p(c|x) by fusing history and
    a global map vector.
    """
    def __init__(self, hidden_size: int, latent_c_dim: int, num_mixtures: int = 1, num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_c_dim = latent_c_dim
        self.num_mixtures = num_mixtures

        # An MLP that processes the FUSED history and map representation.
        # The input size is hidden_size * 2 to account for concatenation.
        fused_mlp_size = hidden_size * 2
        self.fused_mlp = nn.Sequential(
            linear(fused_mlp_size, hidden_size),
            nn.ReLU()
        )
        
        # Final MLP heads to project to the distribution parameters
        self.mean_head = linear(hidden_size, num_mixtures * latent_c_dim, init_scale=1e-10)
        self.log_var_head = nn.Sequential(linear(hidden_size, num_mixtures * latent_c_dim, init_scale=1e-10), nn.Softplus())

    def forward(self, x_hist: torch.Tensor, x_map: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_hist: Encoded agent history. Shape: [B, N, HiddenSize].
            x_map: Encoded map features. Shape: [B, NumMapPoints, HiddenSize].
        """
        B, N, _ = x_hist.shape

        # 1. Create a single global map vector by pooling the map points.
        map_vector = torch.mean(x_map, dim=1, keepdim=True) # Shape: [B, 1, HiddenSize]
        
        # 2. Expand the global map vector to match the number of agents.
        map_vector_expanded = map_vector.expand(-1, N, -1) # Shape: [B, N, HiddenSize]

        # 3. Concatenate with the per-agent history embeddings.
        fused_input = torch.cat([x_hist, map_vector_expanded], dim=-1) # Shape: [B, N, HiddenSize * 2]

        # 4. Process with the MLP.
        fused_context = self.fused_mlp(fused_input) # Shape: [B, N, HiddenSize]

        # 5. Project to the distribution parameters.
        mean = self.mean_head(fused_context)
        log_var = self.log_var_head(fused_context)

        # Reshape for the Mixture of Gaussians output
        mean = mean.view(B, N, self.num_mixtures, -1)
        log_var = torch.add(log_var.view(B, N, self.num_mixtures, -1), torch.finfo(x_hist.dtype).eps)

        return mean, torch.exp(log_var)