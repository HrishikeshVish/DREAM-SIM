import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, Distribution

from planner.model.module.init import variance_scaling
from planner.model.module.layers import layer_norm, linear
from planner.model.module.rst import RSTEncoder

from typing import Optional, Tuple

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
    Models the posterior distribution q(c|x,y).
    It infers a distribution for the latent constraint `c` given both historical
    context `x` and the ground-truth future trajectory `y`.
    """
    def __init__(self, hidden_size: int, latent_c_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_c_dim = latent_c_dim
        
        # Layer to combine history and future embeddings
        self.fusion_layer = nn.Linear(hidden_size * 2, hidden_size)

        # Attention block to fuse with map context
        self.attention_block = HistoryEncoderBlock(hidden_size, num_heads=num_heads)
        
        # Two linear heads for the Gaussian parameters
        self.mean_head = nn.Linear(hidden_size, latent_c_dim)
        self.log_var_head = nn.Linear(hidden_size, latent_c_dim)

    def forward(self, x_hist: torch.Tensor, x_future_gt: torch.Tensor, x_map: torch.Tensor) -> Distribution:
        """
        Args:
            x_hist: Encoded agent history. Shape: [B, N, HiddenSize]
            x_future_gt: Encoded ground-truth future. Shape: [B, N, HiddenSize]
            x_map: Encoded map features. Shape: [B, NumMapPoints, HiddenSize]

        Returns:
            A torch.distributions object representing the distribution over c.
        """
        # 1. Fuse the history and future embeddings
        hist_future_combined = torch.cat([x_hist, x_future_gt], dim=-1)
        fused_trajectory_embedding = F.relu(self.fusion_layer(hist_future_combined))
        
        # 2. Use cross-attention to incorporate map context
        fused_context = self.attention_block.cross_attention(
            x=fused_trajectory_embedding,
            context=x_map
        )

        # 3. Project to the mean and log-variance of the latent c
        mean = self.mean_head(fused_context)
        log_var = self.log_var_head(fused_context)

        return Independent(Normal(loc=mean, scale=torch.exp(0.5 * log_var)), 1)
    

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
    Models the prior distribution p(c|x).
    It predicts a distribution for the latent constraint `c` given only historical context.
    """
    def __init__(self, hidden_size: int, latent_c_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_c_dim = latent_c_dim

        # Attention block to fuse map and agent history information
        self.attention_block = HistoryEncoderBlock(hidden_size, num_heads=num_heads)
        
        # Two linear heads to output the parameters of a Gaussian distribution
        self.mean_head = nn.Linear(hidden_size, latent_c_dim)
        self.log_var_head = nn.Linear(hidden_size, latent_c_dim)

    def forward(self, x_hist: torch.Tensor, x_map: torch.Tensor) -> Distribution:
        """
        Args:
            x_hist: Encoded agent history. Shape: [B, N, HiddenSize]
            x_map: Encoded map features. Shape: [B, NumMapPoints, HiddenSize]

        Returns:
            A torch.distributions object representing the distribution over c for each agent.
        """
        # We use the agent history as the query and the map as the key/value context
        # This lets each agent "look at" the map to inform its prior constraint distribution
        fused_context = self.attention_block.cross_attention(
            x=x_hist,
            context=x_map
        )
        
        # Project the fused context to the mean and log-variance of the latent c
        mean = self.mean_head(fused_context)
        log_var = self.log_var_head(fused_context)

        # We use Independent to create a batch of diagonal Gaussians from the means and vars
        # This treats each dimension of `c` as independent for a given agent
        return Independent(Normal(loc=mean, scale=torch.exp(0.5 * log_var)), 1)