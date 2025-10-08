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

class FiLMLayer(nn.Module):
    """A Feature-wise Linear Modulation layer."""
    def __init__(self, channels: int, cond_channels: int):
        super().__init__()
        self.gammas = nn.Linear(cond_channels, channels)
        self.betas = nn.Linear(cond_channels, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gammas = self.gammas(cond)
        betas = self.betas(cond)
        return gammas * x + betas


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
        latent_c_dim: int = 32,
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
        self.sigma_data = 0.5
        self.latent_c_dim = latent_c_dim
        self.film_layer = FiLMLayer(hidden_size, latent_c_dim)
        

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
        constraint_embed: torch.Tensor,
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

        B = tar_valid.shape[0] # Original batch size
        # First, let's get the constraint embedding into a clean [B, L, C] shape.
        # Your shape [4, 1, 1, 32] suggests it might be a global constraint per scene.
        # We'll squeeze the unnecessary dimensions.
        constraint_embed = constraint_embed.squeeze(2)
        B, L, C = constraint_embed.shape
        k = self.num_intentions
        H = self.hidden_size
        c_with_intent_dim = constraint_embed.unsqueeze(2)
        c_expanded_intents = c_with_intent_dim.expand(-1, -1, k, -1)
        c_final_shape = c_expanded_intents.reshape(B, L * k, C)
        query = self.film_layer(query, c_final_shape)


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

        s_mean, s_var = self._gen_forward(x=query, horizon=horizon, emission_head=emission_head, context=context, map_context=map_context,
                                          tar_valid=tar_valid, assigned_types=assigned_types, scenario=scenario, global_step = global_step, batch_dims=batch_dims)
        s_mean = s_mean.reshape(*batch_dims, horizon, self.hidden_size)
        s_var = s_var.reshape(*batch_dims, horizon, self.hidden_size)
        #print("S MEAN RESHAPED: ", s_mean.shape)


        # forward pass the z-proxy network
        z_logits = self.z_proxy.forward(query)
        z_logits = z_logits.reshape(*batch_dims)

        return s_mean, s_var, z_logits

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

            assert torch.isnan(s_mean).sum() == 0
            s_vars[..., t, :] = nn.functional.softplus(
                self.generative_state_var.forward(out)
            )
        return s_mean, s_vars