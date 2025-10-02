# Copyright (c) 2025 Juanwu Lu and Purdue Digital Twin Lab
"""Common layers with customized initializations."""
from typing import Any

import torch.nn as nn

from planner.type import size_t_any

from .init import variance_scaling


def embedding(
    num_embeddings: int,
    embedding_dim: int,
    init_scale: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> nn.Embedding:
    """Create an embedding layer with custom initialization.

    Args:
        num_embeddings (int): The number of embeddings.
        embedding_dim (int): The dimension of the embeddings.
        init_scale (float): The scale factor for the initialization.
        *args, **kwargs: Additional arguments for the embedding layer.

    Returns:
        nn.Embedding: The initialized embedding layer.
    """
    layer = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        *args,
        **kwargs,
    )
    if hasattr(layer, "weight") and layer.weight is not None:
        variance_scaling(layer.weight, scale=init_scale, distribution="normal")

    return layer


def layer_norm(
    normalized_shape: size_t_any,
    eps: float = 1e-9,
    *args: Any,
    **kwargs: Any,
) -> nn.LayerNorm:
    """Create a layer normalization layer.

    Args:
        normalized_shape (Tuple[int, ...]): The shape of the input tensor.
        eps (float, optional): The epsilon value for numerical stability.
            Defaults to :math:`1e-9`.
        *args, **kwargs: Additional arguments for the normalization layer.

    Returns:
        nn.LayerNorm: The layer normalization layer.
    """
    layer = nn.LayerNorm(
        normalized_shape=normalized_shape, eps=eps, *args, **kwargs
    )
    if hasattr(layer, "weight") and layer.weight is not None:
        nn.init.ones_(layer.weight)
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.zeros_(layer.bias)

    return layer


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
    layer = nn.Linear(
        in_features=in_features, out_features=out_features, *args, **kwargs
    )
    if hasattr(layer, "weight") and layer.weight is not None:
        variance_scaling(
            tensor=layer.weight, scale=init_scale, distribution="uniform"
        )
    if hasattr(layer, "bias") and layer.bias is not None:
        nn.init.constant_(layer.bias, val=0.0)

    return layer


def lstm(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    batch_first: bool = True,
    bidirectional: bool = False,
    dropout: float = 0.0,
    init_scale: float = 1.0,
    *args: Any,
    **kwargs: Any,
) -> nn.LSTM:
    """Create an LSTM layer with custom initialization.

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The number of hidden units.
        num_layers (int, optional): Number of layers. Defaults to :math:`1`.
        batch_first (bool, optional): Whether the input tensor is batch-first.
            Defaults to ``True``.
        bidirectional (bool, optional): Whether the LSTM is bidirectional.
            Defaults to ``False``.
        dropout (float, optional): The dropout rate. Defaults to :math:`0.0`.
        init_scale (float, optional): The scale factor for the initialization.
            Defaults to :math:`1.0`.
        *args, **kwargs: Additional arguments for the LSTM network.

    Returns:
        nn.LSTM: The initialized LSTM layer.
    """
    layer = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=batch_first,
        bidirectional=bidirectional,
        dropout=dropout,
        *args,
        **kwargs,
    )
    for name, param in layer.named_parameters():
        if "weight_ih" in name:
            variance_scaling(param, scale=init_scale, distribution="normal")
        elif "weight_hh" in name:
            nn.init.orthogonal_(param)
        elif "bias" in name:
            nn.init.zeros_(param)
        else:
            raise ValueError(f"Initialization for {name} is undefined.")

    return layer
