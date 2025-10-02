from typing import Literal

import torch
import torch.nn as nn


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
