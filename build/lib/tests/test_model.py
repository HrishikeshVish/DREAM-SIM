import sys

import pytest
import torch
import torch.nn as nn

from planner.model.module.layers import layer_norm, linear


# =============================================================================
# Test Building Layers
@pytest.mark.parametrize("batch_size", [1, 10])
@pytest.mark.parametrize("in_features", [1, 10])
@pytest.mark.parametrize("out_features", [1, 10])
def test_linear(batch_size: int, in_features: int, out_features: int) -> None:
    test_input = torch.ones((batch_size, in_features))

    # test: create a layer with bias
    layer = linear(in_features=in_features, out_features=out_features)
    assert isinstance(layer, nn.Linear)
    assert layer.in_features == in_features
    assert layer.out_features == out_features
    assert layer.weight.shape == (out_features, in_features)
    assert layer.bias.shape == (out_features,)
    test_output = layer(test_input)
    assert isinstance(test_output, torch.Tensor)
    assert test_output.shape == (batch_size, out_features)

    # test: create a layer without bias
    layer = linear(
        in_features=in_features, out_features=out_features, bias=False
    )
    assert isinstance(layer, nn.Linear)
    assert layer.in_features == in_features
    assert layer.out_features == out_features
    assert layer.weight.shape == (out_features, in_features)
    assert layer.bias is None
    test_output = layer(test_input)
    assert isinstance(test_output, torch.Tensor)
    assert test_output.shape == (batch_size, out_features)


@pytest.mark.parametrize("in_features", [1, 10])
def test_layer_norm(in_features: int) -> None:
    test_input = torch.ones((1, in_features))

    # test: create a layer with element-wise normalization
    layer = layer_norm(normalized_shape=in_features)
    assert isinstance(layer, nn.LayerNorm)
    assert layer.normalized_shape == (in_features,)
    assert torch.allclose(layer.weight, torch.tensor(1.0))
    assert torch.allclose(layer.bias, torch.tensor(0.0))
    test_output = layer(test_input)
    assert isinstance(test_output, torch.Tensor)

    # test: create a layer with feature-wise normalization
    layer = layer_norm(normalized_shape=in_features, elementwise_affine=False)
    assert isinstance(layer, nn.LayerNorm)
    assert layer.normalized_shape == (in_features,)
    assert layer.weight is None
    assert layer.bias is None
    test_output = layer(test_input)
    assert isinstance(test_output, torch.Tensor)


if __name__ == "__main__":
    sys.exit(pytest.main(["-arqv", __file__]))
