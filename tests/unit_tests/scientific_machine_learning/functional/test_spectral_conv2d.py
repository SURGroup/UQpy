import pytest
import torch
import UQpy.scientific_machine_learning.functional as func
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


@given(
    batch_size=st.integers(min_value=1, max_value=100),
    in_channels=st.integers(min_value=1, max_value=10),
    out_channels=st.integers(min_value=1, max_value=10),
    signal_shape=array_shapes(min_dims=2, max_dims=2, min_side=64, max_side=128),
    modes=array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=32),
)
def test_output_shape(batch_size, in_channels, out_channels, signal_shape, modes):
    """An input (batch_size, in_channels, height, width) has an output (batch_size, out_channels, height, width)
    Note modes1 and modes 2 does *not* affect the shape of the output
    """
    height, width = signal_shape
    x = torch.rand((batch_size, in_channels, height, width))
    weights = (
        torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat),
        torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat),
    )
    y = func.spectral_conv2d(x, weights, out_channels, modes)
    assert y.shape == torch.Size([batch_size, out_channels, height, width])


def test_invalid_modes0_raises_error():
    """If modes[0] > (height // 2) + 1, raise an error"""
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 64
    width = 32
    modes = (height, width // 2)
    x = torch.rand((batch_size, in_channels, height, width))
    weights = (
        torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat),
        torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat),
    )
    with pytest.raises(ValueError):
        func.spectral_conv2d(x, weights, out_channels, modes)


def test_invalid_modes1_raises_error():
    """If modes[1] > (width // 2) + 1, raise an error"""
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 64
    width = 32
    modes = (height // 2, width)
    x = torch.rand((batch_size, in_channels, height, width))
    weights = (
        torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat),
        torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat),
    )
    with pytest.raises(ValueError):
        func.spectral_conv2d(x, weights, out_channels, modes)


@pytest.mark.parametrize(
    "weight_shape", [(2, 1, 1, 1), (1, 2, 1, 1), (1, 1, 2, 1), (1, 1, 1, 2)]
)
def test_invalid_weights_shape_raises_error(weight_shape):
    """If either weight matrix is not of shape (in_channels, out_channels, modes[0], modes[1]), raise an error
    Note the correct weight shape for this example is (1, 1, 1, 1)
    """
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 64
    width = 32
    modes = (1, 1)
    x = torch.rand((batch_size, in_channels, height, width))
    weight_good = torch.rand((in_channels, out_channels, *modes), dtype=torch.cfloat)
    weight_bad = torch.rand(weight_shape, dtype=torch.cfloat)
    with pytest.raises(AssertionError):
        func.spectral_conv2d(x, (weight_good, weight_bad), out_channels, modes)
    with pytest.raises(AssertionError):
        func.spectral_conv2d(x, (weight_bad, weight_good), out_channels, modes)
