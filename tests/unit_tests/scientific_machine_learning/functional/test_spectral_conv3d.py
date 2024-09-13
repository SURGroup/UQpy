import pytest
import torch
import UQpy.scientific_machine_learning.functional as func
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes


@settings(deadline=1_000)
@given(
    batch_size=st.integers(min_value=1, max_value=10),
    in_channels=st.integers(min_value=1, max_value=3),
    out_channels=st.integers(min_value=1, max_value=3),
    signal_shape=array_shapes(min_dims=3, max_dims=3, min_side=64, max_side=128),
    modes=array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=32),
)
def test_output_shape(batch_size, in_channels, out_channels, signal_shape, modes):
    """An input (batch_size, in_channels, H, W, D) has an output (batch_size, out_channels, H, W, D)
    Note modes do *not* affect the shape of the output
    """
    height, width, depth = signal_shape
    x = torch.rand((batch_size, in_channels, height, width, depth))
    weight_shape = (4, in_channels, out_channels, *modes)
    weights = torch.rand(weight_shape, dtype=torch.cfloat)
    y = func.spectral_conv3d(x, weights, out_channels, modes)
    assert y.shape == torch.Size([batch_size, out_channels, height, width, depth])


def test_invalid_modes0_raises_error():
    """If modes[0] > (height // 2) + 1, raise an error"""
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 32
    width = 64
    depth = 128
    modes = (height, width // 2, depth // 2)
    x = torch.rand(batch_size, in_channels, height, width, depth)
    weights = torch.rand(4, in_channels, out_channels, *modes, dtype=torch.cfloat)
    with pytest.raises(ValueError):
        func.spectral_conv3d(x, weights, out_channels, modes)


def test_invalid_modes1_raises_error():
    """If modes[1] > (width // 2) + 1, raise an error"""
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 32
    width = 64
    depth = 128
    modes = (height // 2, width, depth // 2)
    x = torch.rand(batch_size, in_channels, height, width, depth)
    weights = torch.rand(4, in_channels, out_channels, *modes, dtype=torch.cfloat)
    with pytest.raises(ValueError):
        func.spectral_conv3d(x, weights, out_channels, modes)


def test_invalid_modes2_raises_error():
    """If modes[2] > (depth // 2) + 1, raise an error"""
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 32
    width = 64
    depth = 128
    modes = (height // 2, width // 2, depth)
    x = torch.rand(batch_size, in_channels, height, width, depth)
    weights = torch.rand(4, in_channels, out_channels, *modes, dtype=torch.cfloat)
    with pytest.raises(ValueError):
        func.spectral_conv3d(x, weights, out_channels, modes)


def test_invalid_weights_shape_raises_error():
    """If weights is not of shape (4, in_channels, out_channels, modes[0], modes[1], modes[2]), raise an error
    Note the correct weight shape for this example is (4, 1, 1, 1, 1, 1)
    """
    batch_size, in_channels, out_channels = 1, 1, 1
    height = 64
    width = 128
    depth = 256
    modes = (1, 1, 1)
    x = torch.rand((batch_size, in_channels, height, width, depth))
    weights = torch.rand((42, in_channels, out_channels, *modes), dtype=torch.cfloat)
    with pytest.raises(RuntimeError):
        func.spectral_conv3d(x, weights, out_channels, modes)
