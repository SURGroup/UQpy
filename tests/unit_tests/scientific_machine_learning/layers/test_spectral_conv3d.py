import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes


@settings(deadline=1_000)
@given(
    batch_size=st.integers(min_value=1, max_value=1),
    in_channels=st.integers(min_value=1, max_value=10),
    out_channels=st.integers(min_value=1, max_value=10),
    signal_shape=array_shapes(min_dims=3, max_dims=3, min_side=64, max_side=128),
    modes=array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=33),
)
def test_output_shape(batch_size, in_channels, out_channels, signal_shape, modes):
    """An input (batch_size, in_channels, H, W, D) as an output (batch_size, out_channels, H, W, D)"""
    height, width, depth = signal_shape
    x = torch.rand(batch_size, in_channels, height, width, depth)
    spectral_conv = sml.SpectralConv3d(in_channels, out_channels, modes)
    y = spectral_conv(x)
    assert y.shape == torch.Size([batch_size, out_channels, height, width, depth])