import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes


@settings(deadline=1_000)
@given(
    batch_size=st.integers(min_value=1, max_value=2),
    width=st.integers(min_value=1, max_value=8),
    signal_shape=array_shapes(min_dims=3, max_dims=3, min_side=64, max_side=128),
    modes=array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=33),
)
def test_output_shape(batch_size, width, signal_shape, modes):
    """Fourier3d takes in a tensor (batch_size, width, H, W, D) and outputs a tensor of the same shape"""
    H, W, D = signal_shape
    x = torch.rand((batch_size, width, H, W, D))
    fourier = sml.Fourier3d(width, modes)
    y = fourier(x)
    assert x.shape == y.shape
