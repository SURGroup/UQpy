import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes

settings.register_profile("fast", max_examples=1)
settings.load_profile("fast")


@settings(deadline=1_000)
@given(
    batch_size=st.integers(min_value=1, max_value=4),
    width=st.integers(min_value=1, max_value=8),
    signal_shape=array_shapes(min_dims=2, max_dims=2, min_side=64, max_side=128),
    modes=array_shapes(min_dims=2, max_dims=2, min_side=1, max_side=33),
)
def test_output_shape(batch_size, width, signal_shape, modes):
    """Fourier2d takes in a tensor (batch_size, width, H, W) and outputs a tensor of the same shape"""
    H, W = signal_shape
    x = torch.rand((batch_size, width, H, W))
    fourier = sml.Fourier2d(width, modes)
    y = fourier(x)
    assert x.shape == y.shape


def test_extra_repr():
    """Customize all inputs to confirm extra_repr correctly displays non-default configuration"""
    fourier = sml.Fourier2d(width=1, modes=(2, 4), bias=False)
    assert fourier.extra_repr() == "width=1, modes=(2, 4), bias=False"
