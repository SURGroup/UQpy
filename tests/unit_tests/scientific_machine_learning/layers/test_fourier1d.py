import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st


@settings(max_examples=10)
@given(
    batch_size=st.integers(min_value=1, max_value=128),
    width=st.integers(min_value=1, max_value=32),
    length=st.integers(min_value=64, max_value=256),
    modes=st.integers(min_value=1, max_value=32),
)
def test_output_shape(batch_size, width, length, modes):
    """Fourier1d takes in a tensor of (batch_size, width, length) and outputs a tensor of the same shape"""
    x = torch.ones((batch_size, width, length))
    fourier = sml.Fourier1d(width, modes)
    y = fourier(x)
    assert x.shape == y.shape
