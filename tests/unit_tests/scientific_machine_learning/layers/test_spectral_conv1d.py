import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


@given(
    n_batch=st.integers(min_value=1, max_value=100),
    in_channels=st.integers(min_value=1, max_value=10),
    out_channels=st.integers(min_value=1, max_value=10),
    length=st.integers(min_value=64, max_value=128),
    modes=st.integers(min_value=1, max_value=32),
)
def test_output_shape(n_batch, in_channels, out_channels, length, modes):
    """An input of shape (n_batch, in_channels, n_x) has an output of shape (n_batch, out_channels, n_x)
    Note modes does *not* affect the shape of the output
    """
    spectral_conv = sml.SpectralConv1d(in_channels, out_channels, modes)

    x = torch.zeros((n_batch, in_channels, length))
    y = spectral_conv(x)
    assert y.shape == torch.Size([n_batch, out_channels, length])


def test_zero_input():
    """An input of all zeros should produce an output of all zeros"""
    n_batch = 12
    in_channels = 3
    out_channels = 2
    modes = 16
    n_x = 128
    spectral_conv = sml.SpectralConv1d(in_channels, out_channels, modes)

    x = torch.zeros((n_batch, in_channels, n_x))
    y = spectral_conv(x)
    assert torch.all(y == torch.zeros((n_batch, out_channels, n_x)))
