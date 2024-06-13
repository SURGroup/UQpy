import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


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


def get_spectral_convolution(y, modes):
    """Compute the spectral convolution of y with a given number of modes and weights of all ones"""
    spectral_conv = sml.SpectralConv1d(1, 1, modes)
    weights = torch.ones(1, 1, modes)
    spectral_conv.weights = nn.Parameter(weights)
    return spectral_conv(y)


def test_cosine_1mode():
    """Keeping one mode in the FFT leads to an output of all zeros"""
    x = torch.linspace(0, 2 * torch.pi, 10_000, requires_grad=False).reshape(1, 1, -1)
    y = torch.cos(x)
    f_y = get_spectral_convolution(y, modes=1)
    assert torch.allclose(f_y, torch.zeros_like(f_y), atol=1e-3)


def test_cosine_2mode():
    """Keeping 2 modes in the FFT captures the entire cosine"""
    x = torch.linspace(0, 2 * torch.pi, 10_000, requires_grad=False).reshape(1, 1, -1)
    y = torch.cos(x)
    f_y = get_spectral_convolution(y, modes=2)
    assert torch.allclose(f_y, y, atol=1e-3)


def test_sine_cosine_2mode():
    """Keeping 2 modes leads to an output of all zeros"""
    x = torch.linspace(0, 2 * torch.pi, 10_000, requires_grad=False).reshape(1, 1, -1)
    y = torch.sin(x) * torch.cos(x)
    f_y = get_spectral_convolution(y, modes=2)
    assert torch.allclose(f_y, torch.zeros_like(f_y), atol=1e-3)


def test_sine_cosine_3mode():
    """Keeping 3 modes captures the entire signal"""
    x = torch.linspace(0, 2 * torch.pi, 10_000, requires_grad=False).reshape(1, 1, -1)
    y = torch.sin(x) * torch.cos(x)
    f_y = get_spectral_convolution(y, modes=3)
    assert torch.allclose(f_y, y, atol=1e-3)
