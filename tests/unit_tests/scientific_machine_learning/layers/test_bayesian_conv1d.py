"""Unit tests for sml.BayesianConv1d

Note this does not include tests for numerical accuracy as the convolution is performed by torch.nn.functional.conv1d
"""

import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


def compute_l_out(l_in, kernel_size, stride, padding, dilation):
    l_out = ((l_in + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride) + 1
    return int(l_out)


@given(
    n=st.integers(min_value=1, max_value=1_000),
    length=st.integers(min_value=1, max_value=1_000),
    in_channels=st.integers(min_value=1, max_value=10),
    out_channels=st.integers(min_value=1, max_value=10),
)
def test_default_output_shape(n, length, in_channels, out_channels):
    """Test the output shape for various batch sizes, lengths, and channels"""
    x_size = (n, in_channels, length)
    x = torch.rand(size=x_size)
    layer = sml.BayesianConv1d(in_channels, out_channels, kernel_size=1)
    y = layer(x)
    assert y.shape == torch.Size([n, out_channels, length])


@given(
    kernel_size=st.integers(min_value=1, max_value=8),
    stride=st.integers(min_value=1, max_value=8),
    padding=st.integers(min_value=0, max_value=6),
    dilation=st.integers(min_value=1, max_value=4),
)
def test_fancy_output_shape(kernel_size, stride, padding, dilation):
    """Test the output shape for various kernels, strides, paddings, and dilation"""
    n = 2
    in_channels = 1
    out_channels = 1
    length_in = 256
    layer = sml.BayesianConv1d(
        in_channels, out_channels, kernel_size, stride, padding, dilation
    )
    x = torch.rand((n, in_channels, length_in))
    y = layer(x)
    length_out = compute_l_out(length_in, kernel_size, stride, padding, dilation)
    assert y.shape == torch.Size([n, out_channels, length_out])


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    cpu = torch.device("cpu")
    layer = sml.BayesianConv1d(1, 1, 1, device=cpu)
    assert layer.weight_mu.device == cpu
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("mps", 0) if torch.backends.mps.is_available() else "cpu"
    )
    layer.to(device)
    assert layer.weight_mu.device == device


def test_dtype():
    dtype = torch.cfloat
    layer = sml.BayesianConv1d(1, 1, 1, dtype=dtype)
    x = torch.rand((1, 1), dtype=dtype)
    y = layer(x)
    assert y.dtype == dtype


def test_deterministic_output():
    x = torch.rand(1, 1, 256)
    layer = sml.BayesianConv1d(1, 1, 1)
    layer.eval()
    layer.sample(False)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    x = torch.rand(1, 1, 256)
    layer = sml.BayesianConv1d(1, 1, 1)
    layer.eval()
    layer.sample()
    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)
