"""Unit tests for sml.BayesianConv2d

Note this does not include tests for numerical accuracy as the convolution is performed by torch.nn.functional.conv2d
"""

import torch
from torch.nn.modules.utils import _pair
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


def compute_h_w_out(
    h_in, w_in, kernel_size, stride=(1, 1), padding=(0, 0), dilation=(1, 1)
):
    r"""Compute the final height and width of the output based on the formula for torch.nn.Conv2d

    https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Shape:

    - Input: :math:`(N, C_\text{in}, H_\text{in}, W_\text{in})` or :math:`(C_\text{in}, H_\text{in}, W_\text{in})`
    - Output: :math:`(N, C_\text{out}, H_\text{out}, W_\text{out})` or :math:`(C_\text{out}, H_\text{out}, W_\text{out})`
     where :math:`H_\text{out} = \left\lfloor \frac{H_\text{in} + 2 \times \text{padding[0]} - \text{dilation[0]} \times (\text{kernel\_size[0] - 1}) - 1}{\text{stride[0]}} + 1\right\rfloor`

     :math:`W_\text{out} = \left\lfloor \frac{W_\text{in} + 2 \times \text{padding[1]} - \text{dilation[1]} \times (\text{kernel\_size[1] - 1}) - 1}{\text{stride[1]}} + 1\right\rfloor`

    :return: :math:`H_\text{out}, W_\text{out}`
    """
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)
    h_out = (
        (h_in + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]
    ) + 1
    w_out = (
        (w_in + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]
    ) + 1
    return int(h_out), int(w_out)


@given(
    n=st.integers(min_value=1, max_value=16),
    height=st.integers(min_value=1, max_value=256),
    width=st.integers(min_value=1, max_value=256),
    in_channels=st.integers(min_value=1, max_value=10),
    out_channels=st.integers(min_value=1, max_value=10),
)
def test_default_output_shape(n, height, width, in_channels, out_channels):
    """Test the output shape for various batch sizes, heights, width, and channels"""
    x_size = (n, in_channels, height, width)
    x = torch.rand(size=x_size)
    layer = sml.BayesianConv2d(in_channels, out_channels, kernel_size=1)
    y = layer(x)
    assert y.shape == torch.Size([n, out_channels, height, width])


@given(
    kernel_size=st.one_of(
        st.integers(min_value=1, max_value=8),
        st.tuples(
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=8),
        ),
    ),
    stride=st.one_of(
        st.integers(min_value=1, max_value=8),
        st.tuples(
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=8),
        ),
    ),
    padding=st.one_of(
        st.integers(min_value=0, max_value=6),
        st.tuples(
            st.integers(min_value=0, max_value=6),
            st.integers(min_value=0, max_value=6),
        ),
    ),
    dilation=st.one_of(
        st.integers(min_value=1, max_value=4),
        st.tuples(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=1, max_value=4),
        ),
    ),
)
def test_fancy_output_shape(kernel_size, stride, padding, dilation):
    """Test integer and tuple kernel_sizes, strides, paddings, and dilation"""
    n = 2
    in_channels = 1
    out_channels = 1
    h_in = 512
    w_in = 256
    layer = sml.BayesianConv2d(
        in_channels, out_channels, kernel_size, stride, padding, dilation
    )
    x = torch.rand(size=(n, in_channels, h_in, w_in))
    y = layer(x)
    h_out, w_out = compute_h_w_out(h_in, w_in, kernel_size, stride, padding, dilation)
    assert y.shape == torch.Size([n, out_channels, h_out, w_out])


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    cpu = torch.device("cpu")
    layer = sml.BayesianConv2d(1, 1, 1, device=cpu)
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
    layer = sml.BayesianConv2d(1, 1, 1, dtype=dtype)
    x = torch.rand((1, 1, 1), dtype=dtype)
    y = layer(x)
    assert y.dtype == dtype


def test_deterministic_output():
    x = torch.rand((1, 1, 256, 256))
    layer = sml.BayesianConv2d(1, 1, 1)
    layer.sample(False)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    """When sampling, outputs should be slightly different even on the same input"""
    x = torch.rand((1, 1, 256, 256))
    layer = sml.BayesianConv2d(1, 1, 1)
    layer.sample()
    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_bias_false():
    """When bias=False, BayesianConv2d(0) = 0"""
    x = torch.zeros((1, 1, 256, 256))
    layer = sml.BayesianConv2d(1, 1, 1, bias=False)
    y = layer(x)
    assert torch.all(y == torch.zeros_like(y))
