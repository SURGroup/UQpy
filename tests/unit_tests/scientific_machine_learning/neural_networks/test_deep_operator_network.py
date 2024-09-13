import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


@given(
    n=st.integers(min_value=1, max_value=1),
    m=st.integers(min_value=64, max_value=128),
)
def test_n_m_shape(n, m):
    """Test the output shape with varying batch size (n) and points in the domain (m)"""
    b_in = 1
    t_in = 1
    width = 1
    deep_o_net = sml.DeepOperatorNetwork(
        nn.Linear(b_in, width),
        nn.Linear(t_in, width),
    )
    f_x = torch.rand(n, b_in)
    x = torch.rand(n, m, t_in)
    g_x = deep_o_net(x, f_x)
    assert g_x.shape == torch.Size([n, m, 1])


@given(array_shapes(min_side=1, min_dims=1), st.integers(min_value=1, max_value=8))
def test_same_input_shape(shape, out_channels):
    width = 2 * out_channels
    deep_o_net = sml.DeepOperatorNetwork(
        nn.Linear(shape[-1], width),
        nn.Linear(shape[-1], width),
        out_channels=out_channels,
    )
    x = torch.rand(shape)
    f_x = torch.rand(shape)
    output = deep_o_net(x, f_x)
    assert output.shape == torch.Size((*shape[:-1], out_channels))


@given(
    b_in=st.integers(min_value=1, max_value=8),
    t_in=st.integers(min_value=1, max_value=8),
    out_channels=st.integers(min_value=1, max_value=8),
)
def test_channels_shape(b_in, t_in, out_channels):
    """Test the output shape with varying number of input and output channels"""
    n = 1
    m = 100
    width = 2 * out_channels  # width must be divisible by out_channels
    deep_o_net = sml.DeepOperatorNetwork(
        nn.Linear(b_in, width), nn.Linear(t_in, width), out_channels=out_channels
    )
    x = torch.rand(n, m, t_in)
    f_x = torch.rand(n, b_in)
    g_x = deep_o_net(x, f_x)
    assert g_x.shape == torch.Size([n, m, out_channels])


def test_no_bias():
    """With no bias, DeepOperatorNetwork(x, 0) = 0"""
    n = 1
    m = 100
    b_in = 1
    t_in = 1
    width = 16
    deep_o_net = sml.DeepOperatorNetwork(
        nn.Linear(b_in, width, bias=False),
        nn.Linear(t_in, width, bias=False),
    )
    f_x = torch.zeros(n, b_in)
    x = torch.rand(n, m, t_in)
    g_x = deep_o_net(x, f_x)
    assert torch.all(g_x == torch.zeros([n, m, 1]))
