import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


@given(
    st.integers(min_value=1, max_value=100),
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=256),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
)
def test_conv2d_shape(n, height, width, in_channels, out_channels):
    x_size = (n, in_channels, height, width)
    x = torch.rand(size=x_size)
    layer = sml.BayesianConv2d(in_channels, out_channels, kernel_size=1)
    y = layer(x)
    assert y.shape == torch.Size([n, out_channels, height, width])
