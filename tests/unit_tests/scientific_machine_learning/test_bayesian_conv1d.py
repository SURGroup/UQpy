import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


@given(
    st.integers(min_value=1, max_value=1_000),
    st.integers(min_value=1, max_value=1_000),
    st.integers(min_value=1, max_value=10),
    st.integers(min_value=1, max_value=10),
)
def test_conv1d_shape(n, length, in_channels, out_channels):
    x_size = (n, in_channels, length)
    x = torch.rand(size=x_size)
    layer = sml.BayesianConv1d(in_channels, out_channels, kernel_size=1)
    y = layer(x)
    assert y.shape == torch.Size([n, out_channels, length])
