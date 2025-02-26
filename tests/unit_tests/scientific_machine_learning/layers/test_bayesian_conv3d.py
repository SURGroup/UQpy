import torch
from torch.nn.modules.utils import _triple
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes


settings.register_profile("fast", max_examples=1)
settings.load_profile("fast")


def compute_hwl_out(
    h_in,
    w_in,
    l_in,
    kernel_size,
    stride=(1, 1, 1),
    padding=(0, 0, 0),
    dilation=(1, 1, 1),
):
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)
    padding = _triple(padding)
    dilation = _triple(dilation)
    h_out = (
        (h_in + (2 * padding[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]
    ) + 1
    w_out = (
        (w_in + (2 * padding[1]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]
    ) + 1
    l_out = (
        (l_in + (2 * padding[2]) - (dilation[2] * (kernel_size[2] - 1)) - 1) / stride[2]
    ) + 1
    return int(h_out), int(w_out), int(l_out)


@given(
    n=st.integers(min_value=1, max_value=4),
    shape=array_shapes(min_dims=3, max_dims=3, min_side=64, max_side=96),
    in_channels=st.integers(min_value=1, max_value=3),
    out_channels=st.integers(min_value=1, max_value=3),
)
@settings(deadline=1_000)
def test_default_output_shape(n, shape, in_channels, out_channels):
    """Test the output"""
    x = torch.rand((n, in_channels, *shape))
    layer = sml.BayesianConv3d(in_channels, out_channels, 1)
    y = layer(x)
    assert y.shape == torch.Size([n, out_channels, *shape])


@given(
    kernel_size=st.one_of(
        st.integers(min_value=1, max_value=8),
        st.tuples(
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=8),
        ),
    ),
    stride=st.one_of(
        st.integers(min_value=1, max_value=8),
        st.tuples(
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=8),
            st.integers(min_value=1, max_value=8),
        ),
    ),
    padding=st.one_of(
        st.integers(min_value=0, max_value=6),
        st.tuples(
            st.integers(min_value=0, max_value=6),
            st.integers(min_value=0, max_value=6),
            st.integers(min_value=0, max_value=6),
        ),
    ),
    dilation=st.one_of(
        st.integers(min_value=1, max_value=4),
        st.tuples(
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=1, max_value=4),
            st.integers(min_value=1, max_value=4),
        ),
    ),
)
def test_fancy_output_shape(kernel_size, stride, padding, dilation):
    n = 1
    in_channels = 1
    out_channels = 1
    h_in = 64
    w_in = 64
    l_in = 64
    layer = sml.BayesianConv3d(
        in_channels, out_channels, kernel_size, stride, padding, dilation
    )
    h_out, w_out, l_out = compute_hwl_out(
        h_in, w_in, l_in, kernel_size, stride, padding, dilation
    )
    x = torch.rand(size=(n, in_channels, h_in, w_in, l_in))
    y = layer(x)
    assert y.shape == torch.Size([n, out_channels, h_out, w_out, l_out])


def test_device():
    """Note if neither cuda nor mps is available this test will always pass"""
    cpu = torch.device("cpu")
    layer = sml.BayesianConv3d(1, 1, 1, device=cpu)
    assert layer.weight_mu.device == cpu
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")
    layer.to(device)
    assert layer.weight_mu.device == device


def test_deterministic_output():
    x = torch.rand(1, 1, 256, 256, 256)
    layer = sml.BayesianConv3d(1, 1, 1)
    layer.sample(False)
    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    x = torch.rand(1, 1, 256, 256, 256)
    layer = sml.BayesianConv3d(1, 1, 1)
    layer.sample()
    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_bias_false():
    """When bias=False, BayesianConv3d(0) = 0"""
    x = torch.zeros((1, 1, 256, 256, 256))
    layer = sml.BayesianConv3d(1, 1, 1, bias=False)
    y = layer(x)
    assert torch.all(y == torch.zeros_like(y))


def test_extra_repr():
    """Test custom inputs to confirm extra_repr correctly displays non-default inputs"""
    kwargs = {
        "kernel_size": (1, 1, 1),
        "stride": (2, 2, 2),
        "padding": "same",
        "dilation": (2, 2, 2),
        "groups": 2,
        "bias": False,
        "sampling": False,
        "prior_mu": 1.0,
        "prior_sigma": 1.1,
        "posterior_mu_initial": (1.0, 2.0),
        "posterior_rho_initial": (-4.0, 2.0)
    }
    kwargs_str = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    correct_extra_repr = f"4, 6, {kwargs_str}"
    layer = sml.BayesianConv3d(4, 6, **kwargs)
    assert layer.extra_repr() == correct_extra_repr
