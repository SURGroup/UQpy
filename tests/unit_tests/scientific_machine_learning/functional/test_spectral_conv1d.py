import pytest
import torch
import UQpy.scientific_machine_learning.functional as func
from hypothesis import given, strategies as st


@given(
    batch_size=st.integers(min_value=1, max_value=100),
    in_channels=st.integers(min_value=1, max_value=10),
    out_channels=st.integers(min_value=1, max_value=10),
    length=st.integers(min_value=64, max_value=128),
    modes=st.integers(min_value=2, max_value=32),
)
def test_output_shape(batch_size, in_channels, out_channels, length, modes):
    """An input of shape (batch_size, in_channels, length) has an output of shape (batch_size, out_channels, length)
    Note modes does *not* affect the shape of the output
    """
    x = torch.rand((batch_size, in_channels, length))
    weights = torch.rand((in_channels, out_channels, modes), dtype=torch.cfloat)
    y = func.spectral_conv1d(x, weights, out_channels, modes)
    assert y.shape == torch.Size([batch_size, out_channels, length])


def test_invalid_modes_raises_error():
    """If modes > (length // 2) + 1, raise an error"""
    batch_size = 1
    in_channels = 1
    out_channels = 1
    length = 16
    modes = 2 * length
    x = torch.rand((batch_size, in_channels, length))
    weights = torch.rand((in_channels, out_channels, modes), dtype=torch.cfloat)
    with pytest.raises(ValueError):
        func.spectral_conv1d(x, weights, out_channels, modes)


def test_cosine_1mode():
    """Keeping one mode in the FFT leads to an output of all zeros"""
    in_channels = 1
    out_channels = 1
    length = 10_000
    modes = 1
    x = torch.linspace(0, 2 * torch.pi, length, requires_grad=False).reshape(
        1, in_channels, length
    )
    y = torch.cos(x)
    weights = torch.ones(in_channels, out_channels, modes, dtype=torch.cfloat)
    f = func.spectral_conv1d(y, weights, out_channels, modes)
    assert torch.allclose(f, torch.zeros_like(f), atol=1e-3)


def test_cosine_2mode():
    """Keeping 2 modes in the FFT captures the entire cosine"""
    in_channels = 1
    out_channels = 1
    length = 10_000
    modes = 2
    x = torch.linspace(0, 2 * torch.pi, length).reshape(1, in_channels, length)
    y = torch.cos(x)
    weights = torch.ones(in_channels, out_channels, modes, dtype=torch.cfloat)
    f = func.spectral_conv1d(y, weights, out_channels, modes)
    assert torch.allclose(f, y, atol=1e-3)


@pytest.mark.parametrize("modes", [1, 2, 3, 4, 5])
def test_sine(modes):
    """Test the truncation of a sum of sines for various modes"""
    x = torch.linspace(0, 2 * torch.pi, 10_000, requires_grad=False).reshape(1, 1, -1)
    y = torch.zeros_like(x)
    for i in range(1, 5):
        y += torch.sin(i * x)

    analytical_solution = torch.zeros_like(x)
    for j in range(1, modes):
        analytical_solution += torch.sin(j * x)

    weights = torch.ones([1, 1, modes], dtype=torch.cfloat)
    f = func.spectral_conv1d(y, weights, 1, modes)
    assert torch.allclose(f, analytical_solution, atol=1e-2)
