import pytest
import torch
import UQpy.scientific_machine_learning.functional as func


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
    weights = torch.ones(in_channels, out_channels, modes)
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
    weights = torch.ones(in_channels, out_channels, modes)
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

    weights = torch.ones([1, 1, modes])
    f = func.spectral_conv1d(y, weights, 1, modes)
    assert torch.allclose(f, analytical_solution, atol=1e-2)
