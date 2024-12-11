import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings
from hypothesis.strategies import integers


@settings(max_examples=10)
@given(
    batch_size=integers(min_value=1, max_value=1),
    width=integers(min_value=1, max_value=8),
    length=integers(min_value=64, max_value=128),
    modes=integers(min_value=1, max_value=33),
)
def test_output_shape(batch_size, width, length, modes):
    x = torch.ones((batch_size, width, length))
    fourier = sml.BayesianFourier1d(width, modes)
    y = fourier(x)
    assert y.shape == torch.Size([batch_size, width, length])


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    cpu = torch.device("cpu")
    layer = sml.BayesianFourier1d(1, 1, device=cpu)
    assert layer.weight_spectral_mu.device == cpu
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("mps", 0) if torch.backends.mps.is_available() else "cpu"
    )
    layer.to(device)
    assert layer.weight_spectral_mu.device == device


def test_deterministic_output():
    n = 10
    width = 8
    length = 100
    modes = (length // 2) + 1
    layer = sml.BayesianFourier1d(width, modes)
    layer.sample(False)
    x = torch.ones((n, width, length))

    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    n = 10
    width = 8
    length = 100
    modes = (length // 2) + 1
    layer = sml.BayesianFourier1d(width, modes)
    layer.sample(True)
    x = torch.ones((n, width, length))

    y1 = layer(x)
    y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_bias_false():
    """When bias=False, BayesianFourier1d(0) = 0"""
    x = torch.zeros((1, 1, 256))
    layer = sml.BayesianFourier1d(1, 1, bias=False)
    y = layer(x)
    assert torch.all(y == torch.zeros_like(y))

