import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings
from hypothesis.strategies import integers, tuples


@settings(max_examples=10)
@given(
    batch_size=integers(min_value=1, max_value=1),
    width=integers(min_value=1, max_value=8),
    w=integers(min_value=64, max_value=128),
    h=integers(min_value=64, max_value=128),
    modes=tuples(
        integers(min_value=1, max_value=33),
        integers(min_value=1, max_value=33),
    ),
)
def test_output_shape(batch_size, width, w, h, modes):
    """Fourier layers do not change the shape of the input"""
    x = torch.ones((batch_size, width, w, h))
    fourier = sml.BayesianFourier2d(width, modes)
    y = fourier(x)
    assert x.shape == y.shape


def test_device():
    """Note if neither cuda nor mps is available this test will always pass"""
    cpu = torch.device("cpu")
    layer = sml.BayesianFourier2d(1, (1, 1), device=cpu)
    assert layer.weight_spectral_mu.device == cpu
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")
    layer.to(device)
    assert layer.weight_spectral_mu.device == device


def test_deterministic_output():
    x = torch.rand((1, 1, 64, 128))
    layer = sml.BayesianFourier2d(1, (33, 65))
    layer.sample(False)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    x = torch.rand((1, 1, 64, 128))
    layer = sml.BayesianFourier2d(1, (33, 65))
    layer.sample(True)
    with torch.no_grad():
        y1 = layer(x)
        y2 = layer(x)
    assert not torch.allclose(y1, y2)


def test_bias_false():
    """When bias=False, BayesianFourier1d(0) = 0"""
    x = torch.zeros((1, 1, 128, 256))
    layer = sml.BayesianFourier2d(1, (64, 128), bias=False)
    y = layer(x)
    assert torch.all(y == torch.zeros_like(y))
