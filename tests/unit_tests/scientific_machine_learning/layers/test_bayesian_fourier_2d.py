import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given
from hypothesis.strategies import integers, tuples


@given(
    batch_size=integers(min_value=1, max_value=1),
    width=integers(min_value=1, max_value=32),
    w=integers(min_value=64, max_value=256),
    h=integers(min_value=64, max_value=256),
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
