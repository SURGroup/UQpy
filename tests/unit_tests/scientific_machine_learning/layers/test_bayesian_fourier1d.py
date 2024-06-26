import torch
import UQpy.scientific_machine_learning as sml
from hypothesis import given
from hypothesis.strategies import integers


@given(
    batch_size=integers(min_value=1, max_value=1),
    width=integers(min_value=1, max_value=32),
    length=integers(min_value=64, max_value=256),
    modes=integers(min_value=1, max_value=33),
)
def test_output_shape(batch_size, width, length, modes):
    x = torch.ones(batch_size, width, length)
    fourier = sml.BayesianFourier1d(width, modes)
    y = fourier(x)
    assert y.shape == torch.Size([batch_size, width, length])


def test_deterministic_behavior():
    n = 10
    width = 8
    length = 100
    modes = (length // 2) + 1
    layer = sml.BayesianFourier1d(width, modes)
    layer.train(False)
    layer.sample(False)
    x = torch.ones((n, width, length))

    y1 = layer(x)
    y2 = layer(x)
    assert torch.allclose(y1, y2)
