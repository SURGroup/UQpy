import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given
from hypothesis.strategies import integers


@given(
    integers(min_value=1, max_value=10),
    integers(min_value=1, max_value=64),
    integers(min_value=1, max_value=1_000),
)
def test_output_shape(n, width, length):
    modes = (length // 2) + 1
    layer = sml.BayesianFourier1d(width, modes)
    x = torch.ones((n, width, length))

    y = layer(x)
    assert y.shape == torch.Size([n, width, length])


def test_sample_sets_children():
    layer = sml.BayesianFourier1d(8, 51)
    layer.sample(False)
    assert layer.conv.sampling is False
    assert layer.spectral_conv.sampling is False


def test_sample_false_train_false():
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
