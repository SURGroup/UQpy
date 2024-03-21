"""Test the integration of ProbabilisticLayers into a VanillaNeuralNetwork"""
import torch
import torch.nn as nn
import hypothesis
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.neural_networks.VanillaNeuralNetwork import (
    VanillaNeuralNetwork,
)
from UQpy.scientific_machine_learning.layers.ProbabilisticLayer import (
    ProbabilisticLayer,
)


@hypothesis.settings(deadline=500)
@given(integers(min_value=1, max_value=10), integers(min_value=1, max_value=10))
def test_model_output_is_deterministic(in_features, out_features):
    network = nn.Sequential(
        ProbabilisticLayer(in_features, out_features, sample=False),
    )
    model = VanillaNeuralNetwork(network)
    model.train(False)

    x = torch.ones(in_features)
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)


def test_model_output_is_random_via_sample():
    network = nn.Sequential(
        ProbabilisticLayer(5, 7, sample=True),
    )
    model = VanillaNeuralNetwork(network)
    model.train(False)

    x = torch.ones(5)
    y1 = model(x)
    y2 = model(x)
    assert torch.not_equal(y1, y2).all()


def test_model_output_is_random_via_train():
    network = nn.Sequential(ProbabilisticLayer(5, 7, sample=False))
    model = VanillaNeuralNetwork(network)
    model.train(True)

    x = torch.ones(5)
    y1 = model(x)
    y2 = model(x)
    assert torch.not_equal(y1, y2).all()
