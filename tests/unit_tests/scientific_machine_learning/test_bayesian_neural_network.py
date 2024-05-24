import torch
import torch.nn as nn
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.neural_networks.BayesianNeuralNetwork import (
    BayesianNeuralNetwork,
)
from UQpy.scientific_machine_learning.layers.BayesianLinear import BayesianLinear


@given(integers(min_value=1, max_value=1_000), integers(min_value=1, max_value=1_000))
def test_output_shape(in_features, out_features):
    model = BayesianNeuralNetwork(BayesianLinear(in_features, out_features))
    x = torch.ones((in_features,))
    assert model(x).shape == torch.Size([out_features])


def test_deterministic_output():
    in_features = 100
    width = 5
    out_features = 200
    network = nn.Sequential(
        BayesianLinear(in_features, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        BayesianLinear(width, out_features),
    )
    model = BayesianNeuralNetwork(network)
    model.train(False)
    model.sample(False)

    x = torch.ones((in_features,))
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    in_features = 5
    width = 7
    out_features = 3
    network = nn.Sequential(
        BayesianLinear(in_features, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        BayesianLinear(width, out_features),
    )
    model = BayesianNeuralNetwork(network)
    model.train(False)
    model.sample(True)

    x = torch.ones((in_features,))
    y1 = model(x)
    y2 = model(x)
    assert not torch.allclose(y1, y2)
