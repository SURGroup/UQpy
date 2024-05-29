import torch
import torch.nn as nn
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.neural_networks import FeedForwardNeuralNetwork
from UQpy.scientific_machine_learning.layers import BayesianLinear


@given(integers(min_value=1, max_value=1_000), integers(min_value=1, max_value=1_000))
def test_output_shape(in_features, out_features):
    model = FeedForwardNeuralNetwork(BayesianLinear(in_features, out_features))
    x = torch.ones((in_features,))
    assert model(x).shape == torch.Size([out_features])


def test_deterministic_output():
    in_features = 100
    out_features = 200
    network = nn.Sequential(BayesianLinear(in_features, out_features))
    model = FeedForwardNeuralNetwork(network)
    model.train(False)
    model.sample(False)

    x = torch.ones((in_features,))
    y1 = model(x)
    y2 = model(x)
    assert torch.allclose(y1, y2)


def test_probabilistic_output():
    in_features = 100
    out_features = 200
    network = nn.Sequential(BayesianLinear(in_features, out_features))
    model = FeedForwardNeuralNetwork(network)
    model.train(False)
    model.sample(True)

    x = torch.ones((in_features,))
    y1 = model(x)
    y2 = model(x)
    assert not torch.allclose(y1, y2)
