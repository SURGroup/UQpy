import torch
import torch.nn
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.neural_networks.BayesianNeuralNetwork import BayesianNeuralNetwork
from UQpy.scientific_machine_learning.layers.BayesianLayer import BayesianLayer


@given(integers(min_value=1, max_value=1_000), integers(min_value=1, max_value=1_000))
def test_output_shape(in_features, out_features):
    model = BayesianNeuralNetwork(BayesianLayer(in_features, out_features))
    x = torch.ones((in_features,))
    assert model(x).shape == torch.Size([out_features])

