import torch
import torch.nn as nn
from hypothesis import given
from hypothesis.strategies import integers
from UQpy.scientific_machine_learning.neural_networks.BayesianNeuralNetwork import BayesianNeuralNetwork
from UQpy.scientific_machine_learning.layers.BayesianLayer import BayesianLayer


#@given(integers(min_value=1, max_value=1_000), integers(min_value=1, max_value=1_000))
def test_output_shape():
    network = nn.Sequential(
        BayesianLayer(5, 10),
        nn.ReLU(),
        BayesianLayer(10, 10),
        nn.ReLU(),
        BayesianLayer(10, 1),
    )
    model = BayesianNeuralNetwork(BayesianLayer(1, 1))
    kl = model.compute_kl_div()
    assert not torch.isnan(kl)
