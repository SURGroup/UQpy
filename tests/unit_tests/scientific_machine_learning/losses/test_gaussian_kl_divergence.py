import pytest
import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


# ToDo: write device test

@given(width=st.integers(min_value=1, max_value=10))
def test_reduction_shape(width):
    """With the default reduction='sum', the divergence should be a scalar"""
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        sml.BayesianLinear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    divergence_function = sml.GaussianKullbackLeiblerDivergence()
    divergence = divergence_function(model)
    assert divergence.shape == torch.Size()


def test_reduction_none():
    """Only accepts 'mean' or 'sum' as reduction"""
    with pytest.raises(ValueError):
        sml.GaussianKullbackLeiblerDivergence(reduction="none")
