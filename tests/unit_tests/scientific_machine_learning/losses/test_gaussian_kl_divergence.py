import pytest
import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


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


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    device = (
        torch.device("cuda", 0)
        if torch.cuda.is_available()
        else torch.device("mps", 0) if torch.backends.mps.is_available() else "cpu"
    )
    model = sml.FeedForwardNeuralNetwork(sml.BayesianLinear(1, 1))
    model.to(device)
    divergence_function = sml.GaussianKullbackLeiblerDivergence(device=device)
    divergence = divergence_function(model)
    assert divergence.device == device