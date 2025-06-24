import pytest
import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


@given(width=st.integers(min_value=1, max_value=10))
def test_shape(width):
    """Geometric Jensen-Shannon returns a scalar valued divergence"""
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        sml.BayesianLinear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    divergence_loss = sml.GeometricJensenShannonDivergence(alpha=0.5)
    divergence = divergence_loss(model)
    assert divergence.shape == torch.Size()


def test_reduction_none():
    with pytest.raises(ValueError):
        sml.GeometricJensenShannonDivergence(alpha=0.5, reduction="none")


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")

    model = sml.FeedForwardNeuralNetwork(sml.BayesianLinear(1, 1, dtype=torch.float32))
    model.to(device)
    divergence_function = sml.GeometricJensenShannonDivergence(device=device)
    divergence = divergence_function(model)
    assert divergence.device == device
