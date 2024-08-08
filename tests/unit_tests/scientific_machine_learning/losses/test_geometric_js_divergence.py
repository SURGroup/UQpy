import pytest
import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml
from hypothesis import given, strategies as st


@given(width=st.integers(min_value=1, max_value=10))
def test_reduction_shape(width):
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
