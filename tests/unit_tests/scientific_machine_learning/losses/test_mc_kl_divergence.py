import pytest
import torch
import torch.nn as nn
from UQpy.distributions import Normal, Lognormal
import UQpy.scientific_machine_learning as sml
from hypothesis import given, settings, strategies as st


@settings(deadline=1_000)
@given(width=st.integers(min_value=1, max_value=10))
def test_divergence_shape(width):
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        sml.BayesianLinear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    mc_divergence = sml.MCKullbackLeiblerDivergence(
        posterior_distribution=Normal,
        prior_distribution=Normal,
    )
    divergence = mc_divergence(model)
    assert divergence.shape == torch.Size()


def test_reduction_none():
    with pytest.raises(ValueError):
        sml.MCKullbackLeiblerDivergence(
            posterior_distribution=Normal, prior_distribution=Normal, reduction="none"
        )
