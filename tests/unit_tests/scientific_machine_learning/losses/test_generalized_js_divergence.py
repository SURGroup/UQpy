import pytest
import torch
import torch.nn as nn
import UQpy as uq
import UQpy.scientific_machine_learning as sml


def test_reduction_shape():
    width = 4
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        nn.Linear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    divergence_function = sml.GeneralizedJensenShannonDivergence(
        uq.Normal, uq.Uniform, n_samples=1
    )
    divergence = divergence_function(model)
    assert divergence.shape == torch.Size()


def test_reduction_none_raises_error():
    with pytest.raises(ValueError):
        sml.GeneralizedJensenShannonDivergence(uq.Normal, uq.Normal, reduction="none")


def test_device():
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")
    divergence_function = sml.GeneralizedJensenShannonDivergence(
        uq.Normal, uq.Normal, device=device, n_samples=1
    )
    model = sml.FeedForwardNeuralNetwork(sml.BayesianLinear(1, 1))
    model.to(device)
    divergence = divergence_function(model)
    assert divergence.device == device
