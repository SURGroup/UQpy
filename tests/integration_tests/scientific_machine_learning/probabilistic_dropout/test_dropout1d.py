import pytest
import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml


@pytest.fixture
def model():
    width = 5
    network = nn.Sequential(
        nn.Linear(1, width),
        nn.ReLU(),
        sml.ProbabilisticDropout1d(),
        nn.Linear(width, width),
        nn.ReLU(),
        sml.ProbabilisticDropout1d(),
        nn.Linear(width, 1),
    )
    return sml.FeedForwardNeuralNetwork(network)


def test_drop_false(model):
    x = torch.rand(20, 30, 1)
    model.eval()
    model.drop(False)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert model.is_deterministic()
    assert torch.all(y1 == y2)


def test_drop_true(model):
    x = torch.rand(20, 30, 1)
    model.eval()
    model.drop(True)
    with torch.no_grad():
        y1 = model(x)
        y2 = model(x)
    assert not model.is_deterministic()
    assert not torch.all(y1 == y2)
