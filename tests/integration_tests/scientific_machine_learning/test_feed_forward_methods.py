import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml


def test_mismatch_sampling():
    """If any layers have sampling=True, set the FeedForwardNeuralNetwork and all layers to sampling=True"""
    network = nn.Sequential(
        sml.BayesianLinear(1, 1, sampling=True),
        sml.BayesianLinear(1, 1, sampling=False),
        sml.BayesianLinear(1, 1, sampling=False),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    assert model.sampling
    assert all(
        (m.sampling if hasattr(m, "sampling") else True)
        for m in model.network.modules()
    )


def test_mismatch_dropping():
    """If any layers have dropping=True, set the FeedForwardNeuralNetwork and all layers to dropping=True"""
    network = nn.Sequential(
        nn.Linear(1, 1),
        sml.Dropout(dropping=True),
        nn.Linear(1, 1),
        sml.Dropout1d(dropping=False),
        nn.Linear(1, 1),
        sml.Dropout2d(dropping=False),
        nn.Linear(1, 1),
        sml.Dropout3d(dropping=False),
        nn.Linear(1, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    assert model.dropping
    assert all(
        (m.dropping if hasattr(m, "dropping") else True)
        for m in model.network.modules()
    )


def test_set_deterministic_true():
    network = nn.Sequential(
        sml.BayesianLinear(1, 10),
        sml.Dropout(),
        sml.BayesianLinear(10, 10),
        sml.Dropout(),
        sml.BayesianLinear(10, 1)
    )
    model = sml.FeedForwardNeuralNetwork(network)
    model.set_deterministic(True)
    assert model.is_deterministic()
    x = torch.tensor([1.0])
    y1 = model(x)
    y2 = model(x)
    assert y1 == y2


def test_set_deterministic_false():
    network = nn.Sequential(
        sml.BayesianLinear(1, 10),
        sml.Dropout(),
        sml.BayesianLinear(10, 10),
        sml.Dropout(),
        sml.BayesianLinear(10, 1)
    )
    model = sml.FeedForwardNeuralNetwork(network)
    model.set_deterministic(False)
    assert not model.is_deterministic()
    x = torch.tensor([1.0])
    y1 = model(x)
    y2 = model(x)
    assert y1 != y2