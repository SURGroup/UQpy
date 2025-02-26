import torch
import torch.nn as nn
import UQpy.scientific_machine_learning as sml


def test_device():
    """Test switching a model to another device.

    Note if neither cuda nor mps is available, this test will always pass
    """
    cpu = torch.device("cpu")
    network = nn.Sequential(
        nn.Linear(1, 1, device=cpu),
        sml.BayesianLinear(1, 1, device=cpu),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")
    model.to(device)
    assert model.network[0].weight.device == device
    assert model.network[1].weight_mu.device == device


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
        sml.ProbabilisticDropout(dropping=True),
        nn.Linear(1, 1),
        sml.ProbabilisticDropout1d(dropping=False),
        nn.Linear(1, 1),
        sml.ProbabilisticDropout2d(dropping=False),
        nn.Linear(1, 1),
        sml.ProbabilisticDropout3d(dropping=False),
        nn.Linear(1, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    assert model.dropping
    assert all(
        (m.dropping if hasattr(m, "dropping") else True)
        for m in model.network.modules()
    )


def test_deterministic_output():
    network = nn.Sequential(
        sml.BayesianLinear(1, 10),
        sml.ProbabilisticDropout(),
        sml.BayesianLinear(10, 10),
        sml.ProbabilisticDropout(),
        sml.BayesianLinear(10, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    model.set_deterministic(True)
    assert model.is_deterministic()
    x = torch.tensor([1.0])
    y1 = model(x)
    y2 = model(x)
    assert y1 == y2


def test_probabilistic_output():
    network = nn.Sequential(
        sml.BayesianLinear(1, 10),
        sml.ProbabilisticDropout(),
        sml.BayesianLinear(10, 10),
        sml.ProbabilisticDropout(),
        sml.BayesianLinear(10, 1),
    )
    model = sml.FeedForwardNeuralNetwork(network)
    model.set_deterministic(False)
    assert not model.is_deterministic()
    x = torch.tensor([1.0])
    y1 = model(x)
    y2 = model(x)
    assert y1 != y2
