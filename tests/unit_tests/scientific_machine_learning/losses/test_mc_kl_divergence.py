import torch.nn as nn
from UQpy.distributions import Normal, Lognormal
import random
import UQpy.scientific_machine_learning as sml
from UQpy.scientific_machine_learning.neural_networks.FeedForwardNeuralNetwork import FeedForwardNeuralNetwork


def test_non_negativity():
    width = random.randint(1, 10)
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        sml.BayesianLinear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = FeedForwardNeuralNetwork(network)
    loss_mc = sml.MCKullbackLeiblerDivergence(posterior_distribution=Normal, prior_distribution=Normal)
    kl = loss_mc(model)
    assert kl >= 0


def test_accuracy():
    width = random.randint(1, 10)
    network = nn.Sequential(
        sml.BayesianLinear(1, width),
        nn.ReLU(),
        sml.BayesianLinear(width, width),
        nn.ReLU(),
        sml.BayesianLinear(width, 1),
    )
    model = FeedForwardNeuralNetwork(network)
    loss_mc = sml.MCKullbackLeiblerDivergence(posterior_distribution=Normal, prior_distribution=Normal)
    loss_cf = sml.GaussianKullbackLeiblerDivergence()
    kl_mc = loss_mc(model)
    kl_cf = loss_cf(model)
    assert kl_cf * 0.9 < kl_mc < kl_cf * 1.1
