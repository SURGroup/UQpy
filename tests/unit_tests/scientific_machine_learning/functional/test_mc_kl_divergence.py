import pytest
import torch
import UQpy.scientific_machine_learning.functional as func
from UQpy.distributions import Normal, Lognormal
import random


def test_positivity():
    """KL divergence is non negative
        """
    distribution = Lognormal
    prior_distribution = [distribution(random.uniform(0, 1), random.uniform(0, 1))]
    posterior_distribution = [distribution(random.uniform(0, 1), random.uniform(0, 1))]
    kl = func.mc_kullback_leibler_divergence(posterior_distribution, prior_distribution)
    assert kl >= 0


def test_shape():
    """A list with any number of distributions should give a scalar value of KL divergence
        """
    normal1 = Normal(random.uniform(-1, 1), random.uniform(0, 1))
    normal2 = Normal(random.uniform(-1, 1), random.uniform(0, 1))
    normal3 = Normal(random.uniform(-1, 1), random.uniform(0, 1))
    normal4 = Normal(random.uniform(-1, 1), random.uniform(0, 1))
    prior_distribution = [normal1, normal2]
    posterior_distribution = [normal3, normal4]
    kl = func.mc_kullback_leibler_divergence(posterior_distribution, prior_distribution)
    assert kl.shape == torch.Size()


def test_accuracy():
    """Compare the accuracy with closed form expression. Assert if MC is within 10% error of closed form
        """
    mu1 = random.uniform(-1, 1)
    sigma1 = random.uniform(0, 1)
    mu2 = random.uniform(-1, 1)
    sigma2 = random.uniform(0, 1)
    posterior_distribution = [Normal(mu1, sigma1)]
    prior_distribution = [Normal(mu2, sigma2)]
    kl_mc = func.mc_kullback_leibler_divergence(posterior_distribution, prior_distribution)
    kl_cf = func.gaussian_kullback_leiber_divergence(torch.tensor(mu1), torch.tensor(sigma1), torch.tensor(mu2),
                                                     torch.tensor(sigma2))
    assert kl_cf*0.9 <= kl_mc <= kl_cf*1.1
