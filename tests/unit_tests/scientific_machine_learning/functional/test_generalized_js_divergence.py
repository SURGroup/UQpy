import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes
import UQpy as uq


def test_alpha_zero_is_kl():
    posterior_mu = torch.tensor(1.0)
    posterior_sigma = torch.tensor(1.0)
    prior_mu = torch.tensor(0.0)
    prior_sigma = torch.tensor(1.0)
    kl_divergence = func.gaussian_kullback_leiber_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma
    )

    posterior_distribution = [uq.Normal(1, 1)]
    prior_distribution = [uq.Normal(0, 1)]
    js_divergence = func.generalized_jenson_shannon_divergence(
        posterior_distribution, prior_distribution, alpha=0.0
    )
    assert torch.isclose(kl_divergence, js_divergence, rtol=0.05)
