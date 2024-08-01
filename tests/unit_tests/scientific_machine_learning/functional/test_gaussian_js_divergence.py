import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes


@given(
    mu=st.floats(min_value=-10, max_value=10),
    sigma=st.floats(min_value=1e-3, max_value=10),
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_divergence_zero(mu, sigma, shape):
    """For identical distributions P and Q, the divergence is zero"""
    mu = torch.full(shape, mu)
    sigma = torch.full(shape, sigma)
    divergence = func.gaussian_jenson_shannon_divergence(mu, sigma, mu, sigma)
    assert divergence == 0


def test_divergence_one_eighth():
    """For distributions N(0, 1) and N(1, 1) the Gaussian JS divergence is 0.125"""
    posterior_mu = torch.tensor(1.0)
    posterior_sigma = torch.tensor(1.0)
    prior_mu = torch.tensor(0.0)
    prior_sigma = torch.tensor(1.0)
    divergence = func.gaussian_jenson_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma
    )
    assert divergence == 0.125


@given(
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_divergence_symmetric(shape):
    """The JS divergence is symmetric, JS(P, Q) = JS(Q, P)"""
    posterior_mu = torch.rand(shape)
    posterior_sigma = torch.rand(shape)
    prior_mu = torch.rand(shape)
    prior_sigma = torch.rand(shape)
    divergence_1 = func.gaussian_jenson_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma
    )
    divergence_2 = func.gaussian_jenson_shannon_divergence(
        prior_mu, prior_sigma, posterior_mu, posterior_sigma
    )
    assert torch.allclose(divergence_1, divergence_2)


@given(
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_reduction_shape(shape):
    """For mean and sum, the divergence is a scalar.
    For reduction='none', the divergence is a tensor of the same shapes as the input
    """
    posterior_mu = torch.rand(shape)
    posterior_sigma = torch.rand(shape)
    prior_mu = torch.rand(shape)
    prior_sigma = torch.rand(shape)
    divergence = func.gaussian_jenson_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, reduction="sum"
    )
    assert divergence.shape == torch.Size()
    divergence = func.gaussian_jenson_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, reduction="mean"
    )
    assert divergence.shape == torch.Size()
    divergence = func.gaussian_jenson_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, reduction="none"
    )
    assert divergence.shape == torch.Size(shape)