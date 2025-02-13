import torch
import UQpy.scientific_machine_learning.functional as func
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import array_shapes

settings.register_profile("fast", max_examples=1)
settings.load_profile("fast")

@given(
    mu=st.floats(min_value=-10, max_value=10),
    sigma=st.floats(min_value=1e-3, max_value=10),
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_divergence_zero(mu, sigma, shape):
    """For identical distributions P and Q, the divergence is zero"""
    mu = torch.full(shape, mu)
    sigma = torch.full(shape, sigma)
    divergence = func.gaussian_kullback_leibler_divergence(mu, sigma, mu, sigma)
    assert divergence == 0


def test_divergence_one_half():
    """For distributions with equal variance, and means 0, 1, the GKL divergence is 0.5"""
    posterior_mu = torch.tensor(1.0)
    posterior_sigma = torch.tensor(1.0)
    prior_mu = torch.tensor(0.0)
    prior_sigma = torch.tensor(1.0)
    divergence = func.gaussian_kullback_leibler_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma
    )
    assert divergence == 0.5


@given(
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_divergence_zero(shape):
    """For any distributions, the KL divergence is non-negative"""
    prior_mu = torch.rand(shape)
    prior_sigma = torch.rand(shape) + 1  # must be positive
    posterior_mu = torch.rand(shape)
    posterior_sigma = torch.rand(shape) + 1  # must be positive
    divergence = func.gaussian_kullback_leibler_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma
    )
    assert divergence >= 0


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
    divergence = func.gaussian_kullback_leibler_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, reduction="sum"
    )
    assert divergence.shape == torch.Size()
    divergence = func.gaussian_kullback_leibler_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, reduction="mean"
    )
    assert divergence.shape == torch.Size()
    divergence = func.gaussian_kullback_leibler_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, reduction="none"
    )
    assert divergence.shape == torch.Size(shape)
