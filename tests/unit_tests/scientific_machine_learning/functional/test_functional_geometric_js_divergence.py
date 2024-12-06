import torch
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import array_shapes

import UQpy.scientific_machine_learning.functional as func


@given(
    prior_param_1=st.floats(min_value=-1, max_value=1),
    prior_param_2=st.floats(min_value=1e-3, max_value=1),
    posterior_param_1=st.floats(min_value=-1, max_value=1),
    posterior_param_2=st.floats(min_value=1e-3, max_value=1),
    alpha=st.floats(min_value=0, max_value=1),
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_non_negativity(
        prior_param_1,
        prior_param_2,
        posterior_param_1,
        posterior_param_2,
        alpha,
        shape,
):
    """JSG divergence is always non-negative"""
    post_mu = torch.full(shape, posterior_param_1)
    post_sigma = torch.full(shape, posterior_param_2)
    prior_mu = torch.full(shape, prior_param_1)
    prior_sigma = torch.full(shape, prior_param_2)
    jsg = func.geometric_jensen_shannon_divergence(
        post_mu, post_sigma, prior_mu, prior_sigma, alpha
    )
    jsg = torch.round(jsg, decimals=3)
    assert jsg >= 0


@given(
    prior_param_1=st.floats(min_value=1e-3, max_value=1),
    prior_param_2=st.floats(min_value=1e-3, max_value=1),
    posterior_param_1=st.floats(min_value=1e-3, max_value=1),
    posterior_param_2=st.floats(min_value=1e-3, max_value=1),
    shape=array_shapes(min_dims=1, min_side=1, max_side=100),
)
def test_kl_equal(
        prior_param_1,
        prior_param_2,
        posterior_param_1,
        posterior_param_2,
        shape,
):
    """JSG divergence is equal to KL divergence when alpha = 0"""
    post_mu = torch.full(shape, posterior_param_1)
    post_sigma = torch.full(shape, posterior_param_2)
    prior_mu = torch.full(shape, prior_param_1)
    prior_sigma = torch.full(shape, prior_param_2)
    jsg = func.geometric_jensen_shannon_divergence(
        post_mu, post_sigma, prior_mu, prior_sigma, alpha=0.0
    )
    kl = func.gaussian_kullback_leibler_divergence(
        post_mu, post_sigma, prior_mu, prior_sigma
    )
    assert torch.allclose(jsg, kl, rtol=1e-4)


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
    divergence = func.geometric_jensen_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, alpha=0.5, reduction="sum"
    )
    assert divergence.shape == torch.Size()
    divergence = func.geometric_jensen_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, alpha=0.5, reduction="mean"
    )
    assert divergence.shape == torch.Size()
    divergence = func.geometric_jensen_shannon_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma, alpha=0.5, reduction="none"
    )
    assert divergence.shape == torch.Size(shape)
