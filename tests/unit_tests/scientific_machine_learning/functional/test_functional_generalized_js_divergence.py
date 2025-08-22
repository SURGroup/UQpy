import torch
import UQpy.scientific_machine_learning.functional as func
import UQpy as uq


def test_gaussian_js_equals_gaussian_kl():
    """When alpha=0, generalized JS on normal distributions is exactly Gaussian KL divergence.
    Because JS is estimated from Monte Carlo, this tests for within 5% accuracy.
    """
    # initialize distributions
    dtype = torch.float32
    posterior_mu = torch.tensor(1.0, dtype=dtype)
    posterior_sigma = torch.tensor(1.0, dtype=dtype)
    prior_mu = torch.tensor(0.0, dtype=dtype)
    prior_sigma = torch.tensor(1.0, dtype=dtype)
    posterior_distribution = [uq.Normal(1, 1)]
    prior_distribution = [uq.Normal(0, 1)]
    # compute divergences
    kl_divergence = func.gaussian_kullback_leibler_divergence(
        posterior_mu, posterior_sigma, prior_mu, prior_sigma
    )
    js_divergence = func.generalized_jensen_shannon_divergence(
        posterior_distribution, prior_distribution, alpha=0.0, n_samples=10_000
    )
    assert torch.isclose(kl_divergence, js_divergence, rtol=0.05)


def test_non_gaussian_js_equals_mc_kl():
    posterior_distribution = [uq.Uniform(3, 1)]
    prior_distribution = [uq.Uniform(2, 5)]
    kl_divergence = func.mc_kullback_leibler_divergence(
        posterior_distribution,
        prior_distribution,
    )
    js_divergence = func.generalized_jensen_shannon_divergence(
        posterior_distribution,
        prior_distribution,
        alpha=0.0,
    )
    assert torch.isclose(kl_divergence, js_divergence, rtol=0.05)


def test_device():
    """Note if neither cuda nor mps is available, this test will always pass"""
    posterior_distribution = [uq.Uniform(3, 1)]
    prior_distribution = [uq.Uniform(2, 5)]
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps", 0)
    else:
        device = torch.device("cpu")
    divergence = func.generalized_jensen_shannon_divergence(
        posterior_distribution, prior_distribution, alpha=0.0, device=device
    )
    assert divergence.device == device
