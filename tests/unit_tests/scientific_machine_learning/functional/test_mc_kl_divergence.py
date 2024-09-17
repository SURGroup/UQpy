import torch
import UQpy.scientific_machine_learning.functional as func
import UQpy.distributions as dist
from hypothesis import given, settings, strategies as st


@settings(deadline=None)
@given(
    prior_param_1=st.floats(min_value=1e-3, max_value=1),
    prior_param_2=st.floats(min_value=1e-3, max_value=1),
    posterior_param_1=st.floats(min_value=1e-3, max_value=1),
    posterior_param_2=st.floats(min_value=1e-3, max_value=1),
)
def test_non_negativity(
    prior_param_1,
    prior_param_2,
    posterior_param_1,
    posterior_param_2,
):
    """KL divergence is always non-negative"""
    prior_distribution = [dist.Lognormal(prior_param_1, prior_param_2)]
    posterior_distribution = [dist.Lognormal(posterior_param_1, posterior_param_2)]
    kl = func.mc_kullback_leibler_divergence(
        posterior_distribution, prior_distribution, n_samples=10_000
    )
    kl = torch.round(kl, decimals=2)
    assert kl >= 0


@given(st.integers(min_value=1, max_value=100))
def test_shape(n):
    """A list with any number of distributions should give a scalar value of KL divergence"""
    prior = [dist.Uniform(0, 1)] * n
    posterior = [dist.Uniform(0, 1)] * n
    kl = func.mc_kullback_leibler_divergence(
        posterior, prior, n_samples=1, reduction="sum"
    )
    assert kl.shape == torch.Size()
    kl = func.mc_kullback_leibler_divergence(
        posterior, prior, n_samples=1, reduction="mean"
    )
    assert kl.shape == torch.Size()
    kl = func.mc_kullback_leibler_divergence(
        posterior, prior, n_samples=1, reduction="none"
    )
    assert kl.shape == torch.Size([n])


@settings(max_examples=2)
@given(
    prior_mu=st.floats(min_value=0, max_value=0),
    prior_sigma=st.floats(min_value=1, max_value=1),
    posterior_mu=st.floats(min_value=1, max_value=1),
    posterior_sigma=st.floats(min_value=1, max_value=1),
)
def test_accuracy(prior_mu, prior_sigma, posterior_mu, posterior_sigma):
    """Compare the accuracy with closed form expression. Assert if MC is within 10% error of closed form"""
    posterior_distribution = [dist.Normal(posterior_mu, posterior_sigma)]
    prior_distribution = [dist.Normal(prior_mu, prior_sigma)]
    kl_mc = func.mc_kullback_leibler_divergence(
        posterior_distribution, prior_distribution, n_samples=100
    )
    kl_cf = func.gaussian_kullback_leibler_divergence(
        torch.tensor(posterior_mu),
        torch.tensor(posterior_sigma),
        torch.tensor(prior_mu),
        torch.tensor(prior_sigma),
    )
    assert torch.allclose(
        kl_mc, kl_cf, rtol=0.1
    )  # FixMe: the divergences are not within 10% of each other, even for 100 samples
