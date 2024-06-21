import torch
import UQpy.scientific_machine_learning.functional as func
from beartype import beartype


@beartype
def gaussian_jenson_shannon_divergence(
    posterior_mu: torch.Tensor,
    posterior_sigma: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma: torch.Tensor,
) -> torch.Tensor:
    r"""Compute the Gaussian Jenson-Shannon divergence for a prior and posterior distribution

    :param posterior_mu: Mean of the posterior distribution
    :param posterior_sigma: Standard deviation of the posterior distribution
    :param prior_mu: Mean of the prior distribution
    :param prior_sigma: Standard deviation of the prior distribution
    :return: Gaussian JS divergence between prior and posterior distributions

    Formula
    -------
    The Jenson-Shannon divergence :math:`D_{JS}` is computed as

    .. math:: D_{JS}(P, Q) = \frac12 \left( D_{KL}(P, M) + D_{KL}(Q, M) \right)

    where :math:`D_{KL}` is the Kullback-Leiber divergence and :math:`M=P+Q` is the mixture distribution.
    """
    mixture_mu = (prior_mu + posterior_mu) / 2
    mixture_sigma = torch.sqrt((prior_sigma**2 + posterior_sigma**2) / 2)
    dkl_prior_mixture = func.gaussian_kullback_leiber_divergence(
        prior_mu, prior_sigma, mixture_mu, mixture_sigma
    )
    dkl_posterior_mixture = func.gaussian_kullback_leiber_divergence(
        posterior_mu, posterior_sigma, mixture_mu, mixture_sigma
    )
    return (dkl_prior_mixture + dkl_posterior_mixture) / 2
