import torch
import numpy as np
from UQpy import MonteCarloSampling
from beartype import beartype
from beartype.vale import Is
from typing import Annotated


@beartype
def generalized_jensen_shannon_divergence(
    posterior_distributions: list,
    prior_distributions: list,
    n_samples: int = 1000,
    alpha: Annotated[float, Is[lambda x: 0 <= x <= 1]] = 0.5,
    reduction: str = "sum",
    device=None,
) -> torch.Tensor:
    r"""Compute the generalized Jensen-Shannon divergence for a prior and posterior distribution

    :param posterior_distributions: List of UQpy distributions defining the variational posterior
    :param prior_distributions: List of UQpy distributions defining the prior
    :param n_samples: Number of samples in the Monte Carlo estimation. Default: 1,000
    :param alpha: Weight of the mixture distribution, :math:`0 \leq \alpha \leq 1`. 
     See formula for details. Default: 0.5
    :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
     'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
     Default: 'sum'
     :param device: An object representing the device on which a :py:class:`torch.Tensor` will be allocated.

    :return: JS divergence between prior and posterior distributions

    :raises ValueError: If ``reduction`` is not one of 'none', 'mean', or 'sum'
    :raises RuntimeError: If ``len(posterior_distributions)`` is not equal to ``len(prior_distributions)``

    Formula
    -------
    The Jenson-Shannon divergence :math:`D_{JS}` is computed as

    .. math:: D_{JS}(P, Q) = (1- \alpha) D_{KL}(P, M) + \alpha D_{KL}(Q, M)

    where :math:`D_{KL}` is the Kullback-Leibler divergence and :math:`M=\alpha Q + (1-\alpha) P` is the mixture distribution.
    """
    if len(prior_distributions) != len(posterior_distributions):
        raise RuntimeError(
            "UQpy: `prior_distributions` and `posterior_distributions` must have the same length"
        )
    mc_posterior = MonteCarloSampling(
        distributions=posterior_distributions, nsamples=n_samples
    )
    mc_prior = MonteCarloSampling(
        distributions=prior_distributions, nsamples=n_samples
    )
    n_distributions = len(posterior_distributions)
    js_divergence = np.zeros(n_distributions, dtype=np.float32)
    for i in range(n_samples):
        posterior_samples = mc_posterior.samples[i]
        prior_samples = mc_prior.samples[i]
        for j in range(n_distributions):
            prior = prior_distributions[j]
            posterior = posterior_distributions[j]

            weighted_q_pdf = alpha * posterior.pdf(posterior_samples[j])
            weighted_p_pdf = (1 - alpha) * prior.pdf(posterior_samples[j])
            mixture_pdf_posterior_samples = weighted_p_pdf + weighted_q_pdf

            weighted_q_pdf = alpha * posterior.pdf(prior_samples[j])
            weighted_p_pdf = (1 - alpha) * prior.pdf(prior_samples[j])
            mixture_pdf_prior_samples = weighted_p_pdf + weighted_q_pdf

            kl_divergence_q_m = posterior.log_pdf(posterior_samples[j]) - np.log(
                mixture_pdf_posterior_samples
            )
            kl_divergence_p_m = prior.log_pdf(prior_samples[j]) - np.log(
                mixture_pdf_prior_samples
            )

            js_divergence[j] += ((1 - alpha) * kl_divergence_q_m) + (
                alpha * kl_divergence_p_m
            )
    js_divergence /= n_samples
    js_divergence = torch.tensor(js_divergence, device=device, dtype=torch.float32)

    if reduction == "none":
        return js_divergence
    elif reduction == "mean":
        return torch.mean(js_divergence)
    elif reduction == "sum":
        return torch.sum(js_divergence)
    else:
        raise ValueError(
            f"UQpy: Invalid reduction={reduction}. Must be one of 'none', 'mean', or 'sum'"
        )
