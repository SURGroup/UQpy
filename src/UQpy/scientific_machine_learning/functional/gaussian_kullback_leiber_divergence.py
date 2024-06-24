import torch
from beartype import beartype
from typing import Union

# @beartype
def gaussian_kullback_leiber_divergence(
    posterior_mu: torch.Tensor,
    posterior_sigma: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """Compute the Gaussian Kullback-Leibler divergence for a prior and posterior distribution

    :param posterior_mu: Mean of the posterior distribution
    :param posterior_sigma: Standard deviation of the posterior distribution
    :param prior_mu: Mean of the prior distribution
    :param prior_sigma: Standard deviation of the prior distribution
    :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
     'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
     Default: 'mean'

    :return: Gaussian KL divergence between prior and posterior distributions

    :raise ValueError: If ``reduction`` is not 'none', 'mean', or 'sum'
    """
    gkl_divergence = 0.5 * (
        2 * torch.log(prior_sigma / posterior_sigma)
        - 1
        + (posterior_sigma / prior_sigma).pow(2)
        + ((prior_mu - posterior_mu) / prior_sigma).pow(2)
    )
    if reduction == "none":
        return gkl_divergence
    elif reduction == "mean":
        return torch.mean(gkl_divergence)
    elif reduction == "sum":
        return torch.sum(gkl_divergence)
    else:
        raise ValueError("UQpy: `reduction` must be one of 'none', 'mean', or 'sum'")
