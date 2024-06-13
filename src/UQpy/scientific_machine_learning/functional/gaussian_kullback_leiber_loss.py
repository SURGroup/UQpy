import torch


def gaussian_kullback_leiber_loss(
    posterior_mu: torch.Tensor,
    posterior_sigma: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma: torch.Tensor,
) -> torch.Tensor:
    """Compute the Gaussian Kullback-Leibler divergence for a prior and posterior distribution

    :param posterior_mu: Mean of the posterior distribution
    :param posterior_sigma: Standard deviation of the posterior distribution
    :param prior_mu: Mean of the prior distribution
    :param prior_sigma: Standard deviation of the prior distribution
    :return: Gaussian KL divergence between prior and posterior distributions
    """
    gkl_divergence = (
        0.5
        * (
            2 * torch.log(prior_sigma / posterior_sigma)
            - 1
            + (posterior_sigma / prior_sigma).pow(2)
            + ((prior_mu - posterior_mu) / prior_sigma).pow(2)
        ).sum()
    )
    return gkl_divergence
