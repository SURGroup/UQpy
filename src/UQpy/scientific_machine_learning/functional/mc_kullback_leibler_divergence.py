import torch
from UQpy.sampling import MonteCarloSampling
from beartype import beartype


@beartype
def mc_kullback_leibler_divergence(
    posterior_distributions: list,
    prior_distributions: list,
    num_samples: int = 1000,
    reduction: str = "sum",
) -> torch.Tensor:
    r"""Compute the Kullback-Leibler divergence by sampling for a prior and posterior distribution
    :param posterior_distributions: Variational posterior distribution
    :param prior_distributions: Prior distribution of parameters
    :param num_samples: Number of samples in the Monte Carlo estimation
    :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
     'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
     Default: 'sum'

    :return: KL divergence between prior and posterior distributions

    :raises ValueError: If ``reduction`` is not one of 'none', 'mean', or 'sum'

    """
    mc = MonteCarloSampling(distributions=posterior_distributions, nsamples=num_samples)
    mc_kl_divergence = torch.zeros(len(posterior_distributions))
    for itr in range(num_samples):
        posterior_samples = mc.samples[itr]
        div_list = []
        for prior_dist, post_dist, post_samp in zip(
            prior_distributions, posterior_distributions, posterior_samples
        ):
            div_list.append(
                (post_dist.log_pdf(post_samp) - prior_dist.log_pdf(post_samp)).item()
            )
        mc_kl_divergence += torch.tensor(div_list) / num_samples
    if reduction == "none":
        return mc_kl_divergence
    elif reduction == "mean":
        return torch.mean(mc_kl_divergence)
    elif reduction == "sum":
        return torch.sum(mc_kl_divergence)
    else:
        raise ValueError("UQpy: `reduction` must be one of 'none', 'mean', or 'sum'")
