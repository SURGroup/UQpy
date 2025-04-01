import torch
from beartype import beartype
from beartype.vale import Is
from typing import Annotated


@beartype
def geometric_jensen_shannon_divergence(
    posterior_mu: torch.Tensor,
    posterior_sigma: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma: torch.Tensor,
    alpha: Annotated[float, Is[lambda x: 0 <= x <= 1]] = 0.5,
    reduction: str = "sum",
) -> torch.Tensor:
    r"""Compute the Geometric Jensen-Shannon divergence for a Gaussian prior and Gaussian posterior distributions

    :param posterior_mu: Mean of the posterior distribution
    :param posterior_sigma: Standard deviation of the posterior distribution
    :param prior_mu: Mean of the prior distribution
    :param prior_sigma: Standard deviation of the prior distribution
    :param alpha: Weight of the mixture distribution, :math:`0 \leq \alpha \leq 1`.
     See formula for details. Default: 0.5
    :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
     'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
     Default: 'sum'

    :return: Geometric JS divergence between prior and posterior distributions

    Formula
    -------
    The Geometric Jensen-Shannon divergence :math:`D_{JSG}` is computed as

    .. math:: D_{JSG}(P, Q) = (1-\alpha)  D_{KL}(P, M) + \alpha D_{KL}(Q, M)

    where :math:`D_{KL}` is the Kullback-Leibler divergence and :math:`M=P^\alpha Q^{(1-\alpha)}` is the geometric
    mean distribution. When the distributions :math:`P` and :math:`Q` are Gaussian, the closed form for Geometric
    Jensen-Shannon divergence is given as

    .. math:: D_{JSG}(P, Q) = \frac12 \left( \frac{(1-\alpha)\sigma_p^2 + \alpha\sigma_q^2}{\sigma_\alpha^2} + \log \frac{\sigma_\alpha^2}{\sigma_p^{2(1-\alpha)} \sigma_q^{2\alpha}} + (1-\alpha) \frac{(\mu_\alpha - \mu_p)^2}{\sigma_\alpha^2} + \frac{\alpha(\mu_\alpha - \mu_q)^2}{\sigma_\alpha^2} -1 \right)

    where :math:`\sigma_\alpha^2 = \left( \frac{\alpha}{\sigma_p^2}+\frac{1-\alpha}{\sigma_q^2} \right)^{-1}`
    and :math:`\mu_\alpha = \sigma_\alpha^2 \left[\frac{\alpha \mu_p}{\sigma_p^2} + \frac{(1-\alpha)\mu_q}{\sigma_q^2}\right]`

    """
    posterior_var = posterior_sigma.pow(2)
    prior_var = prior_sigma.pow(2)
    var_alpha = 1 / (alpha / posterior_var + (1 - alpha) / prior_var)
    mu_alpha = var_alpha * (
        alpha * posterior_mu / posterior_var + (1 - alpha) * prior_mu / prior_var
    )
    geometric_js_divergence = 0.5 * (
        ((1 - alpha) * posterior_var + alpha * prior_var) / var_alpha
        + torch.log(var_alpha / (posterior_var ** (1 - alpha) * prior_var**alpha))
        + (1 - alpha) * (mu_alpha - posterior_mu).pow(2) / var_alpha
        + alpha * (mu_alpha - prior_mu).pow(2) / var_alpha
        - 1
    )
    if reduction == "none":
        return geometric_js_divergence
    elif reduction == "mean":
        return torch.mean(geometric_js_divergence)
    elif reduction == "sum":
        return torch.sum(geometric_js_divergence)
    else:
        raise ValueError(
            f"UQpy: Invalid reduction: {reduction}. Must be one of 'none', 'mean', or 'sum'"
        )
