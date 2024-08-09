import torch
import UQpy.scientific_machine_learning.functional as func
from beartype import beartype


@beartype
def geometric_jenson_shannon_divergence(
    posterior_mu: torch.Tensor,
    posterior_sigma: torch.Tensor,
    prior_mu: torch.Tensor,
    prior_sigma: torch.Tensor,
    alpha: float,
    reduction: str = "sum",
) -> torch.Tensor:
    r"""Compute the Geometric Jenson-Shannon divergence for a Gaussian prior and Gaussian posterior distributions

    :param posterior_mu: Mean of the posterior distribution
    :param posterior_sigma: Standard deviation of the posterior distribution
    :param prior_mu: Mean of the prior distribution
    :param prior_sigma: Standard deviation of the prior distribution
    :param alpha: Geometric mean weight
    :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
     'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
     Default: 'sum'

    :return: Gaussian JS divergence between prior and posterior distributions

    Formula
    -------
    The Geometric Jenson-Shannon divergence :math:`D_{JSG}` is computed as

    .. math:: D_{JSG}(P, Q) = (1-\alpha)  D_{KL}(P, M) + \alpha D_{KL}(Q, M)

    where :math:`D_{KL}` is the Kullback-Leiber divergence and :math:`M=P^\alpha Q^{(1-\alpha)}` is the geometric
    mean distribution. When the distributions P and Q are Gaussian, the closed form for Geometric Jenson-Shannon
    divergence is given as

    .. math:: D_{JSG}(P, Q) &= 0.5* [ \frac{(1-\alpha)\sigma_0^2 + \alpha \sigma_1^2}{\sigma_alpha^2} + \log \frac{\sigma_\alpha^2}{\sigma_0^{2(1-\alpha)} \sigma_1^{2\alpha}} + (1-\alpha) \frac{(\mu_\alpha - \mu_0)^2}{\sigma_\alpha^2} + \frac{\alpha(\mu_alpha - \mu_1)^2}{\sigma_\alpha^2} -1]

    where :math:`\sigma_alpha^2 = \frac{1}{\frac{\alpha}{\sigma_0^2}+\frac{1-\alpha}{\sigma_1^2}}` and :math:`\mu_alpha = \sigma_\alpha^2 [\frac{\alpha \mu_0}{\sigma_0^2} + \frac{(1-\alpha)\mu_1}{\sigma_1^2}]`
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
