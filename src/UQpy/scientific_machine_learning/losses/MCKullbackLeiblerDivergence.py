import torch
import torch.nn as nn
from beartype import beartype

import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss


@beartype
class MCKullbackLeiblerDivergence(Loss):

    def __init__(
        self,
        posterior_distribution,
        prior_distribution,
        reduction: str = "sum",
    ):
        """KL divergence by sampling for all Bayesian layers in a module. Note: This is not same as the implementation in Bayes By Backprop.

        :param posterior_distribution: Specifies the posterior distribution: Function handle to one of UQpy.distributions
        :param prior_distribution: Specifies the prior distribution: Function handle to one of UQpy.distributions
        :param reduction: Specifies the reduction to apply to the output: 'mean', or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'
        """
        super().__init__()
        self.posterior_distribution = posterior_distribution
        self.prior_distribution = prior_distribution
        self.reduction = reduction
        if self.reduction is "none":
            raise ValueError(
                "UQpy: MCKullbackLeiblerDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
                "\nWe are deeply sorry this is inconsistent with the behavior of mc_kullback_leibler_divergence, "
                "but we had no other choice."
            )

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the KL divergence by sampling the distributions on all Bayesian layers in a module

        :param network: Network containing Bayesian layers
        :return: KL divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0)

        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                posterior_distributions_list = []
                prior_distributions_list = []

                for name in layer.parameter_shapes:
                    if layer.parameter_shapes[name] is None:
                        continue
                    mu = getattr(layer, f"{name}_mu")
                    rho = getattr(layer, f"{name}_rho")
                    sigma = torch.log1p(torch.exp(rho))
                    for mu_i, sigma_i in zip(
                        mu.flatten(),
                        sigma.flatten(),
                    ):
                        posterior_distributions_list.append(
                            self.posterior_distribution(mu_i.item(), sigma_i.item())
                        )
                        prior_distributions_list.append(
                            self.prior_distribution(layer.prior_mu, layer.prior_sigma)
                        )

                divergence += func.mc_kullback_leibler_divergence(
                    posterior_distributions_list,
                    prior_distributions_list,
                    reduction=self.reduction,
                )

        return divergence

    def extra_repr(self) -> str:
        s = (
            f"posterior_distribution={self.posterior_distribution}, "
            f"prior_distribution={self.prior_distribution}"
        )
        if self.reduction is not "sum":
            s += f", reduction={self.reduction}"
        return s
