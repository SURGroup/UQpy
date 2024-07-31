import torch
import torch.nn as nn
from beartype import beartype

import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss


@beartype
class MCKullbackLeiblerDivergence(Loss):

    def __init__(self, posterior_distribution, prior_distribution, reduction: str = "sum", **kwargs):
        """KL divergence by sampling for all Bayesian layers in a module. Note: This is not same as the implementation in Bayes By Backprop.
        :param posterior_distribution: Specifies the posterior distribution: Function handle to one of UQpy.distributions
        :param prior_distribution: Specifies the prior distribution: Funtion handle to one of UQpy.distributions
        :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
         'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
         Default: 'sum'
        """
        super().__init__(**kwargs)
        self.posterior_distribution = posterior_distribution
        self.prior_distribution = prior_distribution
        self.reduction = reduction

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the KL divergence by sampling the distributions on all Bayesian layers in a module

        :param network: Network containing Bayesian layers
        :return: KL divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float)
        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                posterior_distributions_list = []
                prior_distributions_list = []
                for post_mu, post_sigma in zip(layer.weight_mu.flatten(),
                                               torch.log1p(torch.exp(layer.weight_rho)).flatten()):
                    posterior_distributions_list.append(self.posterior_distribution(post_mu.item(), post_sigma.item()))
                    prior_distributions_list.append(self.prior_distribution(layer.prior_mu, layer.prior_sigma))

                divergence += func.mc_kullback_leibler_divergence(posterior_distributions_list,
                                                                  prior_distributions_list, reduction=self.reduction)

                if layer.bias:
                    posterior_distributions_list = []
                    prior_distributions_list = []
                    for post_mu, post_sigma in zip(layer.bias_mu, torch.log1p(torch.exp(layer.bias_rho))):
                        posterior_distributions_list.append(
                            self.posterior_distribution(post_mu.item(), post_sigma.item()))
                        prior_distributions_list.append(self.prior_distribution(layer.prior_mu, layer.prior_sigma))

                    divergence += func.mc_kullback_leibler_divergence(posterior_distributions_list,
                                                                      prior_distributions_list,
                                                                      reduction=self.reduction)
        return divergence

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
