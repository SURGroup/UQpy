import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss
from beartype import beartype


@beartype
class GaussianJensonShannonDivergence(Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Gaussian JS divergence on all Bayesian layers in a module

        :param network: Network containing Bayesian layers
        :return: Gaussian JS divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float)
        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                divergence += func.gaussian_jenson_shannon_divergence(
                    layer.weight_mu,
                    torch.log1p(torch.exp(layer.weight_sigma)),
                    layer.prior_mu,
                    layer.prior_sigma,
                )
                if layer.bias:
                    divergence += func.gaussian_jenson_shannon_divergence(
                        layer.bias_mu,
                        torch.log1p(torch.exp(layer.bias_sigma)),
                        layer.prior_mu,
                        layer.prior_sigma,
                    )
        return divergence
