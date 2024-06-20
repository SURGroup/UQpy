import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss
from beartype import beartype


@beartype
class GaussianKullbackLeiblerLoss(Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Gaussian KL divergence on all Bayesian layers in a module

        :param network: Network containing Bayesian layers
        :return: Gaussian KL divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float)
        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                weight_divergence = func.gaussian_kullback_leiber_divergence(
                    layer.weight_mu,
                    torch.log1p(torch.exp(layer.weight_sigma)),
                    layer.prior_mu,
                    layer.prior_sigma,
                )
                divergence += weight_divergence / layer.weight_mu.nelement()
                if layer.bias:
                    bias_divergence = func.gaussian_kullback_leiber_divergence(
                        layer.bias_mu,
                        torch.log1p(torch.exp(layer.bias_sigma)),
                        layer.prior_mu,
                        layer.prior_sigma,
                    )
                    divergence += bias_divergence / layer.bias_mu.nelement()
        return divergence
