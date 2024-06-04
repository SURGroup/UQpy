import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss


class GaussianKullbackLeibler(Loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, network: nn.Module) -> torch.Tensor:
        divergence = torch.tensor(0)
        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                divergence += self.compute_gkl_divergence(
                    layer.weight_mu,
                    torch.log1p(torch.exp(layer.weight_sigma)),
                    layer.prior_mu,
                    layer.prior_sigma,
                )
                if layer.bias:
                    divergence += self.compute_gkl_divergence(
                        layer.bias_mu,
                        torch.log1p(torch.exp(layer.bias_sigma)),
                        layer.prior_mu,
                        layer.prior_sigma,
                    )
        return divergence

    @staticmethod
    def compute_gkl_divergence(
        posterior_mu: torch.Tensor,
        posterior_sigma: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Gaussian Kullback-Liebler divergence

        :param posterior_mu:
        :param posterior_sigma:
        :param prior_mu:
        :param prior_sigma:
        :return:
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
