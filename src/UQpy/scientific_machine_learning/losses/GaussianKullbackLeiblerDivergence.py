import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss
from beartype import beartype


@beartype
class GaussianKullbackLeiblerDivergence(Loss):

    def __init__(self, reduction: str = "sum", **kwargs):
        """Analytic form for Gaussian KL divergence for all Bayesian layers in a module

        :param reduction: Specifies the reduction to apply to the output: 'none', 'mean', or 'sum'.
         'none': no reduction will be applied, 'mean': the output will be averaged, 'sum': the output will be summed.
         Default: 'sum'
        """
        super().__init__(**kwargs)
        self.reduction = reduction

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Gaussian KL divergence on all Bayesian layers in a module

        :param network: Module containing Bayesian layers as class attributes
        :return: Gaussian KL divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float)
        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                for name in layer.parameter_shapes:
                    if layer.parameter_shapes[name] is not None:
                        mu = getattr(layer, f"{name}_mu")
                        rho = getattr(layer, f"{name}_rho")
                        divergence += func.gaussian_kullback_leiber_divergence(
                            mu,
                            torch.log1p(torch.exp(rho)),
                            layer.prior_mu,
                            layer.prior_sigma,
                        )
                # divergence += func.gaussian_kullback_leiber_divergence(
                #     layer.weight_mu,
                #     torch.log1p(torch.exp(layer.weight_sigma)),
                #     layer.prior_mu,
                #     layer.prior_sigma,
                #     reduction=self.reduction,
                # )
                # if layer.bias:
                #     divergence += func.gaussian_kullback_leiber_divergence(
                #         layer.bias_mu,
                #         torch.log1p(torch.exp(layer.bias_sigma)),
                #         layer.prior_mu,
                #         layer.prior_sigma,
                #         reduction=self.reduction,
                #     )
        return divergence

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
