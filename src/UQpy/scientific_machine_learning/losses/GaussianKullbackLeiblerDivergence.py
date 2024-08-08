import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss
from beartype import beartype


@beartype
class GaussianKullbackLeiblerDivergence(Loss):

    def __init__(self, reduction: str = "sum"):
        """Analytic form for Gaussian KL divergence for all Bayesian layers in a module

        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'
        """
        super().__init__()
        if reduction is "none":
            raise ValueError(
                "UQpy: GaussianKullbackLeiblerDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
                "\nWe are deeply sorry this is inconsistent with the behavior of gaussian_kullback_leiber_divergence, "
                "but we had no other choice."
            )
        self.reduction = reduction

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Gaussian KL divergence on all Bayesian layers in a module

        :param network: Module containing Bayesian layers as class attributes
        :return: Gaussian KL divergence between prior and posterior distributions
        """
        device = network.device
        divergence = torch.tensor(0.0, dtype=torch.float, device=device)
        for layer in network.modules():
            if not isinstance(layer, BayesianLayer):
                continue
            for name in layer.parameter_shapes:
                if layer.parameter_shapes[name] is None:
                    continue
                mu = getattr(layer, f"{name}_mu")
                rho = getattr(layer, f"{name}_rho")
                divergence += func.gaussian_kullback_leiber_divergence(
                    mu,
                    torch.log1p(torch.exp(rho)),
                    torch.tensor(layer.prior_mu, device=device),
                    torch.tensor(layer.prior_sigma, device=device),
                    reduction=self.reduction,
                )
        return divergence

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
