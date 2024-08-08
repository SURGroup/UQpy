import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss
from beartype import beartype


@beartype
class GaussianJensonShannonDivergence(Loss):

    def __init__(self, reduction: str = "sum", device=None):
        """Analytic form for Gaussian JS divergence for all Bayesian layers in a module

        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'
        """
        super().__init__()
        if reduction is "none":
            raise ValueError(
                "UQpy: GaussianJensonShannonDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
                "\nWe are deeply sorry this is inconsistent with the behavior of gaussian_jenson_shannon_divergence, "
                "but we had no other choice."
            )
        self.reduction = reduction
        self.device = device

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Gaussian JS divergence on all Bayesian layers in a module

        :param network: Network containing Bayesian layers
        :return: Gaussian JS divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for layer in network.modules():
            if not isinstance(layer, BayesianLayer):
                continue
            for name in layer.parameter_shapes:
                if layer.parameter_shapes[name] is None:
                    continue
                posterior_mu = getattr(layer, f"{name}_mu")
                rho = getattr(layer, f"{name}_rho")
                posterior_sigma = torch.log1p(torch.exp(rho))
                divergence += func.gaussian_jenson_shannon_divergence(
                    posterior_mu,
                    posterior_sigma,
                    torch.tensor(layer.prior_mu, device=self.device),
                    torch.tensor(layer.prior_sigma, device=self.device),
                    reduction=self.reduction,
                )
        return divergence

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
