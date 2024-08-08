import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import BayesianLayer, Loss
from beartype import beartype


@beartype
class GeometricJensenShannonDivergence(Loss):

    def __init__(self, alpha: float, reduction: str = "sum", **kwargs):
        """Analytic form for Geometric JS divergence for all Bayesian layers in a module

        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        if reduction is "none":
            raise ValueError(
                "UQpy: GeometricJensenShannonDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
            )
        self.reduction = reduction

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Geometric JS divergence on all Bayesian layers in a module

        :param network: Module containing Bayesian layers as class attributes
        :return: Geometric JS divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float)
        for layer in network.modules():
            if isinstance(layer, BayesianLayer):
                for name in layer.parameter_shapes:
                    if layer.parameter_shapes[name] is not None:
                        mu = getattr(layer, f"{name}_mu")
                        rho = getattr(layer, f"{name}_rho")
                        divergence += func.geometric_jenson_shannon_divergence(
                            mu,
                            torch.log1p(torch.exp(rho)),
                            torch.tensor(layer.prior_mu, device=mu.device),
                            torch.tensor(layer.prior_sigma, device=mu.device),
                            alpha=self.alpha,
                            reduction=self.reduction,
                        )
        return divergence

    def extra_repr(self) -> str:
        return f"reduction={self.reduction}"
