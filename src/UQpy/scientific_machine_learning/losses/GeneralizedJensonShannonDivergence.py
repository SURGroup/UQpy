import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Loss, BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger
from beartype import beartype
from beartype.vale import Is
from typing import Annotated


@beartype
class GeneralizedJensonShannonDivergence(Loss):

    def __init__(
        self,
        posterior_distribution,
        prior_distribution,
        alpha: Annotated[float, Is[lambda x: 0 <= x <= 1]] = 0.5,
        n_samples: PositiveInteger = 1_000,
        reduction: str = "sum",
    ):
        """

        :param posterior_distribution:
        :param prior_distribution:
        :param alpha:
        :param n_samples:
        :param reduction:
        """
        super().__init__()
        self.posterior_distribution = posterior_distribution
        self.prior_distribution = prior_distribution
        self.alpha = alpha
        self.n_samples = n_samples
        if reduction is "none":
            raise ValueError(
                "UQpy: GeometricJensenShannonDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
            )
        self.reduction = reduction

    def forward(self, network: nn.Module) -> torch.Tensor:
        """

        :param network:
        :return:
        """
        divergence = torch.tensor(0.0)
        for layer in network.modules():
            if not isinstance(layer, BayesianLayer):
                continue
            posterior_distribution_list = []
            prior_distribution_list = []
            for name in layer.parameter_shapes:
                if layer.parameter_shapes[name] is None:
                    continue
                mu = getattr(layer, f"{name}_mu")
                rho = getattr(layer, f"{name}_rho")
                sigma = torch.log1p(torch.exp(rho))
                for mu_i, sigma_i in zip(mu.flatten(), sigma.flatten()):
                    posterior_distribution_list.append(
                        self.posterior_distribution(mu_i.item(), sigma_i.item())
                    )
                    prior_distribution_list.append(
                        self.prior_distribution(layer.prior_mu, layer.prior_sigma)
                    )
            divergence += func.generalized_jenson_shannon_divergence(
                posterior_distribution_list,
                prior_distribution_list,
                self.n_samples,
                self.alpha,
                self.reduction,
            )

    def extra_repr(self) -> str:
        s = (
            "posterior_distribution={posterior_distribution}, "
            "prior_distribution={prior_distribution}"
        )
        if self.alpha != 0.5:
            s += ", alpha={alpha}"
        if self.n_samples != 1_000:
            s += ", n_samples={n_samples}"
        if self.reduction is not "sum":
            s += ", reduction={reduction}"
        return s.format(__dict__)
