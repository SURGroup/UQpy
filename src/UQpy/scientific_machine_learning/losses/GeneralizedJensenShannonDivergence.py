import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import Loss, BayesianLayer

from typing import Annotated
from beartype import beartype
from beartype.vale import Is
from UQpy.distributions.baseclass import Distribution
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
class GeneralizedJensenShannonDivergence(Loss):

    def __init__(
        self,
        posterior_distribution: Annotated[
            object, Is[lambda x: issubclass(x, Distribution)]
        ],
        prior_distribution: Annotated[
            object, Is[lambda x: issubclass(x, Distribution)]
        ],
        alpha: Annotated[float, Is[lambda x: 0 <= x <= 1]] = 0.5,
        n_samples: PositiveInteger = 1_000,
        reduction: str = "sum",
        device=None,
    ):
        r"""Estimate the Jensen-Shannon divergence using Monte Carlo sampling for all Bayesian layers in a module

        :param posterior_distribution: A class, *not an instance*, of a UQpy distribution defining the variational posterior
        :param prior_distribution: A class, *not an instance*, of a UQpy distribution defining the prior
        :param alpha: Weight of the mixture distribution, :math:`0 \leq \alpha \leq 1`.
         See formula for details. Default: 0.5
        :param n_samples: Number of samples using in the Monte Carlo estimates. Default: 1,000
        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'

        Formula
        -------
        The Jenson-Shannon divergence :math:`D_{JS}` is computed as

        .. math:: D_{JS}(Q, P) = (1-\alpha) D_{KL}(Q, M) + \alpha D_{KL}(P, M)

        where :math:`D_{KL}` is the Kullback-Leibler divergence and :math:`M=\alpha Q + (1-\alpha) P` is the mixture distribution.

        """
        super().__init__()
        self.posterior_distribution = posterior_distribution
        self.prior_distribution = prior_distribution
        self.alpha = alpha
        self.n_samples = n_samples
        if reduction is "none":
            raise ValueError(
                "UQpy: GeneralizedJensenShannonDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
                "\nWe are deeply sorry this is inconsistent with the behavior of generalized_jensen_shannon_divergence,"
                " but we had no other choice."
            )
        self.reduction = reduction
        self.device = device

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Generalized Jensen-Shannon divergence on all Bayesian layers in a module

        :param network: Module containing Bayesian layers as class attributes
        :return: Generalized JS divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, device=self.device)
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
            divergence += func.generalized_jensen_shannon_divergence(
                posterior_distribution_list,
                prior_distribution_list,
                self.n_samples,
                self.alpha,
                self.reduction,
                device=self.device,
            )
        return divergence

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
