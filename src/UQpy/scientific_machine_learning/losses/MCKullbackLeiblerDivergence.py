import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.distributions.baseclass import Distribution
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer, Loss
from UQpy.utilities.ValidationTypes import PositiveInteger
from typing import Annotated
from beartype import beartype
from beartype.vale import Is



@beartype
class MCKullbackLeiblerDivergence(Loss):

    def __init__(
        self,
        posterior_distribution: Annotated[
            object, Is[lambda x: issubclass(x, Distribution)]
        ],
        prior_distribution: Annotated[
            object, Is[lambda x: issubclass(x, Distribution)]
        ],
        n_samples: PositiveInteger = 1_000,
        reduction: str = "sum",
        device=None,
    ):
        """KL divergence by sampling for all Bayesian layers in a module.

        .. note::
            This is *not* identical to the Kullback-Leibler divergence computed in Bayes by Backprop

        :param posterior_distribution: A class, *not an instance*, of a UQpy distribution defining the variational posterior
        :param prior_distribution: A class, *not an instance*, of a UQpy distribution defining the prior
        :param reduction: Specifies the reduction to apply to the output: 'mean', or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'

        Examples:

        >>> # Divergence of a single Bayesian Layer
        >>> layer = sml.BayesianLinear(4, 5)
        >>> divergence_function = sml.MCKullbackLeiblerDivergence(UQpy.Normal, UQpy.Normal)
        >>> div = divergence_function(layer)

        >>> # Divergence of a Bayesian neural network
        >>> network = nn.Sequential(
        >>>     sml.BayesianLinear(1, 4),
        >>>     nn.ReLU(),
        >>>     nn.Linear(4, 4),
        >>>     nn.ReLU(),
        >>>     sml.BayesianLinear(4, 1),
        >>> )
        >>> model = sml.FeedForwardNeuralNetwork(network)
        >>> divergence_function = sml.MCKullbackLeiblerDivergence(UQpy.Normal, UQpy.Normal)
        >>> div = divergence_function(model)
        """
        super().__init__()
        self.posterior_distribution = posterior_distribution
        self.prior_distribution = prior_distribution
        self.n_samples = n_samples
        self.reduction = reduction
        if self.reduction == "none":
            raise ValueError(
                "UQpy: MCKullbackLeiblerDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
                "\nWe are deeply sorry this is inconsistent with the behavior of mc_kullback_leibler_divergence, "
                "but we had no other choice."
            )
        self.device = device

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the KL divergence by sampling the distributions on all Bayesian layers in a module

        :param network: Network containing Bayesian layers
        :return: KL divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, device=self.device)
        for layer in network.modules():
            if isinstance(layer, NormalBayesianLayer):
                posterior_distributions_list = []
                prior_distributions_list = []
                for name in layer.parameter_shapes:
                    if layer.parameter_shapes[name] is None:
                        continue
                    mu = getattr(layer, f"{name}_mu")
                    rho = getattr(layer, f"{name}_rho")
                    sigma = torch.log1p(torch.exp(rho))
                    for mu_i, sigma_i in zip(
                        mu.flatten(),
                        sigma.flatten(),
                    ):
                        posterior_distributions_list.append(
                            self.posterior_distribution(mu_i.item(), sigma_i.item())
                        )
                        prior_distributions_list.append(
                            self.prior_distribution(layer.prior_mu, layer.prior_sigma)
                        )

                divergence += func.mc_kullback_leibler_divergence(
                    posterior_distributions_list,
                    prior_distributions_list,
                    n_samples=self.n_samples,
                    reduction=self.reduction,
                )

        return divergence

    def extra_repr(self) -> str:
        s = (
            f"posterior_distribution={self.posterior_distribution}, "
            f"prior_distribution={self.prior_distribution}"
        )
        if self.n_samples != 1_000:
            s += ", n_samples={n_samples}"
        if self.reduction != "sum":
            s += f", reduction={self.reduction}"
        return s
