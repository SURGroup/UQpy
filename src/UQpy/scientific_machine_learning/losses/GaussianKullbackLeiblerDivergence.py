import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer, Loss
from beartype import beartype


@beartype
class GaussianKullbackLeiblerDivergence(Loss):

    def __init__(self, reduction: str = "sum", device=None):
        r"""Analytic form for Gaussian KL divergence for all Bayesian layers in a module

        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'

        The Gaussian Kullback-Leibler divergence :math:`D_{KL}` for two univariate normal distributions is computed as

        .. math:: D_{KL}(p, q) = \frac{1}{2} \left( 2\log \frac{\sigma_q}{\sigma_p} + \frac{\sigma_p^2 + (\mu_q-\mu_p)^2}{\sigma_q^2} - 1 \right)

        Examples:

        >>> # Divergence of a single Bayesian Layer
        >>> layer = sml.BayesianLinear(4, 5)
        >>> divergence_function = sml.GaussianKullbackLeiblerDivergence()
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
        >>> divergence_function = sml.GaussianKullbackLeiblerDivergence()
        >>> div = divergence_function(model)
        """
        super().__init__()
        if reduction == "none":
            raise ValueError(
                "UQpy: GaussianKullbackLeiblerDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
                "\nWe are deeply sorry this is inconsistent with the behavior of gaussian_kullback_leibler_divergence, "
                "but we had no other choice."
            )
        self.reduction = reduction
        self.device = device

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Gaussian KL divergence on all Bayesian layers in a module

        :param network: Module containing Bayesian layers as class attributes
        :return: Gaussian KL divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, dtype=torch.float, device=self.device)
        for layer in network.modules():
            if not isinstance(layer, NormalBayesianLayer):
                continue
            for name in layer.parameter_shapes:
                if layer.parameter_shapes[name] is None:
                    continue
                mu = getattr(layer, f"{name}_mu")
                rho = getattr(layer, f"{name}_rho")
                divergence += func.gaussian_kullback_leibler_divergence(
                    mu,
                    torch.log1p(torch.exp(rho)),
                    torch.tensor(layer.prior_mu, device=self.device),
                    torch.tensor(layer.prior_sigma, device=self.device),
                    reduction=self.reduction,
                )
        return divergence

    def extra_repr(self) -> str:
        if self.reduction != "sum":
            return f"reduction={self.reduction}"
        return ""
