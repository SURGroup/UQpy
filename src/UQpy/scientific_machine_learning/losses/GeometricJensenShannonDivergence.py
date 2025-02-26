import torch
import torch.nn as nn
import UQpy.scientific_machine_learning.functional as func
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer, Loss

from typing import Annotated
from beartype import beartype
from beartype.vale import Is


@beartype
class GeometricJensenShannonDivergence(Loss):

    def __init__(
        self,
        alpha: Annotated[float, Is[lambda x: 0 <= x <= 1]] = 0.5,
        reduction: str = "sum",
        device=None,
    ):
        r"""Analytic form for Geometric JS divergence for all Bayesian layers in a module

        :param alpha: Weight of the mixture distribution, :math:`0 \leq \alpha \leq 1`.
         See formula for details. Default: 0.5
        :param reduction: Specifies the reduction to apply to the output: 'mean' or 'sum'.
         'mean': the output will be averaged, 'sum': the output will be summed. Default: 'sum'

        The Geometric Jensen-Shannon divergence :math:`D_{JSG}` is computed as

        .. math:: D_{JSG}(P, Q) = (1-\alpha)  D_{KL}(P, M) + \alpha D_{KL}(Q, M)

        where :math:`D_{KL}` is the Kullback-Leibler divergence and :math:`M=P^\alpha Q^{(1-\alpha)}` is the geometric
        mean distribution. When the distributions :math:`P` and :math:`Q` are Gaussian, the closed form for Geometric
        Jensen-Shannon divergence is given as

        .. math:: D_{JSG}(P, Q) = \frac12 \left( \frac{(1-\alpha)\sigma_0^2 + \alpha\sigma_1^2}{\sigma_\alpha^2} + \log \frac{\sigma_\alpha^2}{\sigma_0^{2(1-\alpha)} \sigma_1^{2\alpha}} + (1-\alpha) \frac{(\mu_\alpha - \mu_0)^2}{\sigma_\alpha^2} + \frac{\alpha(\mu_\alpha - \mu_1)^2}{\sigma_\alpha^2} -1 \right)

        where :math:`\sigma_\alpha^2 = \left( \frac{\alpha}{\sigma_0^2}+\frac{1-\alpha}{\sigma_1^2} \right)^{-1}`
        and :math:`\mu_\alpha = \sigma_\alpha^2 \left[\frac{\alpha \mu_0}{\sigma_0^2} + \frac{(1-\alpha)\mu_1}{\sigma_1^2}\right]`

        Examples:

        >>> # Divergence of a single Bayesian Layer
        >>> layer = sml.BayesianLinear(4, 5)
        >>> divergence_function = sml.GeometricJensenShannonDivergence()
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
        >>> divergence_function = sml.GeometricJensenShannonDivergence()
        >>> div = divergence_function(model)
        """
        super().__init__()
        self.alpha = alpha
        if reduction == "none":
            raise ValueError(
                "UQpy: GeometricJensenShannonDivergence does not accept reduction='none'. "
                "Must be 'sum' or 'mean'."
            )
        self.reduction = reduction
        self.device = device

    def forward(self, network: nn.Module) -> torch.Tensor:
        """Compute the Geometric JS divergence on all Bayesian layers in a module

        :param network: Module containing Bayesian layers as class attributes
        :return: Geometric JS divergence between prior and posterior distributions
        """
        divergence = torch.tensor(0.0, device=self.device)
        for layer in network.modules():
            if isinstance(layer, NormalBayesianLayer):
                for name in layer.parameter_shapes:
                    if layer.parameter_shapes[name] is not None:
                        mu = getattr(layer, f"{name}_mu")
                        rho = getattr(layer, f"{name}_rho")
                        divergence += func.geometric_jensen_shannon_divergence(
                            mu,
                            torch.log1p(torch.exp(rho)),
                            torch.tensor(layer.prior_mu, device=mu.device),
                            torch.tensor(layer.prior_sigma, device=mu.device),
                            alpha=self.alpha,
                            reduction=self.reduction,
                        )
        return divergence

    def extra_repr(self) -> str:
        s = []
        if self.alpha != 0.5:
            s.append(f"alpha={self.alpha}")
        if self.reduction != "sum":
            s.append(f"reduction={self.reduction}")
        return ", ".join(s)
