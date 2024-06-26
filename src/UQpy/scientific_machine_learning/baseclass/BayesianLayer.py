import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from abc import ABC, abstractmethod
from typing import Union


class BayesianLayer(Layer, ABC):
    def __init__(self, weight_shape, bias_shape, priors, sampling=True, dtype=torch.float, **kwargs):
        """Initialize the random variables governing the parameters of the layer.

        :param weight_shape: Shape of the weight_mu and weight_sigma matrices
        :param bias_shape: Shape of the bias_mu and bias_sigma matrices
        :param priors: Prior and posterior distribution parameters. See table for default values
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``

        ``priors`` Keys and Defaults
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        .. list-table::
           :widths: 50 35 25
           :header-rows: 1

           * - Key
             - Type
             - Default
           * - ``"prior_mu"``
             - float
             - 0.0
           * - ``"prior_sigma"``
             - float
             - 0.1
           * - ``"posterior_mu_initial"``
             - tuple[float, float]
             - (0.0, 0.1)
           * - ``"posterior_rho_initial"``
             - tuple[float, float]
             - (-3.0, 0.1)
        :param dtype:
        """
        super().__init__(**kwargs)
        self.sampling = sampling
        self.priors = priors
        if self.priors is None:
            priors = {
                "prior_mu": 0.0,
                "prior_sigma": 0.1,
                "posterior_mu_initial": (0.0, 0.1),
                "posterior_rho_initial": (-3.0, 0.1),
            }
        self.prior_mu: float = priors["prior_mu"]
        """Prior mean of the normal distribution"""
        self.prior_sigma: float = priors["prior_sigma"]
        """Prior standard deviation of the normal distribution"""
        self.posterior_mu_initial: tuple[float, float] = priors["posterior_mu_initial"]
        """Initial posterior mean of the distribution"""
        self.posterior_rho_initial: tuple[float, float] = priors[
            "posterior_rho_initial"
        ]
        """initial posterior rho of the distribution"""
        self.weight_mu: nn.Parameter = nn.Parameter(torch.empty(weight_shape, dtype=dtype))
        """Distribution means for the weights"""
        self.weight_sigma: nn.Parameter = nn.Parameter(torch.empty(weight_shape, dtype=dtype))
        """Distribution standard deviations for the weights"""

        self.bias: bool = True if bias_shape else False
        """If ``True``, add bias. Inferred from ``bias_shape``"""
        self.bias_mu: Union[None, nn.Parameter] = None
        """Distribution means for the bias. If ``bias`` is ``False``, this is ``None``."""
        self.bias_sigma: Union[None, nn.Parameter] = None
        """Distribution standard deviations for the bias. If ``bias`` is ``False``, this is ``None``."""
        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(bias_shape, dtype=dtype))
            self.bias_sigma = nn.Parameter(torch.empty(bias_shape, dtype=dtype))

        self.sample_parameters()

    def sample_parameters(self):
        """Randomly populate ``weight_mu`` and ``weight_sigma`` with samples from Normal distributions.
        If ``bias`` is ``True``, populate ``bias_mu`` and ``bias_sigma`` with samples from Normal distributions.
        """
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_sigma.data.normal_(*self.posterior_rho_initial)
        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_sigma.data.normal_(*self.posterior_rho_initial)

    def get_weight_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the weights and biases for the Bayesian layer.

        If ``training`` or ``sampling`` is ``True``, then sample weights and biases from their respective distributions.
        Otherwise, use distribution means for weights and biases.

        :return: weights, biases
        """
        if (
            self.training or self.sampling
        ):  # Randomly sample weights and biases from normal distribution
            weight_epsilon = torch.empty(self.weight_mu.size()).normal_(0, 1)
            w_sigma = torch.log1p(torch.exp(self.weight_sigma))
            weights = self.weight_mu + (weight_epsilon * w_sigma)
            if self.bias:
                bias_epsilon = torch.empty(self.bias_mu.size()).normal_(0, 1)
                b_sigma = torch.log1p(torch.exp(self.bias_sigma))
                biases = self.bias_mu + (bias_epsilon * b_sigma)
            else:
                biases = None
        else:  # Use mean values for weights and biases
            weights = self.weight_mu
            biases = self.bias_mu if self.bias else None
        return weights, biases

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def extra_repr(self):
        pass
