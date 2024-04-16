import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


#@beartype
class BayesianLayer(Layer):
    def __init__(
        self,
        in_features: PositiveInteger,
        out_features: PositiveInteger,
        function: nn.Module = F.linear,
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        **kwargs,
    ):
        """Construct a Bayesian layer with weights and bias set by I.I.D. Normal distributions

        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param function: Function to apply to the input on ``self.forward``
        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        :param priors:
        :param sampling:
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.function = function
        self.sampling = sampling

        if priors is None:
            priors = {
                "prior_mu": 0,
                "prior_sigma": 0.1,
                "posterior_mu_initial": (0, 0.1),
                "posterior_rho_initial": (-3, 0.1),
            }
        self.prior_mu = priors["prior_mu"]
        self.prior_sigma = priors["prior_sigma"]
        self.posterior_mu_initial = priors["posterior_mu_initial"]
        self.posterior_rho_initial = priors["posterior_rho_initial"]

        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.bias_mu = None
            self.bias_sigma = None
        self._sample_parameters()

    def _sample_parameters(self):
        """Randomly populate weights and biases with samples from Normal distributions"""
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_sigma.data.normal_(*self.posterior_rho_initial)
        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_sigma.data.normal_(*self.posterior_rho_initial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model evaluation

        :param x: Input tensor
        :param sample: If ``True`` sample weights and bias from Gaussian distribution. If ``False`` use mean values.
        :return: output tensor
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

        return self.function(x, weights, biases)

    def sample(self, mode: bool = True):
        """Set sampling mode for Neural Network and all child modules

        Note: Based on the `torch.nn.Module.train` and `torch.nn.Module.training` implementation

        :param mode:
        :return:
        """
        self.sampling = mode
        for child in self.children():
            child.sample(mode=mode)
        return self

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, sampling={self.sampling}"
