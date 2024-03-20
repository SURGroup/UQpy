import beartype
import torch
import torch.nn as nn
import torch.nn.functional as F
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
class BayesianLayer(Layer):
    def __init__(
        self,
        in_features: PositiveInteger,
        out_features: PositiveInteger,
        function: nn.Module = F.Linear,
        bias: bool = True,
        **kwargs
    ):
        """Construct a Bayesian layer with weights and bias set by I.I.D. Normal distributions

        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param function: Function to apply to the input on ``self.forward``
        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.function = function

        self.weight_mu = nn.Parameter(torch.empty((out_features, in_features)))
        self.weight_sigma = nn.Parameter(torch.empty((out_features, in_features)))
        if self.bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
        else:
            self.bias_mu = None
            self.bias_sigma = None
        self._reset_parameters()

    def _reset_parameters(self):
        """Randomly populate weights and biases with samples from Normal distributions"""
        self.weight_mu.data.normal_(*self.posterior_mu_initial)
        self.weight_sigma.data.normal_(*self.posterior_rho_initial)
        if self.bias:
            self.bias_mu.data.normal_(*self.posterior_mu_initial)
            self.bias_sigma.data.normal_(*self.posterior_rho_initial)

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """Forward model evaluation

        :param x: Input tensor
        :param sample: If ``True`` sample weights and bias from Gaussian distribution. If ``False`` use mean values.
        :return: output tensor
        """
        if self.training or sample:  # Randomly sample weights and biases from normal distribution
            weight_epsilon = torch.empty(self.W_mu.size()).normal_(0, 1)
            self.weight_sigma = torch.log1p(torch.exp(self.weight_sigma))
            weights = self.weight_mu + (weight_epsilon * self.weight_sigma)
            if self.bias:
                bias_epsilon = torch.empty(self.bias_mu.size()).normal_(0, 1)
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                biases = self.bias_mu + (bias_epsilon * self.bias_sigma)
            else:
                biases = None
        else:  # Use mean values for weights and biases
            weights = self.weight_mu
            biases = self.bias_mu if self.bias else None

        return self.function(x, weights, biases)
