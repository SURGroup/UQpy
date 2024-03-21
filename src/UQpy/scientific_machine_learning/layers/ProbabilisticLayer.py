import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from typing import Union
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from UQpy.distributions.baseclass import Distribution
from UQpy.distributions.collection import Normal, JointIndependent
from UQpy.utilities.ValidationTypes import PositiveInteger


@beartype
class ProbabilisticLayer(Layer):
    def __init__(
        self,
        in_features: PositiveInteger,
        out_features: PositiveInteger,
        function: nn.Module = F.linear,
        weight_distribution: Distribution = Normal(),
        bias_distribution: Union[None, Distribution] = Normal(),
        sample: bool = True,
        **kwargs
    ):
        """Construct a Probabilistic layer with weights and bias set by independent distributions

        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param function: Function to apply to the input on ``self.forward``
        :param weight_distribution: Random variable for weight distribution
        :param bias_distribution: Random variable for bias distribution. If ``None`` bias is zero
        :param sample: If ``True`` sample weight and bias from their distribution.
        If ``False`` weight and bias are their means
        """
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.function = function
        self.bias_distribution = bias_distribution
        self.weight_distribution = weight_distribution
        self.sample = sample

        self.weight: torch.Tensor = torch.empty((out_features, in_features))
        """Weights of the layer"""
        self.weight_parameters = nn.Parameter(
            torch.ones(
                (self.weight.nelement(), len(self.weight_distribution.parameters))
            )
        )
        """Parameters of the weight distribution for each element in ``weight``"""
        for i, param in enumerate(self.weight_distribution.parameters):
            self.weight_parameters[:, i].data *= self.weight_distribution.parameters[param]
        if self.bias_distribution:
            self.bias = torch.empty((out_features,))
            """Bias of the layer"""
            self.bias_parameters = nn.Parameter(
                torch.ones(
                    (self.bias.nelement(), len(self.bias_distribution.parameters))
                )
            )
            """Parameters of the bias distribution for each element in ``bias``"""
            for i, param in enumerate(self.bias_distribution.parameters):
                self.bias_parameters[:, i].data *= self.bias_distribution.parameters[param]
        else:
            self.bias = None
            self.bias_parameters = None
        self._sample_weight()
        self._sample_bias()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model evaluation

        Note: If ``self.sample`` or ``self.training`` is ``True``,
        then weight and bias are sampled from their distributions and the output is probabilistic.
        If both are ``False``, weight and bias are mean of their distributions and output is deterministic.

        :param x: Input tensor
        :return: Output tensor
        """
        if self.training or self.sample:
            self._sample_weight()
            self._sample_bias()
        else:
            random_variable = self._get_joint_independent(
                self.weight_distribution, self.weight_parameters
            )
            mean = torch.tensor(
                random_variable.moments(moments2return="m"), dtype=torch.float
            )
            self.weight.data = mean.reshape(self.weight.shape)
            if self.bias_distribution:
                random_variable = self._get_joint_independent(
                    self.bias_distribution, self.bias_parameters
                )
                mean = torch.tensor(
                    random_variable.moments(moments2return="m"), dtype=torch.float
                )
                self.bias.data = mean.reshape(self.bias.shape)
        return self.function(x, self.weight, bias=self.bias)

    def _sample_weight(self):
        random_variable = self._get_joint_independent(
            self.weight_distribution, self.weight_parameters
        )
        samples = torch.tensor(random_variable.rvs(), dtype=torch.float)
        self.weight.data = samples.reshape(self.weight.shape)

    def _sample_bias(self):
        if self.bias_distribution:
            random_variable = self._get_joint_independent(
                self.bias_distribution, self.bias_parameters
            )
            samples = torch.tensor(random_variable.rvs(), dtype=torch.float)
            self.bias.data = samples.reshape(self.bias.shape)

    def _get_joint_independent(self, base_distribution: Distribution, parameters: torch.Tensor):
        """Construct a really, really high dimensional random variable"""
        distributions = []
        for i in range(parameters.shape[0]):
            parameters_dict = {
                param: parameters[i, j].item()
                for j, param in enumerate(base_distribution.ordered_parameters)
            }
            distributions.append(base_distribution.__class__(**parameters_dict))
        return JointIndependent(distributions)
