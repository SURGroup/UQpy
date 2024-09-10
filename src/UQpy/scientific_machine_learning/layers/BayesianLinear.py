import torch
import torch.nn.functional as F
from typing import Union
from UQpy.scientific_machine_learning.baseclass import NormalBayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger, PositiveFloat


class BayesianLinear(NormalBayesianLayer):

    def __init__(
        self,
        in_features: PositiveInteger,
        out_features: PositiveInteger,
        bias: bool = True,
        sampling: bool = True,
        prior_mu: float = 0.0,
        prior_sigma: PositiveFloat = 0.1,
        posterior_mu_initial: tuple[float, PositiveFloat] = (0.0, 0.1),
        posterior_rho_initial: tuple[float, PositiveFloat] = (-3.0, 0.1),
        device: Union[torch.device, str] = None,
        dtype: torch.dtype = None,
    ):
        r"""Construct a Bayesian Linear layer as :math:`xA^T + b`
        where :math:`A` and :math:`b` are normal random variables.

        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True`
        :param prior_mu: Prior mean, :math:`\mu_\text{prior}` of the prior normal distribution.
         Default: 0.0
        :param prior_sigma: Prior standard deviation, :math:`\sigma_\text{prior}`, of the prior normal distribution.
         Default: 0.1
        :param posterior_mu_initial: Mean and standard deviation of the initial posterior distribution for :math:`\mu`.
         The initial posterior is :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
         Default: (0.0, 0.1)
        :param posterior_rho_initial: Mean and standard deviation of the initial posterior distribution for :math:`\rho`.
         The initial posterior is :math:`\mathcal{N}(\rho_\text{posterior}[0], \rho_\text{posterior}[1])`.
         The standard deviation of the posterior is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to ensure it is positive.
         Default: (-3.0, 0.1)

        Shape:

        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out_features}`.

        Attributes:

        Unless otherwise noted, all parameters are initialized using the ``priors`` with values
        from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        - **weight_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the
          weights of shape :math:`(\text{out_features}, \text{in_features})`.
        - **weight_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution standard deviation of the
          weights of shape :math:`(\text{out_features}, \text{in_features})`.
          The standard deviation is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the
          bias of shape :math:`(\text{out_features})`.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_rho** (:py:class:`torch.nn.Parameter`): The learnable distributinon standard deviation of the
          bias of shape :math:`(\text{out_features})`.
          The standard deviation is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.

        Example:

        >>> layer = sml.BayesianLinear(4, 15)
        >>> input = torch.rand(20, 4)
        >>> layer.sample(False)
        >>> deterministic_output = layer(input)
        >>> layer.sample()
        >>> probabilistic_output = layer(input)
        >>> print(torch.all(deterministic_output == probabilistic_output))
        tensor(False)
        """
        parameter_shapes = {
            "weight": (out_features, in_features),
            "bias": out_features if bias else None,
        }
        super().__init__(
            parameter_shapes,
            sampling,
            prior_mu,
            prior_sigma,
            posterior_mu_initial,
            posterior_rho_initial,
            device,
            dtype,
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward model evaluation

        :param x: Tensor of shape :math:`(*, \text{in_features})`
        :return: Tensor of shape :math:`(*, \text{out_features})`
        """
        weight, bias = self.get_bayesian_weights()
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        s = "in_features={in_features}, out_features={out_features}"
        if self.bias is False:
            s += ", bias={bias}"
        if self.priors:
            s += ", priors={priors}"
        if self.sampling is False:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)
