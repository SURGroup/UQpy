import torch
import torch.nn.functional as F
from typing import Union
from UQpy.scientific_machine_learning.baseclass import BayesianLayer
from UQpy.utilities.ValidationTypes import PositiveInteger


class BayesianLinear(BayesianLayer):

    def __init__(
        self,
        in_features: PositiveInteger,
        out_features: PositiveInteger,
        bias: bool = True,
        priors: dict = None,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
        dtype: torch.dtype = None,
    ):
        r"""Construct a Bayesian Linear layer as :math:`xA^T + b`
        where :math:`A` and :math:`b` are normal random variables.

        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        :param priors: Prior and posterior distribution parameters.
         The dictionary keys and their default values are:

         - "prior_mu": 0
         - "prior_sigma" : 0.1
         - "posterior_mu_initial": (0.0, 0.1)
         - "posterior_rho_initial": (-3.0, 0.1)
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``

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
        - **weight_rho** (:py:class:`torch.nn.Parameter`): The learnable distribution variance of the
          weights of shape :math:`(\text{out_features}, \text{in_features})`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
        - **bias_mu** (:py:class:`torch.nn.Parameter`): The learnable distribution mean of the
          bias of shape :math:`(\text{out_features})`.
          If ``bias`` is ``True``, the values are initialized from
          :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - **bias_rho** (:py:class:`torch.nn.Parameter`): The learnable distributinon variance of the
          bias of shape :math:`(\text{out_features})`.
          The variance is computed as :math:`\sigma = \ln( 1 + \exp(\rho))` to guarantee it is positive.
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
        super().__init__(parameter_shapes, priors, sampling, device, dtype)
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
