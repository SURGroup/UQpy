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
        r"""Construct a Bayesian layer with weights and bias set by I.I.D. Normal distributions

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
         If ``False``, use distribution mean as parameter values.

        Shape:

        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

        Attributes:

        - weight_mu: The learnable weights of the module of shape :math:`(\text{out_features}, \text{in_features})`.
          The values are initialized from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`.
        - weight_rho: The learnable weights of the module of shape :math:`(\text{out\_features}, \text{in_features})`.
          The values are initalized from :math:`\mathcal{N}(\rho_\text{posterior}[0], \rho_\text{posterior}[1])`.
        - bias_mu: The learnable bias of the module of shape :math:`(\text{out_features})`.
          If ``bias`` is ``True``, the values are initialized from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`
        - bias_rho: The learnable bias of the module of shaspe :math:`(\text{out_features})`.
          If ``bias`` is ``True``, the values are initialized from :math:`\mathcal{N}(\mu_\text{posterior}[0], \mu_\text{posterior}[1])`

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
        s = f"in_features={self.in_features}, out_features={self.out_features}"
        if not self.bias:
            s += f", bias={self.bias}"
        if self.priors:
            s += f", priors={self.priors}"
        if not self.sampling:
            s += f", sampling={self.sampling}"
        return s.format(**self.__dict__)
