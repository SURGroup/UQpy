import torch
import torch.nn.functional as F
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
        **kwargs,
    ):
        r"""Construct a Bayesian layer with weights and bias set by I.I.D. Normal distributions

        :param in_features: Size of each input sample
        :param out_features: Size of each output sample
        :param function: Function to apply to the input on ``self.forward``
        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        :param priors: Prior and posterior distribution parameters. The dictionary keys and their default values are:

         - ``priors["prior_mu"]`` = :math:`0`
         - ``priors["prior_sigma"]`` = :math:`0.1`
         - ``priors["posterior_mu_initial"]`` = ``(0, 0.1)``
         - ``priors["posterior_rho_initial"]`` = ``(-3, 0.1)``
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values.

        Shape:

        - Input: :math:`(*, H_\text{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_\text{in} = \text{in\_features}`.
        - Output: :math:`(*, H_\text{out})` where all but the last dimension
          are the same shape as the input and :math:`H_\text{out} = \text{out\_features}`.

        Example:

            >>> f = sml.BayesianLinear(4, 15)
            >>> x = torch.rand(20, 4)
            >>> f.sample(False)
            >>> deterministic_output = f(x)
            >>> f.sample()
            >>> probabilistic_output = f(x)
            >>> print(deterministic_output.shape)
            >>> print(probabilistic_output.shape)
            >>> print(torch.all(deterministic_output == probabilistic_output))
            torch.Size([20, 15])
            torch.Size([20, 15])
            tensor(False)
        """
        weight_shape = (out_features, in_features)
        bias_shape = out_features if bias else None
        super().__init__(weight_shape, bias_shape, priors, sampling, **kwargs)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward model evaluation

        :param x: Tensor of shape :math:`(*, \text{in_features})`
        :return: Tensor of shape :math:`(*, \text{out_features})`
        """
        weight, bias = self.get_weight_bias()
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
