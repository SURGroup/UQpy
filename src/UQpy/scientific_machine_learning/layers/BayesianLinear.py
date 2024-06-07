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
        """Construct a Bayesian layer with weights and bias set by I.I.D. Normal distributions

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

        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \\text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \\text{out\_features}`.

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

        :param x: Input tensor
        :return: Output tensor
        """
        weight, bias = self.get_weight_bias()
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        s = "{in_features}, {out_features}"
        if not self.bias:
            s += ", bias={bias}"
        if self.priors:
            s += ", priors={priors}"
        if not self.sampling:
            s += ", sampling={sampling}"
        return s.format(**self.__dict__)
