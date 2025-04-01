import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from abc import ABC, abstractmethod
from typing import Union
from beartype import beartype
from UQpy.utilities.ValidationTypes import PositiveFloat


@beartype
class NormalBayesianLayer(Layer, ABC):
    def __init__(
        self,
        parameter_shapes: dict,
        sampling: bool = True,
        prior_mu: float = 0.0,
        prior_sigma: PositiveFloat = 0.1,
        posterior_mu_initial: tuple[float, PositiveFloat] = (0.0, 0.1),
        posterior_rho_initial: tuple[float, PositiveFloat] = (-3.0, 0.1),
        device: Union[torch.device, str, None] = None,
        dtype: Union[torch.dtype, tuple, None] = None,
    ):
        r"""Initialize the random variables governing the parameters of the layer.

        :param parameter_shapes: Dictionary with ``"name"``: ``shape`` pairs for each parameter.
         For each key in the dictionary, assign learnable parameters ``name_mu`` and ``name_rho`` with the given shape.
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``
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
        :param device: A ``torch.device`` representing the device on which tensors are allocated
        :param dtype: A ``torch.dtype`` (or tuple of them) representing the data type of the tensor

        """
        super().__init__()
        if isinstance(dtype, tuple):
            if len(dtype) != len(parameter_shapes):
                raise RuntimeError(
                    f"UQpy: Invalid dtype: {dtype}.\n"
                    "dtype must be a single data type, or a tuple of data types the same length as `weight_shapes`"
                )
        else:
            dtype = (dtype,) * len(parameter_shapes)
        self.parameter_shapes: dict = parameter_shapes
        """Prefix names and shapes of all learnable parameters"""
        self.sampling: bool = sampling
        """Boolean represents whether this module is in sampling mode or not."""
        self.prior_mu: float = prior_mu
        """Mean of the prior distribution"""
        self.prior_sigma: float = prior_sigma
        """Standard deviation of the prior distribution"""
        self.posterior_mu_initial: tuple[float, float] = posterior_mu_initial
        r"""Posterior means are initialized from a normal distribution :math:`\mathcal{N}(\text{posterior_mu_initial}[0], \text{posterior_mu_initial}[1])`"""
        self.posterior_rho_initial: tuple[float, float] = posterior_rho_initial
        r"""Posterior rhos are initialized from a normal distribution :math:`\mathcal{N}(\text{posterior_rho_initial}[0], \text{posterior_rho_initial}[1])`"""


        for i, name in enumerate(parameter_shapes):
            shape = parameter_shapes[name]
            if shape is None:
                self.register_parameter(f"{name}_mu", None)
                self.register_parameter(f"{name}_rho", None)
            else:
                setattr(
                    self,
                    f"{name}_mu",
                    nn.Parameter(torch.empty(shape, device=device, dtype=dtype[i])),
                )
                setattr(
                    self,
                    f"{name}_rho",
                    nn.Parameter(torch.empty(shape, device=device, dtype=dtype[i])),
                )
        self.reset_parameters()

    def reset_parameters(self):
        """Populate parameters with samples from posterior Normal distributions."""
        for name in self.parameter_shapes:
            if self.parameter_shapes[name] is None:
                continue
            mu = getattr(self, f"{name}_mu")
            mu.data.normal_(*self.posterior_mu_initial)
            rho = getattr(self, f"{name}_rho")
            rho.data.normal_(*self.posterior_rho_initial)

    def get_bayesian_weights(self) -> tuple:
        """Get the weights for the Bayesian layer.

        If ``sampling`` is ``True``, then sample weights from their respective distributions.
        Otherwise, use distribution means for weights.

        :return: Tuple containing weight tensors
        """
        if self.sampling:
            weights = []
            for name in self.parameter_shapes:
                if self.parameter_shapes[name] is None:
                    weights.append(None)
                    continue
                mu = getattr(self, f"{name}_mu")
                rho = getattr(self, f"{name}_rho")
                factory_kwargs = {"device": mu.device, "dtype": mu.dtype}
                epsilon = torch.empty(mu.shape, **factory_kwargs).normal_(0, 1)
                sigma = torch.log1p(torch.exp(rho))
                weight = mu + (epsilon * sigma)
                weights.append(weight)
        else:
            weights = (getattr(self, f"{name}_mu") for name in self.parameter_shapes)
        return tuple(weights)

    def sample(self, mode: bool = True):
        """Set sampling mode for this and all child Modules

        .. note::
            This method and ``self.sampling`` only affects Bayesian layers

        :param mode: If ``True``, sample from distributions, otherwise use distribution means.
         Default: ``True``
        """
        self.sampling = mode
        self.apply(self.__set_sampling)

    @torch.no_grad()
    def __set_sampling(self, m):
        if hasattr(m, "sampling"):
            m.sampling = self.sampling

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def extra_repr(self):
        pass
