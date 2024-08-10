import torch
import torch.nn as nn
from UQpy.scientific_machine_learning.baseclass.Layer import Layer
from abc import ABC, abstractmethod
from typing import Union


class BayesianLayer(Layer, ABC):
    def __init__(
        self,
        parameter_shapes: dict,
        priors: dict,
        sampling: bool = True,
        device: Union[torch.device, str] = None,
        dtype: Union[torch.dtype, tuple] = None,
    ):
        """Initialize the random variables governing the parameters of the layer.

        :param parameter_shapes: Dictionary with ``"name"``: ``shape`` pairs for each parameter
        :param priors: Prior and posterior distribution parameters. See table for default values
        :param sampling: If ``True``, sample layer parameters from their respective Gaussian distributions.
         If ``False``, use distribution mean as parameter values. Default: ``True``
        :param device: A ``torch.device`` representing the device on which tensors are allocated
        :param dtype: A ``torch.dtype`` (or tuple of them) representing the data type of the tensor

        +-----------------------------+---------------------+-------------+
        | Key                         | Type                |     Default |
        +=============================+=====================+=============+
        | ``"prior_mu"``              | float               |         0.0 |
        +-----------------------------+---------------------+-------------+
        | ``"prior_sigma"``           | float               |         0.1 |
        +-----------------------------+---------------------+-------------+
        | ``"posterior_mu_initial"``  | tuple[float, float] |  (0.0, 0.1) |
        +-----------------------------+---------------------+-------------+
        | ``"posterior_rho_initial"`` | tuple[float, float] | (-3.0, 0.1) |
        +-----------------------------+---------------------+-------------+

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
        self.sampling = sampling
        """Boolean represents whether this module is in sampling mode or not."""
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
        """Initial posterior rho of the distribution"""

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
