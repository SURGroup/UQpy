import torch
import torch.nn as nn
import torchinfo
from abc import ABC, abstractmethod
from UQpy.scientific_machine_learning.baseclass import BayesianLayer


class NeuralNetwork(nn.Module, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def forward(self, **kwargs):
        """Define the computation at every model call. Inherited from :code:`torch.nn.Module`.
        See `Pytorch documentation`_ for details

        .. _Pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.forward
        """
        ...

    def summary(self, **kwargs):
        """Call `torchinfo.summary()` on `self`"""
        return torchinfo.summary(self, **kwargs)

    def count_parameters(self):
        """Get the total number of parameters that require a gradient computation in the model"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def sample(self, mode: bool = True):
        """Set sampling mode for Neural Network and all child modules

        Note: Based on the `torch.nn.Module.train` and `torch.nn.Module.training` method and attributes

        :param mode:
        :return: ``self``
        """
        self.sampling = mode
        for m in self.network.modules():
            if hasattr(m, "sample"):
                m.sample(mode)
        return self

    def drop(self, mode: bool = True):
        """

        """
        self.dropping = mode
        for m in self.network:
            if hasattr(m, "drop"):
                m.drop(mode)
        return self

    def is_deterministic(self) -> bool:
        """Check if neural network is behaving deterministically or probabilistically

        :return: ``True`` if output is deterministic, ``False`` if output is probabilistic
        """
        return not self.sampling and not self.training

    def compute_kullback_leibler_divergence(self) -> float:
        """Computes the Kullback-Leibler divergence between the current and prior ``network`` parameters

        :return: Kullback-Leibler divergence
        """
        kl = 0
        for layer in self.network.modules():
            if isinstance(layer, BayesianLayer):
                kl += gaussian_kullback_leibler_divergence(
                    layer.weight_mu,
                    torch.log1p(torch.exp(layer.weight_sigma)),
                    layer.prior_mu,
                    layer.prior_sigma,
                )
                if layer.bias:
                    kl += gaussian_kullback_leibler_divergence(
                        layer.bias_mu,
                        torch.log1p(torch.exp(layer.bias_sigma)),
                        layer.prior_mu,
                        layer.prior_sigma,
                    )
        return kl


def gaussian_kullback_leibler_divergence(
    mu_posterior: torch.Tensor,
    sigma_posterior: torch.Tensor,
    mu_prior: torch.Tensor,
    sigma_prior: torch.Tensor,
) -> torch.Tensor:
    """Compute the Gaussian closed-form Kullback-Leibler Divergence

    :param mu_posterior: Mean of the Gaussian variational posterior
    :param sigma_posterior: Standard deviation of the Gaussian variational posterior
    :param mu_prior: Mean of the Gaussian prior
    :param sigma_prior: Standard deviation of the Gaussian prior
    :return: KL Divergence from Gaussian :math:`p` to Gaussian :math:`q`
    """
    kl = (
        0.5
        * (
            2 * torch.log(sigma_prior / sigma_posterior)
            - 1
            + (sigma_posterior / sigma_prior).pow(2)
            + ((mu_prior - mu_posterior) / sigma_prior).pow(2)
        ).sum()
    )
    assert not torch.isnan(kl)
    return kl
