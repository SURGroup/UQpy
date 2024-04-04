import torch
import torch.nn as nn
import logging
from beartype import beartype
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.scientific_machine_learning.losses.EvidenceLowerBound import (
    EvidenceLowerBound,
)


@beartype
class BayesianNeuralNetwork(NeuralNetwork):

    def __init__(self, network: nn.Module, **kwargs):
        super().__init__(**kwargs)
        # self.is_deterministic = False
        self.sampling = True
        self.network = network

        self.logger = logging.getLogger(__name__)

    def gaussian_kl_divergence(
        self,
        mu_posterior: torch.Tensor,
        sigma_posterior: torch.Tensor,
        mu_prior: torch.Tensor,
        sigma_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Gaussian closed-form Kullback-Leibler Divergence

        # ToDo: Should this function have inputs? or can it read parameters off network?

        :param mu_posterior: Mean of the Gaussian variational posterior
        :param sigma_posterior: Standard deviation of the Gaussian variational posterior
        :param mu_prior: Mean of the Gaussian prior
        :param sigma_prior: Standard deviation of the Gaussian prior
        :return: KL Divergence from Gaussian :math:`p` to Gaussian :math:`q`
        """
        parameters = self.network.get_parameters()
        kl = (
            0.5
            * (
                2 * torch.log(sigma_prior / sigma_posterior)
                - 1
                + (sigma_posterior / sigma_prior).pow(2)
                + ((mu_prior - mu_posterior) / sigma_prior).pow(2)
            ).sum()
        )
        return kl

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def sample(self, mode: bool = True):
        """Set sampling mode for Neural Network and all child modules

        Note: Based on the `torch.nn.Module.train` and `torch.nn.Module.training` method and attributes

        :param mode:
        :return:
        """
        self.sampling = mode
        for child in self.children():
            child.sample(mode=mode)
        return self

    def is_deterministic(self) -> bool:
        return not self.sampling and not self.training
