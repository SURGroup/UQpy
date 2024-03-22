import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork


class BayesianNeuralNetwork(NeuralNetwork):

    def __init__(self):
        self.is_deterministic = False

    @property
    def optimizer(self):
        pass

    @property
    def loss_function(self):
        pass

    def calculate_gaussain_kl_divergence(
        self,
        mu_posterior: torch.Tensor,
        sigma_posterior: torch.Tensor,
        mu_prior: torch.Tensor,
        sigma_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Gaussian closed-form Kullback-Leibler Divergence

        # ToDo: is this class necessary? how is it different from VanillaNN?

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
        return kl

    def forward(self, **kwargs):
        pass

    def learn(self, **kwargs):
        pass
