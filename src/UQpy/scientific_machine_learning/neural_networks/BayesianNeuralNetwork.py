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

    @property
    def optimizer(self):
        return torch.optim.Adam(self.parameters())

    @property
    def loss_function(self):
        train_size = 1
        return EvidenceLowerBound(train_size, nn.MSELoss(reduction="mean"))

    def gaussain_kl_divergence(
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

    def learn(
        self,
        data_loader: torch.utils.data.DataLoader,
        kl_weight: float = 0.1,
        epochs: int = 100,
    ):
        """

        Note: Validation method for neural networks should be able to make multiple forward calls for UQ

        :param data_loader:
        :param kl_weight:
        :param epochs:
        """
        self.network.train(True)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Beginning training BayesianNeuralNetwork"
        )
        self.history["train loss"] = torch.full((epochs,), torch.nan)
        for i in range(epochs):
            for batch, (x, y) in enumerate(data_loader):
                prediction = self.forward(x)
                kl = self.gaussain_kl_divergence()
                loss = self.loss_function(prediction, y, kl, kl_weight)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.logger.info(
                f"UQpy: Scientific Machine Learning: Epoch {i+1} / {epochs} Loss {loss.item()}"
            )
            self.history["train loss"][i] = loss.item()
        self.network.train(False)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Completed training BayesianNeuralNetwork"
        )

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
