import torch
import torch.nn as nn
import logging
from UQpy.utilities.ValidationTypes import PositiveInteger


class DeterministicTrainer:
    def __init__(
        self,
        network: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module = nn.MSELoss(),
        data_loader: torch.utils.data.DataLoader = None,
    ):
        """

        :param network:
        :param optimizer:
        :param loss_function:
        :param data_loader:
        """
        self.network = network
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.data_loader = data_loader

        self.history = None
        self.logger = logging.getLogger(__name__)

        if self.data_loader:
            self.run_training(self.data_loader)

    def run_training(self, data_loader: torch.utils.data.DataLoader, epochs: PositiveInteger = 1_000):
        """

        :param data_loader:
        :param epochs:
        """
        self.network.train(True)
        self.logger.info(
            f"UQpy: Scientific Machine Learning: Beginning training {self.network.__name__}"
        )
        for i in range(epochs):
            for batch, (data, target) in enumerate(data_loader):
                prediction = self.network(data)
                loss = self.loss_function(prediction, target)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.logger.info(
                f"UQpy: Scientific Machine Learning: Epoch {i+1:,} / {epochs:,} Loss {loss.item()}"
            )
        self.network.train(False)
        self.logger.info(
            f"UQpy: Scientific Machine Learning: Completed training {self.network.__name__}"
        )
