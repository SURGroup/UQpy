import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork


class VanillaNeuralNetwork(NeuralNetwork):

    def __init__(self, network: nn.Module, **kwargs):
        """Initialize a typical feed-forward neural network using the architecture provided by ``network``

        :param network: Network defining the function from :math:`f(x)=y`
        """
        super().__init__(**kwargs)
        self.network = network
        self.optimizer = torch.optim.Adam(self.parameters())
        self.loss_function = nn.MSELoss(reduction="mean")

        self.logger = logging.getLogger(__name__)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def loss_function(self):
        return self._loss_function

    @loss_function.setter
    def loss_function(self, value):
        self._loss_function = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train(self, data_loader: torch.utils.data.DataLoader, epochs: int = 100):
        """Train the network parameters using data provided by ``data_loader`` for ``epochs`` number of epochs

        :param data_loader: DataLoader that returns tuple of :math:`(x, y)` at each iteration
        :param epochs: Number of epochs to loop over data provided by ``data_loader``
        :return:
        """
        self.network.train(True)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Beginning training VanillaNeuralNetwork"
        )
        self.history["train loss"] = torch.full((epochs,), torch.nan)
        for i in range(epochs):
            for batch, (x, y) in enumerate(data_loader):
                prediction = self.forward(x)
                loss = self.loss_function(prediction, y)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.logger.info(
                f"UQpy: Scientific Machine Learning: Epoch {i+1} / {epochs} Loss {loss.item()}"
            )
            self.history["train loss"][i] = loss.item()
        self.network.train(False)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Completed training VanillaNeuralNetwork"
        )
