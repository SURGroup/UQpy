import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork


class DeepOperatorNetwork(NeuralNetwork):
    """Implementation of the Deep Operator Network (DeepONet) as defined by [#lu_deeponet]_

    References
    ----------

    .. [#lu_deeponet] Lu, L., Jin, P., Pang, G. et al.
     *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.*
     Nat Mach Intell 3, 218–229 (2021).
     https://doi.org/10.1038/s42256-021-00302-5
    """

    def __init__(self, branch_network: nn.Module, trunk_network: nn.Module, **kwargs):
        """Define DeepONet architecture with trunk and branch networks

        :param branch_network: Encodes mapping of the function :math:`u(x)`
        :param trunk_network: Encodes mapping of the transformed domain :math:`y` for :math:`Gu(y)`
        """
        super().__init__(**kwargs)
        self.branch_network: nn.Module = branch_network
        """Architecture of the branch neural network defined by a ``torch.nn.Module``"""
        self.trunk_network: nn.Module = trunk_network
        """Architecture of the trunk neural network defined by a ``torch.nn.Module``"""

        self.optimizer: torch.optim.Optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.loss_function: nn.Module = nn.MSELoss(reduction="mean")
        """Loss function used during ``train``. Default is ``torch.nn.MSELoss(reduction="mean")``"""

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

    def forward(
        self,
        x: torch.Tensor,
        u_x: torch.Tensor,
    ) -> torch.Tensor:
        """# ToDo: no clue if this einsum stuff is anywhere near reasonable or will generalize to higher dimensions

        :param x: Points in the domain
        :param u_x: evaluations of the function :math:`u(x)` at the points ``x``
        :return: Dot product of the branch and trunk networks
        """
        x = torch.atleast_3d(x)
        u_x = torch.atleast_2d(u_x)
        branch_output = self.branch_network(u_x)
        trunk_output = self.trunk_network(x)
        # return branch_output @ trunk_output
        # return torch.einsum("...i,...i -> ...i", branch_output, trunk_output)
        return torch.einsum("ik, ijk -> ij", branch_output, trunk_output)

    def train(self, data_loader: torch.utils.data.DataLoader, epochs: int = 100):
        """Train the network parameters using data provided by ``data_loader`` for ``epochs`` number of epochs

        Note: Modifying the ``optimizer`` and ``loss_function`` attributes must be done *before* ``train`` is called.

        :param data_loader: Dataloader that returns tuple of :math:`(x, u(x), Gu(x))` at each iteration
        :param epochs: Number of epochs to loop over data provided by ``data_loader``
        """
        self.branch_network.train(True)
        self.trunk_network.train(True)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Beginning training DeepOperatorNetwork"
        )
        self.history["train loss"] = torch.full((epochs,), torch.nan)
        for i in range(epochs):
            for batch, (x, u_x, gu_x) in enumerate(data_loader):
                prediction = self.forward(x, u_x)
                loss = self.loss_function(prediction, gu_x)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.logger.info(
                f"UQpy: Scientific Machine Learning: Epoch {i+1} / {epochs} Loss {loss.item()}"
            )
            self.history["train loss"][i] = loss.item()
        self.branch_network.train(False)
        self.trunk_network.train(False)
        self.logger.info(
            "UQpy: Scientific Machine Learning: Completed training DeepOperatorNetwork"
        )
