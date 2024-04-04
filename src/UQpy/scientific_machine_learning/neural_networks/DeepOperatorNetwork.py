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

        self.logger = logging.getLogger(__name__)

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

