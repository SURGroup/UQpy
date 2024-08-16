import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.utilities.ValidationTypes import PositiveInteger


class DeepOperatorNetwork(NeuralNetwork):
    def __init__(
        self,
        branch_network: nn.Module,
        trunk_network: nn.Module,
        out_channels: PositiveInteger = 1,
        **kwargs,
    ):
        r"""Construct a DeepONet via its branch and trunk networks

        :param branch_network: Encodes mapping of the function :math:`f(x)`
        :param trunk_network: Encodes mapping of the domain :math:`y` for :math:`\mathcal{L}f(y)`
        :param out_channels: Number of channels produced by the Deep Operator Network

        .. note::
            The last layer of the branch and trunk network must have the same number of neurons so the last
            dimension of their outputs match, i.e. both outputs have shape :math:`(*, *, \text{width})`.
            Additionally, :math:`\text{width}` must be divisible by :math:`C_\text{out}`.

        Shape:

        - Input:

            - Branch Network (:math:`f(x)`): :math:`(N, 1, B_\text{in})`
            - Trunk Network (:math:`x`): :math:`(N, m, T_\text{in})`
        - Output: :math:`(N, m, C_\text{out})`
        """
        super().__init__(**kwargs)
        self.branch_network: nn.Module = branch_network
        """Architecture of the branch neural network defined by a :py:class`torch.nn.Module`"""
        self.trunk_network: nn.Module = trunk_network
        """Architecture of the trunk neural network defined by a :py:class:`torch.nn.Module`"""
        self.out_channels = out_channels

        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        x: torch.Tensor,
        f_x: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dot product of branch and trunk outputs

        :param x: Input to the :code:`trunk_network`
        :param f_x: Input to the :code:`branch_network`
        :return: Dot product of the branch and trunk outputs

        :raises RuntimeError: If incompatible trunk and branch outputs are encountered. See Shape for details.
        """
        branch_output = self.branch_network(f_x)
        trunk_output = self.trunk_network(x)
        if branch_output.shape[-1] != trunk_output.shape[-1]:
            raise RuntimeError(
                f"UQpy: Incompatible trunk {trunk_output.shape} and branch {branch_output.shape} output shapes."
                f"\nTrunk output must have shape (N, m, width). "
                f"Branch output must have shape (N, 1, width)."
            )
        n = x.shape[0]
        m = x.shape[1]
        width = branch_output.shape[-1]
        if width % self.out_channels != 0:
            raise RuntimeError(
                f"UQpy: Branch and trunk width {width} must be divisible by out_channels {self.out_channels}"
            )
        if branch_output.shape != torch.Size([n, 1, width]):
            raise RuntimeError(
                f"UQpy: Invalid branch output shape {branch_output.shape}. "
                f"Branch output must have shape (N, 1, width)."
            )
        if trunk_output.shape != torch.Size([n, m, width]):
            raise RuntimeError(
                f"UQpy: Invalid trunk output shape {trunk_output.shape}. "
                f"Trunk output must have shape (N, m, width)."
            )

        return torch.einsum(
            "...i,...i",
            branch_output.view(
                branch_output.shape[0], branch_output.shape[1], self.out_channels, -1
            ),
            trunk_output.view(
                trunk_output.shape[0], trunk_output.shape[1], self.out_channels, -1
            ),
        )

    def extra_repr(self) -> str:
        if self.out_channels != 1:
            return f"out_channels={self.out_channels}"
        return ""
