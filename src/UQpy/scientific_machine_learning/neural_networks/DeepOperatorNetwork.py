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
    ):
        r"""Construct a Deep Operator Network via its branch and trunk networks

        :param branch_network: Encodes mapping of the function :math:`f(x)`
        :param trunk_network: Encodes mapping of the domain :math:`y` for :math:`\mathcal{L}f(y)`
        :param out_channels: Number of channels produced by the Deep Operator Network

        .. note::
            The last layer of the branch and trunk network must have the same number of neurons so the last
            dimension of their outputs match, i.e. both outputs have shape :math:`(*, \text{width})`.
            Additionally, :math:`\text{width}` must be divisible by :math:`C_\text{out}`.

        Shape:

        - Input:

            - Branch Network (:math:`f(x)`): :math:`(N, m_\text{branch})`
            - Trunk Network (:math:`x`): :math:`(N, m_\text{trunk}, d)`
        - Output: :math:`(N, m_\text{trunk}, C_\text{out})`
        """
        super().__init__()
        self.branch_network: nn.Module = branch_network
        """Architecture of the branch neural network defined by a :py:class:`torch.nn.Module`"""
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
                f"\nTrunk output must have shape (N, m_trunk, width). "
                f"Branch output must have shape (N, width)."
            )
        batch_size = f_x.size(0)
        width = branch_output.size(1)
        # check dimensions of branch output
        if branch_output.shape != torch.Size([batch_size, width]):
            raise RuntimeError(
                f"UQpy: Invalid branch output shape {branch_output.shape}. "
                f"Branch output must have shape (N, width)={(batch_size, width)}."
            )
        # check dimensions of trunk output
        if x.ndim == 2:
            m_trunk = x.size(0)
            expected_trunk_output_shape = torch.Size([m_trunk, width])
        elif x.ndim == 3:
            m_trunk = x.size(1)
            expected_trunk_output_shape = torch.Size([batch_size, m_trunk, width])
        else:
            raise RuntimeError(
                f"UQpy: Invalid trunk output shape {trunk_output.shape}. "
                f"Trunk output must have shape (m_trunk, width) or (N, m_trunk, width)."
            )
        if trunk_output.shape != expected_trunk_output_shape:
            raise RuntimeError(
                f"UQpy: Invalid trunk output shape {trunk_output.shape}. "
                f"Trunk output must have shape (m_trunk, width) or (N, m_trunk, width). "
                f"Expected shape: {expected_trunk_output_shape}"
            )
        # check that the number of out channels divides the width
        if width % self.out_channels != 0:
            raise RuntimeError(
                f"UQpy: Branch and trunk width {width} must be divisible by out_channels {self.out_channels}"
            )

        return torch.einsum(
            "...i,...i",
            branch_output.view(batch_size, 1, self.out_channels, -1),
            trunk_output.view(
                1 if x.ndim == 2 else batch_size, m_trunk, self.out_channels, -1
            ),
        )

    def extra_repr(self) -> str:
        if self.out_channels != 1:
            return f"out_channels={self.out_channels}"
        return ""
