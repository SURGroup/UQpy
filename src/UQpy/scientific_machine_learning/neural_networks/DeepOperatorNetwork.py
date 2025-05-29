import torch
import torch.nn as nn
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

        - Input: Two tensors representing :math:`x` and :math:`f(x)`

            - Branch Network (:math:`f(x)`): Any shape (can be different from trunk)
            - Trunk Network (:math:`x`): Any shape (can be different from branch)

        - Intermediary: The output from the branch and trunk network must be of shapes :math:`(*, \text{width})` and :math:`(*, \text{width})`.
          Where :math:`*` refers to any broadcastable dimensions.
          Both the tensors are viewed reshaped as :math:`(*, C_\text{out}, \frac{\text{width}}{C_\text{out}})` before the dot product is computed.

        - Output: Tensor of shape :math:`(*, C_\text{out})`

        """
        super().__init__()
        self.branch_network: nn.Module = branch_network
        """Architecture of the branch neural network defined by a :py:class:`torch.nn.Module`"""
        self.trunk_network: nn.Module = trunk_network
        """Architecture of the trunk neural network defined by a :py:class:`torch.nn.Module`"""
        self.out_channels = out_channels

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
        if branch_output.size(-1) != trunk_output.size(-1):
            raise RuntimeError(
                f"UQpy: Incompatible trunk {trunk_output.shape} and branch {branch_output.shape} output shapes."
                f"\nTrunk and branch output must have shape (*, width). "
            )
        width = branch_output.size(-1)
        if width % self.out_channels != 0:
            raise RuntimeError(
                f"UQpy: Branch and trunk width {width} must be divisible by out_channels {self.out_channels}"
            )

        return torch.einsum(
            "...i,...i",
            branch_output.view(
                *branch_output.shape[:-1], self.out_channels, width // self.out_channels
            ),
            trunk_output.view(
                *trunk_output.shape[:-1], self.out_channels, width // self.out_channels
            ),
        )

    def extra_repr(self) -> str:
        if self.out_channels != 1:
            return f"out_channels={self.out_channels}"
        return ""
