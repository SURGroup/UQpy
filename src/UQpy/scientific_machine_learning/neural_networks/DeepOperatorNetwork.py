import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.scientific_machine_learning.layers import BayesianLayer, BayesianConvLayer
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import gaussian_kullback_leibler_divergence


class DeepOperatorNetwork(NeuralNetwork):
    """Implementation of the Deep Operator Network (DeepONet) as defined by [#lu_deeponet]_

    References
    ----------

    . [#lu_deeponet] Lu, L., Jin, P., Pang, G. et al.
     *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.*
     Nat Mach Intell 3, 218–229 (2021).
     https://doi.org/10.1038/s42256-021-00302-5
    """

    def __init__(
        self,
        branch_network: nn.Module,
        trunk_network: nn.Module,
        num_outputs=1,
        **kwargs
    ):
        """Define DeepONet architecture with trunk and branch networks

        :param branch_network: Encodes mapping of the function :math:`u(x)`
        :param trunk_network: Encodes mapping of the transformed domain :math:`y` for :math:`Gu(y)`
        """
        super().__init__(**kwargs)
        self.branch_network: nn.Module = branch_network
        """Architecture of the branch neural network defined by a ``torch.nn.Module``"""
        self.trunk_network: nn.Module = trunk_network
        """Architecture of the trunk neural network defined by a ``torch.nn.Module``"""
        self.num_outputs = num_outputs
        self.logger = logging.getLogger(__name__)

    def forward(
        self,
        x: torch.Tensor,
        u_x: torch.Tensor,
    ) -> list[torch.Tensor]:
        """# ToDo: Replace lists with  appropriate einsum

        :param x: Points in the domain
        :param u_x: evaluations of the function :math:`u(x)` at the points ``x``
        :return: Dot product of the branch and trunk networks
        """
        x = torch.atleast_3d(x)
        u_x = torch.atleast_2d(u_x)
        branch_output = self.branch_network(u_x)
        trunk_output = self.trunk_network(x)
        # Implementing multiple outputs
        """ ToDo: Find a better way of doing this """
        assert (trunk_output.shape[-1] % self.num_outputs) == 0
        ind = trunk_output.shape[-1] // self.num_outputs
        outputs = []
        for num in range(self.num_outputs):
            outputs.append(
                torch.einsum(
                    "ik, ijk -> ij",
                    branch_output[:, ind * num : ind * (num + 1)],
                    trunk_output[:, :, ind * num : ind * (num + 1)],
                )
            )
        # return branch_output @ trunk_output
        # return torch.einsum("...i,...i -> ...i", branch_output, trunk_output)
        return outputs[0] if self.num_outputs == 1 else outputs

    def compute_kullback_leibler_divergence(self) -> float:
        """Computes the Kullback-Leibler divergence between the current and prior ``network`` parameters

        :return: Kullback-Leibler divergence
        """
        kl = 0
        for layer in self.branch_network.modules():
            if isinstance(layer, BayesianLayer) or isinstance(layer, BayesianConvLayer):
                kl += gaussian_kullback_leibler_divergence(
                    layer.weight_mu,
                    torch.log1p(torch.exp(layer.weight_sigma)),
                    layer.prior_mu,
                    layer.prior_sigma,
                )
                if layer.bias:
                    kl += gaussian_kullback_leibler_divergence(
                        layer.bias_mu,
                        torch.log1p(torch.exp(layer.bias_sigma)),
                        layer.prior_mu,
                        layer.prior_sigma,
                    )
        for layer in self.trunk_network.modules():
            if isinstance(layer, BayesianLayer) or isinstance(layer, BayesianConvLayer):
                kl += gaussian_kullback_leibler_divergence(
                    layer.weight_mu,
                    torch.log1p(torch.exp(layer.weight_sigma)),
                    layer.prior_mu,
                    layer.prior_sigma,
                )
                if layer.bias:
                    kl += gaussian_kullback_leibler_divergence(
                        layer.bias_mu,
                        torch.log1p(torch.exp(layer.bias_sigma)),
                        layer.prior_mu,
                        layer.prior_sigma,
                    )
        return kl
