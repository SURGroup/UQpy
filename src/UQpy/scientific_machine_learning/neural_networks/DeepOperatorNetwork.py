import torch
import torch.nn as nn
import logging
from UQpy.scientific_machine_learning.baseclass.NeuralNetwork import NeuralNetwork
from UQpy.scientific_machine_learning.layers.BayesianLayer import BayesianLayer
from UQpy.scientific_machine_learning.layers.BayesianConvLayer import BayesianConvLayer


def gaussian_kullback_leibler_divergence(
        mu_posterior: torch.Tensor,
        sigma_posterior: torch.Tensor,
        mu_prior: torch.Tensor,
        sigma_prior: torch.Tensor,
) -> torch.Tensor:
    """Compute the Gaussian closed-form Kullback-Leibler Divergence

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
    assert not torch.isnan(kl)
    return kl


class DeepOperatorNetwork(NeuralNetwork):
    """Implementation of the Deep Operator Network (DeepONet) as defined by [#lu_deeponet]_

    References
    ----------

    . [#lu_deeponet] Lu, L., Jin, P., Pang, G. et al.
     *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.*
     Nat Mach Intell 3, 218–229 (2021).
     https://doi.org/10.1038/s42256-021-00302-5
    """

    def __init__(self, branch_network: nn.Module, trunk_network: nn.Module, num_outputs=1, **kwargs):
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
        """# ToDo: no clue if this einsum stuff is anywhere near reasonable or will generalize to higher dimensions

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
            outputs.append(torch.einsum("ik, ijk -> ij", branch_output[:, ind * num:ind * (num + 1)],
                                        trunk_output[:, :, ind * num:ind * (num + 1)]))
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

    def sample(self, mode: bool = True):
        """Set sampling mode for Neural Network and all child modules

        Note: Based on the `torch.nn.Module.train` and `torch.nn.Module.training` method and attributes

        :param mode:
        :return: ``self``
        """
        self.sampling = mode
        for m in self.network.modules():
            if hasattr(m, "sample"):
                m.sample(mode)
        # for child in self.children():
        #     child.sample(mode=mode)
        return self

    def is_deterministic(self) -> bool:
        """Check if neural network is behaving deterministically or probabilistically

        :return: ``True`` if output is deterministic, ``False`` if output is probabilistic
        """
        return not self.sampling and not self.training
