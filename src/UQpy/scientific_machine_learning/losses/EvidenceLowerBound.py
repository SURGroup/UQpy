import beartype
import torch
import torch.functional as F
from UQpy.scientific_machine_learning.baseclass.Loss import Loss


@beartype
class EvidenceLowerBound(Loss):
    def __init__(self, train_size: int, beta_type: str = "", **kwargs):  # ToDo: fix beta type and parameters
        """

        :param train_size:
        :param beta_type: # Tod
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.train_size = train_size

    def _get_beta(self, batch_idx, m, beta_type, epoch, num_epochs):
        if type(beta_type) is float:
            return beta_type

        if beta_type == "Blundell":
            beta = 2 ** (m - (batch_idx + 1)) / (2**m - 1)
        elif beta_type == "Soenderby":
            if epoch is None or num_epochs is None:
                raise ValueError(
                    "Soenderby method requires both epoch and num_epochs to be passed."
                )
            beta = min(epoch / (num_epochs // 4), 1)
        elif beta_type == "Standard":
            beta = 1 / m
        else:
            beta = 0
        return beta

    def forward(self, x, target, kl, beta) -> torch.Tensor:
        """

        :param x:
        :param target:
        :param kl:
        :param beta:
        :return:
        """
        return F.nll_loss(x, target, reduction="mean") * self.train_size + beta * kl
