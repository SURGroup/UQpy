import beartype
import torch
import torch.functional as F
from UQpy.scientific_machine_learning.baseclass.Loss import Loss


@beartype
class EvidenceLowerBound(Loss):
    def __init__(self, train_size: int, **kwargs):  
        """

        :param train_size: Number of training samples
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.train_size = train_size


    def forward(self, x, target, kl, beta) -> torch.Tensor:
        """
        
        :param x:
        :param target:
        :param kl:
        :param beta: Weighting factor for the KL term
        :return:
        """
        return F.nll_loss(x, target, reduction="mean") * self.train_size + beta * kl